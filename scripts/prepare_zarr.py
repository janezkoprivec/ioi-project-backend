from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import copernicusmarine
import xarray as xr

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ZARR_DIR = DATA_DIR / "zarr"
ENV_PATH = ROOT_DIR / ".env"

# Dataset definitions: id plus optional variable renames to harmonize naming.
DATASETS: Dict[str, Dict[str, object]] = {
    "reanalysis": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1M-m",
        "var_renames": {"latitude": "lat", "longitude": "lon"},
        "keep_vars": ("so", "thetao"),
    },
}


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Lightweight .env loader to avoid extra dependencies."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # Always overwrite to guarantee we pick up the .env credentials.
            os.environ[key] = value


def ensure_credentials() -> tuple[str, str]:
    """Make sure Copernicus Marine credentials are configured and return them."""
    load_dotenv()
    username = os.environ.get("COPERNICUSMARINE_USERNAME")
    password = os.environ.get("COPERNICUSMARINE_PASSWORD")
    if not username or not password:
        raise SystemExit(
            "Set COPERNICUSMARINE_USERNAME and COPERNICUSMARINE_PASSWORD "
            "environment variables to enable downloads."
        )
    return username, password


def get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except (PermissionError, FileNotFoundError):
        pass
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


def monitor_progress(output_path: Path, stop_event: threading.Event, interval: float = 3.0) -> None:
    """Background thread that monitors Zarr store size and prints progress."""
    last_size = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        current_size = get_dir_size(output_path)
        elapsed = time.time() - start_time
        
        if current_size > last_size:
            rate = (current_size - last_size) / interval / (1024 * 1024)  # MB/s
            print(f"  Progress: {format_size(current_size)} written "
                  f"(+{rate:.1f} MB/s, {elapsed:.0f}s elapsed)", flush=True)
            last_size = current_size
        
        time.sleep(interval)


def normalize_dataset(
    ds: xr.Dataset,
    var_renames: Mapping[str, str],
    keep_vars: Iterable[str],
    target_chunks: Mapping[str, int],
) -> xr.Dataset:
    """Rename dims/vars, drop extras, and rechunk."""
    rename_vars_map = {k: v for k, v in var_renames.items() if k in ds}
    if rename_vars_map:
        ds = ds.rename_vars(rename_vars_map)
    dim_map = {k: v for k, v in var_renames.items() if k in ds.dims}
    if dim_map:
        # If the target dim already exists (after rename_vars), swap_dims avoids conflicts.
        ds = ds.swap_dims(dim_map)
        # Drop old coordinate names to keep things clean.
        ds = ds.drop_vars([d for d in dim_map if d in ds.coords], errors="ignore")
    ds = ds.drop_vars([v for v in ds.data_vars if v not in keep_vars], errors="ignore")
    # Ensure expected coordinate names exist
    for coord in ("lat", "lon"):
        if coord not in ds.coords:
            raise ValueError(f"Expected coordinate '{coord}' not found after renaming")
    return ds.chunk(target_chunks)


def write_zarr(
    ds: xr.Dataset,
    output_path: Path,
    consolidate: bool = True,
    overwrite: bool = False,
) -> None:
    """Persist dataset to a Zarr store on local disk, processing one variable at a time to limit memory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Estimate size
    total_cells = 1
    for dim in ds.dims:
        total_cells *= ds.sizes[dim]
    n_vars = len(ds.data_vars)
    print(f"  Dataset: {n_vars} variable(s), ~{total_cells:,} cells per variable")
    print(f"  Processing one variable at a time to limit memory usage...\n")
    
    # Process variables one at a time to reduce memory footprint
    var_names = list(ds.data_vars)
    
    for i, var_name in enumerate(var_names, 1):
        print(f"  [{i}/{n_vars}] Writing variable: {var_name}")
        mode = "w" if (overwrite and i == 1) else "a"
        
        # Create a dataset with just this variable plus coordinates
        ds_single = ds[[var_name]]
        
        # Start progress monitoring thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_progress,
            args=(output_path, stop_event, 5.0),
            daemon=True,
        )
        monitor_thread.start()
        
        try:
            ds_single.to_zarr(
                output_path,
                mode=mode,
                consolidated=False,  # Consolidate only at the end
                compute=True,
                zarr_format=2,
            )
        finally:
            stop_event.set()
            monitor_thread.join(timeout=1.0)
        
        var_size = get_dir_size(output_path)
        print(f"    ✓ {var_name} complete. Total size so far: {format_size(var_size)}\n")
    
    # Consolidate metadata at the end
    if consolidate:
        print("  Consolidating metadata...")
        import zarr
        zarr.consolidate_metadata(output_path)
    
    final_size = get_dir_size(output_path)
    print(f"  Final size: {format_size(final_size)}")


def prepare_one(
    name: str,
    dataset_id: str,
    var_renames: Mapping[str, str],
    keep_vars: Iterable[str],
    chunks: Mapping[str, int],
    overwrite: bool,
    username: str,
    password: str,
) -> Path:
    """Open a Copernicus dataset lazily and write a local Zarr copy."""
    import time
    start = time.time()
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {name}")
    print(f"{'='*60}")
    print(f"Opening {dataset_id} ...")
    ds = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        username=username,
        password=password,
    )
    try:
        print(f"Original dims: {dict(ds.dims)}; vars: {list(ds.data_vars)[:5]}...")
        ds_norm = normalize_dataset(ds, var_renames, keep_vars, chunks)
        output_path = ZARR_DIR / f"{name}.zarr"
        print(f"Writing Zarr to {output_path} with chunks {chunks} ...")
        write_zarr(ds_norm, output_path, consolidate=True, overwrite=overwrite)
        elapsed = time.time() - start
        print(f"✓ Completed in {elapsed:.1f}s: {output_path}")
        return output_path
    finally:
        ds.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Copernicus datasets as Zarr.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        help="Which datasets to process (default: all). Example: --datasets observations",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Zarr stores.",
    )
    parser.add_argument(
        "--chunk-time",
        type=int,
        default=1,
        help="Chunk size along time dimension (default: 1).",
    )
    parser.add_argument(
        "--chunk-depth",
        type=int,
        default=10,
        help="Chunk size along depth dimension (default: 10).",
    )
    parser.add_argument(
        "--chunk-lat",
        type=int,
        default=512,
        help="Chunk size along latitude/lat dimension (default: 512, larger = less memory overhead).",
    )
    parser.add_argument(
        "--chunk-lon",
        type=int,
        default=512,
        help="Chunk size along longitude/lon dimension (default: 512, larger = less memory overhead).",
    )
    return parser.parse_args()


def main() -> None:
    # Configure dask to limit memory usage
    import dask
    dask.config.set({
        'array.slicing.split_large_chunks': True,
        'distributed.worker.memory.target': 0.6,  # Start spilling at 60% memory
        'distributed.worker.memory.spill': 0.7,   # Spill to disk at 70%
        'distributed.worker.memory.pause': 0.8,   # Pause at 80%
        'distributed.worker.memory.terminate': 0.95,  # Kill at 95%
    })
    # Limit number of threads to reduce memory pressure
    dask.config.set(scheduler='threads', num_workers=2)
    print(f"Dask configured: 2 workers, memory limits enabled")
    
    # Enable dask progress bars if available
    try:
        from dask.diagnostics import ProgressBar
        ProgressBar().register()
        print("Dask progress bars enabled.")
    except ImportError:
        print("Dask progress bars not available (install dask[diagnostics] for progress).")
    
    args = parse_args()
    chunks: Dict[str, int] = {
        "time": args.chunk_time,
        "depth": args.chunk_depth,
        "lat": args.chunk_lat,
        "lon": args.chunk_lon,
    }
    print(f"Target chunks: {chunks}")
    
    # Determine which datasets to process
    if "all" in args.datasets:
        to_process = list(DATASETS.keys())
    else:
        to_process = args.datasets
    print(f"Processing datasets: {', '.join(to_process)}")
    
    # Get credentials once for all datasets
    username, password = ensure_credentials()
    print("Credentials loaded successfully.\n")
    
    for name in to_process:
        cfg = DATASETS[name]
        prepare_one(
            name=name,
            dataset_id=cfg["dataset_id"],
            var_renames=cfg["var_renames"],
            keep_vars=cfg["keep_vars"],
            chunks=chunks,
            overwrite=args.overwrite,
            username=username,
            password=password,
        )
    print("\n" + "="*60)
    print(f"Successfully prepared {len(to_process)} dataset(s)!")
    print("="*60)


if __name__ == "__main__":
    main()

