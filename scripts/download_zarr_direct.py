#!/usr/bin/env python3
"""
Download Zarr stores directly from Copernicus Marine ARCO service.
Much faster than converting - just copies their chunks as-is.
"""

import argparse
import os
from pathlib import Path

import copernicusmarine
import xarray as xr
import zarr

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ZARR_DIR = DATA_DIR / "zarr"
ENV_PATH = ROOT_DIR / ".env"

# define END and START datetime for the dataset
# for our project we are using only data for 2011
START_DATETIME = "2011-01-01"
END_DATETIME = "2011-12-31"
MAXIMUM_DEPTH = 2

DATASETS = {
  "reanalysis": {
    # monthly dataset
    "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1M-m",
    "service": "arco-geo-series",
    "keep_vars": ["so", "thetao"],
  },
}


def load_dotenv(path: Path = ENV_PATH) -> None:
  """Lightweight .env loader."""
  if not path.exists():
    return
  for line in path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
      key, value = line.split("=", 1)
      key = key.strip()
      value = value.strip().strip('"').strip("'")
      if key:
        os.environ[key] = value


def ensure_credentials() -> tuple[str, str]:
  """Get credentials from .env."""
  load_dotenv()
  username = os.environ.get("COPERNICUSMARINE_USERNAME")
  password = os.environ.get("COPERNICUSMARINE_PASSWORD")
  if not username or not password:
    raise SystemExit(
      "Set COPERNICUSMARINE_USERNAME and COPERNICUSMARINE_PASSWORD in .env"
    )
  return username, password


def download_zarr_direct(
  name: str,
  dataset_id: str,
  service: str,
  keep_vars: list[str],
  output_dir: Path,
  overwrite: bool,
  username: str,
  password: str,
) -> Path:
  """
  Download Zarr store directly from Copernicus ARCO service.
  No conversion, just copies their chunks as-is.
  """
  output_path = output_dir / f"{name}.zarr"

  if output_path.exists() and not overwrite:
    print(f"✓ {output_path} already exists. Use --overwrite to replace.")
    return output_path

  print(f"\n{'=' * 70}")
  print(f"Downloading: {name}")
  print(f"{'=' * 70}")
  print(f"Dataset: {dataset_id}")
  print(f"Service: {service}")
  print(f"Output: {output_path}")
  print(f"Variables: {keep_vars}\n")

  # Open remote Zarr store
  print("Opening remote ARCO store...")
  ds = copernicusmarine.open_dataset(
    dataset_id=dataset_id,
    service=service,
    username=username,
    password=password,
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
    maximum_depth=MAXIMUM_DEPTH,
  )

  try:
    print(f"Remote dims: {dict(ds.dims)}")
    print(f"Remote chunks: {ds.chunks}")
    print(f"Available vars: {list(ds.data_vars)[:10]}...\n")

    # Filter to requested variables
    if keep_vars:
      available = [v for v in keep_vars if v in ds.data_vars]
      if not available:
        raise ValueError(f"None of {keep_vars} found in dataset")
      ds = ds[available]
      print(f"Keeping variables: {available}\n")

    # Use extremely memory-efficient approach with Dask streaming
    print("Downloading Zarr store with minimal memory usage...")
    print("This may take 10-30 minutes depending on network speed...")
    print("Using streaming download to avoid out-of-memory errors...\n")

    # Ensure parent directory exists with explicit error handling
    try:
      output_path.parent.mkdir(parents=True, exist_ok=True)
      print(f"✓ Created directory: {output_path.parent}")
    except Exception as e:
      print(f"✗ Failed to create directory {output_path.parent}: {e}")
      raise
    
    # Also ensure the zarr path itself doesn't exist or is writable
    if output_path.exists() and not overwrite:
      raise FileExistsError(f"Output path {output_path} already exists. Use --overwrite to replace.")
    
    # Remove existing zarr store if overwrite is True
    if output_path.exists() and overwrite:
      import shutil
      print(f"Removing existing {output_path}...")
      shutil.rmtree(output_path)

    # Configure Dask for minimal memory usage
    import dask
    with dask.config.set({
      'array.slicing.split_large_chunks': True,
      'distributed.worker.memory.target': False,
      'distributed.worker.memory.spill': False,
      'distributed.worker.memory.pause': False,
    }):
      # Use compute=False then compute with retries
      print("Creating delayed write task...")
      delayed = ds.to_zarr(
        output_path,
        mode="w" if overwrite else "w-",
        consolidated=True,
        compute=False,
        zarr_format=2,
      )
      
      print("Executing download (streaming from remote to local)...")
      print("Note: This uses lazy evaluation to minimize memory usage.\n")
      
      # Execute with single-threaded scheduler for memory efficiency
      with dask.config.set(scheduler='single-threaded'):
        delayed.compute()

    # Get final size
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    size_gb = total_size / (1024**3)

    print(f"\n✓ Download complete!")
    print(f"  Location: {output_path}")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"  Variables: {list(ds.data_vars)}")

    return output_path

  finally:
    ds.close()


def main():
  parser = argparse.ArgumentParser(
    description="Download Zarr directly from Copernicus ARCO service (no conversion)",
    epilog="This preserves the remote chunking scheme and is much faster than convert+rechunk.",
  )
  parser.add_argument(
    "--datasets",
    nargs="+",
    choices=list(DATASETS.keys()) + ["all"],
    default=["all"],
    help="Which datasets to download (default: all)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing Zarr stores",
  )
  args = parser.parse_args()

  # Determine which to process
  to_process = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

  print("=" * 70)
  print("Direct Zarr Download from Copernicus ARCO")
  print("=" * 70)
  print(f"Datasets: {', '.join(to_process)}")
  print(f"Output directory: {ZARR_DIR}\n")

  # Get credentials
  username, password = ensure_credentials()
  print("✓ Credentials loaded\n")

  # Download each dataset
  for name in to_process:
    cfg = DATASETS[name]
    try:
      download_zarr_direct(
        name=name,
        dataset_id=cfg["dataset_id"],
        service=cfg.get("service", "arco-geo-series"),
        keep_vars=cfg.get("keep_vars", []),
        output_dir=ZARR_DIR,
        overwrite=args.overwrite,
        username=username,
        password=password,
      )
    except Exception as e:
      print(f"\n✗ Error downloading {name}: {e}")
      import traceback

      traceback.print_exc()
      continue

  print("\n" + "=" * 70)
  print(f"✓ Completed {len(to_process)} dataset(s)")
  print("=" * 70)
  print("\nNext steps:")
  print("  1. Update app/main.py line 13:")
  print("     from app.datasets import DatasetManager")
  print("  2. Restart server: uvicorn app.main:app --reload")
  print("  3. Enjoy fast queries!")


if __name__ == "__main__":
  main()
