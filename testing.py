from pathlib import Path
import os

import numpy as np
import xarray as xr
import copernicusmarine


DATA_DIR = Path(__file__).parent / "data"
ENV_PATH = Path(__file__).parent / ".env"
DATASET_IDS = {
    "observations_sss": "cmems_obs-mob_glo_phy-sss_my_multi_P1M",
    "reanalysis_phy": "cmems_mod_glo_phy_my_0.083deg_P1M-m",
}


def load_dotenv(path: Path = ENV_PATH):
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
            # Overwrite to guarantee we pick up the .env credentials.
            os.environ[key] = value


def ensure_credentials():
    """Make sure Copernicus Marine credentials are configured."""
    load_dotenv()
    username = os.environ.get("COPERNICUSMARINE_USERNAME")
    password = os.environ.get("COPERNICUSMARINE_PASSWORD")
    if not username or not password:
        raise SystemExit(
            "Set COPERNICUSMARINE_USERNAME and COPERNICUSMARINE_PASSWORD "
            "environment variables to enable downloads."
        )
    copernicusmarine.login(
        username=username,
        password=password,
        check_credentials_valid=True,
        force_overwrite=False,
    )


def download_sample_subset(
    output_dir: Path = DATA_DIR,
    output_filename: str = "copernicus_subset.nc",
):
    """
    Download a tiny Copernicus Marine subset so we have a local file to explore.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    if output_path.exists():
        print(f"Reusing existing subset at {output_path}")
        return output_path

    ensure_credentials()
    print("Requesting a small subset from Copernicus Marine (January 2021, surface)...")
    resp = copernicusmarine.subset(
        dataset_id="cmems_mod_ibi_phy_anfc_0.027deg-3D_P1M-m",
        variables=["thetao"],
        minimum_longitude=-10.0,
        maximum_longitude=-9.5,
        minimum_latitude=38.0,
        maximum_latitude=38.5,
        minimum_depth=0,
        maximum_depth=5,
        start_datetime="2021-01-01T00:00:00Z",
        end_datetime="2021-01-31T23:59:59Z",
        output_directory=output_dir,
        output_filename=output_filename,
        overwrite=False,
        skip_existing=True,
    )
    print(f"Saved subset to {resp.output_path}")
    return Path(resp.output_path)


def list_nc_files():
    files = sorted(DATA_DIR.glob("*.nc"))
    if not files:
        raise SystemExit(f"No .nc files found in {DATA_DIR}")
    return files


def load_first_dataset(files):
    # Single-file open keeps runtime small while still showing the structure.
    return xr.open_dataset(files[0])


def print_overview(ds):
    print("Dataset overview:")
    print(ds)
    print("\nData variables:", list(ds.data_vars))
    if "time" in ds:
        times = ds["time"].dt.strftime("%Y-%m-%d").values
        print(f"\nTime steps (first few): {times[:5]}")
    print()


def show_sample(ds, var_name="sos"):
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found; available: {list(ds.data_vars)}")

    lat_mid = len(ds["lat"]) // 2
    lon_mid = len(ds["lon"]) // 2
    sample = ds[var_name].isel(
        time=0,
        depth=0,
        lat=slice(lat_mid, lat_mid + 3),
        lon=slice(lon_mid, lon_mid + 3),
    )
    print(
        f"Sample from '{var_name}' (time=0, depth=0, lat[{lat_mid}:{lat_mid+3}], "
        f"lon[{lon_mid}:{lon_mid+3}]):"
    )
    print(np.array(sample))
    print("\nStats on that sample:")
    arr = np.array(sample)
    if np.isnan(arr).all():
        print("  all values are NaN in this window")
    else:
        print(f"  min: {np.nanmin(arr):.3f}, max: {np.nanmax(arr):.3f}, mean: {np.nanmean(arr):.3f}")
    print()


def print_remote_dims(dataset_id, var_candidates=None, skip_login=False):
    """
    Lazily open a Copernicus Marine dataset and print dimension sizes.
    No data arrays are loaded into memory; xarray only fetches metadata.
    """
    if not skip_login:
        ensure_credentials()
    print(f"\n=== {dataset_id} ===")
    ds = copernicusmarine.open_dataset(dataset_id=dataset_id)
    try:
        print(f"Dimensions: {dict(ds.dims)}")
        candidates = var_candidates or ()
        found = False
        for name in candidates:
            if name in ds:
                print(f"  {name}: {ds[name].shape}")
                found = True
        if not found and candidates:
            print(f"  None of {candidates} found in data variables {list(ds.data_vars)[:5]}...")
    finally:
        ds.close()


def main():
    files = list(DATA_DIR.glob("*.nc"))
    if not files:
        subset_path = download_sample_subset()
        files = [subset_path]
    files = sorted(files)
    print(f"Found {len(files)} NetCDF files. First file: {files[0].name}\n")

    ds = load_first_dataset(files)
    print_overview(ds)
    show_sample(ds, var_name="sos")

    print("\nRemote dataset dimension summaries (metadata only):")
    ensure_credentials()
    print_remote_dims(
        DATASET_IDS["observations_sss"],
        var_candidates=["sos", "sss"],
        skip_login=False,
    )
    print_remote_dims(
        DATASET_IDS["reanalysis_phy"],
        var_candidates=["so", "thetao", "sos"],
        skip_login=False,
    )


if __name__ == "__main__":
    main()
