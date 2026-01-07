# Salinity & Temperature API (FastAPI + Zarr)

Serve Copernicus Marine salinity/temperature subsets from local Zarr stores with lazy loading and simple caching.

## Setup

1. Python 3.10+ recommended.
2. `pip install -r requirements.txt`
3. Export credentials (or place them in `.env` at repo root):
   - `COPERNICUSMARINE_USERNAME=...`
   - `COPERNICUSMARINE_PASSWORD=...`

## Prepare data (offline, once)

```
python scripts/prepare_zarr.py --overwrite
```

Outputs to `data/zarr/reanalysis.zarr`. Adjust chunk sizes with `--chunk-*` flags.

## Run the API

```
uvicorn app.main:app --reload
```

Health: `GET /health`  
Subset: `GET /subset?dataset=reanalysis&variable=thetao&min_lon=-10&max_lon=-5&min_lat=30&max_lat=35&time=2021-01-15&depth=5&fmt=netcdf`

Env knobs (prefix `APP_`):
- `APP_ZARR_DIR` (default `data/zarr`)
- `APP_SUBSET_MAX_CELLS` (default 400000)
- `APP_CACHE_SIZE` (default 16)

## Visualize data

Beautiful visualization script that calls the API and displays data on a map:

```bash
# Global surface temperature
python visualize_data.py --variable thetao --depth 0

# Mediterranean Sea
python visualize_data.py --variable thetao --min-lon -5 --max-lon 36 --min-lat 30 --max-lat 46 --depth 0

# Save to file
python visualize_data.py --variable so --output salinity_map.png
```

For better maps, install cartopy: `conda install -c conda-forge cartopy`

## Notes

- The API opens Zarr lazily; ensure `scripts/prepare_zarr.py` has been run beforehand (or use remote Zarr).
- Responses are limited by `APP_SUBSET_MAX_CELLS` to avoid huge payloads.
- Cached responses are kept in-process only.

