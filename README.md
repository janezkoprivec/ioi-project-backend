# Salinity & Temperature API (FastAPI + Zarr)

Serve Copernicus Marine salinity/temperature subsets from local Zarr stores with lazy loading and simple caching.

## Data Source

This project uses data from the [Global Ocean Physics Reanalysis (GLORYS12V1)](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) database provided by Copernicus Marine Service. We utilize the following variables:
- **thetao** [degrees celsius] - Sea water potential temperature
- **so** [psu] - Sea water salinity

## Quick Start with Docker (Recommended)

The easiest way to run this project is using Docker Compose:

1. Create a `.env` file with your Copernicus Marine credentials:
   ```bash
   COPERNICUSMARINE_USERNAME=your_username
   COPERNICUSMARINE_PASSWORD=your_password
   API_PORT=8000  # Optional: change the port (default: 8000)
   ```

2. Run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

This will automatically download the data and start the API server on http://localhost:8000

See [DOCKER_README.md](DOCKER_README.md) for detailed Docker instructions.

## Manual Setup (Alternative)

1. Python 3.10+ recommended.
2. `pip install -r requirements.txt`
3. Create a local `.env` file in the repository root with your Copernicus Marine credentials:
   ```bash
   COPERNICUSMARINE_USERNAME=your_username
   COPERNICUSMARINE_PASSWORD=your_password
   ```
   
   To obtain credentials, register for a free account at: [https://data.marine.copernicus.eu/register](https://data.marine.copernicus.eu/register)
   
   **Important:** Make sure `.env` is in your `.gitignore` file (it should be by default) to avoid committing credentials.

## Prepare data (offline, once)

**Important:** Do NOT use `prepare_zarr.py`. Instead, use `download_zarr_direct.py` to download Zarr stores directly from Copernicus Marine ARCO service (much faster, no conversion needed).

```
python scripts/download_zarr_direct.py --overwrite
```

Outputs to `data/zarr/reanalysis.zarr`. This script downloads data for **2011 only, on a monthly basis** (as defined in the script constants: `START_DATETIME = "2011-01-01"` and `END_DATETIME = "2011-12-31"`).

## Run the API

```
uvicorn app.main:app --reload
```

Health: `GET /health`  
Subset: `GET /subset?dataset=reanalysis&variable=thetao&min_lon=-10&max_lon=-5&min_lat=30&max_lat=35&time=2011-07-01&depth=5&fmt=netcdf`

Mean: `POST /mean?variable=thetao&margin=5&time=2011-07-01&depth=0`  
Request body (JSON): `[{"lon": -10, "lat": 10}, {"lon": -5, "lat": 15}]`  
Returns the mean value of the specified variable (thetao or so) computed across multiple coordinate points. For each (lon, lat) pair, creates a square region with the given margin (in degrees) around the point, computes the mean value in that region, then returns the overall mean of all computed means.

Mean Region: `GET /mean-region?region=world&variable=thetao&time=2011-01&stride=4`  
Returns the mean value of the specified variable (thetao or so) for a predefined geographic region.  
Parameters:
- `region` (required): Geographic region - `"world"` (global: lat -90 to 90, lon -180 to 180) or `"europe"` (lat 29 to 72, lon -15 to 45)
- `variable` (required): Data variable - `"thetao"` (temperature) or `"so"` (salinity)
- `time` (required): Time point in format `YYYY-MM` or `YYYY-MM-DD` (e.g., `"2011-01"` or `"2011-01-15"`)
- `stride` (required): Spatial decimation factor (integer 1-50, e.g., 2 = every 2nd point, 4 = every 4th point). Use higher stride values (4-8) for world region to avoid size limits.
- `dataset` (optional): Dataset name, defaults to `"reanalysis"`  
Depth is automatically set to 0.0 (surface level).  
Example: `GET /mean-region?region=world&variable=thetao&time=2011-01&stride=6`

## Precomputed Cache

The API includes a precomputed cache system for frequently accessed data to improve response times. Precomputed cache data exists for:

- **Variables:** `thetao` (temperature) and `so` (salinity)
- **Strides:** 2, 4, and 8
- **Spatial regions:** 37 predefined regions specified in `cached_data/cache_api_requests.log`

The list of API calls (lat/lon coordinates) was provided by Janez from the Frontend team and is stored in `cached_data/cache_api_requests.log`. The cache system automatically matches incoming requests with normalized spatial bounds (rounded to 4 decimal places) to handle floating-point precision differences.

**Precomputing the cache:**

⚠️ **Warning:** Precomputing the cache takes approximately 2 hours to complete (222 API requests). Only run this script if approved by the backend team, as it will regenerate all cache files.

```bash
python cached_data/precompute_caching_dataset.py
```

This generates 222 cache files (37 regions × 2 variables × 3 strides) in `cached_data/precomputed_cache/` and creates a `cache_index.json` file for fast lookup. The `/subset` endpoint automatically uses cached responses when available, significantly reducing response times for matching requests.

**Testing cache performance:**
```bash
python test/test_precomputed_cache.py
```

Env knobs (prefix `APP_`):
- `APP_ZARR_DIR` (default `data/zarr`)
- `APP_SUBSET_MAX_CELLS` (default 400000)
- `APP_CACHE_SIZE` (default 16)

## Visualize data

Beautiful visualization script that calls the API and displays data on a map:

**Note:** Since we're using monthly data for 2011, use the `--time` flag with format `"2011-MM-01"` where MM is the month with leading zeros (e.g., `"2011-01-01"` for January, `"2011-07-01"` for July). The day doesn't matter since data is monthly, but always use `01` with the month (or any other day for that matter, it should't affect the returned data).

Use the `--stride` flag to lower the resolution for faster downloads (e.g., `--stride 4` for every 4th point).

```bash
# Global surface temperature (January 2011)
python visualize_data.py --variable thetao --depth 0 --time 2011-01-01

# Mediterranean Sea (July 2011) with lower resolution
python visualize_data.py --variable thetao --min-lon -5 --max-lon 36 --min-lat 30 --max-lat 46 --depth 0 --time 2011-07-01 --stride 4

# Asia region temperature (July 2011)
python visualize_data.py --variable so --depth 0 --time 2011-07-01 --stride 2 --max-lat 14 --min-lat -14 --min-lon 90 --max-lon 155

# Save to file (December 2011)
python visualize_data.py --variable so --time 2011-12-01 --output salinity_map.png
```

For better maps, install cartopy: `conda install -c conda-forge cartopy`

## Notes

- The API opens Zarr lazily; ensure `scripts/download_zarr_direct.py` has been run beforehand.
- Data is monthly for 2011 only (12 months: January through December).
- When using the API or `visualize_data.py`, specify dates as `"2011-MM-01"` (day doesn't matter, always use 01).
- Use `stride` parameter to reduce resolution for faster processing of large regions.
- Responses are limited by `APP_SUBSET_MAX_CELLS` to avoid huge payloads.
- Cached responses are kept in-process only.

