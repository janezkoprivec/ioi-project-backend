# API Usage Examples

Once your Docker container is running, you can test these API endpoints:

## Health Check

Check if the API is running and data is loaded:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "datasets": {
    "reanalysis": {
      "loaded": true,
      "variables": ["so", "thetao"]
    }
  }
}
```

## Get Temperature Data (JSON)

Get sea surface temperature for a region in the Atlantic Ocean (January 2011):

```bash
curl "http://localhost:8000/subset?dataset=reanalysis&variable=thetao&min_lon=-80&max_lon=-60&min_lat=20&max_lat=40&time=2011-01-01&depth=0&stride=10&fmt=json"
```

Parameters:
- `variable=thetao`: Sea temperature (use `so` for salinity)
- `min_lon=-80&max_lon=-60`: Longitude range
- `min_lat=20&max_lat=40`: Latitude range
- `time=2011-01-01`: Time (2011 data only, use format YYYY-MM-01)
- `depth=0`: Surface depth
- `stride=10`: Every 10th point (lower resolution, faster)
- `fmt=json`: JSON format (use `netcdf` for NetCDF file)

## Get Salinity Data (NetCDF)

Download salinity data for the Mediterranean Sea (July 2011) as NetCDF file:

```bash
curl "http://localhost:8000/subset?dataset=reanalysis&variable=so&min_lon=-5&max_lon=36&min_lat=30&max_lat=46&time=2011-07-01&depth=0&stride=5&fmt=netcdf" -o mediterranean_salinity.nc
```

## Get Mean Value for Region

Get the mean temperature or salinity for predefined regions (world or europe):

```bash
# World mean temperature in January 2011
curl "http://localhost:8000/mean-region?region=world&variable=thetao&time=2011-01&stride=6"

# Europe mean salinity in July 2011
curl "http://localhost:8000/mean-region?region=europe&variable=so&time=2011-07&stride=2"
```

Parameters:
- `region`: `"world"` (global) or `"europe"`
- `variable`: `"thetao"` (temperature) or `"so"` (salinity)
- `time`: Time in format `YYYY-MM` or `YYYY-MM-DD` (e.g., `"2011-01"` or `"2011-01-15"`)
- `stride`: Spatial decimation factor (1-50). Use 4-8 for world, 2-4 for europe to avoid size limits.

Expected response:
```json
{
  "region": "world",
  "variable": "thetao",
  "time": "2011-01",
  "mean": 15.234,
  "stride": 6
}
```

## Using Python

```python
import requests
import json

# Health check
response = requests.get("http://localhost:8000/health")
print(json.dumps(response.json(), indent=2))

# Get temperature data
params = {
    "dataset": "reanalysis",
    "variable": "thetao",
    "min_lon": -10,
    "max_lon": -5,
    "min_lat": 30,
    "max_lat": 35,
    "time": "2011-07-01",
    "depth": 0,
    "stride": 5,
    "fmt": "json"
}

response = requests.get("http://localhost:8000/subset", params=params)
data = response.json()

print(f"Shape: {data['shape']}")
print(f"Dimensions: {data['dims']}")
print(f"Coordinates: {list(data['coords'].keys())}")

# Get mean value for a region
params = {
    "region": "world",
    "variable": "thetao",
    "time": "2011-01",
    "stride": 6
}

response = requests.get("http://localhost:8000/mean-region", params=params)
result = response.json()
print(f"World mean temperature in January: {result['mean']}°C")
```

## Interactive API Documentation

Visit http://localhost:8000/docs for interactive Swagger UI documentation where you can try all endpoints directly in your browser.

## Available Variables

- `so`: Sea salinity (PSU - Practical Salinity Unit)
- `thetao`: Sea water potential temperature (°C)

## Time Range

The dataset includes monthly data for **2011 only** (12 months):
- January 2011: `time=2011-01-01`
- February 2011: `time=2011-02-01`
- March 2011: `time=2011-03-01`
- ...
- December 2011: `time=2011-12-01`

The day in the date doesn't matter (monthly resolution), but always use `01` for consistency.

## Performance Tips

1. **Use stride for large regions**: Higher stride values (5-50) download much faster
2. **Limit spatial extent**: Smaller regions return faster
3. **Results are cached**: Repeated identical queries return instantly
4. **NetCDF vs JSON**: NetCDF is more efficient for large datasets

## Troubleshooting

**Empty response or error:**
- Check that data download completed successfully (check Docker logs)
- Ensure coordinates are within global bounds: lon [-180, 180], lat [-90, 90]
- Use time values within 2011 range
- Try with larger stride value for faster testing

**Timeout:**
- Reduce region size or increase stride
- Check `APP_SUBSET_MAX_CELLS` limit in docker-compose.yml

