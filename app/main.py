from __future__ import annotations

import io
import os
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from fastapi import Depends, Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from pydantic import BaseModel

from app.cache import LRUCache
from app.config import Settings, get_settings

# Use local or remote datasets based on environment variable
USE_LOCAL_ZARR = os.getenv("USE_LOCAL_ZARR", "false").lower() in ("true", "1", "yes")

if USE_LOCAL_ZARR:
  from app.datasets import DatasetManager

  print("✓ Using local Zarr datasets")
else:
  from app.datasets_remote import RemoteDatasetManager as DatasetManager

  print("✓ Using remote Copernicus Marine ARCO datasets")

app = FastAPI(title="Salinity & Temperature API", version="0.1.0")

# Initialize singletons at import time to avoid repeated setup.
settings = get_settings()
dataset_manager = (
  DatasetManager(zarr_dir=settings.zarr_dir) if USE_LOCAL_ZARR else DatasetManager()
)
response_cache: LRUCache[Tuple, bytes] = LRUCache(maxsize=settings.cache_size)


def get_coord_name(ds: xr.Dataset, candidates) -> str:
  for name in candidates:
    if name in ds.coords:
      return name
  raise HTTPException(
    status_code=400, detail=f"None of {candidates} found in coords {list(ds.coords)}"
  )


def enforce_size_limit(da: xr.DataArray, limit: int) -> None:
  sizes = da.sizes
  total = 1
  for dim in sizes:
    total *= sizes[dim]
  if total > limit:
    raise HTTPException(
      status_code=400,
      detail=f"Requested subset is too large ({total} cells > limit {limit}). "
      "Narrow the spatial/temporal slice.",
    )


def get_region_bounds(region: str) -> Tuple[float, float, float, float]:
  """
  Get spatial bounds (min_lat, max_lat, min_lon, max_lon) for a predefined region.
  
  Args:
    region: Region name, either "world" or "europe"
    
  Returns:
    Tuple of (min_lat, max_lat, min_lon, max_lon)
    
  Raises:
    HTTPException: If region is not "world" or "europe"
  """
  region_bounds = {
    "world": (-90.0, 90.0, -180.0, 180.0),
    "europe": (29.0, 72.0, -15.0, 45.0),
  }
  
  if region not in region_bounds:
    error_msg = f"Invalid region '{region}'. Must be one of: {list(region_bounds.keys())}"
    print(f"[get_region_bounds] ERROR: {error_msg}")
    raise HTTPException(
      status_code=400,
      detail=error_msg
    )
  
  min_lat, max_lat, min_lon, max_lon = region_bounds[region]
  return (min_lat, max_lat, min_lon, max_lon)


@app.get("/health")
def health() -> Dict[str, object]:
  result = {"status": "ok", "datasets": {}}
  for name in dataset_manager.list_datasets():
    try:
      vars_ = list(dataset_manager.available_variables(name))
    except Exception as exc:  # noqa: BLE001
      vars_ = []
      result["datasets"][name] = {"loaded": False, "error": str(exc)}
      continue
    result["datasets"][name] = {"loaded": True, "variables": vars_}
  return result


@app.get("/", response_class=PlainTextResponse)
def root():
  return "Salinity & Temperature API. See /health, /subset, /mean, and /mean-region."


@app.get("/subset", response_model=None)
def subset(
  dataset: str = Query("reanalysis", description="Dataset key: reanalysis."),
  variable: str = Query(
    ..., description="Variable to extract: so (salinity) or thetao (temperature)."
  ),
  min_lon: float = Query(..., ge=-180, le=180),
  max_lon: float = Query(..., ge=-180, le=180),
  min_lat: float = Query(..., ge=-90, le=90),
  max_lat: float = Query(..., ge=-90, le=90),
  time: str | None = Query(
    None, description="ISO timestamp or YYYY-MM-DD; nearest match used."
  ),
  depth: float | None = Query("0", description="Depth value; nearest match used."),
  stride: int = Query(
    1,
    ge=1,
    le=50,
    description="Spatial decimation factor (1=full res, 4=every 4th point). Higher = faster but lower resolution.",
  ),
  fmt: str = Query("netcdf", pattern="^(netcdf|json)$"),
):
  if min_lon > max_lon or min_lat > max_lat:
    raise HTTPException(status_code=400, detail="min_* must be <= max_*")

  try:
    ds = dataset_manager.get_dataset(dataset)
  except (KeyError, FileNotFoundError) as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc

  if variable not in ds.data_vars:
    raise HTTPException(
      status_code=404,
      detail=f"Variable '{variable}' not in dataset '{dataset}'. Available: {list(ds.data_vars)}",
    )

  lat_name = get_coord_name(ds, ("lat", "latitude"))
  lon_name = get_coord_name(ds, ("lon", "longitude"))
  time_name = get_coord_name(ds, ("time",))
  depth_name = (
    get_coord_name(ds, ("depth", "lev", "level")) if "depth" in ds.dims else None
  )

  # Select spatial region first (slices don't use method)
  data_array = ds[variable].sel(
    {
      lat_name: slice(min_lat, max_lat),
      lon_name: slice(min_lon, max_lon),
    }
  )

  # Apply spatial decimation/stride if requested (reduces resolution)
  if stride > 1:
    data_array = data_array.isel(
      {
        lat_name: slice(None, None, stride),
        lon_name: slice(None, None, stride),
      }
    )

  # Then select time/depth with nearest neighbor matching
  point_sel = {}
  if time:
    try:
      parsed_time = np.datetime64(time)
    except ValueError as exc:
      raise HTTPException(
        status_code=400, detail=f"Invalid time format: {time}"
      ) from exc
    point_sel[time_name] = parsed_time
  if depth is not None and depth_name:
    point_sel[depth_name] = depth

  if point_sel:
    data_array = data_array.sel(point_sel, method="nearest")

  enforce_size_limit(data_array, settings.subset_max_cells)

  cache_key = (
    dataset,
    variable,
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    time,
    depth,
    stride,
    fmt,
  )
  cached = response_cache.get(cache_key)
  if cached is not None:
    if fmt == "netcdf":
      return StreamingResponse(io.BytesIO(cached), media_type="application/x-netcdf")
    return JSONResponse(content=cached)

  if fmt == "netcdf":
    buffer = io.BytesIO()
    data_array.to_dataset(name=variable).to_netcdf(buffer)
    payload = buffer.getvalue()
    response_cache.put(cache_key, payload)
    return StreamingResponse(
      io.BytesIO(payload),
      media_type="application/x-netcdf",
      headers={
        "Content-Disposition": f'attachment; filename="{dataset}_{variable}.nc"'
      },
    )

  # JSON path: load into memory then serialize to a light structure.
  loaded = data_array.load()

  # Convert data to numpy array and replace NaN with None for JSON compatibility
  data_array_np = loaded.astype(float).values
  data_list = np.where(np.isnan(data_array_np), None, data_array_np).tolist()

  payload_json = {
    "dims": list(loaded.dims),
    "shape": list(loaded.shape),
    "coords": {
      lat_name: loaded[lat_name].astype(float).values.tolist()
      if lat_name in loaded.coords
      else None,
      lon_name: loaded[lon_name].astype(float).values.tolist()
      if lon_name in loaded.coords
      else None,
      time_name: loaded[time_name].astype(str).values.tolist()
      if time_name in loaded.coords
      else None,
      depth_name: loaded[depth_name].astype(float).values.tolist()
      if depth_name and depth_name in loaded.coords
      else None,
    },
    "data": data_list,
  }
  response_cache.put(cache_key, payload_json)
  return JSONResponse(content=payload_json)


class Coordinate(BaseModel):
  lon: float
  lat: float


@app.post("/mean")
def mean(
  coordinates: List[Coordinate] = Body(
    ..., description="Array of {lon, lat} coordinate pairs"
  ),
  dataset: str = Query("reanalysis", description="Dataset key: reanalysis."),
  variable: str = Query(
    ..., description="Variable to extract: so (salinity) or thetao (temperature)."
  ),
  margin: float = Query(
    ...,
    gt=0,
    description="Margin in degrees to create square region around each coordinate point.",
  ),
  time: str | None = Query(
    None, description="ISO timestamp or YYYY-MM-DD; nearest match used."
  ),
  depth: float | None = Query(None, description="Depth value; nearest match used."),
) -> Dict[str, float | int]:
  """
  Compute mean value of a variable (thetao or so) for multiple coordinate points.
  For each (lon, lat) pair, creates a square region with the given margin,
  computes the mean value in that region, then returns the overall mean of all computed means.
  """
  if not coordinates:
    raise HTTPException(status_code=400, detail="Coordinates array cannot be empty")

  try:
    ds = dataset_manager.get_dataset(dataset)
  except (KeyError, FileNotFoundError) as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc

  if variable not in ds.data_vars:
    raise HTTPException(
      status_code=404,
      detail=f"Variable '{variable}' not in dataset '{dataset}'. Available: {list(ds.data_vars)}",
    )

  lat_name = get_coord_name(ds, ("lat", "latitude"))
  lon_name = get_coord_name(ds, ("lon", "longitude"))
  time_name = get_coord_name(ds, ("time",))
  depth_name = (
    get_coord_name(ds, ("depth", "lev", "level")) if "depth" in ds.dims else None
  )

  # Prepare time/depth selection if provided
  point_sel = {}
  if time:
    try:
      parsed_time = np.datetime64(time)
    except ValueError as exc:
      raise HTTPException(
        status_code=400, detail=f"Invalid time format: {time}"
      ) from exc
    point_sel[time_name] = parsed_time
  if depth is not None and depth_name:
    point_sel[depth_name] = depth

  # Compute mean for each coordinate point
  means = []
  for idx, coord in enumerate(coordinates):
    lon, lat = coord.lon, coord.lat

    print(
      f"[DEBUG] Processing point {idx + 1}/{len(coordinates)}: lon={lon}, lat={lat}"
    )

    # Validate coordinate bounds
    if not (-180 <= lon <= 180):
      print(
        f"[DEBUG] Skipping point {idx + 1}: Longitude {lon} out of range [-180, 180]"
      )
      raise HTTPException(
        status_code=400, detail=f"Longitude {lon} out of range [-180, 180]"
      )
    if not (-90 <= lat <= 90):
      print(f"[DEBUG] Skipping point {idx + 1}: Latitude {lat} out of range [-90, 90]")
      raise HTTPException(
        status_code=400, detail=f"Latitude {lat} out of range [-90, 90]"
      )

    # Calculate square region around the point
    min_lon = lon - margin
    max_lon = lon + margin
    min_lat = lat - margin
    max_lat = lat + margin

    # Clamp to valid ranges
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)

    print(
      f"[DEBUG] Selecting spatial region: lon {min_lon} to {max_lon}, lat {min_lat} to {max_lat}"
    )

    # Select spatial region
    data_array = ds[variable].sel(
      {
        lat_name: slice(min_lat, max_lat),
        lon_name: slice(min_lon, max_lon),
      }
    )

    # Apply time/depth selection if provided
    if point_sel:
      print("[DEBUG] Including time/depth selection:", point_sel)
      data_array = data_array.sel(point_sel, method="nearest")

    print("[DEBUG] Waiting for database/load for selected region...")
    # Compute mean of the region (handling NaN values by excluding them)
    loaded = data_array.load()
    print("[DEBUG] Data loaded for point.")

    # Use nanmean to handle any NaN/missing values
    region_mean = float(np.nanmean(loaded.values))

    # Only include valid (non-NaN) means
    if not np.isnan(region_mean):
      print(f"[DEBUG] Mean for point {idx + 1}: {region_mean}")
      means.append(region_mean)
    else:
      print(f"[DEBUG] No valid data for point {idx + 1} (all NaN).")

  if not means:
    print("[DEBUG] No valid data found for any points; all computed means were NaN.")
    raise HTTPException(
      status_code=400,
      detail="No valid data found for any of the provided coordinates. "
      "All computed means were NaN (possibly due to missing data in the regions).",
    )

  # Compute final mean of all individual means
  final_mean = float(np.mean(means))
  print(f"[DEBUG] Final mean across {len(means)} valid points: {final_mean}")

  return {
    "mean": final_mean,
    "points_processed": len(means),
    "total_points": len(coordinates),
  }


@app.get("/mean-region")
def mean_region(
  region: str = Query(
    ..., pattern="^(world|europe)$", description="Region name: 'world' or 'europe'"
  ),
  variable: str = Query(
    ..., pattern="^(thetao|so)$", description="Variable to extract: 'thetao' (temperature) or 'so' (salinity)"
  ),
  time: str = Query(
    ..., description="Time in format YYYY-MM or YYYY-MM-DD (e.g., '2011-01' or '2011-01-15')"
  ),
  stride: int = Query(
    ..., ge=1, le=50, description="Spatial decimation factor (1=full res, higher=faster but lower resolution)"
  ),
  dataset: str = Query("reanalysis", description="Dataset key: reanalysis."),
) -> Dict[str, float | str | int]:
  """
  Compute mean value of a variable (thetao or so) for a predefined geographic region (world or europe).
  
  Args:
    region: Geographic region - "world" (global) or "europe"
    variable: Data variable - "thetao" (temperature) or "so" (salinity)
    time: Time point in format YYYY-MM or YYYY-MM-DD
    stride: Spatial decimation factor (e.g., 2 = every 2nd point)
    dataset: Dataset name (default: "reanalysis")
    
  Returns:
    Dictionary with region, variable, time, mean value, and stride
  """
  print(f"[mean-region] Request: region={region}, variable={variable}, time={time}, stride={stride}")
  
  try:
    # Get region bounds
    min_lat, max_lat, min_lon, max_lon = get_region_bounds(region)
    print(f"[mean-region] Region bounds: lat=[{min_lat}, {max_lat}], lon=[{min_lon}, {max_lon}]")
  except HTTPException as exc:
    print(f"[mean-region] ERROR: Invalid region '{region}' - {exc.detail}")
    raise
  
  # Depth is hardcoded to 0
  depth = 0.0
  
  try:
    ds = dataset_manager.get_dataset(dataset)
    print(f"[mean-region] Dataset loaded: {dataset}")
  except (KeyError, FileNotFoundError) as exc:
    error_msg = f"Dataset '{dataset}' not found: {str(exc)}"
    print(f"[mean-region] ERROR: {error_msg}")
    raise HTTPException(status_code=404, detail=str(exc)) from exc

  if variable not in ds.data_vars:
    error_msg = f"Variable '{variable}' not in dataset '{dataset}'. Available: {list(ds.data_vars)}"
    print(f"[mean-region] ERROR: {error_msg}")
    raise HTTPException(
      status_code=404,
      detail=error_msg,
    )

  try:
    lat_name = get_coord_name(ds, ("lat", "latitude"))
    lon_name = get_coord_name(ds, ("lon", "longitude"))
    time_name = get_coord_name(ds, ("time",))
    depth_name = (
      get_coord_name(ds, ("depth", "lev", "level")) if "depth" in ds.dims else None
    )
    print(f"[mean-region] Coordinate names: lat={lat_name}, lon={lon_name}, time={time_name}, depth={depth_name}")
  except HTTPException as exc:
    print(f"[mean-region] ERROR: Coordinate name lookup failed - {exc.detail}")
    raise

  try:
    # Select spatial region first (slices don't use method)
    data_array = ds[variable].sel(
      {
        lat_name: slice(min_lat, max_lat),
        lon_name: slice(min_lon, max_lon),
      }
    )
    print(f"[mean-region] Spatial selection completed. Shape before stride: {data_array.shape}")
  except Exception as exc:
    error_msg = f"Failed to select spatial region: {str(exc)}"
    print(f"[mean-region] ERROR: {error_msg}")
    raise HTTPException(status_code=400, detail=error_msg) from exc

  # Apply spatial decimation/stride if requested (reduces resolution)
  if stride > 1:
    try:
      data_array = data_array.isel(
        {
          lat_name: slice(None, None, stride),
          lon_name: slice(None, None, stride),
        }
      )
      print(f"[mean-region] Stride {stride} applied. Shape after stride: {data_array.shape}")
    except Exception as exc:
      error_msg = f"Failed to apply stride {stride}: {str(exc)}"
      print(f"[mean-region] ERROR: {error_msg}")
      raise HTTPException(status_code=400, detail=error_msg) from exc

  # Select time and depth with nearest neighbor matching
  point_sel = {}
  try:
    parsed_time = np.datetime64(time)
    print(f"[mean-region] Parsed time: {parsed_time}")
  except ValueError as exc:
    error_msg = f"Invalid time format: {time}. Use YYYY-MM or YYYY-MM-DD format. Error: {str(exc)}"
    print(f"[mean-region] ERROR: {error_msg}")
    raise HTTPException(
      status_code=400, detail=error_msg
    ) from exc
  point_sel[time_name] = parsed_time
  
  if depth_name:
    point_sel[depth_name] = depth
    print(f"[mean-region] Depth selection: {depth}")

  if point_sel:
    try:
      print(f"[mean-region] Selecting time/depth with: {point_sel}")
      print(f"[mean-region] DataArray dims before time/depth selection: {data_array.dims}")
      print(f"[mean-region] DataArray shape before time/depth selection: {data_array.shape}")
      data_array = data_array.sel(point_sel, method="nearest")
      print(f"[mean-region] Time/depth selection completed. Final shape: {data_array.shape}")
      print(f"[mean-region] Final dimensions: {data_array.dims}")
    except Exception as exc:
      error_msg = f"Failed to select time/depth: {str(exc)}"
      print(f"[mean-region] ERROR: {error_msg}")
      import traceback
      print(f"[mean-region] Traceback: {traceback.format_exc()}")
      raise HTTPException(status_code=400, detail=error_msg) from exc

  # Enforce size limit before loading
  try:
    sizes = data_array.sizes
    total_cells = 1
    for dim in sizes:
      total_cells *= sizes[dim]
    print(f"[mean-region] Total cells: {total_cells}, limit: {settings.subset_max_cells}")
    print(f"[mean-region] DataArray is chunked: {data_array.chunks is not None}")
    if data_array.chunks:
      print(f"[mean-region] Chunk sizes: {data_array.chunks}")
    enforce_size_limit(data_array, settings.subset_max_cells)
    print("[mean-region] Size limit check passed")
  except HTTPException as exc:
    print(f"[mean-region] ERROR: Size limit exceeded - {exc.detail}")
    raise
  except Exception as exc:
    error_msg = f"Error checking size limit: {str(exc)}"
    print(f"[mean-region] ERROR: {error_msg}")
    import traceback
    print(f"[mean-region] Traceback: {traceback.format_exc()}")
    raise HTTPException(status_code=400, detail=error_msg) from exc

  # Load the data into memory
  print("[mean-region] Starting data load... (this may take a while for large datasets)")
  try:
    loaded = data_array.load()
    print(f"[mean-region] Data loaded into memory successfully. Shape: {loaded.shape}")
  except KeyboardInterrupt:
    print("[mean-region] ERROR: Data loading was interrupted")
    raise HTTPException(status_code=500, detail="Data loading was interrupted")
  except Exception as exc:
    error_msg = f"Failed to load data: {str(exc)}"
    print(f"[mean-region] ERROR: {error_msg}")
    import traceback
    print(f"[mean-region] Traceback: {traceback.format_exc()}")
    raise HTTPException(status_code=400, detail=error_msg) from exc

  # Compute mean using nanmean to handle NaN/missing values
  try:
    mean_value = float(np.nanmean(loaded.values))
    print(f"[mean-region] Mean computed: {mean_value}")
  except Exception as exc:
    error_msg = f"Failed to compute mean: {str(exc)}"
    print(f"[mean-region] ERROR: {error_msg}")
    raise HTTPException(status_code=400, detail=error_msg) from exc

  # Check if all values were NaN
  if np.isnan(mean_value):
    error_msg = f"No valid data found for region '{region}' at time '{time}'. All values were NaN."
    print(f"[mean-region] ERROR: {error_msg}")
    raise HTTPException(
      status_code=400,
      detail=error_msg
    )

  return {
    "region": region,
    "variable": variable,
    "time": time,
    "mean": mean_value,
    "stride": stride,
  }
