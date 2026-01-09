from __future__ import annotations

import io
import os
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

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
dataset_manager = DatasetManager(zarr_dir=settings.zarr_dir) if USE_LOCAL_ZARR else DatasetManager()
response_cache: LRUCache[Tuple, bytes] = LRUCache(maxsize=settings.cache_size)


def get_coord_name(ds: xr.Dataset, candidates) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise HTTPException(status_code=400, detail=f"None of {candidates} found in coords {list(ds.coords)}")


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
    return "Salinity & Temperature API. See /health and /subset."


@app.get("/subset", response_model=None)
def subset(
    dataset: str = Query("reanalysis", description="Dataset key: reanalysis."),
    variable: str = Query(..., description="Variable to extract: so (salinity) or thetao (temperature)."),
    min_lon: float = Query(..., ge=-180, le=180),
    max_lon: float = Query(..., ge=-180, le=180),
    min_lat: float = Query(..., ge=-90, le=90),
    max_lat: float = Query(..., ge=-90, le=90),
    time: str | None = Query(None, description="ISO timestamp or YYYY-MM-DD; nearest match used."),
    depth: float | None = Query(None, description="Depth value; nearest match used."),
    stride: int = Query(1, ge=1, le=50, description="Spatial decimation factor (1=full res, 4=every 4th point). Higher = faster but lower resolution."),
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
    depth_name = get_coord_name(ds, ("depth", "lev", "level")) if "depth" in ds.dims else None

    # Select spatial region first (slices don't use method)
    data_array = ds[variable].sel({
        lat_name: slice(min_lat, max_lat),
        lon_name: slice(min_lon, max_lon),
    })
    
    # Apply spatial decimation/stride if requested (reduces resolution)
    if stride > 1:
        data_array = data_array.isel({
            lat_name: slice(None, None, stride),
            lon_name: slice(None, None, stride),
        })
    
    # Then select time/depth with nearest neighbor matching
    point_sel = {}
    if time:
        try:
            parsed_time = np.datetime64(time)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid time format: {time}") from exc
        point_sel[time_name] = parsed_time
    if depth is not None and depth_name:
        point_sel[depth_name] = depth
    
    if point_sel:
        data_array = data_array.sel(point_sel, method="nearest")

    enforce_size_limit(data_array, settings.subset_max_cells)

    cache_key = (dataset, variable, min_lon, max_lon, min_lat, max_lat, time, depth, stride, fmt)
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
            headers={"Content-Disposition": f'attachment; filename="{dataset}_{variable}.nc"'},
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
            lat_name: loaded[lat_name].astype(float).values.tolist() if lat_name in loaded.coords else None,
            lon_name: loaded[lon_name].astype(float).values.tolist() if lon_name in loaded.coords else None,
            time_name: loaded[time_name].astype(str).values.tolist() if time_name in loaded.coords else None,
            depth_name: loaded[depth_name].astype(float).values.tolist() if depth_name and depth_name in loaded.coords else None,
        },
        "data": data_list,
    }
    response_cache.put(cache_key, payload_json)
    return JSONResponse(content=payload_json)

