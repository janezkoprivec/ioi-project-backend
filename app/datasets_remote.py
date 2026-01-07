from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

import xarray as xr

# Try to import copernicusmarine for remote Zarr access
try:
    import copernicusmarine
    COPERNICUS_AVAILABLE = True
except ImportError:
    COPERNICUS_AVAILABLE = False


class RemoteDatasetManager:
    """
    Open datasets directly from Copernicus Marine ARCO (remote Zarr) service.
    No local preprocessing needed - data stays in the cloud.
    """

    # Dataset configuration: maps our names to Copernicus dataset IDs
    DATASETS = {
        "reanalysis": {
            "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1M-m",
            "service": "arco-geo-series",  # or "arco-time-series"
            "variables": ["so", "thetao"],
        },
    }

    def __init__(self):
        if not COPERNICUS_AVAILABLE:
            raise RuntimeError(
                "copernicusmarine library not installed. "
                "Install with: pip install copernicusmarine"
            )
        self._cache: Dict[str, xr.Dataset] = {}
        self._ensure_credentials()

    def _ensure_credentials(self) -> tuple[str, str]:
        """Load credentials from environment."""
        username = os.environ.get("COPERNICUSMARINE_USERNAME")
        password = os.environ.get("COPERNICUSMARINE_PASSWORD")
        if not username or not password:
            raise RuntimeError(
                "Set COPERNICUSMARINE_USERNAME and COPERNICUSMARINE_PASSWORD "
                "environment variables."
            )
        return username, password

    def list_datasets(self) -> Iterable[str]:
        return self.DATASETS.keys()

    def get_dataset(self, name: str) -> xr.Dataset:
        """Open a remote Zarr dataset via Copernicus Marine ARCO service."""
        if name not in self.DATASETS:
            raise KeyError(f"Unknown dataset '{name}'. Known: {list(self.DATASETS)}")
        
        if name in self._cache:
            return self._cache[name]

        cfg = self.DATASETS[name]
        username, password = self._ensure_credentials()
        
        # Open remote Zarr store
        ds = copernicusmarine.open_dataset(
            dataset_id=cfg["dataset_id"],
            service=cfg.get("service", "arco-geo-series"),
            username=username,
            password=password,
        )
        
        # Filter to only the variables we want (optional)
        if cfg.get("variables"):
            available = [v for v in cfg["variables"] if v in ds.data_vars]
            if available:
                ds = ds[available]
        
        self._cache[name] = ds
        return ds

    def available_variables(self, name: str) -> Iterable[str]:
        ds = self.get_dataset(name)
        return ds.data_vars.keys()

    def clear_cache(self) -> None:
        """Close all cached datasets."""
        for ds in self._cache.values():
            ds.close()
        self._cache.clear()
