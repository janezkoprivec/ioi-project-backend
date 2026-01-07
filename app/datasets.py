from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZARR_DIR = PROJECT_ROOT / "data" / "zarr"


class DatasetManager:
    """
    Lazily open Zarr datasets and keep handles cached in-process.
    Intended to be created once at app startup.
    """

    def __init__(self, zarr_dir: Path = DEFAULT_ZARR_DIR, mapping: Mapping[str, Path] | None = None):
        self.zarr_dir = Path(zarr_dir)
        if mapping:
            self.dataset_paths: Dict[str, Path] = {k: Path(v) for k, v in mapping.items()}
        else:
            self.dataset_paths = {
                "observations": self.zarr_dir / "observations.zarr",
                "reanalysis": self.zarr_dir / "reanalysis.zarr",
            }
        self._cache: Dict[str, xr.Dataset] = {}

    def list_datasets(self) -> Iterable[str]:
        return self.dataset_paths.keys()

    def get_dataset(self, name: str) -> xr.Dataset:
        if name not in self.dataset_paths:
            raise KeyError(f"Unknown dataset '{name}'. Known: {list(self.dataset_paths)}")
        if name in self._cache:
            return self._cache[name]

        path = self.dataset_paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Zarr store not found at {path}. Run scripts/prepare_zarr.py first.")

        ds = xr.open_zarr(path, consolidated=True, chunks="auto")
        self._cache[name] = ds
        return ds

    def available_variables(self, name: str) -> Iterable[str]:
        ds = self.get_dataset(name)
        return ds.data_vars.keys()

    def clear_cache(self) -> None:
        self._cache.clear()

