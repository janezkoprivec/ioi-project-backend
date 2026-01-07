from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Lightweight .env loader to avoid extra dependencies."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Always set from .env (overwrite existing)
        if key:
            os.environ[key] = value


@dataclass
class Settings:
    """Application settings loaded from environment variables with APP_ prefix."""
    
    zarr_dir: Path
    subset_max_cells: int = 400_000
    cache_size: int = 16

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables."""
        # Load .env first to ensure credentials are available
        load_dotenv()
        
        zarr_dir = os.getenv(
            "APP_ZARR_DIR",
            str(Path(__file__).resolve().parent.parent / "data" / "zarr")
        )
        subset_max_cells = int(os.getenv("APP_SUBSET_MAX_CELLS", "400000"))
        cache_size = int(os.getenv("APP_CACHE_SIZE", "16"))
        
        return cls(
            zarr_dir=Path(zarr_dir),
            subset_max_cells=subset_max_cells,
            cache_size=cache_size,
        )


def get_settings() -> Settings:
    """Get application settings (can be cached if needed)."""
    return Settings.from_env()

