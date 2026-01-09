#!/usr/bin/env python3
"""
Precompute caching dataset script.

Reads cache_api_requests.log and makes API calls for both thetao and so variables
with strides 2, 4, and 8, saving responses to JSON files in precomputed_cache/ subdirectory.

Total: 37 regions × 2 variables × 3 strides = 222 cache files
"""

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

# Constants
API_BASE_URL = "http://localhost:8000"
CACHE_DIR = Path(__file__).parent / "precomputed_cache"
LOG_FILE = Path(__file__).parent / "cache_api_requests.log"
INDEX_FILE = CACHE_DIR / "cache_index.json"

# Variables for caching (thetao and so)
VARIABLES = ["thetao", "so"]

# Strides to precompute cache for
STRIDES = [2, 4, 8]


def parse_url_line(line: str) -> dict | None:
    """Parse a URL line from the log file and extract parameters (stride is ignored, we use STRIDES constant)."""
    line = line.strip()
    if not line:
        return None

    # Parse the URL
    parsed = urlparse(line)
    if parsed.path != "/subset":
        print(f"Warning: Skipping non-subset URL: {line}")
        return None

    # Parse query parameters
    params = parse_qs(parsed.query)

    # Extract parameters (parse_qs returns lists, take first element)
    # Note: stride from log is ignored, we use STRIDES constant instead
    try:
        return {
            "dataset": params.get("dataset", ["reanalysis"])[0],
            "variable": params.get("variable", [None])[0],
            "min_lon": float(params.get("min_lon", [None])[0]),
            "max_lon": float(params.get("max_lon", [None])[0]),
            "min_lat": float(params.get("min_lat", [None])[0]),
            "max_lat": float(params.get("max_lat", [None])[0]),
            "time": params.get("time", [None])[0],
            "depth": float(params.get("depth", [0])[0]) if params.get("depth", [None])[0] is not None else 0,
            # stride is not extracted from log, we use STRIDES constant
            "fmt": params.get("fmt", ["json"])[0],
        }
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error parsing line '{line}': {e}")
        return None


def normalize_spatial_bounds(params: dict) -> dict:
    """Normalize spatial bounds to .4f precision."""
    return {
        "min_lon": round(params["min_lon"], 4),
        "max_lon": round(params["max_lon"], 4),
        "min_lat": round(params["min_lat"], 4),
        "max_lat": round(params["max_lat"], 4),
    }


def make_api_request(params: dict, variable: str, stride: int) -> dict | None:
    """Make an API request with the given parameters, variable, and stride."""
    api_params = {
        "dataset": params["dataset"],
        "variable": variable,
        "min_lon": params["min_lon"],
        "max_lon": params["max_lon"],
        "min_lat": params["min_lat"],
        "max_lat": params["max_lat"],
        "time": params["time"],
        "depth": params["depth"],
        "stride": stride,
        "fmt": "json",
    }

    try:
        response = requests.get(f"{API_BASE_URL}/subset", params=api_params, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API request for {variable} with stride {stride}: {e}")
        return None


def save_cache_file(data: dict, filename: str) -> bool:
    """Save cache data to a JSON file."""
    filepath = CACHE_DIR / filename
    try:
        with open(filepath, "w") as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving cache file {filename}: {e}")
        return False


def check_api_server() -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def main():
    """Main function to precompute cache dataset."""
    print("=" * 70)
    print("🚀 Starting precompute caching dataset script")
    print("=" * 70)

    # Check if API server is running
    print("Checking if API server is running...")
    if not check_api_server():
        print("❌ ERROR: API server is not running!")
        print(f"   Please start the server first: uvicorn app.main:app --reload")
        sys.exit(1)
    print("✅ API server is running")

    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Cache directory: {CACHE_DIR}")

    # Read log file
    if not LOG_FILE.exists():
        print(f"❌ ERROR: Log file not found: {LOG_FILE}")
        sys.exit(1)

    print(f"📖 Reading log file: {LOG_FILE}")
    with open(LOG_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"📊 Found {len(lines)} entries in log file")

    # Parse all entries
    parsed_entries = []
    for idx, line in enumerate(lines):
        params = parse_url_line(line)
        if params:
            parsed_entries.append((idx, params))

    print(f"✅ Successfully parsed {len(parsed_entries)} entries")

    # Prepare index entries
    index_entries = []
    total_requests = len(parsed_entries) * len(VARIABLES) * len(STRIDES)
    completed = 0
    failed = 0

    print()
    print("=" * 70)
    print(f"🔄 Starting to fetch {total_requests} API requests...")
    print(f"   (37 regions × {len(VARIABLES)} variables × {len(STRIDES)} strides = {total_requests} requests)")
    print("=" * 70)

    # Process each entry for both variables and all strides
    for entry_idx, params in parsed_entries:
        for variable in VARIABLES:
            for stride in STRIDES:
                completed += 1
                print(f"[{completed}/{total_requests}] Fetching {variable} for entry {entry_idx} with stride {stride}...")

                # Make API request
                data = make_api_request(params, variable, stride)

                if data is None:
                    failed += 1
                    print(f"  ❌ Failed to fetch data for {variable} with stride {stride}")
                    continue

                # Generate filename with stride
                filename = f"cache_{variable}_{entry_idx:03d}_stride{stride}.json"

                # Save cache file
                if save_cache_file(data, filename):
                    print(f"  ✅ Saved {filename}")

                    # Normalize bounds for index
                    normalized = normalize_spatial_bounds(params)

                    # Add to index
                    index_entry = {
                        "variable": variable,
                        "min_lon": normalized["min_lon"],
                        "max_lon": normalized["max_lon"],
                        "min_lat": normalized["min_lat"],
                        "max_lat": normalized["max_lat"],
                        "time": params["time"],
                        "depth": params["depth"],
                        "stride": stride,
                        "filename": filename,
                    }
                    index_entries.append(index_entry)
                else:
                    failed += 1
                    print(f"  ❌ Failed to save {filename}")

    print()
    print("=" * 70)
    print("💾 Saving cache index...")
    print("=" * 70)

    # Save index file
    index_data = {"entries": index_entries}
    if save_cache_file(index_data, "cache_index.json"):
        print(f"✅ Cache index saved: {INDEX_FILE}")
        print(f"   Total entries in index: {len(index_entries)}")
    else:
        print(f"❌ Failed to save cache index")
        sys.exit(1)

    print()
    print("=" * 70)
    print("🎉 Precompute caching completed!")
    print("=" * 70)
    print(f"✅ Successfully cached: {len(index_entries)} entries")
    if failed > 0:
        print(f"❌ Failed requests: {failed}")
    print(f"📁 Cache directory: {CACHE_DIR}")
    print(f"📄 Index file: {INDEX_FILE}")


if __name__ == "__main__":
    main()
