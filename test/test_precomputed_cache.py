#!/usr/bin/env python3
"""
Test script for precomputed cache performance.

Reads API requests from cache_api_requests.log and tests them with strides 2, 4, 8
to verify cache matching works correctly with original (non-normalized) coordinates.
"""

import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

# Constants
API_BASE_URL = "http://localhost:8000"
LOG_FILE = Path(__file__).parent.parent / "cached_data" / "cache_api_requests.log"

# Strides to test (matching the precomputed cache)
STRIDES = [2, 4, 8]

# Variables to test (matching the precomputed cache)
VARIABLES = ["thetao", "so"]


def check_api_server() -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def parse_url_line(line: str) -> dict | None:
    """Parse a URL line from the log file and extract parameters."""
    line = line.strip()
    if not line:
        return None

    # Parse the URL
    parsed = urlparse(line)
    if parsed.path != "/subset":
        return None

    # Parse query parameters
    params = parse_qs(parsed.query)

    # Extract parameters (parse_qs returns lists, take first element)
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
            # Note: stride from log is ignored, we test with STRIDES instead
        }
    except (KeyError, ValueError, IndexError) as e:
        print(f"Warning: Error parsing line '{line}': {e}")
        return None


def load_log_entries() -> list[dict]:
    """Load and parse API requests from log file."""
    if not LOG_FILE.exists():
        print(f"❌ ERROR: Log file not found: {LOG_FILE}")
        sys.exit(1)

    try:
        with open(LOG_FILE, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        entries = []
        for line in lines:
            params = parse_url_line(line)
            if params:
                entries.append(params)

        return entries
    except Exception as e:
        print(f"❌ ERROR: Failed to load log file: {e}")
        sys.exit(1)


def make_api_request(params: dict, stride: int) -> tuple[float, bool, str]:
    """
    Make API request with given parameters and stride, return (elapsed_time, success, error_message).
    
    Args:
        params: Dictionary with API parameters (variable, spatial bounds, time, depth)
        stride: Stride value to use in the request
    
    Returns:
        (elapsed_time_in_seconds, success_bool, error_message)
    """
    api_params = {
        "dataset": params["dataset"],
        "variable": params["variable"],
        "min_lon": params["min_lon"],
        "max_lon": params["max_lon"],
        "min_lat": params["min_lat"],
        "max_lat": params["max_lat"],
        "time": params["time"],
        "depth": params["depth"],
        "stride": stride,
        "fmt": "json",
    }

    start_time = time.time()
    try:
        response = requests.get(f"{API_BASE_URL}/subset", params=api_params, timeout=300)
        end_time = time.time()
        elapsed_time = end_time - start_time

        response.raise_for_status()
        # Verify response contains expected data structure
        data = response.json()
        if "data" not in data or "shape" not in data:
            return (elapsed_time, False, "Invalid response structure")

        return (elapsed_time, True, "")
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        return (elapsed_time, False, str(e))




def main():
    """Main test function."""
    print("=" * 70)
    print("🧪 Testing precomputed cache performance")
    print("=" * 70)

    # Check if API server is running
    print("Checking if API server is running...")
    if not check_api_server():
        print("❌ ERROR: API server is not running!")
        print("   Please start the server first: uvicorn app.main:app --reload")
        sys.exit(1)
    print("✅ API server is running")
    print()

    # Load log entries
    print(f"📖 Loading API requests from log file: {LOG_FILE}")
    log_entries = load_log_entries()
    total_log_entries = len(log_entries)
    print(f"✅ Loaded {total_log_entries} log entries")
    print()

    if total_log_entries == 0:
        print("❌ ERROR: No valid entries found in log file")
        sys.exit(1)

    # Calculate total requests: each log entry tested with each variable and each stride
    total_requests = total_log_entries * len(VARIABLES) * len(STRIDES)
    print(f"📊 Will test {total_requests} requests ({total_log_entries} log entries × {len(VARIABLES)} variables × {len(STRIDES)} strides)")
    print()

    print("=" * 70)
    print(f"🔄 Starting to test {total_requests} API requests...")
    print("=" * 70)
    print()

    # Track statistics
    times = []
    successful = 0
    failed = 0

    # Test each log entry with each variable and each stride
    request_num = 0
    for entry_idx, params in enumerate(log_entries):
        # Use original coordinates from log entry, but test with both variables
        for variable in VARIABLES:
            # Create params with the variable we want to test
            test_params = params.copy()
            test_params["variable"] = variable
            
            for stride in STRIDES:
                request_num += 1
                print(f"[{request_num}/{total_requests}] Testing {variable}, stride={stride}, log_entry={entry_idx}...", end=" ")

                elapsed_time, success, error_msg = make_api_request(test_params, stride)

            times.append(elapsed_time)

            if success:
                successful += 1
                # Format time as seconds with 1 decimal place (e.g., "0.1", "1.2")
                print(f"Time: {elapsed_time:.1f}s")
            else:
                failed += 1
                print(f"FAILED - Time: {elapsed_time:.1f}s - Error: {error_msg}")

    print()
    print("=" * 70)
    print("📊 Summary")
    print("=" * 70)

    if times:
        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)
        total_time = sum(times)
    else:
        min_time = max_time = avg_time = total_time = 0.0

    print(f"- Total requests: {total_requests}")
    print(f"- Successful: {successful}")
    print(f"- Failed: {failed}")
    print(f"- Min time: {min_time:.1f}s")
    print(f"- Max time: {max_time:.1f}s")
    print(f"- Avg time: {avg_time:.1f}s")
    print(f"- Total time: {total_time:.1f}s")

    if failed > 0:
        print()
        print("⚠️  Warning: Some requests failed. Check errors above.")
        sys.exit(1)
    else:
        print()
        print("✅ All tests passed successfully!")
        print()
        print("💡 Tip: Check server logs to verify [CACHE HIT] messages")


if __name__ == "__main__":
    main()
