#!/bin/bash
set -e

echo "========================================="
echo "Starting Data Download & API Server"
echo "========================================="

# Ensure data directory exists and is writable
echo "Checking data directory..."
mkdir -p /app/data/zarr
if [ ! -w /app/data/zarr ]; then
    echo "✗ Error: /app/data/zarr is not writable"
    echo "  Check Docker volume permissions"
    exit 1
fi
echo "✓ Data directory is ready: /app/data/zarr"
echo ""

# Check if data already exists
if [ -d "/app/data/zarr/reanalysis.zarr" ] && [ "$(ls -A /app/data/zarr/reanalysis.zarr)" ]; then
    echo "✓ Data already exists at /app/data/zarr/reanalysis.zarr"
    echo "  Skipping download. To re-download, delete the data directory."
else
    echo "Downloading data from Copernicus Marine ARCO..."
    echo "This may take 10-30 minutes depending on network speed."
    echo ""
    
    # Check available memory
    if [ -f /proc/meminfo ]; then
        total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        total_mem_gb=$((total_mem_kb / 1024 / 1024))
        echo "Available memory: ${total_mem_gb}GB"
        
        if [ $total_mem_gb -lt 2 ]; then
            echo "⚠ WARNING: Low memory detected (<2GB)"
            echo "  Download may fail with 'Killed' error"
            echo "  See DOCKER_LOW_MEMORY_WORKAROUND.md for solutions:"
            echo "    1. Download data outside Docker (recommended)"
            echo "    2. Increase Docker memory to 4GB+"
            echo "    3. Use remote data source (no download)"
            echo ""
            echo "Attempting download anyway..."
        fi
    fi
    
    # Run the download script
    python3 scripts/download_zarr_direct.py --overwrite
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Data download completed successfully!"
    else
        echo ""
        echo "✗ Data download failed."
        echo ""
        echo "If you got a 'Killed' error, you have LOW MEMORY."
        echo "Solutions:"
        echo "  1. Download data outside Docker first (see DOCKER_LOW_MEMORY_WORKAROUND.md)"
        echo "  2. Increase Docker memory: Settings → Resources → Memory → 4GB+"
        echo "  3. Use remote data: Set USE_LOCAL_ZARR=false in docker-compose.yml"
        echo ""
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "Starting FastAPI server on port ${API_PORT:-8000}..."
echo "========================================="
echo ""

# Start the server (use API_PORT env var, default to 8000)
exec uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT:-8000}"

