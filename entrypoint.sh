#!/bin/bash
set -e

echo "========================================="
echo "Starting Data Download & API Server"
echo "========================================="

# Check if data already exists
if [ -d "/app/data/zarr/reanalysis.zarr" ] && [ "$(ls -A /app/data/zarr/reanalysis.zarr)" ]; then
    echo "✓ Data already exists at /app/data/zarr/reanalysis.zarr"
    echo "  Skipping download. To re-download, delete the data directory."
else
    echo "Downloading data from Copernicus Marine ARCO..."
    echo "This may take 10-30 minutes depending on network speed."
    echo ""
    
    # Run the download script
    python3 scripts/download_zarr_direct.py --overwrite
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Data download completed successfully!"
    else
        echo ""
        echo "✗ Data download failed. Check credentials and network connection."
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

