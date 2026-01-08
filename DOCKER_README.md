# Docker Setup Guide

This guide will help you run the Salinity & Temperature API using Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- Copernicus Marine credentials (register for free at: https://data.marine.copernicus.eu/register)

## Quick Start

1. **Create a `.env` file** in the project root with your credentials:
   ```bash
   COPERNICUSMARINE_USERNAME=your_username
   COPERNICUSMARINE_PASSWORD=your_password
   API_PORT=8000  # Optional: change the port (default: 8000)
   ```

2. **Build and start the container**:
   ```bash
   docker-compose up --build
   ```

   This will:
   - Build the Docker image
   - Download data from Copernicus Marine ARCO (10-30 minutes on first run)
   - Start the FastAPI server on http://localhost:8000

3. **Access the API**:
   - Health check: http://localhost:8000/health
   - API docs: http://localhost:8000/docs
   - Example query: http://localhost:8000/subset?dataset=reanalysis&variable=thetao&min_lon=-10&max_lon=-5&min_lat=30&max_lat=35&time=2011-07-01&depth=5&fmt=json

## How It Works

The `entrypoint.sh` script:
1. Checks if data already exists in `/app/data/zarr/reanalysis.zarr`
2. If not, downloads it using `scripts/download_zarr_direct.py`
3. Starts the uvicorn server

The downloaded data is persisted in a Docker volume (`./data/zarr`), so it won't be re-downloaded on subsequent container restarts.

## Managing the Container

**Start the container** (after first build):
```bash
docker-compose up
```

**Start in detached mode** (background):
```bash
docker-compose up -d
```

**View logs**:
```bash
docker-compose logs -f
```

**Stop the container**:
```bash
docker-compose down
```

**Rebuild the container** (after code changes):
```bash
docker-compose up --build
```

**Re-download data** (delete existing data and restart):
```bash
rm -rf ./data/zarr/reanalysis.zarr
docker-compose restart
```

## Data Persistence

The downloaded Zarr data is stored in `./data/zarr/` on your host machine and mounted into the container. This means:
- Data persists between container restarts
- You can inspect the data on your host machine
- First download takes 10-30 minutes, but subsequent starts are instant

## Environment Variables

You can customize the behavior using environment variables in your `.env` file:

- `COPERNICUSMARINE_USERNAME`: Your Copernicus Marine username (required)
- `COPERNICUSMARINE_PASSWORD`: Your Copernicus Marine password (required)
- `API_PORT`: Port to expose the API on (default: 8000)
- `USE_LOCAL_ZARR`: Set to `true` to use local Zarr files (default in Docker)
- `APP_ZARR_DIR`: Directory for Zarr stores (default: `/app/data/zarr`)
- `APP_SUBSET_MAX_CELLS`: Maximum cells per request (default: 400000)
- `APP_CACHE_SIZE`: Response cache size (default: 16)

**Example `.env` file:**
```bash
COPERNICUSMARINE_USERNAME=my_username
COPERNICUSMARINE_PASSWORD=my_password
API_PORT=8080
```

## Troubleshooting

**Data download fails:**
- Check your Copernicus Marine credentials in `.env`
- Ensure you have internet connectivity
- Check disk space (download is ~1-2 GB)

**Port 8000 already in use:**
- Add `API_PORT` to your `.env` file:
  ```bash
  API_PORT=8080
  ```
- Or set it when starting:
  ```bash
  API_PORT=8080 docker-compose up
  ```

**Container exits immediately:**
- Check logs: `docker-compose logs`
- Ensure `.env` file exists with valid credentials

## Development

To make changes to the code while the container is running:
1. Stop the container: `docker-compose down`
2. Make your changes
3. Rebuild and restart: `docker-compose up --build`

Alternatively, you can mount the code as a volume for live reloading (add to `docker-compose.yml`):
```yaml
volumes:
  - ./data/zarr:/app/data/zarr
  - ./app:/app/app  # Mount app directory for live reload
```

