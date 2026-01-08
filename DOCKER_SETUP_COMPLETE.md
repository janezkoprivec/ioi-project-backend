# Docker Setup - Complete Summary

## What Was Created

Your repository is now Docker-ready! Here's what was added:

### Core Docker Files
1. **`Dockerfile`** - Defines the Python container image
2. **`docker-compose.yml`** - Orchestrates the container with proper environment and volumes
3. **`entrypoint.sh`** - Startup script that downloads data then starts the server
4. **`.dockerignore`** - Excludes unnecessary files from the Docker image

### Documentation
5. **`DOCKER_README.md`** - Detailed Docker usage guide
6. **`EXAMPLES.md`** - API usage examples and test queries
7. **`test_docker_setup.sh`** - Script to verify the setup is working

### Code Updates
8. **`app/main.py`** - Updated to support both local and remote data sources via `USE_LOCAL_ZARR` environment variable

## How to Use

### Step 1: Create `.env` file

Create a file named `.env` in the project root:

```bash
COPERNICUSMARINE_USERNAME=your_username
COPERNICUSMARINE_PASSWORD=your_password
API_PORT=8000  # Optional: change the port (default: 8000)
```

Get credentials from: https://data.marine.copernicus.eu/register

### Step 2: Start the container

```bash
docker-compose up --build
```

**First run:** Downloads ~1-2GB of data (10-30 minutes)
**Subsequent runs:** Instant startup (data is persisted)

### Step 3: Test the API

Once you see "Uvicorn running on http://0.0.0.0:8000", the API is ready!

Run the test script:
```bash
./test_docker_setup.sh
```

Or test manually:
```bash
curl http://localhost:8000/health
```

## What Happens When You Start

The container follows this sequence:

1. **Check for existing data** at `/app/data/zarr/reanalysis.zarr`
2. **If data doesn't exist:**
   - Runs `python3 scripts/download_zarr_direct.py --overwrite`
   - Downloads from Copernicus Marine ARCO service
   - Saves to local Zarr store
3. **Start the server:**
   - Runs `uvicorn app.main:app --host 0.0.0.0 --port 8000`
   - Loads data from local Zarr files (much faster than remote)
   - API available at http://localhost:8000

## Data Persistence

The downloaded data is stored in `./data/zarr/` on your host machine and mounted into the container as a Docker volume. This means:

- ✅ Data persists between container restarts
- ✅ No re-download needed after first run
- ✅ Can be inspected/used outside Docker
- ✅ Can be backed up or shared

## Quick Commands

```bash
# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Restart after code changes
docker-compose up --build

# Delete data and re-download
rm -rf ./data/zarr/reanalysis.zarr
docker-compose restart
```

## API Endpoints

Once running, visit:
- **Interactive docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Data subset:** http://localhost:8000/subset (see EXAMPLES.md)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ Docker Container                                    │
│                                                     │
│  ┌──────────────┐                                  │
│  │ entrypoint.sh│  (1) Check if data exists        │
│  └──────┬───────┘                                  │
│         │                                           │
│         ├─(if needed)─> download_zarr_direct.py    │
│         │                    ↓                      │
│         │            [Download from Copernicus]    │
│         │                    ↓                      │
│         │            /app/data/zarr/ ←─────────────┼─ Volume Mount
│         │                    ↑                      │    to ./data/zarr/
│         │                                           │
│         └──> uvicorn app.main:app                  │
│                     ↓                               │
│              FastAPI Server (port 8000)             │
│                     ↑                               │
│              [Reads local Zarr]                     │
└─────────────────────┼───────────────────────────────┘
                      │
                      └─> Exposed to localhost:8000
```

## Environment Variables

Set in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `COPERNICUSMARINE_USERNAME` | - | Your Copernicus username (required) |
| `COPERNICUSMARINE_PASSWORD` | - | Your Copernicus password (required) |
| `API_PORT` | `8000` | Port to expose the API on |
| `USE_LOCAL_ZARR` | `true` | Use local Zarr files instead of remote |
| `APP_ZARR_DIR` | `/app/data/zarr` | Directory for Zarr stores |
| `APP_SUBSET_MAX_CELLS` | `400000` | Maximum cells per request |
| `APP_CACHE_SIZE` | `16` | Response cache size |

## Troubleshooting

### Container exits immediately
```bash
# Check logs for errors
docker-compose logs

# Common issues:
# - Missing .env file
# - Invalid credentials
# - Permission issues with data directory
```

### Data download fails with "Killed" error
This means the container ran out of memory:

```bash
# Solution 1: Increase Docker Desktop memory
# Go to: Docker Desktop → Settings → Resources → Memory
# Set to 4GB or more, then restart Docker

# Solution 2: Check system memory
free -h  # On Linux
docker stats  # Monitor container memory

# Solution 3: Uncomment memory limits in docker-compose.yml:
# mem_limit: 4g
# memswap_limit: 4g
```

### Data download fails (other reasons)
```bash
# Check credentials are correct
cat .env

# Check network connectivity
docker-compose exec api ping -c 3 google.com

# Try manual download
docker-compose exec api python3 scripts/download_zarr_direct.py --overwrite
```

### Port already in use
Add to your `.env` file:
```bash
API_PORT=8080
```
Or set when starting:
```bash
API_PORT=8080 docker-compose up
```

### Out of disk space
The Zarr data is ~1-2 GB. Check available space:
```bash
df -h .
```

## Next Steps

1. ✅ Container is running
2. ✅ Data is downloaded
3. ✅ API is responding

Now try:
- Read `EXAMPLES.md` for API usage examples
- Visit http://localhost:8000/docs for interactive API documentation
- Run queries using curl or Python (see EXAMPLES.md)
- Integrate the API into your application

## Need Help?

- Docker issues: See `DOCKER_README.md`
- API usage: See `EXAMPLES.md`
- General setup: See `README.md`
- Test setup: Run `./test_docker_setup.sh`

Enjoy your containerized ocean data API! 🌊🐳

