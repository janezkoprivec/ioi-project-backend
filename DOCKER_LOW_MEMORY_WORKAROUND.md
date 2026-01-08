# Low Memory Workaround Guide

If you're getting "Killed" errors during data download in Docker, it means your system or Docker doesn't have enough memory allocated. Here are your options:

## Option 1: Download Data Outside Docker (Recommended for Low Memory Systems)

Download the data on your host machine first, then mount it into Docker:

### Step 1: Download data locally (requires ~2GB RAM)

```bash
# Make sure you're in the project directory
cd /path/to/project-data-playground

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Download data locally
python3 scripts/download_zarr_direct.py --overwrite
```

This will download to `./data/zarr/reanalysis.zarr` on your host machine.

### Step 2: Skip download in Docker

Modify `entrypoint.sh` to skip the download step since data already exists, or just start with:

```bash
docker-compose up
```

Docker will see the data in `./data/zarr/` (which is mounted as a volume) and skip the download!

## Option 2: Increase Docker Memory Limit

### Docker Desktop (Mac/Windows)

1. Open **Docker Desktop**
2. Click **Settings** (gear icon)
3. Go to **Resources** → **Memory**
4. Increase to **4GB or more**
5. Click **Apply & Restart**
6. Try again: `docker-compose up --build`

### Docker on Linux

Check available memory:
```bash
free -h
```

If you have enough RAM, uncomment memory limits in `docker-compose.yml`:

```yaml
mem_limit: 4g
memswap_limit: 4g
```

Then restart:
```bash
docker-compose up --build
```

## Option 3: Use Remote Data Source (No Download Needed)

Instead of downloading data, use the remote Copernicus ARCO service directly:

### Step 1: Update `docker-compose.yml`

Change:
```yaml
- USE_LOCAL_ZARR=true
```

To:
```yaml
- USE_LOCAL_ZARR=false
```

### Step 2: Update `entrypoint.sh`

Comment out or remove the data download section (lines that run `download_zarr_direct.py`).

### Step 3: Start server

```bash
docker-compose up
```

The API will now fetch data directly from Copernicus Marine ARCO service (requires credentials, slower queries but no download needed).

## Option 4: Use a Remote Server

If your local machine has limited resources:

1. **Use a cloud VM** (AWS EC2, Google Cloud, DigitalOcean, etc.) with 4GB+ RAM
2. **Clone the repo** on the VM
3. **Run docker-compose** there
4. **Expose the API** port and access it remotely

## Checking Your Current Memory

### Docker Desktop
```bash
docker stats
```

### Linux/Mac
```bash
# Total system memory
free -h

# Available memory
free -h | grep "Mem:" | awk '{print $7}'
```

### Recommended Memory Requirements

- **Minimum (no download):** 512MB - For running API only with pre-downloaded data
- **Recommended (with download):** 4GB - For downloading data inside Docker
- **Ideal:** 8GB - For smooth operation

## Which Option Should I Choose?

| Situation | Recommended Option |
|-----------|-------------------|
| Have 4GB+ RAM available | Option 2 (Increase Docker memory) |
| Have 2GB RAM, can run Python locally | Option 1 (Download outside Docker) |
| Have <2GB RAM | Option 3 (Use remote data) or Option 4 (Cloud VM) |
| Slow internet connection | Option 1 (Download once outside Docker) |
| Fast internet, limited storage | Option 3 (Use remote data) |

## Still Having Issues?

1. **Check Docker memory allocation:**
   ```bash
   docker info | grep Memory
   ```

2. **Monitor during download:**
   ```bash
   docker stats
   ```

3. **Check system logs:**
   ```bash
   dmesg | grep -i "out of memory"  # Linux
   docker-compose logs  # Docker logs
   ```

4. **Try downloading just one variable** (edit `scripts/download_zarr_direct.py`):
   ```python
   "keep_vars": ["thetao"],  # Only temperature, not salinity
   ```

Need more help? Open an issue with:
- Your system specs (RAM, OS)
- Docker memory allocation
- Error messages from `docker-compose logs`

