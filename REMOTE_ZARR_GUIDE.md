# Using Remote Zarr (ARCO) vs Local Zarr

Copernicus Marine offers **ARCO** (Analysis Ready, Cloud Optimized) services that expose data as remote Zarr stores. This means you can skip the entire local conversion process.

## Two Approaches

### Option 1: Remote Zarr (Cloud-based) ⭐ RECOMMENDED FOR GETTING STARTED

**Pros:**
- ✅ No preprocessing needed - data stays in the cloud
- ✅ No local storage required (save 15+ GB)
- ✅ Always up-to-date with latest data
- ✅ Faster development iteration
- ✅ Multiple servers can share the same data source

**Cons:**
- ❌ Network latency on every query (~100-500ms)
- ❌ Dependent on Copernicus service availability
- ❌ Potential rate limits/throttling
- ❌ Requires credentials for every request

**How to use:**
```python
# In app/main.py, replace:
from app.datasets import DatasetManager
# with:
from app.datasets_remote import RemoteDatasetManager as DatasetManager
```

That's it! The API code stays the same.

### Option 2: Local Zarr (Preprocessed)

**Pros:**
- ✅ Fast queries (no network latency)
- ✅ No dependency on external services
- ✅ Predictable performance
- ✅ No rate limits

**Cons:**
- ❌ Requires 15+ GB local storage
- ❌ 30-60 minute preprocessing step
- ❌ High memory usage during conversion
- ❌ Data becomes stale (need to re-download for updates)

**How to use:**
1. Run: `python scripts/prepare_zarr.py --overwrite`
2. Use the default `DatasetManager` in `app/main.py`

## Testing Remote Zarr

Run this to test if remote Zarr works with your credentials:

```bash
python3 -c "
import os
from pathlib import Path
import copernicusmarine

# Load .env
env_path = Path('.env')
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('\"').strip(\"'\")

# Try ARCO service
ds = copernicusmarine.open_dataset(
    dataset_id='cmems_mod_glo_phy_my_0.083deg_P1M-m',
    service='arco-geo-series',
)
print('✅ Remote Zarr works!')
print(f'Variables: {list(ds.data_vars)}')
print(f'Dims: {dict(ds.dims)}')
print(f'Chunks: {ds.chunks}')
ds.close()
"
```

## Recommendation

**Start with Remote Zarr** (Option 1):
1. It's simpler and faster to get running
2. Test your API logic without the preprocessing hassle
3. See if performance is acceptable for your use case

**Switch to Local Zarr** (Option 2) only if:
- Remote queries are too slow (>1s)
- You hit rate limits
- You need guaranteed uptime independent of Copernicus
- You're running in production with high query volumes

## Hybrid Approach

You could even support both:
```python
# app/config.py
class Settings:
    use_remote_zarr: bool = True  # Toggle via env: APP_USE_REMOTE_ZARR=false
```

Then choose the manager at startup based on config.

