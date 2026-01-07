#!/usr/bin/env python3
"""
Visualize salinity/temperature data from the FastAPI backend.
Creates beautiful maps similar to Copernicus Marine viewer.
"""
import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

# Try to import cartopy for proper map projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: cartopy not installed. Install with 'conda install -c conda-forge cartopy' for better maps.")


def create_ocean_colormap():
    """Create a colormap similar to Copernicus Marine viewer."""
    # Temperature colormap: deep blue -> cyan -> yellow -> orange/red
    colors = [
        '#0d1847',  # Deep blue (cold)
        '#1e3a8a',  # Dark blue
        '#3b82f6',  # Blue
        '#06b6d4',  # Cyan
        '#10b981',  # Teal/green
        '#fbbf24',  # Yellow
        '#f59e0b',  # Orange
        '#ef4444',  # Red (warm)
        '#991b1b',  # Dark red (very warm)
    ]
    return LinearSegmentedColormap.from_list('ocean_temp', colors, N=256)


def fetch_data(
    api_url: str,
    dataset: str,
    variable: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    time: str | None = None,
    depth: float | None = None,
    stride: int = 1,
) -> xr.Dataset:
    """Fetch data from the API as NetCDF."""
    params = {
        'dataset': dataset,
        'variable': variable,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'min_lat': min_lat,
        'max_lat': max_lat,
        'stride': stride,
        'fmt': 'netcdf',
    }
    if time:
        params['time'] = time
    if depth is not None:
        params['depth'] = depth
    
    print(f"Fetching {variable} data from API...")
    print(f"  Region: [{min_lon}, {max_lon}] lon, [{min_lat}, {max_lat}] lat")
    if time:
        print(f"  Time: {time}")
    if depth is not None:
        print(f"  Depth: {depth} m")
    if stride > 1:
        print(f"  Stride: {stride} (lower resolution for faster download)")
    
    response = requests.get(f"{api_url}/subset", params=params, timeout=60)
    response.raise_for_status()
    
    # Load NetCDF from response bytes
    ds = xr.open_dataset(io.BytesIO(response.content))
    print(f"✓ Received data: {list(ds.data_vars)}, shape: {ds[variable].shape}")
    return ds


def plot_with_cartopy(ds: xr.Dataset, variable: str, title: str, output_path: Path | None = None):
    """Create a beautiful map visualization using cartopy."""
    data = ds[variable]
    
    # Get coordinate names (handle both lat/latitude, lon/longitude)
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    
    # Squeeze out singleton dimensions (time, depth if present)
    data = data.squeeze()
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    
    # Create figure with map projection
    fig = plt.figure(figsize=(16, 10), facecolor='#0a0e27')
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, color='#1a1a1a', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#404040', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#303030', zorder=2)
    
    # Plot data
    cmap = create_ocean_colormap()
    im = ax.pcolormesh(
        lons, lats, data.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading='auto',
        zorder=0,
    )
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='#404040', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label(f"{variable} ({ds[variable].attrs.get('units', '')})", 
                   color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Title
    ax.set_title(title, color='white', fontsize=16, pad=20, weight='bold')
    
    # Style
    ax.set_facecolor('#0a0e27')
    ax.spines['geo'].set_edgecolor('#404040')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='#0a0e27', bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
    
    plt.show()


def plot_simple(ds: xr.Dataset, variable: str, title: str, output_path: Path | None = None):
    """Create a simple visualization without cartopy."""
    data = ds[variable]
    
    # Get coordinate names
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    
    # Squeeze out singleton dimensions
    data = data.squeeze()
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0a0e27')
    ax.set_facecolor('#0a0e27')
    
    # Plot data
    cmap = create_ocean_colormap()
    im = ax.pcolormesh(lons, lats, data.values, cmap=cmap, shading='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label(f"{variable} ({ds[variable].attrs.get('units', '')})", 
                   color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Labels and title
    ax.set_xlabel('Longitude', color='white', fontsize=12)
    ax.set_ylabel('Latitude', color='white', fontsize=12)
    ax.set_title(title, color='white', fontsize=16, pad=20, weight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, color='#404040', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#404040')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='#0a0e27', bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ocean data from the FastAPI backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Global temperature at surface
  python visualize_data.py --variable thetao --depth 0

  # Mediterranean Sea salinity
  python visualize_data.py --variable so --min-lon -5 --max-lon 36 --min-lat 30 --max-lat 46

  # Atlantic temperature on a specific date
  python visualize_data.py --variable thetao --time 2020-06-15 --depth 100
        """,
    )
    parser.add_argument('--api-url', default='http://127.0.0.1:8000', 
                       help='API base URL (default: http://127.0.0.1:8000)')
    parser.add_argument('--dataset', default='reanalysis', 
                       help='Dataset name (default: reanalysis)')
    parser.add_argument('--variable', default='thetao', choices=['thetao', 'so'],
                       help='Variable to plot: thetao (temperature) or so (salinity)')
    parser.add_argument('--min-lon', type=float, default=-180,
                       help='Minimum longitude (default: -180)')
    parser.add_argument('--max-lon', type=float, default=180,
                       help='Maximum longitude (default: 180)')
    parser.add_argument('--min-lat', type=float, default=-90,
                       help='Minimum latitude (default: -90)')
    parser.add_argument('--max-lat', type=float, default=90,
                       help='Maximum latitude (default: 90)')
    parser.add_argument('--time', type=str, default='2020-01-15',
                       help='Time (ISO format or YYYY-MM-DD, default: 2020-01-15)')
    parser.add_argument('--depth', type=float, default=0,
                       help='Depth in meters (default: 0 = surface)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Spatial decimation (1=full res, 4=every 4th point). Use 4-10 for global maps.')
    parser.add_argument('--output', type=str,
                       help='Save figure to file (e.g., output.png)')
    
    args = parser.parse_args()
    
    try:
        # Fetch data
        ds = fetch_data(
            api_url=args.api_url,
            dataset=args.dataset,
            variable=args.variable,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            time=args.time,
            depth=args.depth,
            stride=args.stride,
        )
        
        # Create title
        var_name = 'Temperature' if args.variable == 'thetao' else 'Salinity'
        title = f"{var_name} - {args.time} - {args.depth}m depth"
        
        # Plot
        output_path = Path(args.output) if args.output else None
        if HAS_CARTOPY:
            plot_with_cartopy(ds, args.variable, title, output_path)
        else:
            plot_simple(ds, args.variable, title, output_path)
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API. Is the server running?")
        print(f"  Try: uvicorn app.main:app --reload")
        return 1
    except requests.exceptions.HTTPError as e:
        print(f"✗ API Error: {e}")
        if e.response is not None:
            print(f"  Response: {e.response.text[:200]}")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

