"""
Simplified MVP: points and DEM both in EPSG:2056 (LV95, meters).
pip install geopandas rasterio shapely numpy
"""
import numpy as np
import geopandas as gpd
import rasterio

GPKG = "points.gpkg"
TIF = "dem.tif"
LENGTH = 10.0   # profile length in m
STEP = 1.0      # sample spacing in m (e.g. = pixel size)

# One point as GeoSeries
point = gpd.read_file(GPKG).geometry.iloc[0]

# Sample positions straight north: x stays the same, y increases
dists = np.arange(0, LENGTH + STEP / 2, STEP)        # 0, 1, ..., 10
coords = [(point.x, point.y + d) for d in dists]

# Read heights
with rasterio.open(TIF) as src:
    heights = np.array([v[0] for v in src.sample(coords)], dtype="float64")
    if src.nodata is not None:
        heights[heights == src.nodata] = np.nan

print(dict(zip(dists, heights.round(2))))
print(f"Mean height along profile: {np.nanmean(heights):.2f} m")
