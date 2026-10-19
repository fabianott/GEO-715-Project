"""
Simplified MVP: points and DEM both in EPSG:2056 (LV95, meters).
Ring means of vegetation height, sampled along N_RAYS rays around a point.
pip install geopandas rasterio shapely numpy
"""
import numpy as np
import geopandas as gpd
import rasterio

GPKG = "points.gpkg"
TIF = "dem.tif"
RINGS = [(0, 10)]#, (10, 20), (20, 50)]   # (inner, outer) radius in m
N_RAYS = 8                              # number of rays, evenly spread over 360°
STEP = 1.0                              # sample spacing along each ray in m (e.g. = pixel size)
WEIGHT_BY_DISTANCE = False              # True: weight samples by distance (corrects inner-edge bias)

# One point as GeoSeries
point = gpd.read_file(GPKG).geometry.iloc[0]
R_MAX = max(r_out for _, r_out in RINGS)

# Directions as compass bearings: 0° = north, 90° = east, clockwise
bearings = np.arange(N_RAYS) * 360.0 / N_RAYS

# Distances along each ray, starting after the centre
dists = np.arange(STEP, R_MAX + STEP / 2, STEP)      # 1, 2, ..., 50

# Sample positions: the centre once, then every distance on every ray
coords = [(point.x, point.y)]
sample_d = [0.0]
for b in bearings:
    rad = np.radians(b)
    for d in dists:
        coords.append((point.x + d * np.sin(rad), point.y + d * np.cos(rad)))
        sample_d.append(d)
sample_d = np.array(sample_d)

# Read heights
with rasterio.open(TIF) as src:
    heights = np.array([v[0] for v in src.sample(coords)], dtype="float64")
    if src.nodata is not None:
        heights[heights == src.nodata] = np.nan

# Mean per ring: each sample belongs to exactly one ring
for r_in, r_out in RINGS:
    lower = sample_d >= r_in if r_in == 0 else sample_d > r_in   # centre goes into the first ring
    in_ring = lower & (sample_d <= r_out)

    h = heights[in_ring]
    d = sample_d[in_ring]
    valid = ~np.isnan(h)

    if not valid.any():
        mean = np.nan
    elif WEIGHT_BY_DISTANCE and d[valid].sum() > 0:
        mean = np.average(h[valid], weights=d[valid])
    else:
        mean = h[valid].mean()

    print(f"Ring {r_in}-{r_out} m: mean = {mean:.2f} m, "
          f"n = {in_ring.sum()}, valid = {valid.sum()}")
