"""
Empirical semivariogram of a raster around the kilns
=====================================================

What it does
------------
For every kiln, a square window (2 * half_width wide) is read from the raster.
Inside each window, every pixel is compared with the pixels k steps away in
four directions (east, north, north-east, north-west). Half the mean squared
difference of all pairs at a given distance is the semivariance gamma(h):

    gamma(h) = 0.5 * mean( (z(x) - z(x + h))^2 )

The pairs of all windows are pooled, so the result is one curve for all kilns.

How to read the curve
---------------------
nugget  value at very short distances (noise, detail smaller than a pixel)
sill    level where the curve flattens (= roughly the total variance)
range   distance where the curve reaches the sill: points closer than this
        are spatially autocorrelated, points further apart roughly independent

Usage in a notebook
-------------------
    from semivariogram_function import kiln_semivariogram, plot_semivariogram

    sv = kiln_semivariogram({"tif": "merged_dem.tif"})
    plot_semivariogram(sv)

To compare with control points later: {"gpkg": "controls.gpkg"}.

pip install geopandas rasterio numpy pandas matplotlib
"""
import numpy as np
import pandas as pd

DEFAULTS = {
    "gpkg": "kohlmeiler_2056.gpkg",  # points around which windows are read
    "tif": "merged_dem.tif",         # raster to analyse
    "band": 1,
    "half_width": 50,   # window = 100 m x 100 m around each point
    "max_lag": 40,      # largest distance in metres (keep it below half_width)
    "detrend": False,   # True: remove a fitted plane per window (use if the
                        # raster is absolute elevation, so the slope doesn't
                        # dominate the curve)
}


# =============================================================================
# CORE: semivariogram of one array
# =============================================================================
def _shifted_pairs(z, dr, dc):
    """Return two views of z whose pixels are (dr, dc) pixels apart (dr >= 0)."""
    rows, cols = z.shape
    if dc >= 0:
        a = z[0:rows - dr, 0:cols - dc]
        b = z[dr:rows, dc:cols]
    else:
        a = z[0:rows - dr, -dc:cols]
        b = z[dr:rows, 0:cols + dc]
    return a, b


def _remove_plane(z):
    """Subtract a least-squares plane a + b*row + c*col (NaN pixels ignored)."""
    rr, cc = np.indices(z.shape)
    ok = ~np.isnan(z)
    if ok.sum() < 3:
        return z
    A = np.column_stack([np.ones(ok.sum()), rr[ok], cc[ok]])
    coef, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    return z - (coef[0] + coef[1] * rr + coef[2] * cc)


def add_pairs(z, res, max_lag, sums):
    """
    Add all pixel pairs of array z up to max_lag metres apart to `sums`,
    a dict {distance rounded to res: [sum of squared differences, number of pairs]}.
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]   # E, N, NE, NW (in pixel steps)
    for dr, dc in directions:
        step_len = res * np.hypot(dr, dc)            # 1 pixel, or sqrt(2) on diagonals
        for k in range(1, int(max_lag / step_len) + 1):
            a, b = _shifted_pairs(z, k * dr, k * dc)
            d2 = (a - b) ** 2
            d2 = d2[~np.isnan(d2)]                   # pairs with nodata are skipped
            if d2.size == 0:
                continue
            h = round(k * step_len / res) * res      # bin to the pixel size
            s = sums.setdefault(h, [0.0, 0])
            s[0] += d2.sum()
            s[1] += d2.size
    return sums


def finish(sums):
    """Turn the accumulated sums into a table: lag, gamma, n_pairs."""
    lags = sorted(sums)
    return pd.DataFrame({
        "lag": lags,
        "gamma": [0.5 * sums[h][0] / sums[h][1] for h in lags],
        "n_pairs": [sums[h][1] for h in lags],
    })


# =============================================================================
# MAIN: windows around the kilns
# =============================================================================
def kiln_semivariogram(params=None):
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds

    params = params or {}
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
    p = {**DEFAULTS, **params}
    hw = p["half_width"]

    points = gpd.read_file(p["gpkg"])
    sums = {}
    with rasterio.open(p["tif"]) as src:
        if points.crs != src.crs:
            points = points.to_crs(src.crs)
        res = src.res[0]
        for geom in points.geometry:
            x, y = geom.x, geom.y
            win = from_bounds(x - hw, y - hw, x + hw, y + hw, src.transform)
            z = (src.read(p["band"], window=win, boundless=True, masked=True)
                    .astype("float64").filled(np.nan))   # nodata / outside -> NaN
            if p["detrend"]:
                z = _remove_plane(z)
            add_pairs(z, res, p["max_lag"], sums)

    sv = finish(sums)
    print(f"{len(points)} windows of {2 * hw} x {2 * hw} m, pixel size {res} m")
    return sv


def plot_semivariogram(sv, ax=None, label=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sv["lag"], sv["gamma"], marker="o", markersize=3, label=label)
    ax.set_xlabel("Distance between pixels (m)")
    ax.set_ylabel("Semivariance (m²)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(color="0.92")
    if label:
        ax.legend(frameon=False)
    return ax
