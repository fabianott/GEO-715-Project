import rasterio
import numpy as np

# Eingaben
x = 2810393.7046
y = 1172708.8572

richtung = "N"

zone1 = (0, 50)
zone2 = (50, 100)

band = 2

# Richtungen
richtungen = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0)
}

dx, dy = richtungen[richtung]

# Raster öffnen
with rasterio.open("metriche_z_raster.tiff") as src:

    for zone in [zone1, zone2]:

        # Punkte alle 10 m auf der Linie
        punkte = [
            (x + dx * d, y + dy * d)
            for d in range(zone[0] + 10, zone[1] + 1, 10)
        ]

        # Werte aus Band 2 auslesen
        werte = np.array([
            v[0] for v in src.sample(punkte, indexes=band)
        ])

        # NoData entfernen
        werte = werte[werte != src.nodata]

        # Mean
        print(
            zone,
            "Mean =", np.mean(werte)
        )

    print("CRS:", src.crs)
    print("Auflösung:", src.res)
    print("Anzahl Bänder:", src.count)