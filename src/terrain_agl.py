"""Pull free terrain elevation (AWS Terrarium DEM tiles) and convert the 2011
pigeon GPS altitudes from sea-level (ASL) to height-above-ground (AGL).

The i-gotU GT-120 loggers record GPS altitude above mean sea level (confirmed:
negative values occur). To get true flying height we subtract terrain elevation
sampled from a DEM at each fix's lon/lat.

Source: s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
  elevation_m = R*256 + G + B/256 - 32768   (Mapzen/Terrarium encoding)

Outputs:
  terrain_dem.npz   cached DEM mosaic (so we don't re-fetch)
  agl_hist.json     {centers, counts} AGL histogram for the vision viewer
Run with network access (sandbox disabled).
"""
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Z = 12  # tile zoom (~24 m/px at this latitude)
BBOX = dict(lon0=-1.56, lon1=-0.77, lat0=51.62, lat1=52.07)
TILE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def lonlat_to_pixel(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    latr = np.radians(lat)
    y = (1.0 - np.arcsinh(np.tan(latr)) / np.pi) / 2.0 * n
    return x * 256.0, y * 256.0  # global pixel coords


def fetch_tile(z, x, y):
    url = TILE.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": "pigeon-dem/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        img = Image.open(io.BytesIO(r.read())).convert("RGB")
    a = np.asarray(img, dtype=np.float64)
    return a[:, :, 0] * 256.0 + a[:, :, 1] + a[:, :, 2] / 256.0 - 32768.0


def build_dem():
    if Path("terrain_dem.npz").exists():
        d = np.load("terrain_dem.npz")
        print(f"DEM cached: {d['elev'].shape}, z={int(d['z'])}")
        return d["elev"], int(d["z"]), int(d["px0"]), int(d["py0"])
    n = 2 ** Z
    xs, _ = lonlat_to_pixel(np.array([BBOX["lon0"], BBOX["lon1"]]),
                            np.array([BBOX["lat0"], BBOX["lat0"]]), Z)
    _, ys = lonlat_to_pixel(np.array([BBOX["lon0"], BBOX["lon0"]]),
                            np.array([BBOX["lat1"], BBOX["lat0"]]), Z)  # lat1=north(small y)
    xt0, xt1 = int(xs.min() // 256), int(xs.max() // 256)
    yt0, yt1 = int(ys.min() // 256), int(ys.max() // 256)
    nx, ny = xt1 - xt0 + 1, yt1 - yt0 + 1
    print(f"fetching {nx}x{ny} = {nx*ny} terrarium tiles at z={Z} ...")
    elev = np.full((ny * 256, nx * 256), np.nan)
    for j, ty in enumerate(range(yt0, yt1 + 1)):
        for i, tx in enumerate(range(xt0, xt1 + 1)):
            try:
                elev[j*256:(j+1)*256, i*256:(i+1)*256] = fetch_tile(Z, tx, ty)
            except Exception as e:
                print(f"  tile {tx},{ty} failed: {e}")
    px0, py0 = xt0 * 256, yt0 * 256
    np.savez_compressed("terrain_dem.npz", elev=elev, z=Z, px0=px0, py0=py0)
    print(f"DEM built: {elev.shape}, elev {np.nanmin(elev):.0f}..{np.nanmax(elev):.0f} m")
    return elev, Z, px0, py0


def sample_dem(elev, z, px0, py0, lon, lat):
    gx, gy = lonlat_to_pixel(lon, lat, z)
    x = gx - px0; y = gy - py0
    x0 = np.clip(np.floor(x).astype(int), 0, elev.shape[1]-2)
    y0 = np.clip(np.floor(y).astype(int), 0, elev.shape[0]-2)
    fx = np.clip(x - x0, 0, 1); fy = np.clip(y - y0, 0, 1)
    e00 = elev[y0, x0]; e10 = elev[y0, x0+1]
    e01 = elev[y0+1, x0]; e11 = elev[y0+1, x0+1]
    return (e00*(1-fx)*(1-fy) + e10*fx*(1-fy) + e01*(1-fx)*fy + e11*fx*fy)


def main():
    elev, z, px0, py0 = build_dem()
    edges = np.arange(-40, 405, 5.0)
    counts = np.zeros(len(edges)-1)
    asl_med, agl_med, n_tot = [], [], 0
    terr_at_release = []
    for csv in Path("data").glob("R*/*/*.csv"):
        try:
            df = pd.read_csv(csv, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            la = pd.to_numeric(df["Latitude"], errors="coerce")
            lo = pd.to_numeric(df["Longitude"], errors="coerce")
            al = pd.to_numeric(df["Altitude"], errors="coerce")
            m = la.between(45,60) & lo.between(-10,5) & al.between(-50,2000)
            la, lo, al = la[m].to_numpy(), lo[m].to_numpy(), al[m].to_numpy()
        except Exception:
            continue
        if len(la) < 2:
            continue
        terr = sample_dem(elev, z, px0, py0, lo, la)
        agl = al - terr
        counts += np.histogram(agl, bins=edges)[0]
        n_tot += len(agl)
        asl_med.append(np.median(al)); agl_med.append(np.median(agl))
        terr_at_release.append(terr[0])
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    Path("agl_hist.json").write_text(json.dumps(
        {"centers": [round(c,1) for c in centers], "counts": counts.tolist()},
        separators=(",", ":")))
    aglmed = np.array(agl_med)
    print(f"\nfixes: {n_tot:,}")
    print(f"terrain at release sites: {np.nanmedian(terr_at_release):.0f} m ASL "
          f"({np.nanmin(terr_at_release):.0f}..{np.nanmax(terr_at_release):.0f})")
    print(f"per-flight median ASL altitude:  {np.median(asl_med):.0f} m")
    print(f"per-flight median AGL altitude:  {np.median(aglmed):.0f} m "
          f"(IQR {np.percentile(aglmed,25):.0f}-{np.percentile(aglmed,75):.0f})")
    # AGL percentiles from histogram
    cc = np.cumsum(counts) / counts.sum()
    for p in (0.5, 0.95):
        i = np.searchsorted(cc, p)
        print(f"  AGL p{int(p*100)}: {centers[min(i,len(centers)-1)]:.0f} m")
    print("Saved agl_hist.json")


if __name__ == "__main__":
    main()
