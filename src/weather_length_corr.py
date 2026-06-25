"""Spearman correlation: 2011 path length vs weather, overall and by release stage.

Hypothesis: as a bird is trained (release 1 -> 6), its route stabilizes and path
length should become *less* sensitive to weather (|rho| shrinks with stage).

Joins recomputed per-flight path length to flight_weather.csv on (route,bird,flight).
Adds a derived 'tailwind_home' = wind component blowing toward the home loft.

Outputs a console report + weather_length_corr.png (rho heatmap, stage x variable).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

DATA = Path("data")
R = 6371.0  # km


def haversine_km(lat, lon):
    lat = np.radians(lat); lon = np.radians(lon)
    dlat = np.diff(lat); dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1, np.sqrt(a)))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing deg from point1 to point2 (0=N, clockwise)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360


def load_track(route, bird, flight):
    f = DATA / route / bird / f"{bird}_{route}_{flight:02d}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in ("Latitude", "Longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[df["Latitude"].between(45, 60) & df["Longitude"].between(-10, 5)]
    if len(df) < 2:
        return None
    lat = df["Latitude"].to_numpy(); lon = df["Longitude"].to_numpy()
    return lat, lon


def main():
    w = pd.read_csv("flight_weather.csv")
    lengths, end_lat, end_lon, start_lat, start_lon = [], [], [], [], []
    for _, r in w.iterrows():
        t = load_track(r["route"], r["bird"], int(r["flight"]))
        if t is None:
            lengths.append(np.nan); end_lat.append(np.nan); end_lon.append(np.nan)
            start_lat.append(np.nan); start_lon.append(np.nan); continue
        lat, lon = t
        lengths.append(float(haversine_km(lat, lon).sum()))
        start_lat.append(lat[0]); start_lon.append(lon[0])
        end_lat.append(lat[-1]); end_lon.append(lon[-1])
    w["path_km"] = lengths
    w["end_lat"] = end_lat; w["end_lon"] = end_lon
    w["start_lat"] = start_lat; w["start_lon"] = start_lon

    # home loft = median of flight end points; beeline + tortuosity
    home_lat, home_lon = np.nanmedian(end_lat), np.nanmedian(end_lon)
    w["beeline_km"] = R * np.radians(1) * 0  # placeholder
    bl = haversine_pairs(w["start_lat"], w["start_lon"], home_lat, home_lon)
    w["beeline_km"] = bl
    w["tortuosity"] = w["path_km"] / w["beeline_km"]

    # derived tailwind toward home: + helps the bird, - is a headwind
    brg_home = bearing(w["start_lat"].to_numpy(), w["start_lon"].to_numpy(),
                       home_lat, home_lon)
    wind_to = (w["wind_direction_10m"].to_numpy() + 180) % 360  # blowing-toward dir
    ang = np.radians(wind_to - brg_home)
    w["tailwind_home"] = w["wind_speed_10m"].to_numpy() * np.cos(ang)
    w["crosswind"] = w["wind_speed_10m"].to_numpy() * np.abs(np.sin(ang))

    w = w.dropna(subset=["path_km"]).copy()
    print(f"home loft est: {home_lat:.4f}, {home_lon:.4f}   "
          f"beeline median {w['beeline_km'].median():.1f} km   n={len(w)}")
    print(f"path_km: median {w['path_km'].median():.1f}, "
          f"IQR {w['path_km'].quantile(.25):.1f}-{w['path_km'].quantile(.75):.1f}\n")

    wvars = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
             "wind_speed_10m", "wind_gusts_10m", "tailwind_home", "crosswind",
             "temperature_2m", "precipitation", "shortwave_radiation",
             "surface_pressure"]

    def corr_block(df, label):
        print(f"===== {label}  (n={len(df)}) =====")
        print(f"{'variable':<22}{'rho':>8}{'p':>10}")
        out = {}
        for v in wvars:
            rho, p = spearmanr(df["path_km"], df[v], nan_policy="omit")
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            print(f"{v:<22}{rho:>8.3f}{p:>10.4f} {star}")
            out[v] = rho
        print()
        return out

    overall = corr_block(w, "OVERALL  path_km vs weather")

    # by release stage (flight 1..6)
    stage_rho = {}
    for fl in sorted(w["flight"].unique()):
        stage_rho[fl] = corr_block(w[w["flight"] == fl], f"RELEASE {fl}")

    # summary: mean |rho| per stage -> does weather sensitivity fall with training?
    print("===== mean |rho| across weather vars, by release stage =====")
    print(f"{'release':>8}{'mean|rho|':>12}{'mean|rho| (wind+cloud)':>26}")
    key = ["cloud_cover", "wind_speed_10m", "tailwind_home", "shortwave_radiation"]
    for fl in sorted(stage_rho):
        vals = np.abs(list(stage_rho[fl].values()))
        kv = np.abs([stage_rho[fl][k] for k in key])
        print(f"{fl:>8}{np.nanmean(vals):>12.3f}{np.nanmean(kv):>26.3f}")
    # trend test: does |rho| for each var decline across stage?
    print("\n===== Spearman(release stage, |rho|) per variable  (neg = weakens) =====")
    stages = sorted(stage_rho)
    for v in wvars:
        series = np.array([abs(stage_rho[fl][v]) for fl in stages], float)
        m = ~np.isnan(series)
        if m.sum() < 3:
            print(f"{v:<22} trend n/a (insufficient data)"); continue
        tr, tp = spearmanr(np.array(stages)[m], series[m])
        flag = "  <- weakens with training" if (tr < 0 and tp < 0.1) else ""
        print(f"{v:<22} trend rho={tr:>6.2f}  p={tp:.3f}{flag}")

    make_heatmap(stage_rho, overall, wvars, stages)


def haversine_pairs(lat, lon, lat2, lon2):
    lat = np.radians(np.asarray(lat, float)); lon = np.radians(np.asarray(lon, float))
    p2, l2 = np.radians(lat2), np.radians(lon2)
    dlat = p2 - lat; dlon = l2 - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1, np.sqrt(a)))


def make_heatmap(stage_rho, overall, wvars, stages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[stage_rho[fl][v] for fl in stages] for v in wvars])
    ov = np.array([[overall[v]] for v in wvars])
    fig, ax = plt.subplots(1, 2, figsize=(9, 6),
                           gridspec_kw={"width_ratios": [len(stages), 1]})
    for a, mat, cols, title in [
            (ax[0], M, [f"R{s}" for s in stages], "Spearman ρ by release stage"),
            (ax[1], ov, ["all"], "overall")]:
        im = a.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        a.set_xticks(range(len(cols))); a.set_xticklabels(cols)
        a.set_yticks(range(len(wvars))); a.set_yticklabels(wvars if a is ax[0] else [])
        a.set_title(title, fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                a.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                       fontsize=7, color="#222")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Spearman ρ (path length vs weather)")
    fig.suptitle("2011 pigeon path length vs weather — does training reduce weather sensitivity?",
                 fontsize=11)
    fig.savefig("figures/weather_length_corr.png", dpi=130, bbox_inches="tight")
    print("\nSaved weather_length_corr.png")


if __name__ == "__main__":
    main()
