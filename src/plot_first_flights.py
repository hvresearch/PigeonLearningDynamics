"""Plot the first release flight (*_01.csv) of every pigeon on a map.

Data: Flack et al. (2014) homing-pigeon GPS tracks.
Folders: data/R{1,2,3}/<COND>_<BIRD>/<BIRD>_R{n}_01.csv
Columns: Date, Time, Latitude, Longitude, Altitude, Speed, Course, Type, Distance, Essential
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("data")
ROUTE_COLORS = {"R1": "#e6194B", "R2": "#3cb44b", "R3": "#4363d8"}


def load_track(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Latitude", "Longitude"])
    # Drop obviously bad fixes (0,0 or out of plausible UK range)
    df = df[(df["Latitude"].between(45, 60)) & (df["Longitude"].between(-10, 5))]
    return df


def main():
    flights = sorted(DATA.glob("R*/*/*_01.csv"))
    print(f"Found {len(flights)} first-flight files")

    fig, ax = plt.subplots(figsize=(12, 12))
    plotted = {r: 0 for r in ROUTE_COLORS}
    all_lat, all_lon = [], []

    for f in flights:
        route = f.parts[1]  # R1 / R2 / R3
        color = ROUTE_COLORS.get(route, "#999999")
        df = load_track(f)
        if df.empty:
            continue
        ax.plot(df["Longitude"], df["Latitude"], color=color,
                lw=0.8, alpha=0.6, zorder=2)
        # release point (start) marker
        ax.plot(df["Longitude"].iloc[0], df["Latitude"].iloc[0],
                marker="^", color=color, ms=5, zorder=3)
        plotted[route] += 1
        all_lat += [df["Latitude"].iloc[0], df["Latitude"].iloc[-1]]
        all_lon += [df["Longitude"].iloc[0], df["Longitude"].iloc[-1]]

    # legend
    handles = [plt.Line2D([], [], color=c, lw=2, label=f"{r} (n={plotted[r]})")
               for r, c in ROUTE_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=11, framealpha=0.9)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Pigeon first release flights (Flack et al. 2014)\n"
                 "▲ = release site, lines = GPS trajectory, colored by route")
    ax.set_aspect("equal", adjustable="datalim")

    # basemap underlay
    try:
        import contextily as cx
        cx.add_basemap(ax, crs="EPSG:4326",
                       source=cx.providers.CartoDB.Positron, attribution_size=6)
        print("Basemap added")
    except Exception as e:  # noqa: BLE001
        ax.grid(True, ls=":", alpha=0.4)
        print(f"No basemap ({e}); plotted on plain axes")

    out = "figures/first_flights_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print("Per-route flights plotted:", plotted)


if __name__ == "__main__":
    main()
