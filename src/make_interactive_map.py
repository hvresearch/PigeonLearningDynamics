"""Build an interactive Leaflet map of each pigeon's first and last release flights.

Data: Flack et al. (2014) homing-pigeon GPS tracks.
Folders: data/R{1,2,3}/<COND>_<BIRD>/<BIRD>_R{n}_<NN>.csv
First flight = lowest-numbered CSV, last flight = highest-numbered CSV per bird.
Output: first_flights_map.html  (open in any browser; pan / zoom / toggle layers)
"""

from pathlib import Path

import folium
import pandas as pd
from folium.plugins import GroupedLayerControl

DATA = Path("data")
ROUTE_COLORS = {"R1": "#e6194B", "R2": "#3cb44b", "R3": "#4363d8"}


def load_track(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[df["Latitude"].between(45, 60) & df["Longitude"].between(-10, 5)]
    # Thin to ~every 3rd fix to keep the HTML light without losing shape.
    return df.iloc[::3]


def add_track(csv_path, route, group, color, dash=None, label="flight"):
    bird = csv_path.parts[2]
    flight_no = csv_path.stem.split("_")[-1]
    df = load_track(csv_path)
    if df.empty:
        return []
    coords = list(zip(df["Latitude"], df["Longitude"]))
    folium.PolyLine(
        coords, color=color, weight=2, opacity=0.6, dash_array=dash,
        tooltip=f"{bird} · {route} · {label} (flight {flight_no})",
    ).add_to(group)
    folium.CircleMarker(
        coords[0], radius=4, color=color, fill=True, fill_opacity=1,
        tooltip=f"Release · {bird} ({route}) · {label}",
    ).add_to(group)
    return coords


def main():
    birds = sorted(p for p in DATA.glob("R*/*") if p.is_dir())
    print(f"Found {len(birds)} bird folders")

    fmap = folium.Map(tiles="CartoDB positron", control_scale=True)

    # One feature group per (route, phase). Last flights drawn dashed.
    first_groups = {r: folium.FeatureGroup(name=f"{r} · first flight", show=True)
                    for r in ROUTE_COLORS}
    last_groups = {r: folium.FeatureGroup(name=f"{r} · last flight", show=False)
                   for r in ROUTE_COLORS}

    all_pts = []
    counts = {r: {"first": 0, "last": 0} for r in ROUTE_COLORS}

    for bird_dir in birds:
        route = bird_dir.parts[1]
        color = ROUTE_COLORS.get(route, "#999999")
        flights = sorted(bird_dir.glob("*.csv"))
        if not flights:
            continue
        first, last = flights[0], flights[-1]

        pts = add_track(first, route, first_groups[route], color, label="first")
        if pts:
            all_pts += pts
            counts[route]["first"] += 1

        if last != first:
            pts = add_track(last, route, last_groups[route], color,
                            dash="6", label="last")
            if pts:
                all_pts += pts
                counts[route]["last"] += 1

    for g in list(first_groups.values()) + list(last_groups.values()):
        g.add_to(fmap)

    GroupedLayerControl(
        groups={
            "First flights": list(first_groups.values()),
            "Last flights": list(last_groups.values()),
        },
        collapsed=False, exclusive_groups=False,
    ).add_to(fmap)

    lats = [p[0] for p in all_pts]
    lons = [p[1] for p in all_pts]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    legend = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        'background:white;padding:10px 14px;border:1px solid #888;'
        'border-radius:6px;font:13px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
        '<b>Pigeon release flights</b><br>Flack et al. (2014)<br>'
    )
    for r, c in ROUTE_COLORS.items():
        legend += (f'<span style="color:{c};font-size:16px">&#9644;</span> '
                   f'{r} (n={counts[r]["first"]})<br>')
    legend += ('<hr style="margin:6px 0">'
               'solid = first flight<br>dashed = last flight</div>')
    fmap.get_root().html.add_child(folium.Element(legend))

    out = "viewers/first_flights_map.html"
    fmap.save(out)
    print(f"Saved {out}; counts: {counts}")


if __name__ == "__main__":
    main()
