"""Fetch historical weather (ERA5 via Open-Meteo) for the pigeon flights.

Produces two artifacts:
  flight_weather.csv  - per-flight conditions at release location & start hour
  cloud_grid.json     - hourly cloud-cover field on a grid over the study area,
                        per flight date/hour, for the map overlay

ERA5 reanalysis, ~9-25 km resolution, hourly. Free, no API key.
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path("data")
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Variables pulled at each release point.
POINT_VARS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid",
              "cloud_cover_high", "wind_speed_10m", "wind_direction_10m",
              "wind_gusts_10m", "temperature_2m", "precipitation",
              "shortwave_radiation", "surface_pressure"]

# Study-area grid for the spatial cloud overlay (pad slightly past the data).
GRID_LAT = [round(51.55 + 0.07 * i, 4) for i in range(9)]   # 51.55 .. 52.11
GRID_LON = [round(-1.70 + 0.10 * i, 4) for i in range(11)]  # -1.70 .. -0.70
GRID_PTS = [(la, lo) for la in GRID_LAT for lo in GRID_LON]


def get_json(params):
    url = ARCHIVE + "?" + urllib.parse.urlencode(params, safe=",")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            wait = 2 * (attempt + 1)
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("API failed: " + url)


def scan_flights():
    """Return list of flight dicts with release point, date, start hour."""
    out = []
    for csv_path in sorted(DATA.glob("R*/*/*.csv")):
        df = pd.read_csv(csv_path, skipinitialspace=True,
                         usecols=["Date", "Time", "Latitude", "Longitude"])
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=["Latitude", "Longitude"])
        df = df[df["Latitude"].between(45, 60) & df["Longitude"].between(-10, 5)]
        if df.empty:
            continue
        date = str(df["Date"].iloc[0]).strip().replace("/", "-")
        start = str(df["Time"].iloc[0]).strip()
        out.append({
            "route": csv_path.parts[1],
            "bird": csv_path.parts[2],
            "flight": csv_path.stem.split("_")[-1],
            "date": date,
            "start_time": start,
            "hour": int(start[:2]),
            "rel_lat": round(float(df["Latitude"].iloc[0]), 5),
            "rel_lon": round(float(df["Longitude"].iloc[0]), 5),
        })
    return out


def fetch_point_weather(flights):
    """One multi-location call per date for the distinct release points."""
    by_date = {}
    for f in flights:
        key = (round(f["rel_lat"], 2), round(f["rel_lon"], 2))
        by_date.setdefault(f["date"], {})[key] = None
    for date, pts in sorted(by_date.items()):
        keys = list(pts)
        lat = ",".join(str(k[0]) for k in keys)
        lon = ",".join(str(k[1]) for k in keys)
        print(f"point weather {date}  ({len(keys)} release pts)")
        res = get_json({"latitude": lat, "longitude": lon,
                        "start_date": date, "end_date": date,
                        "hourly": ",".join(POINT_VARS),
                        "timezone": "Europe/London"})
        res = res if isinstance(res, list) else [res]
        for k, r in zip(keys, res):
            pts[k] = r["hourly"]
        time.sleep(0.4)
    return by_date


def fetch_cloud_grid(dates):
    """One multi-location call per date over the fixed grid; keep flight hours."""
    lat = ",".join(str(la) for la, _ in GRID_PTS)
    lon = ",".join(str(lo) for _, lo in GRID_PTS)
    grid = {}
    for date in sorted(dates):
        print(f"cloud grid    {date}  ({len(GRID_PTS)} pts)")
        res = get_json({"latitude": lat, "longitude": lon,
                        "start_date": date, "end_date": date,
                        "hourly": "cloud_cover", "timezone": "Europe/London"})
        res = res if isinstance(res, list) else [res]
        # res[p]["hourly"]["cloud_cover"][hour] -> store per hour as flat list
        per_hour = {}
        for hour in range(5, 17):  # daytime band that covers all flights
            per_hour[hour] = [res[p]["hourly"]["cloud_cover"][hour]
                              for p in range(len(GRID_PTS))]
        grid[date] = per_hour
        time.sleep(0.4)
    return grid


def main():
    flights = scan_flights()
    dates = {f["date"] for f in flights}
    print(f"{len(flights)} flights across {len(dates)} dates")

    point = fetch_point_weather(flights)
    rows = []
    for f in flights:
        key = (round(f["rel_lat"], 2), round(f["rel_lon"], 2))
        hourly = point[f["date"]][key]
        h = f["hour"]
        row = {**{k: f[k] for k in ("route", "bird", "flight", "date",
                                    "start_time", "rel_lat", "rel_lon")}}
        for v in POINT_VARS:
            val = hourly[v][h] if hourly and h < len(hourly[v]) else ""
            row[v] = val
        rows.append(row)

    with open("flight_weather.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved flight_weather.csv ({len(rows)} rows)")

    grid = fetch_cloud_grid(dates)
    Path("cloud_grid.json").write_text(json.dumps({
        "lat": GRID_LAT, "lon": GRID_LON, "grid": grid
    }, separators=(",", ":")))
    print(f"Saved cloud_grid.json ({Path('cloud_grid.json').stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
