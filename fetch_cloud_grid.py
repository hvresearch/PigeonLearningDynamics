"""Fetch the cloud-cover grid for the overlay, rate-limit friendly.

Strategy: one date-RANGE request (whole 2011-05-05 .. 2011-06-08 span) per small
batch of grid points, with generous pauses. Filters to flight dates afterward.
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path("data")
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

GRID_LAT = [round(51.55 + 0.10 * i, 4) for i in range(7)]   # 51.55 .. 52.15
GRID_LON = [round(-1.70 + 0.125 * i, 4) for i in range(9)]  # -1.70 .. -0.70
GRID_PTS = [(la, lo) for la in GRID_LAT for lo in GRID_LON]
BATCH = 10
HOURS = list(range(5, 17))


def get_json(params):
    url = ARCHIVE + "?" + urllib.parse.urlencode(params, safe=",")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            wait = 8 * (attempt + 1)
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("API failed")


def flight_dates():
    dates = set()
    for csv_path in DATA.glob("R*/*/*.csv"):
        df = pd.read_csv(csv_path, skipinitialspace=True, usecols=["Date"], nrows=1)
        dates.add(str(df.iloc[0, 0]).strip().replace("/", "-"))
    return sorted(dates)


def main():
    dates = flight_dates()
    start, end = dates[0], dates[-1]
    print(f"Fetching cloud grid {start} .. {end}, {len(GRID_PTS)} pts "
          f"in batches of {BATCH}")

    # date -> {hour -> [cloud per grid point]}
    grid = {d: {h: [None] * len(GRID_PTS) for h in HOURS} for d in dates}

    for b in range(0, len(GRID_PTS), BATCH):
        batch = GRID_PTS[b:b + BATCH]
        lat = ",".join(str(la) for la, _ in batch)
        lon = ",".join(str(lo) for _, lo in batch)
        print(f"batch {b//BATCH + 1}/{-(-len(GRID_PTS)//BATCH)} "
              f"(pts {b}..{b+len(batch)-1})")
        res = get_json({"latitude": lat, "longitude": lon,
                        "start_date": start, "end_date": end,
                        "hourly": "cloud_cover", "timezone": "Europe/London"})
        res = res if isinstance(res, list) else [res]
        for j, r in enumerate(res):
            gp = b + j
            times = r["hourly"]["time"]
            cloud = r["hourly"]["cloud_cover"]
            # index hourly array by date+hour
            lut = {}
            for i, t in enumerate(times):
                lut[(t[:10], int(t[11:13]))] = cloud[i]
            for d in dates:
                for h in HOURS:
                    grid[d][h][gp] = lut.get((d, h))
        time.sleep(8)

    Path("cloud_grid.json").write_text(json.dumps(
        {"lat": GRID_LAT, "lon": GRID_LON, "grid": grid},
        separators=(",", ":")))
    kb = Path("cloud_grid.json").stat().st_size / 1e3
    print(f"Saved cloud_grid.json ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
