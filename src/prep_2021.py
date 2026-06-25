"""Parse the 2021 Valentini et al. raw GPS tracks into a compact JSON for the viewer.

Raw: 2021/data/raw/<Condition>_<ID>_<release>_bare.csv  (headerless: lat,lon,time)
Conditions: Solo, Pair, Gen1..Gen5.  Each file = one bird's track on one release.
Output: data_2021.json  { groups:[...], units:[{id, group, label, flights:[{n,coords}]}] }
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from geo_simplify import simplify

RAW = Path("2021/data/raw")
EPS_M = 5.0    # RDP tolerance in meters
CAP = 1200     # safety ceiling on points per flight
NAME = re.compile(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d+)$")


def norm_cond(c):
    c = c.capitalize()
    m = re.match(r"Gen(\d+)", c, re.I)
    return f"Gen{m.group(1)}" if m else c


def load_track(path):
    pts = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                la, lo = float(row[0]), float(row[1])
            except ValueError:
                continue
            if 45 < la < 60 and -10 < lo < 5:
                pts.append([la, lo])
    if not pts:
        return []
    return simplify(pts, eps_m=EPS_M, cap=CAP)


def main():
    units = defaultdict(lambda: {"flights": []})
    groups = set()
    files = sorted(RAW.glob("*.csv"))
    for i, f in enumerate(files):
        m = NAME.match(f.stem.replace("_bare", ""))
        if not m:
            continue
        cond = norm_cond(m.group(1))
        bird, rel = m.group(2), int(m.group(3))
        coords = load_track(f)
        if not coords:
            continue
        groups.add(cond)
        key = (cond, bird)
        units[key]["group"] = cond
        units[key]["id"] = f"{cond}·{bird}"
        units[key]["flights"].append({"rel": rel, "coords": coords})
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(files)} files")

    out_units = []
    for (cond, bird), u in units.items():
        fl = sorted(u["flights"], key=lambda x: x["rel"])
        out_units.append({
            "id": u["id"], "group": cond,
            "flights": [{"n": str(x["rel"]), "coords": x["coords"]} for x in fl],
        })
    out_units.sort(key=lambda u: (u["group"], u["id"]))

    def gkey(g):
        m = re.match(r"Gen(\d+)", g)
        return (0, int(m.group(1))) if m else (1 if g == "Pair" else 2, g)

    payload = {"groups": sorted(groups, key=gkey), "units": out_units}
    Path("data_2021.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{len(out_units)} units across {len(groups)} groups "
          f"({sorted(groups, key=gkey)})")
    print(f"Saved data_2021.json ({Path('data_2021.json').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
