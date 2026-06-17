"""Geometry-aware trajectory simplification (Ramer-Douglas-Peucker).

Keeps GPS points where the path actually bends and drops redundant points on
straight/stationary stretches, controlled by a tolerance in METERS. Far better
fidelity-per-point than uniform stride decimation. Works on lat/lon by first
projecting to a local equirectangular metric frame around the track centroid.
"""

import numpy as np

_M_PER_DEG_LAT = 110540.0


def _rdp_mask(xy, eps):
    """Boolean keep-mask over points (Nx2 array, metric units), tolerance eps."""
    n = len(xy)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        seg = xy[i:j + 1]
        ab = b - a
        L = np.hypot(ab[0], ab[1])
        if L == 0:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            # perpendicular distance from each point to line a-b
            d = np.abs(ab[0] * (a[1] - seg[:, 1]) - (a[0] - seg[:, 0]) * ab[1]) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            idx = i + k
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return keep


def simplify(coords, eps_m=4.0, cap=2000, round_to=5):
    """coords: list of [lat, lon]. Returns simplified list of [lat, lon].

    eps_m : RDP tolerance in meters (smaller = more detail/points).
    cap   : hard safety ceiling on kept points (uniform stride if exceeded).
    """
    if len(coords) <= 2:
        return [[round(la, round_to), round(lo, round_to)] for la, lo in coords]
    arr = np.asarray(coords, dtype=float)
    lat0 = float(arr[:, 0].mean())
    mlon = _M_PER_DEG_LAT * np.cos(np.radians(lat0))
    xy = np.column_stack([arr[:, 1] * mlon, arr[:, 0] * _M_PER_DEG_LAT])
    keep = _rdp_mask(xy, eps_m)
    out = arr[keep]
    if len(out) > cap:
        out = out[:: (len(out) // cap + 1)]
    return [[round(la, round_to), round(lo, round_to)] for la, lo in out]
