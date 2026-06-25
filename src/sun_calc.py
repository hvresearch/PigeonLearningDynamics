"""Solar position (NOAA algorithm). Returns sun azimuth & elevation in degrees.

azimuth: degrees clockwise from true north (direction TO the sun).
elevation: degrees above the horizon (negative = below).
Accurate to ~0.1 deg, plenty for a navigation-cue overlay.
"""

import math


def sun_position(lat_deg, lon_deg, dt_utc):
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    hour = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = (int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d
          + B - 1524.5 + hour / 24)
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = math.radians(M)
    C = ((1.914602 - T * (0.004817 + 0.000014 * T)) * math.sin(Mr)
         + (0.019993 - 0.000101 * T) * math.sin(2 * Mr)
         + 0.000289 * math.sin(3 * Mr))
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lamb = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(lamb)))
    yv = math.tan(math.radians(eps / 2)) ** 2
    L0r = math.radians(L0)
    eqtime = 4 * math.degrees(
        yv * math.sin(2 * L0r) - 2 * e * math.sin(Mr)
        + 4 * e * yv * math.sin(Mr) * math.cos(2 * L0r)
        - 0.5 * yv * yv * math.sin(4 * L0r)
        - 1.25 * e * e * math.sin(2 * Mr))
    tst = (hour * 60 + eqtime + 4 * lon_deg) % 1440
    ha = tst / 4 - 180 if tst / 4 >= 0 else tst / 4 + 180
    if tst / 4 < 0:
        ha = tst / 4 + 180
    else:
        ha = tst / 4 - 180
    latr = math.radians(lat_deg)
    har = math.radians(ha)
    zen = math.acos(math.sin(latr) * math.sin(decl)
                    + math.cos(latr) * math.cos(decl) * math.cos(har))
    el = 90 - math.degrees(zen)
    denom = math.cos(latr) * math.sin(zen)
    if abs(denom) < 1e-9:
        az = 0.0
    else:
        arg = (math.sin(latr) * math.cos(zen) - math.sin(decl)) / denom
        arg = max(-1.0, min(1.0, arg))
        az = math.degrees(math.acos(arg))
        az = (az + 180) % 360 if ha > 0 else (540 - az) % 360
    return round(az, 1), round(el, 1)


if __name__ == "__main__":
    import datetime as dt
    # Oxford, 2011-05-05 11:25 UTC (12:25 BST) — near midday, sun ~S, fairly high
    print(sun_position(51.857, -1.284, dt.datetime(2011, 5, 5, 11, 25)))
