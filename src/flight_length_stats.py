"""Power-law analysis of two per-flight quantities:
  (1) path length (km) = sum of great-circle GPS segments
  (2) tortuosity       = path length / straight-line start->end displacement

For each, fit a power law (Clauset MLE) and compare vs lognormal & exponential.
Figure: two log-log CCDF panels (length | tortuosity), each with power-law fits.
"""

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import powerlaw

R = 6371.0


def _gc(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def path_and_net(lat, lon):
    """Return (path length km, straight-line start->end km)."""
    seg = _gc(lat[:-1], lon[:-1], lat[1:], lon[1:])
    path = float(np.sum(seg))
    net = float(_gc(lat[0], lon[0], lat[-1], lon[-1]))
    return path, net


def collect_2011():
    paths, nets = [], []
    for f in glob.glob("data/R*/*/*.csv"):
        df = pd.read_csv(f, skipinitialspace=True, usecols=["Latitude", "Longitude"])
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna()
        df = df[df["Latitude"].between(45, 60) & df["Longitude"].between(-10, 5)]
        if len(df) > 1:
            p, n = path_and_net(df["Latitude"].values, df["Longitude"].values)
            paths.append(p); nets.append(n)
    return np.array(paths), np.array(nets)


def collect_2021():
    paths, nets = [], []
    for f in glob.glob("2021/data/raw/*.csv"):
        try:
            a = pd.read_csv(f, header=None, usecols=[0, 1]).values
        except Exception:
            continue
        a = a[(a[:, 0] > 45) & (a[:, 0] < 60) & (a[:, 1] > -10) & (a[:, 1] < 5)]
        if len(a) > 1:
            p, n = path_and_net(a[:, 0], a[:, 1])
            paths.append(p); nets.append(n)
    return np.array(paths), np.array(nets)


def fit(name, x):
    x = np.asarray(x); x = x[np.isfinite(x) & (x > 0)]
    f = powerlaw.Fit(x, verbose=False)
    print(f"\n[{name}] n={len(x)}  median={np.median(x):.2f}  max={x.max():.2f}")
    print(f"  power law: xmin={f.power_law.xmin:.2f}, alpha={f.power_law.alpha:.2f}, "
          f"tail n={int(np.sum(x>=f.power_law.xmin))}")
    for alt in ("lognormal", "exponential"):
        Rll, p = f.distribution_compare("power_law", alt, normalized_ratio=True)
        print(f"  vs {alt:11s}: R={Rll:+.2f} p={p:.3f} -> "
              f"{'power_law' if Rll>0 else alt}{'' if p<0.1 else ' (ns)'}")
    return f


def main():
    print("Computing (2021 raw is large)...")
    p11, n11 = collect_2011()
    p21, n21 = collect_2021()
    t11 = p11 / np.where(n11 > 0.05, n11, np.nan)
    t21 = p21 / np.where(n21 > 0.05, n21, np.nan)
    np.savez("flight_metrics.npz", p11=p11, n11=n11, p21=p21, n21=n21)

    print("\n===== FLIGHT LENGTH (km) =====")
    fL11 = fit("2011 length", p11); fL21 = fit("2021 length", p21)
    print("\n===== TORTUOSITY (path / straight-line) =====")
    fT11 = fit("2011 tortuosity", t11); fT21 = fit("2021 tortuosity", t21)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    series = [("#e6194B", "2011 Flack"), ("#4363d8", "2021 Valentini")]

    for (c, lab), f in zip(series, (fL11, fL21)):
        f.plot_ccdf(ax=ax[0], color=c, marker=".", linewidth=0, label=f"{lab} data")
        f.power_law.plot_ccdf(ax=ax[0], color=c, linestyle="--", linewidth=1.3,
                              label=f"{lab} power law (α={f.power_law.alpha:.2f})")
    ax[0].set_xlabel("Flight path length (km)"); ax[0].set_ylabel("CCDF  P(X ≥ x)")
    ax[0].set_title("Flight length"); ax[0].legend(fontsize=8)

    for (c, lab), f in zip(series, (fT11, fT21)):
        f.plot_ccdf(ax=ax[1], color=c, marker=".", linewidth=0, label=f"{lab} data")
        f.power_law.plot_ccdf(ax=ax[1], color=c, linestyle="--", linewidth=1.3,
                              label=f"{lab} power law (α={f.power_law.alpha:.2f})")
    ax[1].set_xlabel("Tortuosity  (path ÷ straight-line)"); ax[1].set_ylabel("CCDF  P(X ≥ x)")
    ax[1].set_title("Tortuosity"); ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/flight_length_stats.png", dpi=150, bbox_inches="tight")
    print("\nSaved flight_length_stats.png")


if __name__ == "__main__":
    main()
