"""Pigeon vision-vs-altitude model.

Effective *recognition* radius for a ground landmark = the min of two limits:

  1. Geometric horizon (altitude-limited):  d_h = sqrt(2*R*h + h^2)
       - how far before the Earth curves away; grows with sqrt(altitude).
  2. Visual acuity (landmark-size-limited):  d_a = L / (2*tan(theta/2))
       - a landmark of width L is only recognizable while it subtends an angle
         theta >= the bird's recognition threshold; independent of altitude.
       - theta = recog_elements * MRA, where the minimum resolvable angle
         MRA = 1/(2*acuity) degrees (one bar of the finest resolvable grating).

Optionally also capped by atmospheric/meteorological visibility.

Pigeon acuity ~12 cycles/deg (Hodos et al.); ~1/3 of human. Defaults below are
deliberately conservative knobs, not measured truths — see TODO.md.

Lengths in metres unless noted. R = Earth radius.
"""
import math

R_EARTH = 6_371_000.0  # m


def horizon_distance(h_agl):
    """Geometric horizon distance (m) for eye height h_agl (m) above the surface,
    viewing an object at ground level."""
    h = max(0.0, h_agl)
    return math.sqrt(2 * R_EARTH * h + h * h)


def min_resolvable_angle_deg(acuity_cpd):
    """Finest resolvable detail (one bar = half a cycle), in degrees."""
    return 1.0 / (2.0 * acuity_cpd)


def acuity_range(landmark_m, acuity_cpd=12.0, recog_elements=5.0):
    """Distance (m) at which a landmark of width `landmark_m` drops to the
    recognition threshold angle. recog_elements = how many resolvable bars must
    span the object to 'recognize' it (1 = bare detection, ~5-10 = see shape)."""
    theta = math.radians(recog_elements * min_resolvable_angle_deg(acuity_cpd))
    return landmark_m / (2.0 * math.tan(theta / 2.0))


def recognition_radius(h_agl, landmark_m=100.0, acuity_cpd=12.0,
                       recog_elements=5.0, visibility_m=None):
    """Effective recognition radius (m): min(horizon, acuity[, visibility])."""
    d = min(horizon_distance(h_agl), acuity_range(landmark_m, acuity_cpd, recog_elements))
    if visibility_m:
        d = min(d, visibility_m)
    return d


def crossover_altitude(landmark_m, acuity_cpd=12.0, recog_elements=5.0):
    """Altitude (m AGL) at which horizon == acuity range. Below it the bird is
    horizon-limited (sees less because Earth curves); above it acuity-limited
    (horizon is huge but the landmark is too small to resolve farther)."""
    a = acuity_range(landmark_m, acuity_cpd, recog_elements)
    return (a * a) / (2 * R_EARTH)  # invert d_h = sqrt(2 R h)


if __name__ == "__main__":
    print("altitude(m)  horizon(km)  acuity@100m(km)  effective(km)")
    for h in (10, 25, 50, 75, 100, 150, 300):
        print(f"{h:>8}   {horizon_distance(h)/1e3:>9.1f}   "
              f"{acuity_range(100)/1e3:>13.1f}   {recognition_radius(h,100)/1e3:>10.1f}")
    print(f"\ncrossover altitude (100 m landmark): "
          f"{crossover_altitude(100):.1f} m AGL")
