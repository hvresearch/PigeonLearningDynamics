# TODO

## Bird vision as a function of altitude → predict behavior from visual cues
_Added 2026-06-24. Requires map analysis._

Make "what the bird can see" rigorous, then test whether visible cues explain route choices.
Extends the field-of-view animation in `flight_paths_3d.html` (currently a hand-tuned ~5 km
sight cap rather than a principled altitude→visibility model).

- [ ] **Vision-vs-altitude model** — combine geometric horizon `d = √(2Rh)` with pigeon
      visual acuity (~12 cyc/deg) and a landmark-size assumption to get an effective
      *recognition* radius per altitude. 2011 has per-fix altitude; 2021 does not.
- [ ] **Map / viewshed analysis** — overlay terrain + landmark layers; compute true
      line-of-sight viewshed (hills occlude — not just a flat disc) along each track.
- [ ] **Behavior prediction** — relate visible cues (recognizable landmarks, terrain edges,
      coastline/road features) to heading changes and route stabilization across releases.
      Feeds the IRL feature-design work.
