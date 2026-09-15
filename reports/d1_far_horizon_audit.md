# D1 Task 4 — far-horizon feasibility audit, full shard-0

**Four-axis:** audit only — (a) 0 (b) 0 (c) 0 (d) 0 directly. It exists to stop
~5% of teacher labels teaching the student to leave the road or hit a car, which
is the mechanism by which a distilled zero would appear.
**Smoke:** `~/alpasim/.venv/bin/python rl/audit_far_horizon.py --labels rl/teacher_labels`
→ 6080 audited, 0 skipped, ~8 min CPU.
**Gate state:** audit complete; filtering recommendation below is **not applied**.

---

## Headline

**2.52% of labels drive off the road, and 97% of those do it beyond the window
GT can check.** That is precisely the hiding place we went looking for.

```
audited 6080 labels (0 skipped)

1. OFF-ROAD  (full 8 s -- COMPLETE coverage)
   labels with any off-road waypoint : 153/6080  (2.52%)
   first off-road step: median 60 (6.0s)  min 0  max 79
   of those, first leave within 2.0 s (GT-verifiable window): 5/153
       -> 148 are FAR-HORIZON ONLY
   off-road steps per bad label: median 13 of 80

2. COLLISION with logged neighbours  (LIMITED coverage)
   checkable steps per label: median 15 of 80 (max 35)
   labels with >=1 checkable step    : 5281/6080
   labels with a box overlap         : 147/5281 (2.78%)
   first overlap step: median 9 (0.9s)

3. KINEMATIC FEASIBILITY  (full 8 s -- COMPLETE coverage)
   labels flagged : 19/6080  (0.31%)
        19  JERK
         2  REVERSE
   v_max      median  6.60  p99 11.77  max 13.24   [flag >30]  m/s
   a_lon_max  median  0.77  p99  2.11  max  3.58   [flag >6]   m/s^2
   a_lat_max  median  0.17  p99  2.63  max  3.16   [flag >8]   m/s^2
   jerk_max   median  3.23  p99 11.78  max 31.97   [flag >15]  m/s^3

=== any-issue rate: 313/6080 (5.15%) ===
```

## What the three checks say

### Off-road — the real finding

2.52% is not large in itself. What matters is **when**: the median first
departure is at **step 60, i.e. 6.0 s into an 8 s label**, and only **5 of 153**
leave the road inside the ~2 s window GT can validate.

So GT-based validation — the ADE check in `reports/d1_label_integrity.md`, and
any loss term anchored to GT — would have caught **3%** of these. The remaining
97% are invisible to every check we had before this one. A student supervised
uniformly across 80 steps learns them.

This is also independent evidence for **horizon-weighted loss**: the teacher's
reliability degrades with horizon in a measurable way, not just an unverifiable
one. Previously we had "we cannot check the far horizon"; now we have "the far
horizon is where it fails".

### Collisions — the opposite pattern, and mostly checkable

2.78% overlap a logged neighbour, but the median first overlap is at **step 9
(0.9 s)** — *early*, inside the verifiable window. So collisions and off-road
failures are different failure modes with different signatures: collisions are a
near-horizon problem, off-road is a far-horizon one.

Two caveats, both stated rather than buried:

- **Coverage is limited.** Agent data ends with the scene, so the median label
  has only **15 of 80** steps checkable, and labels at ts=55 have none. The
  2.78% is a rate over the near horizon only; the far-horizon collision rate is
  **unmeasurable with this data** and is not reported.
- **Logged neighbours do not react.** They replay recorded paths and never yield,
  so some flagged overlaps would not have happened against a real driver. Read
  2.78% as an **upper bound** on teacher recklessness, not a score.

### Kinematics — the teacher is physically sound

0.31% flagged, all jerk (19) with 2 reversing. Speeds median 6.60 m/s and p99
11.77; longitudinal accel max 3.58 m/s²; lateral accel max 3.16 m/s². These are
ordinary urban-driving numbers. **Diffusion-Planner is not producing physically
implausible trajectories** — its failures are semantic (wrong place) rather than
dynamic (impossible motion).

That is worth knowing because it rules out a cheap filter: we cannot find the bad
labels by looking for unrealistic dynamics. The off-road check is doing the work.

## Recommendation (not applied)

**Filter or down-weight the 313 flagged labels (5.15%) before distillation.**

Rationale in four-axis terms: zero-rate is the primary bar, and a label that
drives off the road or into a logged car is training signal pointing directly at
a zero. 5.15% is small enough to drop outright without materially shrinking an
already-small set (6080 → 5767), and the filter is one boolean per label — about
as cheap and as ablatable as an intervention gets.

Three ways to use it, in increasing sophistication:

1. **Drop** the 313. Simplest, and the safe default.
2. **Truncate** each label at its first off-road step rather than dropping it —
   keeps the good prefix, which for a median first-departure at step 60 is most
   of the label. Strictly more data than (1).
3. **Weight** per-step by "still on-road and still checkable", which composes
   with the horizon-weighted loss and with the information-gating idea: the
   teacher's own off-road excursions are a measured, per-state confidence signal.

(2) is probably the best value and I would default to it, but it is a D2 design
choice and the numbers above are what it needs. I have not implemented any of
them.

## Method notes — two bugs the controls caught

Both would have produced a falsely clean report, so they are worth recording.

1. **Off-road was silently broken.** `get_road_areas_within` raised *"Road areas
   are not loaded. Use map_api.get_map(..., incl_road_areas=True)"* on every
   call; an except-and-continue handler turned that into **0.00% off-road** on
   the first run. Fixed by loading road areas, and failures now propagate — a
   broken containment check must not be reportable as clean. I also had the API
   wrong: `get_road_areas_within` is a *proximity* query (dist=20 m returns 6
   areas for an on-lane point, dist=200 m returns 55), so containment needs
   `get_road_area_polygon_2d` plus point-in-polygon, indexed per city with an
   STRtree.
   **Positive control, passing:** 1905 polygons indexed for Boston; an
   unmodified label is on-road; the same label displaced 150 m laterally is
   off-road at 80/80 waypoints from step 0.
2. **Lateral accel was an artifact.** I computed `max(v)² · max(κ)`, combining a
   speed and a curvature from different points, which gave 1271 m/s² where
   near-stationary waypoints made curvature explode. Now per-step `v[k]²·κ[k]`
   with sub-5 cm steps excluded. Median a_lat fell 2.73 → 0.17 m/s² and the
   kinematic flag rate 25% → 0.31%. **The earlier flag rate was almost entirely
   this bug.**

## Coverage summary — what this audit can and cannot see

| Check | Horizon coverage | Verdict |
|---|---|---|
| Off-road | **full 8 s** (static map) | 2.52%, 97% far-horizon-only |
| Collision w/ logged neighbours | median 15 / 80 steps; none at ts=55 | 2.78% near-horizon, upper bound; far horizon **unmeasurable** |
| Kinematics | **full 8 s** | 0.31%, teacher is physically sound |

The far-horizon collision rate remains the one thing nothing we have can
measure. If that matters for D2, the only routes are a reactive traffic sim over
teacher trajectories, or accepting off-road as the far-horizon proxy.
