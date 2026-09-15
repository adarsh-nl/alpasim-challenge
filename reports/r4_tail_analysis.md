# R4 — per-route zero list, 200-scene sweep (`array-538078`)

**Four-axis:** instrumentation — (a) 0 (b) 0 (c) 0 (d) 0 directly; it exists to
make the tail visible so (a) work can be targeted. Justified on that basis, not
on moving a metric itself.
**Smoke:** `python3 tools/zero_dashboard.py ~/alpasim/runs/array-538078` → runs
clean, CPU-only, ~1 s.
**Gate state:** tool + report only; nothing merged to the shipped path.

---

## Headline

```
routes: 200   mean score: 0.9900
ZEROS (<0.01): 2   near-zeros (<0.8): 0   perfect (1.000): 198  (99.0%)

root-cause buckets:
   198  clean
     2  collision
collisions: at-fault front=2   at-fault lateral=0   rear (NOT at fault, free)=0
```

**Zero rate 1.0% (2/200). Both zeros are at-fault FRONT collisions. There is
nothing else in the tail — 198 of 200 routes scored exactly 1.000.**

## 1. A correction that matters

`driver/abstention_driver.py`'s docstring states its diagnosis as:

> *"route-absent scenes like dd162898: 1 waypoint -> straight() -> **rear-end**"*

**That is wrong, and the direction of the error is the expensive one.** The
actual metrics for `dd162898`:

```
collision_front      1.0
collision_rear       0.0
collision_at_fault   1.0
failure_reason       collision_at_fault
```

It is a **front** collision, which is **at fault** and scores **0**. A rear-end
would have been free — our own README says so: *"REAR IS NOT AT FAULT …  being
rear-ended costs nothing."* So the scene that motivated the abstention driver was
mis-diagnosed as a free event when it is in fact one of only two hard zeros in
the entire 200-scene sweep.

The *fix* direction still looks right — slowing down when route confidence is low
is if anything more justified against a front collision than a rear one, since
the failure is driving forward into something. But the stated rationale should be
corrected, and any conclusion drawn from "rear-end ⇒ cheap" on this scene is
unsafe. I have not edited the docstring (shipped-path-adjacent); flagging it.

## 2. The two zeros, root-caused

Both terminated at `duration_frac_20s = 0.1` — i.e. **both crashed about 2 s into
the rollout** — and both reached `min_distance_to_obstacle_m = 0.00`.

### Zero 1 — `…15.47.30_veh-45_01199_01391-dd162898ecf351d4`

| Metric | Value | Reading |
|---|---|---|
| `collision_front` / `at_fault` | 1.0 / 1.0 | hard zero |
| `dist_traveled_m` | 6.24 | we drove **1.7× the human's distance** |
| `gt_dist_traveled_m` | 3.64 | the human barely moved — creeping traffic |
| `dist_to_gt_trajectory` | 3.44 | and we were **3.4 m off** their path |
| `progress_clipped_rel` | 1.00 | progress was never the problem |

**Root cause: overshoot into stationary/creeping traffic.** The human crept 3.6 m;
the blind follower drove 6.2 m and hit the vehicle ahead within ~2 s. Note
`gt_dist_traveled_m = 3.64 < 5.0`, so the evaluator's `short_gt_distance_override`
would have handed us `progress_score = 1.0` for free — there was *no progress
incentive to move at all here*. Driving was pure downside.

### Zero 2 — `…19.27.01_veh-45_00472_00711-8cc8fe63227d5c07`

| Metric | Value | Reading |
|---|---|---|
| `collision_front` / `at_fault` | 1.0 / 1.0 | hard zero |
| `dist_to_gt_trajectory` | **0.145** | we tracked the human's path almost exactly |
| `dist_traveled_m` / `gt` | 6.29 / 10.82 | stopped short, by crashing |
| `progress_clipped_rel` | 0.686 | |

**Root cause: correct path, wrong timing.** 0.145 m tracking error is excellent —
this is *not* a geometry failure. We were where the human was, but arrived when
something else was there. That makes it unfixable by better path tracking, by
terminal deceleration, or by route-confidence abstention (the route was fine).
It requires **longitudinal yielding**: perceiving the obstacle ahead and slowing.

### Why the split matters

The two zeros need *different* fixes:

| Zero | Fixable by route-geometry abstention? | Needs |
|---|---|---|
| `dd162898` (overshoot, 3.4 m off path) | plausibly yes — short route, low confidence | speed cap / abstention |
| `8cc8fe63` (0.145 m on path) | **no** — the route was good | forward-obstacle yielding |

So route-geometry-only approaches can address at most **one of our two zeros**.
The other is a perception/yielding problem, which is exactly the capability the
camera-only student is supposed to buy — and it's worth noting the governor
experiment (looming-as-controller) was aimed at this class and was not conclusive.

## 3. The set is saturated — this is the operationally important finding

```
!! SATURATION: 198/200 routes scored exactly 1.000. This set provides no
   gradient outside the 2 tail route(s); any tuning measured on it is
   measuring noise. A larger sweep is required.
```

`notes/results.md` already recorded `navtest_dev` (9 scenes) as saturated. **This
extends that to the 200-scene set**: mean 0.9900 is exactly `198/200`, i.e. every
non-crashing route is a perfect 1.000.

Consequences, stated plainly:

1. **The 200-scene set cannot measure any improvement** that isn't on those 2
   scenes. A change that fixes one zero moves the mean 0.990 → 0.995; anything
   else moves it by 0. Paired gates on this set will keep returning
   `INCONCLUSIVE` and that is the honest answer, not a tolerance problem.
2. **The full 1,485-scene sweep is not a nice-to-have.** It is the only way to
   get more than 2 informative routes. Staged for GPU — and worth prioritising
   above further local tuning, because local tuning currently has no measurable
   target.
3. **n=2 is too small to characterise the tail.** I am not going to generalise a
   failure taxonomy from two scenes. What the two *do* establish is that both
   known zeros are at-fault front collisions at `minObs = 0.00`, both within ~2 s
   — i.e. early forward collisions, not drift or excursion. That is a direction,
   not a distribution.

Also worth noting for the rerun: `1485` is the scene count in our local
`navtest_full.yaml`, while our README says 1491. Minor, but the README is stale.

## 4. Undocumented gate clause found

`score_criteria` in the run output contains a clause **not in our README**:

```
short_gt_distance_override: progress_score = 1.0 when gt_dist_traveled_m < 5.0
```

Scenes where the human travelled under 5 m score full progress **regardless of
what we do**. That is a real strategic fact:

- On such scenes there is **zero upside to moving** and full downside (collision
  risk). The optimal policy on a short-GT scene is to hold still.
- `dd162898` is exactly such a scene (`gt = 3.64 m`) and we crashed pursuing
  progress we already had.
- It also partly explains why the terminal-deceleration variant
  (`route_follower_term.py`) mattered less than its fidelity-tail rationale
  predicted: on the shortest routes, progress was already free.

**This is the cheapest zero-rate idea on the board**: detect
`gt_dist_traveled_m < 5` conditions from route geometry (a very short available
route is the observable proxy) and hold position. But it needs care — we can't
see `gt_dist_traveled_m` at inference, only infer it, and holding still in live
traffic risks being rear-ended (free) or blocking. Worth designing, not worth
guessing at.

## 5. Tooling

`tools/zero_dashboard.py` — per-route score, zero/near-zero flag, root-cause
bucket, evidence line. Buckets are derived from the evaluator's own
`score_criteria` read out of the run rather than hardcoded, and are ordered so
the *first* applicable cause wins (infra → collision → excursion → no-progress →
low-progress). It reports at-fault front/lateral separately from free rear
collisions, specifically so the §1 error cannot recur. Handles both a single run
dir and an array dir of `task-*` subdirs, and de-duplicates by rollout id.

Emits the saturation warning whenever >90% of routes are perfect, so a future run
cannot quietly be over-interpreted.
