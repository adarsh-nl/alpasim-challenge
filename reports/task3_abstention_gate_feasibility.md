# Task 3 — teacher-as-abstention-gate: feasibility verdict

**I did not deliver a caution profile. The premise did not survive contact with
measured data, and the offline model I built to test it was invalid.** Reporting
that rather than shipping a profile fitted to an artifact.

**Four-axis:** nothing landed — (a) 0 (b) 0 (c) 0 (d) 0.
**Gate state:** line **not viable to validate offline** with current data. The
cheap GPU ask that would unblock it is at the end.

---

## 1. My offline follower model was wrong, and I am not reporting its numbers

To find "where the geometric driver overshoots the teacher" without a simulator,
I modelled the shipped follower offline: curvature-limited speed profile,
`v_cap = min(V_MAX, sqrt(A_LAT/kappa))` at the rank-1 config, integrated over 8 s.

It produced a median 8 s distance of **112 m** against the teacher's **39 m** —
a median overshoot of +61 m, with 82% of samples overshooting by >5 m.

**Those numbers are an artifact.** The real overshoot on `dd162898` was 6.24 m
vs a 3.64 m human path — not +61 m. The model omits the dominant constraint: the
shipped follower is limited by the **length of the route it is given**
(`s_cur = min(s_cur + v*dt, s_arc[-1])` in `route_follower.py`), not by
curvature. On an open low-curvature lane my model happily accelerates to
`V_MAX=20` and runs 112 m; the real driver stops at the end of its route.

And route length is precisely what I cannot reconstruct offline: the sim supplies
a navigation route, and approximating it with the human's future path would leak
the answer I am trying to predict. So **the offline model cannot be made faithful
with the data I have.**

## 2. On measured data the premise is largely false

Rather than model the follower, use its measured behaviour. Scenes with **both**
a measured follower distance (200-scene sweep) and teacher labels (shard 0):
**12**.

| scene | follower (m) | GT (m) | score |
|---|---|---|---|
| …94a3b8d43f515d6e | 19.09 | 20.93 | 1.00 |
| …0aa8c67f04b75a41 | 40.46 | 41.80 | 1.00 |
| …2ea6bdf1e8905fd8 | **15.08** | **13.09** | 1.00 |
| …0d8754f6d53d5968 | 18.88 | 19.34 | 1.00 |
| …c1e76b8992fa5182 | 38.64 | 41.11 | 1.00 |
| **…8cc8fe63227d5c07** | **6.29** | **10.82** | **0.00** |
| …7767ce3fbd5f5f50 | 38.12 | 39.65 | 1.00 |
| …223d02be29c65e81 | 43.60 | 47.33 | 1.00 |
| …4e496bfdb6b95697 | 40.92 | 45.02 | 1.00 |
| …ba13a6004a0c5f8d | 41.06 | 41.49 | 1.00 |
| …9b0a5826d2d357d8 | 32.92 | 34.79 | 1.00 |
| …8551466c8066505e | 45.35 | 47.20 | 1.00 |

**The follower does not systematically overshoot.** On 11 of 12 it travels
slightly *less* than the human. Exactly one overshoots (`2ea6bdf1`, +2.0 m). The
zero in this set (`8cc8fe63`) travelled *less* than GT — because it crashed and
terminated early, not because it over-drove.

So "the geometric driver overshoots the teacher" is not a general property to
gate against. n=12 is small and I will not over-read it, but it is measured, and
it points the opposite way to the premise.

## 3. Where overshoot *does* occur, and what that implies

The one case we know overshot badly is `dd162898`: 6.24 m driven against a
3.64 m human path — a **1.7× overshoot on a short-GT scene**. And R4 found the
evaluator clause that makes this the expensive quadrant:

> `short_gt_distance_override: progress_score = 1.0 when gt_dist_traveled_m < 5.0`

On a short-GT scene we **already hold full progress**. Moving buys nothing and
risks everything. Both facts line up: the overshoot that cost us a zero happened
exactly where motion had zero upside.

That reframes the gate. The useful trigger is not "match the teacher's speed"
(the follower already roughly does, per §2) but **"detect the short-route
condition and hold"**. That is a much narrower, cheaper intervention — and it is
essentially `route_follower_term.py`'s terminal deceleration, which already
exists and under-delivered against its *fidelity* rationale. It may have been
aimed at the right scenes for the wrong reason.

**The obstacle is the same one as §1:** the trigger needs the route-length signal
the driver genuinely has in the sim (its route waypoints) but which I cannot
validate offline. I can write the gate; I cannot tell you whether it fires on the
right scenes without the sim.

## 4. Verdict

**Not viable to validate offline with current data.** Specifically:

- The overshoot premise is not supported on the 12 scenes where it is measurable.
- Modelling the follower offline is infeasible without the sim's route.
- The promising trigger (short-route hold) depends on a signal I cannot
  reconstruct offline.

This is the hedge failing its feasibility check *cheaply*, which is what a hedge
evaluation is for. I would rather report that than hand over a caution profile
fitted to a 112 m artifact.

## 5. The cheap unblock

The whole problem is that the sweep∩labels overlap is 12 scenes. Two ways to fix
it, both small:

1. **Run the teacher over the 200 sweep scenes** (`sample200`). ~1,600 samples,
   a few minutes of GPU at DP's 8.3 samples/s. That takes the overlap from 12 to
   200 and makes §2's table a real measurement instead of an anecdote. This is
   analysis of eval scenes, **not training on them**, so it is unaffected by the
   render-set blocker.
2. **Or** capture the follower's per-step driven trajectory (not just the
   aggregate `dist_traveled_m`) from the `.asl` logs of an existing run. The
   `driver_return` entries hold the actual emitted trajectories — I extracted
   camera frames from those logs already, so the plumbing is proven. That is
   CPU-only and needs no new GPU at all.

**Option 2 is free and I can do it next.** It gives per-step follower behaviour
on the 10 scenes I already have logs for, plus any others on disk — enough to
test the short-route-hold trigger against what the follower actually did,
step by step, rather than against a model of it.

Recommend I take option 2 before any more of this line is built.
