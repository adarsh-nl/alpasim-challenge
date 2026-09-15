# Throughput budget — the binding constraint, quantified

**Four-axis:** analysis only — (a) 0 (b) 0 (c) 0 (d) 0. Justified because it
changes what the student may cost, i.e. it constrains (b) and (c) for all of
Track B before any GPU time is spent.
**Gate state:** read-only analysis, nothing built or merged.

---

## Headline

**The 0.1 s "budget" is a target, not a gate. The real constraint is the
aggregate per-track wall-time budget, and at full navtest it allows the driver
roughly 38 ms per Drive call — not 100 ms.**

And the R6 prereq answer: **a slow `Drive` does NOT zero the route.** There is no
per-call deadline anywhere in the path. It consumes aggregate wall time, and the
failure mode is submission-level, not route-level.

---

## 1. Does a slow Drive zero the route? No — evidence

### The contestant contract calls 0.1 s a *target*

`e2e_challenge/README.md`, verbatim:

> - target 0.1 seconds or less of model work per `Drive` call
>
> The 0.1 second target corresponds to a 10 Hz driving loop: a policy should be
> able to produce each driving decision within one control tick. **Official
> evaluation enforces this with a per-track throughput budget.** Submission
> status reports the observed throughput wall time and the applicable limit.

"Target", and enforcement is explicitly *"a per-track throughput budget"* — i.e.
aggregate, not per-call.

### There is no per-call deadline in the runtime

`src/runtime/alpasim_runtime/services/driver_service.py` builds the `DriveRequest`
and awaits the stub with **no `timeout=` argument**, and:

```
$ grep -n -iE "timeout|deadline" src/runtime/alpasim_runtime/services/driver_service.py
  NONE
```

So a slow `Drive` is not cancelled, does not raise `DEADLINE_EXCEEDED`, and does
not lose its trajectory. It blocks that rollout's step loop and burns wall time.

### Consequence — and why it may be *worse* than a route zero

The penalty is not one zeroed route; it is the **submission** failing its
throughput check (`throughput_limit.passed`). Our rank-1 run reports
`passed: true` with a 328 s margin. What `passed: false` actually does —
invalidate the submission, or merely annotate it — is **not documented in the
July SDK and I could not determine it.** Treat it as submission-level risk until
confirmed, because a single overrun spread across all routes could in principle
cost the whole run rather than one score.

This reframes R2's missing-watchdog finding: a watchdog does **not** protect
against a route zero (there is no per-call gate to trip). Its value would be
protecting the *aggregate* budget by capping worst-case per-call cost. Different
rationale, and a weaker one — noted so we don't justify that work on the wrong
grounds.

## 2. The numbers

Known quantities:

| Quantity | Value | Source |
|---|---|---|
| Drive calls per rollout | **200** | `n_sim_steps: 200` @ `base.yaml:106` |
| Control tick | 500 ms | `control_timestep_us: 500_000` @ `base.yaml:108` |
| Parallel rollout slots | **32** | 16 replicas × 2 concurrent (`e2e_challenge/README.md`) |
| `navtest_full` scenes | **1485** | `src/wizard/configs/nuplan_scenes/navtest_full.yaml` (note: our README says 1491 — the config says 1485) |
| Observed wall time | 2157.161 s | API, submission `1bb06e8b` |
| Limit wall time | 2485.438 s | ditto |
| Margin | **328.277 s** (13.2%) | ditto |

Model: added wall time = (extra per-call cost × 200 calls × N rollouts) / 32 slots.
Solving for the per-call headroom, `Δ ≤ margin × 32 / (200 × N)`:

| rollouts N | observed/step | limit/step | **headroom per call** | driver today |
|---|---|---|---|---|
| **1485** | 232.4 ms | 267.8 ms | **35.4 ms** | 2.6 ms |
| 1200 | 287.6 ms | 331.4 ms | 43.8 ms | 2.6 ms |
| 1000 | 345.1 ms | 397.7 ms | 52.5 ms | 2.6 ms |
| 500 | 690.3 ms | 795.3 ms | 105.0 ms | 2.6 ms |
| 200 | 1725.7 ms | 1988.4 ms | 262.6 ms | 2.6 ms |

At N = 1485:

- total per-step budget across **all** services: **267.8 ms**
- currently consumed: 232.4 ms, of which **our driver is 2.6 ms (1.1%)** — the
  rest is renderer + controller + physics
- spare for a student forward: **35.4 ms/call**
- → **total per-call driver budget ≈ 38 ms, not 100 ms**
- sanity check: 2.15× faster than real time (100 s simulated per rollout),
  which is plausible for MTGS renders at 2 Hz and corroborates the model

## 3. Is dual-rate inference optional or mandatory?

**Conditional, and the condition is sharp:**

- student forward **≤ ~35 ms/call** at full navtest → dual-rate is **optional**
- student forward **> ~35 ms/call** → dual-rate (or equivalent amortisation) is
  **mandatory**, because the aggregate budget is breached even though every
  individual call is comfortably inside the 0.1 s target

That second case is the trap: a 60 ms forward looks fine against the advertised
0.1 s per-call target and would still blow the throughput limit. **The per-call
target and the aggregate budget disagree, and the aggregate is the one that is
enforced.**

For scale: a ~2 M-param CNN student on an L40S is plausibly single-digit ms and
would be fine; anything transformer-scale at 8-camera input will not be. The
number to design against is **35 ms**, with a safety factor — not 100 ms.

## 4. Assumptions and how to kill them

The per-call figures are only as good as N, and **N is genuinely unknown**:
`rollout_coverage` is `null` in the submission status, and the rank-1 run was
*soft-open*, which per the organizers used a different (pre-final) scene set.
1485 is our local `navtest_full` count, not a confirmed count for that run.

Three assumptions, in decreasing confidence:

1. **200 Drive calls per rollout** — high confidence, straight from the eval config.
2. **32 parallel slots sustained** — high confidence from the README's topology,
   but if the scheduler does not keep all 32 busy, real headroom is *smaller*
   than the table says.
3. **The limit scales with rollout count** — *unverified*. If `limit_wall_time_s`
   is instead a flat per-track constant, then a larger final scene set means the
   headroom per call shrinks further. `2485.438` is suspiciously precise, which
   hints at a computed rather than constant value, but that is a guess and I am
   not resting anything on it.

Cheapest ways to pin it down, in order:

- ask the organizers for the rollout count / how `limit_wall_time_s` is derived
  (one question, and it is the single number that makes this table exact);
- or recover it from the *rerun* once it lands, since that runs on the final
  scenes and will report both numbers;
- or instrument a local full-navtest sweep and count `Drive` calls directly —
  that is already on the list as a GPU task and would settle assumption 2 as well.

**I have not guessed a single number here.** Everything above is either from the
config, from the API, or shown as a sensitivity row.

## 5. What this changes

1. `README.md`'s conclusion *"Latency is NOT the bottleneck. Deprioritise
   FP8/TensorRT/dual-rate work."* was drawn from the per-call target and is
   **wrong in aggregate**. It should be revised — flagging, not editing, since it
   is a shipped-path document and the correction depends on N.
2. D2's VRAM-first framing is incomplete: the student must be engineered to a
   **~35 ms/call** compute budget alongside the ≤16 GiB one, and that likely
   constrains architecture more than VRAM does.
3. FP8/TensorRT work moves from "deprioritised" to "possibly load-bearing",
   depending entirely on where the student's forward lands relative to 35 ms.
