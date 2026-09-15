# Upstream SDK findings (checked 2026-09-15, current `e2e_challenge`)

Our `~/alpasim` is pinned at `42116f8` (2026-07-07). Upstream has moved to
`7611632`. Three things in the delta matter; one of them changes the
submission plan.

Read-only: fetched the upstream README and CLI via raw.githubusercontent. Nothing
pulled into `~/alpasim`.

---

## 1. `nuplan-warmup` — a free, quota-separate end-to-end validation ⭐

**This did not exist in our pinned checkout and it directly solves the problem
that Phase 1.2 and 1.4 are blocked on.**

Verbatim from the upstream README:

> ### Server warmup tracks
>
> `pai-warmup` and `nuplan-warmup` are server-side smoke tests for a driver
> image. They use the same managed evaluator as official submissions, but run one
> fixed wave only: **32 private scenes (16 driver replicas with two simultaneous
> scenes per replica)**. A warmup run has a 45-minute simulation limit.
>
> Warmups are form-factor smoke tests only: completing one does **not** establish
> that an image meets the timing, throughput, or full-workload requirements of
> either official track.
>
> Warmups are not scored, never appear on a leaderboard, and do not change an
> official submission or its ranking. … **Each team has three warmup submissions
> per warmup track in any rolling 30-day window. The official `pai` and `nuplan`
> limits are unaffected.**

```bash
python3 alpasim_challenge.py submit --track nuplan-warmup \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/na-la:<tag>
```

Confirmed in the current CLI source — `submit` and `submissions` both accept
`("pai", "nuplan", "pai-warmup", "nuplan-warmup")`. **Our July CLI does not**
(choices are `("pai","nuplan")` only), which is why we never saw this. Adarsh's
`~/alpasim-cli` copy has it.

This also explains the two `null` fields we saw in `submissions`:
`current_warmup_submission_count` and `remaining_warmup_submissions` — a separate
pool we have never drawn on.

### Why this matters more than it looks

**It runs the exact eval topology (16×2) on the real evaluator at zero official
quota cost.** That makes it the missing validation step for a fresh image:

- **It would have caught issue #151.** That team's total zero was all 16 replicas
  dying at model load on an unrecognised H100 name — a pure form-factor failure,
  exactly what a warmup surfaces. R5's whole risk class is warmup-detectable
  *before* spending a slot.
- It substitutes for Phase 1.2/1.4, which are blocked on CC having no container
  runtime and no GPU. A warmup exercises push + startup + gRPC readiness +
  concurrency for real.
- 3 per 30 days, independent of the official 3/month.

### What it explicitly does NOT do

The README is blunt: completing a warmup *"does not establish that an image meets
the timing, throughput, or full-workload requirements."* So it does **not**
validate the ~35 ms/call aggregate budget, and it is not a substitute for the
full navtest. 32 scenes is also far too few to say anything about zero-rate.

**Recommended addition to the fresh-submission gate:** a green `nuplan-warmup`
on the exact image, before the official submission. Free, and it converts the
most catastrophic failure class (total zero at startup) from "discovered by
spending a slot" to "discovered for nothing".

## 2. `--controller-gains` — a new, CPU-testable lever we weren't using

Also absent from our pinned checkout. A submission may ship a JSON gain set for
the evaluator's nonlinear MPC controller:

```bash
python3 alpasim_challenge.py submit --track nuplan \
  --controller-gains controller_gains.json \
  <image-uri>
```

Accepted fields (any subset; the CLI validates before submit and the evaluator
re-validates; everything else stays fixed):

| Field | Range |
|---|---|
| `long_position_weight`, `lat_position_weight`, `heading_weight` | 0–10 |
| `acceleration_weight` | 0–10 |
| `rel_front_steering_angle_weight`, `rel_acceleration_weight` | 0–10 |
| `idx_start_penalty` | integer 0–19 |

Our rank-1 submission ran with `controller_gains: null`, i.e. defaults.

**Why this is interesting for zero-rate specifically:** the controller is what
*executes* our planned trajectory. Both of our known zeros are front collisions
where we travelled further than the human (`dd162898`: 6.24 m vs 3.64 m). Some of
that overshoot is the planner, but some is tracking behaviour — and
`long_position_weight` / `acceleration_weight` / `idx_start_penalty` are exactly
the knobs governing how aggressively the controller chases the longitudinal
reference.

**Four-axis (prospective, not a proposal yet):** (a) zero-rate — plausibly ↓ if
overshoot is partly tracking-induced, but *unknown*; (b) capability ~0;
(c) latency 0 — evaluator-side, not our RPC; (d) VRAM 0.

I am explicitly **not** proposing a gain change. Its effect is unmeasurable on
the saturated 200-scene set, it needs the 1,485-scene sweep to evaluate, and
guessing gains against a controller we cannot inspect is how you buy a
regression. Recording it as an available lever with a cheap ablation story — one
gain set per arm, no retraining, no rebuild — which is exactly the kind of thing
the technical report wants.

## 3. `throughput_limit.passed: false` — still undocumented upstream

D2's open question. I searched the current upstream README and CLI README; the
only statements are:

> Official evaluation enforces this with a per-track throughput budget.
> Submission status reports the observed throughput wall time and the applicable
> limit.

There is **no statement of what happens when the check fails** — whether it
invalidates the submission, zeroes it, or scales a penalty. The enforcement lives
in the managed evaluator, which is not in the repo, so this is not discoverable
from source.

**→ Organizer question, as the charter anticipated.** It matters because it
decides whether ≈35 ms/call is a hard wall or a soft one, which in turn decides
whether dual-rate inference is mandatory or merely prudent for D2. Suggested
phrasing for Adarsh:

> When `throughput_limit.passed` is false, is the submission invalidated, scored
> zero, or penalised proportionally? We're sizing a model's per-call budget
> against the wall-time limit and need to know whether it's a hard wall.

## 4. Smaller deltas worth noting

- Local smoke presets are now spelled out per track: `+e2e_challenge=dev` for
  PAI and **`+e2e_challenge_nuplan=dev`** for nuPlan. Our `jobs/*.sbatch` already
  use the nuPlan form, so no change needed — but Phase 1.4's charter text says
  `+e2e_challenge=dev`, which is the PAI preset. Worth correcting there.
- The README now states the `full` preset selects the complete scene set without
  a scene limit (relevant to staging the 1,485-scene sweep).
- Warmup artifacts are stored separately and are **not** inspectable beyond
  submission status — so a warmup tells you pass/fail, not per-scene detail.

## 5. What I did not do

Did not pull upstream into `~/alpasim` (every eval run, venv and sbatch depends
on that pin, and the charter says don't). Did not submit a warmup — that is a
submission action and on the HOLD list. Did not open or comment on any issue.
