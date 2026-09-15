# D1 — teacher label integrity, shard 0

**Four-axis:** validation only — (a) 0 (b) 0 (c) 0 (d) 0. Justified because it
gates a ~1.4 GB / 15-shard GPU spend and the student's supervision target.
**Smoke:** `~/alpasim/.venv/bin/python rl/check_teacher_labels.py --labels rl/teacher_labels`
→ VERDICT PASS (CPU, ~90 s).
**Gate state:** shard 0 verified; shards 1–15 **staged, not submitted**
(`rl/teacher_shards_1_15.sbatch`).

---

## Verdict: PASS on the hard checks

| Check | Result |
|---|---|
| Schema (keys/shapes/dtypes/finite) | **6080/6080 readable, 0 errors** |
| Dedupe | **6080 unique tokens, 0 duplicates** |
| Kinematics (run_teacher's own flag criteria, recomputed) | **0.0329% flagged (2/6080)** |
| Coverage | 760 scenes × exactly 8 samples, ts stride 5 (2 Hz) |
| ADE vs GT | **0.370 m @1.0 s · 0.582 m @available** (see §2) |

Both kinematic flags are `BACKWARDS` — not `NONFINITE`, not velocity or accel
violations. `v_max` median 6.60 m/s (p99 11.77), `a_max` median 0.77 m/s²
(p99 2.11), 8 s path length median 39.77 m. All plausible urban values.
Cities: `boston` 1872, `las_vegas` 2032, `pittsburgh` 1216, `singapore` 960.

**The charter's claimed 0.03% flag rate is confirmed exactly.**

## 1. Correction to something I said earlier

In my previous commit message I wrote that the charter's *"ADE median 0.58 m vs
GT"* was "not reproducible as stated". **That was too strong — the figure is
real.** Measured over the GT horizon that actually exists, ADE median is
**0.582 m**, matching it to three digits.

What was genuinely missing was its *provenance and horizon*: `run_teacher.py`
computes no ADE and no GT comparison anywhere, so the number could not be traced
to any code, and no horizon was stated. Both are now pinned, and the match also
**validates my GT reconstruction** — that was the point of making this check
self-validating (a wrong frame convention yields tens of metres, not 0.58).

## 2. The 8-second horizon is mostly unvalidatable — and that matters for D2

The teacher emits `ego_future` as **80 steps (8.0 s)**. The trajdata cache cannot
check most of it:

- every scene holds **exactly 56 ego timesteps** (5.6 s at dt=0.10)
- labels are sampled at **ts ∈ {20,25,30,35,40,45,50,55}**
- so GT available *after* ts is **median 17 steps**, max 35, and **zero at ts=55**

Measured on a 300-sample draw:

```
compared      : 231/300  (69 had <10 GT steps available)
GT horizon    : median 20 steps (2.0s), max 35 -- teacher predicts 80 (8.0s)
ADE @1.0s     : 0.370 m   <- the comparable window
ADE @available: 0.582 m  (p90 2.325, max 7.612)
```

So **~75% of every teacher label is extrapolation beyond any recorded ground
truth**, and one eighth of samples (all ts=55, i.e. 760 of 6080) have no GT
future at all.

### Why this is not automatically a problem

Distillation deliberately targets the *teacher's* output, not GT — the teacher's
privileged map/agent access is the whole point, and its 8 s extrapolation is
plausibly better than nothing. So unvalidatable ≠ wrong.

### Why it is still a real design constraint for D2

We can only *verify* the first ~2 s. If the student is supervised uniformly
across all 80 steps, three quarters of its loss comes from a signal we have never
checked, and the error growth is steep: ADE rises 0.370 m @1 s → 0.582 m
@2 s, with p90 2.33 m and a max of 7.61 m. That tail is where a confidently-wrong
teacher would hide.

Concrete options for D2, in increasing ambition:

1. **Horizon-weighted loss** — down-weight steps beyond the verifiable window.
   Cheapest, and directly ablatable.
2. **Truncate supervision** to ~2–3 s and let the tracker handle the rest. The
   driver only needs 5 s of trajectory at 10 Hz for the controller, and the
   near-horizon is what collisions depend on.
3. **Use the horizon as the uncertainty signal** — this is a natural fit with the
   planned information-gated distillation contribution: per-step confidence is
   already implicit in how far past GT we are, so "abstain where the teacher is
   extrapolating" comes almost free and is honest about *why* it abstains.

I am not choosing among these — it is a D2 design decision and should be made
with the student's architecture in hand. Flagging that option 1 is nearly free
and worth defaulting to.

## 3. Coverage facts worth knowing before spending 15 shards

- **760 scenes ≈ 1/16 of ~12,160**, consistent with `--n-shards 16`. The full
  set projects to **~97k samples, ~1.4 GB**.
- **8 samples per scene, at ts 20–55**, i.e. we sample the *back 3.5 s* of each
  5.6 s scene and never the opening. Whether that matters depends on whether
  scene starts are systematically different (e.g. more stationary) — worth a
  glance before committing to the same stride for all 16 shards.
- Sampling is uniform: exactly 8 per scene, min = max = 8, so no scene is
  over- or under-represented.

## 4. Housekeeping finding — misleading failure logs

The 32 files `rl/slurm-546098_*.out` are **all one-line failures**:

```
/tmp/slurmd/job546184/slurm_script: 4: source: not found
```

That is a `--wrap`-style submission executed by `sh`, where `source` is not a
builtin — **exactly what charter rule 7 forbids**. `rl/teacher.sbatch` itself is
correct (`#!/bin/bash` + `set -euo pipefail`), so these logs are from a
different, abandoned attempt. The 6080 good labels were produced by a separate
successful run on 2026-07-25 09:51–09:58.

They are pure clutter and read as "the teacher run failed" to anyone glancing at
the directory. Recommend deleting them (they are untracked). Not deleted — they
are not mine to remove and the charter says flag rather than tidy.

## 5. What the harness does not check

- **Inputs.** The labels contain only `ego_future`, `neighbor_future`, `token`,
  `map_name` — no camera frames, no vector-map features. The student's inputs
  must be produced separately (via `trajdata_to_dp.py` for teacher inputs, or
  from MTGS renders for the camera student). **This is an open D2 gap**: there is
  currently no camera-input pipeline keyed to these tokens.
- **`neighbor_future` quality.** Only `ego_future` is audited. Neighbour
  predictions are stored but unvalidated; if the student ever supervises on them,
  extend the harness.
- **Cross-shard consistency.** Cannot be checked until shards 1–15 exist. Re-run
  the harness over the full set afterwards; the staged sbatch prints the exact
  command on completion.
