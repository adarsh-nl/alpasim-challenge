# Staged regression gate — drive-handler fallback fix

**Status: STAGED, NOT RUN.** Scoring needs a GPU eval run. Adarsh runs this, not me.

Candidate: `fix/drive-handler-fallback` (branched from `main` @ `18d921a`)
Baseline: `main` @ `18d921a` — the shipped rank-1 driver
Changed file that ships: `submission/route_follower.py` only

---

## What the gate is being asked to decide

**Not** "is the candidate better?" — it cannot be, on score. The happy path is
already proven byte-identical, so the expected score delta is exactly zero.

The gate is here to **falsify** that claim at scale. It answers: *does the fix
perturb driving on any real scene?* A clean null says the bit-identity proof
held across 1,491 scenes of real closed-loop feedback, and the fix is safe to
ship. Anything else means the local proof missed something and shipping stops.

The **value** of this change is the eliminated crash path, not a score gain.
That value is already demonstrated by `tests/test_drive_handler_fallback.py`;
the gate exists to confirm we did not pay for it.

---

## The commands

### 1. Run both arms (paired, same scenes, same node)

```bash
cd /home/nanjaiyalathaa/alpasim-challenge
sbatch --array=0-31%8 jobs/handler_fix_gate.sbatch
```

The script (`jobs/handler_fix_gate.sbatch`, staged in this branch) extracts both
drivers straight from git —

```bash
git show main:submission/route_follower.py                        > driver_baseline.py
git show fix/drive-handler-fallback:submission/route_follower.py  > driver_candidate.py
```

— then runs **both arms back-to-back inside the same array task**, on the same
node, against the same `/local`-staged assets and the same scene list, with
`RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0` (the rank-1 config) on both. That
pairing matters more than usual here: the expected effect is exactly zero, so
node-to-node or staging variation would be indistinguishable from the signal.

Outputs land in:

```
$HOME/alpasim/runs/handlerfix-<ARRAY_JOB_ID>/baseline/task-*/
$HOME/alpasim/runs/handlerfix-<ARRAY_JOB_ID>/candidate/task-*/
```

### 2. The gate itself

```bash
cd /home/nanjaiyalathaa/alpasim-challenge/rl
RL_REWARD=v2 python regression_gate.py \
  --baseline  $HOME/alpasim/runs/handlerfix-<ARRAY_JOB_ID>/baseline \
  --candidate $HOME/alpasim/runs/handlerfix-<ARRAY_JOB_ID>/candidate \
  --tol 0.02
```

### 3. Supporting reads

```bash
python notes/compare.py $HOME/alpasim/runs/handlerfix-<ID>/{baseline,candidate}
python notes/score.py   $HOME/alpasim/runs/handlerfix-<ID>/candidate
```

---

## Prediction — written before the run, falsifiable

| Quantity | Predicted | Why |
|---|---|---|
| `mean diff` (candidate − baseline) | **0.0000** | Happy path is byte-identical; the driver emits the same trajectory for the same observations |
| quality-among-both-valid | **null, 0.0000** | Same reason — this is the decomposition's driving-quality term |
| valid-drive rate | **flat**, or up by 0–2 scenes | Can only rise, and only if a scene actually hits the unknown/post-close Drive path |
| per-scene diffs | **all exactly 0.0** | Not "small" — *exactly* zero, since the trajectories are bitwise equal |
| worst-5 regressions table | **all 0.000** | |
| `regression_gate.py` verdict | **INCONCLUSIVE** | With all diffs at 0, the bootstrap CI collapses to [0, 0], which spans the ±0.02 tolerance band. The script's own wording — *"need more scenes, not a false 'cleared'"* — is **wrong for this case** and should be ignored; see below. |

### PASS condition

**A clean null.** Specifically, all three must hold:

1. `mean diff` = 0.0000 (tolerance: `|Δ| < 0.005` to absorb any sim nondeterminism)
2. **no scene regresses** — the worst-5 table shows no negative diff
3. valid-drive rate is flat or higher

`INCONCLUSIVE` is the **expected and passing** verdict here. The gate was built
to stop us shipping noise as a win; this change is deliberately a null, so its
"cannot resolve a 0.02 effect" output is the correct answer to a question we are
not asking. Read the per-scene diffs, not the verdict line.

### FAIL condition — stop, do not ship

**Any scene with a non-zero diff.** That contradicts the local bit-identity
proof and means one of the following, in order of likelihood:

1. **Sim nondeterminism**, not the fix. Check this *first*: re-run the baseline
   arm against itself (`--baseline baseline --candidate baseline` across two
   separate runs of the same driver) and see whether the same scenes move. If
   baseline-vs-baseline is also non-zero, the delta is renderer/controller
   noise and the gate needs paired-noise subtraction before it can say anything.
2. The tombstone changed `close_session` timing enough to matter — should be
   impossible (it runs after the last Drive of a rollout).
3. The local proof's observation model missed a branch that real scenes take.

### What the gate almost certainly will *not* show

The runtime opens a session before driving it and does not drive after closing
it, so **navtest probably never triggers the unknown/post-close path at all**.
A completely flat result therefore does **not** demonstrate the fix's value — it
only confirms the fix costs nothing. Do not read a null here as "the crash path
was not real": it was real, it is reproduced by a negative-control unit test
against the baseline driver, and it would have fired on any runtime hiccup,
replica restart, or late retry that navtest happens not to exercise.

---

## Local evidence already in hand (CPU, no GPU needed)

```
$ ~/alpasim/.venv-driver/bin/python -m pytest tests/test_drive_handler_fallback.py -q
12 passed, 2 warnings in 3.56s
```

Happy-path identity vs `main:submission/route_follower.py`:

```
profiles: 6   trajectories compared: 60
  straight_fast      10 drives   32640 bytes  -> BIT-IDENTICAL
  straight_slow      10 drives   32640 bytes  -> BIT-IDENTICAL
  curve_left         10 drives   32640 bytes  -> BIT-IDENTICAL
  curve_right        10 drives   32640 bytes  -> BIT-IDENTICAL
  short_route        10 drives   32640 bytes  -> BIT-IDENTICAL
  degenerate_route   10 drives   32640 bytes  -> BIT-IDENTICAL

happy-path identity: PASS
```

195,840 bytes of trajectory compared, zero differing bits. Profiles cover the
straight branch, both curve directions, a short route, and the degenerate-route
(<2 waypoint) fall-through to `straight()`.

Crash path, before vs after:

| Case | Baseline (`main`) | Candidate |
|---|---|---|
| Drive, session never opened | **raises** out of the handler | empty trajectory, no raise |
| Drive after `close_session` | **raises** out of the handler | decelerating stop from last known pose |
| `submit_route` / `submit_egomotion_observation` on unknown session | aborts NOT_FOUND | aborts NOT_FOUND (unchanged, test-locked) |

---

## Cost note

`navtest_full` is 1,491 scenes with assets on disk, run **twice** (both arms).
At the observed ~6 h for a 20-task 200-scene array, budget roughly 8 h wall with
`--array=0-31%8`. If that is too much for a change whose predicted delta is
zero, the cheaper first pass is the same script with
`nuplan_scenes=sample200` — the 200-scene set used for every prior measurement
in `notes/results.md`. A null on sample200 plus the bit-identity proof is
already strong; full navtest is the belt-and-braces version.
