# AlpaSim E2E Closed-Loop Challenge 2026 — nuPlan track

Read `README.md` first: it holds the confirmed scoring formula, the eval spec,
and the Docker workaround. This file covers how the code is laid out and how to
work in it. `~/CLAUDE.md` holds the cluster rules (they apply here in full —
**nothing but file inspection, git, and config edits runs on the login node**).

## Two repos, and why

| Path | What it is |
|---|---|
| `~/alpasim-challenge` | **This repo.** Our drivers, RL stack, sbatch scripts, analysis. |
| `~/alpasim` | Upstream AlpaSim SDK checkout (`uv`-managed). Not ours; don't edit it except `src/wizard/configs/nuplan_scenes/*.yaml`. |

Everything that invokes the simulator must run with `cwd=~/alpasim` so `uv run`
resolves the SDK env. Job scripts set `REPO=~/alpasim`, `CHAL=~/alpasim-challenge`
and pass `$CHAL/driver/...py` as an absolute path. Two venvs exist there:
`.venv` (the SDK, via `uv run`) and `.venv-driver` (minimal grpc+numpy env used
by older jobs to prove the driver has no SDK dependency). Newer jobs just use
`uv run python` for the driver too.

**Outputs live in the SDK repo, not here:** runs go to `~/alpasim/runs/<name>-$JOBID/`,
slurm logs to `~/alpasim/logs/`. This repo stays small — code, scene lists, notes.

## Running the simulator (the bare-process pattern)

No Docker/enroot on the cluster, so both wizard deployment backends are dead.
Every job follows the same five steps — copy `jobs/smoke2.sbatch` as the template:

1. `uv run alpasim_wizard +e2e_challenge_nuplan=dev wizard.run_method=NONE wizard.log_dir=$RUN`
   — generates configs and exits without launching containers.
2. Launch three background processes on localhost: **driver :6789**,
   `alpasim_controller.server` **:6001**, `alpasim-mtgs-server` **:6000**.
3. Poll all three ports with `/dev/tcp` until they listen (renderer takes minutes).
4. `uv run python -m alpasim_runtime.simulate --user-config ... --network-config ...`
5. `trap cleanup EXIT` to kill the three PIDs.

Non-negotiables:
- `--gres=gpu:lovelace:1` — gsplat kernels are sm_89; an A100 fails.
- `module load nvidia/cuda-12.4`, `TORCH_CUDA_ARCH_LIST=8.9`, `LC_ALL=C.UTF-8`,
  `UV_LINK_MODE=copy`, `ALPASIM_NUPLAN_ROOT=/datasets/eemcs/ps/alpasim-nuplan-2026`.
- Array jobs must offset ports per task (`PORT=$((6789 + TID*10))`) **and** `sed`
  the generated network config to match — see `jobs/array.sbatch`, which also
  stages per-scene assets to `/local/$SLURM_JOB_ID` and retries `simulate` 3x.

Restrict a run to specific scenes with `scenes.limit_to_first_n=0
"scenes.scene_ids=[$(paste -sd, some_scenes.txt)]"`. Add
`eval.video.render_video=false` unless you actually need video.

## Driver contract

Drivers are gRPC `EgodriverServiceServicer` implementations (`driver/*.py`,
`rl/policy_driver.py`). Rules that the scoring depends on:

- **Per-session state, keyed by `session_uuid`.** Official eval runs
  `n_concurrent_rollouts=2`; shared mutable state corrupts both rollouts.
- **`drive()` must never raise.** Wrap in try/except → `fallback()` (decelerate
  along current heading). An exception escaping the RPC scores the scene 0.
- Only `driver/route_follower.py` has no ML dependency; it is what ships.
  `submission/route_follower.py` is its deployed copy — **keep the two in sync**.
- Latency budget is 0.1 s/call and the follower uses 0.003 s. Don't optimize it.

### Driver variants and the regression-safe discipline

`route_follower_gov.py` (camera looming governor), `route_follower_term.py`
(terminal deceleration), `abstention_driver.py` (route-confidence abstention)
are each a **copy** of `route_follower.py` with exactly one behavioral change,
gated behind an env flag, written so that with the flag off — or when the
triggering condition is absent — the trajectory is byte-identical to the
baseline. Preserve that property when adding a variant: it makes every A/B a
clean paired comparison and bounds the downside to the scenes it targets.
Divergence between the copies is the accepted cost; don't refactor them into a
shared base class without a reason.

## RL stack (`rl/`)

Closed-loop PPO on a **residual speed multiplier** `m ∈ (0,1]` scaling the
follower's curvature-limited speed profile — not driving from scratch. 1-D
action, ~2M params.

```
train_loop.py            orchestrator: rollouts -> batch -> PPO -> repeat
  run_rollouts()         re-runs the bare-process pattern in-process (subprocess)
  policy_driver.py       follower geometry + policy multiplier; logs transitions
                         to $RL_LOG_DIR/<session_uuid>.npz
  collect_batch.py       joins transition .npz with aggregate metrics -> batch
  reward.py / reward_v2.py   v2 is DriveIRT-aligned: hard 0 on collision/offroad
  ppo_update.py          clipped surrogate, advantage = return - value (no GAE)
  residual_policy.py     SmallCNN(CAM_F0) + route + ego -> squashed Gaussian + value
```

Env knobs (set in the sbatch, read at import time): `RL_REWARD=v2`,
`RL_LOG_STD_INIT`, `RL_MEAN_BIAS`, `RL_PERSTEP`, `RL_FID_D0`, `RL_BASE`,
`RL_PROG_GATE`, plus `RL_POLICY_CKPT` / `RL_LOG_DIR` / `RL_DETERMINISTIC` on the
driver side. `rl/rl_train_v4_full.sbatch` is the last validated config.

The log-prob in `residual_policy` is squash-corrected (Gaussian minus sigmoid
Jacobian) and `policy_driver` logs the **exact pre-squash sample**, not the mean
— an earlier version logged inconsistently and the policy never moved.

Scene lists are generated, not hand-written: `select_discriminating_scenes.py`
keeps scenes where the follower is neither saturated at 1.0 nor a hard 0 (flat
reward landscape was the v1/v2 failure mode). `make_scene_lists.py` is the older
naive split.

## Measure → diagnose → fix

The working method here, and the thing to preserve:

- `notes/score.py <run_dir>` — per-scene scores, worst 20, failure histogram.
- `notes/compare.py <run_dir>...` — cross-run table (mean, zeros, at-fault,
  offroad, progress, minObs, rpc ms).
- `rl/regression_gate.py --baseline <run> --candidate <run> --tol 0.02` — paired
  bootstrap CI + permutation test on shared scenes, plus a **reliability vs
  quality** split (valid-drive rate vs score gap among scenes both drove
  validly). Reports SHIP / HOLD / INCONCLUSIVE, and reports INCONCLUSIVE
  honestly rather than declaring a noisy win.

Run the gate before claiming an improvement. `notes/results.md` records the key
negative results — navtest_dev is saturated (a blind follower scores 1.000),
V_MAX sweeps were null, A_LON was the binding constraint, and looming-as-classifier
could not separate collisions from clear scenes.

## Scoring reminders that change design decisions

Gated, not weighted: `collision_at_fault == 0` **and** `offroad == 0`, then
`score = min(clamp(progress_rel,0,1)/0.8, 1)`.

- Rear-end is **not** at fault → braking and conservatism are free.
- No credit above 80% of GT distance → don't chase speed past the cap.
- Leaderboard aggregates with zero-inflated IRT → optimize the failure tail, not
  the mean. One zero costs more than many small gains.

## Submission

`submission/` holds the shipped driver + `Dockerfile` (baked
`RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0`, non-root user, port 6789) and the
vendored `grpc/` proto stubs. **Build the image on a laptop, not the cluster.**
Log every submission in `submissions.log` — submissions are scarce.

## In flight (untracked)

`rl/run_teacher.py` + `rl/trajdata_to_dp.py` + `rl/dp_repo/` + `rl/dp_ckpt/`:
Diffusion-Planner as a privileged teacher (reads the trajdata cache directly,
bypassing nuplan-devkit) generating distillation targets in `rl/teacher_labels/`.
Sharded via `rl/teacher.sbatch`. Not committed yet.

Housekeeping: `rl/slurm-*.out` and `rl/teacher_labels/` should not be committed —
`.gitignore` covers `runs/ logs/ data/` but not those.
