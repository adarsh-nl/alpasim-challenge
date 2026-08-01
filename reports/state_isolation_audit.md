# Per-rollout state isolation audit — egodriver

Branch `audit/state-isolation` (do not merge). Scope: the gRPC
`egodriver.EgodriverService` implementations at the eval topology,
**16 replica processes × 2 concurrent rollouts per process**.

---

## What I found

**Per-rollout state isolation is sound. I could not produce cross-rollout
contamination in any shipped driver.** All mutable driving state already lives
in a `Session` container keyed by `session_uuid` and created per rollout — including
the confidence EMA the brief called out as the top risk. Across 4 drivers, 2- and
4-way concurrency, round-robin, 300 randomized interleavings and real threaded
contention, every rollout's trajectory sequence was **bit-identical** to running
that rollout alone.

I did find one real defect on the Drive path, in a different failure mode:

> **`drive()` could raise out of the RPC handler.** It resolved the session via
> `_sess()`, which calls `ctx.abort()` on an unknown id — and `abort()` raises. A
> Drive for a session that was never started, or one arriving after
> `close_session`, produced **no trajectory at all** instead of the safe fallback
> the handler promises. This violated the stated Drive contract and is exactly the
> shape of thing that turns into a DriveIRT zero.

Present in all four drivers, **including the shipped `submission/route_follower.py`**.

Two lower-severity items are documented and accepted rather than changed, with
reasoning below: the process-wide `RLock` held across `_plan()`, and the training-only
`rl/policy_driver.py`, which *does* have genuine shared-mutable state but never ships.

## What I changed

1. **`drive()` never raises** — non-aborting `_sess_opt()` lookup; an unknown or
   closed session returns the same empty trajectory `fallback()` already emits when
   no ego pose has arrived. Observation RPCs keep aborting `NOT_FOUND` (correct
   protocol signalling, not on the Drive path). Commit `c6145a9`.
2. **Dev-only state guard** (`RF_STATE_GUARD=1`, default off) — snapshots module-level
   mutables at first session, asserts they never change, and rejects re-use of a live
   `session_uuid`.
3. **`tests/test_state_isolation.py`** — 52 tests. Runs locally, CPU-only, no
   simulator, no renderer, model forward stubbed.

**The fix is behaviour-preserving.** 60 trajectories per driver over 6 rollouts
(including degenerate and route-flapping profiles) are bit-identical pre- vs
post-fix, so the rank-1 baseline is untouched.

## Residual risk

| Risk | Severity | Why it's left |
|---|---|---|
| `self._lock` held across the whole `_plan()` serializes the 2 concurrent rollouts | low | Correctness-safe (it's what *guarantees* isolation under threads). Costs latency, not accuracy: 2.58 ms mean Drive vs a 100 ms budget — ~39× headroom even fully serialized. Narrowing the critical section would trade proven safety for headroom we don't need. |
| `rl/policy_driver.py` has real shared-mutable state (global torch RNG) | none in prod | Training-only; never containerized. Would be a blocker if it ever ships — see §4. |
| Harness drives the servicer directly, not through a real gRPC channel | low | The grpc layer adds a thread pool, which the threaded test already models. It does not add shared driver state. |
| Isolation verified against *my* observation model, not recorded traffic | medium | Profiles are adversarial by construction (degenerate/flapping routes drive the EMA far from 1.0), but they are synthetic. Replaying real `rollout.asl` streams would be strictly stronger. |

---

## 1. Service and session keying

`alpasim_grpc/v0/egodriver.proto` (SDK checkout) — **verified, not assumed**:

```
service EgodriverService {
    rpc start_session (DriveSessionRequest) returns (common.SessionRequestStatus);
    rpc close_session (DriveSessionCloseRequest) returns (common.Empty);
    rpc submit_image_observation (RolloutCameraImage) returns (common.Empty);
    rpc submit_egomotion_observation (RolloutEgoTrajectory) returns (common.Empty);
    rpc submit_route (RouteRequest) returns (common.Empty);
    rpc submit_recording_ground_truth (GroundTruthRequest) returns (common.Empty);
    rpc drive (DriveRequest) returns (DriveResponse);
    rpc get_version (common.Empty) returns (common.VersionId);
}
```

- **Every** request message carries `string session_uuid = 1`. That is the rollout key,
  and it is the only correct one.
- `DriveSessionRequest` also carries **`fixed64 random_seed = 2`** — a *per-session*
  seed the runtime supplies. **Every driver currently ignores it.** Harmless today
  (no shipped driver samples), but it is the intended seeding channel and the fix for
  §4 below.
- `DriveSessionRequest.debug_info.scene_id` is `optional` and documented *"will not be
  present in benchmark runs to avoid any potential data leakage"* — so keying state on
  scene id would work in dev and silently collapse all rollouts onto one key at eval.
  No driver does this. Worth never doing.

Implementations audited:

| File | Servicer | Ships? |
|---|---|---|
| `submission/route_follower.py` | `RouteFollower` | **yes** — this is the image |
| `driver/route_follower.py` | `RouteFollower` | source of the above (kept byte-identical) |
| `driver/route_follower_term.py` | `RouteFollower` | variant |
| `driver/abstention_driver.py` | `AbstentionDriver` | variant — **carries the confidence EMA** |
| `driver/route_follower_gov.py` | `RouteFollower` | variant (needs Pillow; static audit only) |
| `rl/policy_driver.py` | `PolicyDriver` | **no** — training only |

---

## 2. Static audit: every mutable on the Drive path

Classification: **(a)** stateless/per-request · **(b)** per-rollout keyed by session ·
**(c)** SHARED MUTABLE (risk).

### 2.1 Module level — `driver/route_follower.py`, all variants

| Item | file:line | Class | Risk |
|---|---|---|---|
| `LOG` | `route_follower.py:24` | (a) | None. `logging.Logger` is internally locked; no driving state. |
| `V_MAX`, `A_LAT_MAX`, `A_LON_MAX` | `:26–28` | (a) | None. Floats read from env **once at import**, never written. Immutable. |
| `TAU_CONVERGE`, `HORIZON_S`, `DT_US` | `:29–30` | (a) | None. Constants. |
| `ABSTAIN`, `FLOOR`, `ROUTE_HORIZON` | `abstention_driver.py:45–47` | (a) | None. Same pattern. |
| `TERM_DECEL`, `TERM_V_END` | `route_follower_term.py:43–44` | (a) | None. |
| `GOV_ENABLE`, `GOV_GAIN`, `GOV_FLOOR`, `FWD_CAM`, `DEGEN_CREEP_MPS` | `route_follower_gov.py:52–56` | (a) | None. |
| `STATE_GUARD`, `_GUARD_BASELINE`, `_GUARD_SEEN` | `route_follower.py:38–40` | (a)/(c) by design | **Added by this audit.** Mutable, but written only when `RF_STATE_GUARD=1`, never read on the driving path, and excluded from its own snapshot. Off in prod. |

**No module-level mutable is written on the Drive path.** Asserted continuously by
`test_no_module_level_mutable_written_on_drive_path` (all 4 drivers), and the guard
itself is proved to fire by `test_dev_guard_detects_a_planted_module_level_write`.

### 2.2 Servicer instance level

| Item | file:line | Class | Risk |
|---|---|---|---|
| `self._s: dict[str, Session]` | `route_follower.py:129` | (b) | **This is the isolation mechanism.** One entry per `session_uuid`, created in `start_session`, dropped in `close_session`. All reads/writes under `_lock`. |
| `self._lock: threading.RLock` | `:130` | (c) shared, **safe** | Shared by design — it is what serializes access. Held across all of `_plan()`, so the two concurrent rollouts do not overlap. Correct; costs latency only (see Residual risk). |
| `self._server` | `:131` | (a) | Written once by `attach_server`, read only by `shut_down`. Not on the Drive path. |
| `self.policy` (`nn.Module`) | `policy_driver.py:117` | (c) shared | Read-only in `.eval()` under `no_grad` — weights are not per-rollout state. But see §4 for the RNG it uses. |

### 2.3 Per-rollout `Session` — the state that actually matters

All fields use `field(default_factory=...)` for mutables, so no dataclass
mutable-default aliasing (the classic way this bug appears). Verified in all five.

| Field | file:line | Class | Risk |
|---|---|---|---|
| `poses: dict[int, Pose]` | `route_follower.py:88` | (b) | Ring-bounded to 400 entries at `:63–64`. |
| `pose_ts: list[int]` | `:89` | (b) | Same bound. Sorted via `bisect.insort`. |
| `latest: PoseAtTime` | `:90` | (b) | Monotonic-max update. |
| `speed: float` | `:91` | (b) | Overwritten per egomotion message. |
| `route_rig: np.ndarray` | `:92` | (b) | **Freshly allocated** per `submit_route` (`np.array([...])` at `:181`) — no buffer reuse, no aliasing between rollouts. |
| `route_ts: int` | `:93` | (b) | |
| `last_plan: Trajectory` | `:94` | (b) | Written at `:258`, never read on the Drive path. Dead but harmless. |
| **`conf_ema: float`** | **`abstention_driver.py:106`** | **(b)** | **The named top risk. Per-`Session`, correct.** Recursive: `s.conf_ema = 0.6*s.conf_ema + 0.4*c_stable_inst` (`:160`). If shared, one rollout's flapping route would throttle the other's speed. It is not shared — proved by `test_two_concurrent_rollouts_adversarial_pairing`, which pairs a `flapping` rollout with a `clean_straight` one and requires bit-identity for both. |
| **`last_route_len: float`** | **`abstention_driver.py:107`** | **(b)** | Second half of the EMA recursion (`:154–159`). Per-`Session`, correct. |
| `loom_hist: list[(t, area)]` | `route_follower_gov.py:111` | (b) | Image-history buffer, capped at 20 entries (`:134–135`). Per-`Session`. |
| `yield_factor: float` | `route_follower_gov.py:112` | (b) | Derived from `loom_hist`; per-`Session`. |
| `last_img` | `policy_driver.py:73` | (b) | Per-`Session` decoded frame. |
| `log_img/route/ego/pre/logp/val/t` | `policy_driver.py:75–81` | (b) but **unbounded** | Per-`Session`, so no contamination — but grows without limit; see §4. |

### 2.4 Scratch inside `_plan()`

`route`, `pts`, `s_arc`, `d`, `kap`, `v_cap`, `traj` and every loop temporary
(`route_follower.py:186–241`) are **allocated fresh on each call** from local names.
There are no preallocated or cached tensors/arrays reused across Drive calls in any
shipped driver, so there is nothing to alias. Classification: **(a)** throughout.

`np.errstate` is never modified; no global numpy state is touched.

---

## 3. The defect: `drive()` could raise — found, reproduced, fixed

### Before

```python
def drive(self, req, ctx):
    s = self._sess(req.session_uuid, ctx)   # <-- OUTSIDE the try
    try:
        ...
    except Exception:
        return egodriver_pb2.DriveResponse(trajectory=fallback(s, req.time_now_us))
```

`_sess()` (`route_follower.py:138–143` pre-fix) calls
`ctx.abort(grpc.StatusCode.NOT_FOUND, ...)`, and **`abort()` raises**. Because the
lookup sat outside the `try`, the exception escaped the handler: no `DriveResponse`,
no fallback trajectory. With 2 interleaved rollouts per process a stray or late Drive
is a live possibility, and the controller is left with no plan for that tick.

### Harness output — before (`71eb489`)

```
FAILED test_drive_never_throws_on_unknown_session[route_follower-...]
FAILED test_drive_never_throws_on_unknown_session[submission_route_follower-...]
FAILED test_drive_never_throws_on_unknown_session[route_follower_term-...]
FAILED test_drive_never_throws_on_unknown_session[abstention_driver-...]
FAILED test_drive_never_throws_after_close[route_follower-...]
FAILED test_drive_never_throws_after_close[submission_route_follower-...]
FAILED test_drive_never_throws_after_close[route_follower_term-...]
FAILED test_drive_never_throws_after_close[abstention_driver-...]
8 failed, 35 passed, 2 warnings in 21.20s
```

```
driver/abstention_driver.py:239: in drive
    s = self._sess(req.session_uuid, ctx)
driver/abstention_driver.py:142: in _sess
    ctx.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {uuid}")
E   RuntimeError: aborted: StatusCode.NOT_FOUND: unknown session late
```

Note the 35 passes: **every isolation test was already green before the fix.** The
contamination hypothesis was disproved; this was a separate contract bug.

### After (`c6145a9`)

```
....................................................                     [100%]
52 passed, 2 warnings in 22.79s
```

With the dev guard on:

```
RF_STATE_GUARD=1 ... -> 51 passed, 1 skipped, 2 warnings in 24.27s
```

(the skip is `test_dev_guard_is_off_by_default`, which is not observable when the
env var is explicitly set)

Stress run at 300 interleavings per driver:

```
ISO_INTERLEAVINGS=300 ... -k randomized -> 4 passed, 48 deselected in 49.50s
```

### Behaviour-preservation check

Pre-fix vs post-fix driver, identical observation streams:

```
route_follower        60 trajectories over 6 rollouts  -> BIT-IDENTICAL
abstention_driver     60 trajectories over 6 rollouts  -> BIT-IDENTICAL

RESULT: behaviour preserved
```

---

## 4. `rl/policy_driver.py` — real shared-mutable state, does not ship

Not part of the submission (training only, needs torch, not in the image). Recorded
because it *is* an `EgodriverServiceServicer` and would be a blocker if promoted:

| Item | file:line | Class | Risk |
|---|---|---|---|
| **Global torch RNG** | `residual_policy.py:118` (`torch.randn_like`) via `policy_driver.py:239` | **(c) SHARED MUTABLE** | With `RL_DETERMINISTIC=0`, two rollouts in one process draw from **one global stream**. Rollout B's actions then depend on how many samples rollout A happened to take first → not reproducible, not isolated. Masked at eval only because eval sets `RL_DETERMINISTIC=1` (no sampling). |
| `DriveSessionRequest.random_seed` unused | `policy_driver.py:134` | — | The runtime hands us a per-session seed and we discard it. This is the fix: `torch.Generator` per `Session`, seeded from `req.random_seed`, passed to `torch.normal`. |
| `log_img` and friends | `policy_driver.py:75–81`, appended `:247` | (b), **unbounded** | ~169 KiB per Drive call (3×180×320 uint8). At 200 sim steps that is ~34 MiB per rollout, ~67 MiB for two — then `_flush` (`:291–302`) writes an `.npz` per session. If `RL_LOG_DIR` ever pointed at `/tmp`, sustained runs would approach the 2 GiB budget. Fine as-is because it writes into the run directory, not `/tmp`. |

The RNG class of bug is covered by tests rather than left as prose:
`test_global_rng_policy_is_contaminated_by_interleaving` is a **negative control** —
it asserts a shared global stream *does* leak, proving the harness can detect this —
and `test_per_session_rng_is_isolated_under_interleaving` demonstrates the
per-session-generator fix holding over 100 interleavings. Both use a stub policy, so
they run without torch.

---

## 5. Budgets

Measured on the shipped driver, CPU only:

| Budget | Measured | Verdict |
|---|---|---|
| Per-rollout state | **353 KiB** saturated (pose ring at its 400 cap); **705 KiB** for 2 concurrent | negligible |
| Leak across sessions | 2000 sequential sessions / 12k Drives → RSS **56628 → 56596 KiB (−32 KiB)**, 0 live sessions | no leak; eviction works |
| Cost of *not* evicting | ~488 MiB of populated state for those 2000 sessions | eviction is load-bearing |
| `/tmp` (2 GiB) | shipped driver performs **no disk writes at all** (no `open()`/`savez`/`tempfile` in the module) | 0 B used |
| VRAM (16 GiB) | shipped driver is pure numpy, **no GPU allocation** | 0 used |
| Drive latency (0.1 s mean) | **2.58 ms** mean over 60 calls | 39× headroom |
| Cost of per-rollout keying | dict lookup **51 ns** = **0.002 %** of a Drive | free, as required |

Eviction is by explicit `close_session`. `test_close_session_evicts_state` runs 40
full rollout lifecycles and requires `len(servicer._s) == 0`.

**Caveat, stated plainly:** if the runtime ever fails to send `close_session` (replica
killed mid-rollout, controller crash), that session's ~353 KiB is retained for the life
of the process. There is no idle-timeout eviction. At 2 concurrent rollouts and a
process lifetime of one eval, the exposure is bounded and small, so I did not add a
reaper — it would be new failure surface on the Drive path for no measurable gain.
Recorded here so the choice is visible rather than implicit.

---

## 6. Test inventory — `tests/test_state_isolation.py`

Runs with `~/alpasim/.venv-driver/bin/python -m pytest tests/test_state_isolation.py -q`.
No GPU, no simulator, no renderer, no SLURM. Uses the real `alpasim_grpc` protos when
importable and falls back to an in-process stub otherwise, so it runs in a bare env too.

| Test | What it proves |
|---|---|
| `test_two_concurrent_rollouts_round_robin` | Eval topology exactly: 2 rollouts, round-robin, bit-identical to solo |
| `test_two_concurrent_rollouts_adversarial_pairing` | **The EMA test.** A `flapping`-route rollout paired with a clean one poisons neither |
| `test_randomized_interleavings` | 120 (default; 300 on demand) randomized order-preserving interleavings over 4 rollouts |
| `test_threaded_concurrency_matches_solo` | Real threads + barrier start; no races, no exceptions, bit-identical |
| `test_drive_never_throws_on_unknown_session` | The fixed defect — unknown id |
| `test_drive_never_throws_after_close` | The fixed defect — late Drive after `close_session` |
| `test_drive_before_any_pose_returns_trajectory` | Fallback, not exception, before first egomotion |
| `test_close_session_evicts_state` | 40 lifecycles → 0 residents |
| `test_pose_buffer_is_bounded` | 1200 pose updates → ring stays ≤ 400 |
| `test_no_module_level_mutable_written_on_drive_path` | No module global mutates across the Drive path |
| `test_dev_guard_detects_a_planted_module_level_write` | **Positive control** — the guard fires on a real leak |
| `test_dev_guard_rejects_session_id_reuse` | Re-used `session_uuid` is rejected in dev |
| `test_dev_guard_is_off_by_default` | Zero cost in prod |
| `test_global_rng_policy_is_contaminated_by_interleaving` | **Negative control** — harness can detect an RNG leak |
| `test_per_session_rng_is_isolated_under_interleaving` | Per-session generator fix holds over 100 interleavings |
| `test_session_lookup_is_constant_time` | Keying stays O(1) at 10k sessions |

Each of the first 11 is parametrized over all four numpy drivers → 52 tests.

### Why a leak would have been caught

The rollout profiles are chosen so contamination is *loud*, not subtle:

- `clean_straight` / `clean_curve` — high confidence, `conf_ema → 1.0`
- `degenerate` — 1-waypoint route on alternating steps, forcing the abstention branch
- `flapping` — route length alternating 2↔14 waypoints, driving `c_stable` hard toward 0

A shared EMA would drag a clean rollout's `v_cap` down by the flapping rollout's
instability, changing every pose in the emitted trajectory. The assertion is on the
exact IEEE-754 bytes of every pose (`struct.pack`), not a tolerance — so a single
perturbed multiplier anywhere in the speed profile fails the test.

---

## 7. Guardrails observed

No submission, no Docker build, no ECR push, no SLURM job, no GPU job. Nothing touched
under the model, scoring, or test-set paths. All work on `audit/state-isolation`, not
merged. Everything above ran locally on CPU in under a minute per invocation.
