# CHECK 1 — replay fixture provenance

# VERDICT: REAL

The 10 tracked fixture files are faithful extractions of genuine recorded
`rollout.asl` streams from five real SLURM runs. Nothing was synthesized. The
"real streams" claims in `reports/state_isolation_audit.md` are **confirmed
true** and need no walk-back.

One qualification, stated up front so the claim is not read wider than it is:
the fixtures are a **derived cache**, not the original `.asl` artifacts. They
contain a lossless copy of the fields the driver consumes, and deliberately drop
camera bytes and cap at 10 Drive calls per rollout. See "What 'REAL' does and
does not mean" below.

*Read-only audit. Nothing was edited, committed, merged or pushed.*

---

## 1. The files

10 tracked files, `tests/data/replay/*.json`, single-line JSON (ASCII), 104 KiB
total. All added by commit **`07af779`**.

| File | Bytes |
|---|---|
| `2021.05.25.14.16.10_veh-35_01100_01664-f02b15cc2.json` | 10,576 |
| `2021.05.25.14.16.10_veh-35_01690_02183-0c77bbb19.json` | 8,640 |
| `2021.05.25.14.16.10_veh-35_01690_02183-2cd1545c4.json` | 9,290 |
| `2021.06.03.13.55.17_veh-35_02866_03582-9b58f5bd4.json` | 8,498 |
| `2021.06.28.16.57.59_veh-26_00016_00484-bbb7e4525.json` | 8,666 |
| `2021.08.16.14.23.37_veh-45_00015_00132-5fadffd02.json` | 8,463 |
| `2021.09.09.18.04.06_veh-40_00743_01071-6d2b59f78.json` | 7,674 |
| `2021.09.16.14.39.34_veh-42_00297_00935-c7a44a2e5.json` | 9,438 |
| `2021.09.16.15.47.30_veh-45_01199_01391-dd162898e.json` | 7,592 |
| `2021.10.06.07.26.10_veh-52_01245_02064-24817ec41.json` | 8,383 |

Schema per file: `source`, `session_uuid`, `scene_id`, `random_seed`,
`n_drives`, `events[]` where each event is `start` / `ego` / `route` / `drive` /
`close`.

## 2. Origin

Written by `tests/extract_replay_streams.py` (added in `cd145f8`). That script is
a **reader, not a generator**: its whole data path is

```
async for m in async_read_pb_log(asl_path):      # line 56, SDK log reader
    ...
    "poses": [_pose_at(p) for p in r.trajectory.poses],        # line 75
    "dyn":   [_dyn(d) for d in r.dynamic_states],              # line 76
    "wps":   [[w.x, w.y, w.z] for w in r.route.waypoints],     # line 84
```

Field-by-field copies out of the protobuf. The module imports no RNG, no numpy,
and contains no synthesis, interpolation or curve-fitting of any kind.

Every fixture carries a `source` key naming the `rollout.asl` it came from, and
**all 10 source paths exist on disk**, spread across five real runs:

| Run directory | Fixtures | Corresponding job script |
|---|---|---|
| `runs/a1-537347` | 3 | `jobs/a1.sbatch` |
| `runs/follower-on-v4eval-540018` | 5 | `jobs/v4_vs_follower_gate.sbatch` |
| `runs/degen9-baseline-538542` | 1 | `jobs/degen9_validate.sbatch` |
| `runs/abstain-abstain-539923` | 1 | `jobs/abstention_ab.sbatch` |

These are the actual scored runs this project executed; each maps to a committed
sbatch script and a SLURM job id.

## 3. Evidence — five independent checks, all passing

### 3.1 Byte-exact re-derivation (the decisive one)

I independently re-read each cited `.asl` with the SDK reader and hashed the
result against the committed file. A synthesized fixture cannot survive this.

```
MATCH  2021.05.25.14.16.10_veh-35_01100_01664-f02b1
    committed sha256 f8da24ebe4f478b09c2e3cc21a9e0c3d
    re-derived sha256 f8da24ebe4f478b09c2e3cc21a9e0c3d
MATCH  2021.05.25.14.16.10_veh-35_01690_02183-0c77b
    committed sha256 ffc8ab76a58792ae14be6159855ab1ca
    re-derived sha256 ffc8ab76a58792ae14be6159855ab1ca
...
10 reproduced exactly, 0 did not (of 10 checked)
```

**10/10 reproduce byte-for-byte from the recorded logs.**

### 3.2 No fabricated or defaulted values

The extractor has exactly one defaulting path — `_dyn()` (line 44) returns
`[0.0, 0.0]` if `linear_velocity` is absent. The fixtures contain 13 velocity
entries that are exactly `[0,0]`, which could in principle be that default
firing rather than recorded data. **Re-derivation alone would not catch this**
(it would faithfully reproduce an injected default), so I checked the sources
directly:

```
sources scanned      : 10
drive msgs covered   : 100
dynamic_state entries: 110
  MISSING linear_velocity (would trigger extractor fallback): 0
  genuinely (0.0,0.0) in the recording                     : 13
```

**0 of 110.** The fallback never fired. All 13 zeros are genuine recorded values
(stationary frames), confirmed by `HasField('linear_velocity') == True` on each.
No value in any fixture was invented or defaulted.

### 3.3 Runtime-issued identifiers, not authored ones

| Field | Result |
|---|---|
| distinct `session_uuid` | 10 / 10 — RFC-4122 v1 UUIDs with real MAC/timestamp structure (e.g. `fbd6c64e-7f87-11f1-a333-1110625c0d4b`) |
| distinct `random_seed` | 10 / 10 — e.g. `909262145`, `15388760`, `44525507`; these come from `DriveSessionRequest.random_seed`, issued by the runtime |
| distinct `scene_id` | 10 / 10 |

The `session_uuid` values also match the directory names in the source paths,
which is consistent with the runtime having created them.

### 3.4 Scene IDs are real nuPlan scenes

Cross-checked against the staged dataset, independent of the logs:

```
10/10 scene_ids exist as real asset dirs under
/datasets/eemcs/ps/alpasim-nuplan-2026/navtest/assets
```

### 3.5 Coordinates are not synthetic-looking

Ego x-coordinates span `-32.0 .. 22.1` in local frame with irregular per-frame
deltas, and velocities like `(0.0, -12.5)` in rig frame. The synthetic profiles
in the same test file, by contrast, start at the origin and advance on a smooth
`v·dt·cos(yaw)` schedule. The two are not confusable.

## 4. Commit trail

| Commit | Role |
|---|---|
| `cd145f8` | Adds `tests/extract_replay_streams.py` (the reader) + the 5 replay tests + report §8. **Fixture files not included** — `.gitignore`'s `data/` pattern silently excluded them. |
| `07af779` | Adds the 10 fixture files + narrows `data/` → `/data/`. Message explains the exclusion and the fix. |

The two-commit split is explained by the `.gitignore` bug and is consistent with
the file contents. No commit rewrites or backdates the fixtures.

## 5. What "REAL" does and does not mean here

**Does mean:** the values are byte-exact extractions of messages the driver
genuinely received during real scored runs on real nuPlan scenes. No
fabrication, no defaulting, no synthesis.

**Does not mean:**

1. **They are not the original artifacts.** They are a derived cache. Camera
   bytes are dropped (the route-follower family ignores cameras) and extraction
   stops at 10 Drive calls per rollout, so each fixture is a *prefix* of its
   rollout, not the whole thing.
2. **Reproducibility depends on `~/alpasim/runs/` surviving.** That path is
   outside the repo and is not backed up. If those run directories are cleaned,
   the fixtures remain valid but can no longer be re-derived, and provenance
   would degrade to UNVERIFIABLE for anyone auditing later. **Recommendation
   (not applied): record the source `.asl` sha256 in each fixture**, so
   provenance stays checkable after the runs are deleted.
3. **The recordings came from route-follower-family drivers.** No fixture
   exercises a driver that brakes hard or abstains. The audit report already
   states this limit.

## 6. Bearing on `state_isolation_audit.md`

**No sentence requires walking back.** For completeness, these are the claims
this check was asked to validate, each confirmed:

- Line 19–22: *"**The claim now holds on REAL streams, not only synthetic
  ones.** §8 replays the actual RPC messages recorded in `rollout.asl` logs from
  scored runs — 10 scenes across 4 cities, including `dd162898`, the
  route-absent scene that produced the rear-end collision this project already
  diagnosed. Bit-identical there too."* — **CONFIRMED.** `dd162898` traces to
  `runs/abstain-abstain-539923`, a real run, and re-derives byte-exactly.
- Line 65 (residual-risk row): *"**Closed** by the replay path (§8). Real
  recorded streams from 10 scenes now run through the same bit-identity
  check."* — **CONFIRMED.**
- Line 372–373: *"It proves the isolation claim holds on real nuPlan route
  geometry, real ego motion and the real Drive cadence — not just on my
  synthesis of them."* — **CONFIRMED**, with the prefix caveat in §5.1 above:
  it is the first 10 Drive calls of each rollout, not the full 200-step episode.
  The sentence does not claim otherwise, but a reader could infer full episodes.
  **Optional tightening (not applied):** add "(first 10 Drive calls of each
  rollout)".
- Line 304: *"`rollout.asl` logs record the actual messages the driver received
  during scored runs"* — **CONFIRMED** against `egodriver.proto` and the logs.

The report's own stated limits (10 of 12,146 scenes; all logs from
route-follower runs) are accurate and already present.
