# D2 — camera observation pipeline: design, and three findings that change the plan

**Four-axis (the pipeline itself):** (a) zero-rate 0 · (b) capability — *enables*
the student, which holds all remaining upside · (c) latency 0, this is offline
tooling and never runs on the Drive path · (d) VRAM 0.
**Smoke:** `~/alpasim/.venv/bin/python -m pytest tests/test_obs_pipeline.py -q`
→ **13 passed** (7 s, CPU, read-only).
**Gate state:** resolver built and tested; render pass **not** staged pending the
decisions below. No large scaffold committed, per instruction.

---

## Finding 1 — the fork does not exist as posed: there are no real frames

The charter asks me to surface the real-frames-vs-MTGS fork and default to real
frames. **Real frames are not available**, so that default is not selectable.

Every camera entry carries a `data_path` to a raw nuPlan JPEG:

```
frame_info['cams']['CAM_F0']['data_path']
  = '2021.05.25.14.16.10_veh-35_00083_00485/CAM_F0/25074407c1e256e0.jpg'
```

**None of these exist.** Checked under the dataset root, under `navtest/`, and
under the asset dir — all absent (asserted by
`test_real_source_data_path_is_recorded_but_absent`, which fails loudly if a
real frame ever appears). What *is* staged per renderable scene:

| Item | Size | Use |
|---|---|---|
| `background/<scene>.ckpt` | ~415 MB | Gaussian splat — renderable |
| `road_height_map/*.npy` | ~1.9 MB | ground geometry |
| `video_scene_dict.pkl` | ~3.6 MB | 8-cam intrinsics/extrinsics/distortion, per-frame ego pose, **privileged `gt_boxes`** |

The only image file in an asset scene is a road-height-map preview PNG.

**Consequence, and it is good news:** observations can only come from rendering
the splats — the same MTGS path the evaluator uses — so **there is no domain gap
to fine-tune away.** The planned "renderer fine-tune" is unnecessary because we
were never going to train on a different distribution.

The cost is that observation generation is **GPU-bound**, not CPU. Staging raw
nuPlan sensor data to get real frames would be a multi-TB decision *and* would
*introduce* the gap that rendering avoids. I have implemented `RealFrameSource`
as an explicit, explanatory error rather than deleting it, so the fork stays
visible in code instead of becoming a silent assumption.

**Recommendation: default to RENDERED.** This inverts the charter's default, on
evidence rather than preference.

## Finding 2 — 87% of teacher labels can never have an observation

This is a blocker for D1's remaining work, not just D2's.

```
nuplan_test  (trajdata, teacher-labelable) : 12,147 scenes
navtest/assets (splats, renderable)        :  1,491 scenes   (12.3%)

shard 0 labels                             :  6,080 samples / 760 scenes
  of which renderable                      :    800 samples / 100 scenes (13.2%)
  NOT renderable                           :  5,280 samples
```

The teacher was sharded over the **trajdata** scene set; observations only exist
for the **navtest splat** set; the two are largely disjoint.

**So running shards 1–15 as staged would produce ~90k labels of which ~88k can
never be paired with an observation.** That is the great majority of a ~12.7 h
GPU spend, wasted.

**Recommendation — re-target the teacher at the renderable set before spending
shards 1–15:**

| Plan | Scenes | Samples | Usable | GPU (1×L40S) |
|---|---|---|---|---|
| Current staged shards 1–15 | 12,147 | ~97k | ~12k (12%) | ~12.7 h |
| Re-targeted to splat scenes | 1,491 | ~11.9k | **~11.9k (100%)** | **~1.6 h** |

Shard 0 already covers 100 of the 1,491. This is a ~8× saving *and* a fully
usable dataset. I have **not** modified `rl/teacher_shards_1_15.sbatch` — the
re-targeting changes what the dataset *is*, which is your call, not a silent
edit. If you want it, the change is a scene-list filter in `run_teacher.py`
against `navtest/assets`, and I can stage it as a separate sbatch.

Caveat worth stating: ~11.9k samples is a small supervised set for a
camera-to-trajectory model. If that proves too small, the options are more
samples per scene (currently 8, at ts 20–55 only — the back 3.5 s of each 5.6 s
scene, so there is room) rather than more scenes.

## Finding 3 — decode, not the network, is the binding cost

The charter asked me to four-axis the input choice against the ~35 ms aggregate
budget because decode/resize is on the hot path. Measured on CPU at the eval
spec's 1920×1080:

```
JPEG size: 1531 KiB @ 1920x1080 q85
  decode only            :   38.40 ms
  decode+resize+float    :   59.52 ms   <- what rl/policy_driver.py does today
  draft-decode+resize    :   27.84 ms   <- JPEG DCT-scaling shortcut

  naive : 1 cam  59.52 ms | 8 cams  476.14 ms   vs ~35 ms budget -> OVER
  draft : 1 cam  27.84 ms | 8 cams  222.75 ms   vs ~35 ms budget -> OVER
```

**A single forward camera's decode consumes the entire ~35 ms per-call allowance
before any network runs.** Eight cameras is 6–14× over.

**Honest caveat on these numbers.** I measured against synthetic
maximum-entropy imagery, which is the *worst case* for JPEG — 1531 KiB is far
larger than a real 1080p render (expect ~150–400 KiB). Real decode will be
faster, plausibly by 2–3×. But the ordering does not change: at even 1/3 the
cost, one camera is ~10 ms and eight are ~75 ms, and decode still dominates a
small student's forward. **Re-measure against an actual render before fixing any
budget**, and treat the absolute numbers as an upper bound.

### Four-axis on the input choice

| Choice | (a) zero-rate | (b) capability | (c) latency | (d) VRAM |
|---|---|---|---|---|
| 8 cameras | ↓↓ if it worked | best — full surround | **fatal: 223–476 ms/call** | ~8× activations |
| **CAM_F0 only (default)** | ↓ targets forward collisions — *both* known zeros are `collision_front` | good — forward yielding is the missing capability | 28–60 ms, still over; needs mitigation | smallest |
| CAM_F0 + one rear | ~0 extra — rear-end is not at fault, so rear vision buys no score | low | ~2× forward-only | 2× |

**`DEFAULT_CAMS = ("CAM_F0",)`**, asserted by a test so it cannot drift silently.
The justification is not just budget: R4 showed **both** of our zeros are
at-fault *front* collisions, and rear collisions are explicitly free, so rear and
side cameras buy little score for a large latency bill.

### Mitigations, cheapest first

1. **`draft()` DCT-scaled decode** — ~2× cheaper, one line, no accuracy cost at
   180×320 since we downscale anyway. Already the default in
   `RenderedFrameSource`. Free win, and it applies to `policy_driver.py` too,
   which currently uses the naive path.
2. **Decode at dual rate.** Amortisation is legitimate per the charter, and it
   applies to *perception*, not only planning: decode every K-th Drive call and
   reuse the latent between. At K=4 a 28 ms decode amortises to 7 ms/call. This
   is the single biggest lever and it costs only staleness, which matters little
   at 2 Hz control.
3. **GPU decode (nvJPEG via torch/DALI).** GPU is provided at eval and we have
   16 GiB unused. Moves decode off the CPU critical path entirely. Worth
   benchmarking once a real render exists.
4. **Ask the evaluator for smaller frames** — not available; the render
   resolution is fixed by the eval config.

**Conclusion for D2: dual-rate is mandatory, and it was already the charter's
default plan — but for the wrong reason.** The driver is not
forward-pass-bound, it is *decode*-bound, and the amortisation has to cover
decode. A sub-35 ms single-rate design is not merely unproven; on these numbers
it is out of reach for any camera input at full resolution without GPU decode.

## The pipeline as built

`rl/obs_pipeline.py` — CPU-only, read-only, 13 tests green.

```
resolve(token) -> ObservationSpec
    token          label token, e.g. <scene>_020
    scene          trajdata scene name (== asset dir name)
    label_ts       trajdata timestep (dt=0.10)
    frame_idx      matched video frame
    align_residual_m
    splat_ckpt     path to the Gaussian splat
    cams{}         per-camera intrinsic / distortion / sensor2ego
    gt_boxes       privileged, opt-in only (never fed to the student)
```

### Alignment — derived, not documented anywhere

Label `ts` is a trajdata timestep; `frame_infos` are video frames over a
**different and longer span** (56 trajdata steps vs 115 video frames), and the
trajdata metadata carries **no absolute timestamp**, so the two cannot be joined
on time. They *can* be joined on ego position once lifted into a common frame:

```
global = frame_info['ego2global_translation'] + global2world_translation
```

Nearest-neighbour match against the trajdata ego track gives, on a real scene:

```
residual (m): median 0.091   p90 0.169   max 0.178     monotonic in ts: True
label ts 0..55 -> frame_idx 18..73
```

Sub-20 cm residuals confirm both are in the same global frame. The offset is
per-scene (18 here), so the implementation matches by pose rather than assuming
a constant. Tests assert residual < 1 m, monotonicity, and that distinct
timesteps do not collapse onto one frame.

### Frame sources — the swappable half

| Source | Status |
|---|---|
| `SyntheticFrameSource` | working; deterministic per (token, cam); **tests only** |
| `RenderedFrameSource` | working reader over a pre-rendered cache; `draft()` decode on; render pass itself is GPU and unstaged |
| `RealFrameSource` | deliberate loud failure with an explanation — keeps the fork visible |

## What I did not do, and what I need from you

Not built, pending your decisions:

1. **The render pass sbatch.** It depends on Finding 2 — rendering 1,491 scenes ×
   8 frames × 1 camera is a very different job from 12,147 scenes, and I should
   not stage a large GPU job against a scene set we may re-target.
2. **The student scaffold.** Its input stage depends on Finding 3 (decode rate,
   GPU vs CPU decode), and its loss on D1's horizon question. Building it first
   would bake in assumptions I have just shown to be wrong.

Three decisions, in priority order:

- **Re-target the teacher to the 1,491 renderable scenes?** (saves ~11 h GPU,
  makes the dataset 100% usable; my recommendation: yes)
- **Confirm RENDERED as the observation source?** (my recommendation: yes — no
  domain gap, and REAL is not available)
- **Accept CAM_F0-only + dual-rate decode as the input design?** (my
  recommendation: yes, with decode re-measured on a real render before the
  budget is fixed)

Also worth noting for the technical report: "we discovered the binding constraint
was JPEG decode, not inference" is a genuinely useful negative result, and it is
cheap to ablate (decode rate K is one knob).
