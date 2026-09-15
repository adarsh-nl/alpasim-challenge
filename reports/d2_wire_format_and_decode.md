# D2 verifications 1 & 2 — wire format and real-render decode

**Verification 2 flips the design.** Reporting before designing, per instruction.

**Four-axis:** measurement only — (a) 0 (b) 0 (c) 0 (d) 0. It exists to stop us
baking a wrong budget into the student.
**Gate state:** no design committed; tasks 3–5 not started pending this report.

---

## Verification 1 — images arrive JPEG-encoded; decode IS on our hot path

Traced the **sender**, not our driver's behaviour:

| Step | Evidence |
|---|---|
| Proto carries opaque bytes | `egodriver.proto:63-73` — `CameraImage { … bytes image_bytes = 4; string logical_id = 5; }`. **No format field**, so format is convention, not declared. |
| Runtime passes bytes through verbatim | `driver_service.py:101-110` — `submit_image()` copies `image.image_bytes` straight into the request. |
| The renderer encodes | `sensorsim_service.py:307-308` — `image_format=image_format, image_quality=95`. |
| Default format | `config.py:309` — `image_format: str = "jpeg"  # Literal["jpeg", "png"]` |
| Confirmed on the wire | Real `driver_camera_image` payloads from a scored run decode as **JPEG, 1920×1080, RGB, 274–333 KiB** |

**Result: JPEG q95, decode is ours.** So the amortisation question is real — but
see verification 2 for how much it actually costs.

Worth noting the format is *not* pinned in the egodriver proto. If the evaluator
ever switched to PNG, our decode cost would change sharply with no schema change
to warn us. A driver that detects the format from magic bytes rather than
assuming JPEG is marginally more robust; PIL already does this, so we get it free
as long as we never hand-roll a JPEG parser.

## Verification 2 — real renders are ~6× cheaper than my synthetic estimate

I previously measured decode on synthetic max-entropy JPEG and reported 27.8–59.5
ms/camera, concluding dual-rate decode was *mandatory*. **That conclusion was
wrong.** Measured on **real MTGS-rendered frames** pulled from a scored run's
`rollout.asl` (`driver_camera_image` payloads — the actual eval wire bytes):

```
12 real frames, 274-333 KiB, 1920x1080 JPEG

  naive decode+resize : median 37.87 ms  (min 26.19, max 38.76)   <- policy_driver today
  draft decode+resize : median  5.92 ms  (min  5.52, max  6.57)   <- 6.4x cheaper
```

Why my synthetic estimate was so far off: random noise produced a 1531 KiB frame
versus a real 308 KiB one (5× larger), and `draft()` DCT-scaled decoding helps
far more on real imagery (6.4× here vs 2.1× on noise) because renders have smooth,
compressible content. This is exactly why you asked for the real-render
re-measurement before fixing a budget.

### Revised budget picture

| Config | Decode/call | Leaves for forward (of ~35 ms) | Verdict |
|---|---|---|---|
| **CAM_F0, draft, single-rate** | **5.92 ms** | **~29 ms** | **fits comfortably** |
| CAM_F0, draft, dual-rate K=2 | 2.96 ms | ~32 ms | optional extra headroom |
| CAM_F0, naive (today's code) | 37.87 ms | **−3 ms** | over budget on its own |
| 8 cameras, draft | 47.4 ms | over | confirms CAM_F0-only |

### What this changes

1. **Dual-rate decode is NOT mandatory.** It is a lever to buy forward-pass
   headroom, not a requirement. Single-rate CAM_F0 fits with ~29 ms to spare.
   The approved "CAM_F0 + dual-rate" decision still works, but its *rationale*
   changes: dual-rate now buys network capacity, not feasibility.
2. **The binding constraint moves back to the forward pass.** ~29 ms/call, and
   the forward runs on the GPU we are currently not using at all. A ~2M-param CNN
   is single-digit ms there, so the student has real room — more than I implied.
3. **`draft()` is load-bearing, not a micro-optimisation.** 37.87 → 5.92 ms. It
   is also a **latent bug in `rl/policy_driver.py` today**: its naive decode is
   37.9 ms/call against a 35 ms allowance, so the RL driver would breach the
   aggregate budget on its own if it were ever the shipped path. Training-only
   today, so not urgent — but it should not be promoted without this one-line fix.
4. **CAM_F0-only still stands**, on two independent grounds: 8-cam draft decode
   is 47 ms (over), and R4 showed both known zeros are at-fault *front*
   collisions while rear collisions are free.

### Caveats

- 12 frames from one scene of one run. Sizes are tight (274–333 KiB) so the
  median is probably representative, but frame complexity varies across scenes;
  a denser scene will decode slower.
- Measured on a shared login node, single core, CPU-only. At eval the driver has
  a GPU and competes with 1 co-resident rollout; the R2 harness showed the
  process-wide lock roughly doubles effective per-call cost under 2-way
  concurrency, so budget ~2× these figures for the concurrent case
  (≈12 ms decode, still fine).
- `draft()` reduces decode precision by DCT-scaling before the resize. At a
  180×320 target that should be lossless-in-effect, but if the student is ever
  sensitive to fine texture, verify the accuracy delta rather than assuming it.

## Standing recommendation

Proceed with **CAM_F0 + `draft()` decode + single-rate**, and keep dual-rate as a
tuning knob behind a `K` parameter rather than an architectural commitment. Fix
`policy_driver.py`'s decode when convenient.
