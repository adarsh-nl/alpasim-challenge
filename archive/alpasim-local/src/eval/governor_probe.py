#!/usr/bin/env python
"""Corridor-closing governor: offline trigger-analysis (controller lever, step 1).

Prior probes asked "can a THRESHOLD separate collisions from clear?" and failed --
collisions look like safe scenes at any instant. A GOVERNOR is different: it doesn't
classify, it yields PROPORTIONALLY to how fast the forward corridor is closing. Mild
closing (safe following) -> tiny yield -> negligible trajectory drift. Severe closing
(the rear-end) -> strong yield. The graded response is what should avoid the mxr trap.

Signal, per scene, from CAM_F0 only (the always-present forward view):
  * YOLO+track the largest CENTERED threat (|cx-0.5|<0.33)
  * range proxy r_hat = 1 - (y2 / H)   (bottom-edge low in frame => near => small r_hat)
    (scale-free; we only use its time-derivative)
  * closing rate c = -d r_hat/dt  (positive when object approaching)
  * graded yield demand:
       yield = clip( GAIN * c * proximity , 0, 1 )
    where proximity = (1 - r_hat) emphasises near objects; yield in [0,1] is a
    fraction of max decel. yield==0 when not closing or corridor open.

Outputs:
  (A) per-scene PEAK yield and INTEGRATED yield (sum over frames ~ total slowdown).
  (B) collisions: does peak yield fire, and at what lead time before impact frame?
  (C) regression proxy: distribution of integrated-yield on the 198 clear scenes.
      A clean result = collisions get high peak yield early, while clear scenes get
      near-zero integrated yield (governor idle on open corridors).

This is a PRECONDITION test: it shows whether the governor fires at the right time and
stays quiet on safe scenes. It does NOT prove braking avoids the collision -- that
needs a sim run (the .asl are the blind policy's actual driven trajectories).

Run from ~/alpasim/src/eval:  uv run python governor_probe.py [limit]
"""
from __future__ import annotations
import glob, io, sys, asyncio
import numpy as np
from PIL import Image
from ultralytics import YOLO
from eval.asl_loader import async_read_pb_log

yolo = YOLO("yolov8n.pt")
THREAT_CLASSES = {0, 1, 2, 3, 5, 7}
CENTER_MAX = 0.33
GAIN = 6.0                 # yield gain on closing rate (tunable)
FIRE_THRESH = 0.30         # "the governor meaningfully yields" level, for lead-time calc
COLLISION_IDS = {"8cc8fe63227d5c07", "dd162898ecf351d4"}


async def cam_f0_frames(asl):
    frames = []
    async for e in async_read_pb_log(asl):
        if "driver_camera_image" in [f.name for f, _ in e.ListFields()]:
            ci = e.driver_camera_image.camera_image
            if ci.logical_id == "CAM_F0":
                frames.append((ci.frame_start_us,
                               np.asarray(Image.open(io.BytesIO(ci.image_bytes)).convert("RGB"))))
    frames.sort(key=lambda x: x[0])
    return frames


def scene_yield_series(frames):
    """Return list of (t_s, yield_demand) over the scene."""
    t0 = frames[0][0]
    # first pass: largest centered-threat range-proxy per frame
    series = []     # (t, r_hat or None)
    for ts, rgb in frames:
        h, w = rgb.shape[:2]
        res = yolo.predict(rgb, verbose=False, conf=0.25, imgsz=640)[0]
        best_prox, best_rhat = -1.0, None
        if res.boxes is not None:
            for b in res.boxes:
                if int(b.cls.item()) not in THREAT_CLASSES:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cx = (x1 + x2) / 2 / w
                if abs(cx - 0.5) > CENTER_MAX:
                    continue
                r_hat = 1.0 - (y2 / h)          # small => near
                prox = 1.0 - r_hat              # large => near
                if prox > best_prox:
                    best_prox, best_rhat = prox, r_hat
        series.append(((ts - t0) / 1e6, best_rhat))

    # second pass: closing rate -> graded yield
    ys = []
    for i in range(len(series)):
        t, r = series[i]
        if r is None:
            ys.append((t, 0.0)); continue
        # look back to previous valid sample
        c = 0.0
        for j in range(i - 1, -1, -1):
            tp, rp = series[j]
            if rp is not None:
                dt = t - tp
                if dt > 1e-3:
                    c = -(r - rp) / dt          # closing = range-proxy decreasing
                break
        prox = 1.0 - r
        y = GAIN * max(c, 0.0) * max(prox, 0.0)
        ys.append((t, float(np.clip(y, 0.0, 1.0))))
    return ys


async def main():
    base = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
    asls = sorted(glob.glob(f"{base}/task-*/rollouts/*/*/rollout.asl"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(asls)
    asls = asls[:limit]
    print(f"governor trigger-analysis over {len(asls)} scenes (CAM_F0, CPU)...\n", flush=True)

    rows = []    # (peak_yield, integ_yield, fire_lead_s, is_coll, short)
    for i, asl in enumerate(asls):
        sid = asl.split("/rollouts/")[1].split("/")[0]
        short = sid.split("-")[-1] if "-" in sid else sid
        frames = await cam_f0_frames(asl)
        if not frames:
            continue
        ys = scene_yield_series(frames)
        yvals = [y for _, y in ys]
        peak = max(yvals) if yvals else 0.0
        integ = float(np.sum(yvals))     # ~ total slowdown demand over clip
        # lead time: first frame yield exceeds FIRE_THRESH, seconds before clip end
        t_end = ys[-1][0] if ys else 0.0
        fire_t = next((t for t, y in ys if y >= FIRE_THRESH), None)
        lead = (t_end - fire_t) if fire_t is not None else -1.0
        is_coll = any(c in asl for c in COLLISION_IDS)
        rows.append((peak, integ, lead, is_coll, short))
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(asls)}", flush=True)

    coll = [r for r in rows if r[3]]
    clear = [r for r in rows if not r[3]]

    print("\n=== (B) COLLISIONS: does the governor fire, and how early? ===")
    print(f"{'peak':>6} {'integ':>7} {'lead_s':>7}  scene")
    for peak, integ, lead, _, short in coll:
        ls = f"{lead:.2f}" if lead >= 0 else "never"
        print(f"{peak:6.3f} {integ:7.2f} {ls:>7}  {short}")

    print("\n=== (C) REGRESSION PROXY: integrated yield on 198 clear scenes ===")
    ci = sorted([r[1] for r in clear], reverse=True)
    cp = sorted([r[0] for r in clear], reverse=True)
    if clear:
        print(f"integrated yield: max={ci[0]:.2f}  90th={np.percentile([r[1] for r in clear],90):.2f}  "
              f"median={np.median([r[1] for r in clear]):.2f}")
        print(f"peak yield:       max={cp[0]:.3f}  90th={np.percentile([r[0] for r in clear],90):.3f}  "
              f"median={np.median([r[0] for r in clear]):.3f}")
        nz = sum(1 for r in clear if r[1] > 0.5)
        print(f"clear scenes with integrated yield > 0.5 (would slow noticeably): {nz}/{len(clear)}")

    # separation on PEAK yield (does the governor prefer collisions?)
    if coll and clear:
        print("\n=== SEPARATION on peak yield ===")
        cpk = [r[0] for r in coll]
        clpk = sorted([r[0] for r in clear], reverse=True)
        print(f"collision peak yields = {[round(x,3) for x in sorted(cpk)]}")
        print(f"clear top-10 peak     = {[round(x,3) for x in clpk[:10]]}")
        mc = min(cpk)
        n_above = sum(1 for r in clear if r[0] >= mc)
        print(f"clear scenes with peak >= lowest collision ({mc:.3f}): {n_above}/{len(clear)}")
        print("\nREAD: clean = collisions fire high & early (positive lead_s), while few clear")
        print("      scenes accumulate integrated yield. If collisions DON'T fire, governor")
        print("      can't see them either -> controller lever also closed for that collision.")


if __name__ == "__main__":
    asyncio.run(main())
