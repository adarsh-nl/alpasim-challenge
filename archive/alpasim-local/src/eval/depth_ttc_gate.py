#!/usr/bin/env python
"""Depth+detection metric-TTC gate: the definitive monocular perception attempt.

Prior probes used bbox HEIGHT as a range proxy -- noisy, and it failed (collisions
had safer-looking TTC than clear scenes). Here we get RANGE from a depth model:

  * YOLO+ByteTrack -> WHERE each vehicle/person is (tracked box, persistent id)
  * Depth-Anything-V2 -> per-pixel depth; median depth inside the box = RANGE
  * track range(t) per id -> closing rate -> TTC = range / (-d range/dt)

Depth is relative-scale, but TTC = r/(dr/dt) is scale-invariant (const cancels),
so no metric calibration needed for the SEPARATION test.

Note: Depth-Anything returns LARGER values for NEARER points (inverse depth) in the
HF pipeline 'predicted_depth'. We normalise so that bigger = closer, and treat the
median-in-box as an inverse-range; closing = inverse-range INCREASING.

Gate: does depth-based TTC separate the 2 collisions from the 198 clear scenes,
where bbox-height TTC did not?  If yes -> build the depth governor. If no ->
collisions are likely irreducible from monocular vision; pivot.

Run from ~/alpasim/src/eval:  uv run python depth_ttc_gate.py [limit]
"""
from __future__ import annotations
import glob, io, sys, asyncio
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import pipeline
from eval.asl_loader import async_read_pb_log

yolo = YOLO("yolov8n.pt")
depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

THREAT_CLASSES = {0, 1, 2, 3, 5, 7}
MIN_TRACK_LEN = 3
CENTER_MAX_OFFSET = 0.30
D_INF = 1e9
DOWNSCALE_FOR_DEPTH = 4          # depth on a smaller image for speed


def depth_map(rgb):
    """Return inverse-depth map (bigger = nearer), same HxW as a downscaled rgb."""
    h, w = rgb.shape[:2]
    small = Image.fromarray(rgb).resize((w // DOWNSCALE_FOR_DEPTH, h // DOWNSCALE_FOR_DEPTH))
    out = depth(small)
    d = np.asarray(out["predicted_depth"][0] if hasattr(out["predicted_depth"], "__getitem__")
                   else out["predicted_depth"], dtype=np.float32)
    # HF returns a tensor; normalise robustly to [0,1], bigger=nearer (inverse depth)
    d = np.asarray(out["depth"], dtype=np.float32) if "depth" in out else d
    if d.max() > d.min():
        d = (d - d.min()) / (d.max() - d.min())
    return d                      # shape ~ (h/DS, w/DS)


async def scene_frames(asl):
    frames = []
    async for e in async_read_pb_log(asl):
        if "driver_camera_image" in [f.name for f, _ in e.ListFields()]:
            ci = e.driver_camera_image.camera_image
            if ci.logical_id == "CAM_F0":
                frames.append((ci.frame_start_us,
                               np.asarray(Image.open(io.BytesIO(ci.image_bytes)).convert("RGB"))))
    frames.sort(key=lambda x: x[0])
    return frames


def scene_min_ttc(frames) -> float:
    """Min TTC over sustained, centered, closing (range-decreasing) tracks."""
    tracks = defaultdict(list)     # id -> [(t, inv_range, cx)]
    t0 = frames[0][0]
    for ts, rgb in frames:
        h, w = rgb.shape[:2]
        res = yolo.track(rgb, persist=True, verbose=False, conf=0.25, imgsz=640,
                         tracker="bytetrack.yaml")[0]
        if res.boxes is None or res.boxes.id is None:
            continue
        dm = depth_map(rgb)         # inverse depth, downscaled
        dh, dw = dm.shape
        for b in res.boxes:
            cls = int(b.cls.item())
            if cls not in THREAT_CLASSES:
                continue
            tid = int(b.id.item()) if b.id is not None else -1
            if tid < 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cx = (x1 + x2) / 2 / w
            # sample inverse-depth median inside the box (in downscaled coords)
            bx1, by1 = int(x1 / w * dw), int(y1 / h * dh)
            bx2, by2 = int(x2 / w * dw), int(y2 / h * dh)
            bx1, bx2 = max(0, bx1), min(dw, max(bx1 + 1, bx2))
            by1, by2 = max(0, by1), min(dh, max(by1 + 1, by2))
            patch = dm[by1:by2, bx1:bx2]
            if patch.size == 0:
                continue
            inv_range = float(np.median(patch))   # bigger = nearer
            tracks[tid].append(((ts - t0) / 1e6, inv_range, cx))

    best = D_INF
    for tid, series in tracks.items():
        if len(series) < MIN_TRACK_LEN:
            continue
        if np.mean([abs(cx - 0.5) for _, _, cx in series]) > CENTER_MAX_OFFSET:
            continue
        for i in range(1, len(series)):
            t_p, ir_p, _ = series[i - 1]
            t_c, ir_c, _ = series[i]
            dt = t_c - t_p
            if dt <= 1e-3:
                continue
            dir_dt = (ir_c - ir_p) / dt
            if dir_dt > 1e-4 and ir_c > 1e-4:
                ttc = ir_c / dir_dt
                if ttc < best:
                    best = ttc
    return best


async def main():
    base = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
    collision_ids = {"8cc8fe63227d5c07", "dd162898ecf351d4"}
    asls = sorted(glob.glob(f"{base}/task-*/rollouts/*/*/rollout.asl"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(asls)
    asls = asls[:limit]
    print(f"scanning {len(asls)} rollouts with YOLO+ByteTrack+Depth-Anything (CPU, slow)...\n", flush=True)

    rows = []
    for i, asl in enumerate(asls):
        sid = asl.split("/rollouts/")[1].split("/")[0]
        short = sid.split("-")[-1] if "-" in sid else sid
        frames = await scene_frames(asl)
        if not frames:
            continue
        try:
            if hasattr(yolo, "predictor") and yolo.predictor is not None and hasattr(yolo.predictor, "trackers"):
                for tr in yolo.predictor.trackers:
                    tr.reset()
        except Exception:
            pass
        mttc = scene_min_ttc(frames)
        is_coll = any(c in asl for c in collision_ids)
        rows.append((mttc, is_coll, short))
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(asls)}", flush=True)

    rows.sort(key=lambda r: r[0])
    print(f"\n{'minTTC':>8} {'COLL':>6}  scene")
    for mttc, is_coll, short in rows[:50]:
        disp = "inf" if mttc > 1e8 else f"{mttc:8.2f}"
        print(f"{disp:>8} {'*COLL*' if is_coll else '':>6}  {short}")

    coll = [r[0] for r in rows if r[1]]
    clear = [r[0] for r in rows if not r[1]]
    fmt = lambda xs: [("inf" if x > 1e8 else round(x, 2)) for x in xs]
    print(f"\ncollisions (n={len(coll)}): min TTC = {fmt(sorted(coll))}")
    if clear:
        print(f"clear (n={len(clear)}): lowest-10 = {fmt(sorted(clear)[:10])}")
    if coll and clear:
        print("\nGAP CHECK (brake if minTTC < thr):")
        for thr in [1.5, 2.0, 2.5, 3.0, 3.5]:
            caught = sum(1 for c in coll if c < thr)
            fp = sum(1 for c in clear if c < thr)
            print(f"  thr={thr:.1f}s -> catches {caught}/{len(coll)} collisions, false-brakes {fp}/{len(clear)} clear")
        print("\n  WIN: a threshold catches both collisions with FP in low single digits.")
        print("  FAIL: collisions still buried among clear scenes -> monocular can't separate -> pivot.")


if __name__ == "__main__":
    asyncio.run(main())
