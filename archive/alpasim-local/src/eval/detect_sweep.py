#!/usr/bin/env python
"""Detector probe: does a vehicle detector separate the 2 collisions from the 198
clear scenes, where looming failed?  Offline, CPU-ok, no sim, no submission.

For every saved rollout, run YOLOv8n on each CAM_F0 frame, keep vehicle/person
detections, and compute a PATH-RELEVANCE THREAT per frame:

    threat = (bbox area fraction) * (horizontal centrality) * (verticality)

  * area fraction : bigger box  => closer                 (0..1)
  * centrality    : box centre near image centre-x        (0..1, in-lane vs edge)
  * verticality   : box bottom low in the frame           (0..1, near vs far)

Per scene we take the PEAK threat across frames. Then split collisions vs clear
and check for a usable gap. A clean gap => the detector is a viable brake trigger;
we wire it into the governor. Overlap => detector alone insufficient, iterate the
path-relevance definition or add corridor geometry.

Run from ~/alpasim/src/eval:
    uv run python detect_sweep.py
"""
from __future__ import annotations
import glob, io, sys, asyncio
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    print("ultralytics missing"); sys.exit(1)

from eval.asl_loader import async_read_pb_log

VEHICLE_CLASSES = {1, 2, 3, 5, 7}   # bicycle, car, motorcycle, bus, truck (COCO)
PERSON_CLASS = {0}
THREAT_CLASSES = VEHICLE_CLASSES | PERSON_CLASS

CENTER_SIGMA = 0.25   # how tightly "in-lane" is judged (frac of width)
CONF_MIN = 0.25

_model = None
def model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def decode_rgb(b):
    if not b: return None
    try:
        return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    except Exception:
        return None


def frame_threat(rgb) -> float:
    """Peak path-relevance threat over all threat-class detections in one frame."""
    h, w = rgb.shape[:2]
    res = model().predict(rgb, verbose=False, conf=CONF_MIN, imgsz=640)[0]
    best = 0.0
    if res.boxes is None:
        return 0.0
    for b in res.boxes:
        cls = int(b.cls.item())
        if cls not in THREAT_CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        bw, bh = (x2 - x1), (y2 - y1)
        area_frac = (bw * bh) / (w * h)                     # 0..1
        cx = (x1 + x2) / 2 / w                              # 0..1
        centrality = float(np.exp(-((cx - 0.5) ** 2) / (2 * CENTER_SIGMA ** 2)))
        verticality = (y2 / h)                              # bottom edge low => near
        threat = area_frac * centrality * verticality
        if threat > best:
            best = threat
    return best


async def scene_peak_threat(asl) -> float | None:
    frames = []
    async for e in async_read_pb_log(asl):
        names = [f.name for f, _ in e.ListFields()]
        if "driver_camera_image" in names:
            ci = e.driver_camera_image.camera_image
            if ci.logical_id == "CAM_F0":
                rgb = decode_rgb(ci.image_bytes)
                if rgb is not None:
                    frames.append((ci.frame_start_us, rgb))
    if not frames:
        return None
    frames.sort(key=lambda x: x[0])
    return max(frame_threat(rgb) for _, rgb in frames)


async def main():
    base = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
    collision_ids = {"8cc8fe63227d5c07", "dd162898ecf351d4"}
    asls = sorted(glob.glob(f"{base}/task-*/rollouts/*/*/rollout.asl"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(asls)
    asls = asls[:limit]
    print(f"scanning {len(asls)} rollouts with YOLOv8n (CPU ok)...\n")

    rows = []
    for i, asl in enumerate(asls):
        sid = asl.split("/rollouts/")[1].split("/")[0]
        short = sid.split("-")[-1] if "-" in sid else sid
        pk = await scene_peak_threat(asl)
        if pk is None:
            continue
        is_coll = any(c in asl for c in collision_ids)
        rows.append((pk, is_coll, short))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(asls)}")

    rows.sort(reverse=True)
    print(f"\n{'threat':>8} {'COLL':>6}  scene")
    for pk, is_coll, short in rows[:60]:
        print(f"{pk:8.4f} {'*COLL*' if is_coll else '':>6}  {short}")

    coll = [r[0] for r in rows if r[1]]
    clear = [r[0] for r in rows if not r[1]]
    print(f"\ncollisions (n={len(coll)}): peak threat = {[round(x,4) for x in sorted(coll)]}")
    if clear:
        cl = sorted(clear, reverse=True)
        print(f"clear (n={len(clear)}): top-10 = {[round(x,4) for x in cl[:10]]}")
        print(f"clear max={max(clear):.4f}  90th={np.percentile(clear,90):.4f}  median={np.median(clear):.4f}")
    if coll and clear:
        mc, xc = min(coll), max(clear)
        print(f"\nGAP CHECK: min collision threat={mc:.4f} vs max clear threat={xc:.4f}")
        if mc > xc:
            print(">>> CLEAN SEPARATION: a threat threshold catches both collisions, zero clear. BUILD IT.")
        else:
            n_above = sum(1 for c in clear if c >= mc)
            print(f">>> OVERLAP: {n_above} clear scenes >= lowest collision. Refine path-relevance / add corridor geometry.")
            for thr in sorted(coll):
                fp = sum(1 for c in clear if c >= thr)
                print(f"     threshold {thr:.4f} -> catches collisions>=it, false-brakes {fp}/{len(clear)} clear scenes")


if __name__ == "__main__":
    asyncio.run(main())
