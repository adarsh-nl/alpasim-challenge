#!/usr/bin/env python
"""Range-over-time probe: does TTC separate a real collision from a safe car-ahead?

Static detection failed: a safely-followed car and a car-you'll-hit look identical
in one frame. The distinguishing signal is DYNAMICS -- is range collapsing?

For each scene, per frame, find the nearest in-path vehicle/person and use bbox
HEIGHT as an inverse-range proxy (taller box = closer). Track height over time:
  * growing height  => closing        => range collapsing => low TTC  => THREAT
  * stable height   => matched speed  => safe following   => high TTC => no brake

TTC estimate (pinhole, object of fixed real height H):
  range ∝ 1 / bbox_height_px          (call it r = K / h_px, K unknown const)
  closing_rate = -d(range)/dt
  TTC = range / closing_rate = h_px / (d h_px/dt)      (K cancels!)
So TTC in seconds needs NO calibration -- it's just height / height-growth-rate.

Run from ~/alpasim/src/eval:  uv run python ttc_probe.py
"""
from __future__ import annotations
import glob, io, asyncio
import numpy as np
from PIL import Image
from ultralytics import YOLO
from eval.asl_loader import async_read_pb_log

m = YOLO("yolov8n.pt")
THREAT_CLASSES = {0, 1, 2, 3, 5, 7}
CENTER_SIGMA = 0.30


def nearest_inpath_height(rgb):
    """Return (bbox_height_frac, cx, class) of the most in-path threat, or None."""
    h, w = rgb.shape[:2]
    r = m.predict(rgb, verbose=False, conf=0.25, imgsz=640)[0]
    best = None
    best_score = 0.0
    for b in (r.boxes or []):
        cls = int(b.cls.item())
        if cls not in THREAT_CLASSES:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        cx = (x1 + x2) / 2 / w
        bh = (y2 - y1) / h
        cen = np.exp(-((cx - 0.5) ** 2) / (2 * CENTER_SIGMA ** 2))
        # in-path score prioritises central + tall (near) boxes
        score = bh * cen
        if score > best_score:
            best_score = score
            best = (bh, round(cx, 2), m.names[cls])
    return best


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


async def analyse(label, sid):
    base = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
    asl = glob.glob(f"{base}/task-*/rollouts/*{sid}*/*/rollout.asl")
    if not asl:
        print(f"{label}: NOT FOUND"); return
    frames = await scene_frames(asl[0])
    print(f"\n=== {label} ({len(frames)} frames) ===")
    print(f"{'t(s)':>6} {'h_frac':>7} {'cx':>5} {'class':>7} {'dh/dt':>8} {'TTC(s)':>8}")
    t0 = frames[0][0]
    prev_h = prev_t = None
    min_ttc = 1e9
    for ts, rgb in frames:
        det = nearest_inpath_height(rgb)
        t = (ts - t0) / 1e6
        if det is None:
            print(f"{t:6.2f} {'--':>7} {'--':>5} {'none':>7} {'--':>8} {'--':>8}")
            prev_h = prev_t = None
            continue
        h_frac, cx, cls = det
        ttc_str = "  -"
        dh_str = "  -"
        if prev_h is not None and prev_t is not None:
            dt = (ts - prev_t) / 1e6
            if dt > 1e-3:
                dh = (h_frac - prev_h) / dt          # height growth per second
                dh_str = f"{dh:8.4f}"
                # TTC = h / (dh/dt), only meaningful when closing (dh>0)
                if dh > 1e-4:
                    ttc = h_frac / dh
                    ttc_str = f"{ttc:8.2f}"
                    min_ttc = min(min_ttc, ttc)
                else:
                    ttc_str = "    inf"
        print(f"{t:6.2f} {h_frac:7.3f} {cx:5.2f} {cls:>7} {dh_str:>8} {ttc_str:>8}")
        prev_h, prev_t = h_frac, ts
    mt = "inf" if min_ttc > 1e8 else f"{min_ttc:.2f}s"
    print(f"  --> min TTC = {mt}")


async def main():
    await analyse("FALSE_POS (safe car ahead)", "1786ef7e")
    await analyse("REAR-END collision", "dd162898")
    await analyse("CROSSING collision", "8cc8fe63")
    print("\n--- verdict ---")
    print("GOOD: collisions show min TTC small (few s, range collapsing);")
    print("      false-pos shows min TTC large/inf (range stable, safe follow).")
    print("BAD : all similar -> bbox-height TTC can't separate -> depth model for metric range.")


if __name__ == "__main__":
    asyncio.run(main())
