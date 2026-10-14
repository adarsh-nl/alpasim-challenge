#!/usr/bin/env python
"""Offline probe: does the monocular clearance heuristic separate a collision
scene from a clear scene on gsplat renders?  No sim, no GPU, no driver wiring.

Usage (run from ~/alpasim/src/eval so the eval package imports):
    uv run python probe_perception.py

It reads the CAM_F0 frames straight out of two saved rollout .asl logs:
  * COLLISION: the rear-end scene (obstacle clearly ahead, clearance should drop)
  * CLEAR:     a 1.0-progress scene (road clear, clearance should stay high)

If the two clearance curves separate, the heuristic is viable -> wire it in.
If they overlap, the heuristic is blind on renders -> switch to a depth model.
"""
from __future__ import annotations

import glob
import io
import math
import sys

import numpy as np

try:
    from PIL import Image
except Exception:
    print("PIL missing: pip install pillow (uv add pillow)")
    sys.exit(1)

from eval.asl_loader import async_read_pb_log
import asyncio

# ---- heuristic (mirrors perception.py, inlined so this file is standalone) ----
FS_ROW_FRAC, FS_CENTER_FRAC, FS_EDGE_THRESH = 0.55, 0.40, 0.18
FS_FOCAL_PX, FS_OBST_HEIGHT_M = 1000.0, 1.5
D_INF = 1e6


def decode_gray(image_bytes, downscale=8):
    if not image_bytes:
        return None
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("L")
        if downscale > 1:
            im = im.resize((im.width // downscale, im.height // downscale))
        return np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None


def free_space_distance(gray):
    h, w = gray.shape
    r0 = int(h * (1.0 - FS_ROW_FRAC))
    c0 = int(w * (0.5 - FS_CENTER_FRAC / 2))
    c1 = int(w * (0.5 + FS_CENTER_FRAC / 2))
    band = gray[r0:h, c0:c1]
    if band.shape[0] < 4:
        return D_INF
    gvert = np.abs(np.diff(band, axis=0)).mean(axis=1)
    obstacle_row = None
    for k in range(len(gvert) - 1, -1, -1):
        if gvert[k] > FS_EDGE_THRESH:
            obstacle_row = r0 + k
            break
    if obstacle_row is None:
        return D_INF
    horizon = h * 0.5
    dy = max(1.0, obstacle_row - horizon)
    focal_eff = FS_FOCAL_PX * (h / 135.0)
    return float(max(0.0, (FS_OBST_HEIGHT_M * focal_eff) / dy))


def loom_area(gray):
    h, w = gray.shape
    roi = gray[int(h * 0.5):h, int(w * 0.3):int(w * 0.7)]
    med = float(np.median(roi))
    return float((roi < med * 0.85).mean())


async def frames_from(asl_path, cam="CAM_F0"):
    """Yield (ts_us, gray) for the chosen camera, in order."""
    out = []
    async for entry in async_read_pb_log(asl_path):
        names = [f.name for f, _ in entry.ListFields()]
        if "driver_camera_image" in names:
            ci = entry.driver_camera_image.camera_image
            if ci.logical_id == cam:
                g = decode_gray(ci.image_bytes)
                if g is not None:
                    out.append((ci.frame_start_us, g))
    out.sort(key=lambda x: x[0])
    return out


def analyse(frames, label):
    print(f"\n=== {label}  ({len(frames)} {  'CAM_F0'} frames) ===")
    print(f"{'t(s)':>6} {'free_m':>8} {'loom_area':>10} {'growth':>8}")
    prev_area = None
    t0 = frames[0][0] if frames else 0
    fs_vals = []
    for ts, g in frames:
        fs = free_space_distance(g)
        a = loom_area(g)
        growth = (a / prev_area) if (prev_area and prev_area > 1e-3) else float("nan")
        prev_area = a
        fs_disp = "inf" if fs >= D_INF else f"{fs:8.1f}"
        gd = "  -" if math.isnan(growth) else f"{growth:8.3f}"
        print(f"{(ts - t0)/1e6:6.2f} {fs_disp:>8} {a:10.3f} {gd}")
        fs_vals.append(fs if fs < D_INF else np.nan)
    finite = [v for v in fs_vals if not np.isnan(v)]
    if finite:
        print(f"  free_space: min={min(finite):.1f}m  "
              f"last={finite[-1]:.1f}m  n_finite={len(finite)}/{len(fs_vals)}")
    else:
        print("  free_space: never fired (all inf) -> heuristic sees no obstacle")


async def main():
    base = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
    coll = glob.glob(f"{base}/task-1/rollouts/*8cc8fe63227d5c07*/*/rollout.asl")
    clear = glob.glob(f"{base}/task-0/rollouts/*/*/rollout.asl")

    if not coll:
        print("collision asl not found"); return
    if not clear:
        print("clear asl not found"); return

    coll_frames = await frames_from(coll[0])
    clear_frames = await frames_from(clear[0])

    analyse(coll_frames, "COLLISION (rear-end veh-45)")
    analyse(clear_frames, "CLEAR (task-0 scene)")

    print("\n--- verdict guide ---")
    print("GOOD: collision free_space drops toward 0 near the end; clear stays inf/high.")
    print("BAD : both look the same -> heuristic is blind on renders -> depth model.")


if __name__ == "__main__":
    asyncio.run(main())
