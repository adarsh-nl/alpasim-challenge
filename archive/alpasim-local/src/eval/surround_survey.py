#!/usr/bin/env python
"""Full surround-camera detection survey (Phase 2, step 1).

Runs YOLO on ALL 8 cameras across ALL scenes. Two outputs:

(A) RAW per-camera stats: how often each camera holds a threat, and the nearest
    (largest-box) threat it sees. Descriptive statistics for the report.

(B) SIDE->FORWARD MIGRATION signal: the hypothesis that forward-side cameras
    (CAM_R0, CAM_L0) see a crossing threat BEFORE CAM_F0. Per scene we compute an
    "early-warning score": the peak forward-corridor threat seen in R0/L0 at a time
    when CAM_F0 does NOT yet show a comparable threat. High early-warning on the
    crossing collision (and separation of collisions from clear) validates surround.

Cross-camera object identity is approximated by TIMING (side threat at t, forward
threat at t+dt) rather than true re-ID -- flagged as a heuristic.

Camera roles (challenge rig):
  forward:        CAM_F0
  forward-side:   CAM_R0, CAM_L0   (early warning for crossers -- USED)
  rear-side/rear: CAM_R1/R2/L1/L2/CAM_B0  (surveyed for raw stats, NOT for braking:
                  rear collisions are not at-fault, braking for them is harmful)

Run from ~/alpasim/src/eval:  uv run python surround_survey.py [limit]
"""
from __future__ import annotations
import glob, io, sys, asyncio
from collections import defaultdict
import numpy as np
from PIL import Image
from ultralytics import YOLO
from eval.asl_loader import async_read_pb_log

yolo = YOLO("yolov8n.pt")
THREAT_CLASSES = {0, 1, 2, 3, 5, 7}
CENTER_SIGMA = 0.30

ALL_CAMS = ["CAM_F0", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_B0"]
FORWARD = "CAM_F0"
FWD_SIDE = ["CAM_R0", "CAM_L0"]
COLLISION_IDS = {"8cc8fe63227d5c07", "dd162898ecf351d4"}


def frame_threat(rgb):
    """Return (peak_threat, nearest_box_area_frac) for threat classes in a frame."""
    h, w = rgb.shape[:2]
    res = yolo.predict(rgb, verbose=False, conf=0.25, imgsz=640)[0]
    best_threat, best_area = 0.0, 0.0
    if res.boxes is None:
        return 0.0, 0.0
    for b in res.boxes:
        if int(b.cls.item()) not in THREAT_CLASSES:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        area = (x2 - x1) * (y2 - y1) / (w * h)
        cx = (x1 + x2) / 2 / w
        cen = np.exp(-((cx - 0.5) ** 2) / (2 * CENTER_SIGMA ** 2))
        vert = y2 / h
        threat = area * cen * vert
        best_threat = max(best_threat, threat)
        best_area = max(best_area, area)
    return best_threat, best_area


async def scene_cam_frames(asl):
    """Return {cam: [(t_s, rgb), ...]} for all cameras."""
    byc = defaultdict(list)
    t0 = None
    async for e in async_read_pb_log(asl):
        if "driver_camera_image" in [f.name for f, _ in e.ListFields()]:
            ci = e.driver_camera_image.camera_image
            if ci.logical_id in ALL_CAMS:
                if t0 is None:
                    t0 = ci.frame_start_us
                rgb = np.asarray(Image.open(io.BytesIO(ci.image_bytes)).convert("RGB"))
                byc[ci.logical_id].append(((ci.frame_start_us - (t0 or 0)) / 1e6, rgb))
    for c in byc:
        byc[c].sort(key=lambda x: x[0])
    return byc


def survey_scene(byc):
    # (A) per-camera peak threat + how many frames each camera had a detection
    cam_peak = {}
    cam_hits = {}
    cam_series = {}     # cam -> list of (t, threat)
    for cam in ALL_CAMS:
        series = [(t, frame_threat(rgb)[0]) for t, rgb in byc.get(cam, [])]
        cam_series[cam] = series
        cam_peak[cam] = max((v for _, v in series), default=0.0)
        cam_hits[cam] = sum(1 for _, v in series if v > 0.02)

    # (B) early-warning: peak forward-side threat at a time when F0 threat is still low.
    f0 = dict(cam_series[FORWARD])       # t -> threat
    f0_times = sorted(f0)
    def f0_threat_at(t):
        # nearest F0 sample at or before t
        prior = [tt for tt in f0_times if tt <= t + 1e-6]
        return f0[prior[-1]] if prior else 0.0

    early_warning = 0.0
    for cam in FWD_SIDE:
        for t, thr in cam_series[cam]:
            if thr > 0.05 and f0_threat_at(t) < 0.05:    # side sees it, forward doesn't yet
                early_warning = max(early_warning, thr)

    return cam_peak, cam_hits, early_warning


async def main():
    base = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
    asls = sorted(glob.glob(f"{base}/task-*/rollouts/*/*/rollout.asl"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(asls)
    asls = asls[:limit]
    print(f"surround survey over {len(asls)} scenes x 8 cameras (CPU, slow)...\n", flush=True)

    peak_accum = defaultdict(list)
    hits_accum = defaultdict(list)
    ew_rows = []     # (early_warning, is_coll, short)
    for i, asl in enumerate(asls):
        sid = asl.split("/rollouts/")[1].split("/")[0]
        short = sid.split("-")[-1] if "-" in sid else sid
        byc = await scene_cam_frames(asl)
        if not byc.get(FORWARD):
            continue
        cam_peak, cam_hits, ew = survey_scene(byc)
        for c in ALL_CAMS:
            peak_accum[c].append(cam_peak[c])
            hits_accum[c].append(cam_hits[c])
        is_coll = any(c in asl for c in COLLISION_IDS)
        ew_rows.append((ew, is_coll, short))
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(asls)}", flush=True)

    # (A) per-camera raw stats
    print("\n=== (A) PER-CAMERA THREAT STATISTICS (mean over scenes) ===")
    print(f"{'camera':>8} {'mean_peak':>10} {'mean_hits':>10}  role")
    roles = {"CAM_F0":"forward (brake)", "CAM_R0":"fwd-side (early-warn)", "CAM_L0":"fwd-side (early-warn)",
             "CAM_R1":"rear-side (ignore)", "CAM_R2":"rear-side (ignore)",
             "CAM_L1":"rear-side (ignore)", "CAM_L2":"rear-side (ignore)", "CAM_B0":"rear (ignore)"}
    for c in ALL_CAMS:
        mp = np.mean(peak_accum[c]) if peak_accum[c] else 0
        mh = np.mean(hits_accum[c]) if hits_accum[c] else 0
        print(f"{c:>8} {mp:10.4f} {mh:10.2f}  {roles[c]}")

    # (B) early-warning separation
    ew_rows.sort(reverse=True)
    coll = [r[0] for r in ew_rows if r[1]]
    clear = [r[0] for r in ew_rows if not r[1]]
    print("\n=== (B) SIDE->FORWARD EARLY-WARNING SIGNAL ===")
    print(f"{'earlyW':>8} {'COLL':>6}  scene   (top 25)")
    for ew, is_coll, short in ew_rows[:25]:
        print(f"{ew:8.4f} {'*COLL*' if is_coll else '':>6}  {short}")
    print(f"\ncollisions (n={len(coll)}): early-warning = {[round(x,4) for x in sorted(coll)]}")
    if clear:
        cl = sorted(clear, reverse=True)
        print(f"clear (n={len(clear)}): top-10 = {[round(x,4) for x in cl[:10]]}")
        print(f"clear max={max(clear):.4f}  90th={np.percentile(clear,90):.4f}  median={np.median(clear):.4f}")
    if coll and clear:
        mc, xc = min(coll), max(clear)
        print(f"\nGAP CHECK: min collision early-warning={mc:.4f} vs max clear={xc:.4f}")
        if mc > xc:
            print(">>> Surround early-warning SEPARATES collisions from clear. Forward-only could not. Build BEV.")
        else:
            n_above = sum(1 for c in clear if c >= mc)
            print(f">>> {n_above} clear scenes >= lowest collision. Side cameras also see much irrelevant traffic.")
            print("    (Expected: raw side-detection isn't enough; the crossing case needs path-corridor")
            print("     geometry in BEV, not just 'a side camera saw something'. This motivates the lift.)")


if __name__ == "__main__":
    asyncio.run(main())
