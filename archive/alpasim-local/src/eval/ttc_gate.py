#!/usr/bin/env python
"""Full-200 TTC-tracking gate: does a min-TTC threshold separate collisions from
the 198 clear scenes?  Uses YOLO + ByteTrack persistent IDs so TTC is computed on
sustained tracks of the SAME object (the flicker that wrecked per-frame TTC is
filtered out).

Per scene:
  * track vehicles/persons across frames with model.track(persist=True)
  * for each track: series of (t, bbox_height, cx)
  * keep only SUSTAINED (>=3 frames), CENTERED (mean |cx-0.5| < 0.25) tracks
  * TTC per step = h / (dh/dt) when closing (dh>0); take that track's min TTC
  * scene min-TTC = min over qualifying tracks (inf if none closing)

Then split collisions vs clear and report the separation + a false-brake table
at candidate thresholds.

Run from ~/alpasim/src/eval:  uv run python ttc_gate.py [limit]
"""
from __future__ import annotations
import glob, io, sys, asyncio
from collections import defaultdict
import numpy as np
from PIL import Image
from ultralytics import YOLO
from eval.asl_loader import async_read_pb_log

m = YOLO("yolov8n.pt")
THREAT_CLASSES = {0, 1, 2, 3, 5, 7}
MIN_TRACK_LEN = 3
CENTER_MAX_OFFSET = 0.25       # mean |cx-0.5| must be under this to count as in-path
D_INF = 1e9


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
    """Min TTC over sustained, centered, closing tracks. inf if none."""
    # tracks[id] = list of (t_s, h_frac, cx)
    tracks = defaultdict(list)
    t0 = frames[0][0]
    # reset tracker state between scenes by creating detections with persist=True
    # (ultralytics keeps state on the model; we call track fresh per scene via a new stream)
    for ts, rgb in frames:
        h, w = rgb.shape[:2]
        res = m.track(rgb, persist=True, verbose=False, conf=0.25, imgsz=640,
                      tracker="bytetrack.yaml")[0]
        if res.boxes is None or res.boxes.id is None:
            continue
        for b in res.boxes:
            cls = int(b.cls.item())
            if cls not in THREAT_CLASSES:
                continue
            tid = int(b.id.item()) if b.id is not None else -1
            if tid < 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cx = (x1 + x2) / 2 / w
            bh = (y2 - y1) / h
            tracks[tid].append(((ts - t0) / 1e6, bh, cx))

    best = D_INF
    for tid, series in tracks.items():
        if len(series) < MIN_TRACK_LEN:
            continue
        mean_off = np.mean([abs(cx - 0.5) for _, _, cx in series])
        if mean_off > CENTER_MAX_OFFSET:
            continue                      # not in-path (off to the side / crossing)
        # walk the track, compute TTC where closing
        for i in range(1, len(series)):
            t_prev, h_prev, _ = series[i - 1]
            t_cur, h_cur, _ = series[i]
            dt = t_cur - t_prev
            if dt <= 1e-3:
                continue
            dh = (h_cur - h_prev) / dt
            if dh > 1e-4:
                ttc = h_cur / dh
                if ttc < best:
                    best = ttc
    return best


async def main():
    base = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
    collision_ids = {"8cc8fe63227d5c07", "dd162898ecf351d4"}
    asls = sorted(glob.glob(f"{base}/task-*/rollouts/*/*/rollout.asl"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(asls)
    asls = asls[:limit]
    print(f"scanning {len(asls)} rollouts with YOLO+ByteTrack...\n", flush=True)

    rows = []
    for i, asl in enumerate(asls):
        sid = asl.split("/rollouts/")[1].split("/")[0]
        short = sid.split("-")[-1] if "-" in sid else sid
        frames = await scene_frames(asl)
        if not frames:
            continue
        try:
            if hasattr(m, "predictor") and m.predictor is not None and hasattr(m.predictor, "trackers"):
                for tr in m.predictor.trackers:
                    tr.reset()
        except Exception:
            pass
        mttc = scene_min_ttc(frames)
        is_coll = any(c in asl for c in collision_ids)
        rows.append((mttc, is_coll, short))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(asls)}", flush=True)

    rows.sort(key=lambda r: r[0])
    print(f"\n{'minTTC':>8} {'COLL':>6}  scene")
    for mttc, is_coll, short in rows[:60]:
        disp = "inf" if mttc > 1e8 else f"{mttc:8.2f}"
        print(f"{disp:>8} {'*COLL*' if is_coll else '':>6}  {short}")

    coll = [r[0] for r in rows if r[1]]
    clear = [r[0] for r in rows if not r[1]]
    fmt = lambda xs: [("inf" if x > 1e8 else round(x, 2)) for x in xs]
    print(f"\ncollisions (n={len(coll)}): min TTC = {fmt(sorted(coll))}")
    if clear:
        cs = sorted(clear)
        print(f"clear (n={len(clear)}): lowest-10 min TTC = {fmt(cs[:10])}")
    if coll and clear:
        worst_coll = max(c for c in coll if c < 1e8) if any(c < 1e8 for c in coll) else None
        print(f"\nGAP CHECK (brake if minTTC < thr):")
        if worst_coll is None:
            print("  collisions never show a closing track -> TTC-tracking can't catch them.")
        else:
            for thr in [1.5, 2.0, 2.5, 3.0, 3.5]:
                caught = sum(1 for c in coll if c < thr)
                fp = sum(1 for c in clear if c < thr)
                print(f"  thr={thr:.1f}s -> catches {caught}/{len(coll)} collisions, "
                      f"false-brakes {fp}/{len(clear)} clear")
            print("\n  Ideal: a threshold catching both collisions with FP in low single digits.")


if __name__ == "__main__":
    asyncio.run(main())
