#!/usr/bin/env python
"""Sweep loom_area over MANY scenes to find where a brake threshold can safely sit.

For every rollout .asl under the run dir, compute per-frame loom_area on CAM_F0 and
report each scene's PEAK loom_area and max sustained growth. Then print the
distribution split by whether the scene collided (from score.py's known zeros) vs
clear. The gap between the two distributions is the usable threshold — or its
absence tells us the raw cue is unusable and we need normalization / depth.
"""
from __future__ import annotations
import glob, io, math, sys, asyncio
import numpy as np
from PIL import Image
from eval.asl_loader import async_read_pb_log

def decode_gray(b, downscale=8):
    if not b: return None
    try:
        im = Image.open(io.BytesIO(b)).convert("L")
        im = im.resize((im.width//downscale, im.height//downscale))
        return np.asarray(im, dtype=np.float32)/255.0
    except Exception:
        return None

def loom_area(g):
    h,w = g.shape
    roi = g[int(h*0.5):h, int(w*0.3):int(w*0.7)]
    med = float(np.median(roi))
    return float((roi < med*0.85).mean())

async def scene_stats(asl):
    frames=[]
    async for e in async_read_pb_log(asl):
        names=[f.name for f,_ in e.ListFields()]
        if "driver_camera_image" in names:
            ci=e.driver_camera_image.camera_image
            if ci.logical_id=="CAM_F0":
                g=decode_gray(ci.image_bytes)
                if g is not None: frames.append((ci.frame_start_us,g))
    frames.sort(key=lambda x:x[0])
    if not frames: return None
    areas=[loom_area(g) for _,g in frames]
    peak=max(areas)
    # brightness proxy = mean of lower-centre region on first frame
    h,w=frames[0][1].shape
    bright=float(frames[0][1][int(h*0.5):h,int(w*0.3):int(w*0.7)].mean())
    # max sustained growth: max of min(growth[i],growth[i+1]) over consecutive pairs
    grow=[areas[i]/areas[i-1] if areas[i-1]>1e-3 else 1.0 for i in range(1,len(areas))]
    sustained=max([min(grow[i],grow[i+1]) for i in range(len(grow)-1)], default=1.0)
    return peak, sustained, bright, len(frames)

async def main():
    base="/home/nanjaiyalathaa/alpasim/runs/array-538078"
    # known collisions from earlier analysis
    collision_ids={"8cc8fe63227d5c07","dd162898ecf351d4"}
    asls=sorted(glob.glob(f"{base}/task-*/rollouts/*/*/rollout.asl"))
    print(f"scanning {len(asls)} rollouts...\n")
    rows=[]
    for asl in asls:
        sid = asl.split("/rollouts/")[1].split("/")[0]
        short = sid.split("-")[-1] if "-" in sid else sid
        st = await scene_stats(asl)
        if st is None: continue
        peak,sus,bright,n = st
        is_coll = any(c in asl for c in collision_ids)
        rows.append((peak,sus,bright,n,is_coll,short))
    rows.sort(reverse=True)
    print(f"{'peak':>6} {'sustain':>8} {'bright':>7} {'n':>3} {'COLL':>5}  scene")
    for peak,sus,bright,n,is_coll,short in rows:
        tag="*COLL*" if is_coll else ""
        print(f"{peak:6.3f} {sus:8.3f} {bright:7.3f} {n:3d} {tag:>5}  {short}")
    coll=[r[0] for r in rows if r[4]]
    clear=[r[0] for r in rows if not r[4]]
    print(f"\ncollisions (n={len(coll)}): peak loom_area = {sorted(coll)}")
    if clear:
        cl=sorted(clear,reverse=True)
        print(f"clear (n={len(clear)}): peak loom_area top-10 = {[round(x,3) for x in cl[:10]]}")
        print(f"clear max={max(clear):.3f}  clear 90th pct={np.percentile(clear,90):.3f}  median={np.median(clear):.3f}")
    if coll and clear:
        print(f"\nGAP CHECK: min collision peak={min(coll):.3f} vs max clear peak={max(clear):.3f}")
        if min(coll) > max(clear):
            print(">>> CLEAN SEPARATION: threshold between them works.")
        else:
            n_above = sum(1 for c in clear if c >= min(coll))
            print(f">>> OVERLAP: {n_above} clear scenes >= lowest collision. Raw loom_area unusable as-is.")

if __name__=="__main__":
    asyncio.run(main())
