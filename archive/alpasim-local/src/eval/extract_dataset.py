#!/usr/bin/env python
"""Phase 0: extract (CAM_F0 image, route, ego-history, GT trajectory) from rollout .asl files.

Produces a compact per-frame training set for imitation learning. No GPU needed.

For each rollout, at each Drive timestep we capture:
  - CAM_F0 image (the forward camera, what the policy sees)  -> resized small to keep size sane
  - route waypoints (rig frame, the navigation goal)          -> the follower's key input
  - ego pose/speed history                                    -> state
  - GT future trajectory (the human demonstration)            -> the LABEL to imitate

Saves one .npz per scene under OUT_DIR, plus a manifest.
"""
from __future__ import annotations
import asyncio, glob, io, os, sys
import numpy as np
from PIL import Image

# NOTE: this imports the challenge's asl loader; run from ~/alpasim/src/eval
from eval.asl_loader import async_read_pb_log

ROLLOUT_GLOB = os.environ.get("ROLLOUT_GLOB",
    "/home/nanjaiyalathaa/alpasim/runs/array-538078/task-*/rollouts/*/*/rollout.asl")
OUT_DIR = os.environ.get("OUT_DIR", "/home/nanjaiyalathaa/alpasim/data/phase0")
IMG_HW = (180, 320)   # downsize CAM_F0 (from 1080x1920) to keep dataset compact; ~9x downscale

os.makedirs(OUT_DIR, exist_ok=True)

def _yaw(q):
    import math
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

async def extract_one(asl_path):
    frames_img = []      # list of (H,W,3) uint8
    frames_t   = []      # timestamps for each CAM_F0 frame
    route_rig  = None    # (N,3) most-recent route
    ego_poses  = []      # (t, x, y, yaw, speed)
    gt_traj    = None    # (M,3) GT future if present

    async for e in async_read_pb_log(asl_path):
        names = [f.name for f,_ in e.ListFields()]
        for n in names:
            # CAM_F0 image
            if 'image' in n.lower() or 'camera' in n.lower():
                cam = getattr(getattr(e, n), 'camera_image', None)
                if cam and getattr(cam,'logical_id','')=='CAM_F0' and cam.image_bytes:
                    im = Image.open(io.BytesIO(cam.image_bytes)).convert('RGB').resize((IMG_HW[1], IMG_HW[0]))
                    frames_img.append(np.asarray(im, dtype=np.uint8))
                    frames_t.append(int(cam.frame_start_us))
            # route
            if 'route' in n:
                r = getattr(e, n).route
                wps = np.array([[w.x,w.y,w.z] for w in r.waypoints], dtype=np.float32)
                if len(wps): route_rig = wps
            # egomotion
            if 'egomotion' in n or ('ego' in n.lower() and 'traj' in n.lower()):
                em = getattr(e, n)
                if hasattr(em,'trajectory') and em.trajectory.poses:
                    for p in em.trajectory.poses:
                        import math
                        sp = 0.0
                        if hasattr(em,'dynamic_states') and em.dynamic_states:
                            lv = em.dynamic_states[-1].linear_velocity
                            sp = float(math.hypot(lv.x, lv.y))
                        ego_poses.append((int(p.timestamp_us), float(p.pose.vec.x),
                                          float(p.pose.vec.y), _yaw(p.pose.quat), sp))
            # ground truth
            if 'ground_truth' in n:
                try:
                    tr = getattr(e, n).ground_truth.trajectory
                    if tr and tr.poses:
                        gt_traj = np.array([[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z]
                                            for p in tr.poses], dtype=np.float32)
                except Exception:
                    pass

    n_valid_route = 0 if route_rig is None else int(np.sum(~np.isnan(route_rig).any(axis=1)))
    return {
        'frames_img': np.array(frames_img) if frames_img else np.zeros((0,*IMG_HW,3),np.uint8),
        'frames_t':   np.array(frames_t, dtype=np.int64),
        'route_rig':  route_rig if route_rig is not None else np.zeros((0,3),np.float32),
        'n_valid_route': n_valid_route,
        'ego_poses':  np.array(ego_poses, dtype=np.float32) if ego_poses else np.zeros((0,5),np.float32),
        'gt_traj':    gt_traj if gt_traj is not None else np.zeros((0,3),np.float32),
    }

async def main():
    asls = sorted(glob.glob(ROLLOUT_GLOB))
    print(f"found {len(asls)} rollouts")
    manifest = []
    n_img_total = 0; n_gt = 0; n_degen = 0
    for i, a in enumerate(asls):
        sid = a.split('/rollouts/')[1].split('/')[0]
        try:
            d = await extract_one(a)
        except Exception as ex:
            print(f"  [{i}] FAIL {sid[:40]}: {ex}"); continue
        out = os.path.join(OUT_DIR, sid.split('-')[-1] + ".npz")
        np.savez_compressed(out, **d)
        nf = len(d['frames_img']); has_gt = len(d['gt_traj'])>0
        n_img_total += nf; n_gt += int(has_gt); n_degen += int(d['n_valid_route']<2)
        manifest.append((sid, nf, d['n_valid_route'], int(has_gt)))
        if i % 25 == 0:
            print(f"  [{i}/{len(asls)}] {sid.split('-')[-1]}: {nf} frames, "
                  f"{d['n_valid_route']} valid route wps, gt={has_gt}")
    # summary
    print("\n=== PHASE 0 SUMMARY ===")
    print(f"  scenes extracted:      {len(manifest)}")
    print(f"  total CAM_F0 frames:   {n_img_total}")
    print(f"  scenes WITH GT traj:   {n_gt} / {len(manifest)}")
    print(f"  degenerate-route scenes: {n_degen}")
    print(f"  output dir: {OUT_DIR}")
    # save manifest
    with open(os.path.join(OUT_DIR, "manifest.tsv"), "w") as f:
        f.write("scene\tn_frames\tn_valid_route\thas_gt\n")
        for sid, nf, nv, hg in manifest:
            f.write(f"{sid}\t{nf}\t{nv}\t{hg}\n")
    print(f"  manifest: {OUT_DIR}/manifest.tsv")

if __name__ == "__main__":
    asyncio.run(main())
