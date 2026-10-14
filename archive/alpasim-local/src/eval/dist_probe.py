#!/usr/bin/env python
"""Distance-covered probe: is the route-follower leaving safe distance on the table?

avg_dist_between_incidents_at_fault = distance_driven / at_fault_incidents.
We can't safely cut incidents (5 perception methods failed). So: are we driving
as much DISTANCE as we safely can on the scenes we already win?

For each scene, from the .asl:
  * ego_distance   = arclength of the ego's actual driven trajectory
  * route/gt hints = we also grab the driver_ego_trajectory span and, if present,
                     the recorded route length, to see headroom.

Prints per-scene ego distance + duration + mean speed, and aggregate stats. If ego
mean speed sits well under the V_MAX cap on many scenes, or ego stops early, there
is distance to reclaim. If ego already runs at cap for the full clip, the lever is
empty.

Run from ~/alpasim/src/eval:  uv run python dist_probe.py [limit]
"""
from __future__ import annotations
import glob, sys, asyncio, math
import numpy as np
from eval.asl_loader import async_read_pb_log


def traj_arclength(poses):
    pts = np.array([[p.pose.vec.x, p.pose.vec.y] for p in poses], dtype=float)
    if len(pts) < 2:
        return 0.0, 0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(seg.sum()), len(pts)


async def scene_stats(asl):
    ego_poses = []          # from driver_ego_trajectory (the estimated ego path)
    actor_frames = 0
    t_span = [None, None]
    async for e in async_read_pb_log(asl):
        names = [f.name for f, _ in e.ListFields()]
        if "driver_ego_trajectory" in names:
            tr = e.driver_ego_trajectory.trajectory
            for p in tr.poses:
                ego_poses.append(p)
                ts = p.timestamp_us
                if t_span[0] is None or ts < t_span[0]:
                    t_span[0] = ts
                if t_span[1] is None or ts > t_span[1]:
                    t_span[1] = ts
    # dedup poses by timestamp, sort
    seen = {}
    for p in ego_poses:
        seen[p.timestamp_us] = p
    ordered = [seen[t] for t in sorted(seen)]
    dist, n = traj_arclength(ordered)
    dur = (t_span[1] - t_span[0]) / 1e6 if t_span[0] is not None and t_span[1] is not None else 0.0
    mean_v = dist / dur if dur > 1e-3 else 0.0
    return dist, dur, mean_v, n


async def main():
    base = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
    asls = sorted(glob.glob(f"{base}/task-*/rollouts/*/*/rollout.asl"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(asls)
    asls = asls[:limit]
    print(f"scanning {len(asls)} rollouts for ego distance...\n", flush=True)

    rows = []
    for i, asl in enumerate(asls):
        sid = asl.split("/rollouts/")[1].split("/")[0]
        short = sid.split("-")[-1] if "-" in sid else sid
        dist, dur, mean_v, n = await scene_stats(asl)
        rows.append((dist, dur, mean_v, n, short))
        if (i + 1) % 40 == 0:
            print(f"  ...{i+1}/{len(asls)}", flush=True)

    dists = np.array([r[0] for r in rows])
    durs = np.array([r[1] for r in rows])
    vs = np.array([r[2] for r in rows])

    print(f"\n{'dist_m':>8} {'dur_s':>7} {'mean_v':>7} {'npose':>6}  scene")
    for dist, dur, mean_v, n, short in sorted(rows)[:15]:
        print(f"{dist:8.1f} {dur:7.2f} {mean_v:7.2f} {n:6d}  {short}   (shortest)")
    print("  ...")
    for dist, dur, mean_v, n, short in sorted(rows, reverse=True)[:15]:
        print(f"{dist:8.1f} {dur:7.2f} {mean_v:7.2f} {n:6d}  {short}   (longest)")

    print(f"\n=== AGGREGATE (n={len(rows)}) ===")
    print(f"ego distance:  mean={dists.mean():.1f}m  median={np.median(dists):.1f}m  "
          f"min={dists.min():.1f}m  max={dists.max():.1f}m  total={dists.sum():.0f}m")
    print(f"duration:      mean={durs.mean():.1f}s  median={np.median(durs):.1f}s")
    print(f"mean speed:    mean={vs.mean():.2f}m/s  median={np.median(vs):.2f}m/s  "
          f"max={vs.max():.2f}m/s")
    print(f"\nV_MAX cap is 20 m/s. Scenes with mean_v << cap may have safe speed headroom.")
    slow = np.sum(vs < 8.0)
    print(f"scenes with mean_v < 8 m/s: {slow}/{len(rows)} "
          f"(candidates for safe speed-up IF not speed-limited by curvature/traffic)")


if __name__ == "__main__":
    asyncio.run(main())
