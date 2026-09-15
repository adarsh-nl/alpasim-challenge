#!/usr/bin/env python3
"""D1 Task 4 — far-horizon feasibility audit of the teacher labels.

GT can only validate ~2 s of each 8 s label (see reports/d1_label_integrity.md),
so the far horizon is where a confidently-wrong teacher would hide — and a
distilled zero with it. This audits the FULL 8 s against privileged data the
teacher itself had access to:

  OFF-ROAD        every waypoint against the static road-area polygons from the
                  trajdata vector map. **Covers the full 8 s** — the map does not
                  expire, which makes this the only check with complete horizon
                  coverage, and therefore the most valuable one here.

  COLLISION       2-D oriented-box overlap against logged neighbours at the
                  matching future timestep. Coverage is LIMITED: agent data ends
                  with the scene (56 timesteps), so from ts=20 only ~35 of 80
                  steps are checkable and from ts=55 none are. Reported with its
                  coverage, never extrapolated.

  KINEMATIC       speed, longitudinal accel, lateral accel (v^2*kappa), jerk and
                  reversing over the full 8 s. Complete coverage.

Non-reactivity caveat, stated because it cuts both ways: logged neighbours
replay their recorded paths and do not yield to the teacher's trajectory. A flagged
collision therefore means "this trajectory would have driven into a vehicle that
was actually there", which over-counts (a real driver might have been given room)
but under-counts nothing. Treat the rate as an upper bound on teacher
recklessness, not a score.

CPU-only, read-only. Usage:
    ~/alpasim/.venv/bin/python rl/audit_far_horizon.py --labels rl/teacher_labels \\
        [--limit 800] [--json reports/d1_far_horizon.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter
from functools import lru_cache

import numpy as np

DT = 0.1
EGO_L, EGO_W = 5.0, 2.0          # nuPlan ego footprint (m), approximate
A_LON_MAX = 6.0                  # plausibility thresholds, generous on purpose
A_LAT_MAX = 8.0
JERK_MAX = 15.0
V_MAX = 30.0
ROOT = os.environ.get("ALPASIM_NUPLAN_ROOT", "")
if not ROOT or not os.path.isdir(os.path.join(ROOT, "nuplan_test")):
    ROOT = "/datasets/eemcs/ps/alpasim-nuplan-2026"


@lru_cache(maxsize=8)
def get_map(location: str):
    from pathlib import Path
    from trajdata import MapAPI
    # incl_road_areas=True is REQUIRED: without it every road-area query raises
    # "Road areas are not loaded", which an except-and-continue handler turns
    # into a silent 0% off-road rate. That bug produced a meaningless clean
    # result on the first run of this audit.
    return MapAPI(Path(ROOT)).get_map(f"nuplan_test:{location}",
                                      incl_road_areas=True)


@lru_cache(maxsize=8)
def _drivable_index(location: str):
    """STRtree over a city's road-area polygons, for fast point containment.

    `get_road_areas_within(p, dist)` is a PROXIMITY query, not containment
    (dist=20 m returns 6 areas for an on-lane point, dist=200 m returns 55), so
    containment needs the polygon plus a point-in-polygon test. Building the
    index once per city turns an O(areas) scan per waypoint into a tree lookup.
    """
    from shapely.geometry import Polygon  # noqa: F401
    from shapely.strtree import STRtree
    vm = get_map(location)
    polys = []
    # element type 2 holds the road areas; iterate ids via the proximity query
    # over the whole map extent so we do not depend on private element layout.
    ext = vm.extent  # (min_x, min_y, min_z, max_x, max_y, max_z)
    cx, cy = (ext[0] + ext[3]) / 2.0, (ext[1] + ext[4]) / 2.0
    span = max(ext[3] - ext[0], ext[4] - ext[1])
    seen = set()
    for a in vm.get_road_areas_within(np.array([cx, cy, 0.0]), dist=span):
        if a.id in seen:
            continue
        seen.add(a.id)
        try:
            polys.append(vm.get_road_area_polygon_2d(a.id))
        except Exception:
            pass
    if not polys:
        raise RuntimeError(
            f"{location}: no road-area polygons recovered -- off-road check "
            f"cannot run and must not be reported as 0%")
    return STRtree(polys), polys


def is_on_road(location: str, x: float, y: float) -> bool:
    from shapely.geometry import Point
    tree, polys = _drivable_index(location)
    pt = Point(x, y)
    for i in tree.query(pt):
        if polys[i].contains(pt):
            return True
    return False


@lru_cache(maxsize=256)
def scene_frames(scene: str):
    """(ego Tx3 [x,y,heading], agents dict ts -> list[(x,y,heading,l,w)])."""
    import dill
    import pandas as pd
    d = os.path.join(ROOT, "nuplan_test", scene)
    md = dill.load(open(os.path.join(d, "scene_metadata_dt0.10.dill"), "rb"))
    df = pd.read_feather(os.path.join(d, "agent_data_dt0.10.feather"))
    ego = df[df["agent_id"] == "ego"].sort_values("scene_ts")
    ego_arr = ego[["x", "y", "heading"]].values.astype(np.float64)
    # agent extents come from scene metadata, not the per-step table
    ext = {}
    for a in getattr(md, "agents", []):
        e = getattr(a, "extent", None)
        ext[str(getattr(a, "name", ""))] = (
            float(getattr(e, "length", 4.5)), float(getattr(e, "width", 2.0)))
    others = df[df["agent_id"] != "ego"]
    by_ts: dict[int, list] = {}
    for ts, g in others.groupby("scene_ts"):
        rows = []
        for aid, x, y, h in zip(g["agent_id"], g["x"], g["y"], g["heading"]):
            l, w = ext.get(str(aid), (4.5, 2.0))
            rows.append((float(x), float(y), float(h), l, w))
        by_ts[int(ts)] = rows
    return ego_arr, by_ts, str(md.location)


def ego_to_world(pred_xy: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Inverse of trajdata_to_dp.se2_world_to_ego."""
    c, s = math.cos(anchor[2]), math.sin(anchor[2])
    rot = np.array([[c, -s], [s, c]])
    return pred_xy @ rot.T + anchor[:2]


def _corners(x, y, h, l, w):
    c, s = math.cos(h), math.sin(h)
    dx, dy = l / 2.0, w / 2.0
    loc = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    return loc @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def _boxes_overlap(a, b) -> bool:
    """Separating-axis test on two convex quads."""
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            ax = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(ax)
            if n < 1e-9:
                continue
            ax /= n
            pa, pb = a @ ax, b @ ax
            if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
                return False
    return True


def audit_one(token: str, ego_future: np.ndarray) -> dict:
    scene, _, ts_s = token.rpartition("_")
    ts = int(ts_s)
    ego_arr, agents_by_ts, location = scene_frames(scene)
    if ts >= len(ego_arr):
        return {"skip": "ts beyond scene"}
    anchor = ego_arr[ts]
    world = ego_to_world(ego_future[:, :2].astype(np.float64), anchor)
    H = len(world)

    # --- OFF-ROAD: full-horizon, static map -------------------------------
    # Any failure here propagates: a broken containment check must NOT be
    # reportable as "clean". See is_on_road / _drivable_index.
    off_steps = [k for k in range(H)
                 if not is_on_road(location, world[k][0], world[k][1])]
    first_off = off_steps[0] if off_steps else -1

    # --- COLLISION: only where logged agents exist ------------------------
    checkable = 0
    coll_steps = []
    for k in range(H):
        fut_ts = ts + 1 + k
        rows = agents_by_ts.get(fut_ts)
        if rows is None:
            continue
        checkable += 1
        # ego heading from the trajectory tangent
        if k + 1 < H:
            d = world[k + 1] - world[k]
        else:
            d = world[k] - world[k - 1]
        hh = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else anchor[2]
        ebox = _corners(world[k][0], world[k][1], hh, EGO_L, EGO_W)
        for (ax, ay, ah, al, aw) in rows:
            if abs(ax - world[k][0]) > 15 or abs(ay - world[k][1]) > 15:
                continue
            if _boxes_overlap(ebox, _corners(ax, ay, ah, al, aw)):
                coll_steps.append(k)
                break

    # --- KINEMATIC: full horizon -----------------------------------------
    seg = np.linalg.norm(np.diff(world, axis=0), axis=1)
    v = seg / DT
    a = np.diff(v) / DT if len(v) > 1 else np.array([0.0])
    j = np.diff(a) / DT if len(a) > 1 else np.array([0.0])
    # Lateral accel must be evaluated PER STEP as v[k]^2 * kappa[k]. Taking
    # max(v)^2 * max(kappa) combines a speed and a curvature from different
    # points and produced absurd values (1271 m/s^2) where near-stationary
    # waypoints made kappa explode. Curvature is also meaningless below a
    # minimum step length, so those steps are excluded rather than clipped.
    dx = np.gradient(world[:, 0]); dy = np.gradient(world[:, 1])
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx**2 + dy**2) ** 1.5
    kap = np.abs(dx * ddy - dy * ddx) / np.maximum(denom, 1e-6)
    MIN_STEP_M = 0.05                     # below this, heading is undefined
    v_at = np.concatenate([v, v[-1:]]) if len(v) else np.zeros(H)
    valid = np.concatenate([seg > MIN_STEP_M, [False]]) if len(seg) else np.zeros(H, bool)
    a_lat_steps = (v_at**2) * kap
    a_lat = float(a_lat_steps[valid].max()) if valid.any() else 0.0

    flags = []
    if not np.isfinite(ego_future).all():
        flags.append("NONFINITE")
    if v.max() > V_MAX:
        flags.append("V_MAX")
    if np.abs(a).max() > A_LON_MAX:
        flags.append("A_LON")
    if a_lat > A_LAT_MAX:
        flags.append("A_LAT")
    if np.abs(j).max() > JERK_MAX:
        flags.append("JERK")
    if ego_future[:, 0].min() < -1.0:
        flags.append("REVERSE")

    return {
        "token": token, "horizon": H,
        "off_road": bool(off_steps), "first_off_step": first_off,
        "n_off_steps": len(off_steps),
        "coll_checkable_steps": checkable,
        "collision": bool(coll_steps),
        "first_coll_step": coll_steps[0] if coll_steps else -1,
        "v_max": float(v.max()) if len(v) else 0.0,
        "a_lon_max": float(np.abs(a).max()),
        "a_lat_max": float(a_lat),
        "jerk_max": float(np.abs(j).max()),
        "flags": flags,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.labels, "*.npz")))
    if args.limit:
        step = max(1, len(files) // args.limit)
        files = files[::step][:args.limit]
    print(f"=== far-horizon audit: {len(files)} labels ===\n")

    rows, skipped = [], 0
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        try:
            r = audit_one(str(z["token"]), z["ego_future"])
        except Exception as e:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  skip {os.path.basename(f)[:40]}: {type(e).__name__}: {e}")
            continue
        if "skip" in r:
            skipped += 1
            continue
        rows.append(r)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(files)}", flush=True)

    if not rows:
        print("no labels audited", file=sys.stderr)
        return 1
    n = len(rows)

    off = [r for r in rows if r["off_road"]]
    cov = [r["coll_checkable_steps"] for r in rows]
    coll_rows = [r for r in rows if r["coll_checkable_steps"] > 0]
    coll = [r for r in coll_rows if r["collision"]]
    flagged = [r for r in rows if r["flags"]]

    print(f"\naudited {n} labels ({skipped} skipped)\n")
    print("1. OFF-ROAD  (full 8 s horizon, static map -- COMPLETE coverage)")
    print(f"   labels with any off-road waypoint : {len(off)}/{n}  ({len(off)/n:.2%})")
    if off:
        fo = [r["first_off_step"] for r in off]
        print(f"   first off-road step: median {statistics.median(fo):.0f} "
              f"({statistics.median(fo)*DT:.1f}s)  min {min(fo)}  max {max(fo)}")
        early = sum(1 for x in fo if x < 20)
        print(f"   of those, first leave within 2.0 s (GT-verifiable window): "
              f"{early}/{len(off)}  -> {len(off)-early} are FAR-HORIZON ONLY")
        ns = [r["n_off_steps"] for r in off]
        print(f"   off-road steps per bad label: median {statistics.median(ns):.0f} of 80")

    print("\n2. COLLISION with logged neighbours  (LIMITED coverage)")
    print(f"   checkable steps per label: median {statistics.median(cov):.0f} of 80 "
          f"(max {max(cov)})  -- agent data ends with the scene")
    print(f"   labels with >=1 checkable step    : {len(coll_rows)}/{n}")
    if coll_rows:
        print(f"   labels with a box overlap         : {len(coll)}/{len(coll_rows)} "
              f"({len(coll)/len(coll_rows):.2%})")
        if coll:
            fc = [r["first_coll_step"] for r in coll]
            print(f"   first overlap step: median {statistics.median(fc):.0f} "
                  f"({statistics.median(fc)*DT:.1f}s)")

    print("\n3. KINEMATIC FEASIBILITY  (full 8 s -- COMPLETE coverage)")
    print(f"   labels flagged : {len(flagged)}/{n}  ({len(flagged)/n:.2%})")
    for k, c in Counter(f for r in flagged for f in r["flags"]).most_common():
        print(f"     {c:5d}  {k}")
    for key, lim, unit in (("v_max", V_MAX, "m/s"), ("a_lon_max", A_LON_MAX, "m/s^2"),
                           ("a_lat_max", A_LAT_MAX, "m/s^2"), ("jerk_max", JERK_MAX, "m/s^3")):
        vals = sorted(r[key] for r in rows)
        print(f"   {key:10s} median {statistics.median(vals):7.2f}  "
              f"p99 {vals[int(.99*n)]:8.2f}  max {vals[-1]:9.2f}   [flag >{lim}] {unit}")

    bad = [r for r in rows if r["off_road"] or r["collision"] or r["flags"]]
    print(f"\n=== any-issue rate: {len(bad)}/{n} ({len(bad)/n:.2%}) ===")

    if args.json:
        json.dump({"n": n, "skipped": skipped,
                   "off_road_rate": len(off)/n,
                   "collision_rate_where_checkable":
                       (len(coll)/len(coll_rows) if coll_rows else None),
                   "coll_coverage_median_steps": statistics.median(cov),
                   "kinematic_flag_rate": len(flagged)/n,
                   "any_issue_rate": len(bad)/n,
                   "worst": sorted(bad, key=lambda r: -r["n_off_steps"])[:20]},
                  open(args.json, "w"), indent=2, default=float)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
