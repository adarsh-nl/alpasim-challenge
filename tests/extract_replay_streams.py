#!/usr/bin/env python
"""Extract real egodriver RPC streams from recorded rollout.asl logs.

The .asl log records the ACTUAL messages the driver received during a scored
run -- driver_session_request, driver_ego_trajectory, route_request,
driver_request -- so replaying them exercises the driver on real nuPlan
geometry instead of synthesised routes.

This writes a small JSON per rollout (poses + routes + drive timestamps only;
camera bytes are dropped, the route-follower family ignores them). The replay
tests then run with no SDK dependency and no GPU.

Run ONCE, from the SDK env, on the login node (single-core file I/O):

    cd ~/alpasim
    ./.venv/bin/python ~/alpasim-challenge/tests/extract_replay_streams.py \
        --out ~/alpasim-challenge/tests/data/replay --limit 8 --max-drives 12

It is slow (~15-30 s per rollout: the reader parses embedded camera frames even
though we discard them), so it is deliberately a one-shot cache-builder, not
something the test suite runs.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
import time
from pathlib import Path

DEFAULT_GLOB = "/home/nanjaiyalathaa/alpasim/runs/*/rollouts/*/*/rollout.asl"
WANTED = ("driver_session_request", "driver_ego_trajectory",
          "route_request", "driver_request")


def _pose_at(p):
    return {"t": int(p.timestamp_us),
            "vec": [p.pose.vec.x, p.pose.vec.y, p.pose.vec.z],
            "quat": [p.pose.quat.w, p.pose.quat.x, p.pose.quat.y, p.pose.quat.z]}


def _dyn(d):
    lv = getattr(d, "linear_velocity", None)
    return [getattr(lv, "x", 0.0), getattr(lv, "y", 0.0)] if lv is not None else [0.0, 0.0]


async def extract_one(asl_path: str, max_drives: int) -> dict | None:
    from alpasim_utils.logs import async_read_pb_log

    events: list[dict] = []
    meta: dict = {"source": asl_path}
    n_drives = 0

    async for m in async_read_pb_log(asl_path):
        fields = {f.name for f, _ in m.ListFields()}

        if "rollout_metadata" in fields:
            sm = getattr(m.rollout_metadata, "session_metadata", None)
            if sm is not None:
                meta["session_uuid"] = getattr(sm, "session_uuid", "")
                meta["scene_id"] = getattr(sm, "scene_id", "")

        if "driver_session_request" in fields:
            r = m.driver_session_request
            meta.setdefault("session_uuid", getattr(r, "session_uuid", ""))
            meta["random_seed"] = int(getattr(r, "random_seed", 0) or 0)
            events.append({"kind": "start"})

        elif "driver_ego_trajectory" in fields:
            r = m.driver_ego_trajectory
            events.append({
                "kind": "ego",
                "poses": [_pose_at(p) for p in r.trajectory.poses],
                "dyn": [_dyn(d) for d in getattr(r, "dynamic_states", [])],
            })

        elif "route_request" in fields:
            r = m.route_request
            events.append({
                "kind": "route",
                "ts": int(getattr(r.route, "timestamp_us", 0) or 0),
                "wps": [[w.x, w.y, w.z] for w in r.route.waypoints],
            })

        elif "driver_request" in fields:
            r = m.driver_request
            events.append({
                "kind": "drive",
                "time_now_us": int(getattr(r, "time_now_us", 0) or 0),
                "time_query_us": int(getattr(r, "time_query_us", 0) or 0),
            })
            n_drives += 1
            if n_drives >= max_drives:
                break

    if n_drives == 0:
        return None
    events.append({"kind": "close"})
    meta["events"] = events
    meta["n_drives"] = n_drives
    return meta


async def main_async(args) -> int:
    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"no rollout.asl matched {args.glob}", file=sys.stderr)
        return 1

    # One rollout per scene, STRIDED across the available scenes. Taking the
    # first N alphabetically clusters on a handful of route groups from the same
    # drive log, which is weak evidence -- a stride spans distinct geometry.
    by_scene: dict[str, str] = {}
    for p in paths:
        scene = Path(p).parent.parent.name
        by_scene.setdefault(scene, p)
    scenes = sorted(by_scene.items())
    if args.limit and args.limit < len(scenes):
        step = len(scenes) / args.limit
        picked = [scenes[int(i * step)] for i in range(args.limit)]
    else:
        picked = scenes
    groups = {s.rsplit("-", 1)[0] for s, _ in picked}
    print(f"{len(paths)} logs / {len(by_scene)} scenes -> extracting "
          f"{len(picked)} across {len(groups)} route groups")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for scene, path in picked:
        dst = out / f"{scene[:48]}.json"
        if dst.exists() and not args.force:
            print(f"  skip (exists) {dst.name}")
            ok += 1
            continue
        t0 = time.time()
        try:
            rec = await extract_one(path, args.max_drives)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {scene[:40]}: {type(e).__name__}: {e}")
            continue
        if rec is None:
            print(f"  skip (no drives) {scene[:40]}")
            continue
        rec["scene_id"] = rec.get("scene_id") or scene
        dst.write_text(json.dumps(rec))
        ok += 1
        n_ego = sum(1 for e in rec["events"] if e["kind"] == "ego")
        n_rt = sum(1 for e in rec["events"] if e["kind"] == "route")
        print(f"  {dst.name}  drives={rec['n_drives']} ego={n_ego} route={n_rt} "
              f"({time.time()-t0:.1f}s, {dst.stat().st_size/1024:.0f} KiB)")
    print(f"extracted {ok}/{len(picked)} -> {out}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-drives", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
