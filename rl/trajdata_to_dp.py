"""
trajdata (AlpaSim nuPlan cache)  ->  Diffusion-Planner input tensors.

Bypasses nuplan-devkit entirely: reads the trajdata feather/dill/protobuf cache
directly and emits the 7 tensors Diffusion-Planner's model expects.

Target spec (from Diffusion-Planner defaults + normalization.json):

    ego_current_state        (10,)             constant, ego-centric
    neighbor_agents_past     (32, T, 11)       8 state + 3 one-hot type
    static_objects           (5, 10)           6 state + 4 one-hot type
    lanes                    (70, 20, 12)      xy,tangent,to_left,to_right,tl(4)
    lanes_speed_limit        (70, 1)           NOT IN CACHE -> filled
    lanes_has_speed_limit    (70, 1)           bool mask
    route_lanes              (25, 20, 12)
    route_lanes_speed_limit  (25, 1)           NOT IN CACHE -> filled
    route_lanes_has_speed_limit (25, 1)

Usage
-----
    python trajdata_to_dp.py --root $ALPASIM_NUPLAN_ROOT --limit 4 --out /tmp/dp_feats
    python trajdata_to_dp.py --root $ALPASIM_NUPLAN_ROOT --selftest
"""

from __future__ import annotations

import argparse
import dill
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Spec constants -- match Diffusion-Planner data_process.py argparse defaults
# ----------------------------------------------------------------------------

AGENT_NUM = 32
STATIC_NUM = 5
LANE_NUM = 70
LANE_LEN = 20
ROUTE_NUM = 25
ROUTE_LEN = 20

T_HISTORY = 21          # 2.0s at 10Hz, inclusive of current -> 21 samples
DT = 0.1

LANE_RADIUS_M = 100.0   # crop radius around ego before resampling
STATIC_SPEED_EPS = 0.2  # m/s; track-wide max speed below this => static
ROUTE_MATCH_M = 5.0     # max lateral distance for ego-future -> lane matching
ROUTE_SEED_MAX_M = 10.0 # max distance from ego to its seed lane

# trajdata AgentType enum
TRAJDATA_VEHICLE = 1
TRAJDATA_PEDESTRIAN = 2
TRAJDATA_BICYCLE = 3
TRAJDATA_MOTORCYCLE = 4

# Diffusion-Planner one-hot ordering for neighbours: [VEHICLE, PEDESTRIAN, BICYCLE]
_TYPE_ONEHOT = {
    TRAJDATA_VEHICLE: (1.0, 0.0, 0.0),
    TRAJDATA_PEDESTRIAN: (0.0, 1.0, 0.0),
    TRAJDATA_BICYCLE: (0.0, 0.0, 1.0),
    TRAJDATA_MOTORCYCLE: (0.0, 0.0, 1.0),  # folded into BICYCLE
}

# Traffic-light one-hot width used by LaneSegmentTrafficLightData.encoding_dim()
TL_DIM = 4
# Diffusion-Planner's one-hot slots. trajdata's enum is
# {NO_DATA: -1, UNKNOWN: 0, GREEN: 1, RED: 2} -- there is no YELLOW in this
# cache, so that slot is simply never set.
TL_GREEN, TL_YELLOW, TL_RED, TL_UNKNOWN = 0, 1, 2, 3

# trajdata TrafficLightStatus -> our one-hot slot
_TL_FROM_TRAJDATA = {1: TL_GREEN, 2: TL_RED, 0: TL_UNKNOWN, -1: TL_UNKNOWN}

# ---------------------------------------------------------------------------
# KNOWN GAP: the AlpaSim trajdata cache has no speed limits on RoadLane.
# Diffusion-Planner was trained with real nuPlan speed limits, so feeding the
# normalization mean (0 m/s) is out-of-distribution -- it reads as "stopped
# road". We fill with a plausible urban constant instead and expose it as a
# knob so the choice can be ablated against teacher trajectory quality.
# ---------------------------------------------------------------------------
DEFAULT_SPEED_LIMIT_MPS = 13.4  # ~30 mph


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def se2_world_to_ego(xy: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Transform (..., 2) world points into the ego frame at `anchor` (x, y, heading)."""
    c, s = np.cos(-anchor[2]), np.sin(-anchor[2])
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (xy - anchor[:2]) @ rot.T


def rotate_vec_to_ego(vec: np.ndarray, heading: float) -> np.ndarray:
    """Rotate (..., 2) world vectors (velocity, tangent) into the ego frame."""
    c, s = np.cos(-heading), np.sin(-heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return vec @ rot.T


def resample_polyline(pts: np.ndarray, n: int,
                     f0: float = 0.0, f1: float = 1.0) -> np.ndarray:
    """Arc-length resample an (M, >=2) polyline to n points over fraction [f0, f1].

    The fraction range lets center and boundary polylines be cropped to the
    SAME arc-length span. Cropping only the center leaves the boundaries
    parametrized over the full lane, so to_left/to_right offsets come out as
    tens of metres instead of a lane half-width.
    """
    pts = np.asarray(pts, dtype=np.float64)[:, :2]
    if len(pts) == 0:
        return np.zeros((n, 2))
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(pts[:1], n, axis=0)

    target = np.linspace(f0 * s[-1], f1 * s[-1], n)
    return np.stack([np.interp(target, s, pts[:, 0]),
                     np.interp(target, s, pts[:, 1])], axis=-1)


def polyline_tangents(pts: np.ndarray) -> np.ndarray:
    """Unit tangent per point of an (n, 2) polyline."""
    d = np.gradient(pts, axis=0)
    nrm = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(nrm, 1e-6)


# ----------------------------------------------------------------------------
# Scene loading
# ----------------------------------------------------------------------------

@dataclass
class SceneData:
    name: str
    location: str
    dt: float
    length_ts: int
    agents: pd.DataFrame              # agent_id, scene_ts, x, y, z, vx, vy, ax, ay, heading
    agent_type: dict[str, int]
    agent_extent: dict[str, tuple[float, float]]   # (length, width)
    tls: pd.DataFrame | None          # lane_id, scene_ts, status


def load_scene(scene_dir: Path) -> SceneData:
    md = dill.load(open(scene_dir / "scene_metadata_dt0.10.dill", "rb"))

    a_type, a_ext = {}, {}
    for a in md.agents:
        a_type[a.name] = int(a.type)
        ext = a.extent
        a_ext[a.name] = (float(getattr(ext, "length", 4.5)),
                         float(getattr(ext, "width", 2.0)))

    agents = pd.read_feather(scene_dir / "agent_data_dt0.10.feather")

    tls_path = scene_dir / "tls_data_dt0.05.feather"
    tls = pd.read_feather(tls_path) if tls_path.exists() else None

    return SceneData(
        name=md.name,
        location=md.location,
        dt=float(md.dt),
        length_ts=int(md.length_timesteps),
        agents=agents,
        agent_type=a_type,
        agent_extent=a_ext,
        tls=tls,
    )


# ----------------------------------------------------------------------------
# Traffic lights
# ----------------------------------------------------------------------------

def build_tl_lookup(scene: SceneData, ts: int) -> dict[str, int]:
    """lane_id -> TL_* slot, as of agent-cadence timestep `ts`.

    The tls table has its own timestep index (observed 0..75 on a 56-step
    scene), so it is neither 1:1 nor a clean 2x of the agent cadence. We map
    proportionally onto the tls range, then take each lane's most recent
    status at or before that point -- lanes only appear while observed, so
    carrying the last known state forward is the correct reading.
    """
    if scene.tls is None or len(scene.tls) == 0:
        return {}

    tls = scene.tls
    tls_max = int(tls.scene_ts.max())
    denom = max(scene.length_ts - 1, 1)
    row_ts = int(round(ts / denom * tls_max))

    sub = tls[tls.scene_ts <= row_ts]
    if len(sub) == 0:
        sub = tls[tls.scene_ts == int(tls.scene_ts.min())]

    latest = sub.sort_values("scene_ts").groupby("lane_id").tail(1)
    return {str(l): _map_tl_status(s)
            for l, s in zip(latest.lane_id.values, latest.status.values)}


def _map_tl_status(status) -> int:
    """trajdata stores status as int64 (TrafficLightStatus enum)."""
    try:
        return _TL_FROM_TRAJDATA.get(int(status), TL_UNKNOWN)
    except (TypeError, ValueError):
        return TL_UNKNOWN


# ----------------------------------------------------------------------------
# Feature builders
# ----------------------------------------------------------------------------

def build_ego_current_state() -> np.ndarray:
    """Ego-centric, so this is constant.

    Layout: x, y, cos, sin, vx, vy, ax, ay, steering_angle, yaw_rate.
    Diffusion-Planner only consumes x/y/cos/sin at inference (see their
    data_processor.py line 72), so the trailing entries stay zero.
    """
    v = np.zeros(10, dtype=np.float32)
    v[2] = 1.0  # cos(0)
    return v


def build_neighbors(scene: SceneData, ts: int, anchor: np.ndarray
                    ) -> tuple[np.ndarray, list[str]]:
    """(AGENT_NUM, T_HISTORY, 11) ego-centric neighbour history."""
    lo = ts - T_HISTORY + 1
    df = scene.agents
    win = df[(df.scene_ts >= lo) & (df.scene_ts <= ts) & (df.agent_id != "ego")]

    out = np.zeros((AGENT_NUM, T_HISTORY, 11), dtype=np.float32)
    if len(win) == 0:
        return out, []

    # rank by distance at the query timestep
    cur = win[win.scene_ts == ts]
    if len(cur) == 0:
        return out, []
    d = np.linalg.norm(cur[["x", "y"]].values - anchor[:2], axis=1)
    order = cur.agent_id.values[np.argsort(d)][:AGENT_NUM]

    by_id = {aid: g for aid, g in win.groupby("agent_id")}
    for i, aid in enumerate(order):
        g = by_id[aid].sort_values("scene_ts")
        idx = (g.scene_ts.values - lo).astype(int)
        valid = (idx >= 0) & (idx < T_HISTORY)
        idx = idx[valid]
        if len(idx) == 0:
            continue

        xy = se2_world_to_ego(g[["x", "y"]].values[valid], anchor)
        hd = g["heading"].values[valid] - anchor[2]
        vel = rotate_vec_to_ego(g[["vx", "vy"]].values[valid], anchor[2])
        length, width = scene.agent_extent.get(aid, (4.5, 2.0))
        oh = _TYPE_ONEHOT.get(scene.agent_type.get(aid, TRAJDATA_VEHICLE),
                              (1.0, 0.0, 0.0))

        out[i, idx, 0] = xy[:, 0]
        out[i, idx, 1] = xy[:, 1]
        out[i, idx, 2] = np.cos(hd)
        out[i, idx, 3] = np.sin(hd)
        out[i, idx, 4] = vel[:, 0]
        out[i, idx, 5] = vel[:, 1]
        out[i, idx, 6] = length
        out[i, idx, 7] = width
        out[i, idx, 8:11] = oh

    return out, list(order)


def build_static_objects(scene: SceneData, ts: int, anchor: np.ndarray,
                         exclude: set[str]) -> np.ndarray:
    """(STATIC_NUM, 10). Derived: agents whose track-wide max speed is ~0.

    The cache has no static/parked flag, so this is a heuristic. Layout is
    6 state (x, y, cos, sin, length, width) + 4 one-hot type.
    """
    out = np.zeros((STATIC_NUM, 10), dtype=np.float32)
    df = scene.agents
    cur = df[(df.scene_ts == ts) & (df.agent_id != "ego")]
    if len(cur) == 0:
        return out

    speed = df.groupby("agent_id").apply(
        lambda g: float(np.max(np.hypot(g.vx.values, g.vy.values))),
        include_groups=False,
    )
    static_ids = {a for a, s in speed.items()
                  if s < STATIC_SPEED_EPS and a not in exclude}
    if not static_ids:
        return out

    cur = cur[cur.agent_id.isin(static_ids)]
    if len(cur) == 0:
        return out
    d = np.linalg.norm(cur[["x", "y"]].values - anchor[:2], axis=1)
    cur = cur.iloc[np.argsort(d)[:STATIC_NUM]]

    xy = se2_world_to_ego(cur[["x", "y"]].values, anchor)
    hd = cur["heading"].values - anchor[2]
    for i in range(len(cur)):
        aid = cur.agent_id.values[i]
        length, width = scene.agent_extent.get(aid, (4.5, 2.0))
        out[i, 0:6] = [xy[i, 0], xy[i, 1], np.cos(hd[i]), np.sin(hd[i]),
                       length, width]
        out[i, 6] = 1.0  # generic obstacle class
    return out


def _lane_to_12dim(center: np.ndarray, left: np.ndarray, right: np.ndarray,
                   tl: int, anchor: np.ndarray, n_pts: int,
                   f0: float = 0.0, f1: float = 1.0) -> np.ndarray:
    """[xy(2), tangent(2), to_left(2), to_right(2), tl_onehot(4)] per point.

    All three polylines are resampled over the same arc-length fraction so
    point i of the center pairs with point i of each boundary.
    """
    c = resample_polyline(center, n_pts, f0, f1)
    l = resample_polyline(left, n_pts, f0, f1) if left is not None and len(left) else c
    r = resample_polyline(right, n_pts, f0, f1) if right is not None and len(right) else c

    c_e = se2_world_to_ego(c, anchor)
    l_e = se2_world_to_ego(l, anchor)
    r_e = se2_world_to_ego(r, anchor)
    tan = polyline_tangents(c_e)

    feat = np.zeros((n_pts, 12), dtype=np.float32)
    feat[:, 0:2] = c_e
    feat[:, 2:4] = tan
    feat[:, 4:6] = l_e - c_e
    feat[:, 6:8] = r_e - c_e
    feat[:, 8 + tl] = 1.0
    return feat


def build_lanes(vec_map, anchor: np.ndarray, tl_lookup: dict[str, int],
                speed_limit: float) -> dict[str, np.ndarray]:
    """(LANE_NUM, LANE_LEN, 12) nearest lanes, cropped then resampled."""
    lanes = np.zeros((LANE_NUM, LANE_LEN, 12), dtype=np.float32)
    spd = np.full((LANE_NUM, 1), speed_limit, dtype=np.float32)
    has = np.zeros((LANE_NUM, 1), dtype=bool)

    cand = []
    for ln in vec_map.lanes:
        c = np.asarray(ln.center.xyz)[:, :2]
        d = np.linalg.norm(c - anchor[:2], axis=1)
        if d.min() > LANE_RADIUS_M:
            continue
        # crop to the in-radius span so resampling isn't diluted by long roads
        keep = np.where(d <= LANE_RADIUS_M)[0]
        cand.append((d.min(), ln, keep))

    cand.sort(key=lambda t: t[0])
    for i, (_, ln, keep) in enumerate(cand[:LANE_NUM]):
        c_full = np.asarray(ln.center.xyz)[:, :2]
        seg = np.linalg.norm(np.diff(c_full, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        tot = s[-1] if s[-1] > 1e-6 else 1.0
        f0, f1 = float(s[keep[0]] / tot), float(s[keep[-1]] / tot)
        if f1 - f0 < 1e-3:
            f0, f1 = 0.0, 1.0
        le = np.asarray(ln.left_edge.xyz)[:, :2] if ln.left_edge is not None else None
        re = np.asarray(ln.right_edge.xyz)[:, :2] if ln.right_edge is not None else None
        tl = tl_lookup.get(str(ln.id), TL_UNKNOWN)
        lanes[i] = _lane_to_12dim(c_full, le, re, tl, anchor, LANE_LEN, f0, f1)
        has[i] = False  # no speed limit in this cache -- see DEFAULT_SPEED_LIMIT_MPS

    return {"lanes": lanes, "lanes_speed_limit": spd,
            "lanes_has_speed_limit": has}


def _nearest_lane(vec_map, xy, max_d):
    """Lane whose centerline passes closest to `xy`, or None if beyond max_d."""
    best, bd = None, 1e9
    for ln in vec_map.lanes:
        c = np.asarray(ln.center.xyz)[:, :2]
        d = float(np.linalg.norm(c - xy, axis=1).min())
        if d < bd:
            bd, best = d, ln
    return best if bd <= max_d else None


def build_route_lanes(vec_map, scene: SceneData, ts: int, anchor: np.ndarray,
                      tl_lookup: dict[str, int], speed_limit: float
                      ) -> dict[str, np.ndarray]:
    """(ROUTE_NUM, ROUTE_LEN, 12).

    Seeds from the ego's current lane and walks the lane graph forward via
    next_lanes. The logged ego future is used only to disambiguate branches at
    forks -- privileged, correct for a teacher, unavailable to the student.

    Graph traversal rather than pure future-matching is what gives the route a
    horizon comparable to nuPlan roadblocks; the 5.6s scenes only cover ~20m of
    actual ego travel, which is far too short for the model's route encoder.
    """
    out = np.zeros((ROUTE_NUM, ROUTE_LEN, 12), dtype=np.float32)
    spd = np.full((ROUTE_NUM, 1), speed_limit, dtype=np.float32)
    has = np.zeros((ROUTE_NUM, 1), dtype=bool)
    empty = {"route_lanes": out, "route_lanes_speed_limit": spd,
             "route_lanes_has_speed_limit": has}

    by_id = {str(l.id): l for l in vec_map.lanes}

    fut = scene.agents[(scene.agents.agent_id == "ego") &
                       (scene.agents.scene_ts >= ts)].sort_values("scene_ts")
    fxy = fut[["x", "y"]].values if len(fut) else np.zeros((0, 2))

    seed = _nearest_lane(vec_map, anchor[:2], ROUTE_SEED_MAX_M)
    if seed is None:
        return empty

    def future_dist(ln):
        if len(fxy) == 0:
            return 0.0
        c = np.asarray(ln.center.xyz)[:, :2]
        return float(np.linalg.norm(fxy[:, None, :] - c[None, :, :], axis=-1).min())

    chain, seen, cur = [seed], {str(seed.id)}, seed
    while len(chain) < ROUTE_NUM:
        nxt = [by_id[str(i)] for i in cur.next_lanes
               if str(i) in by_id and str(i) not in seen]
        if not nxt:
            break
        cur = nxt[0] if len(nxt) == 1 else min(nxt, key=future_dist)
        chain.append(cur)
        seen.add(str(cur.id))

    # widen the corridor with parallel lanes in the same roadblock
    for ln in list(chain):
        if len(chain) >= ROUTE_NUM:
            break
        for adj in list(ln.adj_lanes_left) + list(ln.adj_lanes_right):
            if str(adj) in by_id and str(adj) not in seen and len(chain) < ROUTE_NUM:
                chain.append(by_id[str(adj)])
                seen.add(str(adj))

    for i, ln in enumerate(chain[:ROUTE_NUM]):
        c = np.asarray(ln.center.xyz)[:, :2]
        le = np.asarray(ln.left_edge.xyz)[:, :2] if ln.left_edge is not None else None
        re = np.asarray(ln.right_edge.xyz)[:, :2] if ln.right_edge is not None else None
        tl = tl_lookup.get(str(ln.id), TL_UNKNOWN)
        out[i] = _lane_to_12dim(c, le, re, tl, anchor, ROUTE_LEN)

    return {"route_lanes": out, "route_lanes_speed_limit": spd,
            "route_lanes_has_speed_limit": has}


# ----------------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------------

def build_sample(scene: SceneData, vec_map, ts: int,
                 speed_limit: float = DEFAULT_SPEED_LIMIT_MPS) -> dict:
    ego = scene.agents[(scene.agents.agent_id == "ego") &
                       (scene.agents.scene_ts == ts)]
    if len(ego) == 0:
        raise ValueError(f"no ego row at ts={ts} in {scene.name}")
    e = ego.iloc[0]
    anchor = np.array([e.x, e.y, e.heading], dtype=np.float64)

    tl = build_tl_lookup(scene, ts)
    neigh, neigh_ids = build_neighbors(scene, ts, anchor)

    data = {
        "map_name": scene.location,
        "token": f"{scene.name}_{ts:03d}",
        "ego_current_state": build_ego_current_state(),
        "neighbor_agents_past": neigh,
        "static_objects": build_static_objects(scene, ts, anchor, set(neigh_ids)),
    }
    data.update(build_lanes(vec_map, anchor, tl, speed_limit))
    data.update(build_route_lanes(vec_map, scene, ts, anchor, tl, speed_limit))
    return data


def normalize(data: dict, norm_path: Path) -> dict:
    """Apply Diffusion-Planner's normalization.json in-place on a copy."""
    norm = json.load(open(norm_path))
    out = dict(data)
    for key, stats in norm.items():
        if key not in out:
            continue
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        arr = np.asarray(out[key], dtype=np.float32)
        if arr.shape[-1] != mean.shape[-1]:
            continue
        out[key] = (arr - mean) / np.maximum(std, 1e-6)
    return out


def valid_timesteps(scene: SceneData) -> list[int]:
    """Query points with a full T_HISTORY window available."""
    return list(range(T_HISTORY - 1, scene.length_ts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=2, help="number of scenes")
    ap.add_argument("--stride", type=int, default=5,
                    help="timestep stride; 5 => sim policy cadence (2Hz)")
    ap.add_argument("--speed-limit", type=float, default=DEFAULT_SPEED_LIMIT_MPS)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(args.root))
    from trajdata import MapAPI

    cache = args.root / "nuplan_test"
    scenes = sorted(p for p in cache.iterdir() if p.is_dir() and p.name != "maps")
    api = MapAPI(args.root)
    map_cache: dict[str, object] = {}

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    n = 0
    for sd in scenes[: args.limit]:
        scene = load_scene(sd)
        if scene.location not in map_cache:
            map_cache[scene.location] = api.get_map(f"nuplan_test:{scene.location}")
        vm = map_cache[scene.location]

        for ts in valid_timesteps(scene)[:: args.stride]:
            data = build_sample(scene, vm, ts, args.speed_limit)
            if args.normalize:
                data = normalize(data, Path(__file__).parent / "normalization.json")

            if args.selftest:
                _check(data)
            if args.out:
                np.savez(args.out / f"{data['map_name']}_{data['token']}.npz", **data)
            n += 1

        print(f"{scene.name}  loc={scene.location}  "
              f"ts={len(valid_timesteps(scene))}  emitted={n}")

    print(f"\ntotal samples: {n}")
    return 0


_EXPECT = {
    "ego_current_state": (10,),
    "neighbor_agents_past": (AGENT_NUM, T_HISTORY, 11),
    "static_objects": (STATIC_NUM, 10),
    "lanes": (LANE_NUM, LANE_LEN, 12),
    "lanes_speed_limit": (LANE_NUM, 1),
    "route_lanes": (ROUTE_NUM, ROUTE_LEN, 12),
    "route_lanes_speed_limit": (ROUTE_NUM, 1),
}


def _check(data: dict) -> None:
    for k, shape in _EXPECT.items():
        got = np.asarray(data[k]).shape
        assert got == shape, f"{k}: expected {shape}, got {got}"
        assert np.isfinite(np.asarray(data[k], dtype=np.float64)).all(), \
            f"{k}: non-finite values"
    nz_lanes = int((np.abs(data["lanes"]).sum(axis=(1, 2)) > 0).sum())
    nz_route = int((np.abs(data["route_lanes"]).sum(axis=(1, 2)) > 0).sum())
    nz_agents = int((np.abs(data["neighbor_agents_past"]).sum(axis=(1, 2)) > 0).sum())
    assert nz_lanes > 0, "no lanes populated -- check LANE_RADIUS_M / map location"
    print(f"  ok {data['token']}: lanes={nz_lanes}/{LANE_NUM} "
          f"route={nz_route}/{ROUTE_NUM} agents={nz_agents}/{AGENT_NUM}")


if __name__ == "__main__":
    raise SystemExit(main())
