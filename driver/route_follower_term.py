#!/usr/bin/env python
"""Route-follower + TERMINAL DECELERATION (fidelity-tail fix).

Identical to the rank-1 route_follower.py EXCEPT for one added block in _plan:
a terminal-deceleration cap that prevents the follower from overrunning the end
of the available route. The fidelity tail (mean dist_to_gt inflated to ~1.0 by
~20 scenes) is caused by driving 2-3x past short routes: where the human crept
3-4m in slow traffic, the follower drove 11-14m, producing large endpoint error.

The fix caps speed so the car can still stop by the route's end:
    v <= sqrt(2 * A_LON * dist_remaining + V_END^2)
On long-route scenes dist_remaining is large -> no effect -> byte-identical.
Only short-route (tail) scenes are slowed. Pure route geometry, no perception.

Env knobs:
  RF_TERM_DECEL=1     enable terminal deceleration (0 = pure baseline)
  RF_TERM_V_END=0.0   residual speed allowed at route end (raise to ~1.0 if
                      full-stop causes rear-ended-from-behind on live-traffic scenes)
  RF_V_MAX/A_LAT/A_LON as before
"""
from __future__ import annotations
import bisect
import logging
import math
import os
import signal
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
import numpy as np
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
import grpc

LOG = logging.getLogger("route_follower_term")
V_MAX = float(os.environ.get("RF_V_MAX", "12.0"))
A_LAT_MAX = float(os.environ.get("RF_A_LAT", "2.5"))
A_LON_MAX = float(os.environ.get("RF_A_LON", "1.5"))
TAU_CONVERGE = 1.5
HORIZON_S, DT_US = 5.0, 100_000

TERM_DECEL = os.environ.get("RF_TERM_DECEL", "1") == "1"
TERM_V_END = float(os.environ.get("RF_TERM_V_END", "0.0"))


def quat_rotate(q: common_pb2.Quat, v: np.ndarray) -> np.ndarray:
    qv = np.array([q.x, q.y, q.z], dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q.w * v)


def yaw_of(q: common_pb2.Quat) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_of_yaw(yaw: float) -> common_pb2.Quat:
    return common_pb2.Quat(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))


@dataclass
class Session:
    poses: dict[int, common_pb2.Pose] = field(default_factory=dict)
    pose_ts: list[int] = field(default_factory=list)
    latest: common_pb2.PoseAtTime | None = None
    speed: float = 0.0
    route_rig: np.ndarray | None = None
    route_ts: int = 0
    last_plan: common_pb2.Trajectory | None = None

    def add_pose(self, p: common_pb2.PoseAtTime) -> None:
        if p.timestamp_us not in self.poses:
            bisect.insort(self.pose_ts, p.timestamp_us)
        self.poses[p.timestamp_us] = p.pose
        if self.latest is None or p.timestamp_us >= self.latest.timestamp_us:
            self.latest = p
        while len(self.pose_ts) > 400:
            self.poses.pop(self.pose_ts.pop(0), None)

    def pose_at(self, ts: int) -> common_pb2.Pose | None:
        if not self.pose_ts:
            return None
        i = bisect.bisect_left(self.pose_ts, ts)
        cands = [t for t in (self.pose_ts[max(i - 1, 0)],
                             self.pose_ts[min(i, len(self.pose_ts) - 1)])]
        return self.poses[min(cands, key=lambda t: abs(t - ts))]


def resample(pts: np.ndarray, step: float = 0.5):
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-3])
    pts, s = pts[keep], s[keep]
    if len(pts) < 2:
        return pts, s
    su = np.arange(0.0, s[-1], step)
    out = np.stack([np.interp(su, s, pts[:, k]) for k in range(3)], axis=1)
    return out, su


class RouteFollower(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self) -> None:
        self._s: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._server: grpc.Server | None = None

    def attach_server(self, s: grpc.Server) -> None:
        self._server = s

    def _sess(self, uuid: str, ctx) -> Session:
        with self._lock:
            s = self._s.get(uuid)
        if s is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {uuid}")
        return s

    def start_session(self, req, ctx):
        with self._lock:
            self._s[req.session_uuid] = Session()
        LOG.info("start %s (TERM_DECEL=%s V_END=%.2f)", req.session_uuid, TERM_DECEL, TERM_V_END)
        return common_pb2.SessionRequestStatus()

    def close_session(self, req, ctx):
        with self._lock:
            self._s.pop(req.session_uuid, None)
        return common_pb2.Empty()

    def submit_image_observation(self, req, ctx):
        return common_pb2.Empty()

    def submit_recording_ground_truth(self, req, ctx):
        return common_pb2.Empty()

    def submit_egomotion_observation(self, req, ctx):
        s = self._sess(req.session_uuid, ctx)
        with self._lock:
            for p in req.trajectory.poses:
                s.add_pose(p)
            if req.dynamic_states:
                lv = req.dynamic_states[-1].linear_velocity
                s.speed = float(math.hypot(lv.x, lv.y))
        return common_pb2.Empty()

    def submit_route(self, req, ctx):
        s = self._sess(req.session_uuid, ctx)
        wps = req.route.waypoints
        if wps:
            with self._lock:
                s.route_rig = np.array([[w.x, w.y, w.z] for w in wps], dtype=float)
                s.route_ts = req.route.timestamp_us
        return common_pb2.Empty()

    def _plan(self, s: Session, t0: int) -> common_pb2.Trajectory:
        ego = s.latest
        p0 = np.array([ego.pose.vec.x, ego.pose.vec.y, ego.pose.vec.z])
        yaw0 = yaw_of(ego.pose.quat)
        anchor = s.pose_at(s.route_ts) if s.route_rig is not None else None
        if s.route_rig is None or anchor is None or len(s.route_rig) < 2:
            return straight(p0, yaw0, t0, max(s.speed, 3.0))
        base = np.array([anchor.vec.x, anchor.vec.y, anchor.vec.z])
        route = np.stack([quat_rotate(anchor.quat, w) + base for w in s.route_rig])
        route = np.vstack([p0[None, :], route])
        pts, s_arc = resample(route)
        if len(pts) < 2:
            return straight(p0, yaw0, t0, max(s.speed, 3.0))
        d = np.linalg.norm(pts[:, :2] - p0[:2], axis=1)
        i0 = int(np.argmin(d))
        tang = pts[min(i0 + 1, len(pts) - 1), :2] - pts[max(i0 - 1, 0), :2]
        th = math.atan2(tang[1], tang[0]) if np.linalg.norm(tang) > 1e-6 else yaw0
        nvec = np.array([-math.sin(th), math.cos(th)])
        e0 = float(np.dot(p0[:2] - pts[i0, :2], nvec))
        dx = np.gradient(pts[:, 0]); dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx); ddy = np.gradient(dy)
        kap = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-6)
        v_cap = np.minimum(V_MAX, np.sqrt(A_LAT_MAX / np.maximum(kap, 1e-3)))

        # === TERMINAL DECELERATION: don't overrun the end of the available route ===
        # Fidelity tail = driving 2-3x past short routes (endpoint error). Cap speed so
        # we can still stop by route's end. No effect on long routes (regression-safe).
        if TERM_DECEL:
            dist_remaining = np.maximum(s_arc[-1] - s_arc, 0.0)
            v_term = np.sqrt(2.0 * A_LON_MAX * dist_remaining + TERM_V_END**2)
            v_cap = np.minimum(v_cap, v_term)

        traj = common_pb2.Trajectory()
        n = int(HORIZON_S * 1e6 / DT_US) + 1
        s_cur, v = s_arc[i0], max(s.speed, 0.0)
        for i in range(n):
            t = i * DT_US / 1e6
            j = int(np.searchsorted(s_arc, s_cur, side="right")) - 1
            j = max(0, min(j, len(pts) - 2))
            v_t = float(v_cap[j])
            v += np.clip(v_t - v, -A_LON_MAX * 0.1, A_LON_MAX * 0.1) if i else 0.0
            if i:
                s_cur = min(s_cur + v * DT_US / 1e6, s_arc[-1])
            ref = np.array([np.interp(s_cur, s_arc, pts[:, k]) for k in range(3)])
            k2 = max(0, min(int(np.searchsorted(s_arc, s_cur)), len(pts) - 2))
            tv = pts[k2 + 1, :2] - pts[k2, :2]
            yaw = math.atan2(tv[1], tv[0]) if np.linalg.norm(tv) > 1e-6 else yaw0
            off = e0 * math.exp(-t / TAU_CONVERGE)
            pos = ref[:2] + off * np.array([-math.sin(yaw), math.cos(yaw)])
            traj.poses.append(common_pb2.PoseAtTime(
                timestamp_us=t0 + i * DT_US,
                pose=common_pb2.Pose(
                    vec=common_pb2.Vec3(x=float(pos[0]), y=float(pos[1]), z=float(ref[2])),
                    quat=quat_of_yaw(yaw))))
        return traj

    def drive(self, req, ctx):
        s = self._sess(req.session_uuid, ctx)
        try:
            with self._lock:
                if s.latest is None:
                    raise RuntimeError("no ego pose yet")
                traj = self._plan(s, req.time_now_us)
                s.last_plan = traj
            return egodriver_pb2.DriveResponse(trajectory=traj)
        except Exception:
            LOG.exception("drive failed -> safe fallback")
            return egodriver_pb2.DriveResponse(trajectory=fallback(s, req.time_now_us))

    def get_version(self, req, ctx):
        return common_pb2.VersionId(version_id="route-follower-term",
                                    git_hash="local",
                                    grpc_api_version=API_VERSION_MESSAGE)

    def shut_down(self, req, ctx):
        if self._server:
            threading.Thread(target=lambda: (time.sleep(0.05),
                                             self._server.stop(grace=0.0)),
                             daemon=True).start()
        return common_pb2.Empty()


def straight(p0, yaw, t0, v) -> common_pb2.Trajectory:
    tr = common_pb2.Trajectory()
    for i in range(int(HORIZON_S * 1e6 / DT_US) + 1):
        d = v * i * DT_US / 1e6
        tr.poses.append(common_pb2.PoseAtTime(
            timestamp_us=t0 + i * DT_US,
            pose=common_pb2.Pose(
                vec=common_pb2.Vec3(x=float(p0[0] + math.cos(yaw) * d),
                                    y=float(p0[1] + math.sin(yaw) * d),
                                    z=float(p0[2])),
                quat=quat_of_yaw(yaw))))
    return tr


def fallback(s: Session, t0: int) -> common_pb2.Trajectory:
    if s.latest is None:
        return common_pb2.Trajectory()
    p = s.latest.pose
    p0 = np.array([p.vec.x, p.vec.y, p.vec.z])
    yaw, v = yaw_of(p.quat), s.speed
    tr = common_pb2.Trajectory()
    d = 0.0
    for i in range(int(HORIZON_S * 1e6 / DT_US) + 1):
        v = max(0.0, v - 4.0 * DT_US / 1e6)
        d += v * DT_US / 1e6
        tr.poses.append(common_pb2.PoseAtTime(
            timestamp_us=t0 + i * DT_US,
            pose=common_pb2.Pose(
                vec=common_pb2.Vec3(x=float(p0[0] + math.cos(yaw) * d),
                                    y=float(p0[1] + math.sin(yaw) * d),
                                    z=float(p0[2])),
                quat=quat_of_yaw(yaw))))
    return tr


def main() -> None:
    logging.basicConfig(level=os.environ.get("ALPASIM_DRIVER_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0")
    port = int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    svc = RouteFollower()
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(svc, srv)
    svc.attach_server(srv)
    if srv.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: srv.stop(grace=0.0))
    srv.start()
    LOG.info("route-follower-TERM on %s:%d V_MAX=%.1f TERM_DECEL=%s V_END=%.2f",
             host, port, V_MAX, TERM_DECEL, TERM_V_END)
    srv.wait_for_termination()


if __name__ == "__main__":
    main()
