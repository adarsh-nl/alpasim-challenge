#!/usr/bin/env python
"""Route-follower + CAM_F0 corridor-looming speed GOVERNOR (closed-loop test).

Identical route-following geometry to the rank-1 route_follower.py. The ONLY change
is a speed modulation: submit_image_observation (previously a no-op that discarded
the cameras) now decodes CAM_F0, measures a lower-center "looming" signal -- the area
of large near masses in the forward corridor -- tracks its growth across drive()
calls, and sets a per-session yield_factor in [GOV_FLOOR, 1.0]. _plan multiplies the
curvature-limited v_cap by yield_factor.

Regression-safe by construction: when the corridor is open, yield_factor == 1.0 and
the trajectory is byte-identical to the baseline route-follower. Only when a large
mass looms in the forward corridor does the governor slow the car.

This tests LOOMING-AS-CONTROLLER in closed loop. Offline, looming-as-classifier
failed to separate collisions from clear scenes -- but offline it scored the BLIND
policy's fixed frames. In closed loop the governor changes the trajectory (brakes
early, approaches slower), a different system the offline analysis could not evaluate.

Pure numpy + Pillow JPEG decode. No neural net (latency-budget safe, ~15-30ms/frame).

Env knobs:
  GOV_ENABLE=1        turn the governor on (0 = pure baseline)
  GOV_GAIN=8.0        yield gain on looming growth rate
  GOV_FLOOR=0.15      minimum yield_factor (never fully stop from vision alone)
  RF_V_MAX/A_LAT/A_LON as before
"""
from __future__ import annotations
import bisect
import io
import logging
import math
import os
import signal
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
import grpc

LOG = logging.getLogger("route_follower_gov")
V_MAX = float(os.environ.get("RF_V_MAX", "12.0"))
A_LAT_MAX = float(os.environ.get("RF_A_LAT", "2.5"))
A_LON_MAX = float(os.environ.get("RF_A_LON", "1.5"))
TAU_CONVERGE = 1.5
HORIZON_S, DT_US = 5.0, 100_000

GOV_ENABLE = os.environ.get("GOV_ENABLE", "1") == "1"
GOV_GAIN = float(os.environ.get("GOV_GAIN", "8.0"))
GOV_FLOOR = float(os.environ.get("GOV_FLOOR", "0.15"))
FWD_CAM = os.environ.get("GOV_CAM", "CAM_F0")


def quat_rotate(q: common_pb2.Quat, v: np.ndarray) -> np.ndarray:
    qv = np.array([q.x, q.y, q.z], dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q.w * v)


def yaw_of(q: common_pb2.Quat) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_of_yaw(yaw: float) -> common_pb2.Quat:
    return common_pb2.Quat(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))


def looming_area(rgb: np.ndarray) -> float:
    """Fraction of the lower-center corridor occupied by large near masses.

    Heuristic, pure numpy: in the lower-center region, a nearby vehicle is a large
    contiguous block that differs from the road surface. We proxy 'near mass' by
    the fraction of corridor pixels that are markedly darker OR markedly different
    from the corridor's median colour (vehicles, shadows under vehicles). Scale-free;
    only its time-derivative is used by the governor.
    """
    h, w = rgb.shape[:2]
    # forward corridor: central 40% width, lower 45% height
    x0, x1 = int(0.30 * w), int(0.70 * w)
    y0, y1 = int(0.55 * h), int(0.95 * h)
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return 0.0
    gray = crop.mean(axis=2)
    med = np.median(gray)
    # dark masses (under-vehicle shadow, tires, dark car bodies)
    dark = gray < (med * 0.55)
    # colour deviation from road median (vehicles differ from grey asphalt)
    med_rgb = np.median(crop.reshape(-1, 3), axis=0)
    dev = np.linalg.norm(crop - med_rgb[None, None, :], axis=2)
    devhi = dev > (np.median(dev) + 1.5 * (np.std(dev) + 1e-6))
    mass = np.logical_or(dark, devhi)
    return float(mass.mean())


@dataclass
class Session:
    poses: dict[int, common_pb2.Pose] = field(default_factory=dict)
    pose_ts: list[int] = field(default_factory=list)
    latest: common_pb2.PoseAtTime | None = None
    speed: float = 0.0
    route_rig: np.ndarray | None = None
    route_ts: int = 0
    last_plan: common_pb2.Trajectory | None = None
    # governor state
    loom_hist: list[tuple[float, float]] = field(default_factory=list)   # (t_s, area)
    yield_factor: float = 1.0

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

    def update_governor(self, t_us: int, area: float) -> None:
        t = t_us / 1e6
        self.loom_hist.append((t, area))
        if len(self.loom_hist) > 20:
            self.loom_hist.pop(0)
        if len(self.loom_hist) < 2 or not GOV_ENABLE:
            self.yield_factor = 1.0
            return
        # closing = looming area GROWING. Use recent slope over ~last 1s.
        recent = [x for x in self.loom_hist if x[0] >= t - 1.0]
        if len(recent) < 2:
            self.yield_factor = 1.0
            return
        (t_a, a_a), (t_b, a_b) = recent[0], recent[-1]
        dt = t_b - t_a
        growth = (a_b - a_a) / dt if dt > 1e-3 else 0.0
        # also weight by absolute proximity (large area now = nearer)
        prox = a_b
        yield_reduction = GOV_GAIN * max(growth, 0.0) * max(prox, 0.0)
        self.yield_factor = float(np.clip(1.0 - yield_reduction, GOV_FLOOR, 1.0))


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
        LOG.info("start %s (GOV_ENABLE=%s GAIN=%.1f FLOOR=%.2f)",
                 req.session_uuid, GOV_ENABLE, GOV_GAIN, GOV_FLOOR)
        return common_pb2.SessionRequestStatus()

    def close_session(self, req, ctx):
        with self._lock:
            self._s.pop(req.session_uuid, None)
        return common_pb2.Empty()

    def submit_image_observation(self, req, ctx):
        # GOVERNOR: decode CAM_F0, update looming -> yield_factor
        try:
            ci = req.camera_image
            if ci.logical_id == FWD_CAM and ci.image_bytes:
                rgb = np.asarray(Image.open(io.BytesIO(ci.image_bytes)).convert("RGB"))
                area = looming_area(rgb)
                s = self._sess(req.session_uuid, ctx)
                with self._lock:
                    s.update_governor(ci.frame_start_us, area)
        except Exception:
            LOG.exception("governor image path failed -> yield_factor stays 1.0")
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

        # === GOVERNOR: scale the whole speed profile by the yield factor ===
        yf = s.yield_factor if GOV_ENABLE else 1.0
        v_cap = v_cap * yf

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
        return common_pb2.VersionId(version_id="route-follower-gov",
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
    LOG.info("route-follower-GOV on %s:%d  V_MAX=%.1f GOV_ENABLE=%s GAIN=%.1f FLOOR=%.2f",
             host, port, V_MAX, GOV_ENABLE, GOV_GAIN, GOV_FLOOR)
    srv.wait_for_termination()


if __name__ == "__main__":
    main()
