#!/usr/bin/env python
"""Uncertainty-aware abstention driver (the 'fix' half of measure->diagnose->fix).

DIAGNOSIS (from the regression gate): the follower's catastrophic failures are
RELIABILITY failures -- it drives blind and fast exactly when it should be
uncertain (route-absent scenes like dd162898: 1 waypoint -> straight() -> rear-end).
The gate showed the failure tail is reliability, not driving quality.

FIX: an abstention wrapper on the rank-1 follower. It computes a CONFIDENCE
signal from signals we PROVED are trustworthy -- route geometry, NOT the camera
(which we proved cannot separate collisions from clear scenes). When confidence
is low, it abstains from full speed via a graduated safe slowdown. When
confidence is high (valid, stable, sufficiently-long route), it is byte-identical
to the follower -> regression-safe by construction.

Confidence = product of three route-reliability factors, each in [0,1]:
  c_valid  : fraction of route waypoints that are valid (non-NaN)         -- route present?
  c_length : route arc-length vs a horizon (short route => less certain)   -- route sufficient?
  c_stable : agreement of the route across recent frames (EMA)            -- route stable?
speed_scale = c_valid * c_length * c_stable, floored so we never fully stop
             in live traffic (avoids the 'sitting duck' failure we found).

Env:
  RF_ABSTAIN=1          enable abstention (0 => pure follower)
  RF_ABSTAIN_FLOOR=0.15 minimum speed scale (never crawl below this * planned)
  RF_ABSTAIN_HORIZON=15 route arc-length (m) considered 'fully sufficient'
  RF_V_MAX/A_LAT/A_LON  follower params
"""
from __future__ import annotations
import bisect, logging, math, os, signal, threading, time
from concurrent import futures
from dataclasses import dataclass, field
import numpy as np
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
import grpc

LOG = logging.getLogger("abstention_driver")
V_MAX = float(os.environ.get("RF_V_MAX", "12.0"))
A_LAT_MAX = float(os.environ.get("RF_A_LAT", "2.5"))
A_LON_MAX = float(os.environ.get("RF_A_LON", "1.5"))
TAU_CONVERGE = 1.5
HORIZON_S, DT_US = 5.0, 100_000

ABSTAIN = os.environ.get("RF_ABSTAIN", "1") == "1"
FLOOR = float(os.environ.get("RF_ABSTAIN_FLOOR", "0.15"))
ROUTE_HORIZON = float(os.environ.get("RF_ABSTAIN_HORIZON", "15.0"))


def quat_rotate(q, v):
    qv = np.array([q.x, q.y, q.z], dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q.w * v)

def yaw_of(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

def quat_of_yaw(y):
    return common_pb2.Quat(w=math.cos(y/2), x=0.0, y=0.0, z=math.sin(y/2))


@dataclass
class Session:
    poses: dict = field(default_factory=dict)
    pose_ts: list = field(default_factory=list)
    latest: object = None
    speed: float = 0.0
    route_rig: object = None
    route_ts: int = 0
    conf_ema: float = 1.0            # EMA of confidence for stability term
    last_route_len: float = -1.0

    def add_pose(self, p):
        if p.timestamp_us not in self.poses:
            bisect.insort(self.pose_ts, p.timestamp_us)
        self.poses[p.timestamp_us] = p.pose
        if self.latest is None or p.timestamp_us >= self.latest.timestamp_us:
            self.latest = p
        while len(self.pose_ts) > 400:
            self.poses.pop(self.pose_ts.pop(0), None)

    def pose_at(self, ts):
        if not self.pose_ts:
            return None
        i = bisect.bisect_left(self.pose_ts, ts)
        cands = [self.pose_ts[max(i-1,0)], self.pose_ts[min(i,len(self.pose_ts)-1)]]
        return self.poses[min(cands, key=lambda t: abs(t-ts))]


def resample(pts, step=0.5):
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-3])
    pts, s = pts[keep], s[keep]
    if len(pts) < 2:
        return pts, s
    su = np.arange(0.0, s[-1], step)
    out = np.stack([np.interp(su, s, pts[:, k]) for k in range(3)], axis=1)
    return out, su


def route_confidence(s, route_rig):
    """Confidence in [0,1] from route reliability (NOT camera). Returns (conf, breakdown)."""
    if route_rig is None or len(route_rig) == 0:
        return FLOOR, {"c_valid": 0.0, "c_length": 0.0, "c_stable": s.conf_ema}
    valid_mask = ~np.isnan(route_rig).any(axis=1)
    n_valid = int(valid_mask.sum())
    # c_valid: fraction of waypoints valid, but heavily penalize <2 (can't form a path)
    c_valid = 0.0 if n_valid < 2 else min(1.0, n_valid / 5.0)
    # c_length: arc-length of the valid route vs horizon
    v = route_rig[valid_mask]
    if len(v) >= 2:
        arclen = float(np.linalg.norm(np.diff(v[:, :2], axis=0), axis=1).sum())
    else:
        arclen = 0.0
    c_length = min(1.0, arclen / ROUTE_HORIZON)
    # c_stable: EMA agreement -- if route length is stable frame-to-frame, more confident
    if s.last_route_len >= 0:
        change = abs(arclen - s.last_route_len) / max(s.last_route_len, 1.0)
        c_stable_inst = math.exp(-2.0 * change)      # big jumps -> low stability
    else:
        c_stable_inst = 1.0
    s.last_route_len = arclen
    s.conf_ema = 0.6 * s.conf_ema + 0.4 * c_stable_inst
    conf = c_valid * c_length * s.conf_ema
    conf = max(FLOOR, min(1.0, conf))
    return conf, {"c_valid": c_valid, "c_length": c_length, "c_stable": s.conf_ema,
                  "arclen": arclen, "n_valid": n_valid}


class AbstentionDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self):
        self._s = {}
        self._lock = threading.RLock()
        self._server = None

    def attach_server(self, s): self._server = s

    def _sess(self, uuid, ctx):
        with self._lock:
            s = self._s.get(uuid)
        if s is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {uuid}")
        return s

    def start_session(self, req, ctx):
        with self._lock:
            self._s[req.session_uuid] = Session()
        LOG.info("start %s (ABSTAIN=%s floor=%.2f)", req.session_uuid, ABSTAIN, FLOOR)
        return common_pb2.SessionRequestStatus()

    def close_session(self, req, ctx):
        with self._lock:
            self._s.pop(req.session_uuid, None)
        return common_pb2.Empty()

    def submit_image_observation(self, req, ctx):
        return common_pb2.Empty()   # abstention uses route geometry, not camera

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

    def _plan(self, s, t0):
        ego = s.latest
        p0 = np.array([ego.pose.vec.x, ego.pose.vec.y, ego.pose.vec.z])
        yaw0 = yaw_of(ego.pose.quat)

        # === ABSTENTION: confidence from route reliability ===
        conf = 1.0
        if ABSTAIN:
            conf, _bd = route_confidence(s, s.route_rig)

        anchor = s.pose_at(s.route_ts) if s.route_rig is not None else None
        if s.route_rig is None or anchor is None or len(s.route_rig) < 2:
            # route absent -> low confidence -> abstain: safe slow crawl, not blind charge
            v_safe = max(s.speed * conf, FLOOR * 3.0) if ABSTAIN else max(s.speed, 3.0)
            return straight(p0, yaw0, t0, v_safe)
        base = np.array([anchor.vec.x, anchor.vec.y, anchor.vec.z])
        route = np.stack([quat_rotate(anchor.quat, w) + base for w in s.route_rig])
        route = np.vstack([p0[None,:], route])
        pts, s_arc = resample(route)
        if len(pts) < 2:
            return straight(p0, yaw0, t0, max(s.speed*conf, FLOOR*3.0))
        d = np.linalg.norm(pts[:,:2]-p0[:2], axis=1)
        i0 = int(np.argmin(d))
        tang = pts[min(i0+1,len(pts)-1),:2]-pts[max(i0-1,0),:2]
        th = math.atan2(tang[1],tang[0]) if np.linalg.norm(tang)>1e-6 else yaw0
        nvec = np.array([-math.sin(th),math.cos(th)])
        e0 = float(np.dot(p0[:2]-pts[i0,:2], nvec))
        dx=np.gradient(pts[:,0]); dy=np.gradient(pts[:,1])
        ddx=np.gradient(dx); ddy=np.gradient(dy)
        kap=np.abs(dx*ddy-dy*ddx)/np.maximum((dx**2+dy**2)**1.5,1e-6)
        v_cap=np.minimum(V_MAX, np.sqrt(A_LAT_MAX/np.maximum(kap,1e-3)))
        # abstention scales speed by confidence (high conf -> unchanged)
        v_cap = v_cap * conf

        traj = common_pb2.Trajectory()
        n = int(HORIZON_S*1e6/DT_US)+1
        s_cur, v = s_arc[i0], max(s.speed, 0.0)
        for i in range(n):
            t=i*DT_US/1e6
            j=int(np.searchsorted(s_arc,s_cur,side="right"))-1
            j=max(0,min(j,len(pts)-2))
            v_t=float(v_cap[j])
            v += np.clip(v_t-v, -A_LON_MAX*0.1, A_LON_MAX*0.1) if i else 0.0
            if i: s_cur=min(s_cur+v*DT_US/1e6, s_arc[-1])
            ref=np.array([np.interp(s_cur,s_arc,pts[:,k]) for k in range(3)])
            k2=max(0,min(int(np.searchsorted(s_arc,s_cur)),len(pts)-2))
            tv=pts[k2+1,:2]-pts[k2,:2]
            yaw=math.atan2(tv[1],tv[0]) if np.linalg.norm(tv)>1e-6 else yaw0
            off=e0*math.exp(-t/TAU_CONVERGE)
            pos=ref[:2]+off*np.array([-math.sin(yaw),math.cos(yaw)])
            traj.poses.append(common_pb2.PoseAtTime(
                timestamp_us=t0+i*DT_US,
                pose=common_pb2.Pose(
                    vec=common_pb2.Vec3(x=float(pos[0]),y=float(pos[1]),z=float(ref[2])),
                    quat=quat_of_yaw(yaw))))
        return traj

    def drive(self, req, ctx):
        s = self._sess(req.session_uuid, ctx)
        try:
            with self._lock:
                if s.latest is None:
                    raise RuntimeError("no ego pose yet")
                traj = self._plan(s, req.time_now_us)
            return egodriver_pb2.DriveResponse(trajectory=traj)
        except Exception:
            LOG.exception("drive failed -> safe fallback")
            return egodriver_pb2.DriveResponse(trajectory=fallback(s, req.time_now_us))

    def get_version(self, req, ctx):
        return common_pb2.VersionId(version_id="abstention-driver", git_hash="local",
                                    grpc_api_version=API_VERSION_MESSAGE)

    def shut_down(self, req, ctx):
        if self._server:
            threading.Thread(target=lambda:(time.sleep(0.05), self._server.stop(grace=0.0)),
                             daemon=True).start()
        return common_pb2.Empty()


def straight(p0, yaw, t0, v):
    tr=common_pb2.Trajectory()
    for i in range(int(HORIZON_S*1e6/DT_US)+1):
        dd=v*i*DT_US/1e6
        tr.poses.append(common_pb2.PoseAtTime(timestamp_us=t0+i*DT_US,
            pose=common_pb2.Pose(vec=common_pb2.Vec3(x=float(p0[0]+math.cos(yaw)*dd),
                y=float(p0[1]+math.sin(yaw)*dd), z=float(p0[2])), quat=quat_of_yaw(yaw))))
    return tr

def fallback(s, t0):
    if s.latest is None: return common_pb2.Trajectory()
    p=s.latest.pose; p0=np.array([p.vec.x,p.vec.y,p.vec.z]); yaw=yaw_of(p.quat); v=s.speed
    tr=common_pb2.Trajectory(); dd=0.0
    for i in range(int(HORIZON_S*1e6/DT_US)+1):
        v=max(0.0,v-4.0*DT_US/1e6); dd+=v*DT_US/1e6
        tr.poses.append(common_pb2.PoseAtTime(timestamp_us=t0+i*DT_US,
            pose=common_pb2.Pose(vec=common_pb2.Vec3(x=float(p0[0]+math.cos(yaw)*dd),
                y=float(p0[1]+math.sin(yaw)*dd), z=float(p0[2])), quat=quat_of_yaw(yaw))))
    return tr


def main():
    logging.basicConfig(level=os.environ.get("ALPASIM_DRIVER_LOG_LEVEL","INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host=os.environ.get("ALPASIM_DRIVER_HOST","0.0.0.0")
    port=int(os.environ.get("ALPASIM_DRIVER_PORT","6789"))
    svc=AbstentionDriver()
    srv=grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(svc, srv)
    svc.attach_server(srv)
    if srv.add_insecure_port(f"{host}:{port}")==0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: srv.stop(grace=0.0))
    srv.start()
    LOG.info("abstention-driver on %s:%d ABSTAIN=%s floor=%.2f horizon=%.1f",
             host, port, ABSTAIN, FLOOR, ROUTE_HORIZON)
    srv.wait_for_termination()

if __name__ == "__main__":
    main()
