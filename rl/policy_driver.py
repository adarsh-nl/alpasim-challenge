#!/usr/bin/env python
"""Milestone 2: gRPC egodriver that drives via the residual speed-control policy.

Reuses the rank-1 route_follower geometry EXACTLY (route projection, curvature
v_cap, lateral convergence, safe fallback). The ONE behavioral change: the
planned speed profile is scaled by a learned per-Drive-call multiplier m in (0,1]
produced by ResidualSpeedPolicy from the live CAM_F0 frame.

Also logs, per Drive call, the (observation, pre-squash action, logprob, value)
tuple to an .npz under $RL_LOG_DIR -- the transitions PPO needs. Logging is
append-in-memory, flushed on close_session.

Env:
  RL_POLICY_CKPT   path to policy state_dict (.pt). If unset, uses random init
                   (near-full-speed prior) -- fine for the M2 smoke.
  RL_LOG_DIR       dir to write per-session transition logs (default: no logging)
  RL_DETERMINISTIC 1 => m=sigmoid(mean) (eval);  0 => sample (rollout). default 0
  RF_V_MAX/A_LAT/A_LON  same as the follower
"""
from __future__ import annotations
import bisect, logging, math, os, signal, threading, time
from concurrent import futures
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import io
import torch

from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
import grpc

# policy net lives alongside this file
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_policy import ResidualSpeedPolicy

LOG = logging.getLogger("policy_driver")
V_MAX = float(os.environ.get("RF_V_MAX", "12.0"))
A_LAT_MAX = float(os.environ.get("RF_A_LAT", "2.5"))
A_LON_MAX = float(os.environ.get("RF_A_LON", "1.5"))
TAU_CONVERGE = 1.5
HORIZON_S, DT_US = 5.0, 100_000
N_ROUTE = 10
IMG_HW = (180, 320)

DETERMINISTIC = os.environ.get("RL_DETERMINISTIC", "0") == "1"
LOG_DIR = os.environ.get("RL_LOG_DIR", "")
CKPT = os.environ.get("RL_POLICY_CKPT", "")

_DEV = "cuda" if torch.cuda.is_available() else "cpu"


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
    last_img: object = None            # most recent CAM_F0 as (3,H,W) float tensor
    # transition log
    log_img: list = field(default_factory=list)
    log_route: list = field(default_factory=list)
    log_ego: list = field(default_factory=list)
    log_pre: list = field(default_factory=list)
    log_logp: list = field(default_factory=list)
    log_val: list = field(default_factory=list)
    log_t: list = field(default_factory=list)

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


class PolicyDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self):
        self._s = {}
        self._lock = threading.RLock()
        self._server = None
        self.policy = ResidualSpeedPolicy(n_route=N_ROUTE).to(_DEV).eval()
        if CKPT and os.path.exists(CKPT):
            sd = torch.load(CKPT, map_location=_DEV)
            self.policy.load_state_dict(sd)
            LOG.info("loaded policy ckpt %s", CKPT)
        else:
            LOG.info("no ckpt -> random-init policy (near-full-speed prior)")

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
        LOG.info("start %s (det=%s log=%s)", req.session_uuid, DETERMINISTIC, bool(LOG_DIR))
        return common_pb2.SessionRequestStatus()

    def close_session(self, req, ctx):
        with self._lock:
            s = self._s.pop(req.session_uuid, None)
        if s is not None and LOG_DIR and s.log_img:
            self._flush(req.session_uuid, s)
        return common_pb2.Empty()

    def submit_image_observation(self, req, ctx):
        # decode CAM_F0 -> (3,H,W) float tensor in [0,1], stash on session
        s = self._sess(req.session_uuid, ctx)
        try:
            for f, _ in req.ListFields():
                pass
            ci = getattr(req, 'camera_image', None) or getattr(req, 'image', None)
            cam = getattr(req, 'camera_image', None)
            # req may wrap the image; try common shapes
            img_bytes = None; logical = None
            if hasattr(req, 'camera_image') and req.camera_image.image_bytes:
                img_bytes = req.camera_image.image_bytes
                logical = req.camera_image.logical_id
            if img_bytes and logical == 'CAM_F0':
                im = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((IMG_HW[1], IMG_HW[0]))
                arr = np.asarray(im, dtype=np.float32) / 255.0     # H,W,3
                s.last_img = torch.from_numpy(arr).permute(2,0,1).contiguous()  # 3,H,W
        except Exception:
            LOG.debug("img decode skipped", exc_info=True)
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

    def _policy_inputs(self, s, pts, s_arc, i0, p0):
        """Build (img, route, ego) tensors for the policy from current state."""
        # image
        if s.last_img is not None:
            img = s.last_img.unsqueeze(0).to(_DEV)
        else:
            img = torch.zeros(1, 3, *IMG_HW, device=_DEV)
        # route: next N_ROUTE points ahead of i0, in ego frame (x fwd, y left), + mask
        route = np.zeros((N_ROUTE, 3), dtype=np.float32)   # x,y,mask
        yaw0 = yaw_of(s.latest.pose.quat)
        c, sn = math.cos(-yaw0), math.sin(-yaw0)
        for k in range(N_ROUTE):
            j = i0 + k
            if j < len(pts):
                dx, dy = pts[j,0]-p0[0], pts[j,1]-p0[1]
                route[k,0] = c*dx - sn*dy
                route[k,1] = sn*dx + c*dy
                route[k,2] = 1.0
        route_t = torch.from_numpy(route).unsqueeze(0).to(_DEV)
        # ego: speed + normalized
        ego = torch.tensor([[s.speed/ max(V_MAX,1.0), 0.0, 0.0]], dtype=torch.float32, device=_DEV)
        return img, route_t, ego

    def _plan(self, s, t0):
        ego = s.latest
        p0 = np.array([ego.pose.vec.x, ego.pose.vec.y, ego.pose.vec.z])
        yaw0 = yaw_of(ego.pose.quat)
        anchor = s.pose_at(s.route_ts) if s.route_rig is not None else None
        if s.route_rig is None or anchor is None or len(s.route_rig) < 2:
            return straight(p0, yaw0, t0, max(s.speed, 3.0))
        base = np.array([anchor.vec.x, anchor.vec.y, anchor.vec.z])
        route = np.stack([quat_rotate(anchor.quat, w) + base for w in s.route_rig])
        route = np.vstack([p0[None,:], route])
        pts, s_arc = resample(route)
        if len(pts) < 2:
            return straight(p0, yaw0, t0, max(s.speed, 3.0))
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

        # === POLICY: speed multiplier from camera ===
        img, route_t, ego_t = self._policy_inputs(s, pts, s_arc, i0, p0)
        m, logp, val = self.policy.act(img, route_t, ego_t, deterministic=DETERMINISTIC)
        m_val = float(m.item())
        v_cap = v_cap * m_val                       # the residual: scale planned speed

        # log transition
        if LOG_DIR:
            # recover pre-squash action for PPO (invert sigmoid)
            pre = math.log(max(min(m_val,1-1e-6),1e-6)/(1-max(min(m_val,1-1e-6),1e-6)))
            s.log_img.append((s.last_img.numpy()*255).astype(np.uint8) if s.last_img is not None
                             else np.zeros((3,*IMG_HW),np.uint8))
            s.log_route.append(route_t.squeeze(0).cpu().numpy())
            s.log_ego.append(ego_t.squeeze(0).cpu().numpy())
            s.log_pre.append(pre)
            s.log_logp.append(float(logp.item()))
            s.log_val.append(float(val.item()))
            s.log_t.append(int(t0))

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

    def _flush(self, uuid, s):
        os.makedirs(LOG_DIR, exist_ok=True)
        out = os.path.join(LOG_DIR, uuid + ".npz")
        np.savez_compressed(out,
            img=np.array(s.log_img, dtype=np.uint8),
            route=np.array(s.log_route, dtype=np.float32),
            ego=np.array(s.log_ego, dtype=np.float32),
            pre_action=np.array(s.log_pre, dtype=np.float32),
            logprob=np.array(s.log_logp, dtype=np.float32),
            value=np.array(s.log_val, dtype=np.float32),
            t_us=np.array(s.log_t, dtype=np.int64))
        LOG.info("flushed %d transitions -> %s", len(s.log_pre), out)

    def get_version(self, req, ctx):
        return common_pb2.VersionId(version_id="policy-driver", git_hash="local",
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
    svc=PolicyDriver()
    srv=grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(svc, srv)
    svc.attach_server(srv)
    if srv.add_insecure_port(f"{host}:{port}")==0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: srv.stop(grace=0.0))
    srv.start()
    LOG.info("policy-driver on %s:%d dev=%s V_MAX=%.1f det=%s", host, port, _DEV, V_MAX, DETERMINISTIC)
    srv.wait_for_termination()

if __name__ == "__main__":
    main()
