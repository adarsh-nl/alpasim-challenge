#!/usr/bin/env python
"""Regression tests for the Drive-handler fallback fix (shipped route_follower).

The bug: drive() resolved the session with _sess(), which calls ctx.abort() --
and abort() RAISES. Because the lookup sat outside the try, a Drive for a
session that was never opened, or one arriving after close_session, escaped the
handler with an exception instead of a trajectory. A dead Drive handler means
the controller gets no plan for that tick, which is how a scene becomes a
DriveIRT zero.

What this file locks in:
  1. the happy path is IEEE-754 BIT-IDENTICAL to the shipped baseline, so the
     rank-1 driver is provably unchanged on every scene that does not hit the
     crash path;
  2. an unknown-session Drive and a post-close Drive both return a SAFE STOP,
     not a raise;
  3. the observation RPCs STILL abort NOT_FOUND on an unknown session -- that is
     correct protocol signalling and is deliberately not part of this fix.

Self-contained: no simulator, no renderer, no GPU, no model. Real alpasim_grpc
protos when importable, otherwise an in-process stub, so it runs anywhere.

    ~/alpasim/.venv-driver/bin/python -m pytest tests/test_drive_handler_fallback.py -q
"""
from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHIPPED = "submission/route_follower.py"
# The shipped baseline this branch was cut from. Happy-path output must match it.
BASELINE_REF = "main"


# ---------------------------------------------------------------------------
# proto layer: real protos if available, else a minimal stub
# ---------------------------------------------------------------------------

def _install_proto_stub() -> None:
    class Vec3:
        __slots__ = ("x", "y", "z")

        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class Quat:
        __slots__ = ("w", "x", "y", "z")

        def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
            self.w, self.x, self.y, self.z = float(w), float(x), float(y), float(z)

    class Pose:
        __slots__ = ("vec", "quat")

        def __init__(self, vec=None, quat=None):
            self.vec = vec if vec is not None else Vec3()
            self.quat = quat if quat is not None else Quat()

    class PoseAtTime:
        __slots__ = ("timestamp_us", "pose")

        def __init__(self, timestamp_us=0, pose=None):
            self.timestamp_us = int(timestamp_us)
            self.pose = pose if pose is not None else Pose()

    class Trajectory:
        __slots__ = ("poses",)

        def __init__(self, poses=None):
            self.poses = list(poses) if poses else []

    class _Bare:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    common_pb2 = types.ModuleType("alpasim_grpc.v0.common_pb2")
    common_pb2.Vec3, common_pb2.Quat = Vec3, Quat
    common_pb2.Pose, common_pb2.PoseAtTime = Pose, PoseAtTime
    common_pb2.Trajectory = Trajectory
    common_pb2.Empty = common_pb2.SessionRequestStatus = _Bare
    common_pb2.VersionId = _Bare

    egodriver_pb2 = types.ModuleType("alpasim_grpc.v0.egodriver_pb2")

    class DriveResponse:
        __slots__ = ("trajectory",)

        def __init__(self, trajectory=None):
            self.trajectory = trajectory if trajectory is not None else Trajectory()

    egodriver_pb2.DriveResponse = DriveResponse

    egodriver_pb2_grpc = types.ModuleType("alpasim_grpc.v0.egodriver_pb2_grpc")
    egodriver_pb2_grpc.EgodriverServiceServicer = type("Servicer", (), {})
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server = lambda *a, **k: None

    v0 = types.ModuleType("alpasim_grpc.v0")
    v0.common_pb2, v0.egodriver_pb2 = common_pb2, egodriver_pb2
    v0.egodriver_pb2_grpc = egodriver_pb2_grpc

    root = types.ModuleType("alpasim_grpc")
    root.API_VERSION_MESSAGE, root.v0 = "stub", v0

    grpc_mod = types.ModuleType("grpc")
    grpc_mod.StatusCode = type("StatusCode", (), {"NOT_FOUND": "NOT_FOUND"})
    grpc_mod.RpcError = type("RpcError", (Exception,), {})
    grpc_mod.server = lambda *a, **k: None

    for name, mod in [("alpasim_grpc", root), ("alpasim_grpc.v0", v0),
                      ("alpasim_grpc.v0.common_pb2", common_pb2),
                      ("alpasim_grpc.v0.egodriver_pb2", egodriver_pb2),
                      ("alpasim_grpc.v0.egodriver_pb2_grpc", egodriver_pb2_grpc)]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("grpc", grpc_mod)


try:
    import alpasim_grpc  # noqa: F401
    import grpc  # noqa: F401
except Exception:  # pragma: no cover
    _install_proto_stub()


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

class Ctx:
    """Stand-in for grpc.ServicerContext. abort() raises, like the real one."""

    def __init__(self):
        self.aborted = None

    def abort(self, code, details):
        self.aborted = (code, details)
        raise RuntimeError(f"aborted: {code}: {details}")


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def vec(x, y, z=0.0):
    return NS(x=float(x), y=float(y), z=float(z))


def quat_yaw(yaw):
    return NS(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))


def pose_at(ts, x, y, yaw, z=0.0):
    return NS(timestamp_us=int(ts), pose=NS(vec=vec(x, y, z), quat=quat_yaw(yaw)))


def load_module(path: Path, alias: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def canon(traj) -> bytes:
    """Exact IEEE-754 fingerprint of a trajectory -- no tolerance."""
    out = bytearray()
    for p in traj.poses:
        out += struct.pack("<q", int(p.timestamp_us))
        v, q = p.pose.vec, p.pose.quat
        out += struct.pack("<7d", float(v.x), float(v.y), float(v.z),
                           float(q.w), float(q.x), float(q.y), float(q.z))
    return bytes(out)


# Six valid-session rollout profiles, deterministic, covering the branches the
# happy path can take: straight, curved, short route, and the degenerate-route
# branch that falls through to straight().
PROFILES = [
    ("straight_fast", 0.0, 14, 11.0),
    ("straight_slow", 0.0, 14, 3.0),
    ("curve_left", 0.05, 14, 7.0),
    ("curve_right", -0.05, 14, 7.0),
    ("short_route", 0.02, 3, 6.0),
    ("degenerate_route", 0.0, 1, 5.0),   # <2 waypoints -> straight() branch
]


def rollout_calls(uuid: str, curve: float, n_wp: int, speed: float, n_steps: int = 10):
    """Deterministic RPC sequence for one valid session."""
    dt_us = 500_000
    yield ("start_session", NS(session_uuid=uuid, random_seed=abs(hash(uuid)) % 2**31))
    x = y = yaw = 0.0
    for step in range(n_steps):
        t = 1_000_000 + step * dt_us
        v = speed + 0.5 * math.sin(step * 0.7)
        x += v * (dt_us / 1e6) * math.cos(yaw)
        y += v * (dt_us / 1e6) * math.sin(yaw)
        yaw += curve
        poses = [pose_at(t - dt_us, x - v * 0.5, y, yaw), pose_at(t, x, y, yaw)]
        yield ("submit_egomotion_observation",
               NS(session_uuid=uuid, trajectory=NS(poses=poses),
                  dynamic_states=[NS(linear_velocity=vec(v * math.cos(yaw),
                                                         v * math.sin(yaw)))
                                  for _ in poses]))
        wps, wx, wy, wth = [], 0.0, 0.0, 0.0
        for _ in range(n_wp):
            wth += curve
            wx += 2.0 * math.cos(wth)
            wy += 2.0 * math.sin(wth)
            wps.append(vec(wx, wy))
        yield ("submit_route",
               NS(session_uuid=uuid, route=NS(waypoints=wps, timestamp_us=t)))
        yield ("drive", NS(session_uuid=uuid, time_now_us=t, time_query_us=t))
    yield ("close_session", NS(session_uuid=uuid))


def run_all_rollouts(mod) -> dict[str, list[bytes]]:
    """Drive every profile through a fresh servicer; collect trajectory bytes."""
    out = {}
    for name, curve, n_wp, speed in PROFILES:
        servicer = mod.RouteFollower()
        ctx = Ctx()
        seq = []
        for rpc, req in rollout_calls(name, curve, n_wp, speed):
            resp = getattr(servicer, rpc)(req, ctx)
            if rpc == "drive":
                seq.append(canon(resp.trajectory))
        out[name] = seq
    return out


@pytest.fixture(scope="module")
def fixed_mod():
    return load_module(REPO / SHIPPED, "_fixed_rf")


@pytest.fixture(scope="module")
def baseline_mod(tmp_path_factory):
    """The shipped driver as it exists at BASELINE_REF, extracted via git."""
    try:
        src = subprocess.run(["git", "show", f"{BASELINE_REF}:{SHIPPED}"],
                             cwd=REPO, capture_output=True, check=True).stdout
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot extract baseline {BASELINE_REF}:{SHIPPED}: {e}")
    p = tmp_path_factory.mktemp("baseline") / "route_follower_baseline.py"
    p.write_bytes(src)
    return load_module(p, "_baseline_rf")


# ---------------------------------------------------------------------------
# 1. happy path is bit-identical to the shipped baseline
# ---------------------------------------------------------------------------

def test_happy_path_bit_identical_to_baseline(fixed_mod, baseline_mod):
    """>=6 valid-session rollouts: every trajectory must match the baseline.

    This is the whole safety argument for shipping the fix: on any scene that
    never hits the crash path, the rank-1 driver is byte-for-byte unchanged.
    """
    base = run_all_rollouts(baseline_mod)
    fixed = run_all_rollouts(fixed_mod)
    assert len(base) >= 6, "need at least 6 rollouts for the identity claim"
    assert set(base) == set(fixed)
    n_traj = sum(len(v) for v in base.values())
    assert n_traj >= 60, f"only {n_traj} trajectories compared"
    for name in base:
        assert fixed[name] == base[name], (
            f"happy-path divergence in profile '{name}': the fix changed "
            f"driving behaviour, which it must not")


def test_baseline_actually_raises_on_unknown_session(baseline_mod):
    """Negative control: prove the bug was real in the shipped baseline."""
    servicer = baseline_mod.RouteFollower()
    with pytest.raises(Exception):
        servicer.drive(NS(session_uuid="never-opened", time_now_us=1_000_000,
                          time_query_us=1_000_000), Ctx())


# ---------------------------------------------------------------------------
# 2. the crash path now returns a safe stop
# ---------------------------------------------------------------------------

def _assert_is_safe_stop(traj, *, expect_poses: bool):
    """A safe stop: monotonic forward motion, non-increasing steps, ends halted."""
    n = len(traj.poses)
    if not expect_poses:
        assert n == 0, (
            "with no ego pose ever received we cannot know where the car is; "
            f"the only honest output is an empty trajectory, got {n} poses")
        return
    assert n > 1, f"expected a stopping trajectory, got {n} poses"
    xs = [(p.pose.vec.x, p.pose.vec.y) for p in traj.poses]
    steps = [math.dist(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
    assert all(s >= -1e-12 for s in steps), "trajectory moves backwards"
    for a, b in zip(steps, steps[1:]):
        assert b <= a + 1e-9, "speed increases mid-fallback: not a deceleration"
    assert steps[-1] <= 1e-9, (
        f"fallback never reaches a stop; final step {steps[-1]:.6f} m")
    ts = [p.timestamp_us for p in traj.poses]
    assert ts == sorted(ts), "timestamps not monotonic"


def test_unknown_session_drive_returns_safe_stop_not_raise(fixed_mod):
    """(a) A Drive for a session that was never opened must not raise."""
    servicer = fixed_mod.RouteFollower()
    resp = servicer.drive(NS(session_uuid="never-opened", time_now_us=1_000_000,
                             time_query_us=1_000_000), Ctx())
    assert resp is not None and hasattr(resp, "trajectory")
    # never opened -> no pose was ever received -> empty is the honest answer
    _assert_is_safe_stop(resp.trajectory, expect_poses=False)


def test_post_close_drive_returns_safe_stop_not_raise(fixed_mod):
    """(b) A late Drive after close_session must emit a real stopping plan."""
    servicer = fixed_mod.RouteFollower()
    ctx = Ctx()
    for rpc, req in rollout_calls("closed-one", 0.0, 14, 9.0, n_steps=6):
        getattr(servicer, rpc)(req, ctx)
    resp = servicer.drive(NS(session_uuid="closed-one", time_now_us=99_000_000,
                             time_query_us=99_000_000), ctx)
    assert resp is not None and hasattr(resp, "trajectory")
    _assert_is_safe_stop(resp.trajectory, expect_poses=True)


def test_post_close_stop_starts_from_last_known_pose(fixed_mod):
    """The stop is anchored at the last pose we saw, not at the origin."""
    servicer = fixed_mod.RouteFollower()
    ctx = Ctx()
    calls = list(rollout_calls("anchored", 0.0, 14, 9.0, n_steps=6))
    last_pose = None
    for rpc, req in calls:
        if rpc == "submit_egomotion_observation":
            last_pose = req.trajectory.poses[-1]
        getattr(servicer, rpc)(req, ctx)
    resp = servicer.drive(NS(session_uuid="anchored", time_now_us=99_000_000,
                             time_query_us=99_000_000), ctx)
    first = resp.trajectory.poses[0]
    d = math.dist((first.pose.vec.x, first.pose.vec.y),
                  (last_pose.pose.vec.x, last_pose.pose.vec.y))
    assert d < 1.0, (
        f"stop trajectory starts {d:.2f} m from the last known pose -- "
        f"that is a teleport, not a stop")


def test_drive_before_first_pose_returns_trajectory(fixed_mod):
    """Session open but no egomotion yet: fallback, not an exception."""
    servicer = fixed_mod.RouteFollower()
    ctx = Ctx()
    servicer.start_session(NS(session_uuid="s", random_seed=1), ctx)
    resp = servicer.drive(NS(session_uuid="s", time_now_us=1_000_000,
                             time_query_us=1_000_000), ctx)
    assert resp is not None and hasattr(resp, "trajectory")


def test_repeated_late_drives_are_stable(fixed_mod):
    """Many late Drives keep returning the same safe stop (no state drift)."""
    servicer = fixed_mod.RouteFollower()
    ctx = Ctx()
    for rpc, req in rollout_calls("repeat", 0.0, 14, 8.0, n_steps=5):
        getattr(servicer, rpc)(req, ctx)
    seen = {canon(servicer.drive(
        NS(session_uuid="repeat", time_now_us=50_000_000, time_query_us=50_000_000),
        ctx).trajectory) for _ in range(20)}
    assert len(seen) == 1, "late-Drive output drifts between calls"


# ---------------------------------------------------------------------------
# 3. observation RPCs must STILL abort NOT_FOUND
# ---------------------------------------------------------------------------

def test_observation_rpcs_still_abort_not_found(fixed_mod):
    """(c) Lock in the protocol behaviour this fix must NOT change.

    Only the Drive path got a non-aborting lookup. The observation RPCs that
    resolve a session keep aborting NOT_FOUND -- that is correct signalling to
    the runtime that it addressed a session we do not have.
    """
    servicer = fixed_mod.RouteFollower()
    cases = {
        "submit_egomotion_observation": NS(
            session_uuid="ghost",
            trajectory=NS(poses=[pose_at(1_000_000, 0.0, 0.0, 0.0)]),
            dynamic_states=[NS(linear_velocity=vec(5.0, 0.0))]),
        "submit_route": NS(
            session_uuid="ghost",
            route=NS(waypoints=[vec(2.0, 0.0), vec(4.0, 0.0)], timestamp_us=1_000_000)),
    }
    for rpc, req in cases.items():
        ctx = Ctx()
        with pytest.raises(RuntimeError, match="aborted"):
            getattr(servicer, rpc)(req, ctx)
        assert ctx.aborted is not None, f"{rpc} did not call ctx.abort()"
        assert "NOT_FOUND" in str(ctx.aborted[0]), (
            f"{rpc} aborted with {ctx.aborted[0]}, expected NOT_FOUND")


def test_observation_rpcs_still_abort_after_close(fixed_mod):
    """A tombstone must not make observation RPCs start succeeding."""
    servicer = fixed_mod.RouteFollower()
    ctx = Ctx()
    for rpc, req in rollout_calls("gone", 0.0, 14, 7.0, n_steps=4):
        getattr(servicer, rpc)(req, ctx)
    ctx2 = Ctx()
    with pytest.raises(RuntimeError, match="aborted"):
        servicer.submit_route(
            NS(session_uuid="gone",
               route=NS(waypoints=[vec(2.0, 0.0), vec(4.0, 0.0)],
                        timestamp_us=9_000_000)), ctx2)
    assert "NOT_FOUND" in str(ctx2.aborted[0])


def test_cameras_and_gt_remain_unconditional_noops(fixed_mod):
    """These never looked up a session and must stay that way."""
    servicer = fixed_mod.RouteFollower()
    ctx = Ctx()
    assert servicer.submit_image_observation(NS(session_uuid="ghost"), ctx) is not None
    assert servicer.submit_recording_ground_truth(NS(session_uuid="ghost"), ctx) is not None


# ---------------------------------------------------------------------------
# 4. the tombstone must not become a leak
# ---------------------------------------------------------------------------

def test_tombstones_are_bounded(fixed_mod):
    """Closed-session stop states are capped, and live state is still evicted."""
    servicer = fixed_mod.RouteFollower()
    ctx = Ctx()
    for i in range(200):
        for rpc, req in rollout_calls(f"churn-{i}", 0.0, 14, 6.0, n_steps=2):
            getattr(servicer, rpc)(req, ctx)
    assert len(servicer._s) == 0, "live sessions leaked"
    assert len(servicer._closed) <= fixed_mod.MAX_TOMBSTONES, (
        f"tombstones grew to {len(servicer._closed)}, "
        f"cap is {fixed_mod.MAX_TOMBSTONES}")


def test_restarting_a_session_id_clears_its_tombstone(fixed_mod):
    """A reopened id must resolve to the live session, never the stale ghost."""
    servicer = fixed_mod.RouteFollower()
    ctx = Ctx()
    for rpc, req in rollout_calls("reused", 0.0, 14, 8.0, n_steps=4):
        getattr(servicer, rpc)(req, ctx)
    assert "reused" in servicer._closed
    servicer.start_session(NS(session_uuid="reused", random_seed=2), ctx)
    assert "reused" not in servicer._closed, "stale tombstone survived restart"
    assert "reused" in servicer._s


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
