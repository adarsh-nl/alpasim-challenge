#!/usr/bin/env python
"""Per-rollout state-isolation harness + regression tests for the egodriver.

Eval topology: 16 replica PROCESSES x 2 concurrent rollouts per process. Any
mutable state shared across rollouts inside one process bleeds between them and
can produce a wrong trajectory -> collision/offroad -> a DriveIRT zero.

This harness proves (or disproves) strict per-rollout isolation WITHOUT a GPU,
without the simulator, and without the renderer:

  * requests are duck-typed stand-ins (the drivers only read attributes),
  * responses are real protobuf when `alpasim_grpc` is importable, else a
    minimal in-process stub, so the file runs in any local env,
  * any model forward is replaced by a deterministic stub (see the RNG tests).

Core assertion: for every rollout, the full sequence of emitted trajectories is
BIT-IDENTICAL whether that rollout ran alone in a fresh servicer or interleaved
with other rollouts in one servicer.

Run:
    ~/alpasim/.venv-driver/bin/python -m pytest tests/test_state_isolation.py -q
"""
from __future__ import annotations

import itertools
import math
import os
import random
import struct
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. proto layer: real alpasim_grpc if present, else a minimal stub
# ---------------------------------------------------------------------------

def _install_proto_stub() -> None:
    """Register a minimal alpasim_grpc/grpc in sys.modules.

    Only the handful of messages the drivers construct on the response path are
    needed. Field semantics match the real protos (see alpasim_grpc/v0/*.proto).
    """
    class _Msg:
        __slots__ = ()

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class Vec3(_Msg):
        __slots__ = ("x", "y", "z")

        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class Quat(_Msg):
        __slots__ = ("w", "x", "y", "z")

        def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
            self.w, self.x, self.y, self.z = float(w), float(x), float(y), float(z)

    class Pose(_Msg):
        __slots__ = ("vec", "quat")

        def __init__(self, vec=None, quat=None):
            self.vec = vec if vec is not None else Vec3()
            self.quat = quat if quat is not None else Quat()

    class PoseAtTime(_Msg):
        __slots__ = ("timestamp_us", "pose")

        def __init__(self, timestamp_us=0, pose=None):
            self.timestamp_us = int(timestamp_us)
            self.pose = pose if pose is not None else Pose()

    class Trajectory(_Msg):
        __slots__ = ("poses",)

        def __init__(self, poses=None):
            self.poses = list(poses) if poses else []

    class Empty(_Msg):
        __slots__ = ()

    class SessionRequestStatus(_Msg):
        __slots__ = ()

    class VersionId(_Msg):
        __slots__ = ("version_id", "git_hash", "grpc_api_version")

        def __init__(self, version_id="", git_hash="", grpc_api_version=""):
            self.version_id, self.git_hash = version_id, git_hash
            self.grpc_api_version = grpc_api_version

    common_pb2 = types.ModuleType("alpasim_grpc.v0.common_pb2")
    for _n, _c in [("Vec3", Vec3), ("Quat", Quat), ("Pose", Pose),
                   ("PoseAtTime", PoseAtTime), ("Trajectory", Trajectory),
                   ("Empty", Empty), ("SessionRequestStatus", SessionRequestStatus),
                   ("VersionId", VersionId)]:
        setattr(common_pb2, _n, _c)

    class DriveResponse(_Msg):
        __slots__ = ("trajectory",)

        def __init__(self, trajectory=None):
            self.trajectory = trajectory if trajectory is not None else Trajectory()

    egodriver_pb2 = types.ModuleType("alpasim_grpc.v0.egodriver_pb2")
    egodriver_pb2.DriveResponse = DriveResponse

    egodriver_pb2_grpc = types.ModuleType("alpasim_grpc.v0.egodriver_pb2_grpc")

    class EgodriverServiceServicer:
        pass

    egodriver_pb2_grpc.EgodriverServiceServicer = EgodriverServiceServicer
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server = lambda *a, **k: None

    v0 = types.ModuleType("alpasim_grpc.v0")
    v0.common_pb2, v0.egodriver_pb2, v0.egodriver_pb2_grpc = (
        common_pb2, egodriver_pb2, egodriver_pb2_grpc)

    root = types.ModuleType("alpasim_grpc")
    root.API_VERSION_MESSAGE = "stub"
    root.v0 = v0

    grpc_mod = types.ModuleType("grpc")

    class _Status:
        NOT_FOUND = "NOT_FOUND"
        INTERNAL = "INTERNAL"

    grpc_mod.StatusCode = _Status
    grpc_mod.RpcError = type("RpcError", (Exception,), {})
    grpc_mod.server = lambda *a, **k: None

    for name, mod in [("alpasim_grpc", root), ("alpasim_grpc.v0", v0),
                      ("alpasim_grpc.v0.common_pb2", common_pb2),
                      ("alpasim_grpc.v0.egodriver_pb2", egodriver_pb2),
                      ("alpasim_grpc.v0.egodriver_pb2_grpc", egodriver_pb2_grpc)]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("grpc", grpc_mod)


try:  # prefer the real protos: closer to production
    import alpasim_grpc  # noqa: F401
    import grpc  # noqa: F401
    USING_REAL_PROTO = True
except Exception:  # pragma: no cover - exercised only in bare envs
    _install_proto_stub()
    USING_REAL_PROTO = False


# ---------------------------------------------------------------------------
# 2. driver loading
# ---------------------------------------------------------------------------

def load_driver_module(relpath: str, alias: str):
    """Import a driver .py by path under a private alias (fresh module state)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# Driver family under audit. Each entry: (label, relpath, servicer attr).
# All four are pure-numpy (no torch), so they run on CPU with no GPU present.
DRIVERS = [
    ("route_follower", "driver/route_follower.py", "RouteFollower"),
    ("submission_route_follower", "submission/route_follower.py", "RouteFollower"),
    ("route_follower_term", "driver/route_follower_term.py", "RouteFollower"),
    ("abstention_driver", "driver/abstention_driver.py", "AbstentionDriver"),
]


def make_servicer(relpath: str, attr: str, alias_suffix: str = ""):
    mod = load_driver_module(relpath, f"_uut_{attr}_{abs(hash(relpath))}{alias_suffix}")
    return getattr(mod, attr)(), mod


# ---------------------------------------------------------------------------
# 3. duck-typed request objects + a grpc-like context
# ---------------------------------------------------------------------------

class Ctx:
    """Stand-in for grpc.ServicerContext. abort() raises, as the real one does."""

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


# ---------------------------------------------------------------------------
# 4. deterministic rollout scenarios
# ---------------------------------------------------------------------------
# Distinct, reproducible observation streams. Profiles are chosen so that a leak
# WOULD be visible: 'flapping' and 'degenerate' drive the abstention confidence
# EMA far from 1.0, so if that EMA were shared, the clean rollouts' speeds would
# be dragged down and the trajectories would differ from their solo runs.

PROFILES = ("clean_straight", "clean_curve", "degenerate", "flapping")


class Rollout:
    """A reproducible observation sequence for one session."""

    def __init__(self, uuid: str, profile: str, seed: int, n_steps: int = 8):
        self.uuid = uuid
        self.profile = profile
        self.seed = seed
        self.n_steps = n_steps
        self.rng = random.Random(seed)
        self._calls = list(self._build())

    # -- geometry ---------------------------------------------------------
    def _route_waypoints(self, step: int):
        """Route waypoints in rig frame, per profile."""
        rng = random.Random(self.seed * 1000 + step)
        if self.profile == "degenerate":
            # half the steps have an unusable route (0 or 1 waypoint)
            if step % 2 == 0:
                return [vec(3.0, 0.0)]
            n, curve = 8, 0.0
        elif self.profile == "flapping":
            # route length jumps hard frame to frame -> hammers the stability EMA
            n = 2 if step % 2 == 0 else 14
            curve = 0.05
        elif self.profile == "clean_curve":
            n, curve = 14, 0.06
        else:
            n, curve = 14, 0.0
        pts = []
        x = y = th = 0.0
        for k in range(n):
            th += curve
            x += 2.0 * math.cos(th)
            y += 2.0 * math.sin(th)
            jitter = rng.uniform(-0.01, 0.01)
            pts.append(vec(x, y + jitter))
        return pts

    def _build(self):
        """Yield (rpc_name, request) in the order the runtime would send them."""
        dt_us = 500_000  # control_timestep = 500 ms
        speed0 = {"clean_straight": 8.0, "clean_curve": 6.0,
                  "degenerate": 4.0, "flapping": 7.0}[self.profile]
        yield ("start_session", NS(session_uuid=self.uuid, random_seed=self.seed))
        x = y = yaw = 0.0
        for step in range(self.n_steps):
            t = 1_000_000 + step * dt_us
            v = speed0 + 0.5 * math.sin(step * 0.7 + self.seed)
            x += v * (dt_us / 1e6) * math.cos(yaw)
            y += v * (dt_us / 1e6) * math.sin(yaw)
            if self.profile == "clean_curve":
                yaw += 0.03
            poses = [pose_at(t - dt_us, x - v * 0.5, y, yaw), pose_at(t, x, y, yaw)]
            dyn = [NS(linear_velocity=vec(v * math.cos(yaw), v * math.sin(yaw)))
                   for _ in poses]
            yield ("submit_egomotion_observation",
                   NS(session_uuid=self.uuid,
                      trajectory=NS(poses=poses), dynamic_states=dyn))
            yield ("submit_route",
                   NS(session_uuid=self.uuid,
                      route=NS(waypoints=self._route_waypoints(step), timestamp_us=t)))
            yield ("drive", NS(session_uuid=self.uuid, time_now_us=t, time_query_us=t))
        yield ("close_session", NS(session_uuid=self.uuid))

    def calls(self):
        return list(self._calls)


# ---------------------------------------------------------------------------
# 5. canonicalisation: exact bitwise trajectory fingerprint
# ---------------------------------------------------------------------------

def canon(traj) -> bytes:
    """Exact byte fingerprint of a trajectory (IEEE-754 bits, not repr)."""
    out = bytearray()
    for p in traj.poses:
        out += struct.pack("<q", int(p.timestamp_us))
        v, q = p.pose.vec, p.pose.quat
        out += struct.pack("<7d", float(v.x), float(v.y), float(v.z),
                           float(q.w), float(q.x), float(q.y), float(q.z))
    return bytes(out)


def drive_outputs(responses) -> list[bytes]:
    return [canon(r.trajectory) for r in responses]


# ---------------------------------------------------------------------------
# 6. execution engines
# ---------------------------------------------------------------------------

def dispatch(servicer, rpc: str, req, ctx: Ctx):
    return getattr(servicer, rpc)(req, ctx)


def run_solo(relpath, attr, rollout: Rollout) -> list[bytes]:
    """Run one rollout alone in a brand-new servicer -> reference outputs."""
    servicer, _ = make_servicer(relpath, attr, alias_suffix="_solo")
    ctx = Ctx()
    out = []
    for rpc, req in rollout.calls():
        resp = dispatch(servicer, rpc, req, ctx)
        if rpc == "drive":
            out.append(canon(resp.trajectory))
    return out


def run_interleaved(relpath, attr, rollouts, order) -> dict[str, list[bytes]]:
    """Run all rollouts in ONE servicer, calls interleaved in `order`.

    `order` is a list of rollout indices; each occurrence consumes that
    rollout's next pending call, so per-rollout call order is preserved while
    the cross-rollout interleaving varies.
    """
    servicer, _ = make_servicer(relpath, attr, alias_suffix="_conc")
    ctx = Ctx()
    pending = [iter(r.calls()) for r in rollouts]
    out: dict[str, list[bytes]] = {r.uuid: [] for r in rollouts}
    for idx in order:
        try:
            rpc, req = next(pending[idx])
        except StopIteration:
            continue
        resp = dispatch(servicer, rpc, req, ctx)
        if rpc == "drive":
            out[rollouts[idx].uuid].append(canon(resp.trajectory))
    return out


def interleaving(rollouts, rng: random.Random) -> list[int]:
    """A random interleaving that preserves each rollout's internal order."""
    counts = [len(r.calls()) for r in rollouts]
    slots = list(itertools.chain.from_iterable(
        [i] * c for i, c in enumerate(counts)))
    rng.shuffle(slots)
    return slots


def round_robin(rollouts) -> list[int]:
    counts = [len(r.calls()) for r in rollouts]
    order = []
    for k in range(max(counts)):
        for i, c in enumerate(counts):
            if k < c:
                order.append(i)
    return order


def make_rollouts(n: int, n_steps: int = 8) -> list[Rollout]:
    return [Rollout(uuid=f"rollout-{i:02d}-{PROFILES[i % len(PROFILES)]}",
                    profile=PROFILES[i % len(PROFILES)],
                    seed=100 + i, n_steps=n_steps)
            for i in range(n)]


# ---------------------------------------------------------------------------
# 7. TESTS -- isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_two_concurrent_rollouts_round_robin(label, relpath, attr):
    """Eval topology: exactly 2 concurrent rollouts, round-robin interleaved."""
    rollouts = make_rollouts(2)
    solo = {r.uuid: run_solo(relpath, attr, r) for r in rollouts}
    conc = run_interleaved(relpath, attr, rollouts, round_robin(rollouts))
    for r in rollouts:
        assert conc[r.uuid] == solo[r.uuid], (
            f"{label}: rollout {r.uuid} ({r.profile}) diverged when run "
            f"concurrently -> cross-rollout state contamination")


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_two_concurrent_rollouts_adversarial_pairing(label, relpath, attr):
    """Pair the WORST offender (flapping route) with a clean rollout.

    If the confidence EMA / route history were shared, the clean rollout's speed
    profile would be dragged down by the flapping one and diverge from solo.
    """
    clean = Rollout("clean", "clean_straight", seed=7, n_steps=10)
    noisy = Rollout("noisy", "flapping", seed=8, n_steps=10)
    solo_clean = run_solo(relpath, attr, clean)
    solo_noisy = run_solo(relpath, attr, noisy)
    conc = run_interleaved(relpath, attr, [clean, noisy],
                           round_robin([clean, noisy]))
    assert conc["clean"] == solo_clean, f"{label}: clean rollout poisoned by flapping peer"
    assert conc["noisy"] == solo_noisy, f"{label}: flapping rollout perturbed by clean peer"


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_randomized_interleavings(label, relpath, attr):
    """>=100 randomized interleavings over N rollouts must all be bit-identical."""
    n_interleavings = int(os.environ.get("ISO_INTERLEAVINGS", "120"))
    rollouts = make_rollouts(4, n_steps=6)
    solo = {r.uuid: run_solo(relpath, attr, r) for r in rollouts}
    for trial in range(n_interleavings):
        rng = random.Random(9000 + trial)
        conc = run_interleaved(relpath, attr, rollouts, interleaving(rollouts, rng))
        for r in rollouts:
            assert conc[r.uuid] == solo[r.uuid], (
                f"{label}: divergence on interleaving trial {trial} "
                f"for {r.uuid} ({r.profile})")


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_threaded_concurrency_matches_solo(label, relpath, attr):
    """Real threads, real contention -> still bit-identical to solo."""
    rollouts = make_rollouts(4, n_steps=6)
    solo = {r.uuid: run_solo(relpath, attr, r) for r in rollouts}
    servicer, _ = make_servicer(relpath, attr, alias_suffix="_thread")
    results: dict[str, list[bytes]] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(rollouts))

    def worker(r: Rollout):
        ctx = Ctx()
        out = []
        try:
            barrier.wait()
            for rpc, req in r.calls():
                resp = dispatch(servicer, rpc, req, ctx)
                if rpc == "drive":
                    out.append(canon(resp.trajectory))
            results[r.uuid] = out
        except BaseException as e:  # noqa: BLE001 - surface in the assert
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(r,)) for r in rollouts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"{label}: exception under threaded concurrency: {errors[:1]}"
    for r in rollouts:
        assert results[r.uuid] == solo[r.uuid], (
            f"{label}: threaded run diverged for {r.uuid} ({r.profile})")


# ---------------------------------------------------------------------------
# 8. TESTS -- the Drive contract: never throw, always emit a trajectory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_drive_never_throws_on_unknown_session(label, relpath, attr):
    """A Drive for a session we never started must NOT raise out of the handler.

    Under 2 concurrent rollouts a stray/late Drive is a real possibility. An
    exception escaping the RPC yields no trajectory -> the controller has no
    plan -> the scene is at risk of a zero.
    """
    servicer, _ = make_servicer(relpath, attr, alias_suffix="_unknown")
    ctx = Ctx()
    resp = servicer.drive(NS(session_uuid="never-started", time_now_us=1_000_000,
                             time_query_us=1_000_000), ctx)
    assert resp is not None and hasattr(resp, "trajectory"), (
        f"{label}: drive() on an unknown session did not return a DriveResponse")


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_drive_never_throws_after_close(label, relpath, attr):
    """A late Drive arriving after close_session must not raise either."""
    servicer, _ = make_servicer(relpath, attr, alias_suffix="_afterclose")
    ctx = Ctx()
    r = Rollout("late", "clean_straight", seed=3, n_steps=3)
    for rpc, req in r.calls():
        dispatch(servicer, rpc, req, ctx)
    resp = servicer.drive(NS(session_uuid="late", time_now_us=9_000_000,
                             time_query_us=9_000_000), ctx)
    assert resp is not None and hasattr(resp, "trajectory"), (
        f"{label}: drive() after close_session did not return a DriveResponse")


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_drive_before_any_pose_returns_trajectory(label, relpath, attr):
    """Drive before the first egomotion observation: fallback, not an exception."""
    servicer, _ = make_servicer(relpath, attr, alias_suffix="_nopose")
    ctx = Ctx()
    servicer.start_session(NS(session_uuid="s", random_seed=1), ctx)
    resp = servicer.drive(NS(session_uuid="s", time_now_us=1_000_000,
                             time_query_us=1_000_000), ctx)
    assert resp is not None and hasattr(resp, "trajectory")


# ---------------------------------------------------------------------------
# 9. TESTS -- eviction / memory bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_close_session_evicts_state(label, relpath, attr):
    """close_session must drop the per-rollout container (no leak across 16x2)."""
    servicer, _ = make_servicer(relpath, attr, alias_suffix="_evict")
    ctx = Ctx()
    for i in range(40):
        r = Rollout(f"ev-{i}", PROFILES[i % len(PROFILES)], seed=200 + i, n_steps=2)
        for rpc, req in r.calls():
            dispatch(servicer, rpc, req, ctx)
    assert len(servicer._s) == 0, (
        f"{label}: {len(servicer._s)} sessions still resident after close_session")


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_pose_buffer_is_bounded(label, relpath, attr):
    """The per-session pose history must stay bounded over a long rollout."""
    servicer, _ = make_servicer(relpath, attr, alias_suffix="_bound")
    ctx = Ctx()
    servicer.start_session(NS(session_uuid="long", random_seed=1), ctx)
    for step in range(1200):
        t = 1_000_000 + step * 100_000
        servicer.submit_egomotion_observation(
            NS(session_uuid="long",
               trajectory=NS(poses=[pose_at(t, step * 0.5, 0.0, 0.0)]),
               dynamic_states=[NS(linear_velocity=vec(5.0, 0.0))]), ctx)
    sess = servicer._s["long"]
    assert len(sess.pose_ts) <= 400, f"{label}: pose_ts grew to {len(sess.pose_ts)}"
    assert len(sess.poses) <= 400, f"{label}: poses grew to {len(sess.poses)}"


# ---------------------------------------------------------------------------
# 10. TESTS -- no module-level mutable global is written on the Drive path
# ---------------------------------------------------------------------------

_IMMUTABLE = (int, float, bool, str, bytes, type(None), tuple, frozenset)
# The dev guard's own bookkeeping is module-level and mutable BY DESIGN; it is
# only populated when RF_STATE_GUARD=1 and is never read on the driving path.
_GUARD_OWN_STATE = {"_GUARD_BASELINE", "_GUARD_SEEN"}


def _module_mutable_snapshot(mod):
    """Every module-level object that is NOT obviously immutable."""
    snap = {}
    for name, val in vars(mod).items():
        if name.startswith("__") or name in _GUARD_OWN_STATE:
            continue
        if isinstance(val, _IMMUTABLE):
            continue
        if isinstance(val, types.ModuleType) or callable(val):
            continue
        snap[name] = (type(val).__name__, repr(val)[:200])
    return snap


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_no_module_level_mutable_written_on_drive_path(label, relpath, attr):
    """Module-level state must be config-only and read-only after import."""
    servicer, mod = make_servicer(relpath, attr, alias_suffix="_globals")
    before = _module_mutable_snapshot(mod)
    ctx = Ctx()
    for r in make_rollouts(3, n_steps=4):
        for rpc, req in r.calls():
            dispatch(servicer, rpc, req, ctx)
    after = _module_mutable_snapshot(mod)
    changed = sorted(set(before) ^ set(after)) + sorted(
        k for k in before if k in after and before[k] != after[k])
    assert before == after, (
        f"{label}: module-level mutable state changed across the Drive path: "
        f"{[(k, before.get(k), after.get(k)) for k in changed]}")


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_dev_guard_detects_a_planted_module_level_write(label, relpath, attr):
    """Positive control: the RF_STATE_GUARD guard must FIRE on a real leak.

    A guard that can only pass is worthless. Plant a module-level mutable, run
    the guard, and require it to raise.
    """
    servicer, mod = make_servicer(relpath, attr, alias_suffix="_guardfire")
    if not hasattr(mod, "_guard_check"):
        pytest.skip(f"{label} has no dev guard")
    mod.STATE_GUARD = True
    mod._GUARD_BASELINE.clear()
    mod._GUARD_SEEN.clear()
    ctx = Ctx()
    mod._guard_new_session("baseline-session")  # snapshots the clean baseline
    mod.LEAKED_SHARED_CACHE = {"rollout-a": [1, 2, 3]}  # the planted leak
    try:
        with pytest.raises(AssertionError, match="module-level mutable"):
            mod._guard_check("test")
    finally:
        del mod.LEAKED_SHARED_CACHE
        mod.STATE_GUARD = False


@pytest.mark.parametrize("label,relpath,attr", DRIVERS)
def test_dev_guard_rejects_session_id_reuse(label, relpath, attr):
    """A repeated session_uuid means state from the previous rollout may persist."""
    servicer, mod = make_servicer(relpath, attr, alias_suffix="_guardreuse")
    if not hasattr(mod, "_guard_new_session"):
        pytest.skip(f"{label} has no dev guard")
    mod.STATE_GUARD = True
    mod._GUARD_BASELINE.clear()
    mod._GUARD_SEEN.clear()
    ctx = Ctx()
    try:
        servicer.start_session(NS(session_uuid="dup", random_seed=1), ctx)
        with pytest.raises(AssertionError, match="re-initialised"):
            servicer.start_session(NS(session_uuid="dup", random_seed=1), ctx)
    finally:
        mod.STATE_GUARD = False


def test_dev_guard_is_off_by_default():
    """Zero cost in prod: the guard must default to disabled."""
    if os.environ.get("RF_STATE_GUARD"):
        pytest.skip("RF_STATE_GUARD set in this environment; default not observable")
    for label, relpath, attr in DRIVERS:
        _, mod = make_servicer(relpath, attr, alias_suffix="_guardoff")
        if hasattr(mod, "STATE_GUARD"):
            assert mod.STATE_GUARD is False, f"{label}: RF_STATE_GUARD defaults on"


# ---------------------------------------------------------------------------
# 11. TESTS -- RNG isolation (the policy-driver class of bug)
# ---------------------------------------------------------------------------
# rl/policy_driver.py samples an action with torch's GLOBAL RNG. Two rollouts in
# one process then consume draws from one shared stream, so rollout B's actions
# depend on how many samples rollout A happened to take first -> not reproducible
# and not isolated. torch is absent in the driver env, so we reproduce the exact
# pattern with a stub policy and prove the per-session-generator fix.

class GlobalRngPolicy:
    """Model stub that samples from a process-global stream (the bug)."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, session_state):
        return self.rng.random()


class PerSessionRngPolicy:
    """Model stub that samples from a per-session stream seeded by random_seed."""

    def act(self, session_state):
        return session_state.rng.random()


class _SessState:
    def __init__(self, seed):
        self.rng = random.Random(seed)


def _sample_sequences(policy, per_session_seed, interleave):
    """Return {sid: [samples]} for two sessions under a given call interleaving."""
    sessions = {sid: _SessState(per_session_seed(sid)) for sid in ("A", "B")}
    out = {sid: [] for sid in sessions}
    for sid in interleave:
        out[sid].append(policy.act(sessions[sid]))
    return out


def test_global_rng_policy_is_contaminated_by_interleaving():
    """Guard test: proves the harness can DETECT an RNG leak (negative control)."""
    solo_a = _sample_sequences(GlobalRngPolicy(0), lambda s: 1, ["A"] * 5)["A"]
    inter = _sample_sequences(GlobalRngPolicy(0), lambda s: 1,
                              ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
    assert inter["A"] != solo_a, (
        "negative control failed: a shared global RNG should have leaked here")


def test_per_session_rng_is_isolated_under_interleaving():
    """The fix: a per-session generator seeded from DriveSessionRequest.random_seed."""
    seed_of = {"A": 11, "B": 22}
    solo = {sid: _sample_sequences(PerSessionRngPolicy(), lambda s: seed_of[s],
                                   [sid] * 5)[sid] for sid in ("A", "B")}
    for trial in range(100):
        rng = random.Random(trial)
        order = ["A"] * 5 + ["B"] * 5
        rng.shuffle(order)
        got = _sample_sequences(PerSessionRngPolicy(), lambda s: seed_of[s], order)
        assert got["A"] == solo["A"] and got["B"] == solo["B"], (
            f"per-session RNG leaked on trial {trial}")


# ---------------------------------------------------------------------------
# 12. cost check: per-rollout keying is a dict lookup
# ---------------------------------------------------------------------------

def test_session_lookup_is_constant_time():
    """Keying by session_uuid must not scale with the number of live sessions."""
    import timeit
    small = {f"s{i}": object() for i in range(2)}
    large = {f"s{i}": object() for i in range(10_000)}
    t_small = timeit.timeit(lambda: small["s1"], number=200_000)
    t_large = timeit.timeit(lambda: large["s1"], number=200_000)
    assert t_large < t_small * 3, (
        f"session dict lookup is not O(1)-ish: {t_small:.4f} vs {t_large:.4f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
