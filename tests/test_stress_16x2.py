#!/usr/bin/env python
"""R2 — 16x2 concurrency stress harness for the shipped egodriver.

Eval topology: 16 replica PROCESSES on GPUs 4-7, each serving 2 concurrent
rollouts. State isolation therefore matters *within* a replica (2 concurrent
sessions sharing one servicer); the 16 replicas are isolated by the process
boundary. This harness models both: 16 independent servicer instances, each
driving 2 rollouts concurrently in threads, so it exercises the isolation
boundary that actually exists AND the aggregate throughput.

Asserts, per the charter:
  1. per-rollout state isolation — no shared mutable state across sessions
  2. fault injection — slow / raising / OOM inference each still yields a safe
     fallback trajectory, never an exception out of the handler, never a leak
  3. latency — full Drive distribution reported; mean must stay <= 0.1 s

MOCKED INFERENCE: the shipped `route_follower` is pure geometry and has no model
forward, so `_plan()` is the compute step and is the injection seam. When the
distillation student lands, its forward sits behind the same seam and these
tests apply unchanged.

SCOPE NOTE (charter R2/R3): the RNG clause — per-`Session` `torch.Generator`
seeded from `DriveSessionRequest.random_seed`, no `torch.manual_seed()` on the
hot path — is `policy_driver`-scoped. The shipped driver has no RNG at all
(asserted by `test_shipped_driver_has_no_rng_at_all`), so that clause is vacuous
here and deliberately does not gate this harness.

CPU-only. No GPU, no simulator, no renderer. Run SERIALLY:
    ~/alpasim/.venv-driver/bin/python -m pytest tests/test_stress_16x2.py -q
"""
from __future__ import annotations

import math
import statistics
import sys
import threading
import time
from dataclasses import fields, MISSING
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

# Reuse the verified harness primitives rather than re-implementing them.
from test_drive_handler_fallback import (  # noqa: E402
    Ctx, NS, canon, load_module, pose_at, rollout_calls, vec,
)

SHIPPED = "submission/route_follower.py"

# Eval topology
REPLICAS = 16
CONCURRENT_PER_REPLICA = 2
LATENCY_BUDGET_S = 0.1

# Keep total work bounded: this runs on a shared login node, single core.
STEPS = 6


def fresh_servicer(alias: str):
    mod = load_module(REPO / SHIPPED, alias)
    return mod.RouteFollower(), mod


@pytest.fixture(scope="module")
def mod():
    return load_module(REPO / SHIPPED, "_stress_mod")


# ---------------------------------------------------------------------------
# rollout definitions: 32 distinct streams (16 replicas x 2)
# ---------------------------------------------------------------------------

def rollout_spec(i: int):
    """Deterministic, distinct per index. Mixes curvature and route health."""
    curve = [0.0, 0.05, -0.05, 0.02][i % 4]
    n_wp = [14, 14, 3, 1][i % 4]          # index 3 -> degenerate route branch
    speed = 4.0 + (i % 5) * 1.7
    return curve, n_wp, speed


def calls_for(uuid: str, i: int, steps: int = STEPS):
    curve, n_wp, speed = rollout_spec(i)
    return list(rollout_calls(uuid, curve, n_wp, speed, n_steps=steps))


def drive_solo(alias: str, uuid: str, i: int) -> list[bytes]:
    servicer, _ = fresh_servicer(alias)
    ctx = Ctx()
    out = []
    for rpc, req in calls_for(uuid, i):
        resp = getattr(servicer, rpc)(req, ctx)
        if rpc == "drive":
            out.append(canon(resp.trajectory))
    return out


def run_replica(servicer, specs, latencies=None, errors=None):
    """Drive `specs` concurrently in threads against ONE servicer instance."""
    results: dict[str, list[bytes]] = {}
    errs: list[BaseException] = [] if errors is None else errors
    barrier = threading.Barrier(len(specs))

    def worker(uuid, i):
        ctx = Ctx()
        out = []
        try:
            barrier.wait()
            for rpc, req in calls_for(uuid, i):
                t0 = time.perf_counter()
                resp = getattr(servicer, rpc)(req, ctx)
                if rpc == "drive":
                    dt = time.perf_counter() - t0
                    if latencies is not None:
                        latencies.append(dt)
                    out.append(canon(resp.trajectory))
            results[uuid] = out
        except BaseException as e:  # noqa: BLE001 — a throw here IS the failure
            errs.append(e)

    threads = [threading.Thread(target=worker, args=(u, i)) for u, i in specs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errs


# ---------------------------------------------------------------------------
# 1. STATE ISOLATION
# ---------------------------------------------------------------------------

def test_session_mutables_use_default_factory(mod):
    """Every mutable Session field must be default_factory, not a shared default.

    A bare mutable default on a dataclass field is shared by every instance —
    the textbook way per-rollout state silently becomes global.
    """
    bad = []
    for f in fields(mod.Session):
        if f.default is not MISSING and isinstance(f.default, (dict, list, set, bytearray)):
            bad.append(f.name)
    assert not bad, f"Session fields with shared mutable defaults: {bad}"


def test_servicer_keys_all_state_by_session_uuid(mod):
    """The only per-rollout containers may be dicts keyed by session_uuid."""
    servicer = mod.RouteFollower()
    ctx = Ctx()
    for i, u in enumerate(("a", "b")):
        for rpc, req in calls_for(u, i, steps=2):
            if rpc == "close_session":
                continue
            getattr(servicer, rpc)(req, ctx)
    assert set(servicer._s) == {"a", "b"}, f"live sessions keyed unexpectedly: {list(servicer._s)}"
    # every dict attribute holding Session objects must be uuid-keyed
    for name, val in vars(servicer).items():
        if isinstance(val, dict) and val:
            assert all(isinstance(k, str) for k in val), f"{name} has non-str keys"


def test_shipped_driver_has_no_rng_at_all(mod):
    """Scope check for the charter's RNG clause: shipped path has no RNG.

    Documents *why* the per-Session torch.Generator requirement is
    policy_driver-scoped and not asserted against the shipped harness.
    """
    src = (REPO / SHIPPED).read_text()
    for token in ("random", "torch", "manual_seed", "Generator", "np.random"):
        assert token not in src, (
            f"shipped driver unexpectedly references '{token}' — the RNG clause "
            f"would no longer be vacuous and must be asserted here")


def test_16x2_topology_bit_identical_to_solo():
    """THE eval topology: 16 replicas x 2 concurrent rollouts, all bit-identical.

    Cross-session contamination would change the emitted trajectory, which is a
    wrong plan, which is a collision/excursion risk — i.e. a zero.
    """
    specs = [(f"r{r:02d}-s{s}", r * CONCURRENT_PER_REPLICA + s)
             for r in range(REPLICAS) for s in range(CONCURRENT_PER_REPLICA)]
    solo = {u: drive_solo(f"_solo_{u}", u, i) for u, i in specs}

    all_errors: list[BaseException] = []
    for r in range(REPLICAS):
        servicer, _ = fresh_servicer(f"_rep_{r}")
        mine = specs[r * CONCURRENT_PER_REPLICA:(r + 1) * CONCURRENT_PER_REPLICA]
        got, errs = run_replica(servicer, mine, errors=all_errors)
        for u, _i in mine:
            assert got.get(u) == solo[u], (
                f"replica {r}: rollout {u} diverged under 2-way concurrency "
                f"-> cross-rollout state contamination")
    assert not all_errors, f"handler raised under load: {all_errors[:1]}"


def test_no_session_leak_across_replica_lifetimes():
    """Sessions must be evicted; 16 replicas x many rollouts must not accumulate."""
    servicer, mod_ = fresh_servicer("_leak")
    ctx = Ctx()
    for i in range(REPLICAS * CONCURRENT_PER_REPLICA * 3):
        u = f"leak-{i}"
        for rpc, req in calls_for(u, i, steps=2):
            getattr(servicer, rpc)(req, ctx)
    assert len(servicer._s) == 0, f"{len(servicer._s)} live sessions leaked"
    assert len(servicer._closed) <= mod_.MAX_TOMBSTONES, (
        f"tombstones unbounded: {len(servicer._closed)}")


# ---------------------------------------------------------------------------
# 2. FAULT INJECTION
# ---------------------------------------------------------------------------
# Each fault is injected at the inference seam (`_plan`). Requirement: the
# handler emits a safe fallback, never throws, and the *other* concurrent
# session in the same replica is unaffected.

def _poses_of(traj):
    return [(p.pose.vec.x, p.pose.vec.y) for p in traj.poses]


def _is_safe_fallback(traj) -> bool:
    """Non-empty, monotone-forward, non-increasing steps, ends stopped."""
    xs = _poses_of(traj)
    if len(xs) < 2:
        return False
    steps = [math.dist(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
    if any(s < -1e-12 for s in steps):
        return False
    if any(b > a + 1e-9 for a, b in zip(steps, steps[1:])):
        return False
    return steps[-1] <= 1e-9


FAULTS = [
    ("exception", lambda: (_ for _ in ()).throw(RuntimeError("injected inference failure"))),
    ("oom", lambda: (_ for _ in ()).throw(MemoryError("injected CUDA OOM"))),
    ("value_error", lambda: (_ for _ in ()).throw(ValueError("injected bad tensor shape"))),
]


@pytest.mark.parametrize("fault_name,thrower", FAULTS)
def test_faulty_inference_yields_safe_fallback_never_throws(fault_name, thrower):
    """A raising/OOM inference step must degrade to a safe stop, not a zero."""
    servicer, _ = fresh_servicer(f"_fault_{fault_name}")
    ctx = Ctx()
    u = "faulty"
    calls = calls_for(u, 0)
    # prime the session so a pose exists (fallback needs a last-known pose)
    for rpc, req in calls:
        if rpc == "drive":
            break
        getattr(servicer, rpc)(req, ctx)

    servicer._plan = lambda *a, **k: thrower()        # inject at the seam
    drive_req = next(req for rpc, req in calls if rpc == "drive")
    resp = servicer.drive(drive_req, ctx)             # must not raise

    assert resp is not None and hasattr(resp, "trajectory")
    assert _is_safe_fallback(resp.trajectory), (
        f"{fault_name}: handler returned a non-safe trajectory "
        f"({len(resp.trajectory.poses)} poses)")


@pytest.mark.parametrize("fault_name,thrower", FAULTS)
def test_fault_in_one_session_does_not_leak_to_its_peer(fault_name, thrower):
    """The co-resident rollout must be untouched when its peer's inference dies.

    This is the case that turns one zero into two.
    """
    healthy_solo = drive_solo(f"_peer_solo_{fault_name}", "healthy", 1)

    servicer, mod_ = fresh_servicer(f"_peer_{fault_name}")
    ctx = Ctx()
    real_plan = mod_.RouteFollower._plan

    def selective_plan(self, s, t0):
        # fail only for the doomed session, by identity of its Session object
        if s is self._s.get("doomed"):
            thrower()
        return real_plan(self, s, t0)

    servicer._plan = selective_plan.__get__(servicer, type(servicer))

    specs = [("doomed", 0), ("healthy", 1)]
    got, errs = run_replica(servicer, specs)
    assert not errs, f"{fault_name}: handler raised: {errs[:1]}"
    assert got["healthy"] == healthy_solo, (
        f"{fault_name}: peer rollout perturbed by the faulting session")
    assert all(_is_safe_fallback_bytes(b) for b in got["doomed"]), (
        f"{fault_name}: doomed session did not emit safe fallbacks")


def _is_safe_fallback_bytes(b: bytes) -> bool:
    """Structural check on a canon() fingerprint: whole poses, at least one.

    canon() packs 8 bytes (timestamp, <q) + 56 bytes (7 doubles) = 64 B per pose.
    This verifies a real trajectory came back rather than an empty or truncated
    one; the kinematic safe-stop shape is asserted separately by
    `_is_safe_fallback` on live trajectory objects.
    """
    return len(b) > 0 and len(b) % 64 == 0


def test_slow_inference_does_not_throw_but_REVEALS_NO_DEADLINE_GUARD():
    """Slow inference: handler survives, but there is NO overrun protection.

    The charter asks that an RPC overrun (>0.1 s) yield a safe fallback. The
    shipped driver has no deadline/watchdog anywhere on the Drive path, so a slow
    inference simply *takes* longer — it is not aborted and no fallback is
    substituted. This test pins the CURRENT behaviour and documents the gap
    rather than asserting a guard that does not exist.

    See reports/r2_stress_findings.md — proposed watchdog, NOT applied (it is a
    behavioural change to the shipped path and needs its own GPU gate).
    """
    servicer, mod_ = fresh_servicer("_slow")
    ctx = Ctx()
    u = "slowpoke"
    calls = calls_for(u, 0)
    for rpc, req in calls:
        if rpc == "drive":
            break
        getattr(servicer, rpc)(req, ctx)

    real_plan = mod_.RouteFollower._plan
    OVERRUN_S = 0.25

    def slow_plan(self, s, t0):
        time.sleep(OVERRUN_S)
        return real_plan(self, s, t0)

    servicer._plan = slow_plan.__get__(servicer, type(servicer))
    drive_req = next(req for rpc, req in calls if rpc == "drive")

    t0 = time.perf_counter()
    resp = servicer.drive(drive_req, ctx)
    dt = time.perf_counter() - t0

    # survives
    assert resp is not None and len(resp.trajectory.poses) > 0, "slow path lost the trajectory"
    # and demonstrably overruns, unguarded
    assert dt >= OVERRUN_S, "injection did not take effect"
    assert dt > LATENCY_BUDGET_S, (
        "unexpected: the call came in under budget despite a 0.25 s injected "
        "stall — re-check the injection seam")
    # If a watchdog is ever added, this assertion is the one to flip.


# ---------------------------------------------------------------------------
# 3. LATENCY UNDER LOAD
# ---------------------------------------------------------------------------

def test_drive_latency_distribution_under_16x2_load(capsys):
    """Report the full Drive latency distribution at eval topology; mean <= 0.1 s."""
    latencies: list[float] = []
    errors: list[BaseException] = []
    specs_all = [(f"L{r:02d}-{s}", r * CONCURRENT_PER_REPLICA + s)
                 for r in range(REPLICAS) for s in range(CONCURRENT_PER_REPLICA)]

    t_wall = time.perf_counter()
    for r in range(REPLICAS):
        servicer, _ = fresh_servicer(f"_lat_{r}")
        mine = specs_all[r * CONCURRENT_PER_REPLICA:(r + 1) * CONCURRENT_PER_REPLICA]
        run_replica(servicer, mine, latencies=latencies, errors=errors)
    wall = time.perf_counter() - t_wall

    assert not errors, f"handler raised under load: {errors[:1]}"
    assert latencies, "no latency samples collected"

    latencies.sort()
    n = len(latencies)

    def pct(p):
        return latencies[min(n - 1, int(p / 100 * n))]

    mean = statistics.fmean(latencies)
    report = (
        f"\n=== Drive latency @ {REPLICAS}x{CONCURRENT_PER_REPLICA} "
        f"({n} calls, {wall:.1f}s wall, GIL-bound single core) ===\n"
        f"  mean   {mean*1000:8.2f} ms   [budget {LATENCY_BUDGET_S*1000:.0f} ms]\n"
        f"  p50    {pct(50)*1000:8.2f} ms\n"
        f"  p90    {pct(90)*1000:8.2f} ms\n"
        f"  p99    {pct(99)*1000:8.2f} ms\n"
        f"  max    {latencies[-1]*1000:8.2f} ms\n"
        f"  over-budget calls: {sum(1 for x in latencies if x > LATENCY_BUDGET_S)}/{n}\n"
        f"  headroom on mean:  {LATENCY_BUDGET_S/mean:.0f}x\n"
    )
    with capsys.disabled():
        print(report)

    assert mean <= LATENCY_BUDGET_S, (
        f"mean Drive latency {mean*1000:.2f} ms exceeds the "
        f"{LATENCY_BUDGET_S*1000:.0f} ms budget")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
