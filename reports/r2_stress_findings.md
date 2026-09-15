# R2 — 16×2 concurrency stress harness: findings

Branch `fix/drive-handler-fallback`. CPU-only, mocked inference, no GPU/sim/renderer.

```
$ ~/alpasim/.venv-driver/bin/python -m pytest tests/test_stress_16x2.py -q
13 passed, 2 warnings in 3.46s
```

**Four-axis:** (a) zero-rate ↓↓ direct — dead handler and cross-session
contamination are both catastrophic · (b) capability 0 — harness only, no driver
behaviour changed · (c) latency **measured, not changed** — mean 5.40 ms at eval
topology vs a 100 ms budget · (d) VRAM 0.

**Gate state: MERGED to `fix/drive-handler-fallback` (tests only, nothing
shipped).** The branch itself remains awaiting-GPU behind R1's navtest gate.

---

## 1. What it exercises

Eval topology is 16 replica *processes* × 2 concurrent rollouts. State isolation
therefore matters **within** a replica — the 16 replicas are isolated by the
process boundary, so a single 32-way test would model a topology that does not
exist. The harness runs 16 independent servicer instances, each driving 2
rollouts concurrently in real threads with a start barrier.

Injection seam: the shipped `route_follower` is pure geometry with no model
forward, so `_plan()` *is* the compute step and is where faults go in. When the
distillation student lands, its forward sits behind the same seam and these tests
apply unchanged.

## 2. Results

### State isolation — PASS

| Assertion | Result |
|---|---|
| No `Session` field carries a shared mutable default (the classic way per-rollout state becomes global) | pass — all mutables use `default_factory` |
| All per-rollout containers keyed by `session_uuid` | pass |
| 16×2 topology, every rollout bit-identical to solo execution | pass — 32 rollouts |
| No session leak across 96 rollout lifetimes; tombstones bounded | pass — 0 live, ≤ `MAX_TOMBSTONES` |

### Fault injection — PASS (3 of 4 fault classes)

Injected at `_plan()`; each asserts the handler emits a **safe stop** (non-empty,
monotone-forward, non-increasing steps, ends halted) and never raises:

| Fault | Handler survives | Safe fallback | Peer rollout unaffected |
|---|---|---|---|
| `RuntimeError` (inference failure) | ✅ | ✅ | ✅ bit-identical to solo |
| `MemoryError` (simulated CUDA OOM) | ✅ | ✅ | ✅ bit-identical to solo |
| `ValueError` (bad tensor shape) | ✅ | ✅ | ✅ bit-identical to solo |
| **RPC overrun (>0.1 s)** | ✅ survives | ❌ **no guard exists** | n/a |

The peer test is the one that matters most: it confirms a faulting session does
not turn one zero into two. The co-resident rollout stays byte-identical to its
solo run while its neighbour's inference is dying.

### Latency under 16×2 load — PASS

```
=== Drive latency @ 16x2 (192 calls, 0.9s wall, GIL-bound single core) ===
  mean       5.40 ms   [budget 100 ms]
  p50        4.87 ms
  p90       12.86 ms
  p99       13.51 ms
  max       13.55 ms
  over-budget calls: 0/192
  headroom on mean:  19x
```

Worth noting: mean rose **2.58 ms solo → 5.40 ms** under 2-way concurrency, and
p90 is ~2.4× the median. That is the process-wide `RLock` held across all of
`_plan()` serializing the two co-resident rollouts — exactly the cost the earlier
isolation audit predicted and accepted. It is still 19× inside budget, so the
trade (proven isolation for latency we have in abundance) remains the right one.

Caveat on the numbers: measured on a shared login node, GIL-bound to one core,
with `_plan` standing in for a model forward that does not exist yet. Treat them
as an upper bound on the *geometric* driver's cost, not as a prediction for the
student.

## 3. FINDING — no deadline guard on the Drive path

**The charter asks that an RPC overrun (>0.1 s) yield a safe fallback. That
behaviour does not exist.**

```
$ grep -n -iE "timeout|deadline|watchdog|perf_counter|signal.alarm|budget" submission/route_follower.py
  (no matches)
```

There is no watchdog anywhere in the driver. A slow inference simply *takes*
longer — it is not aborted, and no fallback is substituted. Verified by
`test_slow_inference_does_not_throw_but_REVEALS_NO_DEADLINE_GUARD`, which injects
a 0.25 s stall and asserts the call both survives **and** demonstrably exceeds
budget. The test pins current behaviour rather than asserting a guard that isn't
there; if a watchdog is added, that assertion is the one to flip.

### Why this is not currently a zero-rate risk

The shipped driver's worst observed Drive is **13.55 ms** — 7× under budget with
no pathological path available to it: `_plan` is bounded work (a 51-step loop over
a resampled polyline), there is no I/O, no allocation cliff, and no model. For the
*geometric* driver the missing guard is latent, not live.

### Why it matters anyway

The budget is on the **mean**, so amortised/dual-rate inference is legitimate —
but a student with a heavy forward, a cold CUDA cache on first call, or a
pathological input could blow a single call badly. With no watchdog, that call
returns late instead of returning a safe stop on time. Whether a late trajectory
scores worse than a safe-stop trajectory depends on evaluator behaviour I have
not verified, so I am not going to claim which is worse — but the *option* to
choose doesn't currently exist.

### Proposed fix — NOT APPLIED

Deliberately not written: it is a behavioural change to the shipped rank-1 path
and would need its own regression gate, and the driver that needs it (the
student) does not exist yet. Landing it now would be unjustifiable under the
four-axis rule — it moves (a) zero-rate only for a driver we haven't built.

Shape, for when the student lands:

```python
# in drive(), around the _plan call
DEADLINE_S = float(os.environ.get("RF_DEADLINE_S", "0.08"))   # < 0.1 budget

t0 = time.monotonic()
try:
    with self._lock:
        ...
        traj = self._plan(s, req.time_now_us)
except Exception:
    traj = fallback(s, req.time_now_us)
if time.monotonic() - t0 > DEADLINE_S:
    LOG.warning("drive overran %.3fs -> emitting cached/safe plan", ...)
    # prefer s.last_plan re-anchored to now, else fallback()
```

Two design notes for whoever implements it:

1. **A post-hoc check cannot un-spend the time.** Detecting the overrun *after*
   `_plan` returns still pays the latency. Genuine protection needs either a
   watchdog thread that returns a fallback while `_plan` keeps running, or a
   co-operative deadline checked inside the plan loop. The watchdog is the only
   version that bounds worst-case latency, and it costs a thread per in-flight
   Drive.
2. **`s.last_plan` is already maintained** (`route_follower.py:244`) and is
   currently dead state. A re-anchored previous plan is a better degradation than
   a hard stop when the route is still valid — that is the natural use for it.

## 4. Scope note — the RNG clause

Per charter R2/R3, the per-`Session` `torch.Generator` requirement is
`policy_driver`-scoped. `test_shipped_driver_has_no_rng_at_all` asserts the
shipped driver references none of `random` / `torch` / `manual_seed` /
`Generator` / `np.random`, so the clause is vacuous on this path — and the test
will **fail loudly** if that ever stops being true, at which point the RNG
assertions must be added here. That is the tripwire, not an exemption.

## 5. New zeros / near-zeros surfaced

None from the driver. The harness found no contamination, no handler death, and
no budget breach. The one gap is the missing deadline guard (§3), which is latent
for the shipped driver and actionable only when the student exists.
