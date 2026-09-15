# Phase 1.1 — container budget, measured

**Source: the evaluator's own record for our rank-1 submission**, not a local
build. `alpasim_challenge.py status 1bb06e8b-9d20-4018-98e1-5ba612e4e44c`
returns the size and digest of the image as the organizers received it, which is
a *better* number than a local `docker images` reading would be — it is what was
actually evaluated. Docker is not installed on this cluster, so a local build
measurement was not available anyway.

## The rank-1 image

| Field | Value |
|---|---|
| `submitted_image_uri` | `696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/na-la:sub1` |
| `image_digest` | `sha256:75e9090a45af58095d87630f8b1f0707d28b638d2e7a0138508a0a95827eee0c` |
| `resolved_image_uri` | `…/teams/na-la@sha256:75e9090a…` (immutable pin) |
| `image_size_bytes` | **99,022,867** |
| `status` | `SUCCEEDED` |
| `score` | `0.9804824187258063` |

## Budget compliance

| Constraint | Limit | Measured | Headroom |
|---|---|---|---|
| Image size | 40 GiB | **94.4 MiB** | **434×** — 0.23% of budget |
| VRAM / replica | 16 GiB | **0 allocated** | full budget unused |
| Mean Drive RPC | 0.1 s | 5.40 ms @ 16×2 (local harness) | 19× |
| `/tmp` | 2 GiB | **0 B** — driver performs no disk writes | full |
| `/run` | 64 MiB | 0 B | full |
| Outbound network | none | none — no sockets opened beyond the gRPC listener | ✅ |
| Read-only rootfs | required | compatible — nothing written outside the listener | ✅ |

VRAM is 0 because `submission/route_follower.py` is pure numpy: no torch, no
CUDA context, no GPU allocation at all. Confirmed by
`tests/test_stress_16x2.py::test_shipped_driver_has_no_rng_at_all`, which asserts
the shipped source references neither `torch` nor any RNG.

The `/tmp` and `/run` figures come from static inspection — the module contains
no `open(...)` for write, no `savez`, no `tempfile`, no `mkstemp` (verified
during the state-isolation audit). Not from a runtime filesystem trace; a
container run would be strictly better evidence and needs Docker.

## The constraint that is actually tight — throughput, not size or latency

```
throughput_limit:
  observed_wall_time_s   2157.161
  limit_wall_time_s      2485.438
  margin_s                328.277        -> 13.2% headroom
  passed                 true
```

Our `README.md` concluded "latency is NOT the bottleneck … 35× headroom" from
the per-call 0.1 s budget. That is correct per-call and **misleading in
aggregate**: there is also a hard wall-time ceiling on the whole evaluation, and
the rank-1 run consumed **87% of it**.

Today the driver is a rounding error inside that number — at ~2.6 ms/call the
wall time is dominated by the renderer and simulator, not by us. But the eval
spec is 200 Drive calls per rollout (`n_sim_steps=200`, `control_timestep=500ms`),
so a student forward costing tens of ms per call multiplies into the same order
of magnitude as the 328 s margin.

**Open item, needs one number I do not have:** converting that margin into a
per-call allowance requires the rollout count for the submission, which is not in
the status JSON (`rollout_coverage: null`). Worth obtaining before D2 — it is
plausibly a harder constraint on the student than the 0.1 s mean, and it
determines whether dual-rate inference is optional or mandatory. I am not going
to guess the arithmetic without that count.

## Status of the remaining Phase 1 items

| Item | State |
|---|---|
| 1.1 image size | ✅ **measured, 94.4 MiB** (from the evaluator's record) |
| 1.1 VRAM | ✅ 0 — pure numpy, no CUDA context |
| 1.1 `/tmp`, `/run`, network, read-only rootfs | 🟡 static inspection only; a container run would be better evidence |
| 1.2 ECR push mechanism | ⛔ needs Docker — not installed here |
| 1.3 shipped path == best image, fix not merged | ✅ verified: `fix/drive-handler-fallback` is not an ancestor of `main`; `main` still has the aborting lookup on the Drive path |
| 1.4 `+e2e_challenge=dev` smoke | ⛔ needs renderer + controller + GPU |
| 1.5 submission → ECR-URI logging | ✅ `tools/log_submission.py`; rank-1 record now complete, audit reports 0 missing |
