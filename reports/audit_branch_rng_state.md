# CHECK 2 — RNG state on `audit/state-isolation`

# VERDICT: ABSENT

**There is no RNG change on the branch.** `rl/policy_driver.py` and
`rl/residual_policy.py` are byte-identical to `main`. No commit on
`main..audit/state-isolation` touches either file, and the full diff of `rl/`
between the two branches is **0 lines**.

The concern that motivated this check — *"a half-done global-RNG edit could
later get cherry-picked onto something submission-bound by accident"* — **does
not apply**: there is nothing to cherry-pick. No commit on this branch carries
any change under `rl/`.

The underlying defect the earlier audit *documented* is still present and still
unfixed, on `main` and on this branch alike: `rl/residual_policy.py:118` samples
from the global torch RNG. That was reported as a finding, never as a change.
§4 below specifies the correct fix for whenever it is actually undertaken —
**not applied here.**

*Read-only audit. Nothing was edited, committed, merged or pushed.*

---

## 1. The five commits

```
$ git log --oneline main..audit/state-isolation
07af779 Commit the replay fixture; narrow the data/ ignore rule to repo root
cd145f8 Isolation harness: replay REAL recorded rollout.asl streams
0ef2c00 Audit report: state isolation proven sound; Drive-contract defect found and fixed
c6145a9 Fix: drive() never raises on unknown/closed session + dev-only state guard
71eb489 State-isolation harness: 2x concurrent + 120 randomized interleavings, bit-identical assertion
```

| Commit | Files touched | Touches `rl/`? |
|---|---|---|
| `71eb489` | `tests/test_state_isolation.py` | no |
| `c6145a9` | `driver/route_follower.py`, `driver/route_follower_term.py`, `driver/abstention_driver.py`, `submission/route_follower.py`, `tests/test_state_isolation.py` | no |
| `0ef2c00` | `reports/state_isolation_audit.md` | no |
| `cd145f8` | `reports/state_isolation_audit.md`, `tests/extract_replay_streams.py`, `tests/test_state_isolation.py` | no |
| `07af779` | `.gitignore`, `tests/data/replay/*.json` (10) | no |

## 2. Evidence

### 2.1 No commit touches the RNG files

```
$ git log --oneline main..audit/state-isolation -- rl/policy_driver.py rl/residual_policy.py
(no output)
```

### 2.2 The whole `rl/` tree is unchanged

```
$ git diff main audit/state-isolation -- rl/ | wc -l
0
```

### 2.3 Both files are byte-identical to `main`

```
$ diff <(git show main:rl/residual_policy.py) <(git show audit/state-isolation:rl/residual_policy.py)
IDENTICAL to main
$ diff <(git show main:rl/policy_driver.py) <(git show audit/state-isolation:rl/policy_driver.py)
policy_driver.py IDENTICAL to main
```

### 2.4 The sample call is untouched — still global RNG

**`rl/residual_policy.py:118`** on `audit/state-isolation`:

```python
115        if deterministic:
116            pre = mean
117        else:
118            eps = torch.randn_like(mean)          # <-- global RNG, unchanged from main
119            pre = mean + std * eps
```

Reached from **`rl/policy_driver.py:239`**:

```python
239        m, logp, val, pre_sample = self.policy.act(img, route_t, ego_t, deterministic=DETERMINISTIC)
```

No `torch.Generator`, no `generator=` argument, no `manual_seed`. Exhaustive grep
for every torch RNG entry point across the branch:

```
$ git grep -n -E "manual_seed|torch\.Generator|randn_like|torch\.normal|torch\.rand|multinomial|\.sample\(|default_generator" audit/state-isolation -- '*.py'
rl/residual_policy.py:118:            eps = torch.randn_like(mean)
rl/residual_policy.py:154:    img = torch.rand(B, 3, 180, 320, device=dev)     # _smoke() only, not the Drive path
rl/residual_policy.py:155:    route = torch.rand(B, 10, 3, device=dev)         #   "
rl/residual_policy.py:156:    ego = torch.rand(B, 3, device=dev)               #   "
```

`torch.manual_seed` appears **nowhere** on the branch, so the BROKEN case
(global reseed at drive time) can be ruled out positively, not just by absence
of a diff.

### 2.5 What *could* be mistaken for an RNG change

The branch does contain RNG-shaped code, all of it confined to
`tests/test_state_isolation.py` and all of it **stdlib `random.Random`, never
torch**:

| Location | What it is |
|---|---|
| `tests/test_state_isolation.py:826` `GlobalRngPolicy` | A **stub** modelling the bug — negative control asserting a shared global stream *does* leak |
| `tests/test_state_isolation.py:841` `PerSessionRngPolicy` | A **stub** demonstrating what a correct per-session generator would do |
| `:242`, `:248`, `:363`, `:501`, `:599`, `:868` | Seeded `random.Random` for reproducible test scenarios and interleavings |

`tests/test_state_isolation.py` imports `random` (line 28) and **does not import
torch at all**. These stubs never touch `rl/`, are never imported by the driver,
and ship nowhere. A reviewer skimming for "RNG work on this branch" will find
them — they are test scaffolding illustrating a documented finding, not a
partial fix.

Similarly, `random_seed` appears at `tests/…:277`, `:419`, `:680` etc. — that is
the harness populating `DriveSessionRequest.random_seed` on synthetic requests,
and `tests/extract_replay_streams.py:68` reading it out of real logs. Neither
feeds any sampler.

## 3. Classification against the four options

| Option | Applies? | Why |
|---|---|---|
| CORRECT | no | No `torch.Generator` exists anywhere on the branch |
| BROKEN | no | `torch.manual_seed` appears nowhere; no global reseed was introduced |
| PARTIAL | no | Nothing was started: `rl/` diff is 0 lines. The specific PARTIAL failure mode to watch for (generator created but not threaded into the sample op, or passed to a `torch.distributions` `.sample()` that silently ignores it) **is not present** — there is no generator and no `.sample()` call in `rl/` |
| **ABSENT** | **yes** | No RNG change present on the branch |

## 4. The fix, specified but NOT applied

Not required by this verdict — recorded so the work is specified when someone
takes it on. **This is training-only code; it does not ship.** `rl/policy_driver.py`
is not in the submission image and needs torch, which the driver env does not have.

The defect: with `RL_DETERMINISTIC=0`, two rollouts in one process draw from one
global torch stream, so rollout B's actions depend on how many samples rollout A
happened to take first. Not reproducible, not isolated. Masked at eval only
because eval sets `RL_DETERMINISTIC=1`.

Correct shape:

1. **`rl/policy_driver.py`** — add a per-`Session` generator, seeded from the
   runtime-issued seed (`DriveSessionRequest.random_seed`, already in the proto
   and currently discarded at `policy_driver.py:134`):

   ```python
   @dataclass
   class Session:
       ...
       gen: object = None          # torch.Generator, per rollout

   def start_session(self, req, ctx):
       s = Session()
       g = torch.Generator(device=_DEV)
       g.manual_seed(int(req.random_seed) or hash(req.session_uuid) & 0x7FFFFFFF)
       s.gen = g                   # seeding a PER-GENERATOR object, never the global
       ...
   ```

2. **`rl/residual_policy.py:118`** — thread it into the sample op:

   ```python
   # act(self, img, route, ego, deterministic=False, gen=None)
   eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=gen)
   ```

   `torch.randn_like` accepts no `generator=`, which is why the call must change
   shape rather than just gain an argument.

3. **`rl/policy_driver.py:239`** — pass it through: `self.policy.act(..., gen=s.gen)`.

Three traps, all of which produce a silent no-op or a worse bug:

- **Do not** call `torch.manual_seed()` at drive time. That reseeds the *global*
  generator and makes the race worse — every concurrent rollout would yank the
  shared stream out from under the others.
- **Do not** route this through `torch.distributions.Normal(...).sample()`; that
  API ignores a generator. Use `torch.normal`/`torch.randn` with `generator=`.
- Creating the generator but not passing it to the sample call leaves behaviour
  unchanged while looking fixed. The regression test must assert two interleaved
  sessions produce identical action sequences to their solo runs — the existing
  `test_per_session_rng_is_isolated_under_interleaving`
  (`tests/test_state_isolation.py:863`) is that assertion's shape, currently
  written against a stub; point it at the real policy once torch is available in
  the test env.

## 5. Bottom line

`audit/state-isolation` is safe to treat as shareable with respect to RNG: it
carries **zero** changes under `rl/`. Anyone cherry-picking from it cannot
accidentally pick up an RNG edit, because none exists. The global-RNG defect
remains open on `main` as a documented, unfixed finding.
