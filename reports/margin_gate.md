# Near-miss margin gate — in, and it separates the tail

**Four-axis:** instrumentation — (a) 0 (b) 0 (c) 0 (d) 0 directly. It exists
because the local set is saturated on score, so nothing else can be measured
without it.
**Smoke:** `python3 tools/zero_dashboard.py --baseline <A> --candidate <B>`
→ runs on all four existing A/B pairs, CPU, <1 s each.
**Gate state:** tool merged to `cc/analysis-and-tooling`; no driver changed.

---

## Does it separate the near-miss tail? Yes — and the answer is uncomfortable

Run against every A/B pair we have on disk:

| pair | tight-scene margin gain | progress Δ | verdict |
|---|---|---|---|
| `degen9` baseline → **stop03** | **+0.831 m** | **−0.4774** | **HOLD** — bought by braking |
| `gov` baseline → **governor** | +0.002 m | −0.0007 | INCONCLUSIVE — below materiality |
| `fidelity` baseline → **term-decel** | +0.043 m | −0.0256 | **HOLD** — bought by braking |
| `abstain` follower → **abstention** | +0.017 m | −0.0980 | **HOLD** — bought by braking |

**Every intervention we have ever built buys margin by braking.** Not one
achieved a margin gain without a progress cost. That is a real finding about the
whole "slow down to be safe" family, and it is exactly what the gate was asked to
detect.

The clearest case is `stop03`: **+0.83 m** of tight-scene margin — which a naive
margin gate would have called a clear win — against a **−0.477** progress
collapse. Held.

## Why the gate is a conjunction, not a margin test

Margin is trivially bought by driving slower, so margin alone is not a signal.
The gate requires all of:

1. **no new zeros** — the primary bar, checked first and overriding;
2. **material margin gain on tight scenes** (baseline margin < 1.0 m), ≥ 0.05 m
   *and* a bootstrap CI excluding zero;
3. **no score/progress regression** beyond tolerance.

Condition 2 exists because of a false positive I hit while building this: an
earlier verdict shipped the `gov-governor` pair on a **2 mm** tight-scene gain
with p = 0.49. Statistically the CI excluded zero; physically 2 mm is nothing
next to a 2 m vehicle. Materiality and significance are different questions and
the gate now asks both.

Condition 3's tell is progress *falling*, not margin rising — so the braking
check fires even when the margin gain is itself trivial, which is why
`abstention` (+0.017 m, −0.098 progress) is correctly held rather than dismissed
as noise.

## Why margin works where score does not

From the 200-scene sweep: 198/200 scenes score exactly 1.000, so a score gate
returns INCONCLUSIVE for anything outside the 2 zeros. On those same saturated
scenes, `min_distance_to_obstacle_m` ranges from **0.02 m to 9 m** — scenes that
came within 2 cm of an obstacle and still scored 1.000.

That is the whole argument: margin is a live signal on a set where score is dead,
and it is the axis the zeros live on (both known zeros reached exactly 0.00 m).
The gate focuses on scenes with baseline margin < 1.0 m because improving a
5 m margin to 6 m is worth nothing.

## A result worth pulling out of the per-scene table

`degen9 stop03` on our known zero:

```
  0.00 ->  4.02 m   score 0.00 -> 1.00   prog 1.00 -> 0.65   ...dd162898ecf351d4
  0.22 ->  0.00 m   score 1.00 -> 0.01   prog 1.00 -> 0.00   ...6d2b59f78d995d9c
```

**stop03 fixes `dd162898` and creates a new near-zero elsewhere.** Net zeros
1 → 1. A mean-score gate would have shown roughly nothing; the per-scene margin
view shows a trade, which is a completely different thing to know. This is the
kind of movement the tail instrument exists to surface.

## Caveats

- **These pairs are tiny** (5–9 scenes). Permutation p-values were 0.38–0.49,
  i.e. the *margin* deltas are not statistically resolved. The HOLD verdicts are
  driven by the progress regressions, which are large and consistent. Read the
  table as directional, not conclusive.
- The gate has **not** been run on a 200-scene pair, because we have only one
  200-scene run. Its real exercise comes with the next paired sweep.
- `MIN_GAIN_M = 0.05` is a judgement call, not a derived threshold. It is a
  constant at the top of the file precisely so it can be argued with.

## What this changes

The local set is informative again for anything that moves near-miss margin,
which is what everything downstream depends on. It also puts a number on
something we suspected: the braking family has been trading progress for margin
all along, and none of it clears a gate that checks both.

`route_follower_term.py` is the interesting one for R7 — +0.043 m margin for
−0.026 progress on 5 scenes. Under R7's corrected rationale (hold on *short
routes*, not decelerate on all routes) the trade might invert, because the
progress cost should fall on exactly the scenes where `short_gt_distance_override`
already grants full progress. That is now a testable claim rather than a hope.
