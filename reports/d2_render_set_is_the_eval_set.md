# STOP — the only renderable scenes are the evaluation scenes

**Design fork, reported before staging anything.** Tasks 3 and 5 are held; the
render pass and re-targeted teacher sbatch are **not** staged, because both would
bake in training on the evaluation set.

**Four-axis:** n/a — this is a scoping/integrity finding, not a change.

---

## The finding

```
navtest_full scene_ids (the OFFICIAL ec2 eval set) : 1485
navtest/assets splat dirs (renderable)             : 1491
  intersection                                     : 1485
  assets NOT in navtest_full                       :    6

shard-0 labelled scenes that are renderable        :  100
  ...of those, in the OFFICIAL eval set            :  100
local gate set sample200                           :  200, all within navtest_full
```

`src/wizard/configs/e2e_challenge_nuplan/ec2.yaml` — documented upstream as the
preset both official tracks use — selects `nuplan_scenes: navtest_full`.

**So the renderable set is the evaluation set.** Of 1,491 scenes with Gaussian
splats, 1,485 are exactly `navtest_full`. Six are not. There is no meaningful
pool of renderable non-eval scenes.

This was not visible earlier: I had established that only ~12% of teacher labels
are renderable, but not *which* 12%. It turns out to be precisely the wrong 12%.

## Why this stops Task 3 rather than being solved by a held-out split

The charter asks for a fixed held-out split "so the local sweep stays a valid
gate". A split does fix **gate validity** — but it does not fix the larger
problem, and I do not want to implement the split and let it read as solved.

Splitting 1,485 into, say, 1,185 train / 300 val gives a clean local gate. But we
would still have trained the student on **1,185 scenes that are scored**. That is
training on the test set. Charter rule 6 — *"Never game the private test set. No
scene fingerprinting, memorized trajectories, or eval-infra exploits"* — is aimed
squarely at this, and a model fitted to renders of the scored scenes is the
memorised-trajectory failure mode arriving by a different route.

Note the asymmetry: the *labels* are fine. They come from the trajdata cache
(12,147 scenes), and only ~12% coincide with eval. It is the **observations**
that are pinned to the eval set, because observations require splats and splats
exist only for navtest.

## One thing I could not resolve, and it changes the severity

**Is `navtest_full` the actual scored set, or only our local copy of a dev set?**

Evidence it is the scored set:
- `ec2.yaml` is documented as the preset both official tracks use, and it selects
  `navtest_full`.
- Our README already recorded "Official (`ec2`): navtest_full".

Evidence it may not be:
- The warmup track runs "**32 private scenes**", so the organizers demonstrably
  hold scenes we do not.
- Issue #122 showed the evaluator syncs scene data from
  `s3://alpasim-scene-data-use1/nuplan/worldengine/` — an organizer-side store
  that need not match our staged copy.
- The migration to "final" nuPlan scenes (still unconfirmed live) implies the
  scored set has changed at least once, independently of what we hold.

If the scored set is private and `navtest_full` is merely a public dev set, then
training on it is **normal and legitimate** — it is a dev set, and the held-out
split is exactly the right answer. If the scored set *is* `navtest_full`, then
training on it is test-set training and we should not do it regardless of whether
the rules technically permit it (§3 permits "any publicly available data", and
navtest_full ships publicly — so this is likely permitted by the letter and still
wrong by our own standard).

**This is an organizer question and it is cheap to ask.** Suggested phrasing:

> Is the official nuPlan-track evaluation run on the public `navtest_full` scene
> set shipped in the challenge dataset, or on a private scene set? We want to be
> sure we don't train on scored scenes.

## Options, assuming the worst case (navtest_full is scored)

| # | Option | Cost | Verdict |
|---|---|---|---|
| 1 | Train on the 6 non-eval renderable scenes | ~48 samples | Useless — too few |
| 2 | Obtain splats for non-eval scenes | Build MTGS splats from raw nuPlan logs we do not have | Very large; effectively a new project |
| 3 | **Train on real nuPlan camera frames** for non-eval scenes | Multi-TB staging; reintroduces the render/real domain gap | **Now the only viable camera route** |
| 4 | Drop camera input; use the teacher as a prior/gating signal only | Small | Preserves integrity, sacrifices the capability bet |
| 5 | Train on eval renders anyway | Free | **Rejected** — rule 6 |

Option 3 deserves attention because **this finding inverts my earlier
recommendation**. I previously argued against real frames on the grounds that
renders have no domain gap. That argument holds only if we may train on renders
of eval scenes. If we may not, real nuPlan data for *non-eval* logs becomes the
only way to build a camera student at all — and the domain gap becomes a cost to
manage (renderer fine-tune, exactly as the original charter proposed) rather than
a reason to avoid it.

So the charter's original instinct — real frames with a planned renderer
fine-tune — may have been right for a reason neither of us had identified.

Option 4 is worth taking seriously rather than treating as a fallback. The
information-gated distillation idea does not require the student to consume
camera input at all: the teacher's map-dependence signal could gate the existing
geometric driver's abstention behaviour, which is cheap, ablatable, needs no
renders, and targets `dd162898`-class zeros directly. That is a smaller bet with
a much shorter path to a gated improvement.

## What I have and have not done

Not staged (deliberately): the re-targeted teacher sbatch, the held-out split,
the splat render pass. All three encode "train on navtest renders" as the plan.

Done and unaffected:
- Alpamayo licence gate — **passes**, see `phase2/alpamayo_teacher_feasibility.md`.
  Its teacher labels come from trajdata and are not affected by this finding.
- `rl/policy_driver.py` decode fix + budget guard (authorised, independent).

Task 4 (far-horizon feasibility audit) is **not** blocked — it audits teacher
labels against privileged map/agent data and needs no renders. The charter says
to prioritise renderable-scene labels; given this finding I would instead audit
the **full** shard-0 set, since which scenes we ultimately train on is now open.
I will proceed on that basis unless told otherwise.
