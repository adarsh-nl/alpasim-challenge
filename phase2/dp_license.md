# Diffusion-Planner licence provenance — evidence

**Gathering evidence, not adjudicating.** I am not qualified to give a legal
verdict and this is not one. Below is what the public record does and does not
establish, separated explicitly.

**Headline: better than feared, and the ambiguity is narrower than it looked.**
The *code* repo has no licence at all. The *weights* — the artifact we actually
distil from — carry an **explicit MIT declaration** on the official HuggingFace
repo the README directs you to. Those are two different artifacts with two
different answers, and conflating them is what made this look worse than it is.

---

## 1. What is determinable

### 1a. The weights are declared MIT — explicitly

The README (line 111) tells you to download the checkpoint from
`https://huggingface.co/ZhengYinan2001/Diffusion-Planner`. That repo's model
card is, in its entirety:

```
---
license: mit
---
```

HF API confirms: `cardData.license = "mit"`, tag `license:mit`. Files present are
exactly the two the README instructs you to fetch — `args.json`, `model.pth`.

**This is the most directly relevant signal we have**, because the teacher
artifact in our pipeline is the checkpoint, not the training code.

### 1b. The code repo has no licence file — confirmed authoritatively

```
$ curl -s api.github.com/repos/ZhengYinan-AIR/Diffusion-Planner | jq .license
  null

$ curl -s api.github.com/repos/.../contents/ | (root listing)
  ['.gitignore','README.md','assets','data_process.py','data_process.sh',
   'diffusion_planner','normalization.json','nuplan_train.json',
   'requirements_torch.txt','run_nuboard.ipynb','setup.py',
   'sim_diffusion_planner_runner.sh','sim_guidance_demo.sh','torch_run.sh',
   'train_predictor.py']
  LICENCE-LIKE FILES: NONE
```

GitHub's own licence detection returns `None`. No `LICENSE`, `LICENCE`,
`COPYING` or `NOTICE` at root. Repo created 2024-11-21, last pushed 2026-03-10,
1057 stars — a well-known repo that simply never added one.

### 1c. Recency cannot adjudicate the `setup.py` contradiction

`setup.py` contains both:

```python
classifiers=[ ..., "License :: Free for non-commercial use" ],   # non-commercial
license="MIT",                                                    # permissive
```

Upstream, **`setup.py` has been touched exactly once**:

```
1 commit(s) touching setup.py
   d3993dc4  2025-01-26  code release
```

Both claims were introduced in the same initial commit and never revisited.
There is no "newer wins" resolution available. (Our vendored copy at
`rl/dp_repo` is a shallow clone — 1 commit, `is-shallow-repository true` — so
local blame was impossible; this came from the upstream API.)

### 1d. No LICENSE file was ever removed or renamed

Searched the upstream history for added/deleted/renamed licence files: none. The
absence is original, not the result of a deletion.

### 1e. The authors' other repos show no non-commercial pattern

| Repo | Licence |
|---|---|
| `airbot_libero_rlds_dataset_builder` | MIT |
| `susie-airbot-libero` | MIT |
| `Vibe-Trading` | MIT |
| `awesome-diffusion-model-in-rl` | Apache-2.0 |
| `Diffusion-Planner` | **NONE** |
| `FISOR`, `Hyper-Diffusion-Planner`, `OMIGA`, … | NONE |
| `ml-starflow` | NOASSERTION |

Where this author does declare a licence, it is MIT or Apache-2.0. **A
non-commercial licence appears on none of their repositories.**

### 1f. The paper adds nothing on code licensing

arXiv 2501.15564 (ICLR 2025, OpenReview `wM2sfVgMDH`): the arXiv page carries
**CC BY-SA 4.0**, which governs the *paper text only*. No code-availability
statement, no ethics statement, no usage restriction, no code or checkpoint
licence on the abstract page.

---

## 2. What is genuinely ambiguous

1. **Whether the training code is MIT or unlicensed.** `setup.py`'s
   `license="MIT"` field is a declaration; the absence of a LICENSE file means
   GitHub — and a cautious reader — treats the repo as all-rights-reserved. These
   point opposite ways and only the authors can settle it.
2. **Whether the non-commercial classifier carries any weight.** It is a PyPI
   trove classifier, in a package never published to PyPI, sitting beside an
   explicit `license="MIT"` in the same file from the same commit. That reads
   like boilerplate carried in from a template — but that is my *inference*, not
   a determination, and I will not present it as one.
3. **Whether a student distilled from these weights is a derivative work of
   them.** This is unsettled in general and I am explicitly not opining. It
   matters less than it might: see §3.

## 3. Why this is probably resolvable without the authors

If the weights' MIT declaration governs the checkpoint, the Track B path is
workable *even on the pessimistic reading of the derivative question*, because
MIT:

- permits commercial use,
- permits relicensing/incorporation into an Apache-2.0 work (MIT → Apache-2.0 is
  a standard, permitted direction), and
- requires only that the notice accompany substantial portions.

So the worst case under MIT is an **attribution obligation**, not a bar. That is
compatible with §5's Apache-2.0-to-the-public requirement, provided we carry the
MIT notice.

The code's ambiguity matters less than it first appears, because the code is
**training infrastructure, not part of the Submission**. §5 reaches the
Submission (the image and what is in it); the teacher never ships.

**Separate issue I found while checking this — flagging it because it is real and
independent:** we have vendored the unlicensed DP code into our own repo at
`rl/dp_repo` (currently untracked). If `alpasim-challenge` is ever published —
and §5 pushes everything toward openness — redistributing third-party code with
no licence is its own problem, unrelated to the Submission question. Cheapest fix
is to not vendor it: keep it outside the repo, or add it as a pinned clone step
in a script. Worth doing regardless of how the licence question lands.

## 4. Drafted question for the authors — 3 lines, for you to send

> Hi — thanks for releasing Diffusion-Planner; we're using it as a distillation
> teacher for the NVIDIA AlpaSim E2E Challenge (nuPlan track), which requires our
> own submission be released under Apache-2.0.
>
> The repo has no LICENSE file, and `setup.py` declares `license="MIT"` alongside
> the classifier `"License :: Free for non-commercial use"` — while the HF
> checkpoint is tagged `license: mit`. Could you confirm which governs the code
> and the released weights?
>
> Specifically: are we permitted to distil from the released checkpoint and
> release the resulting student model under Apache-2.0, with attribution to your
> work?

## 5. Recommendation on the gate

The blocker is **narrower** than when we flagged it: the artifact we distil from
carries an explicit MIT tag, and the author's pattern corroborates MIT intent.
That is decent evidence — but it is a one-line tag against a contradictory
`setup.py`, so my read is that it is **not yet strong enough to commit a large
GPU spend on**, and one email closes it definitively.

Suggested sequencing: send §4, and in the meantime let Task 2's fallback
(§ below / `tuplan_garage`, Apache-2.0) de-risk the schedule so Track B is not
single-threaded on a reply.

---

## Task 2 — cleanly-licensed fallback teachers

**`autonomousvision/tuplan_garage` (PDM-Closed) is the strongest fallback:
Apache-2.0, and in-domain.** GitHub reports `NOASSERTION` for it only because
the LICENSE file opens with a copyright line before the Apache text — the file
itself reads *"Copyright 2023 Daniel Dauner, Marcel Hallgarten, Andreas Geiger,
and Kashyap Chitta / Licensed under the Apache License, Version 2.0"*, so it is
unambiguously Apache-2.0. PDM-Closed won the 2023 nuPlan planning challenge, is
rule-based/hybrid with full map + agent access (exactly the privileged-teacher
profile we want), and being Apache-2.0 it is directly compatible with §5 with no
relicensing question at all. `motional/nuplan-devkit` is the same case —
Apache-2.0 behind a `NOASSERTION` label — and supplies the map/agent interfaces.
Also clean and worth noting: `autonomousvision/navsim` (Apache-2.0, 1099 stars)
and `opendilab/DI-drive` (Apache-2.0), and the organizers' own `NVlabs/alpasim`
is Apache-2.0. I did not chase NVIDIA's "Alpamayo 2 Super" licence in this pass —
our own repo history already records Alpamayo/BEV as ruled out on capability
grounds (commit `e2e8841`), so it is not a live candidate and I deprioritised it
rather than spend the lookup. Net: **Track B need not be single-threaded on the
DP reply** — PDM-Closed is a licence-clean, in-domain substitute, at the cost of
a weaker teacher (rule-based rather than learned) and a re-run of the label
pipeline.
