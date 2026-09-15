# Terms review — `official-2026-08` (ACCEPTED 2026-09-15)

Full text captured in `phase0/terms_text.md` (16 sections, 151 lines).
Status in `phase0/terms_status.txt`: accepted by the captain, `team_ready: true`.

**This review is post-acceptance.** Adarsh accepted before the institutional
check he had flagged wanting to do. Nothing here is legal advice — it is a
reading of what the text says, with the two items that warrant follow-up marked.

---

## Two items needing action

### A. §8(a)/(b) — the university representation, now made

> §8(a) *"…has the full right and authority to submit the Submission and grant
> the rights described in this Agreement, including that the Submission **does
> not violate any applicable employment agreement, company policy, university
> policy**, or third-party obligation"*
>
> §8(b) *"…has obtained all necessary permissions to make the Submission
> available under Apache 2.0"*

By accepting, you warranted both. This is precisely the institutional angle you
wanted to check first, and §5 requires the Submission be licensed **Apache 2.0 to
the public** (see B). If UT asserts institutional ownership of IP created by
employees — common in university IP policy — then releasing it under Apache 2.0
plausibly needs UT sign-off, and §8(b) asserts you already have it.

The consequence is not abstract:

> §8, final paragraph: *"ANY SUBMISSION THAT IS IN BREACH OF ANY OF THE
> FOREGOING REPRESENTATIONS AND WARRANTIES … WILL BE DEEMED VOID AND
> AUTOMATICALLY DISQUALIFIED FROM THIS CHALLENGE."*

**Recommendation:** get a short written confirmation from UT research support /
tech transfer that open-sourcing this work under Apache 2.0 is permitted. Most
likely it is — universities routinely permit open release of research code — but
the check is cheap and the downside is disqualification. Doing it retroactively
is fine; leaving it unchecked is the risk.

Related, same conversation: §7 has you consent to the leaderboard displaying
your **organization name**, and warrant you have *"the permission of any
affiliate"* for any affiliation implied by displayed details. If the displayed
entry implies UT, that permission should be part of the same confirmation.

### B. §5 — everything we submit becomes public Apache 2.0, plus a broad NVIDIA licence

> *"All Submissions must be made available under the Apache License, Version
> 2.0"* … and, additionally, Contestant grants Sponsor *"a perpetual,
> irrevocable, worldwide, non-exclusive, royalty-free, and fully sublicensable
> right and license to use, reproduce, modify, adapt, publish, translate,
> display, create derivative works from, and distribute the Submission for any
> purpose … This includes the right to make the Submission fully open-sourced and
> available to the public. Any recipient … may use the Submissions for both
> commercial and non-commercial purposes."*

Practical consequences:

- The shipped driver, and any student model we submit, is **public Apache 2.0**.
- Any proprietary or patent route for this work is foreclosed.
- This is *aligned* with our goals — the technical report and the Innovation
  prize both want openness — so it is a constraint to note, not to fight.
- It is also the clause that makes A above load-bearing: you cannot grant an
  Apache 2.0 licence over work you do not have the right to license.

---

## C. BLOCKER-CLASS RISK for Track B (distillation) — teacher licence unresolved

**This is the finding with the most consequence, and it is actionable now.**

§3 restricts the nuPlan track to *"any publicly available data"*, and §5 requires
our Submission be Apache-2.0 with commercial use permitted by any recipient,
while §8(c) requires it *"does not infringe, misappropriate, or violate any
intellectual property rights … of any third party."*

Our distillation teacher is Diffusion-Planner
(`rl/dp_repo`, origin `github.com/ZhengYinan-AIR/Diffusion-Planner`, Tsinghua AIR).
Its licence is **ambiguous and self-contradictory**:

```
$ git -C rl/dp_repo ls-files | grep -i licen
  (NONE tracked -- there is no LICENSE file in the repo at all)

$ sed -n '16,22p' rl/dp_repo/setup.py
    classifiers=[
        ...
        "License :: Free for non-commercial use",     # <-- non-commercial
    ],
    license="MIT",                                    # <-- MIT permits commercial
```

`license="MIT"` and `"Free for non-commercial use"` are mutually exclusive
claims, and no LICENSE file exists to adjudicate between them.

### Why this could void a submission

If the teacher is in fact non-commercial-only, then student weights distilled
from it are plausibly a derivative of a non-commercially-licensed model. We
would then be unable to license the student under Apache 2.0 with commercial use
permitted (§5), which would breach §8(b)/(c) — and §8's final paragraph makes
that *"void and automatically disqualified."*

Note the risk is about the **licence**, not availability: the checkpoint is
publicly available, so §3 is satisfied either way.

### Resolve before spending GPU time on D1/D2

Cheapest-first:

1. Check the upstream GitHub repo and the paper for an authoritative licence
   statement (our vendored copy may simply predate one being added).
2. If still ambiguous, ask the authors directly for a written licence
   clarification — they are responsive academics and this is a normal request.
3. If it is non-commercial, the distillation plan needs a differently-licensed
   teacher, or the student must be trained without teacher-derived supervision.

**Recommendation: treat this as a hard gate on Track B, ahead of D1.** Staging 15
shards of teacher labels and a student training run is a large GPU spend to make
on a teacher we may not be permitted to ship a derivative of. This is
inexpensive to resolve and expensive to discover late.

---

## D. Noted, no action required

| Clause | Point |
|---|---|
| §2 | Eligibility: 18+, eligible jurisdiction, not in sanctioned countries (list includes PRC, Russia, Belarus, Iran, NK, Syria, Cuba, occupied Ukrainian regions). NL-based participation is fine. |
| §2 | **Captain binds the whole team.** You warranted that *all* team members have read and agree, and that you have authority to bind each. If `na-la` is more than you, tell them. |
| §6(c) | You **covenant not to assert** IP or other claims against NVIDIA arising from its independent development in AV/CV/AI, to the extent related to your participation or the Submission's subject matter. Broad, and standard for this kind of contest. Nothing to do. |
| §6(b) | Participation creates **no confidentiality or non-use obligation** on NVIDIA. |
| §7 | NVIDIA judges' decisions are **final**. |
| §13 | Sponsor may modify/cancel the Challenge, disqualify for technical tampering, and **remove any Submission subject to a third-party IP claim** — the enforcement hook for C above. |
| §14 | Delaware law; exclusive jurisdiction Santa Clara, CA. Non-negotiable. |
| §15 | No implied licence to NVIDIA software — does not affect our SDK use, which is governed separately. |
| §4 | Submission limits live on the Challenge Page, not in the terms; *"Contestant's responsibility to check … regularly"* for deadline changes. |

## Recorded acceptance details

| Field | Value |
|---|---|
| `terms_version` | `official-2026-08` (footer: "v. August 2026") |
| `terms_sha256` | `8166fed2a553707bb3a467bef029156abb6ef2eaf02580562069a3039b89236f` |
| Accepted by | captain (`actor_role: captain`, `captain_accepted: true`) |
| `team_ready` | `true` — team is now eligible to upload and submit |
| `enforcement_enabled` | `true` |

Re-acceptance is required if the terms are revised, so keep that version+sha.
