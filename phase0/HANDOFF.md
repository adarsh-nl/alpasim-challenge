# PHASE 0 HAND-OFF — secure the free rerun

**Prepared 2026-09-15. Terms acceptance deadline per charter: 2026-09-20 (5 days).**

## Bottom line, three items

1. **BLOCKER (fix first, ~2 min):** the `terms` command **does not exist** in our
   local `~/alpasim` checkout. That checkout is pinned at 2026-07-07 and predates
   the terms feature. The commands in the charter cannot run as written. Fix is
   trivial and does **not** require updating the SDK — see §2.
2. **BLOCKER (yours, G0.a):** all `terms` commands need auth. The saved token at
   `~/.alpasim/challenge.json` is from **2026-07-18** and CLI tokens expire after
   12 hours, so it is long dead. You must re-authenticate before anything else.
3. **Unrelated but important:** issue #151 shows another team took a **total zero
   on all 16 replicas** because their image hardcoded GPU names and the eval host
   is an **H100**. Our current shipped driver is immune (pure numpy, no GPU), but
   this is a direct threat to the Phase 2 student. See §5.

Nothing in Phase 0 has been executed against the API. No credential was read,
stored, or transmitted. `terms accept` has **not** been run.

---

## 1. Phase 0.1 — environment verified ✅

`uv` 0.11.28 present. The CLI runs:

```
$ cd ~/alpasim && uv run e2e_challenge/competitor_cli/alpasim_challenge.py --help
usage: alpasim-challenge [-h] [--token TOKEN]
                         {auth-url,configure-token,me,ecr-login,limits,leaderboard,submit,submissions,status}
```

### The auth commands, quoted verbatim from `e2e_challenge/competitor_cli/README.md`

Not paraphrased, not guessed — lines 23–33 of that file:

```bash
# to get the URL for authentication, go to the URL provided by:
uv run e2e_challenge/competitor_cli/alpasim_challenge.py auth-url
# and copy the token

# run the cli with configure-token and paste the token when prompted
uv run e2e_challenge/competitor_cli/alpasim_challenge.py configure-token

# confirm authentication works by checking authentication status
uv run e2e_challenge/competitor_cli/alpasim_challenge.py me
```

Run these **from the repo root** (`~/alpasim`) — the README says so explicitly.

Notes from reading the CLI source:
- `configure-token` uses `getpass.getpass()`, so the token is **not echoed** and
  will not land in your shell history. Paste it at the prompt.
- It is written to `~/.alpasim/challenge.json`, `chmod 0600`.
- **Tokens expire after 12 hours.** If you authenticate today and accept
  tomorrow, you will need to re-run `configure-token`.
- Auth binds *your* HF identity. I have not touched the credential.

## 2. Phase 0.2 — BLOCKED, and why

Our local CLI has **no `terms` subcommand**. Confirmed three ways:

| Check | Result |
|---|---|
| `--help` output | subcommands are `auth-url, configure-token, me, ecr-login, limits, leaderboard, submit, submissions, status` — no `terms` |
| `grep -rn "terms" e2e_challenge/` | only unrelated maths comments in `sample_submission_vavam/.../rectification.py` |
| local checkout HEAD | `42116f8` **2026-07-07** on branch `e2e_challenge` |

Upstream *does* have it. `git ls-remote` shows `refs/heads/e2e_challenge` at
`7611632`, i.e. the branch has moved since July. Reading the upstream file
confirms:

```python
terms = subparsers.add_parser("terms", help="View, accept, or check competition terms.")
terms_subparsers = terms.add_subparsers(dest="terms_command", required=True)
terms_subparsers.add_parser("show",   help="Print the current private terms.")
terms_subparsers.add_parser("status", help="Show your team terms status.")
terms_subparsers.add_parser("accept", help="Review and accept the current terms as the signed-in user.")
```

Endpoints: `GET /terms/current`, `GET /terms/status`, `POST /terms/accept`.

### Recommended fix — do NOT update the SDK checkout

The upstream CLI imports **only stdlib** (`argparse, getpass, json, math, os,
subprocess, sys, urllib.*, pathlib, typing`) — one new import (`math`) versus our
copy. It has **no dependency on the alpasim SDK at all**. So it can be run
standalone and the shared `~/alpasim` checkout can stay exactly as it is.

That matters: `~/alpasim` is the checkout every eval run, venv and `sbatch` in
this project depends on. Pulling two months of upstream changes into it five days
before a deadline risks breaking the eval path for no benefit. **I deliberately
did not pull it.** Your call, but I'd keep it pinned.

Lowest-risk option — fetch just the one script:

```bash
mkdir -p ~/alpasim-cli && cd ~/alpasim-cli
curl -fsSL -o alpasim_challenge.py \
  https://raw.githubusercontent.com/NVlabs/alpasim/e2e_challenge/e2e_challenge/competitor_cli/alpasim_challenge.py
python3 alpasim_challenge.py --help        # should now list: terms
```

If you would rather stay inside the repo, `git -C ~/alpasim fetch origin
e2e_challenge && git -C ~/alpasim checkout origin/e2e_challenge --
e2e_challenge/competitor_cli/alpasim_challenge.py` updates only that file — but
it leaves the checkout dirty, so the standalone copy above is cleaner.

## 3. Staged commands — read-only ones first, then the accept

Run in this order, from wherever you put the updated CLI. **I have run none of
these.**

```bash
# --- G0.a: authenticate (yours; token not echoed) ---
python3 alpasim_challenge.py auth-url
python3 alpasim_challenge.py configure-token
python3 alpasim_challenge.py me

# --- 0.2: capture terms text + status (read-only, safe) ---
python3 alpasim_challenge.py terms show   > ~/alpasim-challenge/phase0/terms_text.md
python3 alpasim_challenge.py terms status > ~/alpasim-challenge/phase0/terms_status.txt

# --- 0.3: confirm the rerun target (read-only, safe) ---
python3 alpasim_challenge.py submissions --track nuplan > ~/alpasim-challenge/phase0/submissions_nuplan.json
python3 alpasim_challenge.py status 1bb06e8b-9d20-4018-98e1-5ba612e4e44c > ~/alpasim-challenge/phase0/rerun_target_status.json
python3 alpasim_challenge.py limits
```

### G0.b — the accept command, STAGED, HELD FOR YOU

```bash
python3 alpasim_challenge.py terms accept
```

Read §4 before running it. From the source, here is exactly what it does:

1. calls `show_terms()` first, so it **prints the full terms text** plus version
   and SHA-256 before asking anything;
2. if you have already accepted this exact version it says so and exits without
   changing anything;
3. otherwise it prompts **`Type ACCEPT <terms_version> to accept:`** and requires
   that literal string. Anything else → `SystemExit("Terms were not accepted; no
   server-side change was made.")`;
4. only then does it `POST /terms/accept` with the version and SHA-256.

Two useful consequences: it **cannot be triggered accidentally or
non-interactively**, and running `terms show` first is not strictly necessary
because `accept` shows the terms itself. You still want `terms show` captured to
a file for our records.

The response distinguishes `actor_accepted` from `captain_accepted`. You are the
registered captain (`adarshnanjaiya`), so **you** must be the signed-in user when
this runs — an acceptance from a non-captain submitter account will not satisfy
the requirement.

## 4. What to read carefully in the terms before accepting

I cannot see the terms text (blocked on auth), so I cannot review it for you.
Flagging the categories worth your attention, given you are UT-affiliated and
employed:

- **IP / licence grant** — what rights you grant NVIDIA over the submitted image,
  model weights, and the technical report. Check whether it is a broad licence or
  limited to evaluation and publication.
- **Eligibility** — employer/institutional restrictions, and whether prize
  eligibility requires an individual rather than an institution. You flagged
  wanting to check the institutional angle; this is the clause to check.
- **Publication / embargo** — whether accepting constrains publishing the method
  (relevant if any of this becomes a paper).
- **Data handling** — the charter notes non-acceptance by Sept 30 means our
  images, results and data are deleted. Confirm that against the actual text.
- **Version pinning** — acceptance is bound to a specific `terms_version` +
  `terms_sha256`. If the organizers revise the terms, a fresh acceptance is
  needed. Note the version you accept.

**I am not qualified to advise on the legal content and am not doing so.** If the
IP or institutional clauses are non-obvious, that is a question for UT's research
support / tech-transfer office, not for me.

## 5. Phase 0.4 — migration state

### Our failure, root-caused (issue #122, ours, closed 2026-08-17)

Organizer `mwatson-nvidia`, 2026-07-27, verbatim:

> Apologies for this--I was out last week for vacation and had uploaded a few
> things to the runner just before leaving for vacation that caused the system to
> run out of space.
>
> I'll work on fixing it with a migration to the "final" nuplan scenes (the
> previous ones were for the soft open), and rerun all the successful
> submissions/remove the records of the older submissions so that the usage count
> is removed. This should probably be ready later this week.

Another team (`CreaV`, submission `40ce656a`) hit the identical evaluator-side
failure and was also charged quota.

### Migration confirmed live? **NOT CONFIRMED.**

State it plainly: I could not verify from public sources that the migration to
final nuPlan scenes is live.

- The only organizer statement is the **intent** above (2026-07-27, "should
  probably be ready later this week").
- Issue #122 was **closed 2026-08-17 with no closing comment** — closure is the
  only signal, and it is weak evidence.
- A search of the repo's issues for a migration-complete announcement returned
  nothing.

Authoritative check is `terms status` / `limits` after you authenticate, or a
direct question to the organizers. **Recommendation: confirm migration is live
before spending any fresh submission slot**, since the rerun scores on the
migrated scenes.

Also note the organizer's promise to "remove the records of the older
submissions so that the usage count is removed" — worth verifying against
`limits` output that our quota was actually restored.

### Unrelated find that matters more than it looks — issue #151

Another team's image took a **total zero across all 16 replicas**. Organizer's
diagnosis:

> All 16 contestant replicas exited with code 1 (and were not OOM-killed) while
> loading the model. Their logs contain the same error:
>
>     Exception: Unknown GPU name: nvidia h100 80gb hbm3
>
> The traceback originates in the RAPTOR-N/Transfuser model configuration's
> GPU-name handling. The evaluation host was an H100 machine, and the submitted
> image did not recognize that GPU name, so no driver reached gRPC readiness and
> no simulation or metrics were produced.

Three things follow:

1. **The eval host is H100 (sm_90).** Our cluster is L40S/L40 (sm_89), and our
   own `README.md` says gsplat kernels are built for sm_89. Anything we ship that
   is compiled or gated for sm_89 **will die on the eval host exactly like this**.
2. **Our current shipped driver is immune.** `submission/route_follower.py` is
   pure numpy with no torch and no GPU use, so there is no GPU-name path to trip.
   This is an accidental benefit of the geometric baseline.
3. **This is a live threat to Phase 2 / D2.** A camera-only torch student must not
   gate on GPU name, and must be built for sm_90 (or PTX/JIT) rather than sm_89
   only. I have added this to the four-axis notes for D2 — it is a zero-rate
   issue, not a capability one, and it is the single cheapest catastrophic failure
   to avoid on the whole list.

## 6. Phase 0.3 — rerun target, partially identified

From our own `submissions.log`:

```
sub1  1bb06e8b-9d20-4018-98e1-5ba612e4e44c  nuplan  Wed Jul 15 15:21:21 CEST 2026  route-follower V20/A6/A7
sub1 RANK 1  cap=1750.56  at_fault_dist=3.36  Wed Jul 15 16:28:32 CEST 2026
```

Best candidate for "top soft-open submission": **`1bb06e8b-9d20-4018-98e1-5ba612e4e44c`**,
nuPlan track, RANK 1, cap 1750.56 — the route-follower at `RF_V_MAX=20
RF_A_LON=6.0 RF_A_LAT=7.0`, which matches the config baked into
`submission/Dockerfile`.

**Not yet confirmed, and I cannot confirm it without auth:**

- the **ECR image URI/tag** that submission points at. There is no record of it
  anywhere in this repo — I grepped for `ecr`, `696254625193`, `teams/`,
  `image_uri` and `docker tag` and found nothing outside the README. This is a
  gap in our own record-keeping worth closing.
- whether a later submission has since outranked it.

Run the two `submissions` / `status` commands in §3 and the target is confirmed;
I have left `phase0/rerun_target.md` as a stub with the fields to fill.

---

## Held for you (nothing below was done)

| Item | Why held |
|---|---|
| `terms accept` | Captain-only, legally binding (G0.b) |
| HF authentication | Binds your identity; I must not supply or store the credential (G0.a) |
| Updating `~/alpasim` | Shared eval dependency; pinning it is safer, and the standalone CLI avoids needing it |
| Any submission / ECR push | Explicit HOLD |

## Phase 0 completion status

| Item | State |
|---|---|
| 0.1 env + CLI verified, auth commands quoted | ✅ done |
| 0.2 terms text + status captured | ⛔ blocked — CLI lacks `terms` (§2) **and** needs auth (§1) |
| 0.3 rerun target confirmed | 🟡 partial — ID identified from local log, image URI unknown, needs auth |
| 0.4 migration state recorded | ✅ done — **not confirmed live**, evidence in §5 |
| accept command staged + terms surfaced | 🟡 command staged; terms text cannot be surfaced until §2 + auth are cleared |
