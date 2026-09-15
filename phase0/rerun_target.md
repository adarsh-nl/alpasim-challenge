# Rerun target — PARTIALLY CONFIRMED

## Best candidate (from our own records, not yet API-confirmed)

| Field | Value | Source |
|---|---|---|
| Submission ID | `1bb06e8b-9d20-4018-98e1-5ba612e4e44c` | `submissions.log` |
| Track | nuplan | `submissions.log` |
| Submitted | 2026-07-15 15:21 CEST | `submissions.log` |
| Result | **RANK 1**, cap 1750.56, at_fault_dist 3.36 | `submissions.log` |
| Driver | route-follower, `RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0` | `submissions.log`, matches `submission/Dockerfile` |
| **ECR image URI / tag** | **UNKNOWN — not recorded anywhere in this repo** | grep for `ecr`/`696254625193`/`teams/`/`image_uri`/`docker tag` found nothing |
| Still top-ranked? | **UNCONFIRMED** — no later submission in the log, but not API-verified | — |

## To confirm (needs auth)

    python3 alpasim_challenge.py submissions --track nuplan
    python3 alpasim_challenge.py status 1bb06e8b-9d20-4018-98e1-5ba612e4e44c

## Record-keeping gap worth closing

We never recorded the ECR image URI/tag for our only submission. The rerun
"reuses the existing top image", so the organizers resolve it server-side and
this does not block the rerun -- but we cannot currently answer "which image is
our rank-1 result?" from local records alone. Recommend logging the full image
URI in `submissions.log` for every future submission.

## Backfilling the URI once authenticated (Phase 1.5)

`status <submission_id>` should return the image URI the organizers hold, so the
gap is likely recoverable rather than lost:

    python3 alpasim_challenge.py status 1bb06e8b-9d20-4018-98e1-5ba612e4e44c \
      > phase0/rerun_target_status.json

Then backfill the local record:

    python3 tools/log_submission.py --update 1bb06e8b \
      --image-uri <uri-from-that-json>

The record is currently marked `UNRECOVERABLE` in `submissions.tsv` only because
we cannot read it without auth -- not because it is known to be gone.

`tools/log_submission.py --audit` is the pre-submission check: it refuses to
record any NEW submission without an image URI, so this gap cannot recur.
