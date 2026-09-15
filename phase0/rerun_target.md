# Rerun target — CONFIRMED 2026-09-15

## Best candidate (from our own records, not yet API-confirmed)

| Field | Value | Source |
|---|---|---|
| Submission ID | `1bb06e8b-9d20-4018-98e1-5ba612e4e44c` | `submissions.log` |
| Track | nuplan | `submissions.log` |
| Submitted | 2026-07-15 15:21 CEST | `submissions.log` |
| Result | **RANK 1**, cap 1750.56, at_fault_dist 3.36 | `submissions.log` |
| Driver | route-follower, `RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0` | `submissions.log`, matches `submission/Dockerfile` |
| **ECR image URI** | `696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/na-la:sub1` | API `status` — **gap closed** |
| **Image digest (immutable pin)** | `sha256:75e9090a45af58095d87630f8b1f0707d28b638d2e7a0138508a0a95827eee0c` | API `status` |
| Image size | 99,022,867 B = 94.4 MiB (0.23% of the 40 GiB cap) | API `status` |
| Evaluator score | `0.9804824187258063`, status `SUCCEEDED` | API `status` |
| Throughput | 2157.161 s observed vs 2485.438 s limit, margin 328.277 s, passed | API `status` |
| Still top-ranked? | only submission on record for the team; `submissions --track nuplan` captured | — |

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
