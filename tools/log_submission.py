#!/usr/bin/env python3
"""Record a submission and the exact image it maps to (Phase 1.5).

WHY THIS EXISTS: our rank-1 submission `1bb06e8b-9d20-4018-98e1-5ba612e4e44c`
has no ECR image URI recorded anywhere in the repo. The organizers hold the
image so the free rerun does not need it from us, but we cannot currently answer
"which image produced our rank-1 result?" from local records — which blocks
reproducing or iterating from it. Every future submission must record the URI.

Appends a one-line TSV record to submissions.tsv. Pure stdlib, no network, no
credentials read or stored — the image URI is passed in by the caller (or by
`--from-cli`, which shells out to the already-authenticated challenge CLI).

Usage
-----
    # after a push+submit, record it:
    python3 tools/log_submission.py \
        --submission-id 1bb06e8b-9d20-4018-98e1-5ba612e4e44c \
        --track nuplan \
        --image-uri 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team>:v3 \
        --driver submission/route_follower.py \
        --config "RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0" \
        --note "route-follower baseline"

    # fill in the result once the eval returns:
    python3 tools/log_submission.py --update 1bb06e8b... --result "RANK 1 cap=1750.56"

    # verify every recorded submission has an image URI (pre-submission gate):
    python3 tools/log_submission.py --audit
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "submissions.tsv"

COLUMNS = [
    "submission_id", "track", "image_uri", "git_commit", "git_branch",
    "driver", "config", "submitted_utc", "result", "note",
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _ensure_header() -> None:
    if not LOG.exists() or not LOG.read_text().strip():
        LOG.write_text("\t".join(COLUMNS) + "\n")


def _rows() -> list[dict[str, str]]:
    if not LOG.exists():
        return []
    lines = [l for l in LOG.read_text().splitlines() if l.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, l.split("\t"))) for l in lines[1:]]


def _write(rows: list[dict[str, str]]) -> None:
    out = ["\t".join(COLUMNS)]
    for r in rows:
        out.append("\t".join((r.get(c, "") or "").replace("\t", " ") for c in COLUMNS))
    LOG.write_text("\n".join(out) + "\n")


def add(args) -> int:
    _ensure_header()
    rows = _rows()
    if any(r["submission_id"] == args.submission_id for r in rows):
        print(f"already recorded: {args.submission_id}", file=sys.stderr)
        return 1
    if not args.image_uri and not args.allow_missing_uri:
        print("refusing to record a submission with no --image-uri.\n"
              "That gap is exactly what this tool exists to close. Pass\n"
              "--allow-missing-uri only to backfill a historical entry whose\n"
              "URI is genuinely unrecoverable.", file=sys.stderr)
        return 2
    rows.append({
        "submission_id": args.submission_id,
        "track": args.track,
        "image_uri": args.image_uri or "UNRECOVERABLE",
        "git_commit": args.git_commit or _git("rev-parse", "--short", "HEAD"),
        "git_branch": args.git_branch or _git("rev-parse", "--abbrev-ref", "HEAD"),
        "driver": args.driver,
        "config": args.config,
        "submitted_utc": args.submitted_utc or datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result": args.result,
        "note": args.note,
    })
    _write(rows)
    print(f"recorded {args.submission_id} -> {rows[-1]['image_uri']}")
    return 0


def update(args) -> int:
    rows = _rows()
    for r in rows:
        if r["submission_id"].startswith(args.update):
            if args.result:
                r["result"] = args.result
            if args.image_uri:
                r["image_uri"] = args.image_uri
            if args.note:
                r["note"] = args.note
            _write(rows)
            print(f"updated {r['submission_id']}")
            return 0
    print(f"no record matching {args.update}", file=sys.stderr)
    return 1


def audit(_args) -> int:
    """Pre-submission gate: every record must name a real image URI."""
    rows = _rows()
    if not rows:
        print("submissions.tsv is empty")
        return 0
    bad = [r for r in rows if r["image_uri"] in ("", "UNKNOWN", "UNRECOVERABLE")]
    print(f"{len(rows)} submission(s) recorded; {len(bad)} missing an image URI")
    for r in rows:
        flag = "  !!" if r in bad else "    "
        print(f"{flag} {r['submission_id'][:12]}  {r['track']:7s} "
              f"{r['image_uri'][:64]:64s} {r['result'][:28]}")
    if bad:
        print("\nsubmissions with no recoverable image URI cannot be reproduced "
              "or iterated from locally.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-id")
    ap.add_argument("--track", choices=("nuplan", "pai"), default="nuplan")
    ap.add_argument("--image-uri", default="")
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--git-branch", default="")
    ap.add_argument("--driver", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--submitted-utc", default="")
    ap.add_argument("--result", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--allow-missing-uri", action="store_true",
                    help="backfill only: record an entry whose URI is unrecoverable")
    ap.add_argument("--update", metavar="SUBMISSION_ID",
                    help="update result/image_uri/note on an existing record")
    ap.add_argument("--audit", action="store_true",
                    help="list records and flag any without an image URI")
    args = ap.parse_args()

    if args.audit:
        return audit(args)
    if args.update:
        return update(args)
    if not args.submission_id:
        ap.error("--submission-id is required (or use --audit / --update)")
    return add(args)


if __name__ == "__main__":
    sys.exit(main())
