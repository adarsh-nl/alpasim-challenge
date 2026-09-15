# Competition terms — NOT CAPTURED

**Status: BLOCKED. This file contains no terms text. Nothing here is the terms.**

Two blockers, both in `HANDOFF.md`:

1. The local `~/alpasim` CLI (checkout pinned 2026-07-07, `42116f8`) has **no
   `terms` subcommand`. Upstream does. See HANDOFF.md section 2 for the
   lowest-risk fix (fetch the standalone script; do not update the SDK checkout).
2. `terms show` requires authentication, which is Adarsh's action (gate G0.a).
   The saved token at `~/.alpasim/challenge.json` dates from 2026-07-18 and CLI
   tokens expire after 12 hours, so it is dead.

Populate with:

    python3 alpasim_challenge.py terms show > phase0/terms_text.md

Deliberately NOT done: no terms text was guessed, summarised from memory, or
reconstructed. The terms are a legal document; a fabricated or approximate copy
would be worse than an empty file.
