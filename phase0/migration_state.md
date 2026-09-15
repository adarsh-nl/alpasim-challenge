# Infra migration state — NOT CONFIRMED LIVE

## Verdict

The migration to the "final" nuPlan scenes is **NOT CONFIRMED live** from any
source available to me. Only organizer *intent* is on record.

## Evidence

Issue #122 (ours, `adarsh-nl`, opened 2026-07-18, closed 2026-08-17) --
`DIRECT_P5_COMMAND_FAILED` / `p5_ssm_not_online`, S3 sync of
`s3:alpasim-scene-data-use1/nuplan/worldengine/` failed before container start.
Our failure `9f49b1ac`. Another team (`CreaV`, submission `40ce656a`) hit the
same thing and was also charged quota.

Organizer `mwatson-nvidia`, 2026-07-27, verbatim:

> Apologies for this--I was out last week for vacation and had uploaded a few
> things to the runner just before leaving for vacation that caused the system
> to run out of space.
>
> I'll work on fixing it with a migration to the "final" nuplan scenes (the
> previous ones were for the soft open), and rerun all the successful
> submissions/remove the records of the older submissions so that the usage
> count is removed. This should probably be ready later this week.

Root cause: organizer-side runner ran out of disk space. Not our image.

## Why "not confirmed"

- The statement above is intent with a soft ETA ("later this week", from Jul 27).
- Issue #122 was closed 2026-08-17 **with no closing comment**. Closure alone is
  weak evidence that the migration shipped.
- A search of NVlabs/alpasim issues for a migration-complete announcement
  returned nothing.

## Recommended checks (need auth / organizer contact)

1. `python3 alpasim_challenge.py limits` -- did the promised quota restoration
   ("remove the records of the older submissions so that the usage count is
   removed") actually happen?
2. `python3 alpasim_challenge.py terms status` / `submissions --track nuplan` --
   any indication the rerun has occurred or is queued.
3. Ask the organizers directly whether the final-scene migration is live, since
   the free rerun scores against the migrated scenes.

**Do not spend a fresh submission slot until migration is confirmed live.**
