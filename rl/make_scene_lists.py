#!/usr/bin/env python
"""Generate train/eval scene-id lists from the array-538078 rollouts on disk.

Picks a focused SMALL training set: scenes with a spread of dist_to_gt (so the
policy sees both tight and loose tracking), which is where speed modulation has
signal. Writes full scene ids (dir names) one per line.
"""
import glob, os, sys

RUN = "/home/nanjaiyalathaa/alpasim/runs/array-538078"
dirs = glob.glob(f"{RUN}/task-*/rollouts/*")
ids = sorted({os.path.basename(d) for d in dirs})
print(f"found {len(ids)} full scene ids", file=sys.stderr)

# small focused set: first 24 for train, next 8 for eval (deterministic split)
train = ids[:24]
ev = ids[24:32]
with open("train_scenes.txt","w") as f: f.write("\n".join(train)+"\n")
with open("eval_scenes.txt","w") as f: f.write("\n".join(ev)+"\n")
print(f"wrote {len(train)} train, {len(ev)} eval", file=sys.stderr)
