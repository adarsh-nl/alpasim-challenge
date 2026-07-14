#!/usr/bin/env python3
"""Split navtest_full into N shard configs, keeping only scenes whose assets exist."""
import os, sys, yaml, random

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 = all available
SEED = 1234

REPO = os.path.expanduser("~/alpasim")
ROOT = os.environ["ALPASIM_NUPLAN_ROOT"]
SRC = f"{REPO}/src/wizard/configs/nuplan_scenes/navtest_full.yaml"
OUT = f"{REPO}/src/wizard/configs/nuplan_scenes"

ids = yaml.safe_load(open(SRC))["scenes"]["scene_ids"]
have = set(os.listdir(f"{ROOT}/navtest/assets"))
avail = [s for s in ids if s in have]
print(f"navtest_full: {len(ids)} scenes | assets on disk: {len(avail)}")

if LIMIT and LIMIT < len(avail):
    random.Random(SEED).shuffle(avail)
    avail = sorted(avail[:LIMIT])
    print(f"sampled {LIMIT} (seed {SEED})")

for i in range(N):
    chunk = avail[i::N]                       # stride: balances long/short scenes
    if not chunk: continue
    p = f"{OUT}/shard{i:02d}.yaml"
    with open(p, "w") as f:
        f.write("# @package _global_\n# AUTO-GENERATED\nscenes:\n  scene_ids:\n")
        for s in chunk:
            f.write(f"    - {s}\n")
print(f"wrote {N} shards, ~{len(avail)//N} scenes each -> {OUT}/shardNN.yaml")
