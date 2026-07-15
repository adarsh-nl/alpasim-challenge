#!/usr/bin/env python3
"""Per-scene scores from an array run. Usage: score.py <array_dir>"""
import json, sys, os

ad = sys.argv[1]
d = json.load(open(f"{ad}/aggregate/results-summary.json"))
R = d["rollouts"]

def g(r, k, default=0.0):
    v = r["metrics"].get(k, default)
    return default if v is None else v

rows = []
for r in R:
    m = r["metrics"]
    rows.append(dict(
        scene=r["clipgt_id"][-40:],
        score=r["score"],
        passed=r["passed"],
        fail=r.get("failure_reason") or "",
        flt=int(g(r,"collision_at_fault")>0),
        off=int(g(r,"offroad")>0),
        rear=int(g(r,"collision_rear")>0),
        prog=g(r,"progress_clipped_rel"),
        wrong=int(g(r,"wrong_lane")>0),
        obs=g(r,"min_distance_to_obstacle_m",9),
    ))

n = len(rows)
mean = sum(x["score"] for x in rows)/n
zeros = sum(1 for x in rows if x["score"] < 0.01)
flt = sum(x["flt"] for x in rows)
off = sum(x["off"] for x in rows)
wrong = sum(x["wrong"] for x in rows)
print(f"n={n}  mean_score={mean:.3f}  zeros={zeros}  "
      f"at_fault={flt}  offroad={off}  wrong_lane={wrong}")

print("\nworst 20 by score:")
print(f"{'score':>5} {'flt':>3} {'off':>3} {'prog':>5} {'wrong':>5} {'obs':>6}  scene  (failure_reason)")
for x in sorted(rows, key=lambda z:(z["score"], z["prog"]))[:20]:
    print(f"{x['score']:5.2f} {x['flt']:>3} {x['off']:>3} {x['prog']:5.2f} "
          f"{x['wrong']:>5} {x['obs']:6.2f}  {x['scene']}  {x['fail']}")

# failure-reason histogram
from collections import Counter
c = Counter(x["fail"] for x in rows if x["fail"])
if c:
    print("\nfailure reasons:")
    for k,v in c.most_common(): print(f"  {v:>3}  {k}")
