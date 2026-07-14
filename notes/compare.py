#!/usr/bin/env python3
"""Compare runs. Usage: compare.py <run_dir> [run_dir ...]"""
import json, sys, glob, os

def load(d):
    p = os.path.join(d, "aggregate", "results-summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))

rows = []
for d in sorted(sys.argv[1:]):
    js = load(d)
    if not js:
        print(f"  (no results: {os.path.basename(d)})"); continue
    rs = [r for r in js["rollouts"] if r["status"] == "pass"]
    fails = [r for r in js["rollouts"] if r["status"] != "pass"]
    if not rs: continue
    n = len(rs)
    mean = sum(r["score"] for r in rs) / n
    zeros = sum(1 for r in rs if r["score"] < 0.01)
    fault = sum(1 for r in rs if r["metrics"].get("collision_at_fault", 0) > 0)
    offr  = sum(1 for r in rs if r["metrics"].get("offroad", 0) > 0)
    prog  = sum(r["metrics"]["progress_clipped_rel"] for r in rs) / n
    obst  = min(r["metrics"]["min_distance_to_obstacle_m"] for r in rs)
    rpc   = js["telemetry"]["driver_drive_rpc_duration_mean_s"]
    rows.append((os.path.basename(d), n, mean, zeros, fault, offr, prog, obst, rpc, len(fails)))

print(f"{'run':<16} {'n':>2} {'mean':>6} {'zero':>4} {'flt':>3} {'off':>3} "
      f"{'prog':>5} {'minObs':>6} {'rpc_ms':>6} {'err':>3}")
print("-" * 72)
for r in sorted(rows, key=lambda x: -x[2]):
    print(f"{r[0]:<16} {r[1]:>2} {r[2]:>6.3f} {r[3]:>4} {r[4]:>3} {r[5]:>3} "
          f"{r[6]:>5.2f} {r[7]:>6.2f} {r[8]*1000:>6.1f} {r[9]:>3}")

# worst scenes of the best run
if rows:
    best = max(rows, key=lambda x: x[2])[0]
    js = load([d for d in sys.argv[1:] if os.path.basename(d) == best][0])
    rs = sorted([r for r in js["rollouts"] if r["status"] == "pass"],
                key=lambda x: x["score"])[:5]
    print(f"\nworst scenes in {best}:")
    for r in rs:
        m = r["metrics"]
        print(f"  {r['score']:.3f}  flt={m['collision_at_fault']:.0f} "
              f"off={m['offroad']:.0f} prog={m['progress_clipped_rel']:.2f} "
              f"obs={m['min_distance_to_obstacle_m']:5.2f}m  {r['clipgt_id'][-16:]}")
