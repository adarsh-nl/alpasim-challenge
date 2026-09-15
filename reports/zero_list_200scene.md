# Per-route zero list

```
run: /home/nanjaiyalathaa/alpasim/runs/array-538078
routes: 200   mean score: 0.9900
ZEROS (<0.01): 2   near-zeros (<0.8): 0   perfect (1.000): 198  (99.0%)

root-cause buckets:
   198  clean
     2  collision

collisions: at-fault front=2  at-fault lateral=0  rear (NOT at fault, free)=0

 score bucket                prog   gt_m  obs_m  scene  <- evidence
----------------------------------------------------------------------------------------------------
 0.000 collision             0.69   10.8   0.00  19.27.01_veh-45_00472_00711-8cc8fe63227d5c07  <- at-fault (front), minObs=0.00m
 0.000 collision             1.00    3.6   0.00  15.47.30_veh-45_01199_01391-dd162898ecf351d4  <- at-fault (front), minObs=0.00m

!! SATURATION: 198/200 routes scored exactly 1.000. This set provides no gradient outside the 2 tail route(s); any tuning measured on it is measuring noise. A larger sweep is required.

evaluator score_criteria (from the run, not assumed):
  collision_at_fault: == 0
  offroad: == 0
  progress_score: min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)
  short_gt_distance_override: progress_score = 1.0 when gt_dist_traveled_m < 5.0
```
