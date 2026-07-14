# Results log — navtest_dev

| driver | scenes | mean score | at-fault | offroad | notes |
|---|---|---|---|---|---|
| starter (straight, 5 m/s) | 1 | 0.427 | 0 | 0 | rear-ended (free) |
| route-follower V=12 | 1 | 0.966 | 0 | 0 | |
| route-follower V=12 | 8 | **0.900** | **0** | **0** | 1 scene failed: missing asset shard |

## route-follower V=12, per scene (8 scenes)
1.000 / 1.000 / 0.984 / 0.966 / 0.901 / 0.888 / 0.863 / 0.610

All loss is PROGRESS, not safety. Two scenes hit the 0.8 progress cap.
Worst scene (0.610) traveled 25.7m of 52.5m GT.
min_distance_to_obstacle got as low as 0.42m -- blind, got lucky.

## Acceleration sweep (navtest_dev, 9 scenes, RF_V_MAX=20)

| A_LON | A_LAT | mean | prog | at-fault | offroad | minObs |
|---|---|---|---|---|---|---|
| 1.5 | 2.5 | 0.903 | 0.74 | 0 | 0 | 0.42 |
| 2.5 | 3.5 | 0.970 | 0.84 | 0 | 0 | 0.86 |
| 4.0 | 5.0 | 0.993 | 0.92 | 0 | 0 | 0.82 |
| 6.0 | 7.0 | **1.000** | 0.95 | 0 | 0 | 0.79 |

V_MAX sweep (10/12/14/16/18) was a NULL RESULT -- identical above 14.
A_LON was the binding constraint, not V_MAX.

CONCLUSION: navtest_dev is SATURATED. A blind route-follower scores 1.000.
The dev set has no traffic conflicts -- it is not a test of driving.
All further signal must come from a larger scene set.
