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
