## Public set (make_scenarios(300, seed=1), 80 steps) -- official notebook setting

| policy | success | final dist | steps | collision-episode rate | blocked steps / ep |
|---|---|---|---|---|---|
| linear BC (baseline) | 0.830 | 0.090 | 17.9 | 0.170 | 13.14 |
| linear BC, no bias | 0.827 | 0.091 | 18.1 | 0.173 | 13.39 |
| linear BC + D4 aug, no bias | 0.837 | 0.087 | 17.4 | 0.163 | 12.62 |
| MLP BC | 0.790 | 0.113 | 21.2 | 0.210 | 16.24 |
| MLP BC + D4 aug | 0.790 | 0.113 | 21.2 | 0.210 | 16.23 |
| kNN BC | 0.793 | 0.106 | 20.9 | 0.207 | 15.96 |
| 3a random escape | 1.000 | 0.022 | 7.8 | 0.170 | 0.46 |
| 3b bug / wall-follow | 0.997 | 0.022 | 6.7 | 0.170 | 0.32 |
| 3c map + plan (ours, D4 prior) | 1.000 | 0.021 | 6.1 | 0.163 | 0.34 |
| oracle (true map) | 1.000 | 0.023 | 5.7 | 0.000 | - |

## Public-like, mean ± std over seeds 1,2,3

| policy | success | steps | collision-episode rate | blocked steps / ep |
|---|---|---|---|---|
| linear BC (baseline) | 0.808 ± 0.025 | 19.6 ± 1.9 | 0.192 ± 0.025 | 14.88 ± 1.91 |
| linear BC, no bias | 0.806 ± 0.023 | 19.7 ± 1.8 | 0.194 ± 0.023 | 15.04 ± 1.80 |
| linear BC + D4 aug, no bias | 0.808 ± 0.028 | 19.6 ± 2.1 | 0.192 ± 0.028 | 14.88 ± 2.17 |
| MLP BC | 0.769 ± 0.025 | 22.8 ± 1.9 | 0.231 ± 0.025 | 17.86 ± 1.98 |
| MLP BC + D4 aug | 0.773 ± 0.021 | 22.5 ± 1.6 | 0.227 ± 0.021 | 17.52 ± 1.67 |
| kNN BC | 0.777 ± 0.021 | 22.1 ± 1.6 | 0.223 ± 0.021 | 17.26 ± 1.68 |
| 3a random escape | 1.000 ± 0.000 | 8.1 ± 0.2 | 0.192 ± 0.025 | 0.50 ± 0.03 |
| 3b bug / wall-follow | 0.994 ± 0.003 | 7.0 ± 0.2 | 0.193 ± 0.024 | 0.35 ± 0.04 |
| 3c map + plan (ours, D4 prior) | 1.000 ± 0.000 | 6.3 ± 0.2 | 0.192 ± 0.028 | 0.40 ± 0.04 |
| oracle (true map) | 1.000 ± 0.000 | 5.8 ± 0.1 | 0.000 ± 0.000 | - |

## Success rate vs number of blocks (mean over seeds 11-13, 300 episodes each)

| #blocks | baseline | 3a | 3b | 3c | oracle |
|---|---|---|---|---|---|
| 2-4 | 0.793 | 0.998 | 0.997 | 1.000 | 1.000 |
| 4-6 | 0.706 | 0.990 | 0.993 | 0.999 | 1.000 |
| 6-8 | 0.618 | 0.974 | 0.989 | 0.997 | 1.000 |
| 8-10 | 0.532 | 0.943 | 0.983 | 0.999 | 1.000 |

## Success rate vs observation noise std (mean over seeds 11-13, 300 episodes each)

| obs noise | baseline | 3a | 3b | 3c | oracle |
|---|---|---|---|---|---|
| 0.0 | 0.793 | 0.998 | 0.997 | 1.000 | 1.000 |
| 0.005 | 0.801 | 1.000 | 0.998 | 1.000 | 1.000 |
| 0.01 | 0.809 | 1.000 | 0.999 | 1.000 | 1.000 |
| 0.02 | 0.827 | 0.998 | 0.984 | 1.000 | 1.000 |
| 0.03 | 0.839 | 0.998 | 1.000 | 1.000 | 1.000 |

## Success rate vs action noise std (mean over seeds 11-13, 300 episodes each)

| act noise | baseline | 3a | 3b | 3c | oracle |
|---|---|---|---|---|---|
| 0.0 | 0.793 | 0.998 | 0.997 | 1.000 | 1.000 |
| 0.1 | 0.857 | 1.000 | 0.999 | 1.000 | 1.000 |
| 0.2 | 0.901 | 0.999 | 0.999 | 1.000 | 1.000 |
| 0.3 | 0.958 | 1.000 | 1.000 | 0.999 | 1.000 |

## Success rate vs max steps (4-6 blocks) (mean over seeds 11-13, 300 episodes each)

| max steps | baseline | 3a | 3b | 3c | oracle |
|---|---|---|---|---|---|
| 10 | 0.696 | 0.707 | 0.789 | 0.827 | 0.940 |
| 15 | 0.706 | 0.822 | 0.919 | 0.969 | 1.000 |
| 20 | 0.706 | 0.873 | 0.959 | 0.992 | 1.000 |
| 30 | 0.706 | 0.928 | 0.990 | 0.998 | 1.000 |
| 40 | 0.706 | 0.967 | 0.993 | 0.999 | 1.000 |
| 80 | 0.706 | 0.990 | 0.993 | 0.999 | 1.000 |

## Stress test: 6-8 blocks, obs noise .01, action noise .1, 40 steps (seeds 21-23)

| policy | success | steps | collision-episode rate | blocked steps / ep |
|---|---|---|---|---|
| linear BC (baseline) | 0.692 ± 0.022 | 16.4 | 0.391 | 12.17 |
| 3a random escape | 0.911 ± 0.010 | 12.2 | 0.390 | 1.75 |
| 3b bug / wall-follow | 0.983 ± 0.003 | 9.0 | 0.394 | 1.19 |
| 3c map + plan (ours, D4 prior) | 0.998 ± 0.002 | 7.9 | 0.389 | 1.25 |
| oracle (true map) | 1.000 ± 0.000 | 6.1 | 0.081 | - |

## Demonstration budget (public seed 1)

| n demos | linear BC | linear BC + D4 | MLP BC | 3c, plain linear prior | 3c (final, D4 prior) |
|---|---|---|---|---|---|
| 10 | 0.600 | 0.837 | 0.760 | 0.743 | 1.000 |
| 30 | 0.123 | 0.837 | 0.213 | 0.170 | 1.000 |
| 100 | 0.350 | 0.837 | 0.777 | 0.447 | 1.000 |
| 300 | 0.827 | 0.837 | 0.777 | 1.000 | 1.000 |
| 1000 | 0.817 | 0.837 | 0.783 | 1.000 | 1.000 |
| 4000 | 0.830 | 0.837 | 0.790 | 1.000 | 1.000 |

## Demonstration statistics

- 28.9% of the 4000 demo actions deviate > 30° from the goal direction.
- Mean |deviation| of BC output from goal direction: linear BC (baseline) 0.8°, linear BC, no bias 0.8°, linear BC + D4 aug, no bias 0.0°, MLP BC 18.1°, MLP BC + D4 aug 17.9°, kNN BC 17.4°
- Public-set scenarios where the baseline fails: 51; 6-8-block set (seed 12): 109
