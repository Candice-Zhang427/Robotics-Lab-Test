"""Small diagnostic checks quoted in the presentation.

1. The provided A* expert deadlocks when executed with step 0.08.
2. Demo deviation from the goal direction is mostly an A* grid artifact
   (diagonal-first paths), not obstacle avoidance.
3. With few demos the bias term of linear BC is large and makes the robot
   circle the goal; D4 augmentation removes it.

Usage (from the repo root): python experiments/analysis_checks.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dataset, make_scenarios, evaluate, PointRobotEnv  # noqa: E402
from src.obstacles import expert_direction  # noqa: E402
from src.policy import LinearPolicy, SymmetricLinearPolicy  # noqa: E402

# 1) executed expert on the public set
public = make_scenarios(300, seed=1)
env = PointRobotEnv(seed=123)
fails, stuck = 0, 0
for sc in public:
    env.reset(start=sc["start"], target=sc["target"], blocks=sc["blocks"])
    P = [env.pos.copy()]
    for _ in range(80):
        _, _, d, info = env.step(expert_direction(env.pos, env.target, env.blocks))
        P.append(env.pos.copy())
        if d:
            break
    if not info["success"]:
        fails += 1
        stuck += np.allclose(P[-1], P[-10])
print(f"[1] privileged A* expert executed on public set: success {1 - fails / 300:.3f}, "
      f"{stuck}/{fails} failures end stuck in place")

# 2) grid artifact
rng = np.random.default_rng(0)
A, G = [], []
for _ in range(2000):
    s, t = rng.uniform(0.05, 0.95, 2), rng.uniform(0.05, 0.95, 2)
    A.append(expert_direction(s, t, np.zeros((0, 4)))); G.append((t - s) / np.linalg.norm(t - s))
A, G = np.array(A), np.array(G)
dev = np.degrees(np.arccos(np.clip(np.sum(A * G, 1), -1, 1)))
ea = np.degrees(np.arctan2(A[:, 1], A[:, 0])) % 90
ga = np.degrees(np.arctan2(G[:, 1], G[:, 0])) % 90
print(f"[2] expert with NO obstacles/noise: mean deviation {dev.mean():.1f} deg, {np.mean(dev > 30) * 100:.1f}% > 30 deg; "
      f"first step within 8 deg of a diagonal: {np.mean(np.abs(ea - 45) < 8) * 100:.0f}% "
      f"(goal directions: {np.mean(np.abs(ga - 45) < 8) * 100:.0f}%)")

# 3) bias vs demo budget
X, Y = dataset()
for n in [10, 30, 100, 300, 4000]:
    W = LinearPolicy(X[:n], Y[:n]).W
    Ws = SymmetricLinearPolicy(X[:n], Y[:n]).W
    print(f"[3] n={n:5d}: linear BC bias = {np.round(W[2], 3)}, D4-symmetric W = {np.round(Ws, 3).tolist()}")
