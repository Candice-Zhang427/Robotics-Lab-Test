"""Shared helpers for the experiment scripts."""
import os
import sys
import time

import numpy as np

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("RESULTS_DIR") or os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)

from src.env import make_dataset, PointRobotEnv  # noqa: E402
from src.evaluate import evaluate, make_scenarios  # noqa: E402
from src.obstacles import astar_path, point_in_rects  # noqa: E402


def dataset(n=4000, seed=42):
    """make_dataset() is slow (A* per sample) -> cache it. Smaller n = prefix."""
    f = os.path.join(OUT, f"demo_n4000_s{seed}.npz")
    if os.path.exists(f):
        d = np.load(f)
        X, Y = d["X"], d["Y"]
    else:
        X, Y = make_dataset(n=4000, seed=seed)
        np.savez(f, X=X, Y=Y)
    return X[:n], Y[:n]


def collision_steps(traj):
    """Number of blocked steps in an episode (position did not change)."""
    P = traj["positions"]
    return int(np.sum(np.all(np.diff(P, axis=0) == 0, axis=1)))


def run(policy, scenarios, keep_traj=False, **kw):
    t = time.time()
    r = evaluate(policy, scenarios, **kw)
    r["collision_steps"] = float(np.mean([collision_steps(t_) for t_ in r["trajectories"]]))
    r["time"] = time.time() - t
    if not keep_traj:
        r.pop("trajectories")
    return r


def fmt(name, r):
    return (f"{name:26s} succ={r['success_rate']:.3f} dist={r['mean_final_distance']:.3f} "
            f"steps={r['mean_steps']:5.1f} coll_ep={r['collision_episode_rate']:.3f} "
            f"coll_steps={r['collision_steps']:.2f}  ({r['time']:.1f}s)")
