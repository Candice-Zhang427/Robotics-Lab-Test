"""Run all experiments and write results/results.json.

Usage (from the repo root):  python experiments/run_experiments.py
Hyper-parameters of the memory policies were chosen on validation seed 100;
all numbers below use other seeds (public = seed 1, as in the notebook).
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, dataset, make_scenarios, run, astar_path  # noqa: E402
from oracle import eval_oracle  # noqa: E402

POLICIES = ["linear_bc", "linear_nobias", "linear_d4", "mlp_bc", "mlp_d4", "knn_bc", "escape_3a", "bug_3b", "student_3c"]
ROBUST_POLICIES = ["linear_bc", "escape_3a", "bug_3b", "student_3c", "oracle"]
SEEDS = [11, 12, 13]


def make_policy(name, n=4000):
    from src.policy import LinearPolicy, SymmetricLinearPolicy, StudentPolicy
    from src.variants import (LinearNoBiasPolicy, MLPPolicy, MLPD4Policy, KNNPolicy,
                              RandomEscapePolicy, BugPolicy)
    X, Y = dataset(n)
    return {
        "linear_bc": lambda: LinearPolicy(X, Y),
        "linear_nobias": lambda: LinearNoBiasPolicy(X, Y),
        "linear_d4": lambda: SymmetricLinearPolicy(X, Y),
        "mlp_d4": lambda: MLPD4Policy(X, Y),
        "student_3c_linprior": lambda: StudentPolicy(X, Y, prior=LinearPolicy(X, Y)),
        "mlp_bc": lambda: MLPPolicy(X, Y),
        "knn_bc": lambda: KNNPolicy(X, Y, k=min(15, len(X) - 1)),
        "escape_3a": lambda: RandomEscapePolicy(X, Y),
        "bug_3b": lambda: BugPolicy(X, Y),
        "student_3c": lambda: StudentPolicy(X, Y),
    }[name]()


def job(spec):
    """spec = (group, level, policy, seed, scen_kwargs, eval_kwargs, n_demo)"""
    group, level, pol, seed, skw, ekw, n = spec
    scen = make_scenarios(300, seed=seed, **skw)
    if pol == "oracle":
        r = eval_oracle(scen, **ekw)
        r.pop("success_flags", None)
        r["collision_steps"] = float("nan")
    else:
        r = run(make_policy(pol, n), scen, **ekw)
    r = {k: float(v) for k, v in r.items()}
    if pol == "oracle":  # fraction of scenarios the grid A* can solve at all
        r["reachable"] = float(np.mean([astar_path(s["start"], s["target"], s["blocks"]) is not None
                                        for s in scen]))
    return dict(group=group, level=level, policy=pol, seed=seed, n_demo=n, **r)


def build_jobs():
    J = []
    # 1) public set exactly as in the notebook (seed 1) + two more public-like seeds
    for pol in POLICIES + ["oracle"]:
        for s in [1, 2, 3]:
            J.append(("public", "default", pol, s, {}, {}, 4000))
    # 2) robustness sweeps
    for nb in [(2, 4), (4, 6), (6, 8), (8, 10)]:
        for pol in ROBUST_POLICIES:
            for s in SEEDS:
                J.append(("blocks", f"{nb[0]}-{nb[1]}", pol, s, {"n_blocks": nb}, {}, 4000))
    for ns in [0.0, 0.005, 0.01, 0.02, 0.03]:
        for pol in ROBUST_POLICIES:
            for s in SEEDS:
                J.append(("obs_noise", ns, pol, s, {}, {"noise_std": ns}, 4000))
    for an in [0.0, 0.1, 0.2, 0.3]:
        for pol in ROBUST_POLICIES:
            for s in SEEDS:
                J.append(("act_noise", an, pol, s, {}, {"action_noise_std": an}, 4000))
    for H in [10, 15, 20, 30, 40, 80]:
        for pol in ROBUST_POLICIES:
            for s in SEEDS:
                J.append(("horizon", H, pol, s, {"n_blocks": (4, 6)}, {"max_steps": H}, 4000))
    # 3) a combined "held-out-like" stress test
    for pol in ROBUST_POLICIES:
        for s in [21, 22, 23]:
            J.append(("stress", "6-8 blocks, obs .01, act .1, 40 steps", pol, s, {"n_blocks": (6, 8)},
                      {"noise_std": 0.01, "action_noise_std": 0.1, "max_steps": 40}, 4000))
    # 4) demonstration budget
    for n in [10, 30, 100, 300, 1000, 4000]:
        for pol in ["linear_bc", "linear_d4", "mlp_bc", "student_3c_linprior", "student_3c"]:
            J.append(("budget", n, pol, 1, {}, {}, n))
    return J


if __name__ == "__main__":
    dataset(4000)  # build the cache once before forking
    jobs = build_jobs()
    # ONLY=pol1,pol2 re-runs just those policies and merges into an existing results.json
    only = [p for p in os.environ.get("ONLY", "").split(",") if p]
    old = []
    if only and os.path.exists(os.path.join(OUT, "results.json")):
        old = [r for r in json.load(open(os.path.join(OUT, "results.json"))) if r["policy"] not in only]
    if only:
        jobs = [j for j in jobs if j[2] in only]
    print(f"{len(jobs)} jobs", flush=True)
    t = time.time()
    out = []
    with mp.Pool(int(os.environ.get("N_PROC", mp.cpu_count()))) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            out.append(r)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(jobs)} done, {time.time()-t:.0f}s", flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(old + out, f, indent=1)
    print(f"finished in {time.time()-t:.0f}s -> {OUT}/results.json")
