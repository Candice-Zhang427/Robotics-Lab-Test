"""Re-evaluate all policies with a strict, segment-level collision check.

The provided simulator (src/env.py, PointRobotEnv.step) only tests whether the END POINT of a
0.08 step lies inside a block, so a step can cut through the corner of a block. StrictEnv blocks a
step if any point of its straight segment is inside a block. Policies are NOT changed.

Usage (from the repo root): python experiments/strict_collision.py
Writes results/strict_collision.json and prints a markdown table.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, dataset, make_scenarios, evaluate, point_in_rects, PointRobotEnv  # noqa: E402
from oracle import Oracle  # noqa: E402

U = np.linspace(0, 1, 42)[1:-1]  # 40 interior points per step (spacing 0.002)


def seg_hits(a, b, blocks):
    return any(point_in_rects(a + u * (b - a), blocks) for u in U)


def crossings(P, blocks):
    """Number of executed steps whose segment passes through a block."""
    return sum(seg_hits(P[k], P[k + 1], blocks) for k in range(len(P) - 1) if not np.allclose(P[k], P[k + 1]))


class StrictEnv(PointRobotEnv):
    """PointRobotEnv with a segment-level collision check (otherwise identical)."""

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if self.action_noise_std > 0:
            action = action + self.rng.normal(0, self.action_noise_std, size=2)
        n = np.linalg.norm(action)
        action = action / n if n > 1e-8 else np.zeros(2)
        proposed = np.clip(self.pos + self.step_size * action, 0.0, 1.0)
        collision = point_in_rects(proposed, self.blocks) or seg_hits(self.pos, proposed, self.blocks)
        if collision:
            self.collisions += 1
        else:
            self.pos = proposed
        self.t += 1
        dist = float(np.linalg.norm(self.pos - self.target))
        in_obstacle = point_in_rects(self.pos, self.blocks)
        success = dist < self.success_dist and not in_obstacle
        done = success or self.t >= self.max_steps
        return self.observe(), (1.0 if success else -dist), done, {
            "success": success, "distance": dist, "collision": collision, "in_obstacle": in_obstacle}


def run_oracle(scen, strict, **kw):
    env = (StrictEnv if strict else PointRobotEnv)(seed=123, **kw)
    orc = Oracle()
    S, T, C = [], [], []
    for sc in scen:
        env.reset(start=sc["start"], target=sc["target"], blocks=sc["blocks"])
        orc.begin(sc)
        P = [env.pos.copy()]
        for _ in range(kw.get("max_steps", 80)):
            pos = env.pos
            if np.linalg.norm(orc.target - pos) < orc.step:
                a = (orc.target - pos) / np.linalg.norm(orc.target - pos)
            else:
                ends = np.clip(pos + orc.step * orc.dirs, 0, 1)
                c = np.clip(np.round(ends / orc.res).astype(int), 0, orc.N - 1)
                score = orc.V[c[:, 0], c[:, 1]] + 1e-3 * np.linalg.norm(ends - orc.target, axis=1)
                a = orc.dirs[np.argmin(score)]
                for k in np.argsort(score):
                    if np.isfinite(score[k]) and not point_in_rects(ends[k], orc.blocks) and \
                            (not strict or not seg_hits(pos, ends[k], orc.blocks)):
                        a = orc.dirs[k]
                        break
            _, _, done, info = env.step(a)
            P.append(env.pos.copy())
            if done:
                break
        S.append(bool(info["success"])); T.append(len(P) - 1); C.append(crossings(np.array(P), sc["blocks"]))
    return S, T, C


def job(spec):
    group, pol, seed, skw, ekw, strict = spec
    scen = make_scenarios(300, seed=seed, **skw)
    if pol == "oracle":
        S, T, C = run_oracle(scen, strict, **ekw)
    else:
        import src.evaluate as ev
        from src.policy import LinearPolicy, StudentPolicy
        from src.variants import RandomEscapePolicy, BugPolicy
        X, Y = dataset()
        p = {"baseline": LinearPolicy, "3a": RandomEscapePolicy, "3b": BugPolicy, "3c": StudentPolicy}[pol](X, Y)
        ev.PointRobotEnv = StrictEnv if strict else PointRobotEnv
        r = evaluate(p, scen, **ekw)
        S = [bool(np.linalg.norm(t["positions"][-1] - t["target"]) < 0.05
                  and not point_in_rects(t["positions"][-1], t["blocks"])) for t in r["trajectories"]]
        T = [len(t["positions"]) - 1 for t in r["trajectories"]]
        C = [crossings(t["positions"], t["blocks"]) for t in r["trajectories"]]
    return dict(group=group, policy=pol, seed=seed, rule="strict" if strict else "original",
                success=float(np.mean(S)), steps=float(np.mean(T)),
                n_success=int(sum(S)), success_with_crossing=int(sum(1 for s, c in zip(S, C) if s and c > 0)),
                crossing_segments=int(sum(C)))


GROUPS = {
    "public": ([1], {}, {}),
    "6-8 blocks": ([11, 12, 13], {"n_blocks": (6, 8)}, {}),
    "stress": ([21, 22, 23], {"n_blocks": (6, 8)}, {"noise_std": 0.01, "action_noise_std": 0.1, "max_steps": 40}),
}
POLS = ["baseline", "3a", "3b", "3c", "oracle"]

if __name__ == "__main__":
    dataset()
    jobs = [(g, p, s, skw, ekw, strict) for g, (seeds, skw, ekw) in GROUPS.items()
            for p in POLS for s in seeds for strict in (False, True)]
    with mp.Pool(int(os.environ.get("N_PROC", mp.cpu_count()))) as pool:
        out = pool.map(job, jobs)
    with open(os.path.join(OUT, "strict_collision.json"), "w") as f:
        json.dump(out, f, indent=1)

    def agg(g, p, rule, key):
        return float(np.mean([r[key] for r in out if r["group"] == g and r["policy"] == p and r["rule"] == rule]))

    print("| policy | public: original → strict | 6-8 blocks: original → strict | stress: original → strict |")
    print("|---|---|---|---|")
    for p in POLS:
        cells = [f"{agg(g, p, 'original', 'success'):.3f} → {agg(g, p, 'strict', 'success'):.3f}" for g in GROUPS]
        print(f"| {p} | " + " | ".join(cells) + " |")
    print("\nsteps (original → strict):")
    for p in POLS:
        print(f"  {p:8s} " + "  ".join(f"{g}: {agg(g, p, 'original', 'steps'):.1f} → {agg(g, p, 'strict', 'steps'):.1f}"
                                       for g in GROUPS))
    print("\nsuccessful episodes with >= 1 segment through a block (original rule):")
    for p in POLS:
        print(f"  {p:8s} " + "  ".join(
            f"{g}: {sum(r['success_with_crossing'] for r in out if r['group']==g and r['policy']==p and r['rule']=='original')}"
            f"/{sum(r['n_success'] for r in out if r['group']==g and r['policy']==p and r['rule']=='original')}"
            for g in GROUPS))
