"""Privileged reference policy (uses the true block layout) -- NOT a submission.

Cost-to-go field on a fine world grid + choose the action whose real 0.08 end
point is collision-free and has the lowest cost-to-go. Gives an upper bound for
what a perfect-information policy can do in each test suite.
"""
import heapq
import numpy as np
from common import PointRobotEnv, point_in_rects

NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
      (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]


class Oracle:
    def __init__(self, res=0.01, n_dirs=72, step=0.08):
        self.res, self.step = res, step
        self.N = int(round(1 / res)) + 1
        a = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
        self.dirs = np.stack([np.cos(a), np.sin(a)], 1)

    def begin(self, sc):
        N, r = self.N, self.res
        xs = np.arange(N) * r
        blocked = np.zeros((N, N), bool)
        for x0, y0, x1, y1 in sc["blocks"]:
            mx = (xs >= x0 - 0.005) & (xs <= x1 + 0.005)
            my = (xs >= y0 - 0.005) & (xs <= y1 + 0.005)
            blocked[np.ix_(mx, my)] = True
        V = np.full((N, N), np.inf)
        gi, gj = np.clip(np.round(sc["target"] / r).astype(int), 0, N - 1)
        V[gi, gj] = 0
        heap = [(0.0, gi, gj)]
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > V[i, j]:
                continue
            for di, dj, w in NB:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N and not blocked[ni, nj] and d + w < V[ni, nj]:
                    V[ni, nj] = d + w
                    heapq.heappush(heap, (d + w, ni, nj))
        self.V, self.blocks, self.target = V * r, sc["blocks"], sc["target"]

    def act_world(self, pos):
        if np.linalg.norm(self.target - pos) < self.step:
            d = self.target - pos
            return d / np.linalg.norm(d)
        ends = np.clip(pos + self.step * self.dirs, 0, 1)
        c = np.clip(np.round(ends / self.res).astype(int), 0, self.N - 1)
        score = self.V[c[:, 0], c[:, 1]] + 1e-3 * np.linalg.norm(ends - self.target, axis=1)
        for k in np.argsort(score):
            if np.isfinite(score[k]) and not point_in_rects(ends[k], self.blocks):
                return self.dirs[k]
        d = self.target - pos
        return d / max(np.linalg.norm(d), 1e-8)


def eval_oracle(scenarios, noise_std=0.0, action_noise_std=0.0, max_steps=80, seed=123):
    env = PointRobotEnv(noise_std=noise_std, action_noise_std=action_noise_std,
                        max_steps=max_steps, seed=seed)
    orc = Oracle()
    S, D, T, C = [], [], [], []
    for sc in scenarios:
        env.reset(start=sc["start"], target=sc["target"], blocks=sc["blocks"])
        orc.begin(sc)
        info, k = {"success": False, "distance": np.inf}, 0
        for _ in range(max_steps):
            _, _, done, info = env.step(orc.act_world(env.pos))
            k += 1
            if done:
                break
        S.append(info["success"]); D.append(info["distance"]); T.append(k); C.append(env.collisions > 0)
    return {"success_rate": float(np.mean(S)), "mean_final_distance": float(np.mean(D)),
            "mean_steps": float(np.mean(T)), "collision_episode_rate": float(np.mean(C)),
            "success_flags": np.array(S)}
