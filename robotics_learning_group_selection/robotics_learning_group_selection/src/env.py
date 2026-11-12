import numpy as np
from .obstacles import point_in_rects, expert_direction, astar_path

class PointRobotEnv:
    """Tiny CPU-only 2D point robot with axis-aligned rectangular obstacles."""

    def __init__(
        self,
        step_size=0.08,
        max_steps=80,
        success_dist=0.05,
        noise_std=0.0,
        action_noise_std=0.0,
        seed=0,
        grid_size=32,
    ):
        self.step_size = step_size
        self.max_steps = max_steps
        self.success_dist = success_dist
        self.noise_std = noise_std
        self.action_noise_std = action_noise_std
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.blocks = np.zeros((0, 4), dtype=float)
        self.collisions = 0
        self.reset()

    def reset(self, start=None, target=None, blocks=None):
        self.pos = np.array(
            start if start is not None else self.rng.uniform(0.05, 0.95, 2),
            dtype=float,
        )
        self.target = np.array(
            target if target is not None else self.rng.uniform(0.05, 0.95, 2),
            dtype=float,
        )
        if blocks is None:
            self.blocks = np.zeros((0, 4), dtype=float)
        else:
            self.blocks = np.asarray(blocks, dtype=float).reshape(-1, 4)
        self.t = 0
        self.collisions = 0
        return self.observe()

    def observe(self):
        obs = (self.target - self.pos).astype(float)
        if self.noise_std > 0:
            obs = obs + self.rng.normal(0, self.noise_std, size=2)
        return obs.astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if self.action_noise_std > 0:
            action = action + self.rng.normal(0, self.action_noise_std, size=2)
        norm = np.linalg.norm(action)
        if norm > 1.0:
            action = action / norm
        elif norm > 1e-8:
            action = action / norm
        else:
            action = np.zeros(2, dtype=float)

        proposed = np.clip(self.pos + self.step_size * action, 0.0, 1.0)
        collision = point_in_rects(proposed, self.blocks)
        if collision:
            self.collisions += 1
        else:
            self.pos = proposed

        self.t += 1
        dist = float(np.linalg.norm(self.pos - self.target))
        in_obstacle = point_in_rects(self.pos, self.blocks)
        success = dist < self.success_dist and not in_obstacle
        done = success or self.t >= self.max_steps
        reward = 1.0 if success else -dist
        return self.observe(), reward, done, {
            "success": success,
            "distance": dist,
            "collision": collision,
            "in_obstacle": in_obstacle,
        }

def make_dataset(n=4000, seed=42, expert_action_noise=0.12, grid_size=32):
    """Noisy BC demos on reachable scenarios with random obstacles (fixed budget n)."""
    from .evaluate import make_scenarios

    rng = np.random.default_rng(seed)
    scenarios = make_scenarios(n, seed=seed, resample_unreachable=True, grid_size=grid_size)
    obs_list, act_list = [], []
    for sc in scenarios:
        o = (sc["target"] - sc["start"]).astype(np.float32)
        a = expert_direction(sc["start"], sc["target"], sc["blocks"], grid_size=grid_size)
        if expert_action_noise > 0:
            a = a + rng.normal(0, expert_action_noise, size=2).astype(np.float32)
            an = np.linalg.norm(a)
            if an > 1e-8:
                a = (a / an).astype(np.float32)
        obs_list.append(o)
        act_list.append(a)
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)
