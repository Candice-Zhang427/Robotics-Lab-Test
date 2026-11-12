import numpy as np
from .env import PointRobotEnv
from .obstacles import astar_path, sample_blocks

def make_scenarios(
    n=300,
    seed=0,
    region=(0.05, 0.95),
    n_blocks=(2, 4),
    resample_unreachable=False,
    grid_size=32,
):
    rng = np.random.default_rng(seed)
    lo, hi = region
    scenarios = []
    attempts = 0
    max_attempts = n * (40 if resample_unreachable else 1)

    while len(scenarios) < n and attempts < max_attempts:
        attempts += 1
        start = rng.uniform(lo, hi, 2)
        target = rng.uniform(lo, hi, 2)
        blocks = sample_blocks(rng, start, target, n_blocks=n_blocks, region=region)
        if resample_unreachable and astar_path(start, target, blocks, grid_size=grid_size) is None:
            continue
        scenarios.append(
            {"start": start.astype(float), "target": target.astype(float), "blocks": blocks}
        )
    if len(scenarios) < n:
        raise RuntimeError(f"Could only sample {len(scenarios)}/{n} scenarios.")
    return scenarios

def evaluate(
    policy,
    scenarios,
    noise_std=0.0,
    action_noise_std=0.0,
    max_steps=80,
    success_dist=0.05,
    seed=123,
    grid_size=32,
):
    env = PointRobotEnv(
        noise_std=noise_std,
        action_noise_std=action_noise_std,
        max_steps=max_steps,
        success_dist=success_dist,
        seed=seed,
        grid_size=grid_size,
    )
    successes, distances, step_counts, collision_flags = [], [], [], []
    trajectories = []

    for sc in scenarios:
        obs = env.reset(start=sc["start"], target=sc["target"], blocks=sc["blocks"])
        traj = [env.pos.copy()]
        info = {"success": False, "distance": np.inf}
        steps = 0

        for _ in range(env.max_steps):
            action = policy.act(obs)
            obs, _, done, info = env.step(action)
            steps += 1
            traj.append(env.pos.copy())
            if done:
                break

        successes.append(float(info["success"]))
        distances.append(info["distance"])
        step_counts.append(steps)
        collision_flags.append(float(env.collisions > 0))
        trajectories.append(
            {
                "positions": np.asarray(traj),
                "start": sc["start"],
                "target": sc["target"],
                "blocks": sc["blocks"],
            }
        )

    return {
        "success_rate": float(np.mean(successes)),
        "mean_final_distance": float(np.mean(distances)),
        "mean_steps": float(np.mean(step_counts)),
        "collision_episode_rate": float(np.mean(collision_flags)),
        "trajectories": trajectories,
    }
