import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_trajectory(traj_entry, title="Trajectory"):
    if isinstance(traj_entry, dict):
        traj = traj_entry["positions"]
        start = traj_entry["start"]
        target = traj_entry["target"]
        blocks = traj_entry.get("blocks")
    else:
        traj = traj_entry
        start = target = blocks = None

    fig, ax = plt.subplots(figsize=(5, 5))
    if blocks is not None and len(blocks):
        for x0, y0, x1, y1 in blocks:
            ax.add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.75", edgecolor="0.3")
            )
    ax.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2, color="C0")
    if start is not None:
        ax.scatter([start[0]], [start[1]], s=70, label="start", color="C0")
    if target is not None:
        ax.scatter([target[0]], [target[1]], s=70, label="target", color="C1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    return fig
