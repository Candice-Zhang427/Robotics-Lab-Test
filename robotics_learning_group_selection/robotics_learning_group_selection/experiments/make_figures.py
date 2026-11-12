"""Make all figures + summary tables from results/results.json.

Usage (from the repo root): python experiments/make_figures.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, Circle  # noqa: E402
from matplotlib import animation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, dataset, make_scenarios, evaluate, astar_path, point_in_rects  # noqa: E402
from src.policy import LinearPolicy, StudentPolicy  # noqa: E402
from src.variants import LinearNoBiasPolicy, MLPPolicy, KNNPolicy, RandomEscapePolicy, BugPolicy  # noqa: E402
from src.obstacles import expert_direction  # noqa: E402
from src.env import PointRobotEnv  # noqa: E402

FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- style (reference palette, light mode) --------------------------------- #
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"student_3c": "#2a78d6", "linear_bc": "#eb6834", "bug_3b": "#1baf7a", "escape_3a": "#eda100",
       "mlp_bc": "#e87ba4", "knn_bc": "#008300", "linear_nobias": "#6250d6", "linear_d4": "#6250d6",
       "mlp_d4": "#e87ba4", "student_3c_linprior": "#2a78d6", "oracle": MUTED}
MRK = {"student_3c": "o", "linear_bc": "s", "bug_3b": "^", "escape_3a": "D", "mlp_bc": "v",
       "knn_bc": "P", "linear_nobias": "X", "linear_d4": "X", "mlp_d4": "v",
       "student_3c_linprior": "o", "oracle": None}
LAB = {"student_3c": "3c map + plan (ours, D4 prior)", "linear_bc": "linear BC (baseline)", "bug_3b": "3b bug / wall-follow",
       "escape_3a": "3a random escape", "mlp_bc": "MLP BC", "knn_bc": "kNN BC", "linear_nobias": "linear BC, no bias",
       "linear_d4": "linear BC + D4 aug, no bias", "mlp_d4": "MLP BC + D4 aug",
       "student_3c_linprior": "3c with plain linear-BC prior", "oracle": "oracle (true map)"}
SHORT = {"student_3c": "3c", "linear_bc": "baseline", "bug_3b": "3b", "escape_3a": "3a", "oracle": "oracle",
         "mlp_bc": "MLP", "knn_bc": "kNN"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def draw_blocks(ax, blocks, origin=np.zeros(2)):
    for x0, y0, x1, y1 in blocks:
        ax.add_patch(Rectangle((x0 - origin[0], y0 - origin[1]), x1 - x0, y1 - y0,
                               facecolor="#d6d5cf", edgecolor="#a8a7a0", lw=0.8, zorder=1))


def world_ax(ax, title=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(AXIS)
    if title:
        ax.set_title(title, fontsize=9.5)


# ---- load results ---------------------------------------------------------- #
R = json.load(open(os.path.join(OUT, "results.json")))
agg = defaultdict(list)
for r in R:
    agg[(r["group"], str(r["level"]), r["policy"], r["n_demo"])].append(r)


def stat(group, level, pol, key, n=4000):
    v = [r[key] for r in agg[(group, str(level), pol, n)]]
    return float(np.mean(v)), float(np.std(v)), len(v)


METRICS = ["success_rate", "mean_final_distance", "mean_steps", "collision_episode_rate", "collision_steps"]

# ---- summary tables (markdown) -------------------------------------------- #
lines = []
pub_pols = ["linear_bc", "linear_nobias", "linear_d4", "mlp_bc", "mlp_d4", "knn_bc", "escape_3a", "bug_3b",
            "student_3c", "oracle"]
lines.append("## Public set (make_scenarios(300, seed=1), 80 steps) -- official notebook setting\n")
lines.append("| policy | success | final dist | steps | collision-episode rate | blocked steps / ep |")
lines.append("|---|---|---|---|---|---|")
for p in pub_pols:
    r = [x for x in agg[("public", "default", p, 4000)] if x["seed"] == 1][0]
    cs = "-" if np.isnan(r["collision_steps"]) else f"{r['collision_steps']:.2f}"
    lines.append(f"| {LAB[p]} | {r['success_rate']:.3f} | {r['mean_final_distance']:.3f} | {r['mean_steps']:.1f} | "
                 f"{r['collision_episode_rate']:.3f} | {cs} |")
lines.append("\n## Public-like, mean ± std over seeds 1,2,3\n")
lines.append("| policy | success | steps | collision-episode rate | blocked steps / ep |")
lines.append("|---|---|---|---|---|")
for p in pub_pols:
    s = stat("public", "default", p, "success_rate"); t = stat("public", "default", p, "mean_steps")
    c = stat("public", "default", p, "collision_episode_rate"); b = stat("public", "default", p, "collision_steps")
    bs = "-" if np.isnan(b[0]) else f"{b[0]:.2f} ± {b[1]:.2f}"
    lines.append(f"| {LAB[p]} | {s[0]:.3f} ± {s[1]:.3f} | {t[0]:.1f} ± {t[1]:.1f} | {c[0]:.3f} ± {c[1]:.3f} | {bs} |")

rob_pols = ["linear_bc", "escape_3a", "bug_3b", "student_3c", "oracle"]
sweeps = [("blocks", ["2-4", "4-6", "6-8", "8-10"], "number of blocks", "#blocks"),
          ("obs_noise", ["0.0", "0.005", "0.01", "0.02", "0.03"], "observation noise std", "obs noise"),
          ("act_noise", ["0.0", "0.1", "0.2", "0.3"], "action noise std", "act noise"),
          ("horizon", ["10", "15", "20", "30", "40", "80"], "max steps (4-6 blocks)", "max steps")]
for g, levels, title, short in sweeps:
    lines.append(f"\n## Success rate vs {title} (mean over seeds 11-13, 300 episodes each)\n")
    lines.append(f"| {short} | " + " | ".join(SHORT[p] for p in rob_pols) + " |")
    lines.append("|---" * (len(rob_pols) + 1) + "|")
    for lv in levels:
        lines.append(f"| {lv} | " + " | ".join(f"{stat(g, lv, p, 'success_rate')[0]:.3f}" for p in rob_pols) + " |")
lines.append("\n## Stress test: 6-8 blocks, obs noise .01, action noise .1, 40 steps (seeds 21-23)\n")
lines.append("| policy | success | steps | collision-episode rate | blocked steps / ep |")
lines.append("|---|---|---|---|---|")
lv = "6-8 blocks, obs .01, act .1, 40 steps"
for p in rob_pols:
    s, t, c, b = (stat("stress", lv, p, k) for k in ["success_rate", "mean_steps", "collision_episode_rate", "collision_steps"])
    bs = "-" if np.isnan(b[0]) else f"{b[0]:.2f}"
    lines.append(f"| {LAB[p]} | {s[0]:.3f} ± {s[1]:.3f} | {t[0]:.1f} | {c[0]:.3f} | {bs} |")
lines.append("\n## Demonstration budget (public seed 1)\n")
BUD = ["linear_bc", "linear_d4", "mlp_bc", "student_3c_linprior", "student_3c"]
lines.append("| n demos | linear BC | linear BC + D4 | MLP BC | 3c, plain linear prior | 3c (final, D4 prior) |")
lines.append("|---|---|---|---|---|---|")
for n in [10, 30, 100, 300, 1000, 4000]:
    lines.append(f"| {n} | " + " | ".join(f"{stat('budget', n, p, 'success_rate', n)[0]:.3f}"
                                         for p in BUD) + " |")

# ---- Figure 1: what the demonstrations contain ----------------------------- #
X, Y = dataset()
g = X / np.linalg.norm(X, axis=1, keepdims=True)
ang = np.degrees(np.arccos(np.clip(np.sum(g * Y, 1), -1, 1)))
frac30 = float(np.mean(ang > 30))
from src.policy import SymmetricLinearPolicy
from src.variants import MLPD4Policy
bc = {"linear_bc": LinearPolicy(X, Y), "linear_nobias": LinearNoBiasPolicy(X, Y),
      "linear_d4": SymmetricLinearPolicy(X, Y), "mlp_bc": MLPPolicy(X, Y), "mlp_d4": MLPD4Policy(X, Y),
      "knn_bc": KNNPolicy(X, Y)}
rng = np.random.default_rng(0)
Q = rng.uniform(-0.9, 0.9, (3000, 2))
Q = Q[np.linalg.norm(Q, axis=1) > 0.03]
dev = {}
for k, p in bc.items():
    A = np.array([p.act(q.astype(np.float32)) for q in Q])
    gq = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    dev[k] = np.degrees(np.arccos(np.clip(np.sum(A * gq, 1), -1, 1)))

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.3), gridspec_kw={"width_ratios": [1.1, 1.25, 1.0]})
ax = axs[0]
_f = os.path.join(OUT, "dev_empty.npy")
if os.path.exists(_f):
    dev_empty = np.load(_f)
else:  # expert direction vs goal direction with no obstacles and no noise
    _rng = np.random.default_rng(0)
    dev_empty = []
    for _ in range(2000):
        _s, _t = _rng.uniform(0.05, 0.95, 2), _rng.uniform(0.05, 0.95, 2)
        _a = expert_direction(_s, _t, np.zeros((0, 4)))
        dev_empty.append(np.degrees(np.arccos(np.clip(_a @ ((_t - _s) / np.linalg.norm(_t - _s)), -1, 1))))
    dev_empty = np.array(dev_empty)
    np.save(_f, dev_empty)
bins = np.arange(0, 181, 5)
ax.hist(ang, bins=bins, color=COL["student_3c"], edgecolor=SURF, linewidth=1, label="demos (noise + obstacles)")
ax.hist(dev_empty, bins=bins, weights=np.full(len(dev_empty), len(ang) / len(dev_empty)), histtype="step",
        color=INK, lw=1.5, label="expert, no obstacles, no noise")
ax.set_yscale("log")
ax.set_xlabel("angle between expert action and goal direction (deg)")
ax.set_ylabel("number of demos (log)")
ax.set_title("a. Demo deviation is mostly an A* grid artifact")
ax.set_xlim(0, 180)
ax.legend(loc="upper right", fontsize=8.5)
ax.text(0.45, 0.62, f"share > 30° off goal direction\n  demos: {100*frac30:.1f}%\n"
        f"  expert w/o obstacles: {100*np.mean(dev_empty > 30):.1f}%\n→ only ~2% is obstacle detours",
        transform=ax.transAxes, ha="left", va="top", color=INK2, fontsize=8.5)
ax = axs[1]
names = ["linear_bc", "linear_nobias", "linear_d4", "mlp_bc", "mlp_d4", "knn_bc"]
pub_succ = {p: [x for x in agg[("public", "default", p, 4000)] if x["seed"] == 1][0]["success_rate"] for p in names}
vals = [np.mean(dev[p]) for p in names]
ypos = np.arange(len(names))[::-1]
ax.barh(ypos, vals, color=[COL["student_3c"] if p == "linear_d4" else "#b9b8b1" for p in names], height=0.6)
for y, p, v in zip(ypos, names, vals):
    ax.text(v + 0.3, y, f"{v:.1f}°   (public success {pub_succ[p]:.3f})", va="center", fontsize=9, color=INK2)
ax.set_yticks(ypos); ax.set_yticklabels([LAB[p] for p in names], color=INK2)
ax.set_xlim(0, max(vals) * 2.3); ax.grid(axis="y", visible=False)
ax.set_xlabel("mean |deviation| of BC action from goal direction (deg)")
ax.set_title("b. Flexible BC copies the expert's diagonal-first habit")
ax = axs[2]
gx = np.linspace(-0.9, 0.9, 61)
GX, GY = np.meshgrid(gx, gx)
P = np.stack([GX.ravel(), GY.ravel()], 1)
A = np.array([bc["mlp_bc"].act(q.astype(np.float32)) for q in P])
signed = np.degrees(np.arctan2(P[:, 0] * A[:, 1] - P[:, 1] * A[:, 0], np.sum(P * A, 1))).reshape(GX.shape)
from matplotlib.colors import LinearSegmentedColormap
div = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8b8a", "#a32a2a"])
im = ax.imshow(signed, origin="lower", extent=[-0.9, 0.9, -0.9, 0.9], cmap=div, vmin=-20, vmax=20)
ax.grid(False); ax.set_aspect("equal")
ax.set_xlabel("Δx (goal − robot)"); ax.set_ylabel("Δy")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("signed deviation from goal dir (deg)", color=INK2); cb.outline.set_visible(False)
ax.set_title("c. MLP BC bends toward the diagonals")
fig.tight_layout()
save(fig, "fig1_demo_analysis.png")

# ---- Figure 2: the provided expert deadlocks when executed ------------------ #
public = make_scenarios(300, seed=1)
sc = public[2]
env = PointRobotEnv(seed=123)
env.reset(start=sc["start"], target=sc["target"], blocks=sc["blocks"])
traj = [env.pos.copy()]
for _ in range(80):
    _, _, d, info = env.step(expert_direction(env.pos, env.target, env.blocks))
    traj.append(env.pos.copy())
    if d:
        break
traj = np.array(traj)
stuck = traj[-1]
path = np.array(astar_path(stuck, sc["target"], sc["blocks"]))
a = expert_direction(stuck, sc["target"], sc["blocks"])
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.9))
ax = axs[0]
draw_blocks(ax, sc["blocks"])
full = np.array(astar_path(sc["start"], sc["target"], sc["blocks"]))
ax.plot(full[:, 0], full[:, 1], ":", color=MUTED, lw=1.5, label="A* plan (32×32 grid)")
ax.plot(traj[:, 0], traj[:, 1], "-o", color=COL["linear_bc"], ms=3, lw=1.6, label="executed expert")
ax.scatter(*sc["start"], s=60, color=INK, zorder=5, label="start")
ax.scatter(*sc["target"], s=120, marker="*", color=COL["student_3c"], zorder=5, label="goal")
world_ax(ax, f"a. Privileged A* expert, executed: stuck after {np.argmax(np.all(np.diff(traj,axis=0)==0,axis=1))} steps")
ax.legend(loc="lower left", fontsize=8)
ax = axs[1]
draw_blocks(ax, sc["blocks"])
ax.plot(path[:6, 0], path[:6, 1], "o:", color=MUTED, ms=5, lw=1.2, label="next grid waypoints")
ax.scatter(*stuck, s=70, color=COL["linear_bc"], zorder=6, label="robot (stuck)")
ax.annotate("", xy=stuck + 0.08 * a, xytext=stuck, zorder=7,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
ax.scatter(*(stuck + 0.08 * a), s=60, marker="x", color="#d03b3b", zorder=7, label="step end point: inside block")
ax.text(*(stuck + np.array([0.004, 0.012])), f"waypoint {np.linalg.norm(path[1]-stuck):.3f} away\nstep = 0.080",
        fontsize=8.5, color=INK2)
c = stuck
ax.set_xlim(c[0] - 0.13, c[0] + 0.1); ax.set_ylim(c[1] - 0.1, c[1] + 0.13); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.set_title("b. Zoom: grid resolution 0.032 < step 0.08")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
save(fig, "fig2_expert_deadlock.png")

# ---- Figure 3: trajectories ------------------------------------------------- #
def rollout(policy, sc, **kw):
    r = evaluate(policy, [sc], **kw)
    return r["trajectories"][0]["positions"], r["success_rate"] > 0


dense = make_scenarios(300, n_blocks=(6, 8), seed=12)
Xd, Yd = X, Y
lin = LinearPolicy(X, Y)
fails_pub = [i for i, s in enumerate(public) if not rollout(lin, s)[1]]
fails_den = [i for i, s in enumerate(dense) if not rollout(lin, s)[1]]
picks = [("public", public[fails_pub[0]]), ("public", public[fails_pub[3]]),
         ("6-8 blocks", dense[fails_den[1]]), ("6-8 blocks", dense[fails_den[5]])]
cols = ["linear_bc", "escape_3a", "bug_3b", "student_3c"]
fig, axs = plt.subplots(len(picks), 4, figsize=(12, 3.05 * len(picks)))
for r_, (tag, s) in enumerate(picks):
    for c_, p in enumerate(cols):
        pol = {"linear_bc": lambda: LinearPolicy(X, Y), "escape_3a": lambda: RandomEscapePolicy(X, Y),
               "bug_3b": lambda: BugPolicy(X, Y), "student_3c": lambda: StudentPolicy(X, Y)}[p]()
        pos, ok = rollout(pol, s)
        ax = axs[r_, c_]
        draw_blocks(ax, s["blocks"])
        if p == "student_3c":
            for q in pol.hits:
                ax.add_patch(Circle(s["target"] + q, pol.inflate, facecolor="#e34948", alpha=0.18,
                                    edgecolor="none", zorder=2))
                ax.scatter(*(s["target"] + q), marker="x", s=22, color="#d03b3b", zorder=6, lw=1.2)
        blocked = int(np.sum(np.all(np.diff(pos, axis=0) == 0, axis=1)))
        ax.plot(pos[:, 0], pos[:, 1], "-", color=COL[p], lw=1.8, zorder=4)
        ax.scatter(pos[:, 0], pos[:, 1], s=7, color=COL[p], zorder=4)
        ax.scatter(*s["start"], s=40, color=INK, zorder=5)
        ax.scatter(*s["target"], s=110, marker="*", color=COL["student_3c"], edgecolor=SURF, zorder=5)
        status = "success" if ok else "FAIL"
        world_ax(ax, f"{LAB[p]}\n{status}, {len(pos)-1} steps, {blocked} blocked")
        if c_ == 0:
            ax.set_ylabel(tag, color=INK2)
fig.suptitle("Same scenarios, four policies (black dot = start, star = goal, red × = contact points the "
             "3c policy discovered)", x=0.01, ha="left", fontsize=10.5, color=INK2)
fig.tight_layout()
save(fig, "fig3_trajectories.png")

# ---- Figure 4: robustness -------------------------------------------------- #
fig, axs = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)
for ax, (g, levels, title, short) in zip(axs, sweeps):
    xs = np.arange(len(levels))
    for p in rob_pols:
        m = np.array([stat(g, lv, p, "success_rate")[0] for lv in levels])
        s = np.array([stat(g, lv, p, "success_rate")[1] for lv in levels])
        if p == "oracle":
            ax.plot(xs, m, "--", color=MUTED, lw=1.5, zorder=2)
        else:
            ax.errorbar(xs, m, yerr=s, color=COL[p], marker=MRK[p], capsize=2, elinewidth=1, zorder=3,
                        markeredgecolor=SURF, markeredgewidth=1)
    ax.set_xticks(xs); ax.set_xticklabels(levels)
    ax.set_xlabel(title); ax.set_ylim(0.45, 1.02)
    ax.set_title({"blocks": "a. More blocks", "obs_noise": "b. Observation noise",
                  "act_noise": "c. Action noise", "horizon": "d. Shorter horizon"}[g])
axs[0].set_ylabel("success rate")
handles = [plt.Line2D([], [], color=COL[p], marker=MRK[p], ls="--" if p == "oracle" else "-",
                      lw=1.5 if p == "oracle" else 2) for p in rob_pols]
fig.legend(handles, [LAB[p] for p in rob_pols], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
save(fig, "fig4_robustness_success.png")

# ---- Figure 5: efficiency & collisions vs blocks --------------------------- #
levels = sweeps[0][1]; xs = np.arange(len(levels))
fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
for ax, key, ttl, yl in [(axs[0], "mean_steps", "a. Episode length", "mean steps (80 = timeout)"),
                         (axs[1], "collision_episode_rate", "b. Collision-episode rate", "fraction of episodes with ≥1 blocked step"),
                         (axs[2], "collision_steps", "c. Blocked steps per episode", "mean blocked steps / episode (log)")]:
    for p in rob_pols:
        if key == "collision_steps" and p == "oracle":
            continue
        m = np.array([stat("blocks", lv, p, key)[0] for lv in levels])
        if p == "oracle":
            ax.plot(xs, m, "--", color=MUTED, lw=1.5)
        else:
            ax.plot(xs, m, color=COL[p], marker=MRK[p], markeredgecolor=SURF, markeredgewidth=1)
        if key != "collision_episode_rate":
            ax.annotate(SHORT[p], (xs[-1], m[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK2)
    if key == "collision_episode_rate":
        ax.text(0.03, 0.97, "baseline, 3a, 3b, 3c overlap:\nwithout sensors the first contact\nis unavoidable "
                "(only the oracle,\nwhich sees the map, avoids it)", transform=ax.transAxes, va="top",
                fontsize=8.5, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(levels); ax.set_xlabel("number of blocks")
    ax.set_title(ttl); ax.set_ylabel(yl)
    ax.set_xlim(-0.2, len(levels) - 0.4)
axs[2].set_yscale("log")
fig.legend(handles, [LAB[p] for p in rob_pols], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout()
save(fig, "fig5_efficiency_collisions.png")

# ---- Figure 6: demo budget ------------------------------------------------- #
ns = [10, 30, 100, 300, 1000, 4000]
fig, ax = plt.subplots(figsize=(8.6, 4.3))
ends = {}
for p in ["student_3c", "student_3c_linprior", "linear_d4", "linear_bc", "mlp_bc"]:
    m = [stat("budget", n, p, "success_rate", n)[0] for n in ns]
    dashed = p == "student_3c_linprior"
    ax.plot(ns, m, color=COL[p], marker=MRK[p], ls="--" if dashed else "-",
            markerfacecolor=SURF if dashed else COL[p], markeredgecolor=COL[p] if dashed else SURF,
            markeredgewidth=1.2)
    ends[p] = m[0]
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlim(7, 6000); ax.set_ylim(0, 1.04)
ax.legend([LAB[p] for p in ["student_3c", "student_3c_linprior", "linear_d4", "linear_bc", "mlp_bc"]],
          loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, handlelength=4)
ax.annotate("n = 30: fitted bias ≈ (0.13, −0.15)\ndominates near the goal → robot\ncircles at ~0.1, never < 0.05",
            xy=(30, 0.13), xytext=(48, 0.03), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.set_xlabel("number of demonstrations (log)"); ax.set_ylabel("success rate (public, seed 1)")
ax.set_title("Demo budget: symmetry makes 10 demos as good as 4000")
fig.tight_layout()
save(fig, "fig6_demo_budget.png")

# ---- GIF: baseline vs 3c on a dense scenario ------------------------------- #
s = dense[fails_den[5]]
pb, _ = rollout(LinearPolicy(X, Y), s)
stu = StudentPolicy(X, Y)
env = PointRobotEnv(seed=123)
obs = env.reset(start=s["start"], target=s["target"], blocks=s["blocks"])
ps, hit_hist = [env.pos.copy()], [[]]
for _ in range(80):
    obs, _, done, _ = env.step(stu.act(obs))
    ps.append(env.pos.copy())
    hit_hist.append([s["target"] + q for q in stu.hits])
    if done:
        break
ps = np.array(ps)
T = max(len(pb), len(ps)) + 6
T = min(T, 45)
fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.4))
arts = []
for ax, p in zip(axs, ["linear_bc", "student_3c"]):
    draw_blocks(ax, s["blocks"])
    ax.scatter(*s["start"], s=40, color=INK, zorder=5)
    ax.scatter(*s["target"], s=120, marker="*", color=COL["student_3c"], edgecolor=SURF, zorder=5)
    world_ax(ax)
    ln, = ax.plot([], [], "-o", color=COL[p], ms=3, lw=1.8, zorder=4)
    title = ax.set_title(LAB[p], fontsize=10)
    arts.append((ln, title))
hit_sc = axs[1].scatter([], [], marker="x", s=30, color="#d03b3b", zorder=6)
discs = []


def frame(t):
    for (ln, title), P, p in zip(arts, [pb, ps], ["linear_bc", "student_3c"]):
        k = min(t, len(P) - 1)
        ln.set_data(P[:k + 1, 0], P[:k + 1, 1])
        blocked = int(np.sum(np.all(np.diff(P[:k + 1], axis=0) == 0, axis=1)))
        done = "  ✓ reached goal" if (k == len(P) - 1 and np.linalg.norm(P[-1] - s["target"]) < 0.05) else ""
        title.set_text(f"{LAB[p]}\nstep {k}, blocked {blocked}{done}")
    h = hit_hist[min(t, len(hit_hist) - 1)]
    hit_sc.set_offsets(np.array(h) if h else np.empty((0, 2)))
    while len(discs) < len(h):
        discs.append(axs[1].add_patch(Circle(h[len(discs)], stu.inflate, facecolor="#e34948", alpha=0.18,
                                             edgecolor="none", zorder=2)))
    return []


ani = animation.FuncAnimation(fig, frame, frames=T, interval=250)
ani.save(os.path.join(FIG, "demo_baseline_vs_3c.gif"), writer=animation.PillowWriter(fps=4), dpi=110)
plt.close(fig)
print("wrote demo_baseline_vs_3c.gif")

# ---- write tables ----------------------------------------------------------- #
extra = [f"\n## Demonstration statistics\n", f"- {100*frac30:.1f}% of the 4000 demo actions deviate > 30° from the goal direction.",
         "- Mean |deviation| of BC output from goal direction: " +
         ", ".join(f"{LAB[k]} {np.mean(v):.1f}°" for k, v in dev.items()),
         f"- Public-set scenarios where the baseline fails: {len(fails_pub)}; 6-8-block set (seed 12): {len(fails_den)}"]
with open(os.path.join(OUT, "summary.md"), "w") as f:
    f.write("\n".join(lines + extra) + "\n")
print("wrote summary.md")
