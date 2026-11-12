import heapq

import numpy as np


def _unit(a):
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return (a / n).astype(np.float32)


class LinearPolicy:
    """Provided baseline: least-squares linear behavioral cloning."""
    def __init__(self, X, Y):
        Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        self.W = np.linalg.pinv(Xb) @ Y

    def act(self, obs):
        x = np.concatenate([obs, [1.0]])
        a = x @ self.W
        n = np.linalg.norm(a)
        if n > 1.0:
            a = a / n
        elif n > 1e-8:
            a = a / n
        return a.astype(np.float32)


# --------------------------------------------------------------------------- #
# Symmetry-aware behavioural cloning
# --------------------------------------------------------------------------- #
# The 8 symmetries of the square (dihedral group D4) as 2x2 matrices.
D4 = [np.array(m, dtype=float) for m in (
    [[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],   # rotations
    [[1, 0], [0, -1]], [[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]],   # reflections
)]


def d4_augment(X, Y):
    """Start/target are uniform in a square and blocks are axis-aligned with iid
    width/height, so the scenario distribution is invariant under D4. A demo
    (obs, action) therefore stays valid after applying the same g in D4 to both:
    (g obs, g action). This multiplies the demo budget by 8 for free."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    return (np.concatenate([X @ g.T for g in D4]).astype(np.float32),
            np.concatenate([Y @ g.T for g in D4]).astype(np.float32))


class SymmetricLinearPolicy:
    """Linear BC fitted on D4-augmented demos, without a bias term.

    A bias is forbidden by the symmetry (E[a | -obs] = -E[a | obs]); estimating it
    only adds variance, and near the goal (obs -> 0) a non-zero bias dominates the
    action and makes the robot circle around the goal.
    """

    def __init__(self, X, Y):
        Xa, Ya = d4_augment(X, Y)
        self.W = np.linalg.pinv(Xa) @ Ya

    def act(self, obs):
        return _unit(np.asarray(obs, dtype=float) @ self.W)


# --------------------------------------------------------------------------- #
# Turning the observation stream into events
# --------------------------------------------------------------------------- #
class ContactTracker:
    """Extracts the hidden signals in a stream of relative observations.

    obs = target - robot. Within an episode the target is fixed, so
      * robot position in a goal-centred frame is  p = -obs  (no drift: every
        observation is a fresh measurement), and
      * a normal step changes obs by ~step_size; a blocked step changes it by ~0
        -> contact / collision sensing for free.
    The evaluator never calls reset(), so a new episode is detected from
      * the previous observation being inside the success radius, or
      * obs jumping by much more than one step.
    """

    def __init__(self, step=0.08, success_dist=0.05, jump=2.0, stall=0.5):
        self.step = step
        self.success_dist = success_dist
        self.jump = jump * step
        self.stall = stall * step
        self.prev_obs = None
        self.prev_act = None

    def observe(self, obs):
        obs = np.asarray(obs, dtype=float)
        ev = {"new_episode": False, "collided": False, "hit_point": None}
        if self.prev_obs is None or self.prev_act is None:
            ev["new_episode"] = True
        else:
            moved = float(np.linalg.norm(obs - self.prev_obs))
            if np.linalg.norm(self.prev_obs) < self.success_dist or moved > self.jump:
                ev["new_episode"] = True
            elif moved < self.stall and np.linalg.norm(self.prev_act) > 1e-6:
                ev["collided"] = True
                # the point we tried to reach (goal frame) is inside an obstacle
                ev["hit_point"] = -self.prev_obs + self.step * self.prev_act
        if ev["new_episode"]:
            self.prev_act = None
        self.prev_obs = obs
        return ev

    def commit(self, action):
        self.prev_act = np.asarray(action, dtype=float)
        return np.asarray(action, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Final policy: learned free-space prior + contact map + value-field planning
# --------------------------------------------------------------------------- #
_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]


class StudentPolicy:
    """Behavioural-cloning prior + contact mapping + replanning.

    * Free space: follow the BC policy learned from the (obs, action) demos
      (default: SymmetricLinearPolicy = linear BC + D4 augmentation, no bias).
    * Every detected collision marks the attempted end point (goal-centred frame)
      as an obstacle disc on an occupancy grid; unknown cells are assumed free
      (the free-space assumption of D* Lite).
    * When the straight line to the goal is not known to be free, compute a
      cost-to-go field V on the grid (Dijkstra from the goal, i.e. value iteration
      on the map) and pick, among `n_dirs` unit actions, the one whose *actual*
      0.08 end point has the lowest V. Scoring the true end point (not the next
      grid waypoint) is what makes the plan executable with a large step.

    No obstacle layout, target or scenario information is used: act() only sees
    the relative observation [dx, dy].
    """

    def __init__(self, X, Y, prior=None, step=0.08, res=0.02, extent=1.04,
                 inflate=0.06, n_dirs=48, jump=2.0, stall=0.5):
        self.prior = prior if prior is not None else SymmetricLinearPolicy(X, Y)
        self.tracker = ContactTracker(step=step, jump=jump, stall=stall)
        self.step, self.res, self.extent, self.inflate = step, res, extent, inflate
        self.N = int(round(2 * extent / res)) + 1
        r = int(np.ceil(inflate / res))
        ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
        keep = (ii ** 2 + jj ** 2) * res ** 2 <= inflate ** 2 + 1e-12
        self._disc = np.stack([ii[keep], jj[keep]], axis=1)
        ang = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
        self._dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.stats = {"replans": 0, "hits": 0}
        self._reset()

    # ---- map helpers ------------------------------------------------------ #
    def _reset(self):
        self.blocked = np.zeros((self.N, self.N), dtype=bool)
        self.visited = np.zeros((self.N, self.N), dtype=bool)
        self.V = None
        self.hits = []

    def _cells(self, pts):
        c = np.round((np.atleast_2d(pts) + self.extent) / self.res).astype(int)
        ok = np.all((c >= 0) & (c < self.N), axis=1)
        return c, ok

    def _mark_hit(self, q):
        c, _ = self._cells(q)
        cells = c[0] + self._disc
        ok = np.all((cells >= 0) & (cells < self.N), axis=1)
        cells = cells[ok]
        self.blocked[cells[:, 0], cells[:, 1]] = True
        self.blocked &= ~self.visited  # places we have stood on are free
        self.hits.append(np.asarray(q, float))
        self.V = None
        self.stats["hits"] += 1

    def _segment_free(self, a, b):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (0.5 * self.res))) + 1)
        pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
        c, ok = self._cells(pts)
        if not ok.all():
            return False
        return not self.blocked[c[:, 0], c[:, 1]].any()

    def _value_field(self):
        """Dijkstra from the goal cell over unblocked cells (8-connected)."""
        N, blocked = self.N, self.blocked
        V = np.full((N, N), np.inf)
        g = int(round(self.extent / self.res))
        V[g, g] = 0.0
        heap = [(0.0, g, g)]
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > V[i, j]:
                continue
            for di, dj, w in _NEIGHBORS:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N and not blocked[ni, nj]:
                    nd = d + w
                    if nd < V[ni, nj]:
                        V[ni, nj] = nd
                        heapq.heappush(heap, (nd, ni, nj))
        self.stats["replans"] += 1
        return V * self.res

    # ---- policy ------------------------------------------------------------ #
    def act(self, obs):
        obs = np.asarray(obs, dtype=float)
        ev = self.tracker.observe(obs)
        if ev["new_episode"]:
            self._reset()
        p = -obs  # robot position in the goal-centred frame
        c, ok = self._cells(p)
        if ok[0]:
            self.visited[c[0, 0], c[0, 1]] = True
            self.blocked[c[0, 0], c[0, 1]] = False
        if ev["collided"]:
            self._mark_hit(ev["hit_point"])

        if not self.hits or self._segment_free(p, np.zeros(2)):
            return self.tracker.commit(self.prior.act(obs.astype(np.float32)))

        if self.V is None:
            self.V = self._value_field()
        ends = p + self.step * self._dirs
        mids = p + 0.5 * self.step * self._dirs
        ce, oke = self._cells(ends)
        cm, okm = self._cells(mids)
        score = np.full(len(ends), np.inf)
        good = oke & okm
        idx = np.where(good)[0]
        if len(idx):
            free = ~self.blocked[ce[idx, 0], ce[idx, 1]] & ~self.blocked[cm[idx, 0], cm[idx, 1]]
            idx = idx[free]
            score[idx] = self.V[ce[idx, 0], ce[idx, 1]] + 1e-3 * np.linalg.norm(ends[idx], axis=1)
        k = int(np.argmin(score))
        if not np.isfinite(score[k]):  # boxed in by the (inflated) map: fall back
            return self.tracker.commit(self.prior.act(obs.astype(np.float32)))
        return self.tracker.commit(_unit(self._dirs[k]))


# Implement StudentPolicy in this file (optional class name if you document it in your report).
# Training data: make_dataset() -> (X, Y), fixed demo budget unless you justify otherwise.
# At runtime, act(obs) receives obs = [target_x - robot_x, target_y - robot_y].
