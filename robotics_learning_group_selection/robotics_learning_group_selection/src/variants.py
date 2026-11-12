"""Alternative policies used for comparison / ablation (not the final submission).

- MLPPolicy, KNNPolicy, LinearNoBiasPolicy: memoryless behavioural cloning variants.
- RandomEscapePolicy (3a): BC + random escape after a detected collision.
- BugPolicy (3b): BC + map-free wall following (Bug-style) after a collision.
"""
import numpy as np

from .policy import LinearPolicy, ContactTracker, _unit, d4_augment


class MLPD4Policy:
    """MLP BC trained on D4-augmented demos (8x data)."""

    def __init__(self, X, Y, **kw):
        Xa, Ya = d4_augment(X, Y)
        self.net = MLPPolicy(Xa, Ya, **kw)

    def act(self, obs):
        return self.net.act(obs)


# --------------------------------------------------------------------------- #
# Memoryless BC variants
# --------------------------------------------------------------------------- #
def _features(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    d = np.linalg.norm(X, axis=1, keepdims=True)
    u = X / np.maximum(d, 1e-8)
    return np.concatenate([X, d, u], axis=1)


class LinearNoBiasPolicy:
    """Least squares without the bias column: a = W^T obs."""

    def __init__(self, X, Y):
        self.W = np.linalg.pinv(X) @ Y

    def act(self, obs):
        return _unit(np.asarray(obs, float) @ self.W)


class MLPPolicy:
    """2-hidden-layer tanh MLP trained with Adam on MSE (pure NumPy)."""

    def __init__(self, X, Y, hidden=64, epochs=3000, lr=3e-3, wd=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        F = _features(X)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-8
        F = (F - self.mu) / self.sd
        sizes = [F.shape[1], hidden, hidden, Y.shape[1]]
        self.params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.params += [rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)]
        m = [np.zeros_like(p) for p in self.params]
        v = [np.zeros_like(p) for p in self.params]
        b1, b2 = 0.9, 0.999
        for t in range(1, epochs + 1):
            out, cache = self._forward(F)
            g_out = 2 * (out - Y) / len(F)
            grads = self._backward(g_out, cache)
            for i, (p, g) in enumerate(zip(self.params, grads)):
                g = g + wd * p
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                p -= lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + 1e-8)
        self.train_mse = float(np.mean((self._forward(F)[0] - Y) ** 2))

    def _forward(self, F):
        W1, b1, W2, b2, W3, b3 = self.params
        h1 = np.tanh(F @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        return h2 @ W3 + b3, (F, h1, h2)

    def _backward(self, g, cache):
        F, h1, h2 = cache
        W1, b1, W2, b2, W3, b3 = self.params
        gW3, gb3 = h2.T @ g, g.sum(0)
        g2 = (g @ W3.T) * (1 - h2 ** 2)
        gW2, gb2 = h1.T @ g2, g2.sum(0)
        g1 = (g2 @ W2.T) * (1 - h1 ** 2)
        gW1, gb1 = F.T @ g1, g1.sum(0)
        return [gW1, gb1, gW2, gb2, gW3, gb3]

    def act(self, obs):
        F = (_features(obs) - self.mu) / self.sd
        return _unit(self._forward(F)[0][0])


class KNNPolicy:
    """Average the actions of the k nearest demonstrations (in obs space)."""

    def __init__(self, X, Y, k=15):
        self.X, self.Y, self.k = np.asarray(X, float), np.asarray(Y, float), k

    def act(self, obs):
        d = np.sum((self.X - np.asarray(obs, float)) ** 2, axis=1)
        idx = np.argpartition(d, self.k)[: self.k]
        return _unit(self.Y[idx].mean(0))


# --------------------------------------------------------------------------- #
# 3a: random escape
# --------------------------------------------------------------------------- #
class RandomEscapePolicy:
    """BC prior; after a collision, move in a random direction for a few steps."""

    def __init__(self, X, Y, escape_steps=3, seed=0, prior=None):
        self.prior = prior if prior is not None else LinearPolicy(X, Y)
        self.tracker = ContactTracker()
        self.escape_steps = escape_steps
        self.rng = np.random.default_rng(seed)
        self.left, self.dir = 0, None

    def act(self, obs):
        ev = self.tracker.observe(obs)
        if ev["new_episode"]:
            self.left = 0
        if ev["collided"]:
            base = np.arctan2(obs[1], obs[0])
            ang = base + self.rng.choice([-1, 1]) * self.rng.uniform(np.pi / 3, 5 * np.pi / 6)
            self.dir = np.array([np.cos(ang), np.sin(ang)])
            self.left = self.escape_steps
        if self.left > 0:
            self.left -= 1
            a = self.dir
        else:
            a = self.prior.act(obs)
        return self.tracker.commit(_unit(a))


# --------------------------------------------------------------------------- #
# 3b: Bug-style wall following (no map)
# --------------------------------------------------------------------------- #
def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class BugPolicy:
    """Map-free wall following, in the spirit of Bug2 (Lumelsky & Stepanov).

    GO mode: follow the BC prior (≈ head to goal).
    On a collision: enter FOLLOW mode, remember the distance at the hit point and
    turn away from the obstacle (always to the same side).
    FOLLOW mode: after a successful step, turn back toward the wall by
    `probe` radians to keep contact; after a blocked step turn away by `turn`.
    Leave FOLLOW (back to GO) once we are `leave_margin` closer to the goal
    than at the hit point.
    """

    def __init__(self, X, Y, side=1, turn=np.pi / 4, probe=np.pi / 4,
                 leave_margin=0.04, prior=None):
        self.prior = prior if prior is not None else LinearPolicy(X, Y)
        self.tracker = ContactTracker()
        self.side, self.turn, self.probe = side, turn, probe
        self.leave_margin = leave_margin
        self._reset()

    def _reset(self):
        self.mode, self.heading, self.d_hit = "GO", None, None

    def act(self, obs):
        obs = np.asarray(obs, float)
        ev = self.tracker.observe(obs)
        if ev["new_episode"]:
            self._reset()
        dist = np.linalg.norm(obs)
        if self.mode == "GO":
            if ev["collided"]:
                self.mode, self.d_hit = "FOLLOW", dist
                self.heading = _rot(self.tracker.prev_act, self.side * np.pi / 2)
            else:
                return self.tracker.commit(self.prior.act(obs))
        else:  # FOLLOW
            if ev["collided"]:
                self.heading = _rot(self.heading, self.side * self.turn)
            else:
                if dist < self.d_hit - self.leave_margin:
                    self.mode = "GO"
                    return self.tracker.commit(self.prior.act(obs))
                self.heading = _rot(self.heading, -self.side * self.probe)
        return self.tracker.commit(_unit(self.heading))
