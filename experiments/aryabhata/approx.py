"""Ways to approximate sine: series, CORDIC, interpolation, and small learners.

Ground truth for every score is math.sin. The learners never call it except
to build their training targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from experiments.aryabhata.jya import rollout


def taylor_sin(theta: float, terms: int) -> float:
    """Madhava / Taylor sine. `terms` counts summands, so 1 is just theta.

    The series was written out for sine by Madhava of Sangamagrama around
    1400, and again by Taylor in 1715. Same coefficients either way.
    """
    if terms < 1:
        raise ValueError("terms must be positive")
    total = 0.0
    term = theta
    for index in range(terms):
        total += term
        term *= -(theta * theta) / ((2 * index + 2) * (2 * index + 3))
    return total


def cordic_sin(theta: float, iterations: int = 16) -> float:
    """Shift-add sine. The usual current method on hardware without a multiplier.

    Valid on [-pi/2, pi/2], which covers the quadrant Aryabhata tabulated.
    The arctangent constants are the small auxiliary table CORDIC needs
    instead of a sine table.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    sign = -1.0 if theta < 0 else 1.0
    angle = abs(theta)
    if angle > math.pi / 2 + 1e-12:
        raise ValueError("cordic_sin is only defined on [-pi/2, pi/2] here")
    gain = 1.0
    for index in range(iterations):
        gain *= 1.0 / math.sqrt(1.0 + 2.0 ** (-2 * index))
    x = gain
    y = 0.0
    z = angle
    for index in range(iterations):
        power = 2.0 ** (-index)
        sigma = 1.0 if z >= 0.0 else -1.0
        x, y = x - sigma * y * power, y + sigma * x * power
        z -= sigma * math.atan(power)
    return sign * y


def linear_interpolation(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """np.interp clamps outside the training interval."""
    order = np.argsort(train_x)
    return np.interp(test_x, train_x[order], train_y[order])


def odd_polynomial_features(angles: np.ndarray, n_terms: int, scale: float) -> np.ndarray:
    """Columns (angle/scale), (angle/scale)^3, ... so coefficients stay moderate."""
    scaled = angles / scale
    return np.stack([scaled ** (2 * i + 1) for i in range(n_terms)], axis=1)


def fit_odd_polynomial(
    train_x: np.ndarray, train_y: np.ndarray, n_terms: int
) -> tuple[np.ndarray, float]:
    scale = float(np.max(np.abs(train_x)))
    if scale == 0.0:
        scale = 1.0
    design = odd_polynomial_features(train_x, n_terms, scale)
    coef, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    return coef, scale


def predict_odd_polynomial(
    angles: np.ndarray, coef: np.ndarray, scale: float
) -> np.ndarray:
    design = odd_polynomial_features(angles, len(coef), scale)
    return design @ coef


@dataclass(frozen=True)
class MLPWeights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    scale: float

    @property
    def parameter_count(self) -> int:
        return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)


def _adam_step(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    moment: list[np.ndarray],
    velocity: list[np.ndarray],
    step: int,
    lr: float,
) -> None:
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for index, (param, grad) in enumerate(zip(params, grads)):
        moment[index] = beta1 * moment[index] + (1 - beta1) * grad
        velocity[index] = beta2 * velocity[index] + (1 - beta2) * (grad * grad)
        mhat = moment[index] / (1 - beta1**step)
        vhat = velocity[index] / (1 - beta2**step)
        param -= lr * mhat / (np.sqrt(vhat) + eps)


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    hidden: int = 8,
    steps: int = 2500,
    lr: float = 0.03,
    seed: int = 0,
) -> MLPWeights:
    """One-hidden-layer tanh network. Input is angle divided by the training max."""
    rng = np.random.default_rng(seed)
    scale = float(np.max(np.abs(train_x))) or 1.0
    x = (train_x / scale).reshape(-1, 1)
    y = train_y.reshape(-1, 1)
    w1 = rng.normal(0.0, 1.0, size=(1, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, 1))
    b2 = np.zeros(1)
    params = [w1, b1, w2, b2]
    moment = [np.zeros_like(p) for p in params]
    velocity = [np.zeros_like(p) for p in params]
    for step in range(1, steps + 1):
        hidden_pre = x @ w1 + b1
        activated = np.tanh(hidden_pre)
        pred = activated @ w2 + b2
        # d(mean square error) / d(pred)
        dpred = (2.0 / len(x)) * (pred - y)
        dw2 = activated.T @ dpred
        db2 = dpred.sum(axis=0)
        dhidden = (dpred @ w2.T) * (1.0 - activated**2)
        dw1 = x.T @ dhidden
        db1 = dhidden.sum(axis=0)
        _adam_step(params, [dw1, db1, dw2, db2], moment, velocity, step, lr)
    return MLPWeights(w1=w1, b1=b1, w2=w2, b2=b2, scale=scale)


def predict_mlp(angles: np.ndarray, weights: MLPWeights) -> np.ndarray:
    x = (angles / weights.scale).reshape(-1, 1)
    activated = np.tanh(x @ weights.w1 + weights.b1)
    return (activated @ weights.w2 + weights.b2).reshape(-1)


def predict_recurrence(degree_steps: np.ndarray, alpha: float, delta0: float) -> np.ndarray:
    """Sine at integer steps k, where h is the step delta0 and alpha were built for.

    degree_steps are cast with int(). A half-degree such as 0.5 becomes 0.
    That path is integer-grid-only. Half-degree scores belong in robustness.py,
    which interpolates and labels the interpolation.
    """
    cached: dict[int, float] = {0: 0.0}
    out = np.empty(len(degree_steps), dtype=float)
    for index, raw in enumerate(degree_steps):
        k = int(raw)
        if k not in cached:
            cached[k] = rollout(k, alpha, delta0)[-1] if k else 0.0
        out[index] = cached[k]
    return out


def _loss_for_alpha(
    train_steps: np.ndarray, train_y: np.ndarray, alpha: float, delta0: float
) -> float:
    pred = predict_recurrence(train_steps, alpha, delta0)
    err = pred - train_y
    return float(np.mean(err * err))


def fit_alpha(
    train_steps: np.ndarray,
    train_y: np.ndarray,
    delta0: float,
    lo: float = 1e-10,
    hi: float = 0.5,
) -> float:
    """One-parameter fit of the second-difference coefficient. Arc init is fixed."""

    def loss(alpha: float) -> float:
        return _loss_for_alpha(train_steps, train_y, alpha, delta0)

    grid = np.geomspace(lo, hi, 80)
    values = np.array([loss(float(alpha)) for alpha in grid])
    best = int(np.argmin(values))
    left = float(grid[max(0, best - 1)])
    right = float(grid[min(len(grid) - 1, best + 1)])
    if math.isclose(left, right):
        return float(grid[best])
    return _golden(loss, left, right)


def _golden(func, left: float, right: float, iterations: int = 60) -> float:
    phi = (math.sqrt(5) - 1) / 2
    c = right - phi * (right - left)
    d = left + phi * (right - left)
    fc, fd = func(c), func(d)
    for _ in range(iterations):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - phi * (right - left)
            fc = func(c)
        else:
            left, c, fc = c, d, fd
            d = left + phi * (right - left)
            fd = func(d)
    return (left + right) / 2


def fit_alpha_and_delta0(
    train_steps: np.ndarray,
    train_y: np.ndarray,
    alpha0: float,
    delta0: float,
) -> tuple[float, float]:
    """Two-parameter Nelder-Mead on (alpha, delta0)."""

    def loss(point: np.ndarray) -> float:
        alpha = float(point[0])
        init = float(point[1])
        if alpha <= 0:
            return 1e6
        return _loss_for_alpha(train_steps, train_y, alpha, init)

    start = np.array([alpha0, delta0], dtype=float)
    point, _score = _nelder_mead(loss, start, step=0.2, max_iter=120)
    return float(point[0]), float(point[1])


def _nelder_mead(func, x0: np.ndarray, step: float, max_iter: int) -> tuple[np.ndarray, float]:
    n = x0.size
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0
        simplex[i + 1, i] += step * max(abs(x0[i]), 1e-3)
    scores = np.array([func(row) for row in simplex])
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        order = np.argsort(scores)
        simplex = simplex[order]
        scores = scores[order]
        centroid = simplex[:-1].mean(axis=0)
        reflected = centroid + alpha * (centroid - simplex[-1])
        score_r = func(reflected)
        if scores[0] <= score_r < scores[-2]:
            simplex[-1] = reflected
            scores[-1] = score_r
            continue
        if score_r < scores[0]:
            expanded = centroid + gamma * (reflected - centroid)
            score_e = func(expanded)
            if score_e < score_r:
                simplex[-1], scores[-1] = expanded, score_e
            else:
                simplex[-1], scores[-1] = reflected, score_r
            continue
        contracted = centroid + rho * (simplex[-1] - centroid)
        score_c = func(contracted)
        if score_c < scores[-1]:
            simplex[-1], scores[-1] = contracted, score_c
            continue
        simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
        scores[1:] = [func(row) for row in simplex[1:]]
    return simplex[0], float(scores[0])


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def max_abs(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(pred - truth)))
