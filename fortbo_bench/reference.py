"""Independent behavioural references used by the benchmark evidence.

This module deliberately contains direct NumPy implementations rather than
wrappers around FortBO or its generated kernels.  The benchmark is useful only
when it can catch both a wrong implementation and a wrong test.  The formulas
are written from their definitions, with minimisation as the package-wide
convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

Array = np.ndarray
Value = Callable[[Array], Array]
Gradient = Callable[[Array], Array]


def _points(points: Array, dimension: int | None = None) -> Array:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or (dimension is not None and values.shape[1] != dimension):
        raise ValueError(f"expected points with shape (n, {dimension}), got {values.shape}")
    return values


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    dimension: int
    lower: Array
    upper: Array
    optimum: Array
    optimum_value: float
    value: Value
    gradient: Gradient

    def evaluate(self, points: Array) -> Array:
        return np.asarray(self.value(_points(points, self.dimension)), dtype=np.float64)

    def gradients(self, points: Array) -> Array:
        return np.asarray(self.gradient(_points(points, self.dimension)), dtype=np.float64)


def _sphere(x: Array) -> Array:
    return np.sum(x * x, axis=1)


def _sphere_gradient(x: Array) -> Array:
    return 2.0 * x


def _branin(x: Array) -> Array:
    x1, x2 = x[:, 0], x[:, 1]
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8.0 * math.pi)
    return a * (x2 - b*x1*x1 + c*x1 - r)**2 + s*(1.0-t)*np.cos(x1) + s


def _branin_gradient(x: Array) -> Array:
    x1, x2 = x[:, 0], x[:, 1]
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    t = 1.0 / (8.0 * math.pi)
    inner = x2 - b*x1*x1 + c*x1 - 6.0
    return np.column_stack((2.0*inner*(-2.0*b*x1 + c) - 10.0*(1.0-t)*np.sin(x1),
                            2.0*inner))


def _hartmann(x: Array, alpha: Array, a: Array, p: Array) -> Array:
    delta = x[:, None, :] - p[None, :, :]
    exponent = np.sum(a[None, :, :] * delta**2, axis=2)
    return -np.sum(alpha[None, :] * np.exp(-exponent), axis=1)


def _hartmann_gradient(x: Array, alpha: Array, a: Array, p: Array) -> Array:
    delta = x[:, None, :] - p[None, :, :]
    terms = alpha[None, :, None] * a[None, :, :] * delta * np.exp(
        -np.sum(a[None, :, :] * delta**2, axis=2))[:, :, None]
    return 2.0 * np.sum(terms, axis=1)


H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
H3_A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
H3_P = 1.0e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470],
                           [1091, 8732, 5547], [381, 5743, 8828]])
H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
H6_A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
H6_P = 1.0e-4 * np.array([
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381],
])


def _ackley(x: Array) -> Array:
    a, b, c = 20.0, 0.2, 2.0*math.pi
    mean_square = np.mean(x*x, axis=1)
    mean_cosine = np.mean(np.cos(c*x), axis=1)
    return -a*np.exp(-b*np.sqrt(mean_square)) - np.exp(mean_cosine) + a + math.e


def _ackley_gradient(x: Array) -> Array:
    a, b, c = 20.0, 0.2, 2.0*math.pi
    mean_square = np.mean(x*x, axis=1)
    radius = np.sqrt(mean_square)
    mean_cosine = np.mean(np.cos(c*x), axis=1)
    result = np.zeros_like(x)
    nonzero = radius > 0.0
    result[nonzero] = (a*b*np.exp(-b*radius[nonzero]) /
                       (x.shape[1]*radius[nonzero]))[:, None] * x[nonzero]
    result += (c*np.exp(mean_cosine) / x.shape[1])[:, None] * np.sin(c*x)
    result[~nonzero] = 0.0
    return result


def _rosenbrock(x: Array) -> Array:
    return np.sum(100.0*(x[:, 1:] - x[:, :-1]**2)**2 + (1.0-x[:, :-1])**2, axis=1)


def _rosenbrock_gradient(x: Array) -> Array:
    result = np.zeros_like(x)
    result[:, :-1] += -400.0*x[:, :-1]*(x[:, 1:] - x[:, :-1]**2) - 2.0*(1.0-x[:, :-1])
    result[:, 1:] += 200.0*(x[:, 1:] - x[:, :-1]**2)
    return result


def _levy(x: Array) -> Array:
    w = 1.0 + (x-1.0)/4.0
    return (np.sin(math.pi*w[:, 0])**2 +
            np.sum((w[:, :-1]-1.0)**2 * (1.0 + 10.0*np.sin(math.pi*w[:, :-1]+1.0)**2), axis=1) +
            (w[:, -1]-1.0)**2 * (1.0+np.sin(2.0*math.pi*w[:, -1])**2))


def _levy_gradient(x: Array) -> Array:
    # The direct derivative of the definition, including the end terms.
    w = 1.0 + (x-1.0)/4.0
    d = np.zeros_like(x)
    d[:, 0] += 2.0*math.pi*np.sin(math.pi*w[:, 0])*np.cos(math.pi*w[:, 0])/4.0
    d[:, -1] += ((2.0*(w[:, -1]-1.0) *
                  (1.0+np.sin(2.0*math.pi*w[:, -1])**2) +
                  (w[:, -1]-1.0)**2 * 4.0*math.pi*np.sin(2.0*math.pi*w[:, -1]) *
                  np.cos(2.0*math.pi*w[:, -1])) / 4.0)
    inner = w[:, :-1]-1.0
    factor = 1.0 + 10.0*np.sin(math.pi*w[:, :-1]+1.0)**2
    d[:, :-1] += (2.0*inner*factor + inner**2 * 10.0*math.pi *
                   np.sin(2.0*(math.pi*w[:, :-1]+1.0))) / 4.0
    return d


FUNCTION_SPECS: dict[str, FunctionSpec] = {
    "sphere": FunctionSpec("sphere", 2, -5.0*np.ones(2), 5.0*np.ones(2),
                            np.zeros(2), 0.0, _sphere, _sphere_gradient),
    "branin": FunctionSpec("branin", 2, np.array([-5.0, 0.0]), np.array([10.0, 15.0]),
                            np.array([-math.pi, 12.275]), 0.397887357729739, _branin,
                            _branin_gradient),
    "hartmann3": FunctionSpec("hartmann3", 3, np.zeros(3), np.ones(3),
                               np.array([0.114614, 0.555649, 0.852547]), -3.86278,
                               lambda x: _hartmann(x, H3_ALPHA, H3_A, H3_P),
                               lambda x: _hartmann_gradient(x, H3_ALPHA, H3_A, H3_P)),
    "hartmann6": FunctionSpec("hartmann6", 6, np.zeros(6), np.ones(6),
                               np.array([0.20169, 0.150011, 0.476874, 0.275332,
                                         0.311652, 0.6573]), -3.32237,
                               lambda x: _hartmann(x, H6_ALPHA, H6_A, H6_P),
                               lambda x: _hartmann_gradient(x, H6_ALPHA, H6_A, H6_P)),
    "ackley": FunctionSpec("ackley", 8, -5.0*np.ones(8), 10.0*np.ones(8),
                            np.zeros(8), 0.0, _ackley, _ackley_gradient),
    "rosenbrock": FunctionSpec("rosenbrock", 8, -5.0*np.ones(8), 10.0*np.ones(8),
                                np.ones(8), 0.0, _rosenbrock, _rosenbrock_gradient),
    "levy": FunctionSpec("levy", 8, -10.0*np.ones(8), 10.0*np.ones(8),
                          np.ones(8), 0.0, _levy, _levy_gradient),
}


def constrained_fixtures() -> dict[str, dict[str, Callable[[Array], Array]]]:
    """Return constrained fixtures with constraints separate from values."""
    def townsend_value(x: Array) -> Array:
        return -(np.cos((x[:, 0]-0.1)*x[:, 1])**2 + x[:, 0]*np.sin(x[:, 1]))

    def townsend_constraint(x: Array) -> Array:
        theta = np.arctan2(x[:, 0], x[:, 1])
        radius = (2.0*np.cos(theta) - 0.5*np.cos(2.0*theta) -
                  0.25*np.cos(3.0*theta) - 0.125*np.cos(4.0*theta))
        return x[:, 0]**2 + x[:, 1]**2 - radius**2 - (2.0*np.sin(theta))**2

    def gardner_value(x: Array) -> Array:
        return np.sin(x[:, 0]) + x[:, 1]

    def gardner_constraint(x: Array) -> Array:
        return np.sin(x[:, 0])*np.sin(x[:, 1]) + 0.95

    return {
        "townsend": {"lower": np.array([-2.25, -2.5]), "upper": np.array([2.25, 1.75]),
                     "value": townsend_value, "constraint": townsend_constraint},
        "gardner": {"lower": np.zeros(2), "upper": np.full(2, 6.0),
                    "value": gardner_value, "constraint": gardner_constraint},
    }


def zdt(point: Array, kind: int = 1) -> Array:
    """Evaluate ZDT1 or ZDT2, minimizing both objectives."""
    x = _points(point)
    if x.shape[1] < 2 or np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("ZDT inputs must lie in [0, 1]^d with d >= 2")
    g = 1.0 + 9.0*np.mean(x[:, 1:], axis=1)
    ratio = x[:, 0]/g
    f2 = g*(1.0-np.sqrt(ratio) if kind == 1 else 1.0-ratio**2)
    return np.column_stack((x[:, 0], f2))


def zdt_front(f1: Array, kind: int = 1) -> Array:
    first = np.asarray(f1, dtype=np.float64)
    if np.any((first < 0.0) | (first > 1.0)):
        raise ValueError("front parameter must lie in [0, 1]")
    return np.column_stack((first, 1.0-np.sqrt(first) if kind == 1 else 1.0-first**2))


def noisy_observations(values: Array, standard_deviation: float, seed: int) -> Array:
    if standard_deviation < 0.0:
        raise ValueError("noise standard deviation must be nonnegative")
    generator = np.random.default_rng(seed)
    return np.asarray(values, dtype=np.float64) + generator.normal(
        0.0, standard_deviation, size=np.asarray(values).shape)


def rbf_kernel(a: Array, b: Array, lengthscale: float, signal_variance: float) -> Array:
    a, b = _points(a), _points(b)
    if lengthscale <= 0.0 or signal_variance <= 0.0:
        raise ValueError("kernel scales must be positive")
    squared = np.sum((a[:, None, :]-b[None, :, :])**2, axis=2)
    return signal_variance*np.exp(-0.5*squared/lengthscale**2)


def gp_posterior(train_x: Array, train_y: Array, query: Array, *,
                 lengthscale: float = 0.7, signal_variance: float = 1.3,
                 noise_variance: float = 0.05) -> dict[str, Array]:
    """Exact scalar GP posterior and input gradient from direct algebra."""
    x, q = _points(train_x), _points(query)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    if len(y) != len(x) or x.shape[1] != q.shape[1]:
        raise ValueError("GP shapes disagree")
    gram = rbf_kernel(x, x, lengthscale, signal_variance) + noise_variance*np.eye(len(x))
    factor = np.linalg.cholesky(gram)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
    cross = rbf_kernel(q, x, lengthscale, signal_variance)
    mean = cross @ alpha
    solved = np.linalg.solve(factor, cross.T)
    variance = np.maximum(signal_variance - np.sum(solved**2, axis=0), 0.0)
    delta = q[:, None, :] - x[None, :, :]
    d_cross = -cross[:, :, None]*delta/lengthscale**2
    mean_gradient = np.einsum("qnd,n->qd", d_cross, alpha)
    d_rhs = d_cross.transpose(1, 2, 0).reshape(len(x), -1)
    d_solved = np.linalg.solve(factor, d_rhs).reshape(len(x), x.shape[1], len(q))
    d_solved = d_solved.transpose(2, 0, 1)
    variance_gradient = -2.0*np.einsum("nq,qnd->qd", solved, d_solved)
    return {"mean": mean, "variance": variance, "sd": np.sqrt(variance),
            "mean_gradient": mean_gradient, "variance_gradient": variance_gradient}


def gp_posterior_derivative_observations(
        train_x: Array, train_y: Array, train_gradient: Array, query: Array, *,
        lengthscale: float = 0.7, signal_variance: float = 1.3,
        noise_variance: float = 0.05, gradient_noise: float = 0.05) -> dict[str, Array]:
    """Exact GP conditioning on values and first derivatives.

    This is intentionally separate from :func:`gp_posterior`, so a comparison
    can tell whether a derivative-observation implementation has merely called
    the value-only path.
    """
    x, q = _points(train_x), _points(query)
    y, dy = np.asarray(train_y, float).reshape(-1), np.asarray(train_gradient, float)
    n, d = x.shape
    if dy.shape != (n, d):
        raise ValueError("derivative observations must have shape (n, d)")
    k = rbf_kernel(x, x, lengthscale, signal_variance)
    delta = x[:, None, :] - x[None, :, :]
    k_f_d = k[:, :, None] * (x[:, None, :]-x[None, :, :])/lengthscale**2
    k_d_d = np.empty((n, d, n, d), dtype=float)
    for i in range(n):
        for j in range(n):
            difference = x[i]-x[j]
            k_d_d[i, :, j, :] = k[i, j]*(np.eye(d)/lengthscale**2 -
                np.outer(difference, difference)/lengthscale**4)
    gram = np.block([[k, k_f_d.reshape(n, n*d)],
                     [k_f_d.reshape(n, n*d).T, k_d_d.reshape(n*d, n*d)]])
    gram[:n, :n] += noise_variance*np.eye(n)
    gram[n:, n:] += gradient_noise*np.eye(n*d)
    observations = np.concatenate((y, dy.reshape(-1)))
    alpha = np.linalg.solve(gram, observations)
    k_qf = rbf_kernel(q, x, lengthscale, signal_variance)
    difference = q[:, None, :] - x[None, :, :]
    k_qd = k_qf[:, :, None]*(q[:, None, :]-x[None, :, :])/lengthscale**2
    cross = np.concatenate((k_qf, k_qd.reshape(len(q), n*d)), axis=1)
    mean = cross @ alpha
    solved = np.linalg.solve(gram, cross.T)
    variance = np.maximum(signal_variance-np.sum(cross*solved.T, axis=1), 0.0)
    return {"mean": mean, "variance": variance, "sd": np.sqrt(variance)}


def normal_pdf(z: Array | float) -> Array:
    values = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5*values**2)/math.sqrt(2.0*math.pi)


def normal_cdf(z: Array | float) -> Array:
    values = np.asarray(z, dtype=np.float64)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5*(1.0+erf(values/math.sqrt(2.0)))


def expected_improvement(mean: Array, sd: Array, best: float, xi: float = 0.0) -> Array:
    mean, sd = np.asarray(mean, float), np.maximum(np.asarray(sd, float), 0.0)
    gap = best-xi-mean
    result = np.maximum(gap, 0.0)
    positive = sd > 0.0
    z = np.zeros_like(gap)
    z[positive] = gap[positive]/sd[positive]
    result[positive] = gap[positive]*normal_cdf(z[positive]) + sd[positive]*normal_pdf(z[positive])
    return result


def expected_improvement_derivatives(mean: float, sd: float, best: float,
                                     xi: float = 0.0) -> dict[str, float]:
    """Value and parameter products from the defining normal integral."""
    if sd <= 0.0:
        gap = best-xi-mean
        active = 1.0 if gap > 0.0 else 0.0
        return {"value": max(gap, 0.0), "mean": -active, "sd": 0.0,
                "best": active, "xi": -active}
    z = (best-xi-mean)/sd
    cdf, pdf = float(normal_cdf(z)), float(normal_pdf(z))
    return {"value": float((best-xi-mean)*cdf+sd*pdf), "mean": -cdf,
            "sd": pdf, "best": cdf, "xi": -cdf}


def probability_of_improvement(mean: Array, sd: Array, best: float, xi: float = 0.0) -> Array:
    mean, sd = np.asarray(mean, float), np.asarray(sd, float)
    result = (best-xi-mean > 0.0).astype(float)
    positive = sd > 0.0
    result[positive] = normal_cdf((best-xi-mean[positive])/sd[positive])
    return result


def lower_confidence_bound(mean: Array, sd: Array, beta: float = 2.0) -> Array:
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    return np.asarray(mean, float)-beta*np.asarray(sd, float)


def acquisition_values(mean: Array, sd: Array, best: float, *, xi: float = 0.0,
                       beta: float = 2.0) -> dict[str, Array]:
    """All scalar acquisition references in the minimisation convention."""
    return {"ei": expected_improvement(mean, sd, best, xi),
            "pi": probability_of_improvement(mean, sd, best, xi),
            "ucb": lower_confidence_bound(mean, sd, beta)}


def q_expected_improvement(samples: Array, best: float, xi: float = 0.0) -> float:
    values = np.asarray(samples, float)
    if values.ndim != 2:
        raise ValueError("qEI samples must have shape (n_samples, q)")
    return float(np.mean(np.maximum(best-xi-np.min(values, axis=1), 0.0)))


def q_noisy_expected_improvement(candidate_samples: Array, incumbent_samples: Array) -> float:
    candidates, incumbents = np.asarray(candidate_samples, float), np.asarray(incumbent_samples, float)
    if candidates.ndim != 2 or incumbents.ndim != 2 or candidates.shape[0] != incumbents.shape[0]:
        raise ValueError("qNEI sample counts must match")
    return float(np.mean(np.maximum(np.min(incumbents, axis=1)-np.min(candidates, axis=1), 0.0)))


def knowledge_gradient(current_decision_values: Array, fantasies: Array) -> float:
    """Monte-Carlo-free scalar reference for a supplied fantasy table.

    Rows are posterior fantasies and columns are the possible final decisions.
    The value is the current best decision minus the average best decision
    after observing the candidate.  Supplying the fantasies makes the oracle
    independent of any posterior sampler and lets a test pin common draws.
    """
    current = np.asarray(current_decision_values, float).reshape(-1)
    table = np.asarray(fantasies, float)
    if table.ndim != 2:
        raise ValueError("fantasies must have shape (n_draws, n_decisions)")
    return float(np.min(current)-np.mean(np.min(table, axis=1)))


def thompson_indices(samples: Array) -> Array:
    values = np.asarray(samples, float)
    if values.ndim != 2:
        raise ValueError("Thompson samples must have shape (n_draws, n_points)")
    return np.argmin(values, axis=1)


def pending_fantasy(means: Array, incumbent: float, policy: str = "posterior-mean") -> Array:
    """Return the explicit value used to fantasize pending evaluations."""
    values = np.asarray(means, dtype=np.float64)
    if policy == "posterior-mean":
        return values.copy()
    if policy == "incumbent":
        return np.full_like(values, incumbent)
    if policy == "worst-observed":
        return np.full_like(values, np.max(values))
    raise ValueError("unknown pending-point fantasy policy")


def constrained_acquisition(base: Array, feasibility_probability: Array,
                            cost: Array | None = None) -> Array:
    result = np.asarray(base, float)*np.asarray(feasibility_probability, float)
    if np.any(np.asarray(feasibility_probability) < 0.0) or np.any(np.asarray(feasibility_probability) > 1.0):
        raise ValueError("feasibility probabilities must lie in [0, 1]")
    if cost is not None:
        costs = np.asarray(cost, float)
        if np.any(costs <= 0.0):
            raise ValueError("costs must be positive")
        result = result/costs
    return result


def effective_sample_size(weights: Array) -> float:
    values = np.asarray(weights, float).reshape(-1)
    if np.any(values < 0.0) or np.sum(values) <= 0.0:
        raise ValueError("weights must be nonnegative and not all zero")
    normalized = values/np.sum(values)
    return float(1.0/np.sum(normalized**2))


def check_gradient(value: Value, gradient: Gradient, points: Array, *, step: float = 1.0e-5) -> float:
    """Return the largest Richardson-extrapolated central-difference error."""
    x = _points(points)
    analytic = np.asarray(gradient(x), float)
    numerical = np.empty_like(analytic)
    for j in range(x.shape[1]):
        plus = x.copy(); minus = x.copy()
        plus[:, j] += step; minus[:, j] -= step
        plus_fine = x.copy(); minus_fine = x.copy()
        plus_fine[:, j] += 0.5*step; minus_fine[:, j] -= 0.5*step
        coarse = (np.asarray(value(plus))-np.asarray(value(minus)))/(2.0*step)
        fine = (np.asarray(value(plus_fine))-np.asarray(value(minus_fine)))/step
        numerical[:, j] = (4.0*fine-coarse)/3.0
    return float(np.max(np.abs(analytic-numerical)))
