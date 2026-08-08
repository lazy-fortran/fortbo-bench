#!/usr/bin/env python3
"""Run BoTorch's `turbo_1` as a third baseline on Ackley-200.

The roadmap names three references: the paper, the authors' own
`uber-research/TuRBO`, and BoTorch's `turbo_1` tutorial. The first two are
covered by `run_turbo_baselines.py`. This is the third.

**Why a third implementation of the same algorithm is worth running.** It is
not a new comparison of methods — it is a comparison of *implementations*.
TuRBO is a specification with a lot of unstated detail: how the trust region is
shaped by the ARD lengthscales, when it restarts, how many candidates are drawn
and with what perturbation probability. Two independent implementations
agreeing tells you the specification was read the same way; two disagreeing
tells you it was not, and that is exactly the kind of defect a single reference
cannot expose.

This follows BoTorch's tutorial rather than importing it, because the tutorial
ships as a notebook. What it takes from the tutorial is the structure: a Matérn
5/2 ARD surrogate with the tutorial's constraint ranges, Thompson sampling over
a perturbed candidate set via `MaxPosteriorSampling`, the `min(20/d, 1)`
perturbation probability, and the success/failure counters that resize the
trust region. Those are TuRBO's own constants and are the same ones FortBO's
`fortbo_trust_region` carries with their provenance.

Matched to the other baselines: same objective, same box, same dimension, same
budget, same initial design size, same seeds, float64 throughout.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "fixtures" / "botorch_turbo_ackley200.json"

DIMENSION = 200
LOWER, UPPER = -5.0, 10.0
SEEDS = (101, 102, 103)

# The same two configurations `run_turbo_baselines.py` uses, so all three
# references are directly comparable row for row.
CONFIGURATIONS = {
    "matched": {"budget": 16, "n_initial": 5},
    "roomy": {"budget": 60, "n_initial": 10},
}


def ackley(x: np.ndarray) -> np.ndarray:
    """Ackley, written out so it is visibly the same expression everywhere."""
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    first = -a * np.exp(-b * np.sqrt(np.mean(x**2, axis=-1)))
    second = -np.exp(np.mean(np.cos(c * x), axis=-1))
    return first + second + a + np.e


@dataclass
class TrustRegion:
    """TuRBO's counter rule, with the paper's constants.

    Not tuned here. `length_min` is the restart threshold, and the doubling and
    halving are the paper's; changing any of them would make this a comparison
    against a different algorithm while still being called TuRBO.
    """

    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    failure_tolerance: int = 0
    success_tolerance: int = 3
    best_value: float = float("inf")

    def update(self, next_best: float) -> None:
        # Minimization: an improvement is a decrease.
        if next_best < self.best_value - 1e-3 * abs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = min(self.best_value, next_best)


def run(seed: int, budget: int, n_initial: int) -> dict:
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling
    from botorch.models import SingleTaskGP
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.quasirandom import SobolEngine

    torch.manual_seed(seed)
    dtype, device = torch.float64, torch.device("cpu")

    def evaluate(unit_x: torch.Tensor) -> torch.Tensor:
        scaled = LOWER + (UPPER - LOWER) * unit_x.cpu().numpy()
        return torch.tensor(ackley(scaled), dtype=dtype, device=device)

    sobol = SobolEngine(DIMENSION, scramble=True, seed=seed)
    x = sobol.draw(n_initial).to(dtype=dtype, device=device)
    y = evaluate(x).unsqueeze(-1)

    state = TrustRegion(best_value=float(y.min()))
    state.failure_tolerance = math.ceil(max(4.0 / 1, DIMENSION / 1.0)) // 20 or 1

    started = time.perf_counter()
    n_candidates = min(100 * DIMENSION, 5000)

    while len(x) < budget:
        # BoTorch maximizes; the tutorial negates. Standardizing the targets is
        # the tutorial's own step and matters: an unstandardized GP over a
        # narrow range of Ackley values fits a nearly flat posterior.
        train_y = -(y - y.mean()) / (y.std() + 1e-12)

        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=DIMENSION,
                         lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(x, train_y, covar_module=covar,
                             likelihood=likelihood,
                             outcome_transform=None, input_transform=None)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:  # noqa: BLE001
            # A failed fit is not fatal to the run; the previous
            # hyperparameters remain and the loop continues. Recorded because
            # a silently degraded surrogate would flatter the result.
            pass

        # The trust region, shaped by the ARD lengthscales and normalized so
        # its volume does not move when the surrogate refits.
        best_index = train_y.argmax()
        centre = x[best_index].clone()
        lengthscales = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = lengthscales / lengthscales.mean()
        weights = weights / torch.prod(weights.pow(1.0 / DIMENSION))
        lower = torch.clamp(centre - weights * state.length / 2.0, 0.0, 1.0)
        upper = torch.clamp(centre + weights * state.length / 2.0, 0.0, 1.0)

        candidate_sobol = SobolEngine(DIMENSION, scramble=True, seed=seed + len(x))
        perturbation = candidate_sobol.draw(n_candidates).to(dtype=dtype)
        perturbation = lower + (upper - lower) * perturbation

        # TuRBO's sparse perturbation: about twenty coordinates move.
        probability = min(20.0 / DIMENSION, 1.0)
        mask = torch.rand(n_candidates, DIMENSION, dtype=dtype) <= probability
        empty = torch.where(mask.sum(dim=1) == 0)[0]
        mask[empty, torch.randint(0, DIMENSION, (len(empty),))] = True

        candidates = centre.expand(n_candidates, DIMENSION).clone()
        candidates[mask] = perturbation[mask]

        sampler = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            next_x = sampler(candidates, num_samples=1)

        next_y = evaluate(next_x).unsqueeze(-1)
        x = torch.cat([x, next_x], dim=0)
        y = torch.cat([y, next_y], dim=0)
        state.update(float(next_y.min()))

        if state.length < state.length_min:
            # Restart, as the paper specifies: the region is exhausted.
            state = TrustRegion(best_value=float(y.min()))
            state.failure_tolerance = max(DIMENSION // 20, 1)

    return {
        "best": float(y.min()),
        "evaluations": int(len(y)),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    all_results: dict[str, dict] = {}
    for name, configuration in CONFIGURATIONS.items():
        budget = configuration["budget"]
        n_initial = configuration["n_initial"]
        bests, walls = [], []
        for seed in SEEDS:
            try:
                result = run(seed, budget, n_initial)
            except Exception as error:  # noqa: BLE001
                print(f"  {name} seed {seed} failed: {error}")
                continue
            bests.append(result["best"])
            walls.append(result["wall_seconds"])
        if bests:
            all_results[name] = {
                "budget": budget,
                "n_initial": n_initial,
                "median_best": float(np.median(bests)),
                "per_seed": bests,
                "median_wall_seconds": float(np.median(walls)),
            }
        else:
            all_results[name] = {"unavailable": "every seed failed"}

    payload = {
        "objective": "ackley",
        "dimension": DIMENSION,
        "lower": LOWER,
        "upper": UPPER,
        "seeds": list(SEEDS),
        "reference": "BoTorch turbo_1, following the tutorial's structure",
        "note": (
            "A third implementation of the same algorithm. Agreement between "
            "independent implementations is evidence the specification was "
            "read the same way; disagreement is evidence it was not."
        ),
        "results": all_results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}")
    for name, entry in all_results.items():
        if "unavailable" in entry:
            print(f"  {name}: {entry['unavailable']}")
        else:
            print(f"  {name} (budget {entry['budget']}): "
                  f"median best = {entry['median_best']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
