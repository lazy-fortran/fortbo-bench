#!/usr/bin/env python3
"""Emit BoTorch's regret curve on Branin, for FortBO to be compared against.

The roadmap asks for regret and sample efficiency to be reported **separately
from wall time**, and this script exists because those are different claims that
are routinely conflated. Sample efficiency asks how many objective evaluations a
policy needs; wall time asks how long the machinery around them took. Bayesian
optimization is for expensive objectives, so on any problem worth using it on
the first number dominates and the second is nearly irrelevant -- and a table
that reports only the second is measuring Python's overhead against Fortran's,
which is a fact about the two languages rather than about the two optimizers.

What is matched, because a regret comparison is worthless otherwise:

  * the same objective, Branin, on the same box;
  * the same initial design -- **the exact same points**, listed here and read
    by both sides, not merely the same seed. Two frameworks given the same seed
    draw different numbers, and the initial design dominates the early regret
    curve, so a seed-matched comparison mostly reports whose random draw was
    luckier;
  * the same budget, the same acquisition (analytic EI), the same number of
    restarts for the acquisition optimizer;
  * the same stopping criterion, namely the budget and nothing else. A policy
    that stops early on a convergence test is solving a different problem.

Hyperparameters *are* refitted each iteration here, unlike the posterior
comparison, because refitting is what a real loop does and freezing them would
measure a model nobody would deploy. That does mean the two sides run different
optimizers over the marginal likelihood, which is a genuine difference between
the systems rather than a flaw in the comparison -- and it is stated in the
output so nobody reads the gap as a bug.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "fixtures" / "regret_branin.json"
FORTRAN_OUTPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "fortbo" / "test" / "fixtures" / "regret_branin.txt"
)

# Branin, on its standard box. The three global minima all attain this value.
LOWER = np.array([-5.0, 0.0])
UPPER = np.array([10.0, 15.0])
BRANIN_OPTIMUM = 0.397887357729739

BUDGET = 30
RESTARTS = 8


def branin(x: np.ndarray) -> np.ndarray:
    a, b, c = 1.0, 5.1 / (4.0 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8.0 * np.pi)
    x1, x2 = x[..., 0], x[..., 1]
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def initial_design() -> np.ndarray:
    """Ten points, stated rather than drawn.

    Written out so both sides start from the identical design. A shared seed
    would not achieve this: the two frameworks' generators differ, and the
    initial design dominates early regret.
    """
    unit = np.array([
        [0.10, 0.85], [0.45, 0.20], [0.75, 0.60], [0.25, 0.40], [0.90, 0.10],
        [0.60, 0.95], [0.35, 0.70], [0.05, 0.30], [0.80, 0.45], [0.55, 0.05],
    ])
    return LOWER + unit * (UPPER - LOWER)


def botorch_run() -> dict:
    try:
        import torch
        from botorch.acquisition.analytic import LogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as error:
        return {"unavailable": str(error)}

    torch.manual_seed(0)
    dtype = torch.float64
    bounds = torch.tensor(np.stack([LOWER, UPPER]), dtype=dtype)

    x = torch.tensor(initial_design(), dtype=dtype)
    # BoTorch maximizes; the objective is negated on the way in.
    y = torch.tensor(-branin(initial_design()), dtype=dtype).unsqueeze(-1)

    # Regret after the initial design, before any modelling has happened. The
    # curve starts here so the two sides are visibly at the same place.
    regret = [float(-y.max().item() - BRANIN_OPTIMUM)]
    started = time.perf_counter()

    for _ in range(BUDGET):
        model = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # LogEI, not the legacy EI: the logEI paper (arXiv:2310.20708) shows the
        # legacy form's gradients vanish where improvement is unlikely, which is
        # exactly where the acquisition optimizer spends its time.
        acquisition = LogExpectedImprovement(model, best_f=y.max())
        candidate, _ = optimize_acqf(
            acquisition, bounds=bounds, q=1,
            num_restarts=RESTARTS, raw_samples=128,
        )

        value = float(branin(candidate.numpy())[0])
        x = torch.cat([x, candidate], dim=0)
        y = torch.cat([y, torch.tensor([[-value]], dtype=dtype)], dim=0)
        regret.append(float(-y.max().item() - BRANIN_OPTIMUM))

    return {
        "regret": regret,
        # Reported, but kept in its own field and never mixed into the regret
        # curve. This includes hyperparameter refitting and acquisition
        # optimization in Python; it is not a like-for-like against a Fortran
        # binary and is recorded for context only.
        "wall_seconds": time.perf_counter() - started,
        "evaluations": len(regret) - 1,
    }


def main() -> int:
    design = initial_design()
    result = botorch_run()

    payload = {
        "objective": "branin",
        "optimum": BRANIN_OPTIMUM,
        "lower": LOWER.tolist(),
        "upper": UPPER.tolist(),
        "budget": BUDGET,
        "restarts": RESTARTS,
        "note": (
            "Regret and wall time are separate fields on purpose. Sample "
            "efficiency is the claim Bayesian optimization makes; wall time "
            "here largely measures Python against Fortran."
        ),
        "initial_design": design.tolist(),
        "initial_values": branin(design).tolist(),
        "botorch": result,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}")

    if "unavailable" in result:
        print("botorch unavailable:", result["unavailable"])
        return 0

    lines = [
        "# Generated by fortbo-bench/scripts/emit_regret.py. Do not edit.",
        "# BoTorch on Branin: shared initial design, matched budget and",
        "# restarts. Regret is kept apart from wall time deliberately.",
        "# budget restarts optimum",
        f"{BUDGET} {RESTARTS} {BRANIN_OPTIMUM!r}",
        "# n_initial",
        f"{len(design)}",
        "# initial_design (one point per line, x1 x2)",
    ]
    lines.extend(f"{float(p[0])!r} {float(p[1])!r}" for p in design)
    lines.append("# botorch_regret (one per line, initial design first)")
    lines.extend(repr(float(v)) for v in result["regret"])
    lines.append("# botorch_wall_seconds")
    lines.append(repr(float(result["wall_seconds"])))
    FORTRAN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FORTRAN_OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {FORTRAN_OUTPUT}")

    print(f"  botorch final regret = {result['regret'][-1]:.6e} "
          f"after {result['evaluations']} evaluations")
    print(f"  botorch wall time    = {result['wall_seconds']:.2f} s "
          f"(reported separately; not a like-for-like against Fortran)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
