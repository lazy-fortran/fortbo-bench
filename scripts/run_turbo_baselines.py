#!/usr/bin/env python3
"""Run the pinned `uber-research/TuRBO` reference, for FortBO to be compared to.

The roadmap asks for FortBO's TuRBO to be checked against a *pinned* reference
implementation rather than against the paper's printed curves. Printed curves
cannot be matched: they average many seeds at budgets in the thousands, and
their two hardest problems (the rover and the robot pushing task) depend on
code and physics that are not reproduced here. What can be matched exactly is a
closed-form objective evaluated by both implementations from the same box.

**So this runs on Ackley-200 and nothing else.** Ackley is stated in closed
form, so the reference and FortBO are provably optimizing the identical
function — no obstacle map, no rigid-body solver, no substitution to argue
about. The rover and pushing problems are deliberately excluded: FortBO's
versions of them are structurally faithful but numerically its own, so a
comparison against the reference there would measure the fixtures rather than
the optimizers, and would look like a TuRBO comparison while being nothing of
the kind.

The reference is used **unmodified**, at the commit recorded by
`fetch_provenance.py`. Its trust-region constants, its restart bookkeeping and
its `y_cand = inf` rule for never reselecting a candidate within a batch are
all the authors'. Nothing here is ported; the repository is read-only
provenance and its licence is non-commercial.

What is matched: the objective, the box, the dimension, the budget, the initial
design size, the batch size, and float64 throughout. What is *not* matched, and
cannot be: the surrogate's hyperparameter fitting. The reference trains a
GPyTorch model with Adam for a fixed number of steps; FortBO uses a fixed
lengthscale. That difference is real and is reported rather than hidden — it is
the main reason the two are expected to differ, and pretending otherwise would
misattribute the gap to the trust-region logic, which is the part actually
under comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "provenance" / "codes" / "turbo-reference"
OUTPUT = ROOT / "fixtures" / "turbo_baseline_ackley200.json"

DIMENSION = 200
LOWER, UPPER = -5.0, 10.0
SEEDS = (101, 102, 103)

# Two configurations, for two different questions.
#
# `matched` uses exactly the budget and initial-design size FortBO's own
# `test_turbo_ordering_ackley_slow` can afford, so the two implementations are
# optimizing the same function from the same starting budget and the numbers
# are directly comparable. Only the single-region arm is run there: with five
# regions the budget would be spent entirely on initial designs, which is a
# degenerate configuration rather than a comparison.
#
# `roomy` is large enough for the several-versus-one question to mean
# something, and is run only in the reference because FortBO cannot reach that
# budget at 200 dimensions inside its test-suite cap.
CONFIGURATIONS = {
    "matched": {"budget": 16, "n_initial": 5, "regions": (1,)},
    "roomy": {"budget": 60, "n_initial": 10, "regions": (1, 5)},
}


class Ackley:
    """Ackley in its standard form, on the box FortBO uses.

    Written out rather than imported so the objective the reference sees is
    visibly the same expression FortBO evaluates, constant for constant.
    """

    def __init__(self, dimension: int):
        self.dim = dimension
        self.lb = LOWER * np.ones(dimension)
        self.ub = UPPER * np.ones(dimension)

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        first = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
        second = -np.exp(np.mean(np.cos(c * x)))
        return float(first + second + a + np.e)


def run_reference(n_trust_regions: int, seed: int, budget: int,
                  n_initial: int) -> dict:
    sys.path.insert(0, str(REFERENCE))
    import torch
    from turbo import Turbo1, TurboM

    np.random.seed(seed)
    torch.manual_seed(seed)

    objective = Ackley(DIMENSION)
    started = time.perf_counter()
    if n_trust_regions == 1:
        optimizer = Turbo1(
            f=objective, lb=objective.lb, ub=objective.ub,
            n_init=n_initial, max_evals=budget, batch_size=1,
            verbose=False, use_ard=True, max_cholesky_size=2000,
            n_training_steps=50, min_cuda=1024, device="cpu", dtype="float64",
        )
    else:
        optimizer = TurboM(
            f=objective, lb=objective.lb, ub=objective.ub,
            n_init=n_initial, max_evals=budget, n_trust_regions=n_trust_regions,
            batch_size=1, verbose=False, use_ard=True, max_cholesky_size=2000,
            n_training_steps=50, min_cuda=1024, device="cpu", dtype="float64",
        )
    optimizer.optimize()
    elapsed = time.perf_counter() - started

    values = np.asarray(optimizer.fX).ravel()
    return {
        "best": float(values.min()),
        "evaluations": int(values.size),
        "wall_seconds": elapsed,
    }


def run_random(seed: int, budget: int) -> dict:
    """The same undirected baseline FortBO's harness uses."""
    generator = np.random.default_rng(seed)
    objective = Ackley(DIMENSION)
    best = np.inf
    for _ in range(budget):
        x = generator.uniform(LOWER, UPPER, DIMENSION)
        best = min(best, objective(x))
    return {"best": float(best)}


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values)))


def main() -> int:
    if not REFERENCE.exists():
        print(f"reference missing at {REFERENCE}; "
              "run scripts/fetch_provenance.py first")
        return 1

    all_results: dict[str, dict] = {}
    for name, configuration in CONFIGURATIONS.items():
        budget = configuration["budget"]
        n_initial = configuration["n_initial"]
        results: dict[str, dict] = {}
        for regions in configuration["regions"]:
            label = "turbo-1" if regions == 1 else f"turbo-{regions}"
            bests, walls = [], []
            for seed in SEEDS:
                try:
                    run = run_reference(regions, seed, budget, n_initial)
                except Exception as error:  # noqa: BLE001
                    # Recorded, not swallowed. A baseline that failed to run is
                    # a different fact from one that ran badly, and a table
                    # that cannot tell them apart is worse than no table.
                    print(f"  {name}/{label} seed {seed} failed: {error}")
                    continue
                bests.append(run["best"])
                walls.append(run["wall_seconds"])
            if bests:
                results[label] = {
                    "median_best": median(bests),
                    "per_seed": bests,
                    "median_wall_seconds": median(walls),
                }
            else:
                results[label] = {"unavailable": "every seed failed"}

        randoms = [run_random(seed, budget)["best"] for seed in SEEDS]
        results["random"] = {"median_best": median(randoms), "per_seed": randoms}
        results["budget"] = budget
        results["n_initial"] = n_initial
        all_results[name] = results

    payload = {
        "objective": "ackley",
        "dimension": DIMENSION,
        "lower": LOWER,
        "upper": UPPER,
        "seeds": list(SEEDS),
        "reference": "uber-research/TuRBO, unmodified, at the pinned commit",
        "caveat": (
            "The reference fits GP hyperparameters with Adam each step; FortBO "
            "uses a fixed lengthscale. That difference is real and is the main "
            "expected source of any gap, so it must not be attributed to the "
            "trust-region logic under comparison."
        ),
        "results": all_results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}")
    for name, results in all_results.items():
        print(f"  {name} (budget {results['budget']}):")
        for label, entry in results.items():
            if not isinstance(entry, dict):
                continue
            if "unavailable" in entry:
                print(f"    {label}: {entry['unavailable']}")
            else:
                print(f"    {label}: median best = {entry['median_best']:.4f}")

    # A copy FortBO's own suite can read without a JSON parser.
    matched = all_results["matched"]
    lines = [
        "# Generated by fortbo-bench/scripts/run_turbo_baselines.py.",
        "# uber-research/TuRBO, unmodified, at the pinned commit, on Ackley-200",
        "# with the budget FortBO's test suite can afford.",
        "# budget n_initial dimension",
        f"{matched['budget']} {matched['n_initial']} {DIMENSION}",
        "# reference_turbo1_median reference_random_median",
        f"{matched['turbo-1']['median_best']!r} "
        f"{matched['random']['median_best']!r}",
    ]
    fortran_output = (
        ROOT.parent / "fortbo" / "test" / "fixtures" / "turbo_baseline.txt"
    )
    fortran_output.parent.mkdir(parents=True, exist_ok=True)
    fortran_output.write_text("\n".join(lines) + "\n")
    print(f"wrote {fortran_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
