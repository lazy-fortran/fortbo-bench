#!/usr/bin/env python3
"""Emit the complete FortBO feature benchmark matrix.

The matrix is deliberately broader than the rows for which a speedup can be
claimed.  A feature with no like-for-like external implementation still gets
an independent oracle and a contract/refusal lane; it is recorded as
``reference-only`` rather than silently disappearing from the release evidence.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORTBO = ROOT.parent / "fortbo"


def requirement(label: str, module: str, symbol: str) -> dict[str, str]:
    try:
        imported = importlib.import_module(module)
        available = hasattr(imported, symbol)
        reason = "" if available else f"{symbol} is not exported"
    except (ImportError, ModuleNotFoundError) as error:
        available = False
        reason = str(error)
    return {"label": label, "module": module, "symbol": symbol,
            "available": available, "reason": reason}


def row(module: str, feature: str, comparability: str, oracle: str,
        command: str, external: list[tuple[str, str, str]] = (),
        target: str = "correctness and reproducibility") -> dict:
    requirements = [requirement(*item) for item in external]
    return {
        "module": module,
        "feature": feature,
        "source": f"../fortbo/src/{module}.f90",
        "comparability": comparability,
        "external": requirements,
        "oracle": oracle,
        "benchmark_command": command,
        "performance_target": target,
        "performance_status": (
            "speed-gated" if comparability in {"direct-external", "external-policy"}
            else "typed-refusal" if comparability == "oracle-plus-api"
            else "oracle-or-contract"),
        "status": "ready" if not requirements or all(
            item["available"] for item in requirements) else "external-unavailable",
    }


CATALOG = [
    row("fortbo", "public package surface", "contract-only",
        "independent integration and replay suite", "fo test --all"),
    row("fortbo_acquisition", "EI, logEI, PI, lower confidence bound",
        "direct-external", "NumPy definitions plus BoTorch analytic values",
        "python scripts/run_acquisition_speed.py",
        [("EI", "botorch.acquisition.analytic", "ExpectedImprovement"),
         ("logEI", "botorch.acquisition.analytic", "LogExpectedImprovement"),
         ("PI", "botorch.acquisition.analytic", "ProbabilityOfImprovement"),
         ("UCB", "botorch.acquisition.analytic", "UpperConfidenceBound")],
        "FortBO wall time <= BoTorch after value gate"),
    row("fortbo_active", "variance, straddle, and level-set active learning",
        "reference-only", "direct NumPy variance and classification definitions",
        "python scripts/run_benchmark.py"),
    row("fortbo_batch", "qEI, qNEI, qUCB",
        "oracle-plus-api", "frozen joint-draw order-statistic oracle plus BoTorch API availability",
        "python scripts/run_oracle_lane.py batch",
        [("qEI", "botorch.acquisition.monte_carlo", "qExpectedImprovement"),
         ("qNEI", "botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
         ("qUCB", "botorch.acquisition.monte_carlo", "qUpperConfidenceBound")],
        "FortBO wall time <= BoTorch after Monte Carlo tolerance gate"),
    row("fortbo_benchmarks", "analytic objective functions and gradients",
        "reference-only", "known optima, dense/local scans, and independent Richardson gradients",
        "python scripts/run_benchmark.py"),
    row("fortbo_constrained", "feasibility-weighted and cost-aware acquisition",
        "direct-external", "independent constrained-acquisition definition and BoTorch constraint objective",
        "python scripts/run_benchmark.py",
        [("constraint objective", "botorch.acquisition.objective", "ConstrainedMCObjective")]),
    row("fortbo_device", "resident CPU/OpenACC candidate scoring and reduction",
        "device-contract", "bit-identical host/device result, repeated-launch determinism, typed refusal",
        "fo test test_device"),
    row("fortbo_dturbo", "derivative-enabled TuRBO modes 0, 1, and 2",
        "external-policy", "known-function regret, derivative oracle, ratio/trace invariants",
        "fo test --all; python scripts/run_speed_comparison.py",
        [("BoTorch trust-region baseline", "botorch.generation.sampling", "ConstrainedMaxPosteriorSampling")]),
    row("fortbo_entropy", "max-value entropy search",
        "oracle-plus-api", "independent Gaussian entropy oracle plus BoTorch qMES API availability",
        "python scripts/run_oracle_lane.py entropy",
        [("qMES", "botorch.acquisition.max_value_entropy_search", "qMaxValueEntropy")]),
    row("fortbo_feasible", "feasibility filtering and incumbent bookkeeping",
        "reference-only", "brute-force feasible scan and failure/unknown separation",
        "fo test test_feasible; python scripts/run_benchmark.py"),
    row("fortbo_fixtures", "constrained and multi-objective benchmark fixtures",
        "reference-only", "known feasible optima and analytic ZDT fronts",
        "fo test test_fixtures; python scripts/run_benchmark.py"),
    row("fortbo_fortml", "exact and derivative-observation GP adapters",
        "direct-external", "independent augmented covariance GP plus GPyTorch/BoTorch posterior",
        "fo test test_fortml_adapter test_cross_framework; python scripts/emit_reference.py",
        [("SingleTaskGP", "botorch.models", "SingleTaskGP")]),
    row("fortbo_fortml_sparse", "sparse and variational GP adapters",
        "direct-external", "inducing-point NumPy posterior and GPyTorch variational contract",
        "fo test test_fortml_sparse; python scripts/run_benchmark.py",
        [("VariationalGP", "botorch.models.approximate_gp", "SingleTaskVariationalGP")]),
    row("fortbo_generated", "generated acquisition leaves",
        "generator-contract", "independent quadrature and finite-difference gates",
        "fo test test_generated_kernels test_acquisition"),
    row("fortbo_history", "observations, gradients, duplicates, checkpoint/resume",
        "reference-only", "restored-state replay and brute-force incumbent reconstruction",
        "fo test test_history test_workers"),
    row("fortbo_integrated", "integrated and risk-sensitive acquisition helpers",
        "reference-only", "independent quadrature/Monte Carlo utility definitions",
        "fo test test_integrated test_risk"),
    row("fortbo_knowledge_gradient", "sequential and batch knowledge gradient",
        "oracle-plus-api", "independent envelope/Monte Carlo oracle plus BoTorch qKG API availability",
        "python scripts/run_oracle_lane.py knowledge_gradient",
        [("qKG", "botorch.acquisition.knowledge_gradient", "qKnowledgeGradient")],
        "FortBO wall time <= BoTorch after value gate"),
    row("fortbo_linear_posterior", "linear posterior contract",
        "reference-only", "explicit affine posterior moments and samples",
        "fo test test_linear_posterior"),
    row("fortbo_metrics", "regret, feasibility, hypervolume, cost, memory, transfers",
        "reference-only", "independent metric identities and charged failed evaluations",
        "fo test test_metrics; python scripts/run_benchmark.py"),
    row("fortbo_mixed", "mixed continuous/integer/categorical policies",
        "reference-only", "decode/rounding invariants and brute-force finite domain",
        "fo test test_mixed test_space"),
    row("fortbo_monte_carlo", "marginal EI, PI, NEI, CRN, antithetic gradients",
        "direct-external", "frozen NumPy draws and estimator standard-error tolerance",
        "fo test test_monte_carlo; python scripts/run_acquisition_speed.py",
        [("qEI", "botorch.acquisition.monte_carlo", "qExpectedImprovement")],
        "FortBO wall time <= BoTorch for matched q=1 estimators"),
    row("fortbo_normal", "stable Gaussian PDF/CDF and tail branches",
        "reference-only", "mpmath/Simpson normal integrals and tail identities",
        "fo test test_acquisition test_entropy"),
    row("fortbo_optimize", "candidate optimization and refusal boundaries",
        "external-policy", "known constrained minima and independent objective evaluations",
        "fo test test_optimize test_end_to_end"),
    row("fortbo_ordering_bench", "recorded TuRBO ordering lanes",
        "external-policy", "pinned deterministic traces and objective-order oracle",
        "python scripts/record_fortbo_ordering.py"),
    row("fortbo_pareto", "Pareto archive and hypervolume",
        "oracle-plus-api", "analytic ZDT front/dominance oracle plus BoTorch qNEHVI API availability",
        "python scripts/run_oracle_lane.py pareto",
        [("qNEHVI", "botorch.acquisition.multi_objective.monte_carlo", "qNoisyExpectedHypervolumeImprovement")]),
    row("fortbo_pes", "predictive entropy search C1/C2/C3",
        "oracle-plus-api", "paper-definition quadrature/EP constraints plus BoTorch qPES API availability",
        "python scripts/run_oracle_lane.py pes",
        [("qPES", "botorch.acquisition.predictive_entropy_search", "qPredictiveEntropySearch")]),
    row("fortbo_placement", "derivative-bearing device placement refusals",
        "device-contract", "capability matrix and named refusal reasons",
        "fo test test_placement"),
    row("fortbo_posterior", "versioned posterior capability contract",
        "contract-only", "independent scalar normal and two-by-two covariance oracle",
        "fo test test_posterior_contract"),
    row("fortbo_preference", "preference/dueling BO utilities",
        "reference-only", "pairwise probability and utility-order oracle",
        "fo test test_preference"),
    row("fortbo_provenance", "CPU, transfer-inclusive, resident, refusal provenance",
        "contract-only", "schema validation and no-zero-for-refusal invariant",
        "fo test test_provenance"),
    row("fortbo_push", "push-based asynchronous evaluations",
        "reference-only", "event-order replay and failed-attempt accounting",
        "fo test test_push test_workers"),
    row("fortbo_quadratic", "DTuRBO bound-constrained quadratic subproblem",
        "external-policy", "enumerated active-set solution and KKT residual",
        "fo test test_quadratic test_dturbo_modes"),
    row("fortbo_risk", "VaR, CVaR, multi-fidelity/risk utilities",
        "reference-only", "normal-tail formulas and frozen empirical quantiles",
        "fo test test_risk"),
    row("fortbo_rover", "Rover-60 benchmark objective",
        "reference-only", "deterministic path-cost oracle and recorded ordering trace",
        "fo test test_rover; python scripts/run_speed_comparison.py"),
    row("fortbo_space", "continuous, integer, categorical, conditional spaces",
        "reference-only", "round-trip/decode invariants and exhaustive finite-domain scan",
        "fo test test_space test_mixed"),
    row("fortbo_stopping", "budget, target, stall, uncertainty, cost, wall-time rules",
        "reference-only", "boundary truth table and priority-order oracle",
        "fo test test_stopping"),
    row("fortbo_structured", "structured/multi-output posterior helpers",
        "direct-external", "independent output-wise posterior and covariance oracle",
        "fo test test_structured; python scripts/run_benchmark.py"),
    row("fortbo_thompson", "joint posterior Thompson batches",
        "oracle-plus-api", "frozen joint draws and without-replacement arg-min oracle plus BoTorch sampler API availability",
        "python scripts/run_oracle_lane.py thompson",
        [("qPosteriorSampling", "botorch.generation.sampling", "MaxPosteriorSampling")]),
    row("fortbo_trace", "TuRBO/DTuRBO trace and ratio diagnostics",
        "reference-only", "reconstructed radius/counter/ratio state transitions",
        "fo test test_trace"),
    row("fortbo_trust_region", "trust-region geometry, expansion, contraction, restart",
        "external-policy", "paper constants, bounded geometry, and trace transitions",
        "fo test test_trust_region test_turbo"),
    row("fortbo_turbo", "TuRBO-1, TuRBO-m, Thompson selection",
        "external-policy", "known-function regret and Uber/BoTorch pinned baselines",
        "python scripts/run_turbo_baselines.py scripts/run_botorch_turbo.py",
        [("BoTorch sampling", "botorch.generation.sampling", "ConstrainedMaxPosteriorSampling")],
        "FortBO regret no worse than the declared baseline at matched budget"),
    row("fortbo_turbo_driver", "sequential/asynchronous TuRBO driver",
        "external-policy", "objective-count, restart, failure, and replay oracle",
        "fo test test_turbo_driver test_end_to_end"),
    row("fortbo_workers", "asynchronous workers, pending fantasies, distributed replay",
        "reference-only", "deterministic event replay with charged failures",
        "fo test test_workers; python scripts/run_benchmark.py"),
]


def main() -> int:
    source_modules = sorted(
        path.stem for path in (FORTBO / "src").iterdir()
        if path.suffix.lower() == ".f90"
    )
    catalog_modules = sorted(item["module"] for item in CATALOG)
    missing = sorted(set(source_modules) - set(catalog_modules))
    extra = sorted(set(catalog_modules) - set(source_modules))
    if missing or extra:
        print(json.dumps({"missing": missing, "extra": extra}, indent=2))
        return 1
    missing_commands = []
    for item in CATALOG:
        for script in re.findall(r"scripts/([^ ;]+\.py)", item["benchmark_command"]):
            if not (ROOT / "scripts" / script).exists():
                missing_commands.append(script)
    if missing_commands:
        print(json.dumps({"missing_commands": sorted(set(missing_commands))}, indent=2))
        return 1

    payload = {
        "schema": 1,
        "objective": "complete FortBO feature benchmark coverage",
        "fortbo_revision": None,
        "matrix_status": "pass",
        "rows": CATALOG,
        "summary": {
            "features": len(CATALOG),
            "direct_or_external_policy": sum(
                item["comparability"] in {"direct-external", "external-policy"}
                for item in CATALOG),
            "oracle_plus_api": sum(
                item["comparability"] == "oracle-plus-api" for item in CATALOG),
            "reference_or_contract_only": sum(
                item["comparability"] not in {"direct-external", "external-policy"}
                for item in CATALOG),
            "external_requirements_unavailable": sum(
                item["status"] == "external-unavailable" for item in CATALOG),
        },
    }
    fixture = ROOT / "fixtures" / "feature_benchmark_matrix.json"
    fixture.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = ROOT / "results" / "feature_benchmark_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "module", "feature", "comparability", "status", "oracle",
            "benchmark_command", "performance_target", "performance_status",
        ], lineterminator="\n")
        writer.writeheader()
        for item in CATALOG:
            writer.writerow({key: item[key] for key in writer.fieldnames})
    print(json.dumps({"status": payload["matrix_status"], "fixture": str(fixture),
                      "csv": str(csv_path), "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
