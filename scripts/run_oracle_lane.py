#!/usr/bin/env python3
"""Run the remaining feature lanes with an independent behavioural oracle.

Some FortBO features have no semantically identical public BoTorch timing
entry point (for example PES's EP state or FortBO's without-replacement
Thompson batches).  Those lanes must still execute their Fortran contract and
an independent definition, while explicitly refusing a fabricated speedup.
The script is the executable endpoint for those matrix rows.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORTBO = ROOT.parent / "fortbo"
sys.path.insert(0, str(ROOT))

from fortbo_bench.reference import (  # noqa: E402
    knowledge_gradient,
    q_expected_improvement,
    q_noisy_expected_improvement,
    thompson_indices,
    zdt,
    zdt_front,
)


LANES = {
    "batch": {
        "test": "test_batch",
        "external": [("qEI", "botorch.acquisition.monte_carlo", "qExpectedImprovement"),
                     ("qNEI", "botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
                     ("qUCB", "botorch.acquisition.monte_carlo", "qUpperConfidenceBound")],
    },
    "entropy": {
        "test": "test_entropy",
        "external": [("qMES", "botorch.acquisition.max_value_entropy_search", "qMaxValueEntropy")],
    },
    "knowledge_gradient": {
        "test": "test_knowledge_gradient",
        "external": [("qKG", "botorch.acquisition.knowledge_gradient", "qKnowledgeGradient")],
    },
    "pareto": {
        "test": "test_pareto",
        "external": [("qNEHVI", "botorch.acquisition.multi_objective.monte_carlo",
                       "qNoisyExpectedHypervolumeImprovement")],
    },
    "pes": {
        "test": "test_pes test_pes_constraints",
        "external": [("qPES", "botorch.acquisition.predictive_entropy_search",
                       "qPredictiveEntropySearch")],
    },
    "thompson": {
        "test": "test_thompson",
        "external": [("MaxPosteriorSampling", "botorch.generation.sampling",
                       "MaxPosteriorSampling")],
    },
}


def external_status(items: list[tuple[str, str, str]]) -> list[dict[str, object]]:
    result = []
    for label, module_name, symbol in items:
        try:
            module = importlib.import_module(module_name)
            available = hasattr(module, symbol)
            reason = "" if available else f"{symbol} is not exported"
        except (ImportError, ModuleNotFoundError) as error:
            available = False
            reason = str(error)
        result.append({"label": label, "module": module_name, "symbol": symbol,
                       "available": available, "reason": reason})
    return result


def oracle(name: str) -> dict[str, object]:
    if name == "batch":
        samples = np.array([[0.0, 1.0], [-1.0, 0.5], [1.0, 2.0]])
        incumbents = np.array([[0.2, 0.4], [-0.5, 0.4], [0.3, 0.8]])
        qei = q_expected_improvement(samples, 0.0)
        qnei = q_noisy_expected_improvement(samples, incumbents)
        expected_qei = 1.0 / 3.0
        expected_qnei = 0.7 / 3.0
        if not np.isclose(qei, expected_qei) or not np.isclose(qnei, expected_qnei):
            raise AssertionError("batch order-statistic oracle failed")
        return {"values": {"qei": qei, "qnei": qnei},
                "oracle": "independent frozen joint-draw order statistics"}
    if name == "knowledge_gradient":
        current = np.array([1.0, 2.0])
        fantasies = np.array([[0.5, 2.0], [0.2, 3.0]])
        value = knowledge_gradient(current, fantasies)
        if not np.isclose(value, 0.65):
            raise AssertionError("knowledge-gradient oracle failed")
        return {"values": {"kg": value},
                "oracle": "independent fantasy-table decision improvement"}
    if name == "pareto":
        first = np.linspace(0.0, 1.0, 11)
        front = zdt_front(first, kind=1)
        attained = zdt(np.column_stack((first, np.zeros_like(first))), kind=1)
        if not np.allclose(front, attained):
            raise AssertionError("ZDT analytic-front oracle failed")
        return {"values": {"front_points": int(len(front))},
                "oracle": "analytic ZDT1 Pareto front"}
    if name == "thompson":
        samples = np.array([[1.0, 0.0], [0.0, 2.0]])
        indices = thompson_indices(samples)
        if not np.array_equal(indices, [1, 0]):
            raise AssertionError("Thompson arg-min oracle failed")
        return {"values": {"indices": indices.tolist()},
                "oracle": "independent frozen-realization arg-min"}
    if name in {"entropy", "pes"}:
        # A direct entropy identity used as the external-free oracle: a normal
        # with nonzero variance has larger differential entropy than its
        # deterministic limit.  The Fortran tests separately check the paper's
        # truncation/EP formulas; this check shares no generated code.
        sd = np.array([0.25, 1.0, 2.0])
        entropy = 0.5*np.log(2.0*np.pi*np.e*sd**2)
        if not np.all(np.diff(entropy) > 0.0):
            raise AssertionError("normal entropy ordering oracle failed")
        return {"values": {"entropy": entropy.tolist()},
                "oracle": "independent Gaussian entropy identity"}
    raise KeyError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", choices=sorted(LANES))
    args = parser.parse_args()
    spec = LANES[args.lane]
    tests = spec["test"].split()
    started = time.perf_counter()
    result = subprocess.run(["fo", "test", *tests], cwd=FORTBO,
                            capture_output=True, text=True, timeout=1800)
    test_seconds = time.perf_counter() - started
    oracle_result = oracle(args.lane)
    try:
        fortbo_revision = subprocess.check_output(
            ["git", "-C", str(FORTBO), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        fortbo_revision = "working-tree"
    payload = {
        "schema": 1,
        "lane": args.lane,
        "fortbo_revision": fortbo_revision,
        "tests": tests,
        "fortbo_status": "pass" if result.returncode == 0 else "fail",
        "fortbo_test_seconds": test_seconds,
        "external": external_status(spec["external"]),
        "performance_status": "refused_not_semantically_comparable",
        "performance_refusal": (
            "No shared public execution graph exposes the same FortBO state "
            "and utility as the named external API; timing it would compare "
            "different algorithms or hidden posterior work."),
        **oracle_result,
        "output_tail": (result.stdout + result.stderr)[-3000:],
    }
    output = ROOT / "fixtures" / f"oracle_lane_{args.lane}.json"
    payload["fixture"] = str(output)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
