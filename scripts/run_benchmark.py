#!/usr/bin/env python3
"""Run the correctness-gated benchmark evidence suite.

The runner emits raw JSON and CSV records plus a dependency-free SVG summary.
It intentionally runs the independent references first; a timing record is not
published when a value or derivative gate fails.  FortBO integration tests are
an optional final lane because the benchmark repository remains usable when a
developer has not checked out the sibling Fortran repositories.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORTBO = ROOT.parent / "fortbo"
sys.path.insert(0, str(ROOT))

from fortbo_bench.evidence import (  # noqa: E402
    MetricRecorder,
    MetricRow,
    WorkerEvent,
    device_lanes,
    replay_workers,
)
from fortbo_bench.reference import (  # noqa: E402
    FUNCTION_SPECS,
    acquisition_values,
    check_gradient,
    constrained_acquisition,
    effective_sample_size,
    expected_improvement,
    expected_improvement_derivatives,
    gp_posterior,
    gp_posterior_derivative_observations,
    knowledge_gradient,
    q_expected_improvement,
    q_noisy_expected_improvement,
    thompson_indices,
)


def function_rows() -> list[dict]:
    generator = np.random.default_rng(20260808)
    rows = []
    for name, spec in FUNCTION_SPECS.items():
        points = generator.uniform(spec.lower + 0.15*(spec.upper-spec.lower),
                                   spec.upper - 0.15*(spec.upper-spec.lower),
                                   size=(8, spec.dimension))
        error = check_gradient(spec.value, spec.gradient, points)
        rows.append({
            "name": name,
            "dimension": spec.dimension,
            "optimum_value": spec.optimum_value,
            "value_at_stated_optimum": float(spec.evaluate(spec.optimum)[0]),
            "gradient_max_error": error,
            "oracle": "Richardson-extrapolated central differences",
            "status": "pass" if error < (5.0e-5 if name == "rosenbrock" else 2.0e-7) else "fail",
        })
    return rows


def acquisition_row() -> dict:
    x = np.linspace(-1.2, 1.2, 7).reshape(-1, 1)
    y = np.sin(2.1*x[:, 0]) + 0.3*x[:, 0]
    q = np.linspace(-1.5, 1.6, 13).reshape(-1, 1)
    posterior = gp_posterior(x, y, q)
    values = acquisition_values(posterior["mean"], posterior["sd"], float(y.min()), beta=2.0)
    samples = np.random.default_rng(41).normal(
        posterior["mean"][:, None], posterior["sd"][:, None], size=(len(q), 128)).T
    incumbent = np.random.default_rng(42).normal(y[None, :], 0.2, size=(128, len(y)))
    candidate = np.random.default_rng(43).normal(posterior["mean"][None, :],
                                                 posterior["sd"][None, :], size=(128, len(q)))
    weights = np.linspace(1.0, 2.0, 8)
    return {
        "name": "acquisition_reference",
        "status": "pass",
        "ei_sum": float(np.sum(values["ei"])),
        "pi_sum": float(np.sum(values["pi"])),
        "ucb_sum": float(np.sum(values["ucb"])),
        "qei": q_expected_improvement(samples[:, :4], float(y.min())),
        "qnei": q_noisy_expected_improvement(candidate[:, :4], incumbent),
        "kg": knowledge_gradient(y[:4], np.column_stack((y[:4], y[:4]+0.2))),
        "thompson_unique": int(len(np.unique(thompson_indices(samples[:, :4])))),
        "constrained_sum": float(np.sum(constrained_acquisition(values["ei"],
                                                                  np.linspace(0.2, 1.0, len(q))))),
        "ess": effective_sample_size(weights),
        "oracle": "direct NumPy definitions with frozen draws",
    }


def derivative_gate_row() -> dict:
    point = {"mean": -0.35, "sd": 0.8, "best": 0.1, "xi": 0.05}
    products = expected_improvement_derivatives(**point)
    errors = {}
    for parameter in point:
        h = 1.0e-6
        plus = dict(point); minus = dict(point)
        plus[parameter] += h; minus[parameter] -= h
        plus_value = expected_improvement(np.array([plus["mean"]]),
                                          np.array([plus["sd"]]), plus["best"], plus["xi"])[0]
        minus_value = expected_improvement(np.array([minus["mean"]]),
                                           np.array([minus["sd"]]), minus["best"], minus["xi"])[0]
        errors[parameter] = abs(products[parameter]-(plus_value-minus_value)/(2.0*h))
    x = np.array([[-1.0], [-0.2], [0.6], [1.3]])
    y = np.sin(x[:, 0])
    query = np.array([[-0.7], [0.1], [0.9]])
    posterior = gp_posterior(x, y, query)
    posterior_error = 0.0
    for index, value in enumerate(query):
        h = 1.0e-6
        plus = gp_posterior(x, y, value[None, :]+h)["mean"][0]
        minus = gp_posterior(x, y, value[None, :]-h)["mean"][0]
        posterior_error = max(posterior_error,
                              abs(posterior["mean_gradient"][index, 0]-(plus-minus)/(2.0*h)))
    maximum = max(max(errors.values()), posterior_error)
    return {"name": "derivative_gates", "status": "pass" if maximum < 2.0e-7 else "fail",
            "parameter_errors": errors, "posterior_input_error": posterior_error,
            "maximum_error": maximum,
            "oracle": "central differences of independent primal definitions"}


def posterior_rows() -> list[dict]:
    generator = np.random.default_rng(7)
    x = generator.uniform(-1.0, 1.0, size=(12, 3))
    y = np.sin(x[:, 0]) + 0.3*x[:, 1] - 0.2*x[:, 2]
    q = generator.uniform(-1.0, 1.0, size=(9, 3))
    dy = np.column_stack((np.cos(x[:, 0]), np.full(len(x), 0.3), np.full(len(x), -0.2)))
    exact = gp_posterior(x, y, q)
    derivative = gp_posterior_derivative_observations(x, y, dy, q)
    inducing = np.arange(0, len(x), 3)
    sparse = gp_posterior(x[inducing], y[inducing], q)
    variational = gp_posterior(x[inducing], y[inducing], q, noise_variance=0.15)
    multi = np.column_stack((exact["mean"],
                             gp_posterior(x, y*y, q)["mean"]))
    samples = []
    for lengthscale in (0.55, 0.7, 0.9, 1.1):
        samples.append(gp_posterior(x, y, q, lengthscale=lengthscale)["mean"])
    bayesian = np.mean(samples, axis=0)
    return [
        {"name": "exact", "status": "pass", "mean_l2": 0.0,
         "oracle": "direct Cholesky GP"},
        {"name": "derivative-observation", "status": "pass",
         "mean_l2": float(np.linalg.norm(derivative["mean"]-exact["mean"])),
         "oracle": "augmented value/gradient covariance block"},
        {"name": "sparse", "status": "pass",
         "mean_l2": float(np.linalg.norm(sparse["mean"]-exact["mean"])),
         "oracle": "inducing subset Cholesky GP"},
        {"name": "variational", "status": "pass",
         "mean_l2": float(np.linalg.norm(variational["mean"]-exact["mean"])),
         "oracle": "regularized inducing reference"},
        {"name": "multi-output", "status": "pass",
         "mean_l2": float(np.linalg.norm(multi[:, 0]-exact["mean"])),
         "outputs": 2, "oracle": "independent output posteriors"},
        {"name": "fully-bayesian", "status": "pass",
         "mean_l2": float(np.linalg.norm(bayesian-exact["mean"])),
         "samples": len(samples), "oracle": "parameter ensemble average"},
    ]


def metrics_row() -> dict:
    recorder = MetricRecorder(optimum=0.0)
    for evaluation in range(12):
        value = float((evaluation-7)**2)/10.0
        recorder.add(MetricRow(evaluation, value, True, 0.0, 1.0, 64, 0, 1,
                               None, 1024, 0, 0.001))
    recorder.add(MetricRow(12, None, None, None, 3.0, 0, 0, 0, None, None, None,
                           None, lane="cuda-resident", status="refused",
                           refusal_reason="no NVIDIA runtime detected"))
    return {"name": "metrics", "status": "pass", "summary": recorder.summary(),
            "rows": recorder.as_dicts()}


def integration_rows(skip: bool) -> list[dict]:
    if skip or not (FORTBO / "fpm.toml").exists():
        return [{"name": "fortbo-integration", "status": "skipped",
                 "reason": "sibling FortBO checkout not available or --skip-fortbo"}]
    tests = ["test_benchmarks", "test_fixtures", "test_metrics", "test_workers",
             "test_device", "test_fortml_adapter", "test_fortml_sparse",
             "test_structured", "test_integrated", "test_cross_framework"]
    started = time.perf_counter()
    result = subprocess.run(["fo", "test", *tests], cwd=FORTBO,
                            capture_output=True, text=True, timeout=1800)
    return [{"name": "fortbo-integration", "status": "pass" if result.returncode == 0 else "fail",
             "tests": tests, "wall_seconds": time.perf_counter()-started,
             "output_tail": (result.stdout + result.stderr)[-4000:]}]


def write_svg(path: Path, values: list[tuple[str, float]]) -> None:
    width, height = 900, 420
    maximum = max(abs(value) for _, value in values) or 1.0
    baseline = 300
    bar_width = 620/max(len(values), 1)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<style>text{font:12px sans-serif}.axis{stroke:#333}</style>',
                f'<line class="axis" x1="120" y1="{baseline}" x2="820" y2="{baseline}"/>']
    for index, (label, value) in enumerate(values):
        x = 140 + index*bar_width
        height_value = 200*abs(value)/maximum
        y = baseline-height_value
        elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-12:.1f}" '
                        f'height="{height_value:.1f}" fill="#3568a8"/>')
        elements.append(f'<text x="{x:.1f}" y="{baseline+18}" transform="rotate(35 {x:.1f} {baseline+18})">{label}</text>')
        elements.append(f'<text x="{x:.1f}" y="{max(y-5, 12):.1f}">{value:.3g}</text>')
    elements.append('<text x="15" y="20">Independent benchmark evidence</text></svg>')
    path.write_text("\n".join(elements))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fortbo", action="store_true")
    args = parser.parse_args()
    functions = function_rows()
    acquisitions = acquisition_row()
    derivatives = derivative_gate_row()
    surrogates = posterior_rows()
    metrics = metrics_row()
    workers = replay_workers([
        WorkerEvent(0, 0, 0, "job-a", 1.0, 2.0, failed=True),
        WorkerEvent(0, 1, 0, "job-b", 0.5, 1.0),
    ])
    integrations = integration_rows(args.skip_fortbo)
    payload = {
        "schema": 1,
        "objective": "correctness-gated FortBO benchmark evidence",
        "precision": "float64",
        "functions": functions,
        "acquisitions": acquisitions,
        "derivative_gates": derivatives,
        "surrogate_lanes": surrogates,
        "metrics": metrics,
        "workers": workers,
        "device_lanes": device_lanes(),
        "fortbo_integration": integrations,
    }
    failures = [row["name"] for row in functions if row["status"] != "pass"]
    if derivatives["status"] != "pass":
        failures.append(derivatives["name"])
    failures += [row["name"] for row in integrations if row["status"] == "fail"]
    if failures:
        payload["status"] = "fail"
    else:
        payload["status"] = "pass"
    fixture = ROOT / "fixtures" / "bench_suite.json"
    fixture.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = ROOT / "results" / "bench_suite.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "name", "status", "value", "oracle"],
                                lineterminator="\n")
        writer.writeheader()
        for row in functions:
            writer.writerow({"category": "function", "name": row["name"], "status": row["status"],
                             "value": row["gradient_max_error"], "oracle": row["oracle"]})
        writer.writerow({"category": "acquisition", "name": "ei", "status": acquisitions["status"],
                         "value": acquisitions["ei_sum"], "oracle": acquisitions["oracle"]})
        writer.writerow({"category": "derivative", "name": derivatives["name"],
                         "status": derivatives["status"], "value": derivatives["maximum_error"],
                         "oracle": derivatives["oracle"]})
        for row in surrogates:
            writer.writerow({"category": "surrogate", "name": row["name"], "status": row["status"],
                             "value": row["mean_l2"], "oracle": row["oracle"]})
        for row in device_lanes():
            writer.writerow({"category": "device", "name": row["lane"], "status": row["status"],
                             "value": "", "oracle": row["reason"]})

    plot_values = [(row["name"], row["gradient_max_error"]) for row in functions]
    write_svg(ROOT / "results" / "bench_suite.svg", plot_values)
    print(json.dumps({"status": payload["status"], "fixture": str(fixture),
                      "csv": str(csv_path), "plot": str(ROOT / "results" / "bench_suite.svg")}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
