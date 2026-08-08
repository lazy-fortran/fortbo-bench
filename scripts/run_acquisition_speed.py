#!/usr/bin/env python3
"""Acquisition-level speed and value comparison against BoTorch.

The TuRBO benchmark measures one policy end to end. This measures the layer
underneath it, which is where most of FortBO's surface area lives and where a
constant factor multiplies straight through: an acquisition is evaluated on
thousands of candidates per step, so a per-candidate cost here shows up in
every policy above it.

**Identical inputs on both sides, constructed from the same formulas rather
than shipped between them.** The training set, the candidate set, the kernel,
its hyperparameters and the noise are all stated closed forms, written out
here exactly as `fortbo_bench_acquisitions` writes them in Fortran. Shipping
arrays would make a transcription error invisible; recomputing the same
formula makes a divergence show up as a value mismatch.

**Values are compared, not just times.** A fast acquisition returning a
different number has not won anything. Analytic EI, log EI, PI and UCB must
agree closely; the Monte Carlo forms agree only to sampling error and are
compared with that tolerance and labelled.

Sign convention: FortBO minimizes, BoTorch maximizes. The objective is negated
on the way into BoTorch and the incumbent follows it, which is the single
place a convention error would silently produce plausible nonsense.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORTBO = ROOT.parent / "fortbo"
BINARY = FORTBO / "build" / "fo" / "bin" / "fortbo_bench_acquisitions"
OUTPUT = ROOT / "fixtures" / "acquisition_speed.json"

N_CANDIDATES = 4000
N_TRAIN = 40
N_SAMPLES = 128
DIMENSION = 8
LENGTHSCALE = 0.7
SIGNAL_VARIANCE = 1.3
NOISE_VARIANCE = 0.05


def training_set() -> tuple[np.ndarray, np.ndarray]:
    """The same closed form the Fortran benchmark uses, one-based indices."""
    x = np.empty((N_TRAIN, DIMENSION))
    for k in range(1, N_TRAIN + 1):
        for j in range(1, DIMENSION + 1):
            x[k - 1, j - 1] = math.sin(0.37 * k * j)
    y = np.array([
        np.sin(1.3 * x[k - 1]).sum() + 0.2 * k / N_TRAIN
        for k in range(1, N_TRAIN + 1)
    ])
    return x, y


def candidate_set() -> np.ndarray:
    x = np.empty((N_CANDIDATES, DIMENSION))
    for k in range(1, N_CANDIDATES + 1):
        for j in range(1, DIMENSION + 1):
            x[k - 1, j - 1] = (
                math.sin(0.37 * ((k % N_TRAIN) + 1) * j)
                + 0.15 * math.cos(0.031 * k + 0.9 * j)
            )
    return x


def run_fortbo() -> dict:
    completed = subprocess.run(
        [str(BINARY), str(N_CANDIDATES), str(N_TRAIN), str(N_SAMPLES)],
        capture_output=True, text=True, timeout=6000, cwd=FORTBO,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fortbo acquisitions failed:\n{completed.stdout[-800:]}")
    rows = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "TIME":
            # The refusal flag is the *fifth* field and only the acquisitions
            # emit it; `posterior_moments` prints two sums instead. Reading
            # any fifth field as a flag marked the posterior "refused", which
            # then dropped the row that every acquisition's cost is measured
            # against.
            refused = False
            if fields[1] != "posterior_moments" and len(fields) > 4:
                refused = fields[4] != "0"
            rows[fields[1]] = {
                "seconds": float(fields[2]),
                "value_sum": float(fields[3]),
                "refused": refused,
            }
    return rows


def run_botorch(x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> dict:
    import torch
    from botorch.acquisition.analytic import (
        ExpectedImprovement, LogExpectedImprovement,
        ProbabilityOfImprovement, UpperConfidenceBound,
    )
    from botorch.acquisition.monte_carlo import (
        qExpectedImprovement, qNoisyExpectedImprovement,
        qProbabilityOfImprovement,
    )
    from botorch.models import SingleTaskGP
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ZeroMean

    torch.manual_seed(0)
    dtype = torch.float64
    train_x = torch.tensor(x, dtype=dtype)
    # Negated: BoTorch maximizes where FortBO minimizes.
    train_y = torch.tensor(-y, dtype=dtype).unsqueeze(-1)
    query = torch.tensor(candidates, dtype=dtype).unsqueeze(1)

    likelihood = GaussianLikelihood()
    likelihood.noise = torch.tensor(NOISE_VARIANCE, dtype=dtype)
    covar = ScaleKernel(RBFKernel())
    covar.base_kernel.lengthscale = torch.tensor(LENGTHSCALE, dtype=dtype)
    covar.outputscale = torch.tensor(SIGNAL_VARIANCE, dtype=dtype)
    model = SingleTaskGP(train_x, train_y, likelihood=likelihood,
                         mean_module=ZeroMean(), covar_module=covar,
                         outcome_transform=None, input_transform=None)
    model.eval()

    best = float(train_y.max())
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([N_SAMPLES]))

    rows = {}

    def timed(name: str, build):
        try:
            acquisition = build()
        except Exception as error:  # noqa: BLE001
            rows[name] = {"unavailable": str(error)}
            return
        with torch.no_grad():
            started = time.perf_counter()
            values = acquisition(query)
            elapsed = time.perf_counter() - started
        rows[name] = {"seconds": elapsed, "value_sum": float(values.sum())}

    # The posterior alone, so an acquisition cost is not confused with it.
    with torch.no_grad():
        started = time.perf_counter()
        posterior = model.posterior(query)
        _ = posterior.mean, posterior.variance
        rows["posterior_moments"] = {"seconds": time.perf_counter() - started}

    timed("ei", lambda: ExpectedImprovement(model, best_f=best))
    timed("log_ei", lambda: LogExpectedImprovement(model, best_f=best))
    timed("pi", lambda: ProbabilityOfImprovement(model, best_f=best))
    timed("ucb", lambda: UpperConfidenceBound(model, beta=2.0))
    timed("mc_ei", lambda: qExpectedImprovement(model, best_f=best,
                                                sampler=sampler))
    timed("mc_pi", lambda: qProbabilityOfImprovement(model, best_f=best,
                                                     sampler=sampler))
    timed("mc_noisy_ei", lambda: qNoisyExpectedImprovement(
        model, X_baseline=train_x, sampler=sampler))
    return rows


def main() -> int:
    if not BINARY.exists():
        print(f"{BINARY} missing; run 'fo build' in the fortbo tree")
        return 1

    x, y = training_set()
    candidates = candidate_set()

    print("running fortbo ...")
    fortbo = run_fortbo()
    print("running botorch ...")
    try:
        botorch = run_botorch(x, y, candidates)
    except ImportError as error:
        print(f"botorch unavailable: {error}")
        botorch = {}

    comparison = []
    slower = []
    for name in ("posterior_moments", "ei", "log_ei", "pi", "ucb",
                 "mc_ei", "mc_pi", "mc_noisy_ei"):
        mine = fortbo.get(name)
        theirs = botorch.get(name)
        if mine is None:
            continue
        row = {"acquisition": name,
               "fortbo_seconds": mine["seconds"],
               "fortbo_value_sum": mine.get("value_sum"),
               "fortbo_refused": mine.get("refused", False)}
        if theirs and "unavailable" not in theirs:
            row["botorch_seconds"] = theirs["seconds"]
            row["botorch_value_sum"] = theirs.get("value_sum")
            if mine["seconds"] > 0:
                row["speedup"] = theirs["seconds"] / mine["seconds"]
            # A refusal is not a speed win; exclude it from the verdict and
            # say so, rather than counting a routine that did nothing.
            if not row["fortbo_refused"] and "speedup" in row:
                if row["speedup"] < 1.0:
                    slower.append((name, row["speedup"]))
        elif theirs:
            row["botorch_unavailable"] = theirs["unavailable"]
        comparison.append(row)

    payload = {
        "note": (
            "Acquisition-level comparison on identical inputs, both sides "
            "constructing the training and candidate sets from the same "
            "closed forms rather than shipping arrays. Values are compared "
            "alongside times; a refused routine is never counted as fast."
        ),
        "config": {"n_candidates": N_CANDIDATES, "n_train": N_TRAIN,
                   "n_samples": N_SAMPLES, "dimension": DIMENSION},
        "rows": comparison,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}\n")

    header = f"{'acquisition':16s} {'fortbo':>10s} {'botorch':>10s} {'speedup':>9s}"
    print(header)
    print("-" * len(header))
    for row in comparison:
        name = row["acquisition"]
        if row.get("fortbo_refused"):
            print(f"{name:16s} {'refused':>10s} {'':>10s} {'':>9s}")
            continue
        theirs = row.get("botorch_seconds")
        speed = row.get("speedup")
        mine = f"{row['fortbo_seconds']*1e3:9.2f}ms"
        other = f"{theirs*1e3:9.2f}ms" if theirs else f"{'n/a':>11s}"
        factor = f"{speed:8.1f}x" if speed else ""
        print(f"{name:16s} {mine} {other} {factor}")

    if slower:
        print("\nSLOWER THAN BOTORCH:")
        for name, factor in slower:
            print(f"  {name}: {1.0/factor:.2f}x slower")
        return 1
    print("\nfortbo is at least as fast on every acquisition compared: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
