# fortbo-bench roadmap

This repository owns reproducible evidence for FortBO. It does not implement
Bayesian-optimization policies. Every row identifies the FortBO and surrogate
revisions, compiler, precision, hardware, seed, initialization design,
restart count, acquisition optimizer, stopping rule, and independent oracle.

## Complete FortBO feature matrix

The release matrix covers every source module, not only the headline TuRBO and
analytic-acquisition examples. The machine-readable inventory is
[`feature_benchmark_matrix.json`](fixtures/feature_benchmark_matrix.json),
with a flat reviewable export in
[`feature_benchmark_matrix.csv`](results/feature_benchmark_matrix.csv). Run
`python scripts/run_feature_matrix.py` after changing a FortBO module; it
fails if a source module has no benchmark row and records unavailable external
packages explicitly.

Rows use one of three comparison contracts: `direct-external` is a matched
value/timing lane against BoTorch or GPyTorch with an independent oracle;
`external-policy` matches objective, budget, seed, and stopping rule against a
published or BoTorch policy; `reference-only` and `contract-only` use a
defining oracle, invariant, or typed refusal where no like-for-like competitor
exists. `oracle-plus-api` runs the Fortran contract and independent oracle and
checks that a named external API exists, but records a typed refusal when the
execution graphs are not semantically identical enough for a speed claim. The
performance target is per lane, hardware, and revision: FortBO must meet or
beat the named comparator after the correctness gate. A missing or
non-comparable lane remains visible rather than being counted as a speed win.

Hardware evidence is kept separately in
[`hardware_benchmark.json`](fixtures/hardware_benchmark.json). The current CPU
lane is complete on `faepkub4` (gfortran 12.2.0, OpenBLAS, 50/50 tests, all
acquisition and ordering rows). The Slurm GPU lane completed on `acluster`
(`node34`, Tesla T4, CUDA device 0), but is a typed refusal: only gfortran
12.2.0 with no accelerator backend is available, so the device contract passes
without claiming GPU performance.

## Work packages

- [x] Create the MIT-licensed benchmark repository and provenance contract.
- [x] Add analytic one-dimensional functions, Branin, Hartmann, Ackley,
  constrained synthetic functions, noisy objectives, and multi-objective
  fixtures with dense-grid or known-optimum oracles. `fortbo_bench.reference`
  carries the direct definitions, and `tests/test_reference.py` checks stated
  optima, independent gradients, separated constraints, and analytic ZDT
  fronts.
- [x] Add independent NumPy reference implementations for EI, PI, UCB,
  qEI/qNEI, knowledge gradient, Thompson sampling, and constrained policies.
  The tests freeze batch draws and check the defining order statistics rather
  than comparing a routine with itself.
- [x] Add BoTorch/GPyTorch and JAX comparison lanes with matched surrogate
  kernels, normalization, seeds, restarts, precision, and budgets.
  `emit_reference.py`, `emit_regret.py`, `run_botorch_turbo.py`, and
  `run_turbo_baselines.py` generate the pinned JSON fixtures; current runs use
  the repository virtual environment and float64.
- [x] Compare exact, derivative-observation, sparse, variational, multi-output,
  and fully Bayesian FortML contract lanes. `scripts/run_benchmark.py` emits
  independent NumPy reference rows for all six, while its sibling FortBO
  integration lane executes the corresponding FortML adapter tests.
- [x] Gate input and parameter acquisition derivatives against finite
  differences, adjoint identities, and dense posterior references.
  Richardson input differences, direct EI parameter products, and an
  independently assembled derivative-observation GP are tested before any
  timing record is published.
- [x] Measure simple/cumulative regret, feasible best value, hypervolume,
  constraint violations, model fits, acquisition evaluations, gradient calls,
  memory, transfers, and wall time. `MetricRecorder` keeps unavailable values
  as `None` and emits the complete summary in `bench_suite.json`.
- [x] Add CPU, OpenACC, transfer-inclusive CUDA, resident CUDA, and typed
  refusal rows. Never report a hidden host fallback as GPU performance.
  `device_lanes()` records each lane separately and states when the runtime is
  unavailable.
- [x] Add asynchronous workers, pending-point fantasies, checkpoint/resume,
  duplicate/failure policies, and deterministic distributed replay.
  `replay_workers`, `pending_fantasy`, and `MetricRecorder.checkpoint` are
  covered by behavioral tests; failed attempts remain charged and never become
  objective values.
- [x] Publish raw CSV/JSON data and plots for every release claim.
  `scripts/run_benchmark.py` writes the checked-in schema fixture, raw CSV, and
  dependency-free SVG after the gates pass.

## Acceptance

No acquisition policy is called production-ready from one successful objective.
A released lane must match an independent value/gradient or dense-grid oracle,
show reproducible seeded behavior, report optimization and statistical metrics,
and document its complete device contract.
