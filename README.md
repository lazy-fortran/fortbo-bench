# fortbo-bench

Correctness-gated benchmarks for FortBO Bayesian optimization.

The harness compares FortBO against independent analytic grids and pinned
BoTorch/GPyTorch/JAX/NumPy references. It records simple and cumulative regret,
best feasible value, constraint violations, acquisition evaluations, gradient
calls, model fits, memory, transfers, and wall time. Statistical sample
efficiency is kept separate from raw execution throughput.

The benchmark plan is in [`ROADMAP.md`](ROADMAP.md). All contents are MIT
licensed.
