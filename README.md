# fortbo-bench

Correctness-gated benchmarks for FortBO Bayesian optimization.

The harness compares FortBO against independent analytic grids and pinned
BoTorch/GPyTorch/JAX/NumPy references. It records simple and cumulative regret,
best feasible value, constraint violations, acquisition evaluations, gradient
calls, model fits, memory, transfers, and wall time. Statistical sample
efficiency is kept separate from raw execution throughput.

The benchmark plan is in [`ROADMAP.md`](ROADMAP.md). All contents are MIT
licensed.

## Run the evidence suite

The independent reference tests have no FortBO import and use direct NumPy
definitions as their oracle:

```
python3 -m unittest discover -s tests -v
python3 scripts/run_benchmark.py
```

The runner fails before publication if a value or derivative gate fails. A
successful run writes `fixtures/bench_suite.json`, `results/bench_suite.csv`,
and `results/bench_suite.svg`. CPU, OpenACC, transfer-inclusive CUDA, and
resident CUDA are separate rows; an unavailable device is recorded as
unavailable/refused with a reason, never as a zero-time host fallback.

The surrogate rows are independent contract-level references for exact,
derivative-observation, sparse, variational, multi-output, and fully Bayesian
conditioning. When the sibling FortBO checkout is present, the runner also
executes the exact, sparse, structured, integrated, device, metrics, and
cross-framework FortBO tests. The comparison scripts below retain the pinned
BoTorch/GPyTorch/JAX lanes and their provenance fixtures.

Use `python3 scripts/run_benchmark.py --skip-fortbo` when only the independent
benchmark repository is checked out.

## Provenance

`scripts/fetch_provenance.py` downloads the papers and reference source trees
FortBO is built against into a gitignored `provenance/`. Run it before working
on an acquisition or a benchmark claim:

```
python3 scripts/fetch_provenance.py            # papers and source trees
python3 scripts/fetch_provenance.py --papers-only
python3 scripts/fetch_provenance.py --only pes
```

`scripts/sources.py` says what each source is, what FortBO needs *from* it, and
which file consumes it, so an implementation can be traced back to the
definition it was written from. A manifest records what was fetched and at which
commit — without it, "this matches the paper" is uncheckable later, because
nobody can tell which version was read.

Read, not ported. FortBO's first-principles mandate is unchanged: closed forms
are derived through FortSym and implementations are written against the
definition. The point of having the sources on disk is that reconstructing a
method from memory produces something subtly different — as happened with
predictive entropy search, where the paper's own three-constraint decomposition
made precise what a remembered version had left vague.
