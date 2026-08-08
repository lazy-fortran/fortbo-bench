# fortbo-bench

Correctness-gated benchmarks for FortBO Bayesian optimization.

The harness compares FortBO against independent analytic grids and pinned
BoTorch/GPyTorch/JAX/NumPy references. It records simple and cumulative regret,
best feasible value, constraint violations, acquisition evaluations, gradient
calls, model fits, memory, transfers, and wall time. Statistical sample
efficiency is kept separate from raw execution throughput.

The benchmark plan is in [`ROADMAP.md`](ROADMAP.md). All contents are MIT
licensed.

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
