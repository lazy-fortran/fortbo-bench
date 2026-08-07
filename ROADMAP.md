# fortbo-bench roadmap

This repository owns reproducible evidence for FortBO. It does not implement
Bayesian-optimization policies. Every row identifies the FortBO and surrogate
revisions, compiler, precision, hardware, seed, initialization design,
restart count, acquisition optimizer, stopping rule, and independent oracle.

## Work packages

- [x] Create the MIT-licensed benchmark repository and provenance contract.
- [ ] Add analytic one-dimensional functions, Branin, Hartmann, Ackley,
  constrained synthetic functions, noisy objectives, and multi-objective
  fixtures with dense-grid or known-optimum oracles.
- [ ] Add independent NumPy reference implementations for EI, PI, UCB,
  qEI/qNEI, knowledge gradient, Thompson sampling, and constrained policies.
- [ ] Add BoTorch/GPyTorch and JAX comparison lanes with matched surrogate
  kernels, normalization, seeds, restarts, precision, and budgets.
- [ ] Compare exact, derivative-observation, sparse, variational, multi-output,
  and fully Bayesian FortML surrogates.
- [ ] Gate input and parameter acquisition derivatives against finite
  differences, adjoint identities, and dense posterior references.
- [ ] Measure simple/cumulative regret, feasible best value, hypervolume,
  constraint violations, model fits, acquisition evaluations, gradient calls,
  memory, transfers, and wall time.
- [ ] Add CPU, OpenACC, transfer-inclusive CUDA, resident CUDA, and typed
  refusal rows. Never report a hidden host fallback as GPU performance.
- [ ] Add asynchronous workers, pending-point fantasies, checkpoint/resume,
  duplicate/failure policies, and deterministic distributed replay.
- [ ] Publish raw CSV/JSON data and plots for every release claim.

## Acceptance

No acquisition policy is called production-ready from one successful objective.
A released lane must match an independent value/gradient or dense-grid oracle,
show reproducible seeded behavior, report optimization and statistical metrics,
and document its complete device contract.
