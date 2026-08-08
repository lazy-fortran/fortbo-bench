"""The reference material FortBO is built against.

One place to state what a source *is* and why it is needed, so that a fetch is
reproducible and an implementation can be traced back to the definition it was
written from.

Nothing here is a dependency of FortBO. These are read, not ported: the
first-principles mandate stands, and every entry carries the licence it is
distributed under so that stays checkable.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Paper:
    """An arXiv paper, fetched as PDF and as extractable text."""

    key: str
    arxiv_id: str
    title: str
    #: What FortBO needs from it, specifically. Not a summary of the paper.
    needed_for: str
    #: Where the result lands in FortBO, so a reader can check the two agree.
    consumed_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class Repository:
    """A third-party source tree, fetched shallow and read for its definitions."""

    key: str
    url: str
    #: Pinned so a re-fetch reads the same code a claim was checked against.
    ref: str
    licence: str
    needed_for: str
    #: Restrict the checkout to the paths actually worth reading. These trees
    #: are large and the interesting part is usually a few files.
    sparse_paths: tuple[str, ...] = ()
    consumed_by: tuple[str, ...] = ()


PAPERS: tuple[Paper, ...] = (
    Paper(
        key="turbo",
        arxiv_id="1910.01739",
        title="Scalable Global Optimization via Local Bayesian Optimization",
        needed_for=(
            "The TuRBO trust-region constants and update rule, the implicit "
            "bandit across regions, and the reported experimental setup for "
            "Ackley-200, the 60D rover, and 14D robot pushing."
        ),
        consumed_by=(
            "fortbo/src/fortbo_trust_region.f90",
            "fortbo/src/fortbo_turbo_driver.f90",
        ),
    ),
    Paper(
        key="newton-bo",
        arxiv_id="2508.18423",
        title="Enhancing Trust-Region Bayesian Optimization via Newton Methods",
        needed_for=(
            "DTuRBO mode 2: the local quadratic model built from posterior "
            "derivatives, the truncated-normal exploration weight, and the "
            "ratio-test radius rule that replaces the success/failure counters."
        ),
        consumed_by=("fortbo/src/fortbo_dturbo.f90",),
    ),
    Paper(
        key="pes",
        arxiv_id="1406.2541",
        title="Predictive Entropy Search for Efficient Global Optimization",
        needed_for=(
            "The exact conditioning PES applies at x*, which a single-query "
            "reduction provably cannot reproduce. FortBO currently has the "
            "ingredient and not the estimator, and this is the source that "
            "says what the difference is."
        ),
        consumed_by=("fortbo/src/fortbo_pes.f90",),
    ),
    Paper(
        key="mes",
        arxiv_id="1703.01968",
        title="Max-value Entropy Search for Efficient Bayesian Optimization",
        needed_for=(
            "The MES closed form and, in particular, its truncation direction. "
            "FortBO minimizes and the published form maximizes, so the sign of "
            "the second term differs."
        ),
        consumed_by=("fortbo/src/fortbo_entropy.f90",),
    ),
    Paper(
        key="dkg",
        arxiv_id="1703.04389",
        title="Bayesian Optimization with Gradients",
        needed_for=(
            "The derivative-enabled knowledge gradient, and the envelope "
            "construction for the expected minimum of affine functions of one "
            "normal."
        ),
        consumed_by=("fortbo/src/fortbo_knowledge_gradient.f90",),
    ),
    Paper(
        key="hvarfner-prior",
        arxiv_id="2402.02229",
        title="Vanilla Bayesian Optimization Performs Great in High Dimensions",
        needed_for=(
            "The dimension-scaled lengthscale prior, and the claim about what "
            "a correctly-scaled vanilla GP achieves in high dimensions — which "
            "is the baseline any trust-region result has to beat."
        ),
        consumed_by=("fortbo/src/fortbo_fortml.f90",),
    ),
    Paper(
        key="logei",
        arxiv_id="2310.20708",
        title="Unexpected Improvements to Expected Improvement",
        needed_for=(
            "The numerically stable log-EI formulation and the asymptotic "
            "regime where the naive expression cancels to zero."
        ),
        consumed_by=("fortbo/src/fortbo_acquisition.f90",),
    ),
    Paper(
        key="gardner-constrained",
        arxiv_id="1403.5607",
        title="Bayesian Optimization with Inequality Constraints",
        needed_for=(
            "The constrained-EI weighting and the simulation benchmark used as "
            "a constrained fixture."
        ),
        consumed_by=(
            "fortbo/src/fortbo_constrained.f90",
            "fortbo/src/fortbo_fixtures.f90",
        ),
    ),
    Paper(
        key="student-t-process",
        arxiv_id="1402.4306",
        title="Student-t Processes as Alternatives to Gaussian Processes",
        needed_for=(
            "The Student-t process: its predictive marginals, the degrees-of-"
            "freedom update on conditioning, and the claim about which of its "
            "properties actually differ from a GP's. FortML implements it as a "
            "generic model; FortBO only adapts it."
        ),
        consumed_by=("fortml/src/gp/fortml_student_t_process.f90",),
    ),
    Paper(
        key="svgp",
        arxiv_id="1309.6835",
        title="Gaussian Processes for Big Data",
        needed_for=(
            "The stochastic variational sparse GP whose predictive marginals "
            "FortBO's sparse adapter presents, and the sense in which its "
            "variance is not the exact posterior's."
        ),
        consumed_by=("fortbo/src/fortbo_fortml_sparse.f90",),
    ),
    # Heteroscedastic GP regression (Lazaro-Gredilla and Titsias, ICML 2011)
    # has no arXiv posting, so it cannot be fetched here. Recorded rather than
    # replaced with a guessed identifier: an arXiv id that resolves to *some*
    # PDF is worse than none, because it looks fetched. An earlier revision of
    # this file carried a guessed id that resolved to an unrelated Technical
    # Physics Letters article, which is why `fetch_provenance.py` now verifies
    # the title of what it downloads.
)


REPOSITORIES: tuple[Repository, ...] = (
    Repository(
        key="turbo-reference",
        url="https://github.com/uber-research/TuRBO",
        ref="master",
        licence="Uber Non-Commercial (read only; not vendored, not ported)",
        needed_for=(
            "The authors' own constants and restart bookkeeping, to check "
            "FortBO's independently-written trust-region state machine agrees "
            "on the numbers the paper leaves implicit."
        ),
        consumed_by=("fortbo/src/fortbo_trust_region.f90",),
    ),
    Repository(
        key="botorch",
        url="https://github.com/pytorch/botorch",
        ref="main",
        licence="MIT",
        needed_for=(
            "The reference implementations of qEI/qNEI/qKG and the acquisition "
            "API shape, and the benchmark harness FortBO's cross-framework "
            "comparison has to match on setup."
        ),
        sparse_paths=(
            "botorch/acquisition",
            "botorch/test_functions",
            "botorch/utils",
        ),
        consumed_by=(
            "fortbo/src/fortbo_batch.f90",
            "fortbo/src/fortbo_knowledge_gradient.f90",
        ),
    ),
)
