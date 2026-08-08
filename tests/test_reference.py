import unittest

import numpy as np

from fortbo_bench.evidence import MetricRecorder, MetricRow, WorkerEvent, replay_workers
from fortbo_bench.reference import (
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
    lower_confidence_bound,
    q_expected_improvement,
    q_noisy_expected_improvement,
    thompson_indices,
    pending_fantasy,
    zdt,
    zdt_front,
)


class ReferenceFunctionsTest(unittest.TestCase):
    def test_known_optima_and_independent_gradients(self):
        generator = np.random.default_rng(1729)
        for name, spec in FUNCTION_SPECS.items():
            with self.subTest(function=name):
                self.assertAlmostEqual(float(spec.evaluate(spec.optimum)[0]),
                                       spec.optimum_value, delta=5.0e-4)
                points = generator.uniform(spec.lower + 0.15*(spec.upper-spec.lower),
                                           spec.upper - 0.15*(spec.upper-spec.lower),
                                           size=(3, spec.dimension))
                error = check_gradient(spec.value, spec.gradient, points)
                # The Rosenbrock scale is large enough that its Richardson
                # residual is naturally looser than the other functions.
                tolerance = 5.0e-5 if name == "rosenbrock" else 2.0e-7
                self.assertLess(error, tolerance)

    def test_constraint_and_pareto_fixtures_keep_structure(self):
        from fortbo_bench.reference import constrained_fixtures
        fixtures = constrained_fixtures()
        point = np.array([[0.5, 0.5]])
        for fixture in fixtures.values():
            self.assertEqual(fixture["value"](point).shape, (1,))
            self.assertEqual(fixture["constraint"](point).shape, (1,))
        first = zdt_front(np.array([0.0, 0.25, 1.0]), kind=1)
        attained = zdt(np.column_stack((first[:, 0], np.zeros(3))), kind=1)
        np.testing.assert_allclose(attained, first)
        # A sampled ZDT point must not dominate an analytic front point.
        sample = np.random.default_rng(4).uniform(0.0, 1.0, size=(100, 4))
        values = zdt(sample, kind=2)
        front = zdt_front(np.linspace(0.0, 1.0, 201), kind=2)
        self.assertFalse(np.any(np.all(values[:, None, :] <= front[None, :, :], axis=2) &
                                np.any(values[:, None, :] < front[None, :, :], axis=2)))


class AcquisitionReferenceTest(unittest.TestCase):
    def test_scalar_acquisition_definitions(self):
        mean = np.array([-1.0, 0.0, 1.0])
        sd = np.array([0.0, 1.0, 2.0])
        values = acquisition_values(mean, sd, best=0.0, beta=2.0)
        self.assertEqual(values["ei"][0], 1.0)
        np.testing.assert_allclose(values["ucb"], mean-2.0*sd)
        self.assertGreater(values["pi"][0], 0.99)

    def test_acquisition_parameter_products_have_an_independent_oracle(self):
        point = {"mean": -0.35, "sd": 0.8, "best": 0.1, "xi": 0.05}
        products = expected_improvement_derivatives(**point)
        for parameter in ("mean", "sd", "best", "xi"):
            h = 1.0e-6
            plus = dict(point); minus = dict(point)
            plus[parameter] += h; minus[parameter] -= h
            numerical = (expected_improvement(np.array([plus["mean"]]),
                                               np.array([plus["sd"]]), plus["best"], plus["xi"])[0] -
                         expected_improvement(np.array([minus["mean"]]),
                                               np.array([minus["sd"]]), minus["best"], minus["xi"])[0])/(2*h)
            self.assertAlmostEqual(products[parameter], float(numerical), places=7)

    def test_batch_mc_kg_thompson_and_constraints(self):
        candidate = np.array([[0.0, 1.0], [-1.0, 0.5], [1.0, 2.0]])
        incumbent = np.array([[0.2, 0.4], [-0.5, 0.4], [0.3, 0.8]])
        self.assertAlmostEqual(q_expected_improvement(candidate, 0.0),
                               np.mean([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(q_noisy_expected_improvement(candidate, incumbent),
                               np.mean([0.2, 0.5, 0.0]))
        self.assertAlmostEqual(knowledge_gradient(np.array([1.0, 2.0]),
                                                   np.array([[0.5, 2.0], [0.2, 3.0]])), 0.65)
        np.testing.assert_array_equal(thompson_indices(np.array([[1.0, 0.0], [0.0, 2.0]])), [1, 0])
        np.testing.assert_allclose(constrained_acquisition(np.array([2.0, 4.0]),
                                                           np.array([0.5, 0.25]),
                                                           np.array([1.0, 2.0])), [1.0, 0.5])
        self.assertAlmostEqual(effective_sample_size(np.ones(4)), 4.0)


class PosteriorReferenceTest(unittest.TestCase):
    def test_gp_gradient_is_checked_against_the_posterior_itself(self):
        x = np.array([[-1.0], [-0.2], [0.6], [1.3]])
        y = np.sin(x[:, 0])
        query = np.array([[-0.7], [0.1], [0.9]])
        result = gp_posterior(x, y, query)
        for i, point in enumerate(query):
            h = 1.0e-6
            plus = gp_posterior(x, y, point[None, :] + h)["mean"][0]
            minus = gp_posterior(x, y, point[None, :] - h)["mean"][0]
            self.assertAlmostEqual(result["mean_gradient"][i, 0], (plus-minus)/(2*h), places=7)
        gradients = np.cos(x)
        derivative = gp_posterior_derivative_observations(x, y, gradients, query)
        self.assertFalse(np.allclose(derivative["mean"], result["mean"]))


class EvidenceTest(unittest.TestCase):
    def test_metrics_keep_refusals_and_regret_separate(self):
        recorder = MetricRecorder(optimum=0.0)
        recorder.add(MetricRow(0, 2.0, True, 0.0, 1.0, 10, 1, 1, None, 100, 0, 0.1))
        recorder.add(MetricRow(1, None, None, None, 0.0, 0, 0, 0, None, None, None, None,
                               lane="cuda-resident", status="refused", refusal_reason="no device"))
        summary = recorder.summary()
        self.assertEqual(summary["refused"], 1)
        self.assertEqual(summary["best_feasible_value"], 2.0)
        self.assertEqual(summary["simple_regret"], 2.0)
        with self.assertRaises(ValueError):
            recorder.add(MetricRow(2, None, None, None, 0.0, 0, 0, 0, None, None, None, 0.1,
                                   status="refused", refusal_reason="bad timing"))

    def test_worker_replay_charges_failures_and_retries(self):
        result = replay_workers([
            WorkerEvent(0, 0, 0, "a", 2.0, 3.0, failed=True),
            WorkerEvent(0, 1, 0, "b", 1.0, 1.0),
        ])
        self.assertEqual([row["job"] for row in result["completed"]], ["b", "a"])
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(result["total_attempt_cost"], 7.0)

    def test_checkpoint_resume_and_pending_fantasies_are_explicit(self):
        recorder = MetricRecorder(optimum=0.0)
        recorder.add(MetricRow(0, 1.0, True, 0.0, 1.0, 1, 0, 1, None, None, None, 0.1))
        resumed = MetricRecorder.from_checkpoint(recorder.checkpoint())
        self.assertEqual(resumed.summary(), recorder.summary())
        np.testing.assert_allclose(pending_fantasy(np.array([1.0, 2.0]), 0.5), [1.0, 2.0])
        np.testing.assert_allclose(pending_fantasy(np.array([1.0, 2.0]), 0.5, "incumbent"), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
