import unittest
import numpy as np
from src.svm_scratch import LinearSVMScratch


class SVMTests(unittest.TestCase):
    def test_objective_uses_updated_margins(self):
        X = np.array([[-2.0, 0.1], [-1.0, 0.2], [1.0, 0.3], [2.0, 0.4]])
        y = np.array([-1.0, -1.0, 1.0, 1.0])
        initial_w = np.random.default_rng(42).normal(scale=0.01, size=2)
        # All initial margins are below one in this fixture.
        expected_w = initial_w - 0.01 * (initial_w - (X * y[:, None]).sum(axis=0))
        expected_b = 0.01 * y.sum()
        expected_objective = (
            0.5 * (expected_w @ expected_w)
            + np.maximum(0, 1 - y * (X @ expected_w + expected_b)).sum()
        )
        fitted = LinearSVMScratch(max_iter=1).fit(X, y)
        self.assertAlmostEqual(fitted.objective_history_[-1], expected_objective)

    def test_best_parameters_match_reported_objective(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([-1, -1, 1, 1])
        model = LinearSVMScratch(max_iter=1000).fit(X, y)
        objective = (
            0.5 * (model.w_ @ model.w_)
            + np.maximum(0, 1 - y * model.decision_function(X)).sum()
        )
        self.assertAlmostEqual(objective, min(model.objective_history_))
        np.testing.assert_array_equal(model.predict(X), y)

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            LinearSVMScratch().fit([[1], [2]], [0, 1])
        with self.assertRaises(ValueError):
            LinearSVMScratch().fit([[np.nan], [2]], [-1, 1])
        with self.assertRaises(RuntimeError):
            LinearSVMScratch().predict([[1]])


if __name__ == "__main__":
    unittest.main()
