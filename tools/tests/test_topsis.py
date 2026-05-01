"""
Test suite for TOPSIS multi-criteria decision making.

This module tests the TOPSIS implementation for Pareto optimization.
"""

import unittest

import numpy as np

from tools.topsis import (
    ObjectiveWeights,
    TOPSISResult,
    TOPSISSolver,
    create_pareto_visualization_data,
)


class TestTOPSISAlgorithm(unittest.TestCase):
    """Test cases for TOPSIS algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.weights = ObjectiveWeights(ec=0.35, tdhp=0.25, lcc=0.25, lcco2=0.15)
        self.solver = TOPSISSolver(self.weights)

        self.simple_pareto = [
            [100.0, 5.2, 50000, 72.3],
            [120.0, 3.1, 45000, 85.0],
            [95.0, 6.8, 55000, 65.0],
        ]

        self.larger_pareto = [
            [100.0, 5.2, 50000, 72.3],
            [120.0, 3.1, 45000, 85.0],
            [95.0, 6.8, 55000, 65.0],
            [110.0, 4.5, 48000, 75.0],
            [105.0, 4.0, 52000, 70.0],
        ]

    def test_weights_validation(self):
        """Test that weights are validated."""
        valid_weights = ObjectiveWeights(ec=0.25, tdhp=0.25, lcc=0.25, lcco2=0.25)
        self.assertEqual(valid_weights.to_list(), [0.25, 0.25, 0.25, 0.25])

        with self.assertRaises(ValueError):
            ObjectiveWeights(ec=0.5, tdhp=0.5, lcc=0.5, lcco2=0.5)

        with self.assertRaises(ValueError):
            ObjectiveWeights(ec=-0.1, tdhp=0.4, lcc=0.4, lcco2=0.3)

    def test_weights_to_list(self):
        """Test converting weights to list."""
        weights_list = self.weights.to_list()
        self.assertEqual(len(weights_list), 4)
        self.assertAlmostEqual(sum(weights_list), 1.0, places=6)

    def test_topsis_solver_initialization(self):
        """Test TOPSIS solver initialization."""
        solver = TOPSISSolver(self.weights)
        self.assertEqual(solver.weights, self.weights)

        solver_default = TOPSISSolver()
        self.assertEqual(solver_default.weights.to_list(), [0.25, 0.25, 0.25, 0.25])

    def test_topsis_solve_simple(self):
        """Test TOPSIS on simple Pareto frontier."""
        result = self.solver.solve(self.simple_pareto)

        self.assertIsInstance(result, TOPSISResult)
        self.assertIn(result.best_index, [0, 1, 2])
        self.assertEqual(len(result.closeness_scores), 3)
        self.assertEqual(len(result.rankings), 3)
        self.assertEqual(result.weights, self.weights)

    def test_topsis_solve_larger(self):
        """Test TOPSIS on larger Pareto frontier."""
        result = self.solver.solve(self.larger_pareto)

        self.assertEqual(len(result.closeness_scores), 5)
        self.assertEqual(len(result.rankings), 5)
        self.assertEqual(result.best_alternative.shape, (4,))

        self.assertTrue(np.allclose(result.normalized_matrix.shape, (5, 4)))

    def test_topsis_ranking(self):
        """Test that rankings are returned correctly."""
        result = self.solver.solve(self.larger_pareto)

        self.assertEqual(len(result.rankings), 5)
        self.assertEqual(set(result.rankings), {1, 2, 3, 4, 5})
        self.assertEqual(result.rankings.count(1), 1)
        self.assertEqual(result.rankings[result.best_index], 1)

    def test_topsis_get_rankings(self):
        """Test get_rankings convenience method."""
        result = self.solver.solve(self.larger_pareto)
        rankings = self.solver.get_rankings(self.larger_pareto)

        self.assertIsInstance(rankings, list)
        self.assertEqual(len(rankings), 5)
        self.assertEqual(rankings[result.best_index], 1)

    def test_topsis_get_best_alternative(self):
        """Test get_best_alternative convenience method."""
        idx, alternative = self.solver.get_best_alternative(self.larger_pareto)

        self.assertIsInstance(idx, int)
        self.assertIsInstance(alternative, list)
        self.assertEqual(len(alternative), 4)
        self.assertEqual(alternative, self.larger_pareto[idx])

    def test_topsis_closeness_scores_sum_to_one(self):
        """Test that closeness scores are between 0 and 1."""
        result = self.solver.solve(self.larger_pareto)

        for score in result.closeness_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_topsis_ideal_solutions(self):
        """Test that ideal solutions are calculated correctly."""
        result = self.solver.solve(self.larger_pareto)

        self.assertEqual(result.ideal_positive.shape, (4,))
        self.assertEqual(result.ideal_negative.shape, (4,))
        self.assertEqual(len(result.ideal_positive), 4)
        self.assertEqual(len(result.ideal_negative), 4)

    def test_topsis_matrix_normalization(self):
        """Test that normalization preserves relative relationships."""
        result = self.solver.solve(self.larger_pareto)

        normalized = result.normalized_matrix
        self.assertEqual(normalized.shape, (5, 4))
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))

    def test_topsis_invalid_matrix_dimensions(self):
        """Test that invalid matrix dimensions raise error."""
        matrix_3_cols = [[1, 2, 3], [4, 5, 6]]

        with self.assertRaises(ValueError):
            self.solver.solve(matrix_3_cols)

        matrix_1_row = [[1, 2, 3, 4]]

        with self.assertRaises(ValueError):
            self.solver.solve(matrix_1_row)

    def test_topsis_invalid_matrix_ndim(self):
        """Test that non-2D matrix raises error."""
        single_value = [1, 2, 3, 4]

        with self.assertRaises(ValueError):
            self.solver.solve(single_value)

    def test_visualization_data_creation(self):
        """Test creation of visualization data."""
        result = self.solver.solve(self.larger_pareto)
        viz_data = create_pareto_visualization_data(self.larger_pareto, result)

        self.assertIn("pareto_points", viz_data)
        self.assertIn("topsis_selected_index", viz_data)
        self.assertIn("topsis_selected_point", viz_data)
        self.assertIn("closeness_scores", viz_data)
        self.assertIn("rankings", viz_data)
        self.assertIn("ideal_positive", viz_data)
        self.assertIn("ideal_negative", viz_data)
        self.assertIn("weights", viz_data)

        self.assertEqual(viz_data["topsis_selected_index"], result.best_index)
        self.assertEqual(viz_data["pareto_points"], self.larger_pareto)

    def test_visualization_data_with_parameters(self):
        """Test visualization data with parameter sets."""
        result = self.solver.solve(self.larger_pareto)
        parameter_sets = [
            [1.5, 20.0, 26.0],
            [2.0, 21.0, 25.0],
            [1.8, 19.0, 27.0],
            [1.6, 22.0, 24.0],
            [1.7, 20.5, 25.5],
        ]

        viz_data = create_pareto_visualization_data(
            self.larger_pareto, result, parameter_sets
        )

        self.assertIn("selected_parameters", viz_data)
        self.assertEqual(
            viz_data["selected_parameters"], parameter_sets[result.best_index]
        )

    def test_visualization_data_objective_ranges(self):
        """Test objective ranges in visualization data."""
        result = self.solver.solve(self.larger_pareto)
        viz_data = create_pareto_visualization_data(self.larger_pareto, result)

        self.assertIn("objective_ranges", viz_data)
        self.assertIn("EC", viz_data["objective_ranges"])
        self.assertIn("TDHP", viz_data["objective_ranges"])
        self.assertIn("LCC", viz_data["objective_ranges"])
        self.assertIn("LCCO2", viz_data["objective_ranges"])


class TestTOPSISIntegration(unittest.TestCase):
    """Integration tests for TOPSIS with Pareto optimization."""

    def test_topsis_with_equal_weights(self):
        """Test TOPSIS with equal weights."""
        weights = ObjectiveWeights(ec=0.25, tdhp=0.25, lcc=0.25, lcco2=0.25)
        solver = TOPSISSolver(weights)

        pareto = [
            [100.0, 5.0, 50000, 70.0],
            [90.0, 7.0, 55000, 60.0],
            [110.0, 3.0, 45000, 80.0],
        ]

        result = solver.solve(pareto)

        self.assertIn(result.best_index, [0, 1, 2])
        self.assertEqual(len(result.closeness_scores), 3)

    def test_topsis_with_dominant_weight(self):
        """Test TOPSIS when one objective has very high weight."""
        weights = ObjectiveWeights(ec=0.80, tdhp=0.05, lcc=0.10, lcco2=0.05)
        solver = TOPSISSolver(weights)

        pareto = [
            [100.0, 5.0, 50000, 70.0],
            [80.0, 8.0, 60000, 90.0],
            [120.0, 2.0, 40000, 50.0],
        ]

        result = solver.solve(pareto)

        self.assertEqual(result.best_index, 1)

    def test_topsis_closeness_score_ordering(self):
        """Test that closeness scores are ordered correctly."""
        weights = ObjectiveWeights(ec=0.25, tdhp=0.25, lcc=0.25, lcco2=0.25)
        solver = TOPSISSolver(weights)

        pareto = [
            [100.0, 5.0, 50000, 70.0],
            [90.0, 6.0, 45000, 65.0],
            [95.0, 5.5, 47500, 67.5],
        ]

        result = solver.solve(pareto)

        max_closeness_idx = np.argmax(result.closeness_scores)
        self.assertEqual(max_closeness_idx, result.best_index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
