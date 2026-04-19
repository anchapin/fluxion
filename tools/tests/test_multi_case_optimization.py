"""
Test suite for multi-case optimization module.

This module tests the MultiCaseOptimizer class and related functions.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from tools.multi_case_optimization import (
    MultiCaseOptimizer,
    create_multi_case_benchmark_function,
)


class TestMultiCaseOptimization(unittest.TestCase):
    """Test cases for multi-case optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.cases = ["900", "600", "960"]
        self.optimizer = MultiCaseOptimizer(self.cases)

    def test_initialization(self):
        """Test MultiCaseOptimizer initialization."""
        self.assertEqual(self.optimizer.case_ids, self.cases)
        self.assertEqual(len(self.optimizer.case_weights), 3)

        # Check that weights sum to 1.0
        total_weight = sum(self.optimizer.case_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

    def test_custom_weights(self):
        """Test custom case weights."""
        custom_weights = {"900": 0.6, "600": 0.3, "960": 0.1}
        optimizer = MultiCaseOptimizer(self.cases, case_weights=custom_weights)

        self.assertEqual(optimizer.case_weights["900"], 0.6)
        self.assertEqual(optimizer.case_weights["600"], 0.3)
        self.assertEqual(optimizer.case_weights["960"], 0.1)

    def test_parameter_bounds(self):
        """Test parameter bounds setup."""
        bounds = self.optimizer.bounds
        self.assertEqual(len(bounds), 3)  # u_value, heating_sp, cooling_sp

        # Check that bounds are reasonable
        self.assertGreater(bounds[0][1], bounds[0][0])  # u_value max > min
        self.assertGreater(bounds[1][1], bounds[1][0])  # heating max > min
        self.assertGreater(bounds[2][1], bounds[2][0])  # cooling max > min

    def test_single_case_evaluation(self):
        """Test single case evaluation."""
        # Test with reasonable parameters
        params = [1.5, 20.0, 26.0]
        fitness = self.optimizer.evaluate_single_case(params, "900")

        # Fitness should be reasonable (not infinity or NaN)
        self.assertFalse(np.isnan(fitness))
        self.assertFalse(np.isinf(fitness))
        self.assertGreaterEqual(fitness, 0)

    def test_invalid_parameters(self):
        """Test evaluation with invalid parameters."""
        # Test with invalid setpoints (cooling < heating)
        invalid_params = [1.5, 25.0, 20.0]
        fitness = self.optimizer.evaluate_single_case(invalid_params, "900")

        # Should return high fitness for invalid parameters
        self.assertGreater(fitness, 1000)

    def test_multi_case_evaluation_serial(self):
        """Test multi-case evaluation in serial mode."""
        optimizer = MultiCaseOptimizer(self.cases, use_parallel=False)
        params = [1.5, 20.0, 26.0]

        fitness = optimizer.evaluate_multi_case(params)

        self.assertFalse(np.isnan(fitness))
        self.assertFalse(np.isinf(fitness))
        self.assertGreaterEqual(fitness, 0)

    def test_multi_case_evaluation_parallel(self):
        """Test multi-case evaluation in parallel mode."""
        params = [1.5, 20.0, 26.0]

        fitness = self.optimizer.evaluate_multi_case(params)

        self.assertFalse(np.isnan(fitness))
        self.assertFalse(np.isinf(fitness))
        self.assertGreaterEqual(fitness, 0)

    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        params = [1.5, 20.0, 26.0]
        metrics = self.optimizer.get_case_performance_metrics(params)

        # Should have metrics for all cases
        self.assertEqual(len(metrics), 3)
        self.assertIn("900", metrics)
        self.assertIn("600", metrics)
        self.assertIn("960", metrics)

        # Check metric structure
        for case_id, case_metrics in metrics.items():
            self.assertIn("parameters", case_metrics)
            self.assertIn("energy", case_metrics)
            self.assertIn("comfort", case_metrics)
            self.assertIn("construction", case_metrics)

    def test_comfort_score_calculation(self):
        """Test comfort score calculations."""
        # Ideal setpoints
        ideal_heating = self.optimizer._calculate_setpoint_comfort(20.0, "heating")
        ideal_cooling = self.optimizer._calculate_setpoint_comfort(25.0, "cooling")

        self.assertEqual(ideal_heating, 100.0)
        self.assertEqual(ideal_cooling, 100.0)

        # Too cold heating setpoint
        cold_heating = self.optimizer._calculate_setpoint_comfort(16.0, "heating")
        self.assertLess(cold_heating, 100.0)
        self.assertGreater(cold_heating, 0.0)

        # Too hot cooling setpoint
        hot_cooling = self.optimizer._calculate_setpoint_comfort(28.0, "cooling")
        self.assertLess(hot_cooling, 100.0)
        self.assertGreater(hot_cooling, 0.0)

    def test_optimization_report(self):
        """Test optimization report generation."""
        params = [1.5, 20.0, 26.0]
        report = self.optimizer.create_optimization_report(params)

        self.assertIn("MULTI-CASE OPTIMIZATION REPORT", report)
        self.assertIn("Case 900", report)
        self.assertIn("Case 600", report)
        self.assertIn("Case 960", report)
        self.assertIn("OVERALL METRICS", report)

    def test_benchmark_function(self):
        """Test benchmark function creation."""
        benchmark = create_multi_case_benchmark_function()

        # Test with reasonable parameters
        params = [1.5, 20.0, 26.0]
        fitness = benchmark(params)

        self.assertFalse(np.isnan(fitness))
        self.assertFalse(np.isinf(fitness))
        self.assertGreaterEqual(fitness, 0)

        # Test with different parameters
        params2 = [2.0, 19.0, 27.0]
        fitness2 = benchmark(params2)

        # Different parameters should give different fitness
        self.assertNotEqual(fitness, fitness2)

    @patch("tools.multi_case_optimization.ParticleSwarmOptimizer")
    def test_optimize_pso(self, mock_pso):
        """Test optimization with PSO."""
        # Setup mock optimizer
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = ([1.5, 20.0, 26.0], 10.0)
        mock_pso.return_value = mock_instance

        # Run optimization
        best_params, best_fitness = self.optimizer.optimize(
            optimizer_type="pso", max_iterations=10, population_size=10, verbose=False
        )

        # Verify optimization was called
        mock_pso.assert_called_once()
        self.assertEqual(best_params, [1.5, 20.0, 26.0])
        self.assertEqual(best_fitness, 10.0)

    @patch("tools.multi_case_optimization.GeneticAlgorithmOptimizer")
    def test_optimize_ga(self, mock_ga):
        """Test optimization with GA."""
        # Setup mock optimizer
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = ([1.8, 19.5, 26.5], 15.0)
        mock_ga.return_value = mock_instance

        # Run optimization
        best_params, best_fitness = self.optimizer.optimize(
            optimizer_type="ga", max_iterations=10, population_size=10, verbose=False
        )

        # Verify optimization was called
        mock_ga.assert_called_once()
        self.assertEqual(best_params, [1.8, 19.5, 26.5])
        self.assertEqual(best_fitness, 15.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
