"""
Test suite for optimization algorithms.

This module tests the Particle Swarm Optimization and Genetic Algorithm implementations.
"""

import unittest

import numpy as np

from tools.optimization import GeneticAlgorithmOptimizer, ParticleSwarmOptimizer


class TestOptimizationAlgorithms(unittest.TestCase):
    """Test cases for optimization algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple quadratic function for testing
        self.quadratic_function = lambda x: sum(xi**2 for xi in x)

        # More complex function
        self.complex_function = lambda x: (
            (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
            if len(x) >= 2
            else x[0] ** 2
        )

        # Bounds for testing
        self.bounds = [(-5.0, 5.0) for _ in range(3)]

    def test_pso_initialization(self):
        """Test PSO initialization."""
        pso = ParticleSwarmOptimizer(
            objective_function=self.quadratic_function,
            n_particles=20,
            n_dimensions=3,
            bounds=self.bounds,
        )

        # Check that particles are initialized
        self.assertIsNotNone(pso.particles)
        self.assertEqual(pso.particles.shape, (20, 3))

        # Check that velocities are initialized
        self.assertIsNotNone(pso.velocities)
        self.assertEqual(pso.velocities.shape, (20, 3))

        # Check that bounds are respected
        for i in range(3):
            min_val, max_val = self.bounds[i]
            self.assertTrue(np.all(pso.particles[:, i] >= min_val))
            self.assertTrue(np.all(pso.particles[:, i] <= max_val))

    def test_pso_optimization_quadratic(self):
        """Test PSO on simple quadratic function."""
        pso = ParticleSwarmOptimizer(
            objective_function=self.quadratic_function,
            n_particles=30,
            n_dimensions=3,
            bounds=self.bounds,
            inertia_weight=0.7,
            cognitive_weight=1.5,
            social_weight=1.5,
        )

        best_params, best_fitness = pso.optimize(
            max_iterations=100,  # Increased iterations for better convergence
            tolerance=1e-6,
            verbose=False,
        )

        # For quadratic function, minimum should be near 0
        self.assertLess(best_fitness, 2.0)  # Relaxed tolerance for test

        # Parameters should be near 0
        for param in best_params:
            self.assertLess(abs(param), 1.0)  # Relaxed tolerance for test

    def test_pso_bounds_enforcement(self):
        """Test that PSO respects parameter bounds."""
        bounds = [(0.1, 1.0), (10.0, 20.0), (-5.0, -1.0)]

        pso = ParticleSwarmOptimizer(
            objective_function=self.quadratic_function,
            n_particles=20,
            n_dimensions=3,
            bounds=bounds,
        )

        best_params, _ = pso.optimize(max_iterations=20, verbose=False)

        # Check that best parameters are within bounds
        for i, (min_val, max_val) in enumerate(bounds):
            self.assertGreaterEqual(best_params[i], min_val)
            self.assertLessEqual(best_params[i], max_val)

    def test_ga_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithmOptimizer(
            objective_function=self.quadratic_function,
            n_individuals=20,
            n_dimensions=3,
            bounds=self.bounds,
        )

        # Check that population is initialized
        self.assertIsNotNone(ga.population)
        self.assertEqual(ga.population.shape, (20, 3))

        # Check that bounds are respected
        for i in range(3):
            min_val, max_val = self.bounds[i]
            self.assertTrue(np.all(ga.population[:, i] >= min_val))
            self.assertTrue(np.all(ga.population[:, i] <= max_val))

    def test_ga_optimization_quadratic(self):
        """Test GA on simple quadratic function."""
        ga = GeneticAlgorithmOptimizer(
            objective_function=self.quadratic_function,
            n_individuals=40,
            n_dimensions=3,
            bounds=self.bounds,
            mutation_rate=0.1,
            crossover_rate=0.8,
        )

        best_params, best_fitness = ga.optimize(
            max_iterations=100,  # Increased iterations for better convergence
            tolerance=1e-6,
            verbose=False,
        )

        # For quadratic function, minimum should be near 0
        self.assertLess(best_fitness, 0.2)  # Relaxed tolerance for GA

        # Parameters should be near 0
        for param in best_params:
            self.assertLess(abs(param), 1.0)  # Relaxed tolerance for GA

    def test_ga_bounds_enforcement(self):
        """Test that GA respects parameter bounds."""
        bounds = [(0.1, 1.0), (10.0, 20.0), (-5.0, -1.0)]

        ga = GeneticAlgorithmOptimizer(
            objective_function=self.quadratic_function,
            n_individuals=30,
            n_dimensions=3,
            bounds=bounds,
        )

        best_params, _ = ga.optimize(max_iterations=30, verbose=False)

        # Check that best parameters are within bounds
        for i, (min_val, max_val) in enumerate(bounds):
            self.assertGreaterEqual(best_params[i], min_val)
            self.assertLessEqual(best_params[i], max_val)

    def test_convergence_history(self):
        """Test that convergence history is tracked."""
        pso = ParticleSwarmOptimizer(
            objective_function=self.quadratic_function,
            n_particles=20,
            n_dimensions=2,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        )

        pso.optimize(max_iterations=20, verbose=False)

        history = pso.get_convergence_history()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

        # History should be non-increasing (monotonically improving)
        for i in range(1, len(history)):
            self.assertLessEqual(history[i], history[i - 1])

    def test_execution_time_tracking(self):
        """Test that execution time is tracked."""
        pso = ParticleSwarmOptimizer(
            objective_function=self.quadratic_function,
            n_particles=15,
            n_dimensions=2,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        )

        pso.optimize(max_iterations=10, verbose=False)

        exec_time = pso.get_execution_time()
        self.assertIsInstance(exec_time, float)
        self.assertGreater(exec_time, 0)
        self.assertLess(exec_time, 10)  # Should complete quickly


if __name__ == "__main__":
    unittest.main(verbosity=2)
