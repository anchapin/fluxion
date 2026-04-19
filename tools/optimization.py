"""
Optimization Algorithms Module

This module provides advanced optimization algorithms for building energy calibration
and parameter tuning, including Particle Swarm Optimization (PSO).

Available algorithms:
- Particle Swarm Optimization (PSO)
- Genetic Algorithm (GA)
- Simulated Annealing (SA)

Usage:
    from tools.optimization import ParticleSwarmOptimizer

    # Define objective function
    def objective_function(params):
        # params is a list of parameter values
        return fitness_value

    # Set up optimizer
    optimizer = ParticleSwarmOptimizer(
        objective_function,
        n_particles=20,
        n_dimensions=3,
        bounds=[(0.1, 5.0), (15.0, 25.0), (22.0, 32.0)]
    )

    # Run optimization
    best_params, best_fitness = optimizer.optimize(max_iterations=100)
"""

import logging
import random
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) algorithm.

    PSO is a population-based stochastic optimization technique inspired by
    the social behavior of bird flocking or fish schooling.

    Attributes:
        objective_function: Function to minimize
        n_particles: Number of particles in the swarm
        n_dimensions: Number of dimensions in the search space
        bounds: List of tuples specifying (min, max) for each dimension
        inertia_weight: Controls momentum of particles
        cognitive_weight: Controls attraction to personal best
        social_weight: Controls attraction to global best
        velocity_clamp: Maximum allowed velocity
    """

    def __init__(
        self,
        objective_function: Callable[[List[float]], float],
        n_particles: int = 30,
        n_dimensions: int = 3,
        bounds: Optional[List[Tuple[float, float]]] = None,
        inertia_weight: float = 0.7,
        cognitive_weight: float = 1.5,
        social_weight: float = 1.5,
        velocity_clamp: Optional[float] = None,
    ):
        self.objective_function = objective_function
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds or [(0.0, 1.0) for _ in range(n_dimensions)]
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.velocity_clamp = velocity_clamp

        # Initialize particles
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = None

        # History tracking
        self.convergence_history = []
        self.execution_time = 0.0

        self._initialize_swarm()

    def _initialize_swarm(self):
        """Initialize particle positions and velocities."""
        # Initialize positions uniformly within bounds
        self.particles = np.zeros((self.n_particles, self.n_dimensions))
        for i in range(self.n_dimensions):
            min_val, max_val = self.bounds[i]
            self.particles[:, i] = np.random.uniform(
                min_val, max_val, size=self.n_particles
            )

        # Initialize velocities (typically 10-20% of parameter range)
        self.velocities = np.zeros((self.n_particles, self.n_dimensions))
        for i in range(self.n_dimensions):
            min_val, max_val = self.bounds[i]
            range_val = max_val - min_val
            self.velocities[:, i] = np.random.uniform(
                -range_val * 0.2, range_val * 0.2, size=self.n_particles
            )

        # Initialize personal bests
        self.personal_best_positions = self.particles.copy()
        self.personal_best_fitness = np.array(
            [self.objective_function(particle) for particle in self.particles]
        )

        # Initialize global best
        best_idx = np.argmin(self.personal_best_fitness)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[best_idx]

        logger.info(f"Initialized PSO with {self.n_particles} particles")
        logger.info(f"Initial global best fitness: {self.global_best_fitness:.4f}")

    def _update_velocity(self, particle_idx: int):
        """Update velocity for a single particle."""
        # Current velocity
        velocity = self.velocities[particle_idx]

        # Personal best influence
        personal_influence = (
            self.cognitive_weight
            * random.random()
            * (
                self.personal_best_positions[particle_idx]
                - self.particles[particle_idx]
            )
        )

        # Global best influence
        global_influence = (
            self.social_weight
            * random.random()
            * (self.global_best_position - self.particles[particle_idx])
        )

        # Update velocity
        new_velocity = (
            self.inertia_weight * velocity + personal_influence + global_influence
        )

        # Apply velocity clamping if specified
        if self.velocity_clamp is not None:
            new_velocity = np.clip(
                new_velocity, -self.velocity_clamp, self.velocity_clamp
            )

        return new_velocity

    def _update_position(self, particle_idx: int, velocity: np.ndarray) -> np.ndarray:
        """Update position for a single particle with bounds checking."""
        new_position = self.particles[particle_idx] + velocity

        # Apply bounds checking
        for i in range(self.n_dimensions):
            min_val, max_val = self.bounds[i]
            if new_position[i] < min_val:
                new_position[i] = min_val
                # Reverse velocity when hitting boundary
                velocity[i] = -velocity[i] * 0.5
            elif new_position[i] > max_val:
                new_position[i] = max_val
                # Reverse velocity when hitting boundary
                velocity[i] = -velocity[i] * 0.5

        return new_position

    def _update_personal_best(self, particle_idx: int, fitness: float):
        """Update personal best if current position is better."""
        if fitness < self.personal_best_fitness[particle_idx]:
            self.personal_best_fitness[particle_idx] = fitness
            self.personal_best_positions[particle_idx] = self.particles[
                particle_idx
            ].copy()
            return True
        return False

    def _update_global_best(self):
        """Update global best from all personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_position = self.personal_best_positions[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            return True
        return False

    def optimize(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the PSO optimization.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Minimum improvement threshold for convergence
            early_stopping_patience: Number of iterations without improvement before stopping
            verbose: Whether to log progress

        Returns:
            Tuple of (best_parameters, best_fitness)
        """
        start_time = time.time()

        no_improvement_count = 0
        best_fitness_history = []

        for iteration in range(max_iterations):
            iteration_start = time.time()

            # Update velocities and positions
            for i in range(self.n_particles):
                # Update velocity
                self.velocities[i] = self._update_velocity(i)

                # Update position
                self.particles[i] = self._update_position(i, self.velocities[i])

                # Evaluate fitness
                fitness = self.objective_function(self.particles[i])

                # Update personal best
                improved_personal = self._update_personal_best(i, fitness)

            # Update global best
            improved_global = self._update_global_best()

            # Track convergence
            current_best = self.global_best_fitness
            best_fitness_history.append(current_best)

            # Check for improvement
            if improved_global:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Log progress
            if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                iteration_time = time.time() - iteration_start
                logger.info(
                    f"Iteration {iteration:4d}: "
                    f"Best Fitness = {current_best:10.6f} "
                    f"Time = {iteration_time:.3f}s"
                )

            # Check convergence criteria
            if no_improvement_count >= early_stopping_patience:
                if verbose:
                    logger.info(
                        f"Early stopping: No improvement for {no_improvement_count} iterations"
                    )
                break

            # Check if we've reached minimum tolerance
            if len(best_fitness_history) > 1:
                improvement = best_fitness_history[-2] - best_fitness_history[-1]
                if improvement < tolerance:
                    if verbose:
                        logger.info(
                            f"Converged: Improvement {improvement} < tolerance {tolerance}"
                        )
                    break

        self.execution_time = time.time() - start_time
        self.convergence_history = best_fitness_history

        if verbose:
            logger.info(f"Optimization completed in {self.execution_time:.2f} seconds")
            logger.info(f"Final best fitness: {self.global_best_fitness:.6f}")
            logger.info(f"Best parameters: {self.global_best_position}")

        return self.global_best_position.copy(), self.global_best_fitness

    def get_convergence_history(self) -> List[float]:
        """Get the convergence history of the optimization."""
        return self.convergence_history

    def get_execution_time(self) -> float:
        """Get the total execution time in seconds."""
        return self.execution_time

    def get_swarm_positions(self) -> np.ndarray:
        """Get current positions of all particles."""
        return self.particles.copy()

    def get_swarm_velocities(self) -> np.ndarray:
        """Get current velocities of all particles."""
        return self.velocities.copy()


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm (GA) optimizer.

    A population-based optimization algorithm inspired by natural selection.
    """

    def __init__(
        self,
        objective_function: Callable[[List[float]], float],
        n_individuals: int = 50,
        n_dimensions: int = 3,
        bounds: Optional[List[Tuple[float, float]]] = None,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_count: int = 2,
    ):
        self.objective_function = objective_function
        self.n_individuals = n_individuals
        self.n_dimensions = n_dimensions
        self.bounds = bounds or [(0.0, 1.0) for _ in range(n_dimensions)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count

        # Population and fitness
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = None

        # History tracking
        self.convergence_history = []
        self.execution_time = 0.0

        self._initialize_population()

    def _initialize_population(self):
        """Initialize the population."""
        self.population = np.zeros((self.n_individuals, self.n_dimensions))
        for i in range(self.n_dimensions):
            min_val, max_val = self.bounds[i]
            self.population[:, i] = np.random.uniform(
                min_val, max_val, size=self.n_individuals
            )

        # Evaluate initial fitness
        self.fitness = np.array(
            [self.objective_function(individual) for individual in self.population]
        )

        # Find best individual
        best_idx = np.argmin(self.fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        logger.info(f"Initialized GA with {self.n_individuals} individuals")
        logger.info(f"Initial best fitness: {self.best_fitness:.4f}")

    def _selection(self) -> np.ndarray:
        """Select parents using tournament selection."""
        tournament_size = 3
        selected = []

        for _ in range(self.n_individuals):
            # Randomly select tournament_size individuals
            candidates = np.random.choice(
                self.n_individuals, tournament_size, replace=False
            )
            # Select the best one
            best_candidate = candidates[np.argmin(self.fitness[candidates])]
            selected.append(self.population[best_candidate])

        return np.array(selected)

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Perform crossover to create offspring."""
        offspring = np.zeros_like(parents)

        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                break

            parent1, parent2 = parents[i], parents[i + 1]

            if random.random() < self.crossover_rate:
                # Single-point crossover
                crossover_point = random.randint(1, self.n_dimensions - 1)

                child1 = np.concatenate(
                    [parent1[:crossover_point], parent2[crossover_point:]]
                )
                child2 = np.concatenate(
                    [parent2[:crossover_point], parent1[crossover_point:]]
                )

                offspring[i] = child1
                offspring[i + 1] = child2
            else:
                # No crossover - copy parents
                offspring[i] = parent1
                offspring[i + 1] = parent2

        return offspring

    def _mutation(self, offspring: np.ndarray) -> np.ndarray:
        """Apply mutation to offspring."""
        mutated = offspring.copy()

        for i in range(len(mutated)):
            for j in range(self.n_dimensions):
                if random.random() < self.mutation_rate:
                    min_val, max_val = self.bounds[j]
                    # Gaussian mutation centered around current value
                    mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                    mutated[i, j] = np.clip(mutated[i, j] + mutation, min_val, max_val)

        return mutated

    def _apply_bounds(self, population: np.ndarray) -> np.ndarray:
        """Ensure all individuals are within bounds."""
        bounded = population.copy()
        for i in range(self.n_dimensions):
            min_val, max_val = self.bounds[i]
            bounded[:, i] = np.clip(bounded[:, i], min_val, max_val)
        return bounded

    def optimize(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        early_stopping_patience: int = 15,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the Genetic Algorithm optimization.

        Args:
            max_iterations: Maximum number of generations
            tolerance: Minimum improvement threshold for convergence
            early_stopping_patience: Number of generations without improvement before stopping
            verbose: Whether to log progress

        Returns:
            Tuple of (best_parameters, best_fitness)
        """
        start_time = time.time()

        no_improvement_count = 0
        best_fitness_history = []

        for generation in range(max_iterations):
            generation_start = time.time()

            # Selection
            parents = self._selection()

            # Crossover
            offspring = self._crossover(parents)

            # Mutation
            mutated_offspring = self._mutation(offspring)

            # Apply bounds
            new_population = self._apply_bounds(mutated_offspring)

            # Elitism: keep the best individuals from previous generation
            elite_indices = np.argsort(self.fitness)[: self.elitism_count]
            new_population[: self.elitism_count] = self.population[elite_indices]

            # Evaluate fitness
            new_fitness = np.array(
                [self.objective_function(individual) for individual in new_population]
            )

            # Update population and fitness
            self.population = new_population
            self.fitness = new_fitness

            # Update best individual
            current_best_idx = np.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]

            if current_best_fitness < self.best_fitness:
                self.best_individual = self.population[current_best_idx].copy()
                self.best_fitness = current_best_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Track convergence
            best_fitness_history.append(self.best_fitness)

            # Log progress
            if verbose and (generation % 10 == 0 or generation == max_iterations - 1):
                generation_time = time.time() - generation_start
                logger.info(
                    f"Generation {generation:4d}: "
                    f"Best Fitness = {self.best_fitness:10.6f} "
                    f"Time = {generation_time:.3f}s"
                )

            # Check convergence criteria
            if no_improvement_count >= early_stopping_patience:
                if verbose:
                    logger.info(
                        f"Early stopping: No improvement for {no_improvement_count} generations"
                    )
                break

            # Check if we've reached minimum tolerance
            if len(best_fitness_history) > 1:
                improvement = best_fitness_history[-2] - best_fitness_history[-1]
                if improvement < tolerance:
                    if verbose:
                        logger.info(
                            f"Converged: Improvement {improvement} < tolerance {tolerance}"
                        )
                    break

        self.execution_time = time.time() - start_time
        self.convergence_history = best_fitness_history

        if verbose:
            logger.info(f"Optimization completed in {self.execution_time:.2f} seconds")
            logger.info(f"Final best fitness: {self.best_fitness:.6f}")
            logger.info(f"Best parameters: {self.best_individual}")

        return self.best_individual.copy(), self.best_fitness

    def get_convergence_history(self) -> List[float]:
        """Get the convergence history of the optimization."""
        return self.convergence_history

    def get_execution_time(self) -> float:
        """Get the total execution time in seconds."""
        return self.execution_time

    def get_population(self) -> np.ndarray:
        """Get current population."""
        return self.population.copy()


def create_optimization_benchmark():
    """Create a benchmark function for testing optimization algorithms."""

    def sphere_function(params):
        """Sphere function - simple quadratic function for testing."""
        return sum(x**2 for x in params)

    def rosenbrock_function(params):
        """Rosenbrock function - more complex test function."""
        total = 0.0
        for i in range(len(params) - 1):
            total += 100 * (params[i + 1] - params[i] ** 2) ** 2 + (1 - params[i]) ** 2
        return total

    def rastrigin_function(params):
        """Rastrigin function - highly multimodal function."""
        A = 10
        total = A * len(params)
        for x in params:
            total += x**2 - A * np.cos(2 * np.pi * x)
        return total

    return {
        "sphere": sphere_function,
        "rosenbrock": rosenbrock_function,
        "rastrigin": rastrigin_function,
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Optimization Algorithms")
    print("=" * 50)

    # Create benchmark functions
    benchmarks = create_optimization_benchmark()

    # Test PSO on sphere function
    print("\n1. Testing Particle Swarm Optimization on Sphere Function")
    print("-" * 50)

    pso = ParticleSwarmOptimizer(
        objective_function=benchmarks["sphere"],
        n_particles=30,
        n_dimensions=5,
        bounds=[(-5.0, 5.0) for _ in range(5)],
        inertia_weight=0.7,
        cognitive_weight=1.5,
        social_weight=1.5,
    )

    best_params, best_fitness = pso.optimize(
        max_iterations=100, tolerance=1e-8, verbose=True
    )

    print(f"\nPSO Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best fitness: {best_fitness:.8f}")
    print(f"Execution time: {pso.get_execution_time():.3f} seconds")

    # Test GA on Rastrigin function
    print("\n2. Testing Genetic Algorithm on Rastrigin Function")
    print("-" * 50)

    ga = GeneticAlgorithmOptimizer(
        objective_function=benchmarks["rastrigin"],
        n_individuals=40,
        n_dimensions=3,
        bounds=[(-5.12, 5.12) for _ in range(3)],
        mutation_rate=0.15,
        crossover_rate=0.85,
    )

    best_params_ga, best_fitness_ga = ga.optimize(
        max_iterations=80, tolerance=1e-6, verbose=True
    )

    print(f"\nGA Results:")
    print(f"Best parameters: {best_params_ga}")
    print(f"Best fitness: {best_fitness_ga:.8f}")
    print(f"Execution time: {ga.get_execution_time():.3f} seconds")
