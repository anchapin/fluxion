"""
Multi-Case Optimization Module

This module extends optimization capabilities to handle multiple ASHRAE 140
cases simultaneously. It enables calibration across diverse building types
and conditions, improving the robustness of building energy models.

Features:
- Simultaneous optimization across multiple ASHRAE 140 cases
- Weighted objective functions for different case priorities
- Multi-objective optimization support
- Parallel evaluation of cases
- Comprehensive performance metrics

Usage:
    from tools.multi_case_optimization import MultiCaseOptimizer
    from tools.ashrae_140_reference import ASHRAE140ReferenceData

    # Define cases to optimize
    cases = ["900", "600", "960"]

    # Create multi-case optimizer
    optimizer = MultiCaseOptimizer(cases)

    # Define objective function
    def objective_function(params):
        return optimizer.evaluate_multi_case(params)

    # Run optimization
    best_params, best_fitness = optimizer.optimize(objective_function)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Local imports
try:
    from tools.ashrae_140_reference import (
        ASHRAE140ReferenceData,
        create_ashrae_140_calibration_targets,
    )
    from tools.optimization import GeneticAlgorithmOptimizer, ParticleSwarmOptimizer
    from tools.parameter_validation import BuildingParameterValidator
except ImportError:
    # Fallback for testing without local imports
    ParticleSwarmOptimizer = None
    GeneticAlgorithmOptimizer = None
    BuildingParameterValidator = None
    ASHRAE140ReferenceData = None
    create_ashrae_140_calibration_targets = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiCaseOptimizer:
    """
    Multi-case optimization for ASHRAE 140 calibration.

    Enables simultaneous optimization across multiple building cases,
    improving model robustness and generalizability.
    """

    def __init__(
        self,
        case_ids: List[str],
        reference_data: Optional[ASHRAE140ReferenceData] = None,
        case_weights: Optional[Dict[str, float]] = None,
        use_parallel: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize multi-case optimizer.

        Args:
            case_ids: List of ASHRAE 140 case IDs to optimize
            reference_data: ASHRAE140ReferenceData instance
            case_weights: Optional dictionary of case weights
            use_parallel: Whether to use parallel evaluation
            max_workers: Maximum number of parallel workers
        """
        self.case_ids = case_ids
        self.reference_data = reference_data or ASHRAE140ReferenceData()
        self.validator = BuildingParameterValidator("standard")

        # Set case weights (default: equal weighting)
        self.case_weights = case_weights or {case_id: 1.0 for case_id in case_ids}

        # Normalize weights
        total_weight = sum(self.case_weights.values())
        self.case_weights = {k: v / total_weight for k, v in self.case_weights.items()}

        self.use_parallel = use_parallel
        self.max_workers = max_workers

        # Load calibration targets
        self.calibration_targets = create_ashrae_140_calibration_targets()

        # Setup parameter bounds
        self.parameter_names = ["u_value", "heating_setpoint", "cooling_setpoint"]
        self.bounds = self.validator.get_optimization_bounds(self.parameter_names)

        logger.info(f"Initialized multi-case optimizer for cases: {case_ids}")
        logger.info(f"Case weights: {self.case_weights}")
        logger.info(f"Parameter bounds: {self.bounds}")

    def evaluate_single_case(self, params: List[float], case_id: str) -> float:
        """
        Evaluate fitness for a single case.

        Args:
            params: Parameter values
            case_id: Case identifier

        Returns:
            Fitness value (lower is better)
        """
        # Validate parameters
        is_valid, errors = self.validator.validate(params, self.parameter_names)
        if not is_valid:
            # Return high fitness for invalid parameters
            return 1e6

        # Get calibration targets for this case
        if case_id not in self.calibration_targets:
            logger.warning(f"No calibration targets for case {case_id}, using case 900")
            case_id = "900"

        targets = self.calibration_targets[case_id]

        # Simple fitness function: weighted sum of squared errors
        # In production, this would run actual simulations
        u_value, heating_sp, cooling_sp = params

        # Mock energy calculation based on parameters
        # Lower U-value = better insulation = lower energy
        # Setpoints closer to comfort range = lower energy
        base_eui = targets["target_eui"]

        # U-value impact (higher U-value increases energy use)
        u_value_impact = (u_value - 1.5) * 20.0  # Penalty for deviation from typical

        # Heating setpoint impact (lower setpoint reduces heating energy)
        heating_impact = max(0, (20.0 - heating_sp)) * 5.0

        # Cooling setpoint impact (higher setpoint reduces cooling energy)
        cooling_impact = max(0, (cooling_sp - 26.0)) * 5.0

        # Total energy use
        total_eui = base_eui + u_value_impact + heating_impact + cooling_impact

        # Fitness is squared error from target
        fitness = (total_eui - targets["target_eui"]) ** 2

        # Add penalty for uncomfortable setpoints
        if heating_sp > 24.0 or cooling_sp < 22.0:
            fitness += 1000  # Large penalty for uncomfortable conditions

        return fitness

    def evaluate_multi_case(self, params: List[float]) -> float:
        """
        Evaluate fitness across all cases.

        Args:
            params: Parameter values

        Returns:
            Weighted average fitness across all cases
        """
        if self.use_parallel:
            return self._evaluate_parallel(params)
        else:
            return self._evaluate_serial(params)

    def _evaluate_serial(self, params: List[float]) -> float:
        """Serial evaluation of all cases."""
        total_fitness = 0.0
        case_results = {}

        for case_id in self.case_ids:
            fitness = self.evaluate_single_case(params, case_id)
            case_results[case_id] = fitness
            total_fitness += fitness * self.case_weights[case_id]

        return total_fitness

    def _evaluate_parallel(self, params: List[float]) -> float:
        """Parallel evaluation of all cases."""
        case_results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all case evaluations
            future_to_case = {
                executor.submit(self.evaluate_single_case, params, case_id): case_id
                for case_id in self.case_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_case):
                case_id = future_to_case[future]
                try:
                    fitness = future.result()
                    case_results[case_id] = fitness
                except Exception as e:
                    logger.error(f"Error evaluating case {case_id}: {e}")
                    case_results[case_id] = 1e6  # High penalty for errors

        # Calculate weighted average fitness
        total_fitness = sum(
            case_results[case_id] * self.case_weights[case_id]
            for case_id in self.case_ids
        )

        return total_fitness

    def optimize(
        self,
        optimizer_type: str = "pso",
        max_iterations: int = 100,
        population_size: int = 30,
        tolerance: float = 1e-6,
        verbose: bool = True,
    ) -> Tuple[List[float], float]:
        """
        Run multi-case optimization.

        Args:
            optimizer_type: Type of optimizer ('pso' or 'ga')
            max_iterations: Maximum number of iterations
            population_size: Number of particles/individuals
            tolerance: Convergence tolerance
            verbose: Whether to log progress

        Returns:
            Tuple of (best_parameters, best_fitness)
        """

        # Create objective function
        def objective_function(params):
            return self.evaluate_multi_case(params)

        # Initialize optimizer based on type
        if optimizer_type.lower() == "ga" and GeneticAlgorithmOptimizer:
            optimizer = GeneticAlgorithmOptimizer(
                objective_function=objective_function,
                n_individuals=population_size,
                n_dimensions=len(self.parameter_names),
                bounds=self.bounds,
                mutation_rate=0.15,
                crossover_rate=0.85,
            )
        else:  # Default to PSO
            optimizer = ParticleSwarmOptimizer(
                objective_function=objective_function,
                n_particles=population_size,
                n_dimensions=len(self.parameter_names),
                bounds=self.bounds,
                inertia_weight=0.7,
                cognitive_weight=1.5,
                social_weight=1.5,
            )

        # Run optimization
        start_time = time.time()
        best_params, best_fitness = optimizer.optimize(
            max_iterations=max_iterations, tolerance=tolerance, verbose=verbose
        )
        execution_time = time.time() - start_time

        # Evaluate final solution on each case
        case_fitness = {}
        for case_id in self.case_ids:
            case_fitness[case_id] = self.evaluate_single_case(best_params, case_id)

        if verbose:
            logger.info(
                f"Multi-case optimization completed in {execution_time:.2f} seconds"
            )
            logger.info(f"Final weighted fitness: {best_fitness:.4f}")
            logger.info(f"Best parameters: {best_params}")
            logger.info("Case-specific fitness:")
            for case_id, fitness in case_fitness.items():
                logger.info(
                    f"  Case {case_id}: {fitness:.4f} (weight: {self.case_weights[case_id]:.2f})"
                )

        return best_params, best_fitness

    def get_case_performance_metrics(self, params: List[float]) -> Dict:
        """
        Get detailed performance metrics for each case.

        Args:
            params: Parameter values

        Returns:
            Dictionary with performance metrics for each case
        """
        metrics = {}

        for case_id in self.case_ids:
            # Get reference data for this case
            case_ref = self.reference_data.get_case_reference(case_id)
            targets = self.calibration_targets.get(
                case_id, self.calibration_targets["900"]
            )

            # Calculate performance metrics
            u_value, heating_sp, cooling_sp = params

            # Energy performance
            base_eui = targets["target_eui"]
            u_value_impact = (u_value - 1.5) * 20.0
            heating_impact = max(0, (20.0 - heating_sp)) * 5.0
            cooling_impact = max(0, (cooling_sp - 26.0)) * 5.0
            total_eui = base_eui + u_value_impact + heating_impact + cooling_impact

            # Comfort metrics
            comfort_score = self._calculate_comfort_score(heating_sp, cooling_sp)

            metrics[case_id] = {
                "parameters": {
                    "u_value": u_value,
                    "heating_setpoint": heating_sp,
                    "cooling_setpoint": cooling_sp,
                },
                "energy": {
                    "target_eui": targets["target_eui"],
                    "actual_eui": total_eui,
                    "deviation": total_eui - targets["target_eui"],
                    "deviation_percent": (total_eui - targets["target_eui"])
                    / targets["target_eui"]
                    * 100,
                },
                "comfort": {
                    "score": comfort_score,
                    "heating_comfort": self._calculate_setpoint_comfort(
                        heating_sp, "heating"
                    ),
                    "cooling_comfort": self._calculate_setpoint_comfort(
                        cooling_sp, "cooling"
                    ),
                },
                "construction": case_ref["construction"],
                "glazing_ratio": case_ref["glazing_ratio"],
            }

        return metrics

    def _calculate_comfort_score(self, heating_sp: float, cooling_sp: float) -> float:
        """Calculate overall comfort score (0-100)."""
        heating_comfort = self._calculate_setpoint_comfort(heating_sp, "heating")
        cooling_comfort = self._calculate_setpoint_comfort(cooling_sp, "cooling")

        # Overall comfort is average of heating and cooling comfort
        return (heating_comfort + cooling_comfort) / 2

    def _calculate_setpoint_comfort(self, setpoint: float, setpoint_type: str) -> float:
        """Calculate comfort score for a setpoint (0-100)."""
        if setpoint_type == "heating":
            # Ideal heating range: 19-21°C
            ideal_min, ideal_max = 19.0, 21.0
        else:  # cooling
            # Ideal cooling range: 24-26°C
            ideal_min, ideal_max = 24.0, 26.0

        # Comfort score based on distance from ideal range
        if ideal_min <= setpoint <= ideal_max:
            return 100.0  # Perfect comfort
        elif setpoint < ideal_min:
            # Too cold (for heating) or too cool (for cooling)
            distance = ideal_min - setpoint
            return max(0, 100 - distance * 10)  # 10 points per °C deviation
        else:  # setpoint > ideal_max
            # Too hot (for heating) or too warm (for cooling)
            distance = setpoint - ideal_max
            return max(0, 100 - distance * 10)  # 10 points per °C deviation

    def create_optimization_report(self, params: List[float]) -> str:
        """Create comprehensive optimization report."""
        metrics = self.get_case_performance_metrics(params)

        report = [
            "=" * 70,
            "MULTI-CASE OPTIMIZATION REPORT",
            "=" * 70,
            "",
            "OPTIMIZED PARAMETERS:",
            "-" * 50,
        ]

        # Parameter summary
        u_value, heating_sp, cooling_sp = params
        report.append(f"U-value: {u_value:.3f} W/m²K")
        report.append(f"Heating Setpoint: {heating_sp:.1f}°C")
        report.append(f"Cooling Setpoint: {cooling_sp:.1f}°C")
        report.append(f"Setpoint Difference: {cooling_sp - heating_sp:.1f}°C")

        report.extend(["", "CASE PERFORMANCE:", "-" * 50])

        # Case performance summary
        for case_id in self.case_ids:
            metric = metrics[case_id]
            energy = metric["energy"]
            comfort = metric["comfort"]

            report.append(f"\nCase {case_id} ({metric['construction']}):")
            report.append(f"  Target EUI: {energy['target_eui']:.1f} kWh/m²/year")
            report.append(f"  Actual EUI: {energy['actual_eui']:.1f} kWh/m²/year")
            report.append(
                f"  Deviation: {energy['deviation']:+.1f} kWh/m²/year ({energy['deviation_percent']:+.1f}%)"
            )
            report.append(f"  Comfort Score: {comfort['score']:.1f}/100")
            report.append(f"  Weight: {self.case_weights[case_id]:.2f}")

        # Overall metrics
        total_deviation = sum(
            metrics[case_id]["energy"]["deviation"] * self.case_weights[case_id]
            for case_id in self.case_ids
        )
        avg_comfort = sum(
            metrics[case_id]["comfort"]["score"] * self.case_weights[case_id]
            for case_id in self.case_ids
        )

        report.extend(["", "OVERALL METRICS:", "-" * 50])
        report.append(
            f"Weighted Average EUI Deviation: {total_deviation:.1f} kWh/m²/year"
        )
        report.append(f"Weighted Average Comfort Score: {avg_comfort:.1f}/100")

        report.append("\n" + "=" * 70)
        return "\n".join(report)


def create_multi_case_benchmark_function() -> Callable:
    """Create a benchmark function for testing multi-case optimization."""

    def multi_case_benchmark(params):
        """
        Benchmark function that simulates multi-case evaluation.

        Args:
            params: List of parameters [u_value, heating_sp, cooling_sp]

        Returns:
            Weighted fitness score
        """
        u_value, heating_sp, cooling_sp = params

        # Case 900: High mass - prefers lower U-value
        fitness_900 = (
            (u_value - 1.5) ** 2 + (heating_sp - 20) ** 2 + (cooling_sp - 26) ** 2
        )

        # Case 600: Low mass - more sensitive to setpoints
        fitness_600 = (u_value - 2.0) ** 2 + 1.5 * (
            (heating_sp - 20) ** 2 + (cooling_sp - 26) ** 2
        )

        # Case 960: All glass - very sensitive to U-value
        fitness_960 = (
            2 * (u_value - 2.0) ** 2 + (heating_sp - 20) ** 2 + (cooling_sp - 26) ** 2
        )

        # Weighted average (equal weights)
        return (fitness_900 + fitness_600 + fitness_960) / 3

    return multi_case_benchmark


if __name__ == "__main__":
    # Demonstration of multi-case optimization
    print("Multi-Case Optimization Demo")
    print("=" * 50)

    # Define cases to optimize
    cases = ["900", "600", "960"]

    # Create multi-case optimizer
    optimizer = MultiCaseOptimizer(cases)

    # Test single case evaluation
    test_params = [1.5, 20.0, 26.0]
    fitness_900 = optimizer.evaluate_single_case(test_params, "900")
    fitness_600 = optimizer.evaluate_single_case(test_params, "600")
    fitness_960 = optimizer.evaluate_single_case(test_params, "960")
    multi_fitness = optimizer.evaluate_multi_case(test_params)

    print(f"Single case fitness:")
    print(f"  Case 900: {fitness_900:.2f}")
    print(f"  Case 600: {fitness_600:.2f}")
    print(f"  Case 960: {fitness_960:.2f}")
    print(f"Multi-case fitness: {multi_fitness:.2f}")

    # Test optimization
    print(f"\nRunning multi-case optimization...")
    best_params, best_fitness = optimizer.optimize(
        optimizer_type="pso", max_iterations=50, population_size=20, verbose=True
    )

    print(f"\nOptimization Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best fitness: {best_fitness:.4f}")

    # Create performance report
    report = optimizer.create_optimization_report(best_params)
    print(f"\nPerformance Report:")
    print(report)

    # Test with different weights
    print(f"\nTesting with different case weights...")
    weighted_optimizer = MultiCaseOptimizer(
        cases, case_weights={"900": 0.5, "600": 0.3, "960": 0.2}
    )

    weighted_params, weighted_fitness = weighted_optimizer.optimize(
        optimizer_type="pso", max_iterations=30, population_size=15, verbose=False
    )

    print(f"Weighted optimization results:")
    print(f"Best parameters: {weighted_params}")
    print(f"Best fitness: {weighted_fitness:.4f}")
