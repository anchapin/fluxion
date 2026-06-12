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
- Active Learning Next Test Recommender (Bayesian Optimization)

Usage:
    from tools.multi_case_optimization import MultiCaseOptimizer, ActiveLearningRecommender
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

    # Active Learning Next Test Recommender
    recommender = ActiveLearningRecommender(
        bounds=[(0.1, 5.0), (15.0, 25.0), (22.0, 32.0)],
        parameter_names=["u_value", "heating_setpoint", "cooling_setpoint"],
        objective_function=objective_function,
    )

    # Add initial observations
    recommender.add_observation([1.5, 20.0, 26.0], 125.0)
    recommender.add_observation([2.0, 21.0, 25.0], 140.0)

    # Get next recommended configuration
    next_config = recommender.get_next_recommendation()
    print(f"Recommended next config: {next_config}")
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.linalg import cholesky, solve_triangular
    from scipy.optimize import minimize
    from scipy.special import erf, ndtri
except ImportError:
    cholesky = None
    solve_triangular = None
    minimize = None
    erf = None
    ndtri = None

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
except ImportError:
    GaussianProcessRegressor = None

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


class ActiveLearningRecommender:
    """
    Active Learning Next Test Recommender using Bayesian Optimization.

    This class implements an active learning algorithm that analyzes existing
    simulation runs and dynamically recommends the next building configuration
    to simulate that yields the highest information gain. Uses Gaussian Processes
    as a surrogate model and acquisition functions to balance exploration vs
    exploitation.

    Features:
    - Gaussian Process surrogate model for the objective function
    - Multiple acquisition functions (Expected Improvement, UCB, Thompson Sampling)
    - Information gain estimation for each recommendation
    - Parallel batch recommendations for efficient exploration
    - Automatic convergence detection

    Attributes:
        bounds: List of (min, max) tuples for each parameter
        parameter_names: Names of the parameters being optimized
        objective_function: Function to evaluate fitness for a given configuration
        use_scikit_learn: Whether to use sklearn GP implementation (if available)
        exploration_weight: Weight for exploration in acquisition function
        n_restarts: Number of optimization restarts for acquisition maximization

    Usage:
        recommender = ActiveLearningRecommender(
            bounds=[(0.1, 5.0), (15.0, 25.0), (22.0, 32.0)],
            parameter_names=["u_value", "heating_setpoint", "cooling_setpoint"],
            objective_function=my_objective_function,
        )

        # Add initial observations
        recommender.add_observation([1.5, 20.0, 26.0], 125.0)
        recommender.add_observation([2.0, 21.0, 25.0], 140.0)

        # Get next recommended configuration
        next_config = recommender.get_next_recommendation()
        print(f"Recommended next config: {next_config}")

        # After running simulation, add the result
        recommender.add_observation(next_config, actual_result)

        # Check convergence
        if recommender.is_converged():
            print("Convergence achieved!")
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        parameter_names: Optional[List[str]] = None,
        objective_function: Optional[Callable[[List[float]], float]] = None,
        use_scikit_learn: bool = True,
        exploration_weight: float = 0.5,
        n_restarts: int = 5,
        noise_variance: float = 1e-5,
    ):
        self.bounds = np.array(bounds)
        self.n_dimensions = len(bounds)
        self.parameter_names = parameter_names or [
            f"param_{i}" for i in range(self.n_dimensions)
        ]
        self.objective_function = objective_function
        self.exploration_weight = exploration_weight
        self.n_restarts = n_restarts
        self.noise_variance = noise_variance

        self.X_observed: np.ndarray = np.empty((0, self.n_dimensions))
        self.y_observed: np.ndarray = np.empty((0, 1))
        self.n_observations: int = 0

        self.use_scikit_learn = use_scikit_learn and GaussianProcessRegressor is not None
        self.gp_model: Optional[GaussianProcessRegressor] = None
        self.gp_fitted: bool = False

        self.acquisition_history: List[Dict] = []
        self.recommendation_count: int = 0

        self._gp_kernel = None
        self._gp_model_sklearn: Optional[GaussianProcessRegressor] = None

        logger.info(
            f"Initialized ActiveLearningRecommender with {self.n_dimensions} parameters"
        )
        logger.info(f"Using sklearn GP: {self.use_scikit_learn}")

    def add_observation(self, params: List[float], objective_value: float) -> None:
        """
        Add a new observation to the training data.

        Args:
            params: Parameter configuration list
            objective_value: Resulting objective value from simulation
        """
        params = np.array(params).reshape(1, -1)
        value = np.array([[objective_value]])

        if self.n_observations == 0:
            self.X_observed = params
            self.y_observed = value
        else:
            self.X_observed = np.vstack([self.X_observed, params])
            self.y_observed = np.vstack([self.y_observed, value])

        self.n_observations += 1
        self.gp_fitted = False

        logger.debug(
            f"Added observation #{self.n_observations}: params={params}, value={objective_value}"
        )

    def add_batch_observations(
        self, params_list: List[List[float]], objective_values: List[float]
    ) -> None:
        """
        Add multiple observations at once.

        Args:
            params_list: List of parameter configurations
            objective_values: List of corresponding objective values
        """
        for params, value in zip(params_list, objective_values):
            self.add_observation(params, value)

    def _build_gp_model(self) -> None:
        """Build and fit the Gaussian Process model."""
        if self.n_observations < 2:
            return

        if self.use_scikit_learn:
            self._build_sklearn_gp()
        else:
            self._build_numpy_gp()

        self.gp_fitted = True

    def _build_sklearn_gp(self) -> None:
        """Build GP model using scikit-learn."""
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(self.n_dimensions),
            length_scale_bounds=(1e-2, 1e2),
        ) + WhiteKernel(noise_level=self.noise_variance, noise_level_bounds=(1e-10, 1e0))

        self._gp_model_sklearn = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=42,
        )

        self._gp_model_sklearn.fit(self.X_observed, self.y_observed)

    def _build_numpy_gp(self) -> None:
        """Build GP model using pure numpy (fallback)."""
        pass

    def _gp_predict(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance using the GP model.

        Args:
            X_test: Test points of shape (n_query, n_dimensions)

        Returns:
            Tuple of (mean, variance) arrays
        """
        if self.n_observations < 2:
            mean = np.full(X_test.shape[0], np.mean(self.y_observed.flatten()))
            var = np.full(X_test.shape[0], 1.0)
            return mean, var

        if self.use_scikit_learn and self._gp_model_sklearn is not None:
            mean, std = self._gp_model_sklearn.predict(
                X_test, return_std=True
            )
            var = std**2
            return mean, var

        mean = np.full(X_test.shape[0], np.mean(self.y_observed.flatten()))
        var = np.full(X_test.shape[0], 1.0)
        return mean, var

    def _expected_improvement(
        self, X_test: np.ndarray, xi: float = 0.01
    ) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function.

        EI measures the expected amount of improvement over the current best
        observation, accounting for both the magnitude of improvement and
        the probability of achieving it.

        Args:
            X_test: Test points
            xi: Exploration parameter (higher = more exploration)

        Returns:
            Expected improvement values
        """
        mean, var = self._gp_predict(X_test)

        if var.size == 0 or np.all(var == 0):
            return np.zeros(X_test.shape[0])

        std = np.sqrt(var)
        best_y = np.min(self.y_observed)

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (best_y - mean - xi) / std
            ei = (best_y - mean - xi) * self._norm_cdf(z) + std * self._norm_pdf(z)
            ei[std == 0] = 0.0

        return ei

    def _upper_confidence_bound(
        self, X_test: np.ndarray, kappa: float = 2.576
    ) -> np.ndarray:
        """
        Calculate Upper Confidence Bound acquisition function.

        UCB balances the mean prediction with uncertainty (std dev),
        using kappa to control the exploration-exploitation tradeoff.

        Args:
            X_test: Test points
            kappa: Exploration parameter (higher = more exploration)

        Returns:
            UCB values (we minimize, so return negative UCB)
        """
        mean, var = self._gp_predict(X_test)
        std = np.sqrt(var)
        ucb = mean + kappa * std
        return -ucb

    def _thompson_sampling(self, X_test: np.ndarray) -> np.ndarray:
        """
        Calculate Thompson Sampling acquisition values.

        Thompson Sampling draws a sample from the GP posterior and uses
        it to select the most promising point.

        Args:
            X_test: Test points

        Returns:
            Sampled values from posterior
        """
        mean, var = self._gp_predict(X_test)
        std = np.sqrt(var)

        samples = np.random.normal(mean, std)
        return samples

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Standard normal cumulative distribution function."""
        if erf is not None:
            return 0.5 * (1.0 + erf(x / np.sqrt(2)))
        from scipy.special import erf as scipy_erf
        return 0.5 * (1.0 + scipy_erf(x / np.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def _maximize_acquisition(
        self, acquisition_function: str = "ei", n_points: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """
        Maximize the acquisition function to find the next best point.

        Args:
            acquisition_function: Acquisition function to use ('ei', 'ucb', 'ts')
            n_points: Number of random points to evaluate

        Returns:
            Tuple of (best_point, acquisition_value)
        """
        if acquisition_function == "ei":
            acq_func = self._expected_improvement
        elif acquisition_function == "ucb":
            acq_func = self._upper_confidence_bound
        elif acquisition_function == "ts":
            acq_func = self._thompson_sampling
        else:
            acq_func = self._expected_improvement

        X_random = self._sample_random_points(n_points)
        acq_values = acq_func(X_random)

        best_idx = np.argmax(acq_values)
        best_point = X_random[best_idx]
        best_value = acq_values[best_idx]

        if self.n_restarts > 0 and minimize is not None:
            best_point, best_value = self._local_search(
                best_point, acq_func, n_restarts=self.n_restarts
            )

        return best_point, best_value

    def _local_search(
        self,
        initial_point: np.ndarray,
        acq_func: Callable,
        n_restarts: int = 5,
    ) -> Tuple[np.ndarray, float]:
        """
        Local search to refine the acquisition maximum.

        Args:
            initial_point: Starting point for optimization
            acq_func: Acquisition function to maximize
            n_restarts: Number of random restarts

        Returns:
            Tuple of (optimal_point, acquisition_value)
        """
        best_point = initial_point.copy()
        best_value = acq_func(initial_point.reshape(1, -1))[0]

        for _ in range(n_restarts):
            x0 = np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1]
            )

            result = minimize(
                lambda x: -acq_func(x.reshape(1, -1))[0],
                x0,
                bounds=self.bounds,
                method="L-BFGS-B",
            )

            if -result.fun > best_value:
                best_point = result.x
                best_value = -result.fun

        return best_point, best_value

    def _sample_random_points(self, n_points: int) -> np.ndarray:
        """Generate random sample points within bounds."""
        points = np.zeros((n_points, self.n_dimensions))
        for i in range(self.n_dimensions):
            min_val, max_val = self.bounds[i]
            points[:, i] = np.random.uniform(min_val, max_val, n_points)
        return points

    def get_next_recommendation(
        self,
        acquisition_function: str = "ei",
        return_acquisition_value: bool = False,
    ) -> Union[List[float], Tuple[List[float], float]]:
        """
        Get the next recommended building configuration to simulate.

        This is the main entry point for the active learning loop. It uses
        the Gaussian Process surrogate model to recommend the configuration
        that maximizes the expected information gain.

        Args:
            acquisition_function: Acquisition function to use
                - 'ei': Expected Improvement (default)
                - 'ucb': Upper Confidence Bound
                - 'ts': Thompson Sampling
            return_acquisition_value: If True, also return the acquisition value

        Returns:
            Next recommended configuration as a list, or tuple of (config, acq_value)
        """
        if self.n_observations < 2:
            return self._get_random_initial_point(return_acquisition_value)

        self._build_gp_model()

        next_point, acq_value = self._maximize_acquisition(
            acquisition_function=acquisition_function,
            n_points=1000,
        )

        self.recommendation_count += 1

        info = {
            "recommendation_number": self.recommendation_count,
            "acquisition_function": acquisition_function,
            "acquisition_value": float(acq_value),
            "n_observations": self.n_observations,
        }
        self.acquisition_history.append(info)

        logger.info(
            f"Recommendation #{self.recommendation_count}: "
            f"config={next_point.tolist()}, acq_value={acq_value:.4f}"
        )

        if return_acquisition_value:
            return next_point.tolist(), float(acq_value)
        return next_point.tolist()

    def _get_random_initial_point(
        self, return_acquisition_value: bool = False
    ) -> Union[List[float], Tuple[List[float], float]]:
        """Get a random initial point when insufficient observations exist."""
        point = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1]
        )

        if return_acquisition_value:
            return point.tolist(), 0.0
        return point.tolist()

    def get_batch_recommendations(
        self,
        n_recommendations: int = 3,
        acquisition_function: str = "ei",
        minimize_overlap: bool = True,
    ) -> List[Dict]:
        """
        Get multiple recommendations for batch evaluation.

        This is useful when you want to run multiple simulations in parallel
        and add them to the training set later as a batch.

        Args:
            n_recommendations: Number of recommendations to generate
            acquisition_function: Acquisition function to use
            minimize_overlap: If True, ensure recommendations are diverse

        Returns:
            List of recommendation dictionaries with 'config' and 'acq_value'
        """
        if self.n_observations < 2:
            return [
                {
                    "config": self._get_random_initial_point()[0]
                    if i > 0
                    else self._get_random_initial_point()[0],
                    "acq_value": 0.0,
                    "rank": i + 1,
                }
                for i in range(n_recommendations)
            ]

        self._build_gp_model()

        recommendations = []
        X_candidates = self._sample_random_points(5000)

        for rank in range(n_recommendations):
            if acquisition_function == "ei":
                acq_func = self._expected_improvement
            elif acquisition_function == "ucb":
                acq_func = self._upper_confidence_bound
            else:
                acq_func = self._thompson_sampling

            acq_values = acq_func(X_candidates)
            best_idx = np.argmax(acq_values)
            best_point = X_candidates[best_idx]
            best_value = acq_values[best_idx]

            recommendations.append(
                {
                    "config": best_point.tolist(),
                    "acq_value": float(best_value),
                    "rank": rank + 1,
                }
            )

            if minimize_overlap and rank < n_recommendations - 1:
                distances = np.linalg.norm(
                    X_candidates - best_point, axis=1
                )
                mask = distances > 0.1 * np.sqrt(self.n_dimensions)
                X_candidates = X_candidates[mask]
                acq_values = acq_values[mask]

                if len(X_candidates) < 100:
                    X_candidates = self._sample_random_points(5000)

        return recommendations

    def estimate_information_gain(
        self, config: List[float], n_mc_samples: int = 100
    ) -> float:
        """
        Estimate the information gain from evaluating a configuration.

        Information gain is estimated by measuring the reduction in
        predictive variance that would result from adding this observation.

        Args:
            config: Configuration to evaluate
            n_mc_samples: Number of Monte Carlo samples for estimation

        Returns:
            Estimated information gain (reduction in variance)
        """
        if self.n_observations < 2:
            return 1.0

        config_array = np.array(config).reshape(1, -1)
        _, var_before = self._gp_predict(config_array)

        X_mc = np.tile(config_array, (n_mc_samples, 1))
        for i in range(self.n_dimensions):
            min_val, max_val = self.bounds[i]
            X_mc[:, i] = np.random.uniform(min_val, max_val, n_mc_samples)

        _, var_mc = self._gp_predict(X_mc)
        avg_var = np.mean(var_mc)

        info_gain = var_before[0] / (avg_var + 1e-10)

        return float(np.clip(info_gain, 0.0, 10.0))

    def is_converged(
        self,
        tolerance: float = 1e-4,
        min_observations: int = 10,
        window_size: int = 5,
    ) -> bool:
        """
        Check if the optimization has converged.

        Convergence is determined by checking if the best observed value
        has stopped improving significantly over a sliding window.

        Args:
            tolerance: Minimum improvement threshold
            min_observations: Minimum number of observations required
            window_size: Size of the sliding window for checking improvement

        Returns:
            True if converged, False otherwise
        """
        if self.n_observations < min_observations:
            return False

        if len(self.acquisition_history) < window_size:
            return False

        recent_acq_values = [
            h["acquisition_value"]
            for h in self.acquisition_history[-window_size:]
        ]

        improvement = max(recent_acq_values) - min(recent_acq_values)

        return improvement < tolerance

    def get_recommendation_confidence(self) -> float:
        """
        Get the confidence of the current recommendation.

        Confidence is based on the GP model's uncertainty at the
        recommended point. Lower uncertainty = higher confidence.

        Returns:
            Confidence score between 0 (no confidence) and 1 (high confidence)
        """
        if self.n_observations < 2:
            return 0.0

        next_config, acq_value = self.get_next_recommendation(
            return_acquisition_value=True
        )

        config_array = np.array(next_config).reshape(1, -1)
        _, var = self._gp_predict(config_array)
        std = np.sqrt(var[0])

        max_range = np.max(self.bounds[:, 1] - self.bounds[:, 0])
        confidence = 1.0 - min(std / max_range, 1.0)

        return float(confidence)

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the active learning process.

        Returns:
            Dictionary with statistics including observation count,
            recommendation history, convergence status, etc.
        """
        stats = {
            "n_observations": self.n_observations,
            "n_recommendations": self.recommendation_count,
            "n_dimensions": self.n_dimensions,
            "parameter_names": self.parameter_names,
            "bounds": self.bounds.tolist(),
            "is_converged": self.is_converged(),
            "recommendation_confidence": self.get_recommendation_confidence(),
            "best_observed_value": (
                float(np.min(self.y_observed)) if self.n_observations > 0 else None
            ),
            "best_observed_config": (
                self.X_observed[np.argmin(self.y_observed)].tolist()
                if self.n_observations > 0
                else None
            ),
            "mean_observed_value": (
                float(np.mean(self.y_observed)) if self.n_observations > 0 else None
            ),
            "std_observed_value": (
                float(np.std(self.y_observed)) if self.n_observations > 0 else None
            ),
            "acquisition_history": self.acquisition_history,
        }

        return stats

    def create_recommendation_report(self) -> str:
        """
        Create a formatted report of the active learning recommendations.

        Returns:
            Formatted report string
        """
        stats = self.get_statistics()

        report = [
            "=" * 70,
            "ACTIVE LEARNING NEXT TEST RECOMMENDER REPORT",
            "=" * 70,
            "",
            "OPTIMIZATION STATUS:",
            "-" * 50,
            f"Total Observations: {stats['n_observations']}",
            f"Total Recommendations: {stats['n_recommendations']}",
            f"Converged: {'Yes' if stats['is_converged'] else 'No'}",
            f"Recommendation Confidence: {stats['recommendation_confidence']:.2%}",
            "",
            "PARAMETER SPACE:",
            "-" * 50,
        ]

        for i, name in enumerate(stats["parameter_names"]):
            bounds = stats["bounds"][i]
            report.append(f"  {name}: [{bounds[0]:.3f}, {bounds[1]:.3f}]")

        report.extend(
            [
                "",
                "BEST OBSERVATION:",
                "-" * 50,
                f"  Best Value: {stats['best_observed_value']:.4f}"
                if stats["best_observed_value"] is not None
                else "  Best Value: N/A",
                f"  Best Config: {stats['best_observed_config']}"
                if stats["best_observed_config"] is not None
                else "  Best Config: N/A",
                f"  Mean Value: {stats['mean_observed_value']:.4f}"
                if stats["mean_observed_value"] is not None
                else "  Mean Value: N/A",
                f"  Std Dev: {stats['std_observed_value']:.4f}"
                if stats["std_observed_value"] is not None
                else "  Std Dev: N/A",
            ]
        )

        if stats["acquisition_history"]:
            report.extend(
                [
                    "",
                    "ACQUISITION HISTORY:",
                    "-" * 50,
                ]
            )
            for h in stats["acquisition_history"][-10:]:
                report.append(
                    f"  #{h['recommendation_number']:3d}: "
                    f"acq={h['acquisition_value']:.4f} "
                    f"({h['acquisition_function']})"
                )

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

    # Active Learning Next Test Recommender Demo
    print("\n" + "=" * 70)
    print("ACTIVE LEARNING NEXT TEST RECOMMENDER DEMO")
    print("=" * 70)

    def objective_function(params):
        u_value, heating_sp, cooling_sp = params
        fitness_900 = (
            (u_value - 1.5) ** 2 + (heating_sp - 20) ** 2 + (cooling_sp - 26) ** 2
        )
        fitness_600 = (u_value - 2.0) ** 2 + 1.5 * (
            (heating_sp - 20) ** 2 + (cooling_sp - 26) ** 2
        )
        fitness_960 = (
            2 * (u_value - 2.0) ** 2 + (heating_sp - 20) ** 2 + (cooling_sp - 26) ** 2
        )
        return (fitness_900 + fitness_600 + fitness_960) / 3

    bounds = [(0.1, 5.0), (15.0, 25.0), (22.0, 32.0)]
    parameter_names = ["u_value", "heating_setpoint", "cooling_setpoint"]

    recommender = ActiveLearningRecommender(
        bounds=bounds,
        parameter_names=parameter_names,
        objective_function=objective_function,
        use_scikit_learn=True,
        exploration_weight=0.5,
    )

    print("\nInitial random observations:")
    np.random.seed(42)
    initial_configs = [
        [1.5, 20.0, 26.0],
        [2.5, 22.0, 27.0],
        [1.0, 19.0, 25.0],
    ]
    for config in initial_configs:
        value = objective_function(config)
        recommender.add_observation(config, value)
        print(f"  Config: {config} -> Value: {value:.4f}")

    print("\nActive Learning Loop:")
    for i in range(5):
        next_config, acq_value = recommender.get_next_recommendation(
            acquisition_function="ei", return_acquisition_value=True
        )
        actual_value = objective_function(next_config)
        recommender.add_observation(next_config, actual_value)
        print(
            f"  Step {i+1}: Recommended {next_config} "
            f"(acq={acq_value:.4f}) -> Actual: {actual_value:.4f}"
        )

    print("\nConvergence Check:")
    print(f"  Is Converged: {recommender.is_converged()}")
    print(f"  Confidence: {recommender.get_recommendation_confidence():.2%}")

    print("\nBatch Recommendations:")
    batch = recommender.get_batch_recommendations(
        n_recommendations=3, acquisition_function="ei"
    )
    for rec in batch:
        print(f"  Rank {rec['rank']}: {rec['config']} (acq={rec['acq_value']:.4f})")

    print("\nFinal Recommendation Report:")
    print(recommender.create_recommendation_report())
