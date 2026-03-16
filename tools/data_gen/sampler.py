"""
Parameter variation sampler for training data generation.

Implements systematic parameter variation sampling with support for:
- Stratified sampling across parameter space
- Uniform and normal distributions
- Constraint handling
- Seed control for reproducibility

References:
- Latin Hypercube Sampling (LHS)
- Monte Carlo methods
"""

import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Supported distribution types for parameter sampling."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    LOG_UNIFORM = "log_uniform"
    TRIANGULAR = "triangular"


@dataclass
class ParameterSpec:
    """
    Specification for a single parameter to be sampled.

    Attributes:
        name: Parameter name
        dist_type: Distribution type (UNIFORM, NORMAL, etc.)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        base_val: Baseline value for relative sampling
        mean: Mean value (for normal distribution)
        std: Standard deviation (for normal distribution)
        mode: Mode value (for triangular distribution)
        constraints: Optional function to validate sampled values
        description: Human-readable description
    """

    name: str
    dist_type: DistributionType = DistributionType.UNIFORM
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    base_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    mode: Optional[float] = None
    constraints: Optional[Callable[[float], bool]] = None
    description: str = ""

    def __post_init__(self):
        """Validate parameter specification."""
        if self.dist_type == DistributionType.UNIFORM:
            if self.min_val is None or self.max_val is None:
                raise ValueError(
                    f"UNIFORM distribution requires min_val and max_val for {self.name}"
                )
        elif self.dist_type == DistributionType.NORMAL:
            if self.mean is None or self.std is None:
                raise ValueError(
                    f"NORMAL distribution requires mean and std for {self.name}"
                )
        elif self.dist_type == DistributionType.LOG_UNIFORM:
            if self.min_val is None or self.max_val is None or self.min_val <= 0:
                raise ValueError(
                    f"LOG_UNIFORM requires min_val > 0 and max_val for {self.name}"
                )
        elif self.dist_type == DistributionType.TRIANGULAR:
            if self.min_val is None or self.max_val is None or self.mode is None:
                raise ValueError(
                    f"TRIANGULAR distribution requires min_val, max_val, and mode for {self.name}"
                )

    def validate(self, value: float) -> bool:
        """Validate a sampled value against constraints."""
        # Basic bounds check
        if self.min_val is not None and value < self.min_val:
            return False
        if self.max_val is not None and value > self.max_val:
            return False

        # Custom constraint check
        if self.constraints is not None:
            if not self.constraints(value):
                return False

        return True


@dataclass
class SamplingConfig:
    """
    Configuration for the parameter sampling process.

    Attributes:
        seed: Random seed for reproducibility
        method: Sampling method (RANDOM, LHS, SOBOL)
        num_samples: Number of samples to generate
        enforce_constraints: Whether to reject invalid samples
        max_retries: Maximum retries for constraint satisfaction
    """

    seed: int = 42
    method: str = "RANDOM"  # RANDOM, LHS, SOBOL
    num_samples: int = 100
    enforce_constraints: bool = True
    max_retries: int = 100


class ParameterSampler:
    """
    Systematic parameter variation sampler.

    Implements multiple sampling strategies for generating diverse parameter sets
    for training data generation.

    Example:
        >>> sampler = ParameterSampler(seed=42)
        >>> sampler.add_parameter(ParameterSpec(
        ...     name="u_value",
        ...     dist_type=DistributionType.UNIFORM,
        ...     min_val=0.5,
        ...     max_val=3.0,
        ...     base_val=1.0,
        ...     description="Wall U-value (W/m²K)"
        ... ))
        >>> samples = sampler.sample(num_samples=10, method="LHS")
    """

    def __init__(self, config: Optional[SamplingConfig] = None):
        """
        Initialize the parameter sampler.

        Args:
            config: Sampling configuration (uses defaults if None)
        """
        self.config = config or SamplingConfig()
        self.parameters: List[ParameterSpec] = []
        self._rng = random.Random(self.config.seed)
        self._np_rng = np.random.RandomState(self.config.seed)

    def add_parameter(self, param: ParameterSpec) -> None:
        """
        Add a parameter specification to the sampler.

        Args:
            param: Parameter specification
        """
        self.parameters.append(param)
        logger.debug(f"Added parameter: {param.name} ({param.dist_type.value})")

    def add_parameter_range(
        self,
        name: str,
        base_val: float,
        variation_range: Tuple[float, float] = (0.5, 1.5),
        dist_type: DistributionType = DistributionType.UNIFORM,
        **kwargs,
    ) -> None:
        """
        Convenience method to add a parameter with relative variation from baseline.

        Args:
            name: Parameter name
            base_val: Baseline value
            variation_range: (min_mult, max_mult) multipliers for baseline
            dist_type: Distribution type
            **kwargs: Additional arguments passed to ParameterSpec
        """
        min_mult, max_mult = variation_range
        min_val = base_val * min_mult
        max_val = base_val * max_mult

        param = ParameterSpec(
            name=name,
            dist_type=dist_type,
            min_val=min_val,
            max_val=max_val,
            base_val=base_val,
            **kwargs,
        )
        self.add_parameter(param)

    def sample(
        self, num_samples: Optional[int] = None, method: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """
        Generate parameter samples.

        Args:
            num_samples: Number of samples to generate (uses config default if None)
            method: Sampling method (RANDOM, LHS, SOBOL)

        Returns:
            List of parameter dictionaries
        """
        n = num_samples or self.config.num_samples
        method = method or self.config.method

        logger.info(f"Generating {n} samples using {method} method")

        if not self.parameters:
            raise ValueError("No parameters added to sampler")

        if method == "RANDOM":
            samples = self._sample_random(n)
        elif method == "LHS":
            samples = self._sample_lhs(n)
        elif method == "SOBOL":
            samples = self._sample_sobol(n)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Apply constraints if enabled
        if self.config.enforce_constraints:
            samples = self._apply_constraints(samples)

        logger.info(f"Successfully generated {len(samples)} valid samples")
        return samples

    def _sample_random(self, n: int) -> List[Dict[str, float]]:
        """
        Generate samples using simple random sampling.

        Args:
            n: Number of samples

        Returns:
            List of parameter dictionaries
        """
        samples = []
        for _ in range(n):
            sample = {}
            for param in self.parameters:
                sample[param.name] = self._sample_parameter(param)
            samples.append(sample)
        return samples

    def _sample_lhs(self, n: int) -> List[Dict[str, float]]:
        """
        Generate samples using Latin Hypercube Sampling (LHS).

        LHS ensures stratified sampling by dividing each parameter's range
        into n equal intervals and sampling exactly once from each interval.

        Args:
            n: Number of samples

        Returns:
            List of parameter dictionaries
        """
        num_params = len(self.parameters)
        samples = []

        # Generate LHS matrix
        lhs_matrix = np.zeros((n, num_params))
        for j in range(num_params):
            # Random permutation of [0, 1, ..., n-1]
            perm = self._np_rng.permutation(n)
            # Generate samples: (perm + u) / n where u ~ U(0,1)
            u = self._np_rng.uniform(0, 1, n)
            lhs_matrix[:, j] = (perm + u) / n

        # Map LHS values to parameter ranges
        for i in range(n):
            sample = {}
            for j, param in enumerate(self.parameters):
                lhs_value = lhs_matrix[i, j]
                sample[param.name] = self._map_lhs_to_parameter(lhs_value, param)
            samples.append(sample)

        return samples

    def _sample_sobol(self, n: int) -> List[Dict[str, float]]:
        """
        Generate samples using Sobol sequence (quasi-random).

        Sobol sequences provide low-discrepancy quasi-random sampling,
        which covers the parameter space more evenly than random sampling.

        Args:
            n: Number of samples

        Returns:
            List of parameter dictionaries
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            logger.warning("scipy not available, falling back to LHS")
            return self._sample_lhs(n)

        # Generate Sobol sequence
        sampler = qmc.Sobol(
            d=len(self.parameters), scramble=True, seed=self.config.seed
        )
        sobol_points = sampler.random(n)

        # Map to parameter ranges
        samples = []
        for point in sobol_points:
            sample = {}
            for j, param in enumerate(self.parameters):
                sample[param.name] = self._map_lhs_to_parameter(point[j], param)
            samples.append(sample)

        return samples

    def _sample_parameter(self, param: ParameterSpec) -> float:
        """
        Sample a single parameter based on its distribution type.

        Args:
            param: Parameter specification

        Returns:
            Sampled value
        """
        if param.dist_type == DistributionType.UNIFORM:
            value = self._rng.uniform(param.min_val, param.max_val)

        elif param.dist_type == DistributionType.NORMAL:
            # Sample from normal and clip to bounds
            value = self._rng.gauss(param.mean, param.std)
            if param.min_val is not None:
                value = max(value, param.min_val)
            if param.max_val is not None:
                value = min(value, param.max_val)

        elif param.dist_type == DistributionType.LOG_UNIFORM:
            # Sample log-uniform: exp(U(log(min), log(max)))
            log_min = math.log(param.min_val)
            log_max = math.log(param.max_val)
            value = math.exp(self._rng.uniform(log_min, log_max))

        elif param.dist_type == DistributionType.TRIANGULAR:
            value = self._rng.triangular(param.min_val, param.mode, param.max_val)

        else:
            raise ValueError(f"Unknown distribution type: {param.dist_type}")

        return value

    def _map_lhs_to_parameter(self, lhs_value: float, param: ParameterSpec) -> float:
        """
        Map LHS value (0-1) to parameter range based on distribution.

        Args:
            lhs_value: LHS sample value in [0, 1]
            param: Parameter specification

        Returns:
            Mapped parameter value
        """
        if param.dist_type == DistributionType.UNIFORM:
            # Linear mapping: min + lhs * (max - min)
            return param.min_val + lhs_value * (param.max_val - param.min_val)

        elif param.dist_type == DistributionType.LOG_UNIFORM:
            # Log-uniform mapping
            log_min = math.log(param.min_val)
            log_max = math.log(param.max_val)
            log_val = log_min + lhs_value * (log_max - log_min)
            return math.exp(log_val)

        elif param.dist_type == DistributionType.NORMAL:
            # For normal distribution in LHS, we use inverse CDF
            # First convert LHS value to standard normal using inverse CDF
            try:
                from scipy.stats import norm

                z = norm.ppf(lhs_value)
                value = param.mean + z * param.std
                # Clip to bounds
                if param.min_val is not None:
                    value = max(value, param.min_val)
                if param.max_val is not None:
                    value = min(value, param.max_val)
                return value
            except ImportError:
                logger.warning("scipy not available, using uniform approximation")
                return param.min_val + lhs_value * (param.max_val - param.min_val)

        elif param.dist_type == DistributionType.TRIANGULAR:
            # Triangular distribution inverse CDF
            a, m, b = param.min_val, param.mode, param.max_val
            f = (m - a) / (b - a)

            if lhs_value < f:
                # Left side of triangle
                return a + math.sqrt(lhs_value * (b - a) * (m - a))
            else:
                # Right side of triangle
                return b - math.sqrt((1 - lhs_value) * (b - a) * (b - m))

        else:
            raise ValueError(f"Unknown distribution type: {param.dist_type}")

    def _apply_constraints(
        self, samples: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Apply constraints to samples, rejecting invalid ones.

        Args:
            samples: List of sampled parameter dictionaries

        Returns:
            List of valid samples
        """
        valid_samples = []
        retries = 0

        for sample in samples:
            if retries >= self.config.max_retries:
                logger.warning(f"Max retries reached, skipping constraint enforcement")
                break

            is_valid = True
            for param in self.parameters:
                value = sample[param.name]
                if not param.validate(value):
                    is_valid = False
                    # Resample this parameter
                    sample[param.name] = self._sample_parameter(param)
                    retries += 1
                    break

            if is_valid:
                valid_samples.append(sample)
                retries = 0  # Reset retry counter

        if len(valid_samples) < len(samples):
            logger.warning(
                f"Rejected {len(samples) - len(valid_samples)} samples due to constraints"
            )

        return valid_samples

    def get_default_ashrae_140_specs(self) -> List[ParameterSpec]:
        """
        Get default parameter specifications for ASHRAE 140 cases.

        Returns:
            List of ParameterSpec objects for common ASHRAE 140 parameters
        """
        specs = [
            # U-value variations (0.5× to 1.5× baseline)
            ParameterSpec(
                name="u_value",
                dist_type=DistributionType.UNIFORM,
                min_val=0.257,
                max_val=0.771,
                base_val=0.514,
                description="Wall U-value (W/m²K) - low-mass construction",
            ),
            ParameterSpec(
                name="roof_u_value",
                dist_type=DistributionType.UNIFORM,
                min_val=0.159,
                max_val=0.477,
                base_val=0.318,
                description="Roof U-value (W/m²K) - low-mass construction",
            ),
            # Setpoint variations (±5°C)
            ParameterSpec(
                name="heating_setpoint",
                dist_type=DistributionType.UNIFORM,
                min_val=18.0,
                max_val=22.0,
                base_val=20.0,
                description="HVAC heating setpoint (°C)",
            ),
            ParameterSpec(
                name="cooling_setpoint",
                dist_type=DistributionType.UNIFORM,
                min_val=25.0,
                max_val=29.0,
                base_val=27.0,
                description="HVAC cooling setpoint (°C)",
            ),
            # Infiltration rate variations (±0.2 ACH)
            ParameterSpec(
                name="infiltration_rate",
                dist_type=DistributionType.UNIFORM,
                min_val=0.3,
                max_val=0.7,
                base_val=0.5,
                description="Air infiltration rate (ACH)",
            ),
            # Window-to-Wall Ratio (WWR) variations
            ParameterSpec(
                name="wwr",
                dist_type=DistributionType.UNIFORM,
                min_val=0.2,
                max_val=0.6,
                base_val=0.4,
                description="Window-to-wall ratio",
            ),
            # Geometric variations
            ParameterSpec(
                name="width",
                dist_type=DistributionType.UNIFORM,
                min_val=6.0,
                max_val=12.0,
                base_val=8.0,
                description="Building width (m)",
            ),
            ParameterSpec(
                name="length",
                dist_type=DistributionType.UNIFORM,
                min_val=6.0,
                max_val=12.0,
                base_val=8.0,
                description="Building length (m)",
            ),
            ParameterSpec(
                name="aspect_ratio",
                dist_type=DistributionType.TRIANGULAR,
                min_val=0.5,
                max_val=2.0,
                mode=1.0,
                description="Building aspect ratio (width/length)",
                constraints=lambda x: 0.3 < x < 3.0,
            ),
            # Window properties
            ParameterSpec(
                name="window_shgc",
                dist_type=DistributionType.UNIFORM,
                min_val=0.6,
                max_val=0.85,
                base_val=0.789,
                description="Window solar heat gain coefficient",
            ),
            ParameterSpec(
                name="window_u_value",
                dist_type=DistributionType.UNIFORM,
                min_val=2.0,
                max_val=3.5,
                base_val=2.5,
                description="Window U-value (W/m²K)",
            ),
        ]

        return specs

    def create_ashrae_140_sampler(
        self, seed: int = 42, method: str = "LHS"
    ) -> "ParameterSampler":
        """
        Create a sampler pre-configured for ASHRAE 140 parameter variations.

        Args:
            seed: Random seed
            method: Sampling method

        Returns:
            Configured ParameterSampler instance
        """
        self.config.seed = seed
        self.config.method = method

        # Add default ASHRAE 140 specs
        for spec in self.get_default_ashrae_140_specs():
            self.add_parameter(spec)

        return self


# Convenience functions for common sampling patterns


def create_stratified_sampler(
    num_samples: int,
    params: List[Tuple[str, float, float]],
    seed: int = 42,
) -> ParameterSampler:
    """
    Create a simple stratified sampler for uniform parameters.

    Args:
        num_samples: Number of samples to generate
        params: List of (name, min_val, max_val) tuples
        seed: Random seed

    Returns:
        Configured ParameterSampler instance

    Example:
        >>> sampler = create_stratified_sampler(
        ...     num_samples=10,
        ...     params=[("u_value", 0.5, 3.0), ("wwr", 0.1, 0.8)],
        ...     seed=42
        ... )
        >>> samples = sampler.sample()
    """
    config = SamplingConfig(seed=seed, num_samples=num_samples, method="LHS")
    sampler = ParameterSampler(config)

    for name, min_val, max_val in params:
        param = ParameterSpec(
            name=name,
            dist_type=DistributionType.UNIFORM,
            min_val=min_val,
            max_val=max_val,
        )
        sampler.add_parameter(param)

    return sampler


def create_normal_sampler(
    num_samples: int,
    params: List[Tuple[str, float, float, float, float]],
    seed: int = 42,
) -> ParameterSampler:
    """
    Create a sampler for normally-distributed parameters.

    Args:
        num_samples: Number of samples to generate
        params: List of (name, mean, std, min_val, max_val) tuples
        seed: Random seed

    Returns:
        Configured ParameterSampler instance

    Example:
        >>> sampler = create_normal_sampler(
        ...     num_samples=10,
        ...     params=[("temperature", 20.0, 2.0, 15.0, 25.0)],
        ...     seed=42
        ... )
        >>> samples = sampler.sample()
    """
    config = SamplingConfig(seed=seed, num_samples=num_samples, method="RANDOM")
    sampler = ParameterSampler(config)

    for name, mean, std, min_val, max_val in params:
        param = ParameterSpec(
            name=name,
            dist_type=DistributionType.NORMAL,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
        )
        sampler.add_parameter(param)

    return sampler


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Creating ASHRAE 140 parameter sampler...")
    sampler = ParameterSampler(SamplingConfig(seed=42, num_samples=5, method="LHS"))
    sampler.create_ashrae_140_sampler()

    print("\nGenerating 5 samples using LHS:")
    samples = sampler.sample()
    for i, sample in enumerate(samples):
        print(f"\nSample {i + 1}:")
        for key, value in sample.items():
            print(f"  {key}: {value:.4f}")

    print("\n\nComparing sampling methods:")
    print("\nRandom sampling:")
    sampler_random = ParameterSampler(
        SamplingConfig(seed=42, num_samples=3, method="RANDOM")
    )
    for spec in sampler.get_default_ashrae_140_specs()[:3]:
        sampler_random.add_parameter(spec)
    for sample in sampler_random.sample():
        print(f"  u_value: {sample['u_value']:.4f}")

    print("\nLHS sampling:")
    sampler_lhs = ParameterSampler(SamplingConfig(seed=42, num_samples=3, method="LHS"))
    for spec in sampler.get_default_ashrae_140_specs()[:3]:
        sampler_lhs.add_parameter(spec)
    for sample in sampler_lhs.sample():
        print(f"  u_value: {sample['u_value']:.4f}")
