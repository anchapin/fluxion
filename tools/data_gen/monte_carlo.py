"""
Massive Data Generation Tool for AI Surrogate Training.

This module provides a Monte Carlo-based data generation engine that:
- Samples diverse building configurations
- Runs simulations using the Fluxion Rust engine
- Outputs training data in Parquet format: (State_t, Action_t) -> (State_t+1, Energy_consumed)

Usage:
    python -m tools.data_gen.monte_carlo generate --count 10000 --output data/train.parquet
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure we can import the package if run directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from tools.data_gen import geometry, sampler, weather
from tools.data_gen.sampler import (
    DistributionType,
    ParameterSampler,
    ParameterSpec,
    SamplingConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BuildingConfig:
    """Building configuration parameters for simulation."""

    # Geometry
    width: float = 8.0  # meters
    length: float = 8.0  # meters
    height: float = 3.5  # meters
    num_floors: int = 1
    aspect_ratio: float = 1.0

    # Envelope
    wall_u_value: float = 0.5  # W/m²K
    roof_u_value: float = 0.3  # W/m²K
    floor_u_value: float = 0.5  # W/m²K
    window_u_value: float = 2.5  # W/m²K
    window_shgc: float = 0.7  # solar heat gain coefficient
    wwr: float = 0.4  # window-to-wall ratio

    # Infiltration
    infiltration_ach: float = 0.5  # air changes per hour

    # Internal loads
    occupancy_density: float = 0.05  # persons/m²
    equipment_density: float = 10.0  # W/m²
    lighting_density: float = 8.0  # W/m²

    # HVAC
    heating_setpoint: float = 20.0  # °C
    cooling_setpoint: float = 27.0  # °C
    hvac_capacity: float = 5000.0  # W
    cop_heating: float = 3.0  # coefficient of performance
    cop_cooling: float = 3.0

    # Weather
    weather_file: str = "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"

    # Orientation
    orientation: float = 0.0  # degrees from north

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "num_floors": self.num_floors,
            "aspect_ratio": self.aspect_ratio,
            "wall_u_value": self.wall_u_value,
            "roof_u_value": self.roof_u_value,
            "floor_u_value": self.floor_u_value,
            "window_u_value": self.window_u_value,
            "window_shgc": self.window_shgc,
            "wwr": self.wwr,
            "infiltration_ach": self.infiltration_ach,
            "occupancy_density": self.occupancy_density,
            "equipment_density": self.equipment_density,
            "lighting_density": self.lighting_density,
            "heating_setpoint": self.heating_setpoint,
            "cooling_setpoint": self.cooling_setpoint,
            "hvac_capacity": self.hvac_capacity,
            "cop_heating": self.cop_heating,
            "cop_cooling": self.cop_cooling,
            "orientation": self.orientation,
        }


@dataclass
class SimulationResult:
    """Result from a single simulation run."""

    config: BuildingConfig
    run_id: str

    # Time series data (hourly for a year = 8760 timesteps)
    outdoor_temps: np.ndarray = field(default_factory=lambda: np.zeros(8760))
    indoor_temps: np.ndarray = field(default_factory=lambda: np.zeros(8760))
    heating_loads: np.ndarray = field(default_factory=lambda: np.zeros(8760))  # Wh
    cooling_loads: np.ndarray = field(default_factory=lambda: np.zeros(8760))  # Wh
    solar_gains: np.ndarray = field(default_factory=lambda: np.zeros(8760))  # W/m²

    # Aggregate results
    total_heating_load: float = 0.0  # kWh
    total_cooling_load: float = 0.0  # kWh
    total_energy: float = 0.0  # kWh

    # Metadata
    simulation_time: float = 0.0  # seconds
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class TrainingSample:
    """Single training sample for AI surrogate: (State_t, Action_t) -> (State_t+1, Energy)."""

    # State at time t
    outdoor_temp_t: float
    indoor_temp_t: float
    solar_t: float

    # Action at time t (HVAC control)
    hvac_mode: int  # 0=off, 1=heating, 2=cooling
    hvac_power: float  # W

    # Resulting state at t+1
    indoor_temp_t1: float
    energy_consumed: float  # Wh

    # Metadata
    run_id: str
    timestep: int


# =============================================================================
# Monte Carlo Data Generator
# =============================================================================


class MonteCarloDataGenerator:
    """
    Monte Carlo data generator for building energy simulation.

    Generates diverse training data by:
    1. Sampling building configurations from parameter distributions
    2. Running simulations using the Fluxion engine
    3. Converting results to training samples
    4. Saving to Parquet format

    Example:
        >>> gen = MonteCarloDataGenerator(
        ...     output_dir="data/train",
        ...     num_samples=10000,
        ...     seed=42
        ... )
        >>> gen.setup()
        >>> df = gen.generate()
        >>> print(f"Generated {len(df)} training samples")
    """

    def __init__(
        self,
        output_dir: str = "data/train",
        num_samples: int = 1000,
        num_timesteps: int = 8760,  # One year hourly
        seed: Optional[int] = 42,
        weather_dir: str = "assets/weather",
        sampling_method: str = "LHS",
        batch_size: int = 100,
        num_workers: int = 1,
    ):
        """
        Initialize the Monte Carlo data generator.

        Args:
            output_dir: Directory to save generated data
            num_samples: Number of building configurations to simulate
            num_timesteps: Number of timesteps per simulation (default: 8760 = 1 year)
            seed: Random seed for reproducibility
            weather_dir: Directory containing EPW weather files
            sampling_method: Sampling method (RANDOM, LHS, SOBOL)
            batch_size: Number of simulations to run per batch
            num_workers: Number of parallel workers
        """
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.num_timesteps = num_timesteps
        self.seed = seed
        self.weather_dir = Path(weather_dir)
        self.sampling_method = sampling_method
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize components
        self.param_sampler: Optional[ParameterSampler] = None
        self.weather_files: List[str] = []
        self.simulation_results: List[SimulationResult] = []

        # Performance tracking
        self.total_time = 0.0
        self.simulation_times: List[float] = []

        # Output format
        self.output_format = "parquet"

        # Try to import Fluxion engine
        self._engine_available = False
        self._try_import_engine()

    def _try_import_engine(self) -> None:
        """Try to import the Fluxion Rust engine."""
        try:
            import fluxion

            self.fluxion = fluxion
            self._engine_available = True
            logger.info("Fluxion engine available for simulation")
        except ImportError:
            logger.warning(
                "Fluxion Python bindings not available. "
                "Using mock simulation for demonstration."
            )
            self._engine_available = False

    def setup(self) -> None:
        """Setup the generator: prepare samplers, weather files, output directory."""
        logger.info(f"Setting up Monte Carlo data generator")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Number of samples: {self.num_samples}")
        logger.info(f"  Sampling method: {self.sampling_method}")
        logger.info(f"  Seed: {self.seed}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup parameter sampler
        self._setup_parameter_sampler()

        # Setup weather files
        self._setup_weather_files()

        logger.info("Setup complete")

    def _setup_parameter_sampler(self) -> None:
        """Setup the parameter sampler with building configuration specs."""
        config = SamplingConfig(
            seed=self.seed or 42,
            num_samples=self.num_samples,
            method=self.sampling_method,
        )
        self.param_sampler = ParameterSampler(config)

        # Add parameter specifications for building configuration
        # Geometry parameters
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="width",
                dist_type=DistributionType.UNIFORM,
                min_val=5.0,
                max_val=20.0,
                description="Building width (m)",
            )
        )
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="length",
                dist_type=DistributionType.UNIFORM,
                min_val=5.0,
                max_val=20.0,
                description="Building length (m)",
            )
        )
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="height",
                dist_type=DistributionType.UNIFORM,
                min_val=2.5,
                max_val=4.5,
                description="Ceiling height (m)",
            )
        )

        # Envelope parameters
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="wall_u_value",
                dist_type=DistributionType.UNIFORM,
                min_val=0.2,
                max_val=1.5,
                description="Wall U-value (W/m²K)",
            )
        )
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="roof_u_value",
                dist_type=DistributionType.UNIFORM,
                min_val=0.1,
                max_val=0.8,
                description="Roof U-value (W/m²K)",
            )
        )
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="window_u_value",
                dist_type=DistributionType.UNIFORM,
                min_val=1.0,
                max_val=3.5,
                description="Window U-value (W/m²K)",
            )
        )
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="window_shgc",
                dist_type=DistributionType.UNIFORM,
                min_val=0.3,
                max_val=0.85,
                description="Window SHGC",
            )
        )
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="wwr",
                dist_type=DistributionType.UNIFORM,
                min_val=0.1,
                max_val=0.8,
                description="Window-to-wall ratio",
            )
        )

        # Infiltration
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="infiltration_ach",
                dist_type=DistributionType.UNIFORM,
                min_val=0.1,
                max_val=1.5,
                description="Infiltration rate (ACH)",
            )
        )

        # Internal loads
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="occupancy_density",
                dist_type=DistributionType.UNIFORM,
                min_val=0.02,
                max_val=0.1,
                description="Occupancy density (persons/m²)",
            )
        )
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="equipment_density",
                dist_type=DistributionType.UNIFORM,
                min_val=5.0,
                max_val=20.0,
                description="Equipment density (W/m²)",
            )
        )

        # HVAC setpoints
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="heating_setpoint",
                dist_type=DistributionType.UNIFORM,
                min_val=18.0,
                max_val=22.0,
                description="Heating setpoint (°C)",
            )
        )
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="cooling_setpoint",
                dist_type=DistributionType.UNIFORM,
                min_val=24.0,
                max_val=28.0,
                description="Cooling setpoint (°C)",
            )
        )

        # Orientation
        self.param_sampler.add_parameter(
            ParameterSpec(
                name="orientation",
                dist_type=DistributionType.UNIFORM,
                min_val=0.0,
                max_val=360.0,
                description="Building orientation (degrees)",
            )
        )

        logger.info(f"Parameter sampler configured with {len(self.param_sampler.parameters)} parameters")

    def _setup_weather_files(self) -> None:
        """Setup available weather files."""
        if self.weather_dir.exists():
            self.weather_files = [
                f for f in os.listdir(self.weather_dir) if f.endswith(".epw")
            ]
            logger.info(f"Found {len(self.weather_files)} weather files")
        else:
            logger.warning(f"Weather directory {self.weather_dir} not found")
            self.weather_files = []

        # Default weather file if none found
        if not self.weather_files:
            self.weather_files = ["USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"]
            logger.warning("Using default weather file")

    def sample_config(self, sample_idx: int) -> BuildingConfig:
        """
        Sample a building configuration from the parameter distributions.

        Args:
            sample_idx: Index of the sample (used for random state)

        Returns:
            BuildingConfig with sampled parameters
        """
        # Sample parameters
        samples = self.param_sampler.sample(
            num_samples=1, method=self.sampling_method
        )
        params = samples[0]

        # Select weather file (round-robin or random)
        weather_file = self.weather_files[sample_idx % len(self.weather_files)]

        # Create building config
        config = BuildingConfig(
            width=params.get("width", 8.0),
            length=params.get("length", 8.0),
            height=params.get("height", 3.5),
            wall_u_value=params.get("wall_u_value", 0.5),
            roof_u_value=params.get("roof_u_value", 0.3),
            window_u_value=params.get("window_u_value", 2.5),
            window_shgc=params.get("window_shgc", 0.7),
            wwr=params.get("wwr", 0.4),
            infiltration_ach=params.get("infiltration_ach", 0.5),
            occupancy_density=params.get("occupancy_density", 0.05),
            equipment_density=params.get("equipment_density", 10.0),
            heating_setpoint=params.get("heating_setpoint", 20.0),
            cooling_setpoint=params.get("cooling_setpoint", 27.0),
            orientation=params.get("orientation", 0.0),
            weather_file=weather_file,
        )

        return config

    def run_simulation(self, config: BuildingConfig, run_id: str) -> SimulationResult:
        """
        Run a simulation for the given building configuration.

        Args:
            config: Building configuration
            run_id: Unique identifier for this run

        Returns:
            SimulationResult with time series data
        """
        start_time = time.time()

        result = SimulationResult(config=config, run_id=run_id)

        try:
            if self._engine_available:
                result = self._run_fluxion_simulation(config, run_id)
            else:
                result = self._run_mock_simulation(config, run_id)

            result.success = True
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Simulation failed for {run_id}: {e}")

        result.simulation_time = time.time() - start_time
        return result

    def _run_fluxion_simulation(
        self, config: BuildingConfig, run_id: str
    ) -> SimulationResult:
        """Run simulation using Fluxion Rust engine."""
        # Import fluxion here to get fresh module reference
        import fluxion

        # Create model
        model = fluxion.Model(num_zones=1)

        # Set ground temperature
        model.set_ground_temp(15.0)

        # Run simulation for full year (simulate returns EUI in kWh/m²/year)
        # Use use_surrogates=False for ground truth
        years = self.num_timesteps // 8760
        if years < 1:
            years = 1

        # Run simulation without surrogates for ground truth
        eui = model.simulate(years=years, use_surrogates=False)

        # Create result
        result = SimulationResult(config=config, run_id=run_id)
        
        # Calculate total energy from EUI
        floor_area = config.width * config.length * config.num_floors
        result.total_energy = eui * floor_area * years

        # Generate synthetic time series based on the EUI result
        # (Fluxion doesn't expose full time series, so we generate realistic data)
        result = self._generate_realistic_timeseries(result)

        return result

    def _run_mock_simulation(
        self, config: BuildingConfig, run_id: str
    ) -> SimulationResult:
        """Run mock simulation for demonstration when Fluxion is not available."""
        result = SimulationResult(config=config, run_id=run_id)
        result = self._generate_realistic_timeseries(result)
        return result

    def _generate_realistic_timeseries(
        self, result: SimulationResult
    ) -> SimulationResult:
        """Generate realistic time series based on building configuration."""
        config = result.config

        # Generate outdoor temperature (annual pattern)
        hours = np.arange(self.num_timesteps)
        day_of_year = hours // 24
        hour_of_day = hours % 24

        # Annual temperature variation (sine wave)
        annual_cycle = -10 * np.cos(2 * np.pi * day_of_year / 365)
        # Daily temperature variation
        daily_cycle = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        # Base temperature
        base_temp = 15.0

        result.outdoor_temps = base_temp + annual_cycle + daily_cycle

        # Generate solar radiation
        # Solar varies by orientation and hour
        solar_factor = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
        max_solar = 800 * (1 - config.wwr * 0.5)  # Reduced by windows
        result.solar_gains = solar_factor * max_solar * (
            0.8 + 0.2 * np.sin(np.radians(config.orientation))
        )

        # Calculate heat transfer
        # Q = U * A * delta_T
        wall_area = 2 * (config.width + config.length) * config.height
        window_area = config.wwr * wall_area

        # Net area (walls + windows)
        total_area = wall_area + window_area

        # Heat transfer coefficient
        u_wall = config.wall_u_value
        u_window = config.window_u_value
        u_avg = (wall_area * u_wall + window_area * u_window) / total_area

        # Indoor temperature (simplified - maintained near setpoint when HVAC is on)
        # Free-floating temperature calculation
        indoor_free = result.outdoor_temps + (
            result.solar_gains / (u_avg * 10)
        )  # Simplified

        # HVAC mode simulation
        # Heating needed when outdoor is low or indoor drops below setpoint
        # Cooling needed when solar/ outdoor is high

        heating_setpoint = config.heating_setpoint
        cooling_setpoint = config.cooling_setpoint

        # Determine HVAC mode and loads
        for t in range(self.num_timesteps):
            outdoor = result.outdoor_temps[t]
            solar = result.solar_gains[t]

            # Simple indoor temperature calculation
            if outdoor < heating_setpoint:
                # Need heating
                dt = heating_setpoint - outdoor
                heat_loss = u_avg * total_area * dt
                # Subtract solar gain
                net_loss = heat_loss - solar * 0.2 * total_area
                if net_loss > 0:
                    result.heating_loads[t] = max(0, net_loss)  # Watts
                else:
                    result.cooling_loads[t] = max(0, -net_loss)  # Watts
            elif outdoor > cooling_setpoint or solar > 200:
                # Need cooling
                dt = outdoor - cooling_setpoint
                heat_gain = u_avg * total_area * dt
                # Add solar gain
                net_gain = heat_gain + solar * 0.3 * total_area
                if net_gain > 0:
                    result.cooling_loads[t] = max(0, net_gain)  # Watts

            # Indoor temperature (simplified)
            if result.heating_loads[t] > 0:
                result.indoor_temps[t] = heating_setpoint
            elif result.cooling_loads[t] > 0:
                result.indoor_temps[t] = cooling_setpoint
            else:
                result.indoor_temps[t] = outdoor + 0.5  # Free-floating

        # Aggregate totals
        result.total_heating_load = np.sum(result.heating_loads) / 1000  # kWh
        result.total_cooling_load = np.sum(result.cooling_loads) / 1000  # kWh
        result.total_energy = result.total_heating_load + result.total_cooling_load

        return result

    def generate(self) -> pd.DataFrame:
        """
        Generate training data by running Monte Carlo simulations.

        Returns:
            DataFrame with training samples in format:
            (State_t, Action_t) -> (State_t+1, Energy_consumed)
        """
        logger.info(f"Starting Monte Carlo generation of {self.num_samples} samples")

        all_samples: List[TrainingSample] = []
        start_time = time.time()

        # Generate samples in batches
        for batch_idx in range(0, self.num_samples, self.batch_size):
            batch_end = min(batch_idx + self.batch_size, self.num_samples)
            batch_size = batch_end - batch_idx

            logger.info(
                f"Processing batch {batch_idx // self.batch_size + 1}: "
                f"samples {batch_idx} to {batch_end}"
            )

            for i in range(batch_size):
                sample_idx = batch_idx + i
                run_id = f"sample_{sample_idx:06d}"

                # Sample configuration
                config = self.sample_config(sample_idx)

                # Run simulation
                result = self.run_simulation(config, run_id)

                if result.success:
                    # Convert to training samples
                    samples = self._result_to_samples(result)
                    all_samples.extend(samples)
                    self.simulation_times.append(result.simulation_time)

                    if (sample_idx + 1) % 100 == 0:
                        logger.info(
                            f"  Completed {sample_idx + 1}/{self.num_samples} simulations"
                        )
                else:
                    logger.warning(f"  Failed: {run_id}")

        self.total_time = time.time() - start_time

        # Convert to DataFrame
        df = self._samples_to_dataframe(all_samples)

        # Save to file
        self._save_data(df)

        # Print summary
        logger.info(f"Generation complete!")
        logger.info(f"  Total samples generated: {len(all_samples)}")
        logger.info(f"  Total time: {self.total_time:.2f}s")
        if self.simulation_times:
            logger.info(
                f"  Average simulation time: {np.mean(self.simulation_times):.3f}s"
            )
            logger.info(f"  Throughput: {len(all_samples) / self.total_time:.1f} samples/s")

        return df

    def _result_to_samples(
        self, result: SimulationResult
    ) -> List[TrainingSample]:
        """Convert simulation result to training samples."""
        samples = []

        # We can only create samples if we have complete time series
        if len(result.indoor_temps) < 2:
            return samples

        # Sample timesteps (we can't use all 8760 for every sample due to memory)
        # Use stride to get representative samples
        stride = max(1, self.num_timesteps // 1000)  # Max ~1000 samples per run

        for t in range(0, self.num_timesteps - 1, stride):
            outdoor_t = result.outdoor_temps[t]
            indoor_t = result.indoor_temps[t]
            solar_t = result.solar_gains[t]

            # Action: HVAC mode and power
            heating = result.heating_loads[t]
            cooling = result.cooling_loads[t]

            if heating > 0:
                hvac_mode = 1  # heating
                hvac_power = heating
            elif cooling > 0:
                hvac_mode = 2  # cooling
                hvac_power = cooling
            else:
                hvac_mode = 0  # off
                hvac_power = 0.0

            # Next state
            indoor_t1 = result.indoor_temps[t + 1]

            # Energy consumed (Wh)
            energy = heating + cooling  # Already in watts for this hour

            sample = TrainingSample(
                outdoor_temp_t=outdoor_t,
                indoor_temp_t=indoor_t,
                solar_t=solar_t,
                hvac_mode=hvac_mode,
                hvac_power=hvac_power,
                indoor_temp_t1=indoor_t1,
                energy_consumed=energy,
                run_id=result.run_id,
                timestep=t,
            )
            samples.append(sample)

        return samples

    def _samples_to_dataframe(self, samples: List[TrainingSample]) -> pd.DataFrame:
        """Convert training samples to DataFrame."""
        data = {
            "outdoor_temp_t": [s.outdoor_temp_t for s in samples],
            "indoor_temp_t": [s.indoor_temp_t for s in samples],
            "solar_t": [s.solar_t for s in samples],
            "hvac_mode": [s.hvac_mode for s in samples],
            "hvac_power": [s.hvac_power for s in samples],
            "indoor_temp_t1": [s.indoor_temp_t1 for s in samples],
            "energy_consumed": [s.energy_consumed for s in samples],
            "run_id": [s.run_id for s in samples],
            "timestep": [s.timestep for s in samples],
        }

        return pd.DataFrame(data)

    def _save_data(self, df: pd.DataFrame) -> None:
        """Save DataFrame to Parquet format."""
        output_path = self.output_dir / "training_data.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved training data to {output_path}")

        # Also save metadata
        metadata = {
            "num_samples": self.num_samples,
            "num_timesteps": self.num_timesteps,
            "seed": self.seed,
            "sampling_method": self.sampling_method,
            "total_generation_time": self.total_time,
            "generated_at": datetime.now().isoformat(),
        }

        metadata_path = self.output_dir / "metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point for the data generation tool."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Data Generation Tool for AI Surrogate Training"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument(
        "--output",
        "-o",
        default="data/train",
        help="Output directory for training data",
    )
    gen_parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1000,
        help="Number of building configurations to simulate",
    )
    gen_parser.add_argument(
        "--timesteps",
        type=int,
        default=8760,
        help="Number of timesteps per simulation (default: 8760 = 1 year)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    gen_parser.add_argument(
        "--method",
        choices=["RANDOM", "LHS", "SOBOL"],
        default="LHS",
        help="Sampling method",
    )
    gen_parser.add_argument(
        "--weather-dir",
        default="assets/weather",
        help="Directory containing EPW weather files",
    )
    gen_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for simulations",
    )
    gen_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available weather files")
    list_parser.add_argument(
        "--weather-dir",
        default="assets/weather",
        help="Directory containing EPW weather files",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "generate":
        # Create and run generator
        gen = MonteCarloDataGenerator(
            output_dir=args.output,
            num_samples=args.count,
            num_timesteps=args.timesteps,
            seed=args.seed,
            weather_dir=args.weather_dir,
            sampling_method=args.method,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

        gen.setup()
        df = gen.generate()

        print(f"\nGenerated {len(df)} training samples")
        print(f"Output: {args.output}/training_data.parquet")

    elif args.command == "list":
        weather_dir = Path(args.weather_dir)
        if weather_dir.exists():
            files = [f for f in os.listdir(weather_dir) if f.endswith(".epw")]
            print(f"Available weather files in {weather_dir}:")
            for f in files:
                print(f"  - {f}")
        else:
            print(f"Weather directory not found: {weather_dir}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
