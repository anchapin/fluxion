#!/usr/bin/env python3
"""
Synthetic Training Data Generation for Surrogate Models (v2.1)

This script generates diverse, physics-based training data for Phase 4 surrogate
model training covering:
- Zone thermal solver (6R2C network)
- Solar gain calculation
- Conduction heat transfer

Prerequisites:
- Phase 1 (v1.3) must achieve ≥80% ASHRAE 140 blind validation pass rate
- Fluxion physics modules must be validated

Usage:
    python scripts/generate_training_data.py --n-scenarios 10000 --output-dir data/synthetic/v2.1

Output:
    data/synthetic/v2.1/
    ├── zone_thermal/
    │   ├── train.parquet
    │   ├── val.parquet
    │   ├── test.parquet
    │   └── metadata.json
    ├── solar_gain/
    │   ├── train.parquet
    │   ├── val.parquet
    │   ├── test.parquet
    │   └── metadata.json
    └── conduction/
        ├── train.parquet
        ├── val.parquet
        ├── test.parquet
        └── metadata.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ClimateZone(str, Enum):
    ZONE_1A = "1A"
    ZONE_2A = "2A"
    ZONE_2B = "2B"
    ZONE_3A = "3A"
    ZONE_3B = "3B"
    ZONE_3C = "3C"
    ZONE_4A = "4A"
    ZONE_4B = "4B"
    ZODE_4C = "4C"
    ZONE_5A = "5A"
    ZONE_5B = "5B"
    ZONE_6A = "6A"
    ZONE_6B = "6B"
    ZONE_7 = "7"
    ZONE_8 = "8"


class HvacMode(str, Enum):
    HEATING = "heating"
    COOLING = "cooling"
    OFF = "off"
    AUTO = "auto"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ZoneThermalRecord:
    """Single timestep record for zone thermal model training."""

    scenario_id: str
    timestep: int

    # Input features (match SurrogateInputs from surrogate.rs)
    exterior_temp: float  # °C
    zone_temp: float  # °C
    solar_rad: float  # W/m²
    humidity: float  # %
    occupancy: float  # fraction 0-1
    climate_zone: str

    # Additional context
    hour_of_day: int  # 0-23
    day_of_year: int  # 1-365
    hvac_mode: str

    # Building characteristics
    zone_area_m2: float
    zone_volume_m3: float
    window_ratio: float  # fraction
    wall_u_value: float  # W/m²K
    thermal_mass_mj_k: float

    # Training targets
    zone_temp_next: float  # °C - next timestep zone temp
    hvac_power: float  # W - heating (+) / cooling (-)
    energy_storage: float  # W - thermal mass charge/discharge


@dataclass
class SolarGainRecord:
    """Single timestep record for solar gain model training."""

    scenario_id: str
    timestep: int

    # Solar position inputs
    latitude: float  # degrees
    longitude: float  # degrees
    hour_of_day: float  # decimal hours
    day_of_year: int

    # Weather inputs
    dni: float  # W/m² Direct Normal Irradiance
    dhi: float  # W/m² Diffuse Horizontal Irradiance
    ghi: float  # W/m² Global Horizontal Irradiance
    ground_albedo: float  # fraction

    # Surface properties
    surface_tilt: float  # degrees from horizontal
    surface_azimuth: float  # degrees from north
    surface_area: float  # m²

    # Output targets
    beam_gain: float  # W
    diffuse_gain: float  # W
    ground_reflected_gain: float  # W
    total_gain: float  # W
    sol_air_temp: float  # °C


@dataclass
class ConductionRecord:
    """Single timestep record for conduction model training."""

    scenario_id: str
    timestep: int

    # Wall construction
    wall_type: str
    num_layers: int

    # Boundary conditions
    t_int: float  # °C interior surface temp
    t_ext: float  # °C exterior surface temp (or sol-air)
    h_int: float  # W/m²K interior heat transfer coefficient
    h_ext: float  # W/m²K exterior heat transfer coefficient

    # Wall properties (aggregated)
    wall_r_value: float  # m²K/W
    wall_thermal_mass: float  # J/m²K

    # Output targets
    heat_flux_inward: float  # W/m² positive = inward
    heat_flux_outward: float  # W/m² positive = outward
    energy_storage_rate: float  # W/m²
    t_surface_int: float  # °C interior surface temperature


@dataclass
class BuildingConfig:
    """Building configuration for synthetic scenario generation."""

    zone_area_m2: float = 150.0  # 20-500 m²
    zone_volume_m3: float = 405.0  # derived
    window_ratio: float = 0.25  # 0.1-0.5
    ceiling_height_m: float = 2.7  # 2.4-4.0
    wall_u_value: float = 0.5  # W/m²K 0.2-2.0
    roof_u_value: float = 0.3  # W/m²K 0.15-0.8
    floor_u_value: float = 0.4  # W/m²K
    lighting_density: float = 10.0  # W/m² 5-20
    equipment_density: float = 12.0  # W/m² 5-25
    occupancy_density: float = 0.1  # persons/m² 0.05-0.2
    heating_setpoint: float = 20.0  # °C 15-22
    cooling_setpoint: float = 24.0  # °C 22-30
    deadband: float = 2.0  # K 2-5

    def thermal_mass_mj_k(self) -> float:
        """Approximate thermal mass in MJ/K."""
        # Simplified: thermal mass proportional to zone volume and surface area
        surface_area = (
            2 * (self.zone_area_m2)  # floor + ceiling
            + 2 * (self.zone_area_m2 ** 0.5 * self.ceiling_height_m)  # two walls
            + 2 * (self.zone_area_m2 ** 0.5 * self.ceiling_height_m)  # other two
        )
        return surface_area * 0.05  # rough approximation MJ/K per m²


@dataclass
class WeatherProfile:
    """Weather profile for synthetic scenario generation."""

    profile_type: str  # "temperate", "continental", "tropical", "desert", "coastal"
    season_factor: float = 0.5  # 0.0 = summer, 1.0 = winter
    transient_probability: float = 0.1  # probability of weather front


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================


def latin_hypercube_sample(
    bounds: List[Tuple[str, float, float]],
    n_samples: int,
    seed: int = 42,
) -> List[dict]:
    """
    Generate Latin Hypercube Samples for diverse parameter coverage.

    Args:
        bounds: List of (parameter_name, min, max) tuples
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of parameter dictionaries
    """
    rng = np.random.default_rng(seed)
    n_params = len(bounds)

    # Generate LHS points in [0, 1]^n_params
    points = rng.uniform(size=(n_samples, n_params))

    # Shift to align with intervals
    for j in range(n_params):
        points[:, j] = (points[:, j] + rng.uniform(size=n_samples)) / n_samples

    # Scale to bounds
    samples = []
    for i in range(n_samples):
        sample = {}
        for j, (name, low, high) in enumerate(bounds):
            sample[name] = low + (high - low) * points[i, j]
        samples.append(sample)

    return samples


# =============================================================================
# Weather Generation
# =============================================================================


def generate_weather_sequence(
    profile: WeatherProfile,
    n_timesteps: int,
    dt_hours: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate realistic weather sequence using profile statistics.

    Args:
        profile: Weather profile type
        n_timesteps: Number of hourly timesteps (8760 for annual)
        dt_hours: Timestep in hours (1.0 for hourly)
        seed: Random seed

    Returns:
        Array of (exterior_temp, dni, dhi, ghi) per timestep
    """
    rng = np.random.default_rng(seed)

    # Profile-specific statistics
    profile_stats = {
        "temperate": {"temp_mean": 15.0, "temp_std": 8.0, "solar_max": 800},
        "continental": {"temp_mean": 12.0, "temp_std": 15.0, "solar_max": 1000},
        "tropical": {"temp_mean": 28.0, "temp_std": 5.0, "solar_max": 900},
        "desert": {"temp_mean": 25.0, "temp_std": 12.0, "solar_max": 1100},
        "coastal": {"temp_mean": 18.0, "temp_std": 6.0, "solar_max": 850},
    }

    stats = profile_stats.get(profile.profile_type, profile_stats["temperate"])

    # Base temperature with seasonal and diurnal variation
    hours = np.arange(n_timesteps) * dt_hours
    day_of_year = (hours // 24) % 365
    hour_of_day = hours % 24

    # Seasonal component (cosine wave)
    seasonal = stats["temp_std"] * np.cos(2 * np.pi * (day_of_year - 172) / 365)

    # Diurnal component (sine wave peaking at 3pm)
    diurnal = 5.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)

    # Random daily variation (autocorrelated)
    daily_residual = rng.normal(0, 2.0, 365)
    daily_residual = np.convolve(
        daily_residual, np.exp(-np.arange(5) / 1.5), mode="full"
    )[:365]
    daily_residual = daily_residual / daily_residual.std() * 2.0

    temp = stats["temp_mean"] + seasonal + diurnal

    # Add weather fronts (rare transient events)
    for _ in range(int(n_timesteps * profile.transient_probability * 0.1)):
        front_day = rng.integers(0, 365)
        front_hour = rng.integers(0, 24)
        front_idx = int(front_day * 24 + front_hour)
        if front_idx < n_timesteps:
            front_magnitude = rng.choice([-10, 10])
            for i in range(front_idx, min(front_idx + 24, n_timesteps)):
                decay = np.exp(-(i - front_idx) / 6)
                temp[i] += front_magnitude * decay

    # Solar radiation (simplified)
    solar_factor = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    solar_factor *= np.maximum(0, np.cos(2 * np.pi * (day_of_year - 172) / 365))

    dni = stats["solar_max"] * solar_factor * rng.uniform(0.7, 1.0, n_timesteps)
    dhi = stats["solar_max"] * 0.3 * solar_factor * rng.uniform(0.5, 1.0, n_timesteps)
    ghi = dni * 0.5 + dhi

    return np.column_stack([temp, dni, dhi, ghi])


# =============================================================================
# Zone Thermal Simulation (Simplified)
# =============================================================================


def simulate_zone_thermal(
    building: BuildingConfig,
    weather: np.ndarray,
    occupancy_schedule: np.ndarray,
    seed: int = 42,
) -> List[ZoneThermalRecord]:
    """
    Simulate zone thermal dynamics for one scenario.

    This is a simplified 6R2C thermal network simulation.
    For full fidelity, use the validated Fluxion physics engine.

    Args:
        building: Building configuration
        weather: (n_timesteps, 4) array of (temp, dni, dhi, ghi)
        occupancy_schedule: (n_timesteps,) array of occupancy fractions
        seed: Random seed

    Returns:
        List of ZoneThermalRecord for each timestep
    """
    n_timesteps = len(weather)
    rng = np.random.default_rng(seed)

    # Thermal model parameters (simplified 6R2C)
    C_mass = building.thermal_mass_mj_k() * 1e6  # J/K
    C_air = building.zone_volume_m3 * 1200  # J/K (air thermal mass)
    R_wall = 1.0 / building.wall_u_value  # K/W
    R_window = 0.01 / (building.window_ratio * building.zone_area_m2)  # K/W
    R_vent = 0.05 / building.zone_volume_m3  # K/W (simplified)
    R_total = 1.0 / (1.0 / R_wall + 1.0 / R_window + 1.0 / R_vent)

    # Initialize state
    T_zone = 20.0  # °C initial
    T_mass = 20.0  # °C initial

    records = []
    for t in range(n_timesteps):
        T_ext = weather[t, 0]
        solar_rad = weather[t, 3] * building.window_ratio * building.zone_area_m2 * 0.7
        occupancy = occupancy_schedule[t]
        internal_gains = (
            building.lighting_density
            + building.equipment_density
            + occupancy * 100
        ) * building.zone_area_m2

        # HVAC control
        if T_zone < building.heating_setpoint - building.deadband / 2:
            hvac_mode = HvacMode.HEATING.value
            hvac_power = 5000.0  # W (simplified)
        elif T_zone > building.cooling_setpoint + building.deadband / 2:
            hvac_mode = HvacMode.COOLING.value
            hvac_power = -5000.0  # W (cooling)
        else:
            hvac_mode = HvacMode.OFF.value
            hvac_power = 0.0

        # 6R2C simplified update
        Q_in = (T_ext - T_zone) / R_total + solar_rad + internal_gains + hvac_power
        Q_storage = C_mass * (T_mass - T_zone)

        # Time step update (1 hour)
        dt = 3600.0
        T_mass_new = T_mass + dt / C_mass * (Q_in - Q_storage)
        T_zone_new = T_zone + dt / C_air * (Q_in + Q_storage)

        # Clamp to reasonable bounds
        T_zone_new = np.clip(T_zone_new, -50, 80)
        T_mass_new = np.clip(T_mass_new, -50, 80)

        hour_of_day = t % 24
        day_of_year = (t // 24) % 365 + 1

        record = ZoneThermalRecord(
            scenario_id="",  # Set by caller
            timestep=t,
            exterior_temp=T_ext,
            zone_temp=T_zone,
            solar_rad=solar_rad,
            humidity=50.0,  # Simplified
            occupancy=occupancy,
            climate_zone="4A",  # Default, set by caller
            hour_of_day=hour_of_day,
            day_of_year=day_of_year,
            hvac_mode=hvac_mode,
            zone_area_m2=building.zone_area_m2,
            zone_volume_m3=building.zone_volume_m3,
            window_ratio=building.window_ratio,
            wall_u_value=building.wall_u_value,
            thermal_mass_mj_k=building.thermal_mass_mj_k(),
            zone_temp_next=T_zone_new,
            hvac_power=hvac_power,
            energy_storage=Q_storage,
        )

        records.append(record)

        # Update state
        T_zone = T_zone_new
        T_mass = T_mass_new

    return records


# =============================================================================
# Solar Gain Simulation
# =============================================================================


def calculate_solar_position(
    lat: float, lon: float, day_of_year: int, hour: float
) -> Tuple[float, float, float]:
    """
    Calculate solar position (simplified).

    Returns:
        (altitude, azimuth, zenith) in degrees
    """
    # Day angle
    gamma = 2 * np.pi * (day_of_year - 1) / 365

    # Declination
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    # Hour angle
    hour_angle = 15 * (hour - 12)  # degrees

    # Latitude in radians
    lat_rad = np.radians(lat)

    # Solar altitude
    sin_altitude = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(np.radians(hour_angle))
    )
    altitude = np.degrees(np.arcsin(sin_altitude))

    # Solar azimuth
    cos_azimuth = (
        np.sin(declination) - np.sin(lat_rad) * sin_altitude
    ) / (np.cos(lat_rad) * np.cos(np.radians(altitude)))
    azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))
    if hour > 12:
        azimuth = 360 - azimuth

    zenith = 90 - altitude

    return altitude, azimuth, zenith


def simulate_solar_gain(
    latitude: float,
    longitude: float,
    weather: np.ndarray,
    surface_configs: List[dict],
    seed: int = 42,
) -> List[SolarGainRecord]:
    """
    Simulate solar gains for multiple surface configurations.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        weather: (n_timesteps, 4) array of (temp, dni, dhi, ghi)
        surface_configs: List of surface configurations
        seed: Random seed

    Returns:
        List of SolarGainRecord for each timestep and surface
    """
    n_timesteps = len(weather)
    rng = np.random.default_rng(seed)
    records = []

    for t in range(n_timesteps):
        day_of_year = (t // 24) % 365 + 1
        hour_of_day = t % 24 + rng.uniform(0, 1)  # Add sub-hourly variation

        altitude, azimuth, zenith = calculate_solar_position(
            latitude, longitude, day_of_year, hour_of_day
        )

        dni = weather[t, 1]
        dhi = weather[t, 2]
        ghi = weather[t, 3]
        ground_albedo = 0.2  # Typical grass albedo

        for surf_idx, config in enumerate(surface_configs):
            tilt = config["tilt"]
            surf_azimuth = config["azimuth"]
            area = config["area"]

            # Incident angle (simplified)
            incidence = np.cos(np.radians(tilt)) * np.cos(np.radians(zenith)) + np.sin(
                np.radians(tilt)
            ) * np.sin(np.radians(zenith)) * np.cos(np.radians(azimuth - surf_azimuth))
            incidence = np.clip(incidence, 0, 1)

            # Beam gain
            beam_gain = dni * incidence * area * 0.7  # 0.7 = glass transmission

            # Diffuse gain (isotropic sky diffuse)
            diffuse_factor = (1 + np.cos(np.radians(tilt))) / 2
            diffuse_gain = dhi * diffuse_factor * area * 0.7

            # Ground reflected
            ground_factor = (1 - np.cos(np.radians(tilt))) / 2
            ground_reflected_gain = ghi * ground_albedo * ground_factor * area * 0.7

            # Sol-air temperature (simplified)
            sol_air_temp = weather[t, 0] + (beam_gain + diffuse_gain) / (
                area * 10.0
            )  # Rough convective transfer

            record = SolarGainRecord(
                scenario_id="",
                timestep=t,
                latitude=latitude,
                longitude=longitude,
                hour_of_day=hour_of_day,
                day_of_year=day_of_year,
                dni=dni,
                dhi=dhi,
                ghi=ghi,
                ground_albedo=ground_albedo,
                surface_tilt=tilt,
                surface_azimuth=surf_azimuth,
                surface_area=area,
                beam_gain=beam_gain,
                diffuse_gain=diffuse_gain,
                ground_reflected_gain=ground_reflected_gain,
                total_gain=beam_gain + diffuse_gain + ground_reflected_gain,
                sol_air_temp=sol_air_temp,
            )
            records.append(record)

    return records


# =============================================================================
# Conduction Simulation
# =============================================================================


def simulate_conduction(
    wall_type: str,
    weather: np.ndarray,
    seed: int = 42,
) -> List[ConductionRecord]:
    """
    Simulate conduction through wall construction.

    Args:
        wall_type: Wall construction type
        weather: (n_timesteps, 4) array of (temp, dni, dhi, ghi)
        seed: Random seed

    Returns:
        List of ConductionRecord for each timestep
    """
    n_timesteps = len(weather)

    # Wall properties
    wall_specs = {
        "concrete_200mm": {
            "r_value": 0.12,
            "thermal_mass": 460000,  # J/m²K
            "layers": 1,
        },
        "lightweight": {
            "r_value": 2.1,
            "thermal_mass": 30000,  # J/m²K
            "layers": 3,
        },
        "medium": {
            "r_value": 1.0,
            "thermal_mass": 150000,  # J/m²K
            "layers": 2,
        },
    }

    spec = wall_specs.get(wall_type, wall_specs["medium"])
    r_value = spec["r_value"]
    C_wall = spec["thermal_mass"]

    # Initialize state
    T_surface_int = 20.0
    T_surface_ext = weather[0, 0]

    records = []
    for t in range(n_timesteps):
        T_ext = weather[t, 0]
        h_ext = 25.0  # W/m²K exterior convective coefficient
        h_int = 8.0  # W/m²K interior convective coefficient

        # Conduction through wall (steady-state for simplicity)
        r_total = r_value + 1 / h_ext + 1 / h_int
        heat_flux = (T_ext - T_surface_int) / r_total

        # Storage rate (simplified)
        energy_storage_rate = C_wall * (T_surface_ext - T_surface_int) / 3600

        hour_of_day = t % 24

        record = ConductionRecord(
            scenario_id="",
            timestep=t,
            wall_type=wall_type,
            num_layers=spec["layers"],
            t_int=T_surface_int,
            t_ext=T_ext,
            h_int=h_int,
            h_ext=h_ext,
            wall_r_value=r_value,
            wall_thermal_mass=C_wall,
            heat_flux_inward=heat_flux,
            heat_flux_outward=-heat_flux,
            energy_storage_rate=energy_storage_rate,
            t_surface_int=T_surface_int,
        )

        records.append(record)

        # Update surface temperatures
        T_surface_ext = T_ext
        T_surface_int = T_surface_int + heat_flux / h_int * 0.1

    return records


# =============================================================================
# Scenario Generation
# =============================================================================


def generate_scenarios(
    n_scenarios: int,
    climate_zones: List[str],
    seed: int = 42,
) -> List[dict]:
    """
    Generate diverse building + weather scenarios using LHS.

    Args:
        n_scenarios: Number of scenarios to generate
        climate_zones: List of climate zones to include
        seed: Random seed

    Returns:
        List of scenario dictionaries
    """

    def building_bounds():
        return [
            ("zone_area_m2", 20.0, 500.0),
            ("window_ratio", 0.1, 0.5),
            ("wall_u_value", 0.2, 2.0),
            ("lighting_density", 5.0, 20.0),
            ("equipment_density", 5.0, 25.0),
            ("occupancy_density", 0.05, 0.2),
            ("heating_setpoint", 15.0, 22.0),
            ("cooling_setpoint", 22.0, 30.0),
        ]

    samples = latin_hypercube_sample(building_bounds(), n_scenarios, seed)

    profiles = ["temperate", "continental", "tropical", "desert", "coastal"]

    scenarios = []
    for i, sample in enumerate(samples):
        rng = np.random.default_rng(seed + i)
        building = BuildingConfig(
            zone_area_m2=sample["zone_area_m2"],
            zone_volume_m3=sample["zone_area_m2"] * 2.7,
            window_ratio=sample["window_ratio"],
            wall_u_value=sample["wall_u_value"],
            lighting_density=sample["lighting_density"],
            equipment_density=sample["equipment_density"],
            occupancy_density=sample["occupancy_density"],
            heating_setpoint=sample["heating_setpoint"],
            cooling_setpoint=sample["cooling_setpoint"],
        )

        weather_profile = WeatherProfile(
            profile_type=rng.choice(profiles),
            season_factor=rng.uniform(0, 1),
            transient_probability=rng.uniform(0.05, 0.2),
        )

        scenario = {
            "id": f"scenario_{i:05d}",
            "building": building,
            "weather_profile": weather_profile,
            "climate_zone": rng.choice(climate_zones),
            "seed": seed + i,
        }
        scenarios.append(scenario)

    return scenarios


# =============================================================================
# Main Generation Functions
# =============================================================================


def generate_zone_thermal_dataset(
    scenarios: List[dict],
    output_dir: Path,
    n_timesteps: int = 8760,
) -> dict:
    """Generate zone thermal training dataset."""
    logger.info(f"Generating zone thermal dataset for {len(scenarios)} scenarios...")

    all_records = []

    for scenario in tqdm(scenarios, desc="Zone Thermal"):
        # Generate weather for this scenario
        weather = generate_weather_sequence(
            scenario["weather_profile"], n_timesteps, seed=scenario["seed"]
        )

        # Generate occupancy schedule (simplified office pattern)
        hours = np.arange(n_timesteps)
        day_of_year = (hours // 24) % 365
        hour_of_day = hours % 24
        is_weekend = (day_of_year % 7) >= 5
        occupancy = np.where(
            is_weekend,
            0.1 * np.ones(n_timesteps),  # Weekend: low occupancy
            np.where(
                (hour_of_day >= 8) & (hour_of_day <= 18),
                0.8 * np.ones(n_timesteps),  # Business hours
                0.1 * np.ones(n_timesteps),  # Other hours
            ),
        )

        # Simulate
        records = simulate_zone_thermal(
            scenario["building"], weather, occupancy, seed=scenario["seed"]
        )

        # Set scenario ID
        for record in records:
            record.scenario_id = scenario["id"]
            record.climate_zone = scenario["climate_zone"]

        all_records.extend(records)

    # Convert to dict format
    records_dict = [asdict(r) for r in all_records]

    logger.info(f"Generated {len(records_dict)} zone thermal records")
    return {"records": records_dict, "n_scenarios": len(scenarios)}


def generate_solar_gain_dataset(
    scenarios: List[dict],
    output_dir: Path,
    n_timesteps: int = 8760,
) -> dict:
    """Generate solar gain training dataset."""
    logger.info(f"Generating solar gain dataset for {len(scenarios)} scenarios...")

    # Standard surface configurations
    surface_configs = [
        {"name": "south_wall", "tilt": 90, "azimuth": 180, "area": 20.0},
        {"name": "east_wall", "tilt": 90, "azimuth": 90, "area": 15.0},
        {"name": "west_wall", "tilt": 90, "azimuth": 270, "area": 15.0},
        {"name": "north_wall", "tilt": 90, "azimuth": 0, "area": 20.0},
        {"name": "roof", "tilt": 0, "azimuth": 0, "area": 50.0},
    ]

    all_records = []

    for scenario in tqdm(scenarios[: min(len(scenarios), 2000)], desc="Solar Gain"):
        # Generate weather
        weather = generate_weather_sequence(
            scenario["weather_profile"], n_timesteps, seed=scenario["seed"]
        )

        # Lat/lon based on climate zone (simplified)
        climate_latlons = {
            "1A": (25.0, -80.0),  # Miami
            "2A": (30.0, -90.0),  # New Orleans
            "3A": (33.0, -96.0),  # Dallas
            "4A": (40.0, -105.0),  # Denver
            "5A": (42.0, -88.0),  # Chicago
            "6A": (45.0, -93.0),  # Minneapolis
            "7": (45.0, -100.0),  # Montana
            "8": (64.0, -147.0),  # Fairbanks
        }
        lat, lon = climate_latlons.get(scenario["climate_zone"], (40.0, -105.0))

        records = simulate_solar_gain(
            lat, lon, weather, surface_configs, seed=scenario["seed"]
        )

        for record in records:
            record.scenario_id = scenario["id"]

        all_records.extend(records)

    records_dict = [asdict(r) for r in all_records]
    logger.info(f"Generated {len(records_dict)} solar gain records")
    return {"records": records_dict, "n_scenarios": min(len(scenarios), 2000)}


def generate_conduction_dataset(
    scenarios: List[dict],
    output_dir: Path,
    n_timesteps: int = 8760,
) -> dict:
    """Generate conduction training dataset."""
    logger.info(f"Generating conduction dataset for {len(scenarios)} scenarios...")

    wall_types = ["concrete_200mm", "lightweight", "medium"]
    all_records = []

    for scenario in tqdm(scenarios[: min(len(scenarios), 1000)], desc="Conduction"):
        weather = generate_weather_sequence(
            scenario["weather_profile"], n_timesteps, seed=scenario["seed"]
        )

        for wall_type in wall_types:
            records = simulate_conduction(wall_type, weather, seed=scenario["seed"])

            for record in records:
                record.scenario_id = f"{scenario['id']}_{wall_type}"

            all_records.extend(records)

    records_dict = [asdict(r) for r in all_records]
    logger.info(f"Generated {len(records_dict)} conduction records")
    return {"records": records_dict, "n_scenarios": min(len(scenarios), 1000)}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for surrogate models"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic/v2.1"),
        help="Output directory for generated datasets",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=5000,
        help="Number of scenarios to generate",
    )
    parser.add_argument(
        "--timesteps-per-scenario",
        type=int,
        default=8760,
        help="Number of timesteps per scenario (default: 8760 = 1 year hourly)",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="zone_thermal,solar_gain,conduction",
        help="Comma-separated components to generate",
    )
    parser.add_argument(
        "--climate-zones",
        type=str,
        default="1A,2A,3A,4A,5A,6A,7,8",
        help="Comma-separated ASHRAE climate zones",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Synthetic Training Data Generation v2.1")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Scenarios: {args.n_scenarios}")
    logger.info(f"Timesteps per scenario: {args.timesteps_per_scenario}")
    logger.info(f"Components: {args.components}")
    logger.info(f"Climate zones: {args.climate_zones}")
    logger.info(f"Seed: {args.seed}")

    # Parse components and climate zones
    components = args.components.split(",")
    climate_zones = args.climate_zones.split(",")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate scenarios
    logger.info("\nGenerating scenarios with LHS sampling...")
    scenarios = generate_scenarios(
        n_scenarios=args.n_scenarios,
        climate_zones=climate_zones,
        seed=args.seed,
    )

    # Generate requested components
    generation_start = datetime.now(timezone.utc)

    if "zone_thermal" in components:
        logger.info("\n" + "=" * 40)
        logger.info("Generating ZONE THERMAL dataset")
        logger.info("=" * 40)
        data = generate_zone_thermal_dataset(
            scenarios, args.output_dir / "zone_thermal", args.timesteps_per_scenario
        )
        # Save metadata
        metadata = {
            "component": "zone_thermal",
            "n_scenarios": data["n_scenarios"],
            "timesteps_per_scenario": args.timesteps_per_scenario,
            "n_records": len(data["records"]),
            "seed": args.seed,
            "generated_at": generation_start.isoformat(),
            "features": list(data["records"][0].keys()),
        }
        with open(args.output_dir / "zone_thermal" / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    if "solar_gain" in components:
        logger.info("\n" + "=" * 40)
        logger.info("Generating SOLAR GAIN dataset")
        logger.info("=" * 40)
        data = generate_solar_gain_dataset(
            scenarios, args.output_dir / "solar_gain", args.timesteps_per_scenario
        )
        metadata = {
            "component": "solar_gain",
            "n_scenarios": data["n_scenarios"],
            "timesteps_per_scenario": args.timesteps_per_scenario,
            "n_records": len(data["records"]),
            "seed": args.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "features": list(data["records"][0].keys()),
        }
        with open(args.output_dir / "solar_gain" / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    if "conduction" in components:
        logger.info("\n" + "=" * 40)
        logger.info("Generating CONDUCTION dataset")
        logger.info("=" * 40)
        data = generate_conduction_dataset(
            scenarios, args.output_dir / "conduction", args.timesteps_per_scenario
        )
        metadata = {
            "component": "conduction",
            "n_scenarios": data["n_scenarios"],
            "timesteps_per_scenario": args.timesteps_per_scenario,
            "n_records": len(data["records"]),
            "seed": args.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "features": list(data["records"][0].keys()),
        }
        with open(args.output_dir / "conduction" / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    generation_time = (datetime.now(timezone.utc) - generation_start).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info(f"Generation complete in {generation_time:.1f} seconds")
    logger.info(f"Output written to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
