"""
Data Generation Tool for Fluxion Surrogate Models.

This package provides utilities to generate synthetic building performance data
using EnergyPlus and OpenStudio.
"""

__version__ = "0.2.0"

from .ashrae_140_generator import ASHRAE140CaseGenerator, save_cases_to_json
from .monte_carlo import MonteCarloDataGenerator, BuildingConfig, SimulationResult, TrainingSample

__all__ = [
    "ASHRAE140CaseGenerator",
    "save_cases_to_json",
    "MonteCarloDataGenerator",
    "BuildingConfig",
    "SimulationResult",
    "TrainingSample",
]
