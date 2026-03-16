"""
Simulation execution engine.
Handles running OpenStudio/EnergyPlus workflows.
"""

import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path

from . import geometry

logger = logging.getLogger(__name__)

# Try importing OpenStudio for RunManager check
try:
    import openstudio
except ImportError:
    openstudio = None


def run_simulation(
    model,
    weather_file_path: str,
    output_dir: str,
    run_name: str = "run",
    params: dict = None,
):
    """
    Run an OpenStudio simulation.

    Args:
        model: OpenStudio Model object (or mock)
        weather_file_path: Path to EPW file
        output_dir: Directory to store results
        run_name: Name of the run subdirectory
        params: Dictionary of simulation parameters (optional)
    """

    run_dir = os.path.join(output_dir, run_name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    # Save params if provided
    if params:
        with open(os.path.join(run_dir, "simulation_params.json"), "w") as f:
            json.dump(params, f, indent=2)

    # Save the model to OSM
    osm_path = os.path.join(run_dir, "model.osm")
    model.save(osm_path, True)
    logger.info(f"Saved model to {osm_path}")

    # Check if we are running in Mock mode
    if isinstance(model, geometry.MockOpenStudio.model.Model):
        logger.warning("Mock model detected. Skipping actual simulation.")
        # Create dummy output files to satisfy expectations
        Path(os.path.join(run_dir, "eplusout.sql")).touch()

        # Generate realistic dummy CSV output
        create_mock_csv_output(os.path.join(run_dir, "eplusout.csv"), params)
        return True

    # If we have real OpenStudio, we need to convert OSM -> IDF and run EnergyPlus.
    # The easiest way programmatically with OpenStudio python bindings is using
    # the WorkflowJSON
    # and the OSW (OpenStudio Workflow) format, or manually forwarding it.

    # Simple approach: Use OpenStudio CLI if available in path, or Python bindings.
    # Python bindings for running simulation can be complex (RunManager is
    # deprecated in older versions,
    # Workflow is newer).

    # Let's try the `openstudio` CLI command if available, as it's the standard
    # way to run OSW.
    # But we have the model object in memory.

    # Alternative: Translate to IDF and run `energyplus` CLI.
    # This requires `ForwardTranslator`.

    try:
        ft = openstudio.energyplus.ForwardTranslator()
        workspace = ft.translateModel(model)
        idf_path = os.path.join(run_dir, "in.idf")
        workspace.save(idf_path, True)
        logger.info(f"Translated to IDF: {idf_path}")

        # Now run EnergyPlus
        # Assume `energyplus` is in PATH.
        # Command: energyplus -w <weather> -d <output_dir> <idf>

        cmd = ["energyplus", "-w", weather_file_path, "-d", run_dir, idf_path]

        logger.info(f"Running EnergyPlus: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("EnergyPlus failed!")
            logger.error(result.stderr)
            return False

        logger.info("Simulation completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Simulation execution failed: {e}")
        # If translation failed, maybe we are in a partial environment.
        return False


def create_mock_csv_output(filepath, params):
    """Generates a realistic-looking CSV output file."""
    import csv
    import math

    # Default params if None
    if not params:
        params = {"u_value": 2.5, "hvac_setpoint": 21.0}

    u_value = params.get("u_value", 2.5)
    setpoint = params.get("hvac_setpoint", 21.0)

    # Headers
    headers = [
        "Date/Time",
        "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)",
        "Zone Mean Air Temperature [C](Hourly)",  # Simplification: Just one zone
        "Zone Ideal Loads Zone Total Heating Energy [J](Hourly)",
        "Zone Ideal Loads Zone Total Cooling Energy [J](Hourly)",
        "Surface Outside Face Incident Solar Radiation Amount per Area [W/m2](Hourly)",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # Generate 8760 hours
        for t in range(8760):
            # Time string
            # Format:  01/01  01:00:00
            # Simpified logic for date/time string is hard, maybe just skip or use index
            # EnergyPlus CSV usually has "01/01  01:00:00"

            # Outdoor temp (simple sine wave)
            hour_of_day = t % 24
            day_of_year = t / 24.0
            daily_cycle = math.sin(hour_of_day / 24.0 * 2.0 * math.pi - math.pi / 2.0)
            annual_cycle = -math.cos(day_of_year / 365.0 * 2.0 * math.pi)

            outdoor_temp = 10.0 + 15.0 * annual_cycle + 5.0 * daily_cycle

            # Indoor temp (floating or setpoint)
            # For Ideal Loads, indoor temp is maintained at setpoint if load > 0
            indoor_temp = setpoint  # Simplified

            # Solar
            solar = max(0.0, math.sin((hour_of_day - 6) / 12.0 * math.pi) * 800.0)
            if solar < 0:
                solar = 0

            # Loads (Joules)
            # Q = U * A * dT * time
            # Assume Area = 100 for mock
            area = 100.0
            dt = indoor_temp - outdoor_temp
            load_rate_watts = (
                u_value * area * dt - solar * 0.1 * area
            )  # Solar gain subtraction

            heating_load = 0.0
            cooling_load = 0.0

            if load_rate_watts > 0:
                heating_load = load_rate_watts * 3600.0  # Watts to Joules
            else:
                cooling_load = -load_rate_watts * 3600.0

            writer.writerow(
                [
                    f"{t}",  # Placeholder for date
                    f"{outdoor_temp:.2f}",
                    f"{indoor_temp:.2f}",
                    f"{heating_load:.2f}",
                    f"{cooling_load:.2f}",
                    f"{solar:.2f}",
                ]
            )
