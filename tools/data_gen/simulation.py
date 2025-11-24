"""
Simulation execution engine.
Handles running OpenStudio/EnergyPlus workflows.
"""

import logging
import os
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
    model, weather_file_path: str, output_dir: str, run_name: str = "run"
):
    """
    Run an OpenStudio simulation.

    Args:
        model: OpenStudio Model object (or mock)
        weather_file_path: Path to EPW file
        output_dir: Directory to store results
        run_name: Name of the run subdirectory
    """

    run_dir = os.path.join(output_dir, run_name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    # Save the model to OSM
    osm_path = os.path.join(run_dir, "model.osm")
    model.save(osm_path, True)
    logger.info(f"Saved model to {osm_path}")

    # Check if we are running in Mock mode
    if isinstance(model, geometry.MockOpenStudio.model.Model):
        logger.warning("Mock model detected. Skipping actual simulation.")
        # Create dummy output files to satisfy expectations
        Path(os.path.join(run_dir, "eplusout.sql")).touch()
        Path(os.path.join(run_dir, "eplusout.csv")).touch()
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
