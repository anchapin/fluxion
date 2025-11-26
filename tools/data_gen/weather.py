"""
Weather file management utilities.
"""

import logging
import os
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard weather files from EnergyPlus GitHub repository (or similar stable source)
# Using the EnergyPlus/Weatherdata repository for reliable downloads
BASE_URL = "https://github.com/NREL/EnergyPlus/raw/develop/weather"
# Fallback to a known stable URL or specific repo if needed.
# For now, let's use the official EnergyPlus auxiliary repo or just direct links
# to a few TMY3 files.
# Actually, the most reliable source is usually the DOE or a dedicated repo.
# Let's use a few hardcoded URLs for MVP.

WEATHER_FILES = {
    "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw": "https://raw.githubusercontent.com/NREL/EnergyPlus/develop/weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw": "https://raw.githubusercontent.com/NREL/EnergyPlus/develop/weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
    "USA_FL_Miami.Intl.AP.722020_TMY3.epw": "https://raw.githubusercontent.com/NREL/EnergyPlus/develop/weather/USA_FL_Miami.Intl.AP.722020_TMY3.epw",
}


def download_standard_files(output_dir: str):
    """
    Download standard weather files to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    for name, url in WEATHER_FILES.items():
        destination = os.path.join(output_dir, name)
        if os.path.exists(destination):
            logger.info(f"File exists, skipping: {destination}")
            continue

        logger.info(f"Downloading {name} from {url}...")
        try:
            urllib.request.urlretrieve(url, destination)
            logger.info(f"Successfully downloaded {name}")
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")


def get_weather_file_path(output_dir: str, name: str) -> str:
    """
    Get the full path to a weather file, ensuring it exists.
    If it doesn't exist locally, attempts to download it.
    """
    path = os.path.join(output_dir, name)
    if os.path.exists(path):
        return path

    # Try to find it in the known list
    if name in WEATHER_FILES:
        download_standard_files(output_dir)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Weather file {name} not found in {output_dir} and could not be downloaded."
    )
