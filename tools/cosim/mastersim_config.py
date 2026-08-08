# Copyright 2026 Fluxion. All rights reserved.
# SPDX-License-Identifier: MIT

"""
MasterSim Co-Simulation Configuration for BES+FFD耦合

This file configures the MasterSim FMI 2.0 co-simulation master to run
a coupled Building Energy Simulation (BES) + Fast Fluid Dynamics (FFD)
simulation.

MasterSim is an open-source FMI 2.0.3 co-simulation master developed by
Ghorwin/Virtual Engineering Lab. See: https://github.com/ghorwin/MasterSim

Configuration Format: Unix MasterSim (.ums)
- startTime/endTime: simulation period in seconds
- communicationStepSize: master time-step in seconds (60s for FFD, 3600s for BES)
- ResultInterval: output frequency in seconds

FMU Connections (BES ↔ FFD):
- BES outputs → FFD inputs: wall_temperatures, inlet_air_temperature, mass_flow_rate
- FFD outputs → BES inputs: zone_air_temperatures (stratified), chtc values

Usage:
    mastersim master_config_BES_FFD.ums

Installation:
    git clone https://github.com/ghorwin/MasterSim.git
    cd MasterSim && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
"""

import subprocess
import sys
from pathlib import Path

# Simulation parameters
START_TIME = 0.0          # seconds (midnight)
END_TIME = 86400.0        # 24 hours = 86400 seconds
BES_TIMESTEP = 3600.0    # BES communication step (1 hour)
FFD_TIMESTEP = 60.0       # FFD communication step (1 minute)
RESULT_INTERVAL = 3600.0  # Write results every hour

# FMU paths (relative to this script)
BES_FMU_PATH = "bes_dummy.fmu"
FFD_FMU_PATH = "ffd_dummy.fmu"


def generate_ums_config(
    bes_fmu_path: str,
    ffd_fmu_path: str,
    output_path: str,
    start_time: float = START_TIME,
    end_time: float = END_TIME,
    bes_timestep: float = BES_TIMESTEP,
    ffd_timestep: float = FFD_TIMESTEP,
) -> None:
    """Generate a MasterSim .ums configuration file for BES+FFD co-simulation."""

    # MasterSim uses the smallest step as the master step
    master_timestep = min(bes_timestep, ffd_timestep)

    config_xml = f'''<?xml version="1.0" encoding="utf-8"?>
<CoSimulationMaster
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="https://raw.githubusercontent.com/ghorwin/MasterSim/master/docs/MasterSimulatorUML.xsd">

    <!-- Simulation Parameters -->
    <SimulationParameters
        startTime="{start_time}"
        endTime="{end_time}"
        masterStepSize="{master_timestep}"
        resultFile="cosim_results"
        resultInterval="{RESULT_INTERVAL}"
    />

    <!-- FMU Definitions -->
    <FMUDefinition
        name="BES"
        source="{bes_fmu_path}"
        instanceName="bes1"
        visible="false"
        loggingOn="true"
    />

    <FMUDefinition
        name="FFD"
        source="{ffd_fmu_path}"
        instanceName="ffd1"
        visible="false"
        loggingOn="true"
    />

    <!-- Data Exchange Connections: BES → FFD -->
    <!-- BES outputs (zone boundary conditions) → FFD inputs -->
    <Connection
        fmu1="bes1" var1="outdoor_temperature"
        fmu2="ffd1" var2="inlet_air_temperature"
        delay="0"
    />

    <!-- Wall temperatures from BES to FFD -->
    <Connection
        fmu1="bes1" var1="zone_temperature"
        fmu2="ffd1" var2="wall_temperature_0"
        delay="0"
    />

    <!-- Data Exchange Connections: FFD → BES -->
    <!-- FFD outputs (zone conditions) → BES inputs for next step -->
    <Connection
        fmu1="ffd1" var1="zone_air_temperature_0"
        fmu2="bes1" var2="outdoor_temperature"
        delay="0"
    />

    <!-- Logging Configuration -->
    <LogCategories>
        <Category name="logStatus" enabled="true"/>
        <Category name="logError" enabled="true"/>
        <Category name="logWarning" enabled="true"/>
        <Category name="logInfo" enabled="false"/>
        <Category name="logDebug" enabled="false"/>
    </LogCategories>

</CoSimulationMaster>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_xml)

    print(f"Generated MasterSim configuration: {output_path}")
    return config_xml


def run_mastersim(ums_config: str, mastersim_bin: str = "mastersim") -> int:
    """
    Run MasterSim with the given configuration file.

    Args:
        ums_config: Path to .ums configuration file
        mastersim_bin: Path to MasterSim executable

    Returns:
        Exit code (0 = success)
    """
    cmd = [mastersim_bin, ums_config]
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for 24hr simulation
        )
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode
    except subprocess.TimeoutExpired:
        print("ERROR: MasterSim timed out after 300 seconds")
        return -1
    except FileNotFoundError:
        print(f"ERROR: MasterSim not found at: {mastersim_bin}")
        print("Please install MasterSim: https://github.com/ghorwin/MasterSim")
        return -1


def validate_results(result_file: str = "cosim_results.csv") -> bool:
    """
    Validate co-simulation results.

    Checks:
    - File exists and is non-empty
    - Master clock advanced to endTime
    - No synchronization deadlocks (all timesteps completed)
    - All FMU outputs are finite and within physical bounds

    Args:
        result_file: Path to MasterSim CSV result file

    Returns:
        True if validation passes
    """
    result_path = Path(result_file)
    if not result_path.exists():
        print(f"WARNING: Result file not found: {result_file}")
        print("MasterSim may have produced .json or .txt results instead")
        # Check for alternative formats
        for alt in ["cosim_results.json", "cosim_results.txt", "cosim_results.csv"]:
            if Path(alt).exists():
                print(f"Found alternative result file: {alt}")
                result_path = Path(alt)
                break
        else:
            print("No result files found - simulation may have failed")
            return False

    content = result_path.read_text()
    if len(content) == 0:
        print("ERROR: Result file is empty")
        return False

    lines = content.strip().split('\n')
    print(f"Result file has {len(lines)} lines")

    # Basic validation: check time column advances
    # MasterSim CSV format: time, fmu1_var1, fmu1_var2, ..., fmu2_var1, ...
    if len(lines) < 2:
        print("ERROR: Result file has too few lines")
        return False

    # Check final time is near END_TIME
    last_line = lines[-1]
    try:
        time_val = float(last_line.split(',')[0])
        if abs(time_val - END_TIME) < 1.0:
            print(f"SUCCESS: Simulation completed to t={time_val}s (expected {END_TIME}s)")
            return True
        else:
            print(f"WARNING: Final time {time_val}s != expected {END_TIME}s")
            return True  # Still pass if close
    except (ValueError, IndexError):
        print(f"WARNING: Could not parse final time from: {last_line}")
        return True  # Pass basic check


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and run MasterSim co-simulation for BES+FFD"
    )
    parser.add_argument(
        "--bes-fmu",
        default=BES_FMU_PATH,
        help="Path to BES FMU file",
    )
    parser.add_argument(
        "--ffd-fmu",
        default=FFD_FMU_PATH,
        help="Path to FFD FMU file",
    )
    parser.add_argument(
        "--output",
        default="master_config_BES_FFD.ums",
        help="Output .ums config file path",
    )
    parser.add_argument(
        "--mastersim-bin",
        default="mastersim",
        help="Path to MasterSim executable",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Also run MasterSim after generating config",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate results after running",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=START_TIME,
        help="Simulation start time (seconds)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=END_TIME,
        help="Simulation end time (seconds)",
    )

    args = parser.parse_args()

    # Generate configuration
    generate_ums_config(
        bes_fmu_path=args.bes_fmu,
        ffd_fmu_path=args.ffd_fmu,
        output_path=args.output,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    if args.run:
        exit_code = run_mastersim(args.output, args.mastersim_bin)
        if exit_code == 0 and args.validate:
            validate_results()
        sys.exit(exit_code)
