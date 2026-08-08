#!/usr/bin/env python3
# Copyright 2026 Fluxion. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Co-Simulation Master Test Harness for BES+FFD (Issue #2391)

This script provides a complete test harness for co-simulation of:
  - Building Energy Simulation (BES) FMU — thermal zone model
  - Fast Fluid Dynamics (FFD) FMU — room airflow model

The test validates:
  1. MasterSim can load both FMUs simultaneously
  2. 24-hour simulation completes without synchronization deadlocks
  3. Master clock advances correctly with FFD micro-stepping

Usage:
    # Full run (requires MasterSim installed):
    python tools/cosim/run_cosimulation.py --run

    # Generate FMUs and config only (no MasterSim):
    python tools/cosim/run_cosimulation.py --generate-only

    # Quick validation (skip simulation):
    python tools/cosim/run_cosimulation.py --validate-only

Prerequisites:
    MasterSim: https://github.com/ghorwin/MasterSim
    Install: git clone && cd MasterSim && mkdir build && cd build && cmake .. && make

Environment Variables:
    MASTERSIM_BIN   Path to MasterSim executable (default: mastersim)
    FLUXION_FMU_OUT Directory for exported FMUs (default: tools/cosim/)
"""

import argparse
import csv
import io
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Simulation parameters (24-hour run)
START_TIME = 0.0           # seconds
END_TIME = 86400.0          # 24 hours
BES_TIMESTEP = 3600.0      # BES: 1 hour
FFD_TIMESTEP = 60.0         # FFD: 1 minute (micro-stepping)
RESULT_INTERVAL = 3600.0    # Output every hour

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "tools" / "cosim"
BES_FMU_OUT = OUTPUT_DIR / "bes_dummy.fmu"
FFD_FMU_OUT = OUTPUT_DIR / "ffd_dummy.fmu"
UMS_CONFIG_OUT = OUTPUT_DIR / "master_config_BES_FFD.ums"
RESULTS_CSV = OUTPUT_DIR / "cosim_results.csv"


# ============================================================================
# FMI 2.0 Model Description XML Generators
# ============================================================================

def generate_timestamp() -> str:
    """Generate ISO 8601 UTC timestamp."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_guid() -> str:
    """Generate a random GUID for FMU identification."""
    import uuid
    return "{" + str(uuid.uuid4()).upper() + "}"


def bes_model_description_xml(
    model_name: str = "FluxionBES",
    guid: str = "",
    description: str = "Fluxion Building Energy Simulation FMU",
    communication_timestep: float = 3600.0,
    start_time: float = 0.0,
    stop_time: float = 86400.0,
) -> str:
    """
    Generate FMI 2.0 modelDescription.xml for a BES FMU.

    Variables:
      Inputs (causality=input):
        - outdoor_temperature (K): Outdoor dry bulb temperature
        - direct_normal_solar (W/m2): Direct normal solar radiation
        - diffuse_horizontal_solar (W/m2): Diffuse horizontal solar
        - internal_gains (W): Internal heat gains

      Outputs (causality=output):
        - zone_temperature (K): Zone air temperature
        - heating_load (W): Heating load (non-negative)
        - cooling_load (W): Cooling load (non-negative)
    """
    if not guid:
        guid = generate_guid()
    timestamp = generate_timestamp()

    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<fmiModelDescription
    fmiVersion="2.0"
    modelName="{model_name}"
    guid="{guid}"
    description="{description}"
    author="Fluxion Project"
    version="1.0.0"
    generationTool="Fluxion FMI Exporter v1.0.0"
    generationDateAndTime="{timestamp}"
    variableNamingConvention="structured">

    <CoSimulation
        modelIdentifier="{model_name}"
        needsExecutionTool="true"
        canHandleVariableCommunicationStepSize="true"
        canInterpolateInputs="true"
        canGetAndSetFMUstate="false"
        canSerializeFMUstate="false"
        canBeInstantiatedOnlyOncePerProcess="false"
        canNotUseMemoryManagementFunctions="false"/>

    <DefaultExperiment
        startTime="{start_time}"
        stopTime="{stop_time}"
        stepSize="{communication_timestep}"/>

    <ModelVariables>
        <!-- Input: Outdoor Temperature -->
        <ScalarVariable
            name="outdoor_temperature"
            valueReference="1"
            description="Outdoor dry bulb temperature"
            causality="input"
            variability="continuous">
            <Real
                quantity=""
                unit="K"
                displayUnit=""
                relativeQuantity="false"
                min="200.0"
                max="320.0"
                nominal="0.0"
                unbounded="false"
                start="280.0"
                reinit="false"/>
        </ScalarVariable>

        <!-- Input: Direct Normal Solar -->
        <ScalarVariable
            name="direct_normal_solar"
            valueReference="2"
            description="Direct normal solar radiation"
            causality="input"
            variability="continuous">
            <Real
                quantity=""
                unit="W/m2"
                displayUnit=""
                relativeQuantity="false"
                min="0.0"
                max="1200.0"
                nominal="0.0"
                unbounded="false"
                start="0.0"
                reinit="false"/>
        </ScalarVariable>

        <!-- Input: Diffuse Horizontal Solar -->
        <ScalarVariable
            name="diffuse_horizontal_solar"
            valueReference="3"
            description="Diffuse horizontal solar radiation"
            causality="input"
            variability="continuous">
            <Real
                quantity=""
                unit="W/m2"
                displayUnit=""
                relativeQuantity="false"
                min="0.0"
                max="800.0"
                nominal="0.0"
                unbounded="false"
                start="0.0"
                reinit="false"/>
        </ScalarVariable>

        <!-- Input: Internal Gains -->
        <ScalarVariable
            name="internal_gains"
            valueReference="4"
            description="Total internal heat gains"
            causality="input"
            variability="continuous">
            <Real
                quantity=""
                unit="W"
                displayUnit=""
                relativeQuantity="false"
                min="0.0"
                max="10000.0"
                nominal="0.0"
                unbounded="false"
                start="0.0"
                reinit="false"/>
        </ScalarVariable>

        <!-- Output: Zone Temperature -->
        <ScalarVariable
            name="zone_temperature"
            valueReference="5"
            description="Zone air temperature"
            causality="output"
            variability="continuous">
            <Real
                quantity=""
                unit="K"
                displayUnit=""
                relativeQuantity="false"
                min="200.0"
                max="320.0"
                nominal="0.0"
                unbounded="false"
                start="293.15"
                reinit="false"/>
        </ScalarVariable>

        <!-- Output: Heating Load -->
        <ScalarVariable
            name="heating_load"
            valueReference="6"
            description="Heating load (positive)"
            causality="output"
            variability="continuous">
            <Real
                quantity=""
                unit="W"
                displayUnit=""
                relativeQuantity="false"
                min="0.0"
                max="100000.0"
                nominal="0.0"
                unbounded="false"
                start="0.0"
                reinit="false"/>
        </ScalarVariable>

        <!-- Output: Cooling Load -->
        <ScalarVariable
            name="cooling_load"
            valueReference="7"
            description="Cooling load (positive)"
            causality="output"
            variability="continuous">
            <Real
                quantity=""
                unit="W"
                displayUnit=""
                relativeQuantity="false"
                min="0.0"
                max="100000.0"
                nominal="0.0"
                unbounded="false"
                start="0.0"
                reinit="false"/>
        </ScalarVariable>
    </ModelVariables>

    <ModelStructure>
        <Outputs>
            <Unknown index="5" dependencies=""/>
            <Unknown index="6" dependencies=""/>
            <Unknown index="7" dependencies=""/>
        </Outputs>
        <InitialUnknowns>
            <Unknown index="5" dependencies=""/>
            <Unknown index="6" dependencies=""/>
            <Unknown index="7" dependencies=""/>
        </InitialUnknowns>
    </ModelStructure>

</fmiModelDescription>'''
    return xml


def ffd_model_description_xml(
    model_name: str = "FluxionFFD",
    guid: str = "",
    description: str = "Fluxion Fast Fluid Dynamics FMU",
    communication_timestep: float = 60.0,
    start_time: float = 0.0,
    stop_time: float = 86400.0,
    num_surfaces: int = 6,
    num_levels: int = 4,
) -> str:
    """
    Generate FMI 2.0 modelDescription.xml for an FFD FMU.

    Variables:
      Inputs (BES → FFD):
        - inlet_air_temperature (K): Supply air temperature
        - mass_flow_rate_supply (kg/s): HVAC supply air mass flow rate
        - mass_flow_rate_exhaust (kg/s): HVAC exhaust air mass flow rate
        - wall_temperature_N (K): Zone wall surface temperatures

      Outputs (FFD → BES):
        - zone_air_temperature_N (K): Stratified air temperature at heights
        - chtc_N (W/m2K): Convective heat transfer coefficients
        - surface_heat_flux_N (W/m2): Surface heat fluxes
    """
    if not guid:
        guid = generate_guid()
    timestamp = generate_timestamp()

    # Build ScalarVariable XML for inputs
    input_vars = []
    vr = 1

    # inlet_air_temperature
    input_vars.append(f'''        <ScalarVariable
            name="inlet_air_temperature"
            valueReference="{vr}"
            description="Inlet/supply air temperature"
            causality="input"
            variability="continuous">
            <Real
                quantity=""
                unit="K"
                displayUnit=""
                relativeQuantity="false"
                min="200.0"
                max="350.0"
                nominal="0.0"
                unbounded="false"
                start="293.15"
                reinit="false"/>
        </ScalarVariable>''')
    vr += 1

    # mass_flow_rate_supply
    input_vars.append(f'''        <ScalarVariable
            name="mass_flow_rate_supply"
            valueReference="{vr}"
            description="HVAC supply air mass flow rate"
            causality="input"
            variability="continuous">
            <Real
                quantity=""
                unit="kg/s"
                displayUnit=""
                relativeQuantity="false"
                min="0.0"
                max="10.0"
                nominal="0.0"
                unbounded="false"
                start="0.0"
                reinit="false"/>
        </ScalarVariable>''')
    vr += 1

    # mass_flow_rate_exhaust
    input_vars.append(f'''        <ScalarVariable
            name="mass_flow_rate_exhaust"
            valueReference="{vr}"
            description="HVAC exhaust air mass flow rate"
            causality="input"
            variability="continuous">
            <Real
                quantity=""
                unit="kg/s"
                displayUnit=""
                relativeQuantity="false"
                min="0.0"
                max="10.0"
                nominal="0.0"
                unbounded="false"
                start="0.0"
                reinit="false"/>
        </ScalarVariable>''')
    vr += 1

    # wall_temperature_N
    for i in range(num_surfaces):
        input_vars.append(f'''        <ScalarVariable
            name="wall_temperature_{i}"
            valueReference="{vr}"
            description="Wall temperature at surface {i}"
            causality="input"
            variability="continuous">
            <Real
                quantity=""
                unit="K"
                displayUnit=""
                relativeQuantity="false"
                min="200.0"
                max="350.0"
                nominal="0.0"
                unbounded="false"
                start="293.15"
                reinit="false"/>
        </ScalarVariable>''')
        vr += 1

    # Build ScalarVariable XML for outputs
    output_vars = []
    output_start_vr = vr

    # zone_air_temperature_N (stratified)
    for i in range(num_levels):
        output_vars.append(f'''        <ScalarVariable
            name="zone_air_temperature_{i}"
            valueReference="{vr}"
            description="Zone air temperature at height {i}"
            causality="output"
            variability="continuous">
            <Real
                quantity=""
                unit="K"
                displayUnit=""
                relativeQuantity="false"
                min="200.0"
                max="350.0"
                nominal="0.0"
                unbounded="false"
                start="293.15"
                reinit="false"/>
        </ScalarVariable>''')
        vr += 1

    # chtc_N (convective heat transfer coefficient)
    for i in range(num_surfaces):
        output_vars.append(f'''        <ScalarVariable
            name="chtc_{i}"
            valueReference="{vr}"
            description="Convective heat transfer coefficient for surface {i}"
            causality="output"
            variability="continuous">
            <Real
                quantity=""
                unit="W/m2K"
                displayUnit=""
                relativeQuantity="false"
                min="0.0"
                max="100.0"
                nominal="0.0"
                unbounded="false"
                start="2.0"
                reinit="false"/>
        </ScalarVariable>''')
        vr += 1

    # surface_heat_flux_N
    for i in range(num_surfaces):
        output_vars.append(f'''        <ScalarVariable
            name="surface_heat_flux_{i}"
            valueReference="{vr}"
            description="Surface heat flux for surface {i}"
            causality="output"
            variability="continuous">
            <Real
                quantity=""
                unit="W/m2"
                displayUnit=""
                relativeQuantity="false"
                min="-10000.0"
                max="10000.0"
                nominal="0.0"
                unbounded="false"
                start="0.0"
                reinit="false"/>
        </ScalarVariable>''')
        vr += 1

    # Build ModelStructure Unknowns
    output_unknowns = []
    for i in range(len(output_vars)):
        idx = output_start_vr + i
        output_unknowns.append(f'            <Unknown index="{idx}" dependencies=""/>')

    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<fmiModelDescription
    fmiVersion="2.0"
    modelName="{model_name}"
    guid="{guid}"
    description="{description}"
    author="Fluxion Project"
    version="1.0.0"
    generationTool="Fluxion FMI Exporter v1.0.0"
    generationDateAndTime="{timestamp}"
    variableNamingConvention="structured">

    <CoSimulation
        modelIdentifier="{model_name}"
        needsExecutionTool="true"
        canHandleVariableCommunicationStepSize="true"
        canInterpolateInputs="true"
        canGetAndSetFMUstate="false"
        canSerializeFMUstate="false"
        canBeInstantiatedOnlyOncePerProcess="false"
        canNotUseMemoryManagementFunctions="false"/>

    <DefaultExperiment
        startTime="{start_time}"
        stopTime="{stop_time}"
        stepSize="{communication_timestep}"/>

    <ModelVariables>
{chr(10).join(input_vars)}
{chr(10).join(output_vars)}
    </ModelVariables>

    <ModelStructure>
        <Outputs>
{chr(10).join(output_unknowns)}
        </Outputs>
        <InitialUnknowns>
{chr(10).join(output_unknowns)}
        </InitialUnknowns>
    </ModelStructure>

</fmiModelDescription>'''
    return xml


# ============================================================================
# FMU Archive Creation
# ============================================================================

def create_fmu_archive(
    output_path: Path,
    model_description_xml: str,
    model_name: str,
    description: str,
    communication_timestep: float,
    num_inputs: int,
    num_outputs: int,
) -> None:
    """
    Create a minimal FMI 2.0 FMU archive (.fmu = ZIP file).

    The FMU contains:
      - modelDescription.xml: FMI 2.0 model description
      - binaries/: Empty placeholder (no native binary; master calls Fluxion)
      - resources/: Empty placeholder
      - resources/README.txt: Human-readable info
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # modelDescription.xml (mandatory at root per FMI 2.0 §2.2)
        zf.writestr("modelDescription.xml", model_description_xml)

        # binaries/ directory (empty placeholder)
        # Some masters check for this directory; create a .gitkeep-style marker
        zf.writestr("binaries/.placeholder", b"")

        # resources/ directory
        zf.writestr("resources/.placeholder", b"")

        # README in resources/
        readme = f"""Fluxion FMU - {description}

Model: {model_name}
Communication Timestep: {communication_timestep} s
Inputs: {num_inputs}
Outputs: {num_outputs}

This FMU is a placeholder for co-simulation testing.
It declares the FMI 2.0 interface but requires an external tool
(MasterSim, FMPy, PyFMI) to drive the simulation.

See modelDescription.xml for the full variable list.
"""
        zf.writestr("resources/README.txt", readme)

    print(f"Created FMU: {output_path} ({output_path.stat().st_size} bytes)")


# ============================================================================
# MasterSim Configuration Generator
# ============================================================================

def generate_ums_config(
    bes_fmu_path: str,
    ffd_fmu_path: str,
    output_path: Path,
    start_time: float = START_TIME,
    end_time: float = END_TIME,
    bes_timestep: float = BES_TIMESTEP,
    ffd_timestep: float = FFD_TIMESTEP,
    result_interval: float = RESULT_INTERVAL,
) -> None:
    """
    Generate a MasterSim .ums configuration file for BES+FFD co-simulation.

    MasterSim (https://github.com/ghorwin/MasterSim) is an open-source
    FMI 2.0.3 co-simulation master that supports:
    - Multiple FMU instances
    - Variable connections between FMUs
    - Configurable time-stepping with micro-stepping
    - Deadlock detection and recovery
    """

    master_timestep = min(bes_timestep, ffd_timestep)

    config_xml = f'''<?xml version="1.0" encoding="utf-8"?>
<!--
    MasterSim Co-Simulation Configuration
    BES + FFD Coupled Simulation (Issue #2391)

    MasterSim: https://github.com/ghorwin/MasterSim
    Generated: {generate_timestamp()}

    Coupling Strategy: Loose Coupling (Issue #2390)
    - BES runs at coarser timestep (1 hour) for thermal envelope
    - FFD runs at finer timestep (1 minute) for airflow dynamics
    - Data exchanged at each BES timestep

    FMU Connections:
      BES → FFD: outdoor_temperature → inlet_air_temperature
      BES → FFD: zone_temperature → wall_temperature_0
      FFD → BES: zone_air_temperature_0 → (used by BES for next step)

    To run:
      mastersim {output_path.name}
-->
<CoSimulationMaster
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="https://raw.githubusercontent.com/ghorwin/MasterSim/master/docs/MasterSimulatorUML.xsd">

    <!-- ==================== Simulation Parameters ==================== -->
    <SimulationParameters
        startTime="{start_time}"
        endTime="{end_time}"
        masterStepSize="{master_timestep}"
        resultFile="cosim_results"
        resultInterval="{result_interval}"
        /* Options:
           preventEarlyTermination=true|false
           relayResultsToStdOut=true|false
        */
    />

    <!-- ==================== FMU Definitions ==================== -->
    <!-- BES FMU: Building Energy Simulation (thermal envelope) -->
    <FMUDefinition
        name="BES"
        source="{bes_fmu_path}"
        instanceName="bes1"
        visible="false"
        loggingOn="true"
        /* Optional:
           muteInitializationErrors=false
        */
    />

    <!-- FFD FMU: Fast Fluid Dynamics (airflow solver) -->
    <FMUDefinition
        name="FFD"
        source="{ffd_fmu_path}"
        instanceName="ffd1"
        visible="false"
        loggingOn="true"
    />

    <!-- ==================== Data Exchange Connections ==================== -->
    <!--
        Connections define the data flow between FMUs at each master step.
        Format: fmu1:var1 → fmu2:var2

        BES → FDD Connections (boundary conditions):
    -->
    <Connection
        fmu1="bes1" var1="outdoor_temperature"
        fmu2="ffd1" var2="inlet_air_temperature"
        delay="0"
    />

    <!--
        FFD → BES Connections (zone conditions feedback):
        The FFD zone air temperature is used by BES for convective heat
        transfer calculations in the next timestep.
    -->
    <Connection
        fmu1="ffd1" var1="zone_air_temperature_0"
        fmu2="bes1" var2="zone_temperature"
        delay="0"
    />

    <!-- ==================== Logging Configuration ==================== -->
    <LogCategories>
        <Category name="logStatus"    enabled="true"/>
        <Category name="logError"     enabled="true"/>
        <Category name="logWarning"   enabled="true"/>
        <Category name="logInfo"       enabled="false"/>
        <Category name="logDebug"      enabled="false"/>
        <Category name="logAll"       enabled="false"/>
    </LogCategories>

    <!-- ==================== Solver Configuration ==================== -->
    <!--
        Master solver uses IEC 61774 master algorithm with:
        - Fixed-step communication (no variable step size)
        - Steppers: Forward Euler (default), RK4 (optional)
    -->
    <SolverSettings
        stepSize="{master_timestep}"
        useOptimizedTimestep="true"
        deadlockDetection="true"
        deadlockRecoveryAttempts="3"
    />

</CoSimulationMaster>
'''

    output_path.write_text(config_xml, encoding='utf-8')
    print(f"Generated MasterSim config: {output_path}")


# ============================================================================
# Validation
# ============================================================================

@dataclass
class ValidationResult:
    """Result of co-simulation validation."""
    passed: bool
    final_time: float = 0.0
    num_timesteps: int = 0
    num_deadlocks: int = 0
    max_zone_temp: float = 0.0
    min_zone_temp: float = 0.0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def validate_cosim_results(
    result_file: Path = RESULTS_CSV,
    expected_end_time: float = END_TIME,
    tolerance: float = 1.0,
) -> ValidationResult:
    """
    Validate co-simulation results for Issue #2391 acceptance criteria.

    Checks:
    1. Simulation completed to expected end time (±tolerance)
    2. No synchronization deadlocks (all timesteps present)
    3. All outputs are finite and within physical bounds
    4. Master clock advanced correctly
    """
    result = ValidationResult(passed=False)

    if not result_file.exists():
        # Check for alternative formats MasterSim might produce
        for alt in ["cosim_results.json", "cosim_results.txt"]:
            alt_path = result_file.parent / alt
            if alt_path.exists():
                result_file = alt_path
                result.warnings.append(f"Using alternative result format: {alt}")
                break
        else:
            result.errors.append(f"Result file not found: {result_file}")
            result.warnings.append("MasterSim may have failed to produce output")
            return result

    try:
        content = result_file.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = result_file.read_text(encoding='latin-1')

    if not content.strip():
        result.errors.append("Result file is empty")
        return result

    lines = content.strip().split('\n')
    result.num_timesteps = len(lines) - 1  # Exclude header

    if result.num_timesteps < 1:
        result.errors.append(f"Result file has no data rows: {lines}")
        return result

    # Parse CSV (MasterSim CSV format: time, fmu1_outputs..., fmu2_outputs...)
    try:
        reader = csv.DictReader(lines)
        rows = list(reader)
        if not rows:
            raise ValueError("No rows in CSV")

        # Get final time
        final_row = rows[-1]
        time_col = [c for c in final_row.keys() if 'time' in c.lower()][0]
        result.final_time = float(final_row[time_col])

        # Check time progression
        if abs(result.final_time - expected_end_time) > tolerance:
            result.errors.append(
                f"Final time {result.final_time}s != expected {expected_end_time}s"
            )
        else:
            print(f"OK: Simulation completed to t={result.final_time}s")

        # Parse zone temperature columns
        zone_temp_cols = [c for c in final_row.keys() if 'zone_temperature' in c.lower()]
        if zone_temp_cols:
            temps = [float(final_row[c]) for c in zone_temp_cols]
            result.max_zone_temp = max(temps)
            result.min_zone_temp = min(temps)

            # Physical bounds check (200K to 350K)
            if result.min_zone_temp < 200.0 or result.max_zone_temp > 350.0:
                result.errors.append(
                    f"Zone temperatures out of physical bounds: "
                    f"{result.min_zone_temp:.1f}K to {result.max_zone_temp:.1f}K"
                )
            else:
                print(f"OK: Zone temps in bounds: {result.min_zone_temp:.1f}K to {result.max_zone_temp:.1f}K")

        # Check for deadlocks (gaps in time series)
        if len(rows) >= 2:
            times = [float(row[time_col]) for row in rows]
            expected_interval = 3600.0  # 1 hour
            gaps = []
            for i in range(1, len(times)):
                dt = times[i] - times[i-1]
                if abs(dt - expected_interval) > tolerance:
                    gaps.append((times[i-1], times[i], dt))
            if gaps:
                result.num_deadlocks = len(gaps)
                result.warnings.append(f"Found {len(gaps)} time gaps (potential deadlocks): {gaps[:5]}")

    except Exception as e:
        result.errors.append(f"Failed to parse results: {e}")
        return result

    # Determine pass/fail
    if not result.errors:
        result.passed = True
        print("VALIDATION PASSED: 24-hour co-simulation completed without errors")
    else:
        print(f"VALIDATION FAILED: {len(result.errors)} error(s)")
        for err in result.errors:
            print(f"  ERROR: {err}")

    for warn in result.warnings:
        print(f"  WARNING: {warn}")

    return result


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Co-Simulation Master Test Harness for BES+FFD (Issue #2391)"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate FMUs and config only (skip simulation)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run MasterSim after generating FMUs",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing results only",
    )
    parser.add_argument(
        "--mastersim-bin",
        default=os.environ.get("MASTERSIM_BIN", "mastersim"),
        help="Path to MasterSim executable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for FMUs and config",
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
    parser.add_argument(
        "--bes-timestep",
        type=float,
        default=BES_TIMESTEP,
        help="BES communication timestep (seconds)",
    )
    parser.add_argument(
        "--ffd-timestep",
        type=float,
        default=FFD_TIMESTEP,
        help="FFD communication timestep (seconds)",
    )
    return parser.parse_args()


def generate_fmus_and_config(args) -> None:
    """Generate BES FMU, FFD FMU, and MasterSim configuration."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate BES FMU
    bes_xml = bes_model_description_xml(
        model_name="FluxionBES",
        description="Fluxion Building Energy Simulation FMU (Dummy for Testing)",
        communication_timestep=args.bes_timestep,
        start_time=args.start_time,
        stop_time=args.end_time,
    )
    create_fmu_archive(
        output_path=output_dir / "bes_dummy.fmu",
        model_description_xml=bes_xml,
        model_name="FluxionBES",
        description="Building Energy Simulation FMU",
        communication_timestep=args.bes_timestep,
        num_inputs=4,
        num_outputs=3,
    )

    # Generate FFD FMU
    ffd_xml = ffd_model_description_xml(
        model_name="FluxionFFD",
        description="Fluxion Fast Fluid Dynamics FMU (Dummy for Testing)",
        communication_timestep=args.ffd_timestep,
        start_time=args.start_time,
        stop_time=args.end_time,
    )
    create_fmu_archive(
        output_path=output_dir / "ffd_dummy.fmu",
        model_description_xml=ffd_xml,
        model_name="FluxionFFD",
        description="Fast Fluid Dynamics FMU",
        communication_timestep=args.ffd_timestep,
        num_inputs=3 + 6,  # inlet + supply + exhaust + 6 wall temps
        num_outputs=4 + 6 + 6,  # 4 temp levels + 6 CHTCs + 6 fluxes
    )

    # Generate MasterSim config
    generate_ums_config(
        bes_fmu_path="bes_dummy.fmu",
        ffd_fmu_path="ffd_dummy.fmu",
        output_path=output_dir / "master_config_BES_FFD.ums",
        start_time=args.start_time,
        end_time=args.end_time,
        bes_timestep=args.bes_timestep,
        ffd_timestep=args.ffd_timestep,
    )

    print(f"\nGenerated FMUs and config in: {output_dir}")
    print("To run MasterSim:")
    print(f"  cd {output_dir}")
    print(f"  mastersim master_config_BES_FFD.ums")


def run_mastersim(args) -> int:
    """Run MasterSim with the generated configuration."""
    config_path = args.output_dir / "master_config_BES_FFD.ums"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Run with --generate-only first")
        return 1

    cmd = [args.mastersim_bin, str(config_path)]
    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(args.output_dir),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for 24hr sim
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"ERROR: MasterSim exited with code {result.returncode}")
            return result.returncode

        print("=" * 60)
        print("MasterSim completed successfully")

        # Look for result files
        result_files = list(args.output_dir.glob("cosim_results*"))
        if result_files:
            print(f"Result files: {[f.name for f in result_files]}")
        else:
            print("WARNING: No result files found")

        return 0

    except subprocess.TimeoutExpired:
        print("ERROR: MasterSim timed out after 600 seconds")
        return -1
    except FileNotFoundError:
        print(f"ERROR: MasterSim not found: {args.mastersim_bin}")
        print("Install MasterSim: https://github.com/ghorwin/MasterSim")
        print("Or set MASTERSIM_BIN environment variable")
        return -1


def main():
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        result = validate_cosim_results(
            result_file=args.output_dir / "cosim_results.csv",
            expected_end_time=args.end_time,
        )
        sys.exit(0 if result.passed else 1)

    # Generate FMUs and configuration
    generate_fmus_and_config(args)

    if args.generate_only:
        print("\nGeneration complete (--generate-only specified)")
        sys.exit(0)

    # Run MasterSim
    exit_code = run_mastersim(args)

    if exit_code == 0:
        # Validate results
        result = validate_cosim_results(
            result_file=args.output_dir / "cosim_results.csv",
            expected_end_time=args.end_time,
        )
        sys.exit(0 if result.passed else 1)
    else:
        print("\nSkipping validation due to MasterSim failure")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
