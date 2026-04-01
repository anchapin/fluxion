"""
Extract Component Data from EnergyPlus Simulation Results.

This module extracts component-level heat balance data from EnergyPlus
simulations via the OpenStudio MCP server or by parsing SQL files directly.

Supported components (EnergyPlus output variables):
- Surface Inside Face Conduction Heat Transfer Rate [W]
- Surface Inside Face Convection Heat Transfer Rate [W]
- Surface Inside Face Transmitted Solar Radiation Rate [W]
- Zone Air Heat Balance Rate [W]
- Zone Ideal Loads Supply Air Total Heating/Cooling Rate [W]
- Zone Infiltration Total Heat Transfer Rate [W]
- Surface Inside Face Temperature [°C]
- Zone Air Temperature [°C]

Usage:
    # Using OpenStudio MCP (recommended)
    python -m tools.extract_energyplus_components \
        --osm benchmarks/outputs/bestest_gsr/case_900/in.osm \
        --output energyplus_components.csv

    # Using SQL file directly
    python -m tools.extract_energyplus_components \
        --sql eplusout.sql \
        --output energyplus_components.csv
"""

import argparse
import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class EnergyPlusComponent:
    """EnergyPlus output variable data."""

    timestep: int
    hour: float
    component_type: str
    key_value: str  # Surface name, zone name, or "*"
    value: float
    units: str
    metadata: Dict[str, float] = field(default_factory=dict)


class EnergyPlusSQLParser:
    """Parser for EnergyPlus SQL output files."""

    # Mapping of component types to EnergyPlus output variables
    OUTPUT_VARIABLES = {
        "ctf_flux": "Surface Inside Face Conduction Heat Transfer Rate",
        "convective_flux": "Surface Inside Face Convection Heat Transfer Rate",
        "solar_flux": "Surface Inside Face Transmitted Solar Radiation Rate",
        "zone_balance": "Zone Air Heat Balance Rate",
        "hvac_heating": "Zone Ideal Loads Supply Air Total Heating Rate",
        "hvac_cooling": "Zone Ideal Loads Supply Air Total Cooling Rate",
        "infiltration": "Zone Infiltration Total Heat Transfer Rate",
        "surface_temp": "Surface Inside Face Temperature",
        "zone_temp": "Zone Air Temperature",
    }

    def __init__(self, sql_path: Path):
        """Initialize parser with SQL file path.

        Args:
            sql_path: Path to EnergyPlus SQL file
        """
        self.sql_path = sql_path
        self.data: List[EnergyPlusComponent] = []
        self._cache: Dict[str, List[Tuple[int, float]]] = {}

    def extract_component(
        self, component_type: str, key_filter: Optional[str] = None
    ) -> List[EnergyPlusComponent]:
        """Extract data for a specific component type.

        Args:
            component_type: Component type (e.g., "ctf_flux", "convective_flux")
            key_filter: Optional filter for key_value (surface/zone name)

        Returns:
            List of EnergyPlusComponent objects
        """
        if component_type not in self.OUTPUT_VARIABLES:
            raise ValueError(f"Unknown component type: {component_type}")

        variable_name = self.OUTPUT_VARIABLES[component_type]

        # Check cache
        cache_key = f"{variable_name}:{key_filter or '*'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Query SQL file
        results = self._query_sql(variable_name, key_filter)

        # Convert to EnergyPlusComponent objects
        components = []
        for timestep, value in results:
            comp = EnergyPlusComponent(
                timestep=timestep,
                hour=timestep / 6.0,  # Assuming 10-minute timesteps
                component_type=component_type,
                key_value=key_filter or "*",
                value=value,
                units="W" if "temp" not in component_type else "°C",
            )
            components.append(comp)

        # Cache results
        self._cache[cache_key] = components
        self.data.extend(components)

        return components

    def _query_sql(
        self, variable_name: str, key_filter: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """Query EnergyPlus SQL file for a variable.

        Args:
            variable_name: EnergyPlus output variable name
            key_filter: Optional key filter

        Returns:
            List of (timestep, value) tuples
        """
        # Use sqlite3 to query the SQL file
        try:
            import sqlite3
        except ImportError:
            raise ImportError("sqlite3 is required for SQL parsing")

        conn = sqlite3.connect(str(self.sql_path))
        cursor = conn.cursor()

        # Build query
        if key_filter and key_filter != "*":
            query = """
                SELECT TimeIndex, Value
                FROM ReportVariableData
                WHERE VariableName = ? AND KeyName = ?
                ORDER BY TimeIndex
            """
            cursor.execute(query, (variable_name, key_filter))
        else:
            query = """
                SELECT TimeIndex, Value
                FROM ReportVariableData
                WHERE VariableName = ?
                ORDER BY TimeIndex
            """
            cursor.execute(query, (variable_name,))

        results = [(int(row[0]), float(row[1])) for row in cursor.fetchall()]
        conn.close()

        return results

    def extract_all_surfaces(
        self, component_type: str, surface_names: List[str]
    ) -> Dict[str, List[EnergyPlusComponent]]:
        """Extract data for all surfaces of a type.

        Args:
            component_type: Component type
            surface_names: List of surface names

        Returns:
            Dictionary mapping surface name to component data
        """
        results = {}
        for surf_name in surface_names:
            results[surf_name] = self.extract_component(component_type, surf_name)
        return results

    def to_csv(self, output_path: Path):
        """Export all extracted data to CSV.

        Args:
            output_path: Output CSV file path
        """
        if not self.data:
            print("No data to export")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestep", "hour", "component_type", "key_value", "value", "units"]
            )

            for comp in self.data:
                writer.writerow(
                    [
                        comp.timestep,
                        comp.hour,
                        comp.component_type,
                        comp.key_value,
                        comp.value,
                        comp.units,
                    ]
                )

        print(f"Exported {len(self.data)} records to {output_path}")


class OpenStudioMCPExtractor:
    """Extract EnergyPlus data using OpenStudio MCP server."""

    def __init__(self, osm_path: Optional[Path] = None):
        """Initialize extractor.

        Args:
            osm_path: Optional path to OSM file to load
        """
        self.osm_path = osm_path
        self.model_loaded = False

    def run_simulation(
        self, case_name: str = "case_900", weather_path: Optional[Path] = None
    ) -> str:
        """Run EnergyPlus simulation via OpenStudio.

        Args:
            case_name: Case name (e.g., "case_900")
            weather_path: Optional EPW weather file path

        Returns:
            Run ID for retrieving results
        """
        # This would use the OpenStudio MCP tools
        # For now, we'll use subprocess to call the run_osw tool

        print("Running EnergyPlus simulation via OpenStudio...")

        # Build OSW file path
        osm_file = self.osm_path or Path(
            f"benchmarks/outputs/bestest_gsr/{case_name}/in.osm"
        )

        if not osm_file.exists():
            raise FileNotFoundError(f"OSM file not found: {osm_file}")

        # Run simulation
        cmd = [
            "python",
            "-m",
            "tools.run_energyplus_simulations",
            "--osm",
            str(osm_file),
            "--case",
            case_name,
        ]

        if weather_path:
            cmd.extend(["--weather", str(weather_path)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed: {result.stderr}")

        # Parse run ID from output
        for line in result.stdout.splitlines():
            if "Run ID:" in line:
                run_id = line.split(":")[1].strip()
                return run_id

        raise RuntimeError("Could not find run ID in output")

    def extract_output_variables(
        self, run_id: str, variable_names: List[str]
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Extract output variables from a completed simulation.

        Args:
            run_id: Run ID from simulation
            variable_names: List of EnergyPlus output variable names

        Returns:
            Dictionary mapping variable name to (timestep, value) list
        """
        # This would use the OpenStudio MCP query_timeseries tool
        # For now, return empty dict as placeholder
        print(f"Extracting variables: {variable_names}")

        results = {}
        for var_name in variable_names:
            # Would call: mcp__openstudio-mcp__query_timeseries
            results[var_name] = []

        return results

    def list_available_variables(self, run_id: str) -> List[str]:
        """List available output variables from a simulation.

        Args:
            run_id: Run ID

        Returns:
            List of available variable names
        """
        # Would use OpenStudio MCP list_output_variables tool
        return list(EnergyPlusSQLParser.OUTPUT_VARIABLES.values())


def extract_from_existing_sql(
    sql_path: Path, output_path: Path, components: Optional[List[str]] = None
) -> int:
    """Extract component data from existing EnergyPlus SQL file.

    Args:
        sql_path: Path to eplusout.sql file
        output_path: Output CSV path
        components: List of component types to extract (None = all)

    Returns:
        Exit code (0 = success)
    """
    if not sql_path.exists():
        print(f"Error: SQL file not found: {sql_path}")
        return 1

    print(f"Parsing EnergyPlus SQL: {sql_path}")
    parser = EnergyPlusSQLParser(sql_path)

    # Extract requested components
    if components is None:
        components = list(EnergyPlusSQLParser.OUTPUT_VARIABLES.keys())

    for comp_type in components:
        try:
            data = parser.extract_component(comp_type)
            print(f"  Extracted {len(data)} records for {comp_type}")
        except Exception as e:
            print(f"  Warning: Could not extract {comp_type}: {e}")

    # Export to CSV
    parser.to_csv(output_path)

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract component data from EnergyPlus simulations"
    )

    group = group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--osm", type=Path, help="Path to OSM file (runs simulation via OpenStudio MCP)"
    )
    group.add_argument("--sql", type=Path, help="Path to existing eplusout.sql file")

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="energyplus_components.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=list(EnergyPlusSQLParser.OUTPUT_VARIABLES.keys()),
        help="Specific components to extract (default: all)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="case_900",
        help="Case name for simulation (default: case_900)",
    )
    parser.add_argument("--weather", type=Path, help="Weather file path (EPW)")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    if args.sql:
        # Extract from existing SQL file
        return extract_from_existing_sql(args.sql, args.output, args.components)

    elif args.osm:
        # Run simulation via OpenStudio MCP and extract
        print("Using OpenStudio MCP to run simulation and extract data")

        extractor = OpenStudioMCPExtractor(args.osm)

        try:
            # Run simulation
            run_id = extractor.run_simulation(args.case, args.weather)
            print(f"Simulation completed, run ID: {run_id}")

            # Extract variables
            # (This would use OpenStudio MCP tools)
            print("Extraction via OpenStudio MCP not yet implemented")
            print("Please use --sql option with existing SQL file")
            return 1

        except Exception as e:
            print(f"Error: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
