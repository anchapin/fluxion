#!/usr/bin/env python3
"""
EnergyPlus Test Oracle for Physics Validation

Generates EnergyPlus reference data for physics test validation.

Usage:
    python tools/ep_oracle.py generate --case 600
    python tools/ep_oracle.py generate --all-cases
    python tools/ep_oracle.py compare --fluxion output.json --ep ep_output.json
    python tools/ep_oracle.py validate --test-case 600
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EnergyPlusOracle:
    """EnergyPlus test oracle for generating reference data."""

    def __init__(self, ep_path: Optional[Path] = None):
        """Initialize with EnergyPlus installation path."""
        self.ep_path = ep_path or self._find_energyplus()
        self.data_dir = Path(__file__).parent.parent / "refdata"

    def _find_energyplus(self) -> Path:
        """Find EnergyPlus installation."""
        # Check common paths
        paths = [
            Path("/usr/local/EnergyPlus"),
            Path("/opt/EnergyPlus"),
            Path(os.path.expanduser("~/EnergyPlus")),
            Path(os.getenv("ENERGYPLUS_INSTALL_DIR", "")),
        ]

        for path in paths:
            if path.exists():
                return path

        raise RuntimeError(
            "EnergyPlus not found. Set ENERGYPLUS_INSTALL_DIR or install EnergyPlus."
        )

    def generate_idf(
        self,
        case_id: str,
        floor_area: float,
        u_walls: float,
        u_roof: float,
        u_floor: float,
        u_windows: float,
        window_area: float,
        setpoint_heating: float,
        setpoint_cooling: float,
        location: str = "Denver",
        epw_path: Optional[str] = None,
    ) -> str:
        """
        Generate an OpenStudio IDF for a simple box model.

        Args:
            case_id: Test case identifier
            floor_area: Floor area in m²
            u_walls: Wall U-value (W/m²K)
            u_roof: Roof U-value (W/m²K)
            u_floor: Floor U-value (W/m²K)
            u_windows: Window U-value (W/m²K)
            window_area: Window area in m²
            setpoint_heating: Heating setpoint in °C
            setpoint_cooling: Cooling setpoint in °C
            location: Location name (for weather file lookup)
            epw_path: Optional explicit path to EPW file

        Returns:
            Path to generated IDF file
        """
        # Simple IDF template for single zone box model
        idf_content = f"""Version, 9.4;

! Building
Building,
  {case_id},                  ! Name
  0.0,                       ! North Axis
  City,                       ! Terrain
  0.04,                       ! Loads Convergence Tolerance Value
  0.4,                        ! Temperature Convergence Tolerance Value
  FullExterior,                ! Solar Distribution
  25,                         ! Maximum Number of Warmup Days
  6;                          ! Minimum Number of Warmup Days

! Location
Site:Location,
  {location},                  ! Name
  39.7392,                    ! Latitude
  -104.9903,                  ! Longitude
  -7.0,                       ! Time Zone
  1609.0;                     ! Elevation

! Zone
Zone,
  Zone 1,                     ! Name
  0.0,                        ! Direction of Relative North
  0.0,                        ! X Origin
  0.0,                        ! Y Origin
  0.0,                        ! Z Origin
  1,                          ! Type
  1,                          ! Multiplier

! Schedule: Always On
Schedule:Compact,
  Always On,
  Fraction,
  Through: 12/31,
  For: Weekdays, For: AllDays,
  Until: 24:00, 1.0;

! Schedule: Heating Setpoint
Schedule:Compact,
  Heating Setpoint,
  Temperature,
  Through: 12/31,
  For: Weekdays, For: AllDays,
  Until: 24:00, {setpoint_heating};

! Schedule: Cooling Setpoint
Schedule:Compact,
  Cooling Setpoint,
  Temperature,
  Through: 12/31,
  For: Weekdays, For: AllDays,
  Until: 24:00, {setpoint_cooling};

! Thermostat
ZoneControl:Thermostat,
  Zone 1,
  Heating Setpoint,
  Cooling Setpoint;

! Construction: Walls
Construction,
  Wall,
  Concrete;   ! Outside Layer

! Material: Concrete (customized for U-value)
Material,
  Concrete,
  MediumRough,
  0.15,                       ! Thickness (m)
  {1.0 / (u_walls * 0.15):.4f},  ! Conductivity (W/mK)
  {1600.0},                   ! Density (kg/m3)
  {840.0},                    ! Specific Heat (J/kgK)
  {0.9};                      ! Thermal Absorptance

! Construction: Roof
Construction,
  Roof,
  Concrete;   ! Outside Layer

! Construction: Floor
Construction,
  Floor,
  Concrete;   ! Outside Layer

! Construction: Window
Construction,
  Window,
  SimpleGlazing;

! Simple Glazing (for U-value)
WindowMaterial:SimpleGlazingSystem,
  SimpleGlazing,
  {u_windows:.3f},             ! U-factor
  0.7,                        ! Solar Heat Gain Coefficient
  0.9;                       ! Visible Transmittance

! BuildingSurfaceDetailed: South Wall
BuildingSurfaceDetailed,
  South Wall,                 ! Name
  Wall,                       ! Surface Type
  Wall,                       ! Construction Name
  Zone 1,                     ! Zone Name
  Outdoors,                    ! Outside Boundary Condition
  ,                           ! Outside Boundary Condition Object
  SunExposed,                 ! Sun Exposure
  WindExposed,                 ! Wind Exposure
  0.0,                        ! View Factor to Ground
  4,                          ! Number of Vertices
  0.0, 0.0, 3.0,            ! X,Y,Z Vertex 1
  0.0, 0.0, 0.0,            ! X,Y,Z Vertex 2
  {floor_area**0.5:4f}, 0.0, 0.0,   ! X,Y,Z Vertex 3
  {floor_area**0.5:4f}, 0.0, 3.0;   ! X,Y,Z Vertex 4

! BuildingSurfaceDetailed: North Wall
BuildingSurfaceDetailed,
  North Wall,
  Wall,
  Wall,
  Zone 1,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.0,
  4,
  {floor_area**0.5:4f}, {floor_area**0.5:4f}, 3.0,
  {floor_area**0.5:4f}, {floor_area**0.5:4f}, 0.0,
  0.0, {floor_area**0.5:4f}, 0.0,
  0.0, {floor_area**0.5:4f}, 3.0;

! BuildingSurfaceDetailed: East Wall
BuildingSurfaceDetailed,
  East Wall,
  Wall,
  Wall,
  Zone 1,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.0,
  4,
  0.0, {floor_area**0.5:4f}, 3.0,
  0.0, {floor_area**0.5:4f}, 0.0,
  {floor_area**0.5:4f}, {floor_area**0.5:4f}, 0.0,
  {floor_area**0.5:4f}, {floor_area**0.5:4f}, 3.0;

! BuildingSurfaceDetailed: West Wall (with window)
BuildingSurfaceDetailed,
  West Wall,
  Wall,
  Wall,
  Zone 1,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.0,
  4,
  {floor_area**0.5:4f}, 0.0, 3.0,
  {floor_area**0.5:4f}, 0.0, 0.0,
  0.0, {floor_area**0.5:4f}, 0.0,
  0.0, {floor_area**0.5:4f}, 3.0;

! BuildingSurfaceDetailed: Roof
BuildingSurfaceDetailed,
  Roof,
  Roof,
  Roof,
  Zone 1,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.0,
  4,
  0.0, {floor_area**0.5:4f}, 3.0,
  {floor_area**0.5:4f}, {floor_area**0.5:4f}, 3.0,
  {floor_area**0.5:4f}, 0.0, 3.0,
  0.0, 0.0, 3.0;

! BuildingSurfaceDetailed: Floor
BuildingSurfaceDetailed,
  Floor,
  Floor,
  Floor,
  Zone 1,
  Ground,
  ,
  NoSun,
  NoWind,
  0.0,
  4,
  0.0, 0.0, 0.0,
  {floor_area**0.5:4f}, {floor_area**0.5:4f}, 0.0,
  {floor_area**0.5:4f}, 0.0, 0.0,
  0.0, 0.0, 0.0;

! FenestrationSurfaceDetailed: West Window
FenestrationSurfaceDetailed,
  West Window,
  Window,
  SimpleGlazing,
  Zone 1,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  0.0,
  4,
  0.0, {floor_area**0.5:4f * 0.5}, 1.5,
  0.0, {floor_area**0.5:4f * 0.5}, 2.1,
  0.0, {floor_area**0.5:4f * 0.5 + window_area/floor_area**0.5:4f}, 2.1,
  0.0, {floor_area**0.5:4f * 0.5 + window_area/floor_area**0.5:4f}, 1.5;

! Output:SQL,SQLite
Output:SQLite,
  Options,
  SimpleAndTabular;

! OutputControl:Table:Style
OutputControl:Table:Style,
  Comma,
  HTML;

! Output:Variable,*,Zone Air Temperature
Output:Variable,
  Zone Air Temperature,
  Zone 1,
  Hourly;

! Output:Variable,*,Zone Mean Air Temperature
Output:Variable,
  Zone Mean Air Temperature,
  Zone 1,
  Hourly;

! Output:Variable,*,Site Outdoor Air Drybulb Temperature
Output:Variable,
  Site Outdoor Air Drybulb Temperature,
  ,
  Hourly;

! Output:Variable,*,Zone Thermostat Heating Setpoint Temperature
Output:Variable,
  Zone Thermostat Heating Setpoint Temperature,
  Zone 1,
  Hourly;

! Output:Variable,*,Zone Thermostat Cooling Setpoint Temperature
Output:Variable,
  Zone Thermostat Cooling Setpoint Temperature,
  Zone 1,
  Hourly;

! Output:Variable,*,Zone Ideal Loads Supply Air Total Heating Energy
Output:Variable,
  Zone Ideal Loads Supply Air Total Heating Energy,
  Zone 1,
  Hourly;

! Output:Variable,*,Zone Ideal Loads Supply Air Total Cooling Energy
Output:Variable,
  Zone Ideal Loads Supply Air Total Cooling Energy,
  Zone 1,
  Hourly;

! Output:Variable,*,Surface Outside Face Temperature
Output:Variable,
  Surface Outside Face Temperature,
  West Wall,
  Hourly;

! Output:Variable,*,Surface Inside Face Temperature
Output:Variable,
  Surface Inside Face Temperature,
  West Wall,
  Hourly;
"""
        return idf_content

    def run_energyplus(
        self,
        idf_path: str,
        epw_path: str,
        output_dir: str,
        weather_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run EnergyPlus simulation.

        Args:
            idf_path: Path to IDF file
            epw_path: Path to EPW weather file
            output_dir: Output directory for results
            weather_dir: Directory containing weather files

        Returns:
            Dictionary with simulation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build EnergyPlus command
        ep_cmd = [
            str(self.ep_path / "energyplus"),
            "-w",
            str(epw_path),
            "-d",
            str(output_dir),
            "-r",  # Read variables
            idf_path,
        ]

        print(f"Running EnergyPlus: {' '.join(ep_cmd)}")

        result = subprocess.run(
            ep_cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"EnergyPlus failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"EnergyPlus simulation failed: {result.returncode}")

        # Parse SQL output
        sql_file = output_dir / "eplusout.sql"
        if sql_file.exists():
            return self._parse_sql_results(sql_file)
        else:
            raise RuntimeError("EnergyPlus SQL output file not found")

    def _parse_sql_results(self, sql_file: Path) -> Dict:
        """Parse EnergyPlus SQL output."""
        import sqlite3

        results = {}

        try:
            conn = sqlite3.connect(sql_file)
            cursor = conn.cursor()

            # Get zone temperatures
            cursor.execute(
                """
                SELECT Value
                FROM ReportData
                JOIN ReportDataDictionary
                    ON ReportDataDictionary.ReportDataDictionaryIndex = ReportData.ReportDataDictionaryIndex
                WHERE ReportDataDictionary.Name = 'Zone Mean Air Temperature'
                ORDER BY TimeIndex
            """
            )
            results["zone_temperatures"] = [row[0] for row in cursor.fetchall()]

            # Get outdoor temperatures
            cursor.execute(
                """
                SELECT Value
                FROM ReportData
                JOIN ReportDataDictionary
                    ON ReportDataDictionary.ReportDataDictionaryIndex = ReportData.ReportDataDictionaryIndex
                WHERE ReportDataDictionary.Name = 'Site Outdoor Air Drybulb Temperature'
                ORDER BY TimeIndex
            """
            )
            results["outdoor_temperatures"] = [row[0] for row in cursor.fetchall()]

            # Get heating energy
            cursor.execute(
                """
                SELECT SUM(Value)
                FROM ReportData
                JOIN ReportDataDictionary
                    ON ReportDataDictionary.ReportDataDictionaryIndex = ReportData.ReportDataDictionaryIndex
                WHERE ReportDataDictionary.Name = 'Zone Ideal Loads Supply Air Total Heating Energy'
            """
            )
            row = cursor.fetchone()
            results["heating_energy"] = row[0] if row and row[0] else 0.0

            # Get cooling energy
            cursor.execute(
                """
                SELECT SUM(Value)
                FROM ReportData
                JOIN ReportDataDictionary
                    ON ReportDataDictionary.ReportDataDictionaryIndex = ReportData.ReportDataDictionaryIndex
                WHERE ReportDataDictionary.Name = 'Zone Ideal Loads Supply Air Total Cooling Energy'
            """
            )
            row = cursor.fetchone()
            results["cooling_energy"] = row[0] if row and row[0] else 0.0

            # Get surface temperatures
            cursor.execute(
                """
                SELECT TimeIndex, Value
                FROM ReportData
                JOIN ReportDataDictionary
                    ON ReportDataDictionary.ReportDataDictionaryIndex = ReportData.ReportDataDictionaryIndex
                WHERE ReportDataDictionary.Name = 'Surface Outside Face Temperature'
                  AND ReportDataDictionary.KeyValue = 'West Wall'
                ORDER BY TimeIndex
            """
            )
            results["surface_outside_temps"] = [row[1] for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT TimeIndex, Value
                FROM ReportData
                JOIN ReportDataDictionary
                    ON ReportDataDictionary.ReportDataDictionaryIndex = ReportData.ReportDataDictionaryIndex
                WHERE ReportDataDictionary.Name = 'Surface Inside Face Temperature'
                  AND ReportDataDictionary.KeyValue = 'West Wall'
                ORDER BY TimeIndex
            """
            )
            results["surface_inside_temps"] = [row[1] for row in cursor.fetchall()]

            conn.close()

        except sqlite3.Error as e:
            print(f"SQL error: {e}")
            # Return empty results if SQL parsing fails
            results = {
                "zone_temperatures": [],
                "outdoor_temperatures": [],
                "heating_energy": 0.0,
                "cooling_energy": 0.0,
                "surface_outside_temps": [],
                "surface_inside_temps": [],
            }

        return results

    def compare_results(
        self,
        fluxion_results: Dict,
        ep_results: Dict,
        tolerance_abs: float = 1.0,
        tolerance_rel: float = 0.05,
    ) -> Dict:
        """
        Compare Fluxion and EnergyPlus results.

        Returns:
            Comparison report with metrics
        """
        report = {
            "passed": True,
            "metrics": {},
            "details": {},
        }

        # Compare zone temperatures
        if "zone_temperatures" in fluxion_results and "zone_temperatures" in ep_results:
            fluxion_temps = fluxion_results["zone_temperatures"]
            ep_temps = ep_results["zone_temperatures"]

            if len(fluxion_temps) != len(ep_temps):
                report["details"]["temperature_length_mismatch"] = {
                    "fluxion": len(fluxion_temps),
                    "ep": len(ep_temps),
                }

            # Calculate RMSE
            import math

            n = min(len(fluxion_temps), len(ep_temps))
            if n > 0:
                se = sum((fluxion_temps[i] - ep_temps[i]) ** 2 for i in range(n))
                rmse = math.sqrt(se / n)
                report["metrics"]["temperature_rmse"] = rmse

                # Check max absolute error
                max_abs = max(abs(fluxion_temps[i] - ep_temps[i]) for i in range(n))
                report["metrics"]["temperature_max_abs"] = max_abs

                # Check max relative error
                max_rel = max(
                    (
                        abs(fluxion_temps[i] - ep_temps[i]) / abs(ep_temps[i])
                        if ep_temps[i] != 0
                        else 0
                    )
                    for i in range(n)
                )
                report["metrics"]["temperature_max_rel"] = max_rel

                # Check criteria
                report["details"]["temperature_criteria"] = {
                    "rmse_pass": rmse < tolerance_abs,
                    "max_abs_pass": max_abs < tolerance_abs,
                    "max_rel_pass": max_rel < tolerance_rel,
                }

                if not all(report["details"]["temperature_criteria"].values()):
                    report["passed"] = False

        # Compare energy consumption
        if "heating_energy" in fluxion_results and "heating_energy" in ep_results:
            fluxion_heat = fluxion_results["heating_energy"]
            ep_heat = ep_results["heating_energy"]
            heat_error = abs(fluxion_heat - ep_heat)
            heat_rel = heat_error / abs(ep_heat) if ep_heat != 0 else 0

            report["metrics"]["heating_energy_abs_error"] = heat_error
            report["metrics"]["heating_energy_rel_error"] = heat_rel
            report["details"]["heating_energy_pass"] = heat_rel < tolerance_rel

            if heat_rel >= tolerance_rel:
                report["passed"] = False

        if "cooling_energy" in fluxion_results and "cooling_energy" in ep_results:
            fluxion_cool = fluxion_results["cooling_energy"]
            ep_cool = ep_results["cooling_energy"]
            cool_error = abs(fluxion_cool - ep_cool)
            cool_rel = cool_error / abs(ep_cool) if ep_cool != 0 else 0

            report["metrics"]["cooling_energy_abs_error"] = cool_error
            report["metrics"]["cooling_energy_rel_error"] = cool_rel
            report["details"]["cooling_energy_pass"] = cool_rel < tolerance_rel

            if cool_rel >= tolerance_rel:
                report["passed"] = False

        return report


def load_test_cases() -> Dict:
    """Load test cases from catalog."""
    catalog_path = Path(__file__).parent.parent / "refdata" / "ep_test_cases.toml"

    if not catalog_path.exists():
        return {}

    try:
        import tomli

        with open(catalog_path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        print("Warning: tomli not installed, using built-in test cases")
        return get_builtin_test_cases()


def get_builtin_test_cases() -> Dict:
    """Return built-in test cases (fallback if TOML not available)."""
    return {
        "case": [
            {
                "id": "600",
                "name": "Heavyweight - Summer",
                "category": "conduction",
                "floor_area": 48.0,
                "walls_u": 0.358,
                "roof_u": 0.226,
                "floor_u": 0.398,
                "window_u": 2.943,
                "window_area": 6.0,
                "setpoint_heating": 20.0,
                "setpoint_cooling": 27.0,
                "epw": "refdata/epw/Denver.epw",
            },
            {
                "id": "900",
                "name": "Lightweight - Summer",
                "category": "conduction",
                "floor_area": 48.0,
                "walls_u": 0.358,
                "roof_u": 0.226,
                "floor_u": 0.398,
                "window_u": 2.943,
                "window_area": 6.0,
                "setpoint_heating": 20.0,
                "setpoint_cooling": 27.0,
                "epw": "refdata/epw/Denver.epw",
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="EnergyPlus Test Oracle")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate EP reference data")
    gen_parser.add_argument("--case", help="Specific test case ID")
    gen_parser.add_argument(
        "--all-cases", action="store_true", help="Generate all cases"
    )
    gen_parser.add_argument(
        "--output-dir", default="refdata/ep", help="Output directory"
    )

    # Compare command
    comp_parser = subparsers.add_parser(
        "compare", help="Compare Fluxion and EP results"
    )
    comp_parser.add_argument("--fluxion", required=True, help="Fluxion output JSON")
    comp_parser.add_argument("--ep", required=True, help="EP output JSON")
    comp_parser.add_argument(
        "--tol-abs", type=float, default=1.0, help="Absolute tolerance"
    )
    comp_parser.add_argument(
        "--tol-rel", type=float, default=0.05, help="Relative tolerance"
    )

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate Fluxion against EP")
    val_parser.add_argument("--test-case", required=True, help="Test case ID")
    val_parser.add_argument(
        "--fluxion-output", required=True, help="Fluxion output JSON"
    )

    args = parser.parse_args()

    if args.command == "generate":
        oracle = EnergyPlusOracle()
        test_cases = load_test_cases()

        if args.all_cases:
            cases = test_cases.get("case", [])
        elif args.case:
            cases = [c for c in test_cases.get("case", []) if c["id"] == args.case]
            if not cases:
                print(f"Test case {args.case} not found")
                sys.exit(1)
        else:
            print("Please specify --case or --all-cases")
            sys.exit(1)

        for case in cases:
            print(f"\nGenerating EP reference for case {case['id']}...")
            idf_content = oracle.generate_idf(**case)

            # Write IDF
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            idf_path = output_dir / f"Case_{case['id']}.idf"

            with open(idf_path, "w") as f:
                f.write(idf_content)

            print(f"  IDF written to: {idf_path}")

            # Check if EPW file exists
            epw_path = Path(case.get("epw", f"refdata/epw/Denver.epw"))
            if not epw_path.exists():
                print(f"  Warning: EPW file not found: {epw_path}")
                print(f"  Skipping simulation for case {case['id']}")
                continue

            # Run EnergyPlus
            try:
                results = oracle.run_energyplus(
                    str(idf_path),
                    str(epw_path),
                    str(output_dir / f"case_{case['id']}"),
                )

                # Save results as JSON
                result_path = output_dir / f"Case_{case['id']}_results.json"
                with open(result_path, "w") as f:
                    json.dump(results, f, indent=2)

                print(f"  Results saved to: {result_path}")
                print(f"  Heating energy: {results.get('heating_energy', 0):.2f} Wh")
                print(f"  Cooling energy: {results.get('cooling_energy', 0):.2f} Wh")

            except Exception as e:
                print(f"  Error running EnergyPlus: {e}")
                continue

    elif args.command == "compare":
        with open(args.fluxion) as f:
            fluxion_data = json.load(f)

        with open(args.ep) as f:
            ep_data = json.load(f)

        oracle = EnergyPlusOracle()
        report = oracle.compare_results(
            fluxion_data, ep_data, args.tol_abs, args.tol_rel
        )

        print("\nComparison Report:")
        print(f"  Overall: {'PASS' if report['passed'] else 'FAIL'}")
        print("\n  Metrics:")
        for key, value in report.get("metrics", {}).items():
            print(f"    {key}: {value:.4f}")

        print("\n  Details:")
        for key, value in report.get("details", {}).items():
            print(f"    {key}: {value}")

        sys.exit(0 if report["passed"] else 1)

    elif args.command == "validate":
        oracle = EnergyPlusOracle()
        test_cases = load_test_cases()

        case = next(
            (c for c in test_cases.get("case", []) if c["id"] == args.test_case), None
        )
        if not case:
            print(f"Test case {args.test_case} not found")
            sys.exit(1)

        # Load EP reference
        ep_path = Path(args.output_dir) / f"Case_{case['id']}_results.json"
        if not ep_path.exists():
            print(f"EP reference not found: {ep_path}")
            print("Run 'generate' first")
            sys.exit(1)

        with open(ep_path) as f:
            ep_data = json.load(f)

        with open(args.fluxion_output) as f:
            fluxion_data = json.load(f)

        report = oracle.compare_results(fluxion_data, ep_data)

        print(f"\nValidation for Case {case['id']}:")
        print(f"  Status: {'PASS' if report['passed'] else 'FAIL'}")

        if not report["passed"]:
            print("\n  Failures:")
            for key, value in report.get("details", {}).items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if v is False:
                            print(f"    {key}.{k}: FAILED")

        sys.exit(0 if report["passed"] else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
