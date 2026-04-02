#!/usr/bin/env python3
"""
Generate EnergyPlus reference data for ASHRAE 140 cases.

This script uses OpenStudio Python API to generate EnergyPlus simulations
and extract hourly data for comparison with Fluxion results.
"""

import argparse
import sys
from pathlib import Path

# Add OpenStudio Python path
sys.path.insert(0, "/usr/local/openstudio-3.11.0/Python")


def create_idf_for_case(case_id: str, output_dir: Path):
    """Create an OpenStudio IDF file for an ASHRAE 140 case."""
    # Get case specification from fluxion
    import importlib.util

    importlib.util.spec_from_file_location(
        "ashrae_140_cases",
        str(Path(__file__).parent / "src/validation/ashrae_140_cases.rs"),
    )
    print(f"Case {case_id}: Would need to integrate with Rust case specs")
    return None


def run_energyplus(idf_path: Path, epw_path: Path, output_dir: Path):
    """Run EnergyPlus simulation."""
    # OpenStudio includes EnergyPlus runner
    # This would typically use openstudio.run_manager or similar
    print(f"Would run EnergyPlus with IDF: {idf_path} and EPW: {epw_path}")
    return None


def extract_hourly_results(output_dir: Path, case_id: str):
    """Extract hourly results from EnergyPlus output."""
    print(f"Would extract hourly results for case {case_id}")
    return None


def generate_reference_data(case_id: str, output_dir: Path):
    """Generate EnergyPlus reference data for a specific case."""
    print(f"Generating EnergyPlus reference for case {case_id}")

    # This is a placeholder - full implementation would:
    # 1. Create IDF using OpenStudio API
    # 2. Run EnergyPlus simulation
    # 3. Extract hourly data (temperatures, heating/cooling loads)
    # 4. Save to JSON for comparison

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate EnergyPlus reference data for ASHRAE 140 cases"
    )
    parser.add_argument(
        "--case",
        type=str,
        help="Case ID to generate (e.g., 600, 620, 920, 930)",
    )
    parser.add_argument(
        "--all-discrepancies",
        action="store_true",
        help="Generate for all cases with significant discrepancies",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="refdata",
        help="Output directory for reference data",
    )

    args = parser.parse_args()

    # Cases with significant discrepancies from Fluxion validation
    discrepancy_cases = [
        "600",
        "610",
        "620",
        "630",
        "640",
        "650",
        "900",
        "910",
        "920",
        "930",
        "940",
        "950",
    ]

    cases_to_generate = (
        discrepancy_cases
        if args.all_discrepancies
        else [args.case]
        if args.case
        else []
    )

    if not cases_to_generate:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("OpenStudio version check:")
    try:
        import openstudio  # noqa: F811

        openstudio.model.Model()
        print("  OpenStudio Model created successfully")
    except Exception as e:
        print(f"  Error creating OpenStudio model: {e}")

    for case_id in cases_to_generate:
        result = generate_reference_data(case_id, output_dir)
        print(f"Generated reference for case {case_id}: {result}")


if __name__ == "__main__":
    main()
