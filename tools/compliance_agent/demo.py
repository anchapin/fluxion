#!/usr/bin/env python3
"""
Demo Script for Code Compliance Agent

This script demonstrates the automated code compliance agent checking building
energy models against ASHRAE 90.1 and IECC standards.

Usage:
    # Run with mock backend (default, no external dependencies)
    python tools/compliance_agent/demo.py
    
    # Run with Ollama backend (requires local Ollama running)
    python tools/compliance_agent/demo.py --backend ollama --model llama2
    
    # Run with OpenAI backend (requires API key)
    python tools/compliance_agent/demo.py --backend openai --model gpt-4

Requirements:
    - Mock backend: No additional dependencies
    - Ollama: pip install httpx
    - OpenAI: pip install openai
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.compliance_agent import CodeComplianceAgent, Standard
from tools.compliance_agent.agent import DEFAULT_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Sample building model data for demonstration
SAMPLE_OFFICE_BUILDING = {
    "model_name": "Office Building A - Downtown",
    "building_type": "Commercial Office",
    "location": "Chicago, IL",
    "climate_zone": "5A",
    "floor_area_sqft": 50000,
    "number_of_floors": 10,
    
    # Envelope properties
    "wall_r_value": 15.0,  # ft²·°F·h/BTU
    "roof_r_value": 35.0,
    "floor_r_value": 12.0,
    "window_u_factor": 0.50,  # BTU/(h·ft²·°F)
    "window_shgc": 0.30,
    "door_u_factor": 0.65,
    "window_to_wall_ratio": 0.40,
    "infiltration_rate": 0.15,  # ACH
    
    # HVAC properties
    "hvac_type": "VAV with reheat",
    "heating_setpoint": 72.0,  # °F
    "cooling_setpoint": 74.0,  # °F
    "thermostat_deadband": 4.0,  # °F
    "hvac_cop": 3.2,  # Coefficient of Performance
    "hvac_ieer": 12.0,  # Integrated Energy Efficiency Ratio
    "ventilation_rate": 15.0,  # cfm/person
    "outdoor_air_per_person": 5.0,  # cfm/person
    "economizer": True,
    
    # Lighting properties
    "lighting_power_density": 0.85,  # W/sq ft
    "lighting_control_type": "Occupancy sensors",
    "daylighting_controls": True,
    "occupancy_sensors": True,
    
    # Service water heating
    "water_heater_type": "Heat pump",
    "water_heater_efficiency": 0.95,
    
    # Electric power
    "transformer_efficiency": 0.98,
    "motor_efficiency": 0.96,
    
    # Additional info
    "operating_hours": "8am-6pm weekdays",
    "occupancy_density": 150,  # sq ft/person
}


SAMPLE_RESIDENTIAL_BUILDING = {
    "model_name": "Single Family Home - Climate Zone 4",
    "building_type": "Residential",
    "location": "Atlanta, GA",
    "climate_zone": "3A",
    "floor_area_sqft": 2500,
    "number_of_floors": 2,
    
    # Envelope properties
    "wall_r_value": 20.0,
    "roof_r_value": 38.0,
    "floor_r_value": 19.0,
    "window_u_factor": 0.28,
    "window_shgc": 0.25,
    "door_u_factor": 0.35,
    "window_to_wall_ratio": 0.20,
    "infiltration_rate": 0.08,
    
    # HVAC properties
    "hvac_type": "Split system",
    "heating_setpoint": 70.0,
    "cooling_setpoint": 76.0,
    "thermostat_deadband": 2.0,
    "hvac_seer": 16.0,  # Seasonal Energy Efficiency Ratio
    "hvac_hspf": 10.0,  # Heating Seasonal Performance Factor
    "ventilation_rate": 0.03,  # ACH
    
    # Lighting properties
    "lighting_power_density": 0.7,
    "lighting_control_type": "Manual",
    "occupancy_sensors": False,
    
    # Service water heating
    "water_heater_type": "Heat pump water heater",
    "water_heater_efficiency": 3.0,  # COP
    
    # Additional info
    "operating_hours": "24/7",
    "occupancy_density": 4,  # persons
}


def run_demo(backend: str = "mock", model: str = "llama2", standard: str = "ASHRAE90.1-2019"):
    """Run the compliance agent demo."""
    
    print("\n" + "=" * 70)
    print("CODE COMPLIANCE AGENT DEMO")
    print("=" * 70)
    print(f"\nUsing backend: {backend}")
    if backend != "mock":
        print(f"Model: {model}")
    print(f"Standard: {standard}")
    print("-" * 70)
    
    # Initialize the agent
    agent_kwargs = {"backend": backend}
    
    if backend == "ollama":
        agent_kwargs["model"] = model
    elif backend == "openai":
        agent_kwargs["model"] = model
    
    agent = CodeComplianceAgent(**agent_kwargs)
    
    # Check if backend is available
    if not agent.is_available():
        print(f"\n⚠️  Warning: {backend} backend is not available.")
        print("   Falling back to rules engine only (no LLM).")
        print("   Install required packages or start the service to enable LLM.\n")
    
    # Demo 1: Commercial Office Building (ASHRAE 90.1)
    print("\n" + "-" * 70)
    print("Demo 1: Commercial Office Building")
    print("-" * 70)
    
    report = agent.check_compliance(
        model_data=SAMPLE_OFFICE_BUILDING,
        standard=standard,
        use_rules_engine=True
    )
    
    print(report.print_summary())
    
    # Save the report
    report_file = Path("compliance_report_office.json")
    agent.save_report(report, report_file)
    print(f"\nFull report saved to: {report_file}")
    
    # Demo 2: Residential Building (IECC)
    print("\n" + "-" * 70)
    print("Demo 2: Residential Building (IECC)")
    print("-" * 70)
    
    report2 = agent.check_compliance(
        model_data=SAMPLE_RESIDENTIAL_BUILDING,
        standard="IECC-2021",
        use_rules_engine=True
    )
    
    print(report2.print_summary())
    
    # Demo 3: Check with different configurations
    print("\n" + "-" * 70)
    print("Demo 3: Testing Different Configurations")
    print("-" * 70)
    
    # Non-compliant building
    non_compliant_building = SAMPLE_OFFICE_BUILDING.copy()
    non_compliant_building["model_name"] = "Non-Compliant Office Building"
    non_compliant_building["window_u_factor"] = 0.75  # Too high
    non_compliant_building["wall_r_value"] = 8.0  # Too low
    non_compliant_building["hvac_cop"] = 2.5  # Too low
    
    report3 = agent.check_compliance(
        model_data=non_compliant_building,
        standard=standard,
        use_rules_engine=True
    )
    
    print(report3.print_summary())
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


def run_demo_with_file(file_path: str, backend: str = "mock", standard: str = "ASHRAE90.1-2019"):
    """Run the compliance agent on a model data file."""
    
    print("\n" + "=" * 70)
    print("CODE COMPLIANCE AGENT - FILE MODE")
    print("=" * 70)
    
    # Initialize the agent
    agent = CodeComplianceAgent(backend=backend)
    
    # Load model data
    try:
        model_data = agent.load_model_data(file_path)
    except Exception as e:
        logger.error(f"Failed to load model data: {e}")
        sys.exit(1)
    
    print(f"\nLoaded model: {model_data.get('model_name', 'Unknown')}")
    print(f"File: {file_path}")
    print("-" * 70)
    
    # Run compliance check
    report = agent.check_compliance(
        model_data=model_data,
        standard=standard,
        use_rules_engine=True
    )
    
    print(report.print_summary())
    
    # Save the report
    output_path = Path(file_path).with_suffix(".compliance_report.json")
    agent.save_report(report, output_path)
    print(f"\nFull report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for Code Compliance Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with mock backend
  %(prog)s --backend ollama --model llama2   # Run with Ollama
  %(prog)s --backend openai --model gpt-4   # Run with OpenAI
  %(prog)s --file model_data.json            # Load from file
        """
    )
    
    parser.add_argument(
        "--backend",
        choices=["mock", "ollama", "openai"],
        default="mock",
        help="LLM backend to use (default: mock)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama2",
        help="Model name for Ollama/OpenAI (default: llama2)"
    )
    parser.add_argument(
        "--standard",
        type=str,
        default="ASHRAE90.1-2019",
        choices=["ASHRAE90.1-2019", "ASHRAE90.1-2022", "IECC-2021", "IECC-2024"],
        help="Compliance standard to check against (default: ASHRAE90.1-2019)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to model data JSON file (instead of demo data)"
    )
    
    args = parser.parse_args()
    
    if args.file:
        run_demo_with_file(args.file, args.backend, args.standard)
    else:
        run_demo(args.backend, args.model, args.standard)


if __name__ == "__main__":
    main()
