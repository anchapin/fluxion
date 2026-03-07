"""
Compliance Agent CLI Tool

This module provides a command-line interface for generating ASHRAE 90.1 and IECC
compliance reports using Fluxion simulation data and optional LLM integration.

Usage:
    python -m tools.compliance_agent --help
    python -m tools.compliance_agent check --simulation output.json
    python -m tools.compliance_agent report --proposed proposed.json --baseline baseline.json
    python -m tools.compliance_agent llm-report --metrics metrics.json

Requirements:
    - api.compliance module (already implemented)
    - Optional: LLM model for natural language report generation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.compliance import (
    ComplianceDataAggregator,
    ComplianceMetrics,
    create_sample_metrics,
    generate_compliance_report,
    create_prompt_for_llm,
)


class ComplianceCLI:
    """Command-line interface for the compliance agent."""
    
    def __init__(self):
        from api.compliance.agent import ComplianceAgent as APIComplianceAgent
        self.api_agent = APIComplianceAgent()
    
    def check_compliance(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
        standard: str = "ASHRAE 90.1-2019"
    ) -> Dict[str, Any]:
        """
        Check compliance for proposed design against baseline.
        
        Args:
            proposed_metrics: Metrics for the proposed design
            baseline_metrics: Metrics for baseline design (optional)
            standard: Compliance standard to use
        
        Returns:
            Dictionary with compliance results
        """
        # Use the API agent directly
        return self.api_agent.check_compliance(
            proposed_metrics=proposed_metrics,
            baseline_metrics=baseline_metrics,
        )
    
    def generate_report(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
        project_name: str = "Project",
        building_name: str = "Building",
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate Markdown compliance report.
        
        Args:
            proposed_metrics: Proposed design metrics
            baseline_metrics: Baseline design metrics
            project_name: Project name
            building_name: Building name
            output_path: Optional path to save report
        
        Returns:
            Markdown report content
        """
        report = generate_compliance_report(
            proposed_metrics=proposed_metrics,
            baseline_metrics=baseline_metrics,
            project_name=project_name,
            building_name=building_name,
        )
        
        if output_path:
            output_path.write_text(report)
            print(f"Report saved to: {output_path}")
        
        return report
    
    def generate_llm_prompt(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
    ) -> Dict[str, str]:
        """
        Generate prompts for LLM-based report generation.
        
        Args:
            proposed_metrics: Proposed design metrics
            baseline_metrics: Baseline design metrics
        
        Returns:
            Dictionary with system_prompt and user_prompt_template
        """
        template = create_prompt_for_llm(
            metrics=proposed_metrics,
            baseline_metrics=baseline_metrics,
        )
        
        return {
            "system_prompt": template.system_prompt,
            "user_prompt_template": template.user_prompt_template,
        }


def load_metrics_from_file(filepath: Path) -> ComplianceMetrics:
    """
    Load compliance metrics from JSON file.
    
    Args:
        filepath: Path to JSON file with metrics
    
    Returns:
        ComplianceMetrics object
    """
    data = json.loads(filepath.read_text())
    
    # Create metrics from loaded data
    metrics = ComplianceMetrics()
    
    # Building info
    metrics.building_name = data.get("building_name", "Unknown")
    metrics.building_area_m2 = data.get("building_area_m2", 0)
    metrics.building_type = data.get("building_type", "Commercial")
    metrics.climate_zone = data.get("climate_zone", "4A")
    
    # Energy
    metrics.total_energy_kwh = data.get("total_energy_kwh", 0)
    metrics.total_eui_kwh_m2 = data.get("total_eui_kwh_m2", 0)
    metrics.total_electricity_kwh = data.get("total_electricity_kwh", 0)
    metrics.total_natural_gas_kwh = data.get("total_natural_gas_kwh", 0)
    
    # Peak loads
    metrics.peak_heating_load_kw = data.get("peak_heating_load_kw", 0)
    metrics.peak_cooling_load_kw = data.get("peak_cooling_load_kw", 0)
    metrics.peak_electric_demand_kw = data.get("peak_electric_demand_kw", 0)
    
    # Unmet hours
    metrics.unmet_heating_hours = data.get("unmet_heating_hours", 0)
    metrics.unmet_cooling_hours = data.get("unmet_cooling_hours", 0)
    metrics.total_unmet_hours = data.get("total_unmet_hours", 0)
    
    # End uses
    metrics.heating_energy_kwh = data.get("heating_energy_kwh", 0)
    metrics.cooling_energy_kwh = data.get("cooling_energy_kwh", 0)
    metrics.lighting_energy_kwh = data.get("lighting_energy_kwh", 0)
    metrics.plug_loads_kwh = data.get("plug_loads_kwh", 0)
    metrics.ventilation_energy_kwh = data.get("ventilation_energy_kwh", 0)
    
    # Financial
    metrics.annual_energy_cost_usd = data.get("annual_energy_cost_usd", 0)
    metrics.electricity_rate_usd_kwh = data.get("electricity_rate_usd_kwh", 0.12)
    metrics.gas_rate_usd_kwh = data.get("gas_rate_usd_kwh", 0.08)
    
    return metrics


def save_metrics_to_file(metrics: ComplianceMetrics, filepath: Path):
    """
    Save compliance metrics to JSON file.
    
    Args:
        metrics: ComplianceMetrics to save
        filepath: Path to output JSON file
    """
    data = {
        "building_name": metrics.building_name,
        "building_area_m2": metrics.building_area_m2,
        "building_type": metrics.building_type,
        "climate_zone": metrics.climate_zone,
        "total_energy_kwh": metrics.total_energy_kwh,
        "total_eui_kwh_m2": metrics.total_eui_kwh_m2,
        "total_electricity_kwh": metrics.total_electricity_kwh,
        "total_natural_gas_kwh": metrics.total_natural_gas_kwh,
        "peak_heating_load_kw": metrics.peak_heating_load_kw,
        "peak_cooling_load_kw": metrics.peak_cooling_load_kw,
        "peak_electric_demand_kw": metrics.peak_electric_demand_kw,
        "unmet_heating_hours": metrics.unmet_heating_hours,
        "unmet_cooling_hours": metrics.unmet_cooling_hours,
        "total_unmet_hours": metrics.total_unmet_hours,
        "heating_energy_kwh": metrics.heating_energy_kwh,
        "cooling_energy_kwh": metrics.cooling_energy_kwh,
        "lighting_energy_kwh": metrics.lighting_energy_kwh,
        "plug_loads_kwh": metrics.plug_loads_kwh,
        "ventilation_energy_kwh": metrics.ventilation_energy_kwh,
        "annual_energy_cost_usd": metrics.annual_energy_cost_usd,
    }
    filepath.write_text(json.dumps(data, indent=2))


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Fluxion Compliance Agent - ASHRAE 90.1 / IECC compliance reporting"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Check compliance
    check_parser = subparsers.add_parser("check", help="Check compliance")
    check_parser.add_argument(
        "--proposed", "-p", required=True, type=Path,
        help="Path to proposed design metrics JSON"
    )
    check_parser.add_argument(
        "--baseline", "-b", type=Path,
        help="Path to baseline design metrics JSON"
    )
    check_parser.add_argument(
        "--standard", "-s", default="ASHRAE 90.1-2019",
        choices=["ASHRAE 90.1-2019", "ASHRAE 90.1-2022", "IECC 2021", "IECC 2024"],
        help="Compliance standard"
    )
    check_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output file for results (JSON)"
    )
    
    # Generate report
    report_parser = subparsers.add_parser("report", help="Generate compliance report")
    report_parser.add_argument(
        "--proposed", "-p", required=True, type=Path,
        help="Path to proposed design metrics JSON"
    )
    report_parser.add_argument(
        "--baseline", "-b", type=Path,
        help="Path to baseline design metrics JSON"
    )
    report_parser.add_argument(
        "--project", default="Project",
        help="Project name"
    )
    report_parser.add_argument(
        "--building", default="Building",
        help="Building name"
    )
    report_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output file for report (Markdown)"
    )
    
    # Generate LLM prompt
    prompt_parser = subparsers.add_parser("prompt", help="Generate LLM prompts")
    prompt_parser.add_argument(
        "--proposed", "-p", required=True, type=Path,
        help="Path to proposed design metrics JSON"
    )
    prompt_parser.add_argument(
        "--baseline", "-b", type=Path,
        help="Path to baseline design metrics JSON"
    )
    prompt_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output file for prompts (JSON)"
    )
    
    # Sample data
    sample_parser = subparsers.add_parser("sample", help="Generate sample metrics")
    sample_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output file for sample metrics (JSON)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ComplianceCLI()
    
    if args.command == "check":
        # Load proposed metrics
        proposed = load_metrics_from_file(args.proposed)
        baseline = load_metrics_from_file(args.baseline) if args.baseline else None
        
        # Check compliance
        result = cli.check_compliance(
            proposed_metrics=proposed,
            baseline_metrics=baseline,
            standard=args.standard,
        )
        
        # Output result
        output = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(output)
            print(f"Compliance check saved to: {args.output}")
        else:
            print(output)
    
    elif args.command == "report":
        # Load metrics
        proposed = load_metrics_from_file(args.proposed)
        baseline = load_metrics_from_file(args.baseline) if args.baseline else None
        
        # Generate report
        report = cli.generate_report(
            proposed_metrics=proposed,
            baseline_metrics=baseline,
            project_name=args.project,
            building_name=args.building,
            output_path=args.output,
        )
        
        if not args.output:
            print(report)
    
    elif args.command == "prompt":
        # Load metrics
        proposed = load_metrics_from_file(args.proposed)
        baseline = load_metrics_from_file(args.baseline) if args.baseline else None
        
        # Generate prompts
        prompts = cli.generate_llm_prompt(
            proposed_metrics=proposed,
            baseline_metrics=baseline,
        )
        
        output = json.dumps(prompts, indent=2)
        if args.output:
            args.output.write_text(output)
            print(f"Prompts saved to: {args.output}")
        else:
            print(output)
    
    elif args.command == "sample":
        # Generate sample metrics
        metrics = create_sample_metrics()
        
        if args.output:
            save_metrics_to_file(metrics, args.output)
            print(f"Sample metrics saved to: {args.output}")
        else:
            # Print as JSON
            data = {
                "building_name": metrics.building_name,
                "building_area_m2": metrics.building_area_m2,
                "building_type": metrics.building_type,
                "climate_zone": metrics.climate_zone,
                "total_energy_kwh": metrics.total_energy_kwh,
                "total_eui_kwh_m2": metrics.total_eui_kwh_m2,
                "peak_heating_load_kw": metrics.peak_heating_load_kw,
                "peak_cooling_load_kw": metrics.peak_cooling_load_kw,
                "total_unmet_hours": metrics.total_unmet_hours,
            }
            print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
