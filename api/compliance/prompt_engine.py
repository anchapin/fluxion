"""
Prompt Engineering Engine for Code Compliance

This module provides templated prompts that inject building energy metrics
into an LLM context window for generating ASHRAE 90.1 and IECC compliance reports.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from api.compliance.data_aggregation import ComplianceMetrics


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    ASHRAE_90_1_2019 = "ASHRAE 90.1-2019"
    ASHRAE_90_1_2022 = "ASHRAE 90.1-2022"
    IECC_2021 = "IECC 2021"
    IECC_2024 = "IECC 2024"


class ReportFormat(Enum):
    """Output format for compliance reports."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"
    HTML = "html"


@dataclass
class PromptTemplate:
    """Container for prompt templates."""
    system_prompt: str
    user_prompt_template: str
    description: str


class CompliancePromptEngine:
    """
    Prompt engineering engine for code compliance reporting.
    
    Creates templated prompts that inject building energy metrics into an LLM
    context window to generate standardized compliance documents.
    """
    
    # ASHRAE 90.1 Appendix G compliance requirements
    ASHRAE_90_1_REQUIREMENTS = {
        "performance_improvement": {
            "required": True,
            "minimum_percent": 50.0,  # 50% better than baseline for EUI
        },
        "peak_demand": {
            "required": True,
            "maximum_percent": 100.0,  # Cannot exceed baseline
        },
        "unmet_hours": {
            "required": True,
            "maximum_hours": 300,  # Hours outside setpoint
        },
    }
    
    # IECC compliance requirements
    IECC_REQUIREMENTS = {
        "performance_improvement": {
            "required": True,
            "minimum_percent": 50.0,
        },
        "unmet_hours": {
            "required": True,
            "maximum_hours": 400,
        },
    }
    
    def __init__(self, standard: ComplianceStandard = ComplianceStandard.ASHRAE_90_1_2019):
        """
        Initialize the prompt engine.
        
        Args:
            standard: The compliance standard to use
        """
        self.standard = standard
    
    def generate_compliance_report_prompt(
        self,
        metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
        report_format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> PromptTemplate:
        """
        Generate a prompt for creating a compliance report.
        
        Args:
            metrics: The proposed building metrics
            baseline_metrics: Optional baseline building metrics for comparison
            report_format: Desired output format
        
        Returns:
            PromptTemplate with system and user prompts
        """
        # Build context with metrics
        metrics_context = self._build_metrics_context(metrics, baseline_metrics)
        
        # Build system prompt based on standard
        system_prompt = self._build_system_prompt(report_format)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(
            metrics_context,
            baseline_metrics is not None,
            report_format
        )
        
        return PromptTemplate(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            description=f"Generate {report_format.value} compliance report"
        )
    
    def _build_metrics_context(
        self,
        metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None
    ) -> str:
        """
        Build the metrics context string for the prompt.
        
        Args:
            metrics: Proposed building metrics
            baseline_metrics: Optional baseline metrics
        
        Returns:
            Formatted metrics context string
        """
        lines = []
        
        # Building Information
        lines.append("# Building Information")
        lines.append(f"- Building Name: {metrics.building_name}")
        lines.append(f"- Building Area: {metrics.building_area_m2:.1f} m²")
        lines.append(f"- Building Type: {metrics.building_type}")
        lines.append(f"- Climate Zone: {metrics.climate_zone}")
        lines.append("")
        
        # Annual Energy
        lines.append("# Annual Energy Consumption")
        lines.append(f"- Total Energy: {metrics.total_energy_kwh:,.1f} kWh")
        lines.append(f"- Energy Use Intensity (EUI): {metrics.total_eui_kwh_m2:.2f} kWh/m²/year")
        lines.append(f"- Annual Energy Cost: ${metrics.annual_energy_cost_usd:,.2f}")
        lines.append("")
        
        # Peak Loads
        lines.append("# Peak Loads")
        lines.append(f"- Peak Heating: {metrics.peak_heating_load_kw:.1f} kW")
        lines.append(f"- Peak Cooling: {metrics.peak_cooling_load_kw:.1f} kW")
        lines.append(f"- Peak Electric Demand: {metrics.peak_electric_demand_kw:.1f} kW")
        lines.append("")
        
        # Unmet Hours
        lines.append("# Thermal Comfort (Unmet Hours)")
        lines.append(f"- Unmet Heating Hours: {metrics.unmet_heating_hours:.0f}")
        lines.append(f"- Unmet Cooling Hours: {metrics.unmet_cooling_hours:.0f}")
        lines.append(f"- Total Unmet Hours: {metrics.total_unmet_hours:.0f}")
        lines.append("")
        
        # End-use breakdown
        lines.append("# End-Use Breakdown")
        lines.append(f"- Heating: {metrics.heating_energy_kwh:,.1f} kWh")
        lines.append(f"- Cooling: {metrics.cooling_energy_kwh:,.1f} kWh")
        lines.append(f"- Lighting: {metrics.lighting_energy_kwh:,.1f} kWh")
        lines.append(f"- Plug Loads: {metrics.plug_loads_kwh:,.1f} kWh")
        lines.append(f"- Ventilation: {metrics.ventilation_energy_kwh:,.1f} kWh")
        
        # Add baseline comparison if available
        if baseline_metrics:
            lines.append("")
            lines.append("# BASELINE vs PROPOSED Comparison")
            
            # Calculate improvements
            energy_reduction = (
                (baseline_metrics.total_energy_kwh - metrics.total_energy_kwh) /
                baseline_metrics.total_energy_kwh * 100
            ) if baseline_metrics.total_energy_kwh > 0 else 0
            
            cost_savings = (
                baseline_metrics.annual_energy_cost_usd - metrics.annual_energy_cost_usd
            )
            
            lines.append(f"- Baseline EUI: {baseline_metrics.total_eui_kwh_m2:.2f} kWh/m²/year")
            lines.append(f"- Proposed EUI: {metrics.total_eui_kwh_m2:.2f} kWh/m²/year")
            lines.append(f"- Energy Reduction: {energy_reduction:.1f}%")
            lines.append(f"- Annual Cost Savings: ${cost_savings:,.2f}")
        
        return "\n".join(lines)
    
    def _build_system_prompt(self, report_format: ReportFormat) -> str:
        """
        Build the system prompt based on the standard and format.
        
        Args:
            report_format: Desired output format
        
        Returns:
            System prompt string
        """
        base_system = """You are an expert building energy analyst and code compliance specialist.
You generate professional compliance reports for building energy codes including ASHRAE 90.1 and IECC.
Your reports are precise, well-structured, and follow industry-standard formats."""
        
        if self.standard == ComplianceStandard.ASHRAE_90_1_2019:
            base_system += """
You are familiar with ASHRAE Standard 90.1-2019 (Energy Standard for Buildings Except Low-Rise Residential Buildings).
Specifically, you understand Appendix G: Performance Rating Method and can generate compliance documentation."""
        elif self.standard == ComplianceStandard.ASHRAE_90_1_2022:
            base_system += """
You are familiar with ASHRAE Standard 90.1-2022 (Energy Standard for Buildings Except Low-Rise Residential Buildings).
Specifically, you understand Appendix G: Performance Rating Method and can generate compliance documentation."""
        elif self.standard in (ComplianceStandard.IECC_2021, ComplianceStandard.IECC_2024):
            base_system += """
You are familiar with the International Energy Conservation Code (IECC) and its compliance requirements.
You understand the performance-based and prescriptive compliance paths."""
        
        if report_format == ReportFormat.MARKDOWN:
            base_system += """
Output your response in clean, well-formatted Markdown.
Use tables, headers, and bullet points for clarity.
Include a summary section at the beginning."""
        
        return base_system
    
    def _build_user_prompt(
        self,
        metrics_context: str,
        has_baseline: bool,
        report_format: ReportFormat
    ) -> str:
        """
        Build the user prompt.
        
        Args:
            metrics_context: Formatted metrics context
            has_baseline: Whether baseline metrics are available
            report_format: Desired output format
        
        Returns:
            User prompt string
        """
        prompt = f"""Generate a code compliance report based on the following building energy simulation results:

{metrics_context}

"""
        
        if has_baseline:
            prompt += """Please generate an ASHRAE 90.1 Appendix G compliance report that includes:

1. **Executive Summary**: A brief overview of the compliance determination
2. **Building Description**: Summary of the building characteristics
3. **Energy Analysis Summary Table**: Present the baseline vs. proposed comparison in a clear table format
4. **Performance Metrics**: Detailed breakdown of all energy performance metrics
5. **Compliance Determination**: State whether the proposed design meets the required performance threshold
6. **Recommendations**: Any suggestions for improving energy performance

Use the following compliance criteria:
- The proposed design must demonstrate at least 50% energy cost improvement over the baseline (ASHRAE 90.1 Appendix G)
- Total unmet hours should not exceed 300 hours per year
- Peak demand should not exceed baseline

"""
        else:
            prompt += """Please generate an energy compliance summary that includes:

1. **Building Summary**: Overview of the building and its energy characteristics
2. **Energy Performance**: Detailed breakdown of energy consumption and EUI
3. **Thermal Comfort Analysis**: Analysis of unmet hours and thermal comfort
4. **Compliance Notes**: Notes on how this building would typically comply with energy codes

"""
        
        if report_format == ReportFormat.MARKDOWN:
            prompt += "Format the entire response as clean Markdown."
        
        return prompt
    
    def generate_determination_prompt(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: ComplianceMetrics,
    ) -> PromptTemplate:
        """
        Generate a focused prompt for compliance determination.
        
        Args:
            proposed_metrics: Proposed building metrics
            baseline_metrics: Baseline building metrics
        
        Returns:
            PromptTemplate for compliance determination
        """
        # Calculate key metrics
        energy_reduction = (
            (baseline_metrics.total_energy_kwh - proposed_metrics.total_energy_kwh) /
            baseline_metrics.total_energy_kwh * 100
        ) if baseline_metrics.total_energy_kwh > 0 else 0
        
        cost_savings = (
            baseline_metrics.annual_energy_cost_usd - proposed_metrics.annual_energy_cost_usd
        )
        
        system_prompt = """You are a building code compliance officer.
Determine whether a proposed building design complies with energy code requirements.
Provide a clear YES or NO determination with supporting rationale."""
        
        user_prompt = f"""Based on the following ASHRAE 90.1 Appendix G performance comparison:

**Baseline Building:**
- Annual Energy: {baseline_metrics.total_energy_kwh:,.0f} kWh
- EUI: {baseline_metrics.total_eui_kwh_m2:.2f} kWh/m²/year
- Annual Cost: ${baseline_metrics.annual_energy_cost_usd:,.2f}
- Peak Heating: {baseline_metrics.peak_heating_load_kw:.1f} kW
- Peak Cooling: {baseline_metrics.peak_cooling_load_kw:.1f} kW

**Proposed Building:**
- Annual Energy: { proposed_metrics.total_energy_kwh:,.0f} kWh
- EUI: { proposed_metrics.total_eui_kwh_m2:.2f} kWh/m²/year
- Annual Cost: ${ proposed_metrics.annual_energy_cost_usd:,.2f}
- Peak Heating: { proposed_metrics.peak_heating_load_kw:.1f} kW
- Peak Cooling: { proposed_metrics.peak_cooling_load_kw:.1f} kW
- Unmet Hours: { proposed_metrics.total_unmet_hours:.0f}

**Performance Improvement:**
- Energy Reduction: {energy_reduction:.1f}%
- Annual Savings: ${cost_savings:,.2f}

Does this building COMPLY with ASHRAE 90.1-2019 Appendix G requirements?
(Required: ≥50% improvement, unmet hours ≤300)

Provide:
1. COMPLIANT or NON-COMPLIANT determination
2. Summary of key findings
3. Any conditions or caveats"""
        
        return PromptTemplate(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            description="ASHRAE 90.1 compliance determination"
        )
    
    def create_json_prompt(
        self,
        metrics: ComplianceMetrics,
    ) -> Dict[str, Any]:
        """
        Create a structured prompt for JSON output.
        
        Args:
            metrics: Building metrics
        
        Returns:
            Dictionary with structured prompt data
        """
        return {
            "prompt_type": "compliance_report_json",
            "standard": self.standard.value,
            "metrics": {
                "building": {
                    "name": metrics.building_name,
                    "area_m2": metrics.building_area_m2,
                    "type": metrics.building_type,
                    "climate_zone": metrics.climate_zone,
                },
                "annual_energy": {
                    "total_kwh": round(metrics.total_energy_kwh, 2),
                    "eui_kwh_m2": round(metrics.total_eui_kwh_m2, 2),
                    "cost_usd": round(metrics.annual_energy_cost_usd, 2),
                },
                "peak_loads": {
                    "heating_kw": round(metrics.peak_heating_load_kw, 2),
                    "cooling_kw": round(metrics.peak_cooling_load_kw, 2),
                },
                "unmet_hours": round(metrics.total_unmet_hours, 0),
            },
            "output_format": "structured_json",
            "required_fields": [
                "compliance_determination",
                "performance_improvement_percent",
                "meets_eui_requirement",
                "meets_unmet_hours_requirement",
                "summary",
            ],
        }


def create_prompt_for_llm(
    metrics: ComplianceMetrics,
    baseline_metrics: Optional[ComplianceMetrics] = None,
    standard: ComplianceStandard = ComplianceStandard.ASHRAE_90_1_2019,
    format: ReportFormat = ReportFormat.MARKDOWN,
) -> PromptTemplate:
    """
    Convenience function to create a compliance prompt.
    
    Args:
        metrics: Proposed building metrics
        baseline_metrics: Optional baseline metrics
        standard: Compliance standard to use
        format: Output format
    
    Returns:
        PromptTemplate ready for LLM
    """
    engine = CompliancePromptEngine(standard=standard)
    return engine.generate_compliance_report_prompt(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        report_format=format,
    )


if __name__ == "__main__":
    # Demo
    from api.compliance.data_aggregation import create_sample_metrics
    
    metrics = create_sample_metrics()
    prompt = create_prompt_for_llm(
        metrics=metrics,
        baseline_metrics=metrics,  # Using same for demo
        standard=ComplianceStandard.ASHRAE_90_1_2019,
    )
    
    print("=== SYSTEM PROMPT ===")
    print(prompt.system_prompt)
    print("\n=== USER PROMPT ===")
    print(prompt.user_prompt_template[:1000] + "...")
