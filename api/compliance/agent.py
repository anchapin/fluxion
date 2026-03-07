"""
Automated Code Compliance Agent (ASHRAE 90.1 / IECC) via LLM

This module provides an LLM-powered agent that checks building energy models
for compliance with ASHRAE 90.1 and IECC standards.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from api.compliance.data_aggregation import (
    ComplianceMetrics,
    ComplianceDataAggregator,
    create_sample_metrics,
)
from api.compliance.prompt_engine import (
    ComplianceStandard,
    ReportFormat,
    CompliancePromptEngine,
    create_prompt_for_llm,
)
from api.compliance.report_generator import (
    ComplianceReportGenerator,
    ReportMetadata,
    generate_compliance_report,
)


@dataclass
class ComplianceAgentConfig:
    """Configuration for the compliance agent."""
    standard: str = "ASHRAE 90.1-2019"
    output_format: str = "markdown"
    electricity_rate: float = 0.12  # $/kWh
    gas_rate: float = 0.08  # $/kWh
    require_baseline: bool = True  # Require baseline comparison
    max_unmet_hours: float = 300
    min_improvement_percent: float = 50.0


class ComplianceAgent:
    """
    LLM-powered agent for building energy code compliance.
    
    This agent processes building energy simulation results and generates
    human-readable compliance reports using LLM technology.
    """
    
    def __init__(self, config: Optional[ComplianceAgentConfig] = None):
        """
        Initialize the compliance agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config or ComplianceAgentConfig()
        self._standard = self._parse_standard(self.config.standard)
        self._format = self._parse_format(self.config.output_format)
    
    def _parse_standard(self, standard: str) -> ComplianceStandard:
        """Parse standard string to enum."""
        mapping = {
            "ASHRAE 90.1-2019": ComplianceStandard.ASHRAE_90_1_2019,
            "ASHRAE 90.1-2022": ComplianceStandard.ASHRAE_90_1_2022,
            "IECC 2021": ComplianceStandard.IECC_2021,
            "IECC 2024": ComplianceStandard.IECC_2024,
        }
        return mapping.get(standard, ComplianceStandard.ASHRAE_90_1_2019)
    
    def _parse_format(self, format: str) -> ReportFormat:
        """Parse format string to enum."""
        mapping = {
            "markdown": ReportFormat.MARKDOWN,
            "pdf": ReportFormat.PDF,
            "json": ReportFormat.JSON,
            "html": ReportFormat.HTML,
        }
        return mapping.get(format, ReportFormat.MARKDOWN)
    
    def check_compliance(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
    ) -> Dict[str, Any]:
        """
        Check compliance with the configured standard.
        
        Args:
            proposed_metrics: Proposed building metrics
            baseline_metrics: Optional baseline metrics
        
        Returns:
            Dictionary with compliance determination and details
        """
        result = {
            "compliant": False,
            "standard": self.config.standard,
            "checks": [],
            "summary": "",
        }
        
        # Check if baseline is required
        if self.config.require_baseline and baseline_metrics is None:
            result["checks"].append({
                "name": "Baseline Comparison",
                "status": "FAIL",
                "message": "Baseline metrics required for compliance check"
            })
            result["summary"] = "Cannot determine compliance without baseline"
            return result
        
        if baseline_metrics:
            # Energy improvement check
            energy_reduction = (
                (baseline_metrics.total_energy_kwh - proposed_metrics.total_energy_kwh) /
                baseline_metrics.total_energy_kwh * 100
            ) if baseline_metrics.total_energy_kwh > 0 else 0
            
            result["checks"].append({
                "name": "Energy Cost Improvement",
                "threshold": f">={self.config.min_improvement_percent}%",
                "actual": f"{energy_reduction:.1f}%",
                "status": "PASS" if energy_reduction >= self.config.min_improvement_percent else "FAIL",
            })
            
            # Unmet hours check
            unmet = proposed_metrics.total_unmet_hours
            result["checks"].append({
                "name": "Unmet Hours",
                "threshold": f"<={self.config.max_unmet_hours}",
                "actual": f"{unmet:.0f} hours",
                "status": "PASS" if unmet <= self.config.max_unmet_hours else "FAIL",
            })
            
            # Peak demand check
            peak_heat_ok = proposed_metrics.peak_heating_load_kw <= baseline_metrics.peak_heating_load_kw
            peak_cool_ok = proposed_metrics.peak_cooling_load_kw <= baseline_metrics.peak_cooling_load_kw
            
            result["checks"].append({
                "name": "Peak Heating Demand",
                "threshold": "<=Baseline",
                "actual": f"{proposed_metrics.peak_heating_load_kw:.1f} kW",
                "status": "PASS" if peak_heat_ok else "FAIL",
            })
            
            result["checks"].append({
                "name": "Peak Cooling Demand",
                "threshold": "<=Baseline",
                "actual": f"{proposed_metrics.peak_cooling_load_kw:.1f} kW",
                "status": "PASS" if peak_cool_ok else "FAIL",
            })
            
            # Calculate cost savings
            cost_savings = (
                baseline_metrics.annual_energy_cost_usd - proposed_metrics.annual_energy_cost_usd
            )
            result["annual_savings_usd"] = cost_savings
            result["energy_reduction_percent"] = energy_reduction
            
            # Overall compliance
            all_passed = all(
                check["status"] == "PASS"
                for check in result["checks"]
            )
            result["compliant"] = all_passed
            
            if all_passed:
                result["summary"] = (
                    f"COMPLIANT: Building meets {self.config.standard} requirements "
                    f"with {energy_reduction:.1f}% energy improvement and "
                    f"${cost_savings:,.0f} annual savings."
                )
            else:
                failed = [c["name"] for c in result["checks"] if c["status"] == "FAIL"]
                result["summary"] = f"NON-COMPLIANT: Failed checks: {', '.join(failed)}"
        
        return result
    
    def generate_prompt(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
    ) -> Dict[str, Any]:
        """
        Generate an LLM prompt for compliance report generation.
        
        Args:
            proposed_metrics: Proposed building metrics
            baseline_metrics: Optional baseline metrics
        
        Returns:
            Dictionary with system_prompt and user_prompt
        """
        engine = CompliancePromptEngine(standard=self._standard)
        template = engine.generate_compliance_report_prompt(
            metrics=proposed_metrics,
            baseline_metrics=baseline_metrics,
            report_format=self._format,
        )
        
        return {
            "system_prompt": template.system_prompt,
            "user_prompt": template.user_prompt_template,
        }
    
    def generate_markdown_report(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
        project_name: str = "Project",
        building_name: str = "Building",
        building_address: str = "Address",
    ) -> str:
        """
        Generate a Markdown compliance report.
        
        Args:
            proposed_metrics: Proposed building metrics
            baseline_metrics: Optional baseline metrics
            project_name: Name of the project
            building_name: Name of the building
            building_address: Address of the building
        
        Returns:
            Markdown-formatted compliance report
        """
        return generate_compliance_report(
            proposed_metrics=proposed_metrics,
            baseline_metrics=baseline_metrics,
            project_name=project_name,
            building_name=building_name,
            building_address=building_address,
            standard=self._standard,
        )
    
    def run(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
        llm_client=None,
    ) -> Dict[str, Any]:
        """
        Run the full compliance agent workflow.
        
        Args:
            proposed_metrics: Proposed building metrics
            baseline_metrics: Optional baseline metrics
            llm_client: Optional LLM client for generating natural language reports
        
        Returns:
            Complete results including compliance check, report, and LLM response
        """
        results = {
            "compliance": self.check_compliance(proposed_metrics, baseline_metrics),
            "metrics": {
                "proposed": proposed_metrics,
                "baseline": baseline_metrics,
            },
        }
        
        # Generate markdown report
        results["markdown_report"] = self.generate_markdown_report(
            proposed_metrics=proposed_metrics,
            baseline_metrics=baseline_metrics,
        )
        
        # Generate LLM prompt
        results["llm_prompt"] = self.generate_prompt(
            proposed_metrics=proposed_metrics,
            baseline_metrics=baseline_metrics,
        )
        
        # Optionally call LLM
        if llm_client:
            try:
                response = llm_client.complete(
                    system_prompt=results["llm_prompt"]["system_prompt"],
                    user_prompt=results["llm_prompt"]["user_prompt"],
                )
                results["llm_response"] = response
            except Exception as e:
                results["llm_error"] = str(e)
        
        return results


def create_compliance_agent(
    standard: str = "ASHRAE 90.1-2019",
    output_format: str = "markdown",
) -> ComplianceAgent:
    """
    Convenience function to create a compliance agent.
    
    Args:
        standard: Compliance standard to use
        output_format: Output format (markdown, json)
    
    Returns:
        Configured ComplianceAgent
    """
    config = ComplianceAgentConfig(
        standard=standard,
        output_format=output_format,
    )
    return ComplianceAgent(config=config)


# Export key classes and functions
__all__ = [
    "ComplianceAgent",
    "ComplianceAgentConfig",
    "ComplianceMetrics",
    "ComplianceDataAggregator",
    "ComplianceStandard",
    "ReportFormat",
    "CompliancePromptEngine",
    "ComplianceReportGenerator",
    "ReportMetadata",
    "generate_compliance_report",
    "create_prompt_for_llm",
    "create_sample_metrics",
    "create_compliance_agent",
]
