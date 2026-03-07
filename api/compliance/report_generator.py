"""
Report Generation Module for Code Compliance

This module generates standardized Markdown or PDF compliance reports
from building energy metrics for AHJ (Authorities Having Jurisdiction) submission.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from api.compliance.data_aggregation import ComplianceMetrics, ComplianceDataAggregator
from api.compliance.prompt_engine import ComplianceStandard, ReportFormat


@dataclass
class ReportMetadata:
    """Metadata for compliance reports."""
    report_id: str
    project_name: str
    building_name: str
    building_address: str
    building_type: str
    climate_zone: str
    building_area_m2: float
    prepared_by: str = "Fluxion Automated Compliance Agent"
    date: str = ""
    standard: str = "ASHRAE 90.1-2019"
    
    def __post_init__(self):
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d")


class ComplianceReportGenerator:
    """
    Generates standardized compliance reports from building energy metrics.
    
    Supports both Markdown (for review) and structured formats for AHJ submission.
    """
    
    def __init__(
        self,
        metadata: ReportMetadata,
        standard: ComplianceStandard = ComplianceStandard.ASHRAE_90_1_2019,
    ):
        """
        Initialize the report generator.
        
        Args:
            metadata: Report metadata
            standard: Compliance standard being used
        """
        self.metadata = metadata
        self.standard = standard
    
    def generate_report(
        self,
        proposed_metrics: ComplianceMetrics,
        baseline_metrics: Optional[ComplianceMetrics] = None,
    ) -> str:
        """
        Generate a complete compliance report.
        
        Args:
            proposed_metrics: Proposed building metrics
            baseline_metrics: Optional baseline metrics for comparison
        
        Returns:
            Markdown-formatted compliance report
        """
        sections = []
        
        # Header
        sections.append(self._generate_header())
        
        # Executive Summary
        sections.append(self._generate_executive_summary(proposed_metrics, baseline_metrics))
        
        # Building Description
        sections.append(self._generate_building_description())
        
        # Energy Analysis Summary (Appendix G table)
        if baseline_metrics:
            sections.append(self._generate_energy_table(proposed_metrics, baseline_metrics))
        
        # Detailed Metrics
        sections.append(self._generate_detailed_metrics(proposed_metrics))
        
        # Compliance Determination
        if baseline_metrics:
            sections.append(self._generate_compliance_determination(proposed_metrics, baseline_metrics))
        
        # Appendices
        sections.append(self._generate_appendix())
        
        return "\n\n".join(sections)
    
    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Building Energy Compliance Report

**Report ID:** {self.metadata.report_id}  
**Date:** {self.metadata.date}  
**Standard:** {self.metadata.standard}  
**Prepared By:** {self.metadata.prepared_by}

---

## Project Information

| Field | Value |
|-------|-------|
| Project Name | {self.metadata.project_name} |
| Building Name | {self.metadata.building_name} |
| Building Address | {self.metadata.building_address} |
| Building Type | {self.metadata.building_type} |
| Climate Zone | {self.metadata.climate_zone} |
| Building Area | {self.metadata.building_area_m2:,.1f} m² |
"""
    
    def _generate_executive_summary(
        self,
        proposed: ComplianceMetrics,
        baseline: Optional[ComplianceMetrics],
    ) -> str:
        """Generate executive summary section."""
        lines = ["## Executive Summary\n"]
        
        if baseline:
            # Calculate improvements
            energy_reduction = (
                (baseline.total_energy_kwh - proposed.total_energy_kwh) /
                baseline.total_energy_kwh * 100
            ) if baseline.total_energy_kwh > 0 else 0
            
            cost_savings = baseline.annual_energy_cost_usd - proposed.annual_energy_cost_usd
            unmet_hours = proposed.total_unmet_hours
            
            # Determine compliance status
            compliant = energy_reduction >= 50.0 and unmet_hours <= 300
            status = "✅ COMPLIANT" if compliant else "❌ NON-COMPLIANT"
            
            lines.append(f"""This report evaluates the proposed building design against the {self.standard.value} 
Appendix G baseline for code compliance.

### Compliance Determination: **{status}**

| Metric | Baseline | Proposed | Change |
|--------|----------|----------|--------|
| Annual Energy (kWh) | {baseline.total_energy_kwh:,.0f} | {proposed.total_energy_kwh:,.0f} | -{energy_reduction:.1f}% |
| EUI (kWh/m²/year) | {baseline.total_eui_kwh_m2:.2f} | {proposed.total_eui_kwh_m2:.2f} | - |
| Annual Cost ($) | ${baseline.annual_energy_cost_usd:,.0f} | ${proposed.annual_energy_cost_usd:,.0f} | ${cost_savings:,.0f} saved |
| Unmet Hours | {baseline.total_unmet_hours:.0f} | {proposed.total_unmet_hours:.0f} | - |

### Key Findings
- The proposed design achieves **{energy_reduction:.1f}%** energy reduction compared to baseline
- Annual energy cost savings: **${cost_savings:,.0f}**
- Total unmet hours: **{unmet_hours:.0f}** (requirement: ≤300 hours)
""")
        else:
            lines.append(f"""This report presents the energy performance analysis for {proposed.building_name}.

| Metric | Value |
|--------|-------|
| Annual Energy | {proposed.total_energy_kwh:,.0f} kWh |
| EUI | {proposed.total_eui_kwh_m2:.2f} kWh/m²/year |
| Annual Cost | ${proposed.annual_energy_cost_usd:,.0f} |
| Peak Heating | {proposed.peak_heating_load_kw:.1f} kW |
| Peak Cooling | {proposed.peak_cooling_load_kw:.1f} kW |
| Unmet Hours | {proposed.total_unmet_hours:.0f} |
""")
        
        return "".join(lines)
    
    def _generate_building_description(self) -> str:
        """Generate building description section."""
        return f"""## Building Description

### General Information
- **Building Name:** {self.metadata.building_name}
- **Building Type:** {self.metadata.building_type}
- **Climate Zone:** {self.metadata.climate_zone} (ASHRAE/ IECC)
- **Conditioned Floor Area:** {self.metadata.building_area_m2:,.1f} m²

### Building Characteristics
- **Heating Setpoint:** 21°C (70°F)
- **Cooling Setpoint:** 24°C (75°F)
- **Ventilation:** Per ASHRAE 62.1
- **Lighting:** Per ASHRAE 90.1

### Simulation Parameters
- **Simulation Period:** 1 year (8,760 hours)
- **Timestep:** 1 hour
- **Weather Data:** Typical Meteorological Year (TMY3)
"""
    
    def _generate_energy_table(
        self,
        proposed: ComplianceMetrics,
        baseline: ComplianceMetrics,
    ) -> str:
        """Generate the ASHRAE 90.1 Appendix G energy table."""
        energy_reduction = (
            (baseline.total_energy_kwh - proposed.total_energy_kwh) /
            baseline.total_energy_kwh * 100
        ) if baseline.total_energy_kwh > 0 else 0
        
        return f"""## Energy Analysis Summary

### Appendix G Performance Comparison Table

| Metric | Baseline | Proposed | % Change | Notes |
|--------|----------|----------|----------|-------|
| **Total Energy Consumption** | | | | |
| Annual Electricity (kWh) | {baseline.total_energy_kwh:,.0f} | {proposed.total_energy_kwh:,.0f} | -{energy_reduction:.1f}% | |
| EUI (kWh/m²/year) | {baseline.total_eui_kwh_m2:.2f} | {proposed.total_eui_kwh_m2:.2f} | - | |
| **Peak Loads** | | | | |
| Peak Heating Demand (kW) | {baseline.peak_heating_load_kw:.1f} | {proposed.peak_heating_load_kw:.1f} | - | |
| Peak Cooling Demand (kW) | {baseline.peak_cooling_load_kw:.1f} | {proposed.peak_cooling_load_kw:.1f} | - | |
| **Annual Energy Cost** | | | | |
| Electricity ($) | ${baseline.annual_energy_cost_usd:,.0f} | ${proposed.annual_energy_cost_usd:,.0f} | - | |
| **Thermal Comfort** | | | | |
| Unmet Heating Hours | {baseline.unmet_heating_hours:.0f} | {proposed.unmet_heating_hours:.0f} | - | ≤300 |
| Unmet Cooling Hours | {baseline.unmet_cooling_hours:.0f} | {proposed.unmet_cooling_hours:.0f} | - | ≤300 |
| Total Unmet Hours | {baseline.total_unmet_hours:.0f} | {proposed.total_unmet_hours:.0f} | - | ≤300 |

### End-Use Energy Breakdown (Proposed)

| End Use | Energy (kWh) | Percentage |
|---------|--------------|------------|
| Heating | {proposed.heating_energy_kwh:,.0f} | {proposed.heating_energy_kwh/proposed.total_energy_kwh*100:.1f}% |
| Cooling | {proposed.cooling_energy_kwh:,.0f} | {proposed.cooling_energy_kwh/proposed.total_energy_kwh*100:.1f}% |
| Lighting | {proposed.lighting_energy_kwh:,.0f} | {proposed.lighting_energy_kwh/proposed.total_energy_kwh*100 if proposed.total_energy_kwh > 0 else 0:.1f}% |
| Plug Loads | {proposed.plug_loads_kwh:,.0f} | {proposed.plug_loads_kwh/proposed.total_energy_kwh*100 if proposed.total_energy_kwh > 0 else 0:.1f}% |
| Ventilation | {proposed.ventilation_energy_kwh:,.0f} | {proposed.ventilation_energy_kwh/proposed.total_energy_kwh*100 if proposed.total_energy_kwh > 0 else 0:.1f}% |
| **Total** | **{proposed.total_energy_kwh:,.0f}** | **100%** |
"""
    
    def _generate_detailed_metrics(self, metrics: ComplianceMetrics) -> str:
        """Generate detailed metrics section."""
        return f"""## Detailed Performance Metrics

### Energy Use Intensity (EUI) Analysis
- **Total EUI:** {metrics.total_eui_kwh_m2:.2f} kWh/m²/year
- **Electricity EUI:** {metrics.electricity_eui_kwh_m2:.2f} kWh/m²/year

### Peak Demand Analysis
- **Peak Heating Load:** {metrics.peak_heating_load_kw:.1f} kW
- **Peak Cooling Load:** {metrics.peak_cooling_load_kw:.1f} kW  
- **Peak Electric Demand:** {metrics.peak_electric_demand_kw:.1f} kW

### HVAC System Performance
- **Heating COP:** {metrics.heating_cop:.1f}
- **Cooling COP:** {metrics.cooling_cop:.1f}

### Thermal Comfort Summary
- **Unmet Heating Hours:** {metrics.unmet_heating_hours:.0f} hours/year
- **Unmet Cooling Hours:** {metrics.unmet_cooling_hours:.0f} hours/year
- **Total Unmet Hours:** {metrics.total_unmet_hours:.0f} hours/year

*Note: ASHRAE 90.1 typically requires total unmet hours ≤300 per year*
"""
    
    def _generate_compliance_determination(
        self,
        proposed: ComplianceMetrics,
        baseline: ComplianceMetrics,
    ) -> str:
        """Generate compliance determination section."""
        energy_reduction = (
            (baseline.total_energy_kwh - proposed.total_energy_kwh) /
            baseline.total_energy_kwh * 100
        ) if baseline.total_energy_kwh > 0 else 0
        
        cost_savings = baseline.annual_energy_cost_usd - proposed.annual_energy_cost_usd
        
        meets_eui = energy_reduction >= 50.0
        meets_unmet = proposed.total_unmet_hours <= 300
        meets_peak_heating = proposed.peak_heating_load_kw <= baseline.peak_heating_load_kw
        meets_peak_cooling = proposed.peak_cooling_load_kw <= baseline.peak_cooling_load_kw
        
        compliant = meets_eui and meets_unmet and meets_peak_heating and meets_peak_cooling
        
        status = "✅ COMPLIANT" if compliant else "❌ NON-COMPLIANT"
        
        return f"""## Compliance Determination

### {self.standard.value} Appendix G Performance Rating Method

| Requirement | Threshold | Proposed | Status |
|-------------|-----------|----------|--------|
| Energy Cost Improvement | ≥50% | {energy_reduction:.1f}% | {"✅ PASS" if meets_eui else "❌ FAIL"} |
| Unmet Hours | ≤300 | {proposed.total_unmet_hours:.0f} | {"✅ PASS" if meets_unmet else "❌ FAIL"} |
| Peak Heating | ≤Baseline | {proposed.peak_heating_load_kw:.1f} kW | {"✅ PASS" if meets_peak_heating else "❌ FAIL"} |
| Peak Cooling | ≤Baseline | {proposed.peak_cooling_load_kw:.1f} kW | {"✅ PASS" if meets_peak_cooling else "❌ FAIL"} |

### Overall Determination: **{status}**

{"The proposed building design meets all performance requirements of ASHRAE 90.1-2019 Appendix G and is approved for code compliance." if compliant else "The proposed building design does not meet one or more performance requirements. See above for specific items requiring attention."}

### Summary
- Energy Cost Improvement: **{energy_reduction:.1f}%** (Required: ≥50%)
- Annual Energy Cost Savings: **${cost_savings:,.0f}**
- Unmet Hours: **{proposed.total_unmet_hours:.0f}** (Required: ≤300)
"""
    
    def _generate_appendix(self) -> str:
        """Generate appendix section."""
        return f"""## Appendix: Simulation Methodology

### Software
- **Simulation Engine:** Fluxion (Rust-based BEM engine)
- **AI Surrogates:** Optional ONNX-based neural network acceleration
- **Validation:** ASHRAE 140 compliant

### Input Parameters
- Building envelope properties (U-values, SHGC, WWR)
- HVAC system efficiency (COP, SEER)
- Internal loads (lighting, plug loads, occupancy)
- Weather data (TMY3 for specified climate zone)

### Output Metrics
- Annual energy consumption (kWh)
- Energy Use Intensity (kWh/m²/year)
- Peak heating and cooling loads (kW)
- Unmet hours (hours/year)

---
*Report generated by Fluxion Automated Compliance Agent*  
*For technical questions, contact the building energy consultant*
"""
    
    def save_report(self, filepath: Path) -> None:
        """Save the generated report to a file."""
        content = self.generate_report(self.metadata)
        filepath.write_text(content)
        print(f"Report saved to: {filepath}")


def generate_compliance_report(
    proposed_metrics: ComplianceMetrics,
    baseline_metrics: Optional[ComplianceMetrics] = None,
    project_name: str = "Unnamed Project",
    building_name: str = "Unnamed Building",
    building_address: str = "Not Specified",
    standard: ComplianceStandard = ComplianceStandard.ASHRAE_90_1_2019,
) -> str:
    """
    Convenience function to generate a compliance report.
    
    Args:
        proposed_metrics: Proposed building metrics
        baseline_metrics: Optional baseline metrics
        project_name: Name of the project
        building_name: Name of the building
        building_address: Address of the building
        standard: Compliance standard to use
    
    Returns:
        Markdown-formatted compliance report
    """
    import uuid
    
    metadata = ReportMetadata(
        report_id=str(uuid.uuid4())[:8].upper(),
        project_name=project_name,
        building_name=building_name,
        building_address=building_address,
        building_type=proposed_metrics.building_type,
        climate_zone=proposed_metrics.climate_zone,
        building_area_m2=proposed_metrics.building_area_m2,
        standard=standard.value,
    )
    
    generator = ComplianceReportGenerator(metadata=metadata, standard=standard)
    return generator.generate_report(proposed_metrics, baseline_metrics)


if __name__ == "__main__":
    # Demo
    from api.compliance.data_aggregation import create_sample_metrics
    
    proposed = create_sample_metrics()
    proposed.building_name = "Demo Office Building"
    proposed.building_type = "Commercial"
    proposed.climate_zone = "4A"
    
    # Create a slightly worse baseline for demo
    baseline = create_sample_metrics()
    baseline.total_energy_kwh = proposed.total_energy_kwh * 1.5
    baseline.total_eui_kwh_m2 = proposed.total_eui_kwh_m2 * 1.5
    baseline.annual_energy_cost_usd = proposed.annual_energy_cost_usd * 1.5
    
    # Generate report
    report = generate_compliance_report(
        proposed_metrics=proposed,
        baseline_metrics=baseline,
        project_name="Demo Project",
        building_name="Demo Office Building",
        building_address="123 Main St, City, State",
    )
    
    print(report)
