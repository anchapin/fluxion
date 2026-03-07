# ASHRAE 90.1 / IECC Compliance Agent Specification

## Overview

This document specifies the design and implementation of an LLM-powered compliance agent for Fluxion that autonomously generates human-readable compliance reports for ASHRAE 90.1 Appendix G and IECC standards.

## Issue Reference

- **Issue**: #451 - [Feature] Automated Code Compliance Agent (ASHRAE 90.1 / IECC) via LLM
- **Labels**: documentation, llm-integration, phase-10
- **Repository**: anchapin/fluxion

---

## 1. Data Aggregation Requirements

### 1.1 Input Data Sources

The compliance agent consumes time-series output tensors from Fluxion simulations:

| Data Source | Tensor Shape | Unit | Description |
|-------------|--------------|------|-------------|
| Zone Temperatures | (8760, num_zones) | °C | Indoor zone temperatures for each hour |
| Heating Loads | (8760, num_zones) | W | Heating energy demand |
| Cooling Loads | (8760, num_zones) | W | Cooling energy demand |
| Lighting Loads | (8760, num_zones) | W | Lighting energy consumption |
| Plug Loads | (8760, num_zones) | W | Plug/equipment loads |
| HVAC Energy | (8760, num_zones) | kWh | HVAC system energy consumption |

### 1.2 Aggregate Metrics

The data aggregator computes the following annual metrics:

#### Peak Loads
- **Peak Heating Load** (kW): Maximum hourly heating demand
- **Peak Cooling Load** (kW): Maximum hourly cooling demand  
- **Peak Electric Demand** (kW): Maximum hourly electricity demand

#### Annual Energy Consumption
- **Total Electricity** (kWh): Annual electricity consumption
- **Total Natural Gas** (kWh): Annual gas consumption
- **Total Energy** (kWh): Sum of all energy sources
- **EUI** (kWh/m²/year): Energy Use Intensity normalized by floor area

#### End-Use Breakdown
- Heating Energy (kWh)
- Cooling Energy (kWh)
- Lighting Energy (kWh)
- Plug Loads (kWh)
- Ventilation Energy (kWh)

#### Thermal Comfort
- **Unmet Heating Hours**: Hours where zone temp < heating setpoint - tolerance
- **Unmet Cooling Hours**: Hours where zone temp > cooling setpoint + tolerance
- **Total Unmet Hours**: Sum of unmet heating and cooling hours

### 1.3 Data Extraction Interface

```python
class SimulationDataExtractor:
    """Extracts and processes simulation outputs for compliance."""
    
    def extract_hourly_temperatures(simulation_output) -> List[float]:
        """Extract zone temperature time series."""
        
    def extract_hourly_loads(simulation_output, load_type: str) -> List[float]:
        """Extract specific load type time series."""
        
    def compute_annual_metrics(simulation_output) -> ComplianceMetrics:
        """Compute all compliance metrics from simulation."""
```

---

## 2. Prompt Engineering Template System

### 2.1 Template Architecture

The prompt engineering system uses a hierarchical template structure:

```
compliance_prompts/
├── system_prompts/
│   ├── ashrae_90_1_2019.txt
│   ├── ashrae_90_1_2022.txt
│   ├── iecc_2021.txt
│   └── iecc_2024.txt
├── user_prompts/
│   ├── performance_report.txt
│   ├── compliance_summary.txt
│   └── detailed_analysis.txt
└── templates.yaml
```

### 2.2 System Prompt Template

```system
You are an expert building energy analyst specializing in ASHRAE 90.1 Appendix G 
compliance determinations. Your role is to analyze building energy simulation 
results and generate professional compliance reports.

## Your Expertise
- ASHRAE 90.1-2019 and 2022 Energy Standard for Buildings
- IECC 2021 and 2024 International Energy Conservation Code
- Building energy modeling and simulation
- Performance rating method (Appendix G)
- HVAC, lighting, and envelope requirements

## Your Task
Given the building energy simulation metrics provided, generate a comprehensive 
compliance report that:
1. Summarizes the building's energy performance
2. Compares proposed design to baseline (Appendix G)
3. Provides compliance determination with supporting analysis
4. Identifies areas of strength and improvement opportunities

## Output Format
Generate a professional Markdown report with:
- Executive Summary
- Building Description
- Performance Comparison Tables
- Compliance Determination
- Detailed Analysis
```

### 2.3 User Prompt Template

```user
## Building Information
- Building Name: {building_name}
- Building Type: {building_type}
- Climate Zone: {climate_zone}
- Floor Area: {building_area_m2:.1f} m² ({building_area_ft2:.1f} ft²)

## Proposed Design Annual Metrics
- Total Energy: {total_energy_kwh:.0f} kWh
- Total EUI: {total_eui_kwh_m2:.1f} kWh/m²/year
- Peak Heating: {peak_heating_kw:.1f} kW
- Peak Cooling: {peak_cooling_kw:.1f} kW
- Unmet Hours: {unmet_hours:.0f} hours/year

## Baseline Design Annual Metrics (Appendix G)
- Total Energy: {baseline_energy_kwh:.0f} kWh
- Total EUI: {baseline_eui_kwh_m2:.1f} kWh/m²/year
- Peak Heating: {baseline_heating_kw:.1f} kW
- Peak Cooling: {baseline_cooling_kw:.1f} kW
- Unmet Hours: {baseline_unmet_hours:.0f} hours/year

## Performance Improvement
- Energy Cost Improvement: {cost_improvement:.1f}%
- Annual Energy Cost Savings: ${annual_savings:,.0f}

## Compliance Requirements (ASHRAE 90.1-2019)
| Requirement | Threshold | Proposed | Status |
|-------------|-----------|----------|--------|
| Energy Cost Improvement | ≥50% | {cost_improvement:.1f}% | {cost_status} |
| Unmet Hours | ≤300 hours | {unmet_hours:.0f} hours | {unmet_status} |
| Peak Heating | ≤Baseline | {heating_ratio:.1%} of baseline | {heating_status} |
| Peak Cooling | ≤Baseline | {cooling_ratio:.1%} of baseline | {cooling_status} |

Generate a comprehensive compliance report in Markdown format.
```

### 2.4 Prompt Variables

| Variable | Type | Description |
|----------|------|-------------|
| `building_name` | string | Building name |
| `building_type` | string | Building type (Commercial, Residential) |
| `climate_zone` | string | ASHRAE climate zone (e.g., 4A) |
| `building_area_m2` | float | Floor area in square meters |
| `total_energy_kwh` | float | Annual total energy in kWh |
| `total_eui_kwh_m2` | float | Energy Use Intensity |
| `peak_heating_kw` | float | Peak heating demand in kW |
| `peak_cooling_kw` | float | Peak cooling demand in kW |
| `unmet_hours` | float | Total unmet hours |
| `cost_improvement` | float | Energy cost improvement percentage |
| `annual_savings` | float | Annual cost savings in USD |

---

## 3. Output Format Specification

### 3.1 Primary Output: Markdown Report

The compliance agent generates Markdown reports conforming to ASHRAE 90.1 Appendix G:

```markdown
# ASHRAE 90.1-2019 Appendix G Compliance Report

## Executive Summary
**COMPLIANT** / **NON-COMPLIANT**

The proposed design achieves {cost_improvement:.1f}% energy cost improvement
over the baseline building, {meeting_or_not} the ASHRAE 90.1-2019 requirement
of ≥50% improvement.

## Building Description
| Attribute | Value |
|-----------|-------|
| Building Name | {building_name} |
| Building Type | {building_type} |
| Climate Zone | {climate_zone} |
| Floor Area | {area_m2} m² |

## Appendix G Performance Comparison Table

| Metric | Baseline | Proposed | % of Baseline |
|--------|----------|----------|---------------|
| Energy Cost ($) | ${baseline_cost:,.0f} | ${proposed_cost:,.0f} | {cost_pct:.1f}% |
| Source Energy (GJ) | {baseline_src:.0f} | {proposed_src:.0f} | {src_pct:.1f}% |
| Peak Heating (kW) | {baseline_heat:.1f} | {proposed_heat:.1f} | {heat_pct:.1f}% |
| Peak Cooling (kW) | {baseline_cool:.1f} | {proposed_cool:.1f} | {cool_pct:.1f}% |
| Unmet Hours | {baseline_unmet:.0f} | {proposed_unmet:.0f} | - |

## Compliance Determination

### Requirements Checklist
- [x] Energy Cost Improvement ≥ 50%
- [x] Unmet Hours ≤ 300 hours/year
- [x] Peak Heating ≤ Baseline
- [x] Peak Cooling ≤ Baseline

**Final Determination: COMPLIANT**

## Detailed Performance Analysis
[LLM-generated analysis based on metrics]
```

### 3.2 Secondary Output: PDF Generation

For formal submissions, Markdown can be converted to PDF using:

```python
def generate_pdf_report(markdown_content: str, output_path: str):
    """Convert Markdown report to PDF."""
    # Using pandoc or markdown-pdf
    pass
```

### 3.3 Data Export Formats

| Format | Use Case |
|--------|----------|
| JSON | API responses, data interchange |
| CSV | Spreadsheet analysis |
| XML | Building energy model interchange |
| PDF | Formal submissions |

---

## 4. Module Architecture

### 4.1 Existing Implementation (api/compliance/)

The compliance agent is already implemented in `api/compliance/`:

```
api/compliance/
├── __init__.py          # Module exports
├── agent.py             # ComplianceAgent orchestration
├── data_aggregation.py  # ComplianceMetrics & aggregation
├── prompt_engine.py     # Prompt template system
└── report_generator.py  # Markdown report generation
```

### 4.2 Tools Module Extension

A new module in `tools/` provides CLI and standalone capabilities:

```
tools/
├── compliance_agent.py   # CLI tool for compliance reporting
└── extract_simulation.py # Extract data from Fluxion outputs
```

### 4.3 Integration with LLM API

The compliance agent integrates with the existing LLM API (`api/llm.py`):

```python
from api.llm import query_with_function_calling
from api.compliance import ComplianceAgent, ComplianceMetrics

def generate_llm_compliance_report(
    proposed_metrics: ComplianceMetrics,
    baseline_metrics: ComplianceMetrics,
    llm_model_path: str = None
) -> str:
    """Generate natural language compliance report using LLM."""
    
    # Create compliance agent
    agent = ComplianceAgent()
    
    # Get structured prompt
    prompt = agent.generate_prompt(proposed_metrics, baseline_metrics)
    
    # Query LLM
    response = query_with_function_calling(
        query=prompt.user_prompt_template,
        model_path=llm_model_path,
    )
    
    return response["response"]
```

---

## 5. Compliance Requirements Summary

### 5.1 ASHRAE 90.1-2019 Appendix G

| Requirement | Threshold | Description |
|-------------|-----------|-------------|
| Energy Cost Improvement | ≥50% | Performance rating method |
| Unmet Hours | ≤300 hours/year | Total unmet hours |
| Peak Heating | ≤Baseline | Peak heating ≤ baseline |
| Peak Cooling | ≤Baseline | Peak cooling ≤ baseline |

### 5.2 ASHRAE 90.1-2022

| Requirement | Threshold | Description |
|-------------|-----------|-------------|
| Energy Cost Improvement | ≥50% | Performance rating method |
| Total UA | ≤Baseline | Envelope trade-off option |

### 5.3 IECC 2021/2024

| Requirement | Threshold | Description |
|-------------|-----------|-------------|
| Energy Cost Improvement | ≥50% | Performance approach |
| Unmet Hours | ≤400 hours/year | Total unmet hours |

---

## 6. API Endpoints

### 6.1 REST API (api/main.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/compliance/check` | POST | Check compliance |
| `/compliance/report` | POST | Generate report |
| `/compliance/prompt` | POST | Get LLM prompt |
| `/compliance/standards` | GET | List supported standards |
| `/compliance/sample` | GET | Get sample metrics |

### 6.2 Request/Response Examples

```json
// POST /compliance/check
{
  "building_name": "Office Building A",
  "building_area_m2": 1000,
  "building_type": "Commercial",
  "climate_zone": "4A",
  "hourly_temperatures": [22.0, 21.5, ...],
  "hourly_heating_loads": [50000, 48000, ...],
  "hourly_cooling_loads": [-40000, -42000, ...]
}

// Response
{
  "compliant": true,
  "standard": "ASHRAE 90.1-2019",
  "checks": [
    {"name": "Energy Cost Improvement", "status": "PASS", "actual": "52.3%"},
    {"name": "Unmet Hours", "status": "PASS", "actual": "125 hours"}
  ],
  "summary": "COMPLIANT: Building meets ASHRAE 90.1-2019 requirements"
}
```

---

## 7. Implementation Roadmap

### Phase 1: Core Infrastructure (Completed)
- [x] Compliance data aggregation module
- [x] Prompt engineering template system
- [x] Markdown report generator
- [x] REST API endpoints

### Phase 2: LLM Integration (Completed)
- [x] Integration with api/llm.py
- [x] Function calling support
- [x] Natural language report generation

### Phase 3: CLI Tool (In Progress)
- [ ] tools/compliance_agent.py CLI
- [ ] tools/extract_simulation.py for Fluxion output extraction

### Phase 4: Enhanced Features (Future)
- [ ] PDF generation
- [ ] Multi-language support
- [ ] Interactive compliance dashboard

---

## 8. Testing Requirements

### Unit Tests
- Data aggregation calculations
- Compliance check logic
- Prompt template rendering

### Integration Tests
- End-to-end compliance workflow
- LLM API integration
- REST API endpoints

### Validation
- ASHRAE 140 test case validation
- Sample compliance reports review

---

## 9. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| llama-cpp-python | >=0.2.0 | Local LLM inference |
| fastapi | >=0.100.0 | REST API |
| pydantic | >=2.0 | Data validation |
| jinja2 | >=3.0 | Template rendering |

---

## 10. References

- ASHRAE 90.1-2019: Energy Standard for Buildings
- ASHRAE 90.1-2022: Energy Standard for Buildings
- IECC 2021: International Energy Conservation Code
- IECC 2024: International Energy Conservation Code
- ASHRAE Handbook: Fundamentals (thermal properties)
- ASHRAE Handbook: HVAC Applications (system requirements)
