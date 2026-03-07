# Automated Code Compliance Agent (ASHRAE 90.1 / IECC)

This module provides an LLM-powered agent that checks building energy models for compliance with ASHRAE 90.1 and IECC standards.

## Overview

The Compliance Agent processes building energy simulation results and generates:
- Automated compliance determinations
- Human-readable Markdown reports
- LLM prompts for natural language report generation

## Architecture

The module consists of four main components:

1. **Data Aggregation** (`data_aggregation.py`) - Extracts and computes compliance metrics from simulation outputs
2. **Prompt Engineering** (`prompt_engine.py`) - Creates templated prompts for LLM integration
3. **Report Generation** (`report_generator.py`) - Generates standardized Markdown compliance reports
4. **Compliance Agent** (`agent.py`) - Orchestrates the compliance workflow

## Quick Start

### Python API

```python
from api.compliance import (
    ComplianceAgent,
    ComplianceDataAggregator,
    ComplianceMetrics,
    create_sample_metrics,
    create_prompt_for_llm,
    generate_compliance_report,
)

# Option 1: Process hourly simulation data
aggregator = ComplianceDataAggregator(building_area_m2=1000)
metrics = aggregator.process_simulation_results(
    hourly_temperatures=temperature_list,  # 8760 values
    hourly_heating_loads=heating_list,      # 8760 values (W)
    hourly_cooling_loads=cooling_list,      # 8760 values (W)
)

# Option 2: Use sample metrics for testing
metrics = create_sample_metrics()

# Create compliance agent
agent = ComplianceAgent()

# Check compliance
result = agent.check_compliance(proposed_metrics=metrics, baseline_metrics=baseline)

# Generate Markdown report
report = generate_compliance_report(
    proposed_metrics=metrics,
    baseline_metrics=baseline,
    project_name="Project Name",
    building_name="Building Name",
)

# Get LLM prompt for natural language generation
prompt = create_prompt_for_llm(metrics, baseline)
# Send prompt.system_prompt and prompt.user_prompt_template to your LLM
```

### REST API Endpoints

Start the API server:

```bash
cd api
uvicorn main:app --reload
```

Then use these endpoints:

#### Check Compliance
```bash
POST /compliance/check
```

Request:
```json
{
  "building_name": "Office Building",
  "building_area_m2": 1000,
  "building_type": "Commercial",
  "climate_zone": "4A",
  "hourly_temperatures": [22.0, 21.5, ...],  // 8760 values
  "hourly_heating_loads": [50000, 48000, ...],  // 8760 values (W)
  "hourly_cooling_loads": [-40000, -42000, ...]  // 8760 values (W)
}
```

Response:
```json
{
  "compliant": true,
  "standard": "ASHRAE 90.1-2019",
  "checks": [
    {"name": "Energy Cost Improvement", "status": "PASS", "actual": "50.0%"},
    {"name": "Unmet Hours", "status": "PASS", "actual": "0 hours"}
  ],
  "summary": "COMPLIANT: Building meets ASHRAE 90.1-2019 requirements..."
}
```

#### Generate Report
```bash
POST /compliance/report
```

#### Get LLM Prompt
```bash
POST /compliance/prompt
```

#### List Standards
```bash
GET /compliance/standards
```

#### Get Sample Metrics
```bash
GET /compliance/sample
```

## Supported Standards

- **ASHRAE 90.1-2019** - Energy Standard for Buildings
- **ASHRAE 90.1-2022** - Energy Standard for Buildings  
- **IECC 2021** - International Energy Conservation Code
- **IECC 2024** - International Energy Conservation Code

## Compliance Requirements

### ASHRAE 90.1 Appendix G

| Requirement | Threshold |
|-------------|-----------|
| Energy Cost Improvement | ≥50% vs. baseline |
| Unmet Hours | ≤300 hours/year |
| Peak Heating | ≤Baseline |
| Peak Cooling | ≤Baseline |

### IECC

| Requirement | Threshold |
|-------------|-----------|
| Energy Cost Improvement | ≥50% vs. baseline |
| Unmet Hours | ≤400 hours/year |

## Integration with LLM

For natural language report generation, use the `/compliance/prompt` endpoint to get structured prompts:

```python
# Get prompts
prompt = create_prompt_for_llm(metrics, baseline)

# Send to your LLM
response = llm.chat(
    system=prompt.system_prompt,
    user=prompt.user_prompt_template
)

# The LLM will generate a professional compliance report
```

## Data Requirements

### Hourly Data (8760 values = 1 year)

| Parameter | Unit | Description |
|-----------|------|-------------|
| `hourly_temperatures` | °C | Indoor zone temperatures |
| `hourly_heating_loads` | W | Heating loads (positive) |
| `hourly_cooling_loads` | W | Cooling loads (negative) |
| `hourly_lighting` | W | Lighting loads (optional) |
| `hourly_plug_loads` | W | Plug loads (optional) |

### Annual Metrics

| Parameter | Unit | Description |
|-----------|------|-------------|
| `total_energy_kwh` | kWh | Annual total energy |
| `total_eui_kwh_m2` | kWh/m²/year | Energy Use Intensity |
| `peak_heating_kw` | kW | Peak heating demand |
| `peak_cooling_kw` | kW | Peak cooling demand |
| `unmet_hours` | hours | Total unmet hours |

## Example: Full Workflow

```python
from api.compliance import (
    ComplianceAgent,
    ComplianceDataAggregator,
    ComplianceStandard,
    create_sample_metrics,
    generate_compliance_report,
)

# 1. Run simulation with Fluxion and get hourly data
# (or use sample data for testing)
proposed_metrics = create_sample_metrics()
proposed_metrics.building_name = "My Office Building"
proposed_metrics.building_type = "Commercial"
proposed_metrics.climate_zone = "4A"

# 2. Create baseline (standard reference building)
baseline = create_sample_metrics()
baseline.total_energy_kwh = proposed_metrics.total_energy_kwh * 2.0
baseline.total_eui_kwh_m2 = proposed_metrics.total_eui_kwh_m2 * 2.0

# 3. Check compliance
agent = ComplianceAgent()
result = agent.check_compliance(proposed_metrics, baseline)

print(f"Compliant: {result['compliant']}")
print(f"Summary: {result['summary']}")

# 4. Generate detailed Markdown report
report = generate_compliance_report(
    proposed_metrics=proposed_metrics,
    baseline_metrics=baseline,
    project_name="My Project",
    building_name="My Office Building",
    building_address="123 Main St, City, State",
    standard=ComplianceStandard.ASHRAE_90_1_2019,
)

# Save report to file
with open("compliance_report.md", "w") as f:
    f.write(report)

print("Report saved to compliance_report.md")
```

## Output Example

The generated Markdown report includes:

- Executive Summary with compliance determination
- Building Description
- ASHRAE 90.1 Appendix G Performance Comparison Table
- Detailed Performance Metrics
- Compliance Determination with pass/fail status
- Methodology Appendix

## License

This module is part of the Fluxion project.
