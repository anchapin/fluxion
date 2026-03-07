# Code Compliance Agent

An automated code compliance agent that uses LLMs to check building energy models against ASHRAE 90.1 and IECC standards.

## Overview

The Code Compliance Agent provides automated compliance checking for building energy modeling projects. It supports:

- **ASHRAE 90.1**: Commercial building energy standard (2019, 2022)
- **IEC**: International Energy Conservation Code (2021, 2024)

The agent can analyze building parameters such as:
- Envelope insulation (R-values)
- Fenestration (U-factors, SHGC)
- HVAC efficiency (COP, SEER, IEER)
- Lighting power density
- Service water heating
- Ventilation rates

## Installation

The compliance agent is included in the Fluxion tools package. Additional dependencies for different LLM backends:

```bash
# For Ollama backend
pip install httpx

# For OpenAI backend
pip install openai
```

## Quick Start

```python
from tools.compliance_agent import CodeComplianceAgent

# Create agent with mock backend (for testing)
agent = CodeComplianceAgent(backend="mock")

# Building model data
model_data = {
    "model_name": "My Office Building",
    "wall_r_value": 15.0,
    "window_u_factor": 0.35,
    "hvac_cop": 3.5,
    "lighting_power_density": 0.8,
}

# Check compliance
report = agent.check_compliance(
    model_data=model_data,
    standard="ASHRAE90.1-2019"
)

# Print summary
print(report.print_summary())

# Save report
agent.save_report(report, "compliance_report.json")
```

## LLM Backends

### Mock Backend (Default)

For testing without external dependencies:

```python
agent = CodeComplianceAgent(backend="mock")
```

### Ollama Backend

For local LLM using Ollama:

```python
agent = CodeComplianceAgent(
    backend="ollama",
    model="llama2"  # or "mistral", "codellama", etc.
)
```

Requires [Ollama](https://ollama.ai/) to be installed and running.

### OpenAI Backend

For cloud-based LLM:

```python
agent = CodeComplianceAgent(
    backend="openai",
    model="gpt-4"  # or "gpt-3.5-turbo"
)
```

Set the `OPENAI_API_KEY` environment variable or pass the key directly:

```python
agent = CodeComplianceAgent(
    backend="openai",
    model="gpt-4",
    api_key="your-api-key"
)
```

## Demo

Run the demo script to see the agent in action:

```bash
# Using mock backend
python tools/compliance_agent/demo.py

# Using Ollama
python tools/compliance_agent/demo.py --backend ollama --model llama2

# Using OpenAI
python tools/compliance_agent/demo.py --backend openai --model gpt-4
```

## Compliance Rules

### ASHRAE 90.1 Rules

| Rule ID | Category | Description |
|---------|----------|-------------|
| 5.1.1 | Envelope | Building envelope thermal resistance |
| 5.1.2 | Envelope | Fenestration U-Factor |
| 5.1.3 | Envelope | Fenestration SHGC |
| 6.1.1 | HVAC | HVAC Equipment Efficiency |
| 6.2.2 | HVAC | HVAC Controls - Temperature |
| 6.3.1 | HVAC | Ventilation Requirements |
| 7.1.1 | Lighting | Lighting Power Density |
| 7.4.2 | Lighting | Lighting Controls |
| 8.1.1 | Service Water Heating | Water Heating Efficiency |
| 9.1.1 | Electric Power | Transformer and Motor Efficiency |

### IECC Rules

| Rule ID | Category | Description |
|---------|----------|-------------|
| R103.2 | Envelope | Envelope Thermal Performance |
| R402.1.3 | Envelope | Fenestration U-Factor |
| R402.1.4 | Envelope | Fenestration SHGC |
| R403.5.1 | HVAC | Mechanical Ventilation |
| R404.1 | Lighting | Lighting Efficiency |

## API Reference

### CodeComplianceAgent

Main class for compliance checking.

```python
CodeComplianceAgent(
    backend: Union[str, LLMBackend] = "mock",
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    **backend_kwargs
)
```

#### Methods

##### check_compliance()

```python
report = agent.check_compliance(
    model_data: Dict[str, Any],
    standard: Union[str, Standard] = Standard.ASHRAE90_1_2019,
    rules: Optional[List[ComplianceRule]] = None,
    use_rules_engine: bool = True
) -> ComplianceReport
```

##### save_report()

```python
agent.save_report(
    report: ComplianceReport,
    output_path: Union[str, Path],
    format: str = "json"  # or "txt"
)
```

##### load_model_data()

```python
model_data = agent.load_model_data(file_path: Union[str, Path]) -> Dict[str, Any]
```

### ComplianceReport

The report object returned from compliance checks.

```python
report.model_name          # Building model name
report.standard            # Standard checked against
report.timestamp           # ISO timestamp
report.overall_status      # COMPLIANT, NON_COMPLIANT, NEEDS_REVIEW
report.checks             # List of individual check results
report.summary            # Summary counts
report.metadata           # LLM and processing metadata

# Methods
report.to_dict()          # Convert to dictionary
report.to_json()          # Convert to JSON string
report.print_summary()    # Get human-readable summary
```

## File Format Support

Load model data from JSON or CSV files:

```python
# JSON file
agent.load_model_data("building_model.json")

# CSV file (uses first row)
agent.load_model_data("building_model.csv")
```

Example JSON format:

```json
{
    "model_name": "Office Building A",
    "wall_r_value": 15.0,
    "roof_r_value": 30.0,
    "window_u_factor": 0.40,
    "window_shgc": 0.25,
    "hvac_cop": 3.5,
    "lighting_power_density": 0.9
}
```
