# Fluxion Validation CLI

## Overview

The Fluxion CLI provides comprehensive commands for building energy model validation, cross-validation, and automation workflows.

## Installation

The CLI is included with the Fluxion binary. After building the project:

```bash
cargo build --release
./target/release/fluxion --help
```

## Validation Commands

### Basic Validation

Validate the engine against ASHRAE Standard 140:

```bash
fluxion validate
```

**Options:**
- `--all`: Run complete validation suite (baseline + diagnostics)
- `--diagnostics`: Run diagnostic cases only
- `--range <RANGE>`: Run specific diagnostic range (e.g., "195-470", "800-810")
- `--case <CASE>`: Run a specific case (e.g., "600")
- `--statistical`: Enable statistical validation (ASHRAE 140 Addendum B)
- `--alpha <VALUE>`: Alpha threshold for statistical FDR correction (default: 0.05)
- `--format <FORMAT>`: Output format (markdown, csv, json, html)
- `--output-file <PATH>`: Save report to file
- `--ci`: Enable CI mode with guardrails and exit codes

**Examples:**

```bash
# Run complete validation suite
fluxion validate --all --format=markdown --output-file=full_report.md

# Run diagnostic cases only
fluxion validate --diagnostics --format=json

# Run specific case range with statistical validation
fluxion validate --range=800-810 --statistical --alpha=0.01

# CI mode with guardrails
fluxion validate --ci
```

### Validate Specific Cases

Validate individual diagnostic cases:

```bash
fluxion validate-case <CASE_SPEC>
```

**Supported cases:**
- Single HVAC equipment cases: 800, 801, 802, ..., 810
- Case ranges: 195-470, 800-810

**Examples:**

```bash
# Validate single HVAC case
fluxion validate-case 800

# Validate case range
fluxion validate-case 800-810
```

## Cross-Validation Commands

### Run Cross-Validation Tests

Execute automated cross-validation workflows:

```bash
fluxion validation cross-validate --test-cases <DIRECTORY> --output <DIRECTORY>
```

**Options:**
- `--test-cases`: Directory containing test case data
- `--output`: Output directory for reports
- `--tolerance`: Temperature tolerance for validation (default: 0.5°C)
- `--verbose`: Enable verbose output
- `--format`: Output format (markdown, json)

**Example:**

```bash
fluxion validation cross-validate \
  --test-cases tests/fixtures/cross_validation \
  --output target/cross_validation_reports \
  --tolerance 0.5 \
  --format markdown \
  --verbose
```

### ESP-r Validation

Validate against ESP-r reference data:

```bash
fluxion validation esp-r --reference <FILE> --fluxion <FILE> --tolerance <VALUE>
```

**Options:**
- `--reference`: Path to ESP-r reference CSV file
- `--fluxion`: Path to Fluxion results CSV file
- `--tolerance`: Temperature tolerance (default: 0.5°C)
- `--output`: Output report file
- `--format`: Output format (markdown, json)

**Example:**

```bash
fluxion validation esp-r \
  --reference refdata/esp_r/case_600.csv \
  --fluxion results/case_600_fluxion.csv \
  --tolerance 0.5 \
  --output esp_r_comparison.md
```

## Automation Commands

### Run Test Automation

Execute automated test workflows:

```bash
fluxion validation automate --test-cases <DIRECTORY> --output <DIRECTORY>
```

**Options:**
- `--test-cases`: Directory containing test cases
- `--output`: Output directory for reports
- `--tolerance`: Temperature tolerance (default: 0.5°C)
- `--verbose`: Enable verbose output
- `--format`: Output format (markdown, json)

**Example:**

```bash
fluxion validation automate \
  --test-cases tests/automation/test_cases \
  --output reports/automation \
  --tolerance 0.5 \
  --format markdown \
  --verbose
```

## Performance Commands

### Run Performance Benchmarks

Execute performance validation benchmarks:

```bash
fluxion benchmark --model <PATH> --runs <COUNT>
```

**Options:**
- `--model`: Path to ONNX model for benchmarking
- `--runs`: Number of inference runs (default: 100)

**Example:**

```bash
fluxion benchmark --model models/thermal_model.onnx --runs 200
```

## Sensitivity Analysis

### Run Sensitivity Analysis

Perform parameter sensitivity analysis:

```bash
fluxion sensitivity --config <YAML_FILE> --output <DIRECTORY>
```

**Configuration YAML format:**
```yaml
case_id: "600"
method: "oat"  # or "sobol", "random"
levels: 10     # for OAT method
samples: 100   # for random method
parameters:
  - name: "wall_u_value"
    min: 0.1
    max: 1.0
  - name: "window_area"
    min: 0.5
    max: 2.0
```

**Options:**
- `--config`: Path to sensitivity configuration YAML
- `--output`: Output directory (default: current directory)
- `--use-surrogates`: Use AI surrogates for faster evaluation

**Example:**

```bash
fluxion sensitivity \
  --config configs/sensitivity_config.yaml \
  --output results/sensitivity \
  --use-surrogates
```

## Component Analysis

### Generate Component Energy Breakdown

Analyze energy components for specific cases:

```bash
fluxion components --case <CASE_ID> --output <CSV_FILE>
```

**Options:**
- `--case`: ASHRAE case ID (e.g., "600", "900FF")
- `--output`: Output CSV file path

**Example:**

```bash
fluxion components --case 600 --output components_600.csv
```

## Swing Analysis

### Calculate Swing Metrics

Analyze free-floating temperature swing metrics:

```bash
fluxion swing --case <CASE_ID> --comfort-min <TEMP> --comfort-max <TEMP>
```

**Options:**
- `--case`: Free-floating case ID (e.g., "600FF", "900FF")
- `--comfort-min`: Comfort band minimum temperature (°C, default: 18.0)
- `--comfort-max`: Comfort band maximum temperature (°C, default: 26.0)

**Example:**

```bash
fluxion swing --case 600FF --comfort-min 20.0 --comfort-max 24.0
```

## Visualization Commands

### Generate Interactive Visualization

Create interactive HTML visualizations from diagnostics CSV:

```bash
fluxion visualize --input <CSV_FILE> --output <HTML_FILE>
```

**Options:**
- `--input`: Input diagnostics CSV file
- `--output`: Output HTML file path

**Example:**

```bash
fluxion visualize --input diagnostics.csv --output visualization.html
```

### Generate Animation

Create animated visualizations:

```bash
fluxion animate --input <CSV_FILE> --output <HTML_FILE>
```

**Options:**
- `--input`: Input diagnostics CSV file
- `--output`: Output HTML file path

**Example:**

```bash
fluxion animate --input diagnostics.csv --output animation.html
```

## Reference Data Management

### Update Reference Data

Fetch and update reference data from configured sources:

```bash
fluxion references update --url <URL>
```

**Options:**
- `--url`: URL to fetch reference data from (optional)

**Example:**

```bash
fluxion references update --url https://example.com/refdata/latest.zip
```

## Delta Testing

### Run Delta Testing

Compare results between different configurations:

```bash
fluxion delta --config <YAML_FILE> --output <DIRECTORY>
```

**Configuration YAML format:**
```yaml
baseline:
  case_id: "600"
  parameters:
    wall_u_value: 0.35
    window_area: 1.2
variant:
  case_id: "600"
  parameters:
    wall_u_value: 0.25
    window_area: 1.2
tolerance: 0.05
hourly: true
```

**Options:**
- `--config`: Path to delta configuration YAML
- `--output`: Output directory
- `--hourly`: Include hourly differences in output

**Example:**

```bash
fluxion delta \
  --config configs/delta_config.yaml \
  --output results/delta \
  --hourly
```

## Error Handling and Exit Codes

The CLI uses standard exit codes:
- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: Validation failed (CI mode)

## Environment Variables

- `RUST_LOG`: Set logging level (e.g., `RUST_LOG=debug`)
- `FLUXION_DATA_DIR`: Custom data directory path
- `CI`: Set to "true" to enable CI mode automatically

## Troubleshooting

### Common Issues

**"Case not found" errors:**
- Ensure the case ID is correct and supported
- Check that reference data is available

**Validation failures:**
- Review tolerance settings
- Check input parameters
- Examine detailed validation reports

**Performance issues:**
- Reduce number of parallel tests
- Use release builds (`cargo build --release`)
- Consider using AI surrogates for sensitivity analysis

### Getting Help

For additional help and examples, use:

```bash
fluxion --help
fluxion <command> --help
```

## Examples Directory

The `examples/` directory contains sample configuration files:
- `sensitivity_config.yaml`: Sensitivity analysis configuration
- `delta_config.yaml`: Delta testing configuration
- `validation_workflow.sh`: Complete validation workflow script

## Best Practices

1. **Start with simple cases**: Begin validation with basic cases before complex scenarios
2. **Use statistical validation**: For comprehensive analysis, enable `--statistical` flag
3. **Review reports thoroughly**: Validation reports contain detailed diagnostics
4. **Leverage automation**: Use automation commands for repetitive testing
5. **Monitor performance**: Regularly run benchmarks to track performance
6. **Document configurations**: Keep records of validation parameters and settings
