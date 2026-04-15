# ESP-r Cross-Validation

This document provides comprehensive documentation for the ESP-r cross-validation functionality in Fluxion. Cross-validation allows you to compare Fluxion simulation results against ESP-r reference data to ensure accuracy and compliance with building energy modeling standards.

## Table of Contents

- [Overview](#overview)
- [Setup and Installation](#setup-and-installation)
- [Basic Usage](#basic-usage)
- [Advanced Configuration](#advanced-configuration)
- [CLI Commands](#cli-commands)
- [API Usage](#api-usage)
- [Examples](#examples)
- [Integration](#integration)
- [Reporting Options](#reporting-options)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Configuration Options](#configuration-options)
- [Error Codes](#error-codes)

## Overview

### Purpose and Benefits

ESP-r cross-validation in Fluxion provides several key benefits:

- **Accuracy Verification**: Ensure Fluxion results match established ESP-r reference data
- **Compliance**: Demonstrate compliance with ASHRAE 140 and other standards
- **Quality Assurance**: Automated validation workflows for CI/CD pipelines
- **Performance Benchmarking**: Compare computational efficiency while maintaining accuracy

### Key Features

- Configurable tolerance bands for temperature comparisons
- Multi-zone validation support
- Multiple output formats (JSON, Markdown)
- Detailed statistical reporting
- Integration with existing validation frameworks
- Command-line and programmatic interfaces

### Use Cases

1. **Model Development**: Validate new physics models against ESP-r references
2. **Regression Testing**: Ensure code changes don't introduce accuracy regressions
3. **Compliance Reporting**: Generate validation reports for certification
4. **Research**: Compare different simulation approaches
5. **Education**: Teaching tool for building energy modeling

## Setup and Installation

### Prerequisites

- Rust 1.70+ (recommended: latest stable version)
- Cargo package manager
- ESP-r reference data files in CSV format
- Fluxion validation results in JSON format

### Installation

The ESP-r cross-validation functionality is included in the main Fluxion crate. No additional installation is required beyond the standard Fluxion setup:

```bash
# Clone the Fluxion repository
git clone https://github.com/anchapin/fluxion.git
cd fluxion

# Build the project
cargo build --release
```

### Configuration

Add the following to your `Cargo.toml` to use the validation features:

```toml
[dependencies]
fluxion = { git = "https://github.com/anchapin/fluxion.git" }
serde_json = "1.0"
```

## Basic Usage

### Simple Validation

The simplest way to run cross-validation is using the `EspRValidator` struct:

```rust
use fluxion::validation::esp_r::EspRValidator;
use fluxion::validation::MultiZoneValidationResults;
use std::path::PathBuf;

// Create validator with default tolerance (0.5°C)
let validator = EspRValidator::new(
    PathBuf::from("path/to/esp_r_reference.csv"),
    0.5
);

// Create Fluxion results (typically loaded from simulation)
let mut fluxion_results = MultiZoneValidationResults::default();
fluxion_results.add_zone_result("Zone1".to_string(), vec![22.1, 22.3, 22.2]);
fluxion_results.add_zone_result("Zone2".to_string(), vec![21.8, 21.9, 22.0]);

// Run validation
let report = validator.validate(&fluxion_results)?;

println!("Overall pass status: {}", report.overall_pass);
println!("Average temperature difference: {:.2}°C", report.average_temperature_difference);
```

### Command Line Interface

Run cross-validation from the command line:

```bash
# Basic validation
cargo run --example cross_validation_example -- \
    --esp-r path/to/esp_r_reference.csv \
    --tolerance 0.5 \
    --format json

# With Fluxion results file
cargo run --example cross_validation_example -- \
    --esp-r path/to/esp_r_reference.csv \
    --fluxion path/to/fluxion_results.json \
    --tolerance 0.25 \
    --format markdown
```

## Advanced Configuration

### Custom Tolerance Settings

Adjust the tolerance based on your validation requirements:

```rust
// High precision validation (0.25°C tolerance)
let validator = EspRValidator::new(PathBuf::from("reference.csv"), 0.25);

// Loose validation for early development (1.0°C tolerance)
let validator = EspRValidator::new(PathBuf::from("reference.csv"), 1.0);
```

### Multiple Zone Validation

Validate multiple zones with comprehensive reporting:

```rust
let mut fluxion_results = MultiZoneValidationResults::default();

// Add results for multiple zones
fluxion_results.add_zone_result("LivingRoom".to_string(), vec![
    20.5, 20.7, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5
]);

fluxion_results.add_zone_result("Bedroom".to_string(), vec![
    19.0, 19.2, 19.5, 19.8, 20.0, 20.2, 20.5, 20.8, 21.0, 21.2
]);

let report = validator.validate(&fluxion_results)?;

// Access per-zone results
for (zone_name, zone_result) in &report.zone_results {
    println!("Zone {}: avg_diff={:.2}°C, max_diff={:.2}°C",
             zone_name, zone_result.average_difference, zone_result.max_difference);
}
```

### Error Handling

Proper error handling for robust validation workflows:

```rust
match validator.validate(&fluxion_results) {
    Ok(report) => {
        println!("Validation successful!");
        // Process report
    },
    Err(e) => {
        eprintln!("Validation failed: {}", e);
        // Handle error appropriately
    }
}
```

## CLI Commands

### Available Commands

```bash
# Show help
cargo run --example cross_validation_example -- --help

# Basic validation with required arguments
cargo run --example cross_validation_example -- \
    --esp-r <path_to_esp_r_csv> \
    [--fluxion <path_to_fluxion_json>] \
    [--tolerance <tolerance_in_degrees>] \
    [--format <json|markdown>]
```

### Command Line Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--esp-r` | Path to ESP-r reference CSV file | - | Yes |
| `--fluxion` | Path to Fluxion validation results JSON file | - | No |
| `--tolerance` | Temperature tolerance for comparison (°C) | 0.5 | No |
| `--format` | Output format (json or markdown) | json | No |

### Example Workflows

**Basic validation with sample data:**
```bash
cargo run --example cross_validation_example -- \
    --esp-r examples/reference_data/esp_r_basic.csv
```

**Validation with custom tolerance and Markdown output:**
```bash
cargo run --example cross_validation_example -- \
    --esp-r path/to/reference.csv \
    --fluxion path/to/results.json \
    --tolerance 0.3 \
    --format markdown
```

## API Usage

### Programmatic Validation

Integrate cross-validation into your Rust applications:

```rust
use fluxion::validation::esp_r::{EspRValidator, EspRTestConfig, ReportFormat};

// Create test configuration
let config = EspRTestConfig {
    esp_r_output_path: PathBuf::from("reference.csv"),
    fluxion_results_path: PathBuf::from("results.json"),
    tolerance: 0.5,
    report_format: ReportFormat::Json,
};

// Run automated test
let test_result = fluxion::validation::esp_r::run_automated_test(
    config.esp_r_output_path,
    config.fluxion_results_path,
    config.tolerance,
    config.report_format
)?;

println!("Test passed: {}", test_result.pass);
```

### Using the Integration Adapter

For framework integration:

```rust
use fluxion::validation::esp_r::create_integration_adapter;

// Create adapter
let adapter = create_integration_adapter(
    PathBuf::from("reference.csv"),
    0.5
);

// Run validation through adapter
let report = adapter.run_validation(&fluxion_results)?;
```

## Examples

### Basic Cross-Validation Example

```rust
use fluxion::validation::esp_r::examples::basic_cross_validation_example;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    basic_cross_validation_example()
}
```

**Expected Output:**
```
=== Basic Cross-Validation Example ===
Validation completed successfully!
Overall pass status: true
Number of zones validated: 2
Average temperature difference: 0.15°C
```

### Advanced Configuration Example

```rust
use fluxion::validation::esp_r::examples::advanced_cross_validation_example;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    advanced_cross_validation_example()
}
```

**Expected Output:**
```
=== Advanced Cross-Validation Example ===
Advanced validation completed!
Tolerance: 0.25°C
Overall pass status: true
Zones validated: 3
  Zone 'LivingRoom': pass=true, avg_diff=0.23°C, max_diff=0.35°C
  Zone 'Bedroom': pass=true, avg_diff=0.18°C, max_diff=0.28°C
  Zone 'Kitchen': pass=true, avg_diff=0.27°C, max_diff=0.38°C
Report saved to: examples/advanced_validation_report.json
```

### Error Handling Example

```rust
use fluxion::validation::esp_r::examples::error_handling_example;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    error_handling_example()
}
```

### Report Generation Example

```rust
use fluxion::validation::esp_r::examples::report_generation_example;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    report_generation_example()
}
```

## Integration

### CI/CD Integration

Add cross-validation to your GitHub Actions workflow:

```yaml
name: Cross-Validation CI

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable

    - name: Run cross-validation
      run: |
        cargo run --example cross_validation_example -- \
          --esp-r tests/reference_data/esp_r_standard.csv \
          --fluxion tests/expected_results/fluxion_standard.json \
          --tolerance 0.5 \
          --format json
```

### Framework Integration

Integrate with testing frameworks:

```rust
#[test]
fn test_esp_r_cross_validation() {
    use fluxion::validation::esp_r::examples::basic_cross_validation_example;

    let result = basic_cross_validation_example();
    assert!(result.is_ok());
}
```

### Multi-Reference Validation

Combine ESP-r validation with other reference tools:

```rust
use fluxion::validation::esp_r::EspRValidationAdapter;
use fluxion::validation::MultiZoneValidationResults;

// Create ESP-r adapter
let esp_r_adapter = EspRValidationAdapter::new(
    PathBuf::from("esp_r_reference.csv"),
    0.5
);

// Run ESP-r validation
let esp_r_report = esp_r_adapter.run_validation(&fluxion_results)?;

// Combine with other validation results
let overall_pass = esp_r_report.overall_pass
    && energy_plus_report.overall_pass
    && trnsys_report.overall_pass;
```

## Reporting Options

### JSON Reports

Generate machine-readable JSON reports:

```rust
let json_report = serde_json::to_string_pretty(&report)?;
std::fs::write("validation_report.json", &json_report)?;
```

**JSON Report Structure:**
```json
{
  "overall_pass": true,
  "average_temperature_difference": 0.25,
  "maximum_temperature_difference": 0.45,
  "zone_results": {
    "Zone1": {
      "pass": true,
      "average_difference": 0.23,
      "max_difference": 0.35,
      "standard_deviation": 0.08
    },
    "Zone2": {
      "pass": true,
      "average_difference": 0.27,
      "max_difference": 0.45,
      "standard_deviation": 0.12
    }
  }
}
```

### Markdown Reports

Generate human-readable Markdown reports:

```rust
let markdown_report = format!(
    "# Cross-Validation Report\n\n"
    + &format!("## Summary\n\n"
    + &format!("- Overall Status: {}\n", report.overall_pass)
    + &format!("- Average Difference: {:.2}°C\n", report.average_temperature_difference)
    + &format!("- Zones Validated: {}\n\n", report.zone_results.len()))
    + "## Zone Results\n\n"
    + &report.zone_results.iter().map(|(zone_name, zone_result)|
        format!("- **{}**: Avg={:.2}°C, Max={:.2}°C, Pass={}\n",
                zone_name, zone_result.average_difference,
                zone_result.max_difference, zone_result.pass)).collect::<String>()
);

std::fs::write("validation_report.md", markdown_report)?;
```

**Markdown Report Example:**
```markdown
# Cross-Validation Report

## Summary

- Overall Status: true
- Average Difference: 0.25°C
- Zones Validated: 3

## Zone Results

- **LivingRoom**: Avg=0.23°C, Max=0.35°C, Pass=true
- **Bedroom**: Avg=0.18°C, Max=0.28°C, Pass=true
- **Kitchen**: Avg=0.27°C, Max=0.38°C, Pass=true
```

### Custom Report Formatting

Create custom report formats for specific requirements:

```rust
let custom_report = format!(
    "Cross-Validation Summary - {}\n"
    + "==================================\n"
    + &format!("Tolerance: {:.2}°C\n", validator.tolerance)
    + &format!("Overall: {}\n", report.overall_pass)
    + &format!("Avg Diff: {:.2}°C\n", report.average_temperature_difference)
    + "\nZone Details:\n"
    + &report.zone_results.iter().map(|(zone_name, zone_result)|
        format!("  {}: {:.2}°C avg, {:.2}°C max {}\n",
                zone_name, zone_result.average_difference,
                zone_result.max_difference,
                if zone_result.pass { "✓" } else { "✗" })).collect::<String>()
);
```

## Troubleshooting

### Common Issues

**Missing Reference File:**
```
Error: "No such file or directory (os error 2)"
```

**Solution:** Verify the ESP-r reference file path is correct and the file exists.

**Invalid Tolerance:**
```
Error: "tolerance must be positive"
```

**Solution:** Use a positive tolerance value (e.g., 0.5 instead of -0.5).

**Empty Results:**
```
Warning: "No zones found in validation results"
```

**Solution:** Ensure your Fluxion results contain zone data before validation.

**Format Errors:**
```
Error: "Invalid CSV format"
```

**Solution:** Check that your ESP-r reference file is properly formatted CSV.

### Debugging Tips

1. **Verify File Paths:** Use absolute paths during development
2. **Check File Permissions:** Ensure read access to reference files
3. **Validate CSV Format:** Use `csvlint` to validate ESP-r files
4. **Start with Sample Data:** Use the provided examples before your own data
5. **Enable Debug Logging:** Set `RUST_LOG=debug` for detailed output

### Performance Issues

If validation is slow:
- Reduce the number of time steps in your reference data
- Use a larger tolerance for initial testing
- Validate zones separately if working with large models
- Ensure you're using release mode (`cargo build --release`)

## API Reference

### EspRValidator

```rust
pub struct EspRValidator {
    pub reference_path: PathBuf,
    pub tolerance: f64,
}
```

**Methods:**

- `new(reference_path: PathBuf, tolerance: f64) -> Self`
- `validate(&self, fluxion_results: &MultiZoneValidationResults) -> Result<CrossValidationReport, Box<dyn Error>>`

### EspRTestConfig

```rust
pub struct EspRTestConfig {
    pub esp_r_output_path: PathBuf,
    pub fluxion_results_path: PathBuf,
    pub tolerance: f64,
    pub report_format: ReportFormat,
}
```

### ReportFormat

```rust
pub enum ReportFormat {
    Json,
    Markdown,
}
```

### CrossValidationReport

```rust
pub struct CrossValidationReport {
    pub overall_pass: bool,
    pub average_temperature_difference: f64,
    pub maximum_temperature_difference: f64,
    pub zone_results: HashMap<String, ZoneValidationResult>,
}
```

### ZoneValidationResult

```rust
pub struct ZoneValidationResult {
    pub pass: bool,
    pub average_difference: f64,
    pub max_difference: f64,
    pub standard_deviation: f64,
}
```

## Configuration Options

### Tolerance Settings

| Tolerance (°C) | Use Case |
|---------------|----------|
| 0.1 - 0.2 | High precision research |
| 0.3 - 0.5 | Standard validation (default) |
| 0.6 - 1.0 | Early development, loose validation |
| > 1.0 | Debugging, conceptual validation |

### File Format Requirements

**ESP-r Reference CSV:**
- Header row with zone names
- Each column represents a zone
- Each row represents a time step
- Comma-separated values
- No missing values

**Fluxion Results JSON:**
- Valid JSON format
- Zone results in array format
- Temperature values as floats
- Consistent time steps

## Error Codes

| Error Code | Description | Solution |
|-----------|-------------|----------|
| E001 | Missing reference file | Verify file path and permissions |
| E002 | Invalid CSV format | Check CSV structure and headers |
| E003 | Empty validation results | Add zone data to results |
| E004 | Tolerance out of range | Use positive tolerance value |
| E005 | Zone mismatch | Ensure zones match between reference and results |
| E006 | Time step mismatch | Align time steps between datasets |
| E007 | JSON serialization error | Check result data structure |
| E008 | File I/O error | Verify disk space and permissions |

## Additional Resources

- [Fluxion GitHub Repository](https://github.com/anchapin/fluxion)
- [ESP-r Documentation](https://www.esru.strath.ac.uk/Programs/ESP-r.htm)
- [ASHRAE 140 Standard](https://www.ashrae.org/technical-resources/standards-and-guidelines/standards-addenda)
- [Fluxion API Reference](../API_REFERENCE.md)
- [Validation Examples](../../examples/)

## Support

For issues or questions:

- Open a GitHub issue: https://github.com/anchapin/fluxion/issues
- Check the [Known Issues](../KNOWN_ISSUES.md) document
- Review the [Troubleshooting Guide](../TROUBLESHOOTING.md)

## License

This documentation and the associated code are licensed under the Apache License 2.0. See the [LICENSE](../../LICENSE) file for details.
