# Validation Reporting System

The Fluxion validation reporting system provides comprehensive reporting capabilities for all validation modules, including ASHRAE 140, climate zone validation, and occupancy pattern validation.

## Overview

The validation reporting system consolidates results from multiple validation modules into unified reports that can be exported in various formats for analysis, compliance reporting, and quality assurance.

## Report Structure

A comprehensive validation report includes:

- **Metadata**: Generation timestamp, Fluxion version, validation coverage
- **ASHRAE 140 Validation Results**: Case-by-case validation results with energy metrics
- **Climate Zone Validation Results**: Climate zone-specific validation metrics
- **Occupancy Pattern Validation Results**: Occupancy pattern validation status
- **Cross-Validation Results**: Comparison with external tools (ESP-r, EnergyPlus, etc.)
- **Summary Statistics**: Overall validation counts and pass rates
- **Quality Metrics**: Error metrics and coverage scores

## CLI Usage

### Generate Comprehensive Validation Report

```bash
# Generate Markdown report (default)
fluxion validation report generate --output reports/comprehensive.md

# Generate JSON report
fluxion validation report generate --format json --output reports/comprehensive.json

# Generate HTML report
fluxion validation report generate --format html --output reports/comprehensive.html

# Generate comprehensive report with diagnostics
fluxion validation report generate --comprehensive --diagnostics --output reports/full_report.md
```

### CLI Command Reference

```
fluxion validation report generate [OPTIONS]

OPTIONS:
  -f, --format <FORMAT>    Report format: json, html, or markdown [default: markdown]
  -o, --output <PATH>     Output file path [required]
  -c, --comprehensive      Include comprehensive data from all validation modules
  -d, --diagnostics        Include detailed diagnostics in report
  -h, --help              Print help information
```

## Programmatic API

### Basic Usage

```rust
use fluxion::validation::reporting::{ValidationReporter, ReportingConfig, ReportFormat};

let config = ReportingConfig {
    output_dir: "reports".to_string(),
    format: ReportFormat::Markdown,
    include_diagnostics: true,
    comprehensive: true,
};

let reporter = ValidationReporter::new(config);
let report = reporter.generate_comprehensive_report()?;
```

### Report Generation Methods

```rust
// Generate and export JSON report
reporter.generate_json_report("reports/comprehensive.json")?;

// Generate and export HTML report
reporter.generate_html_report("reports/comprehensive.html")?;

// Generate and export Markdown report
reporter.generate_markdown_report("reports/comprehensive.md")?;
```

## Report Formats

### JSON Format

JSON reports provide machine-readable validation results suitable for automated processing and integration with other systems.

```json
{
  "metadata": {
    "generated_at": "2024-01-01T00:00:00Z",
    "fluxion_version": "1.0.0",
    "validation_coverage": "Comprehensive (ASHRAE 140 + Climate + Occupancy)",
    "total_test_cases": 42,
    "passing_cases": 38,
    "warning_cases": 3,
    "failing_cases": 1
  },
  "ashrae140_results": [
    {
      "case_id": "600",
      "case_description": "ASHRAE 140 Case 600 with residential occupancy in climate zone 4A",
      "annual_heating_mwh": 12.5,
      "annual_cooling_mwh": 8.3,
      "peak_heating_kw": 5.2,
      "peak_cooling_kw": 4.1,
      "min_temp_celsius": 18.5,
      "max_temp_celsius": 26.2,
      "status": "Pass",
      "reference_range": {
        "min": 0.0,
        "max": 0.0,
        "source": "ASHRAE 140"
      }
    }
  ],
  "climate_results": [
    {
      "zone_id": "4A",
      "zone_description": "ASHRAE Climate Zone 4A - Mixed-Humid",
      "validation_results": [
        {
          "metric": "Temperature Range",
          "value": 35.0,
          "reference_min": 10.0,
          "reference_max": 80.0,
          "status": "Pass"
        }
      ],
      "overall_status": "Pass"
    }
  ],
  "occupancy_results": [
    {
      "pattern_name": "residential",
      "pattern_description": "Occupancy pattern: residential",
      "validation_status": "Pass",
      "coverage_percentage": 100.0
    }
  ],
  "cross_validation_results": [],
  "summary": {
    "total_validations": 42,
    "pass_count": 38,
    "warning_count": 3,
    "fail_count": 1,
    "pass_rate": 0.9047619047619048,
    "overall_status": "Pass"
  },
  "quality_metrics": {
    "mean_absolute_error": 0.5,
    "root_mean_square_error": 0.7,
    "max_deviation": 1.2,
    "coverage_score": 100.0,
    "completeness_score": 95.0
  }
}
```

### HTML Format

HTML reports provide interactive, browser-readable validation results with styled tables and visual formatting.

```html
<html>
<head><title>Comprehensive Validation Report</title>
<style>
  /* CSS styling for tables and formatting */
</style>
</head>
<body>
  <h1>Comprehensive Validation Report</h1>
  <p><strong>Generated:</strong> 2024-01-01T00:00:00Z</p>
  <p><strong>Fluxion Version:</strong> 1.0.0</p>
  <p><strong>Validation Coverage:</strong> Comprehensive (ASHRAE 140 + Climate + Occupancy)</p>

  <h2>Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Validations</td><td>42</td></tr>
    <tr><td>Pass Count</td><td>38</td></tr>
    <tr><td>Warning Count</td><td>3</td></tr>
    <tr><td>Fail Count</td><td>1</td></tr>
    <tr><td>Pass Rate</td><td>90.48%</td></tr>
  </table>

  <!-- Additional sections for ASHRAE 140, Climate, Occupancy results -->
</body>
</html>
```

### Markdown Format

Markdown reports provide human-readable validation results suitable for documentation and version control.

```markdown
# Comprehensive Validation Report

**Generated:** 2024-01-01T00:00:00Z
**Fluxion Version:** 1.0.0
**Validation Coverage:** Comprehensive (ASHRAE 140 + Climate + Occupancy)

## Summary

- **Total Validations:** 42
- **Pass Count:** 38
- **Warning Count:** 3
- **Fail Count:** 1
- **Pass Rate:** 90.48%

## Quality Metrics

- **Mean Absolute Error:** 0.5000
- **Root Mean Square Error:** 0.7000
- **Max Deviation:** 1.2000
- **Coverage Score:** 100.00%
- **Completeness Score:** 95.00%

## ASHRAE 140 Validation Results

| Case ID | Description | Status | Annual Heating (MWh) | Annual Cooling (MWh) |
|---------|-------------|--------|---------------------|----------------------|
| 600 | ASHRAE 140 Case 600 with residential occupancy in climate zone 4A | Pass | 12.50 | 8.30 |

## Climate Zone Validation Results

| Zone ID | Description | Status |
|---------|-------------|--------|
| 4A | ASHRAE Climate Zone 4A - Mixed-Humid | Pass |

## Occupancy Pattern Validation Results

| Pattern Name | Description | Status | Coverage |
|--------------|-------------|--------|----------|
| residential | Occupancy pattern: residential | Pass | 100.0% |
```

## Quality Metrics

The validation reporting system calculates several quality metrics to assess the overall validation quality:

### Mean Absolute Error (MAE)

Measures the average magnitude of errors in validation results, without considering their direction.

### Root Mean Square Error (RMSE)

Measures the square root of the average squared errors, giving more weight to larger errors.

### Max Deviation

Identifies the maximum deviation from reference values across all validation tests.

### Coverage Score

Indicates the percentage of validation test cases covered by the reporting system.

### Completeness Score

Assesses how complete the validation data is across all modules.

## Validation Status Levels

The reporting system uses the following validation status levels:

- **Pass**: Validation test meets all acceptance criteria
- **Warning**: Validation test has minor issues or deviations
- **Fail**: Validation test fails to meet acceptance criteria
- **NotApplicable**: Validation test is not applicable to the current configuration

## Integration with Validation Workflows

### Pre-commit Validation

```bash
#!/bin/bash
# Run validation and generate report before commit
fluxion validation report generate --output reports/pre_commit_report.md
if [ $? -ne 0 ]; then
  echo "Validation failed. Commit aborted."
  exit 1
fi
echo "Validation passed. Proceeding with commit."
```

### CI/CD Pipeline Integration

```yaml
# GitHub Actions example
- name: Run Validation Tests
  run: |
    fluxion validation report generate --format json --output validation_report.json
    # Check overall validation status
    if [ $(jq '.summary.overall_status' validation_report.json) != "Pass" ]; then
      echo "Validation failed"
      exit 1
    fi
```

### Compliance Reporting

```rust
// Generate compliance report for auditing
let config = ReportingConfig {
    output_dir: "compliance".to_string(),
    format: ReportFormat::Json,
    include_diagnostics: true,
    comprehensive: true,
};

let reporter = ValidationReporter::new(config);
let report = reporter.generate_comprehensive_report()?;

// Check compliance
if report.summary.pass_rate >= 0.95 {
    println!("✅ System compliant: {}% pass rate", report.summary.pass_rate * 100.0);
} else {
    println!("⚠️  Compliance issue: {}% pass rate", report.summary.pass_rate * 100.0);
}

reporter.generate_json_report("compliance/audit_report.json")?;
```

## Troubleshooting

### Common Issues

#### Report generation fails with permission errors

**Solution:** Ensure the output directory exists and is writable:

```bash
mkdir -p reports
chmod 755 reports
```

#### Missing validation data

**Solution:** Run validation tests before generating reports:

```bash
fluxion validation run-series 800-810
fluxion validation report generate --output reports/comprehensive.md
```

#### Invalid output format

**Solution:** Use one of the supported formats: `json`, `html`, or `markdown`

```bash
fluxion validation report generate --format markdown --output report.md
```

## Best Practices

### Regular Validation

- Run validation tests regularly to catch issues early
- Generate reports after significant code changes
- Include validation reports in pull requests

### Report Analysis

- Monitor pass rates over time
- Investigate warnings and failures promptly
- Track quality metrics to identify trends

### Compliance

- Archive validation reports for auditing
- Include validation status in release notes
- Use JSON format for automated compliance checking

## Examples

See `examples/validation_reporting.rs` for comprehensive examples of:

- Basic report generation
- Custom configuration and output formats
- Programmatic report analysis
- Integration with validation workflows
- Error handling

## API Reference

### `ValidationReporter`

Main reporting struct for generating comprehensive validation reports.

#### Methods

- `new(config: ReportingConfig) -> Self`
- `generate_comprehensive_report() -> Result<ComprehensiveValidationReport, String>`
- `generate_json_report(path: &str) -> Result<(), String>`
- `generate_html_report(path: &str) -> Result<(), String>`
- `generate_markdown_report(path: &str) -> Result<(), String>`

### `ReportingConfig`

Configuration for validation reporting.

#### Fields

- `output_dir: String` - Output directory for reports
- `format: ReportFormat` - Report format (Markdown, Html, Json)
- `include_diagnostics: bool` - Include detailed diagnostics
- `comprehensive: bool` - Generate comprehensive reports

### `ComprehensiveValidationReport`

Main report structure containing all validation results.

#### Fields

- `metadata: ReportMetadata`
- `ashrae140_results: Vec<ASHRAE140ReportSection>`
- `climate_results: Vec<ClimateZoneReportSection>`
- `occupancy_results: Vec<OccupancyPatternReportSection>`
- `cross_validation_results: Vec<CrossValidationReportSection>`
- `summary: ReportSummary`
- `quality_metrics: QualityMetrics`

## Future Enhancements

- PDF report generation
- Interactive HTML dashboards
- Historical trend analysis
- Automated compliance checking
- Integration with monitoring systems
