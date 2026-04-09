// validation/reporting/examples.rs
/// Examples for validation reporting functionality
use std::error::Error;

/// Generate example validation report
pub fn generate_example_report() -> Result<String, Box<dyn Error>> {
    Ok(r#"{
  "validation_results": {
    "total_cases": 10,
    "passed_cases": 9,
    "failures": [
      {
        "case_id": "case_600",
        "reason": "Temperature deviation exceeded threshold"
      }
    ]
  },
  "statistics": {
    "mean_deviation": 0.23,
    "max_deviation": 0.87,
    "standard_deviation": 0.15
  }
}"#
    .to_string())
}

/// Generate example markdown report
pub fn generate_example_markdown_report() -> Result<String, Box<dyn Error>> {
    Ok(r#"# Validation Report

## Summary
- Total Cases: 10
- Passed: 9 (90%)
- Failed: 1 (10%)

## Failures
- **case_600**: Temperature deviation exceeded threshold (0.87°C)

## Statistics
- Mean Deviation: 0.23°C
- Max Deviation: 0.87°C
- Standard Deviation: 0.15°C
"#
    .to_string())
}

/// Generate example HTML report
pub fn generate_example_html_report() -> Result<String, Box<dyn Error>> {
    Ok(r#"<html>
<head><title>Validation Report</title></head>
<body>
  <h1>Validation Report</h1>
  <h2>Summary</h2>
  <ul>
    <li>Total Cases: 10</li>
    <li>Passed: 9 (90%)</li>
    <li>Failed: 1 (10%)</li>
  </ul>
  <h2>Failures</h2>
  <ul>
    <li><strong>case_600</strong>: Temperature deviation exceeded threshold (0.87°C)</li>
  </ul>
</body>
</html>"#
        .to_string())
}
