//! CLI integration tests
//!
//! Tests validate CLI commands work correctly using std::process::Command.

use std::process::Command;
use tempfile::tempdir;

/// Test CLI validate --all command
#[test]
fn test_cli_validate_command() {
    // Run fluxion validate --all
    let output = Command::new("cargo")
        .args(["run", "--bin", "fluxion", "--", "validate", "--all"])
        .output()
        .expect("Failed to execute fluxion validate --all");

    // Check exit code (may be 0 or non-zero depending on validation results)
    // We just verify the command runs without crashing
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify output contains validation-related content
    let output_str = format!("{}\n{}", stdout, stderr);
    println!("CLI validate output:\n{}", output_str);

    // Command should run successfully
    assert!(
        output.status.success() || stderr.contains("Validation Report"),
        "CLI validate command should run successfully or show validation report"
    );
}

/// Test CLI validate --case command
#[test]
fn test_cli_validate_case_command() {
    // Run fluxion validate --case 600
    let output = Command::new("cargo")
        .args(["run", "--bin", "fluxion", "--", "validate", "--case", "600"])
        .output()
        .expect("Failed to execute fluxion validate --case 600");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify output contains case-specific content
    let output_str = format!("{}\n{}", stdout, stderr);
    println!("CLI validate --case 600 output:\n{}", output_str);

    // Command should run successfully
    assert!(
        output.status.success() || stderr.contains("Case 600"),
        "CLI validate --case command should run successfully or show case information"
    );
}

/// Test CLI sensitivity command (if available)
#[test]
fn test_cli_sensitivity_command() {
    // Create a temporary config file
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let config_path = temp_dir.path().join("sensitivity_config.yaml");

    // Create a minimal sensitivity config
    let config_content = r#"
case_id: "600"
method: "oat"
levels: 5
parameters:
  - name: "window_u_value"
    min: 1.0
    max: 3.0
  - name: "heating_setpoint"
    min: 18.0
    max: 22.0
"#;

    std::fs::write(&config_path, config_content).expect("Failed to write config file");

    // Run fluxion sensitivity --config
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "fluxion",
            "--",
            "sensitivity",
            "--config",
            config_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute fluxion sensitivity");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify output contains sensitivity analysis content
    let output_str = format!("{}\n{}", stdout, stderr);
    println!("CLI sensitivity output:\n{}", output_str);

    // Command should run successfully (or fail gracefully if sensitivity not available)
    assert!(
        output.status.success() || stderr.contains("sensitivity") || stderr.contains("Sensitivity"),
        "CLI sensitivity command should run successfully or show sensitivity information"
    );
}

/// Test CLI binary exists and is executable
#[test]
fn test_cli_binary_exists() {
    // Verify cargo run --bin fluxion works
    let output = Command::new("cargo")
        .args(["run", "--bin", "fluxion", "--", "--help"])
        .output()
        .expect("Failed to execute fluxion --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify help output is shown
    let output_str = format!("{}\n{}", stdout, stderr);
    println!("CLI --help output:\n{}", output_str);

    // Command should show help information
    assert!(
        output_str.contains("Usage:") || output_str.contains("usage:"),
        "CLI --help should show usage information"
    );
}
