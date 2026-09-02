use std::env;
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;

fn fluxion_bin() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    if PathBuf::from(&manifest_dir)
        .join("target/release/fluxion")
        .exists()
    {
        PathBuf::from(&manifest_dir).join("target/release/fluxion")
    } else {
        PathBuf::from(manifest_dir).join("target/debug/fluxion")
    }
}

#[test]
fn test_sensitivity_command() {
    let temp_dir = tempdir().unwrap();
    let config_content = r#"
case_id: "600"
method: "oat"
levels: 2
parameters:
  - name: "window_u"
    min: 0.1
    max: 5.0
  - name: "heating_setpoint"
    min: 15.0
    max: 25.0
"#;
    let config_path = temp_dir.path().join("sensitivity.yaml");
    std::fs::write(&config_path, config_content).unwrap();

    let output = Command::new(fluxion_bin())
        .arg("sensitivity")
        .arg("--config")
        .arg(&config_path)
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to execute fluxion sensitivity");
    assert!(
        output.status.success(),
        "Sensitivity command failed: stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Default output dir is ./tmp relative to the current directory
    let csv_path = temp_dir.path().join("tmp").join("sensitivity_report.csv");
    let md_path = temp_dir.path().join("tmp").join("sensitivity_report.md");
    assert!(csv_path.exists(), "CSV report not found: {:?}", csv_path);
    assert!(md_path.exists(), "Markdown report not found: {:?}", md_path);
    // No scratch files may land directly in the working directory
    assert!(
        !temp_dir.path().join("sensitivity_report.csv").exists(),
        "CSV report must not be written to cwd"
    );
    assert!(
        !temp_dir.path().join("sensitivity_report.md").exists(),
        "Markdown report must not be written to cwd"
    );
}

#[test]
fn test_sensitivity_command_explicit_output_dir() {
    let temp_dir = tempdir().unwrap();
    let out_dir = temp_dir.path().join("reports").join("sens");
    let config_content = r#"
case_id: "600"
method: "oat"
levels: 2
parameters:
  - name: "window_u"
    min: 0.1
    max: 5.0
  - name: "heating_setpoint"
    min: 15.0
    max: 25.0
"#;
    let config_path = temp_dir.path().join("sensitivity.yaml");
    std::fs::write(&config_path, config_content).unwrap();

    let output = Command::new(fluxion_bin())
        .arg("sensitivity")
        .arg("--config")
        .arg(&config_path)
        .arg("--output-dir")
        .arg(&out_dir)
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to execute fluxion sensitivity");
    assert!(
        output.status.success(),
        "Sensitivity command failed: stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let csv_path = out_dir.join("sensitivity_report.csv");
    let md_path = out_dir.join("sensitivity_report.md");
    assert!(csv_path.exists(), "CSV report not found: {:?}", csv_path);
    assert!(md_path.exists(), "Markdown report not found: {:?}", md_path);
    assert!(
        !temp_dir.path().join("tmp").exists(),
        "default ./tmp must not be created when --output-dir is explicit"
    );
}

#[test]
fn test_delta_command() {
    use fluxion::analysis::delta::{DeltaConfig, Variant};
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    use std::collections::HashMap;

    let temp_dir = tempdir().unwrap();

    // Build a simple DeltaConfig
    let base_spec = ASHRAE140Case::Case600.spec();
    let mut patch = HashMap::new();
    patch.insert(
        "infiltration_ach".to_string(),
        serde_yaml::to_value(1.5).unwrap(),
    );
    let delta_config = DeltaConfig {
        base: base_spec,
        variants: vec![Variant {
            name: "high_infiltration".to_string(),
            patch: Some(patch),
            sweep: None,
        }],
        warm_up_years: 2,
    };
    let yaml = serde_yaml::to_string(&delta_config).unwrap();
    let config_path = temp_dir.path().join("delta.yaml");
    std::fs::write(&config_path, yaml).unwrap();

    let output = Command::new(fluxion_bin())
        .arg("delta")
        .arg("--config")
        .arg(&config_path)
        .arg("--hourly")
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to execute fluxion delta");
    assert!(
        output.status.success(),
        "Delta command failed: stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Default output dir is ./tmp relative to the current directory
    let md_path = temp_dir.path().join("tmp").join("delta_report.md");
    let csv_path = temp_dir.path().join("tmp").join("hourly_differences.csv");
    assert!(md_path.exists(), "Delta markdown report not found");
    assert!(csv_path.exists(), "Hourly differences CSV not found");
}

#[test]
fn test_components_command() {
    let temp_dir = tempdir().unwrap();
    let case_id = "600FF";

    let output = Command::new(fluxion_bin())
        .arg("components")
        .arg("--case")
        .arg(case_id)
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to execute fluxion components");
    assert!(
        output.status.success(),
        "Components command failed: stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let expected_csv = temp_dir
        .path()
        .join("tmp")
        .join(format!("{}_components.csv", case_id));
    assert!(
        expected_csv.exists(),
        "Components CSV not found: {:?}",
        expected_csv
    );
}

#[test]
fn test_swing_command() {
    let temp_dir = tempdir().unwrap();
    let case_id = "600FF";

    let output = Command::new(fluxion_bin())
        .arg("swing")
        .arg("--case")
        .arg(case_id)
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to execute fluxion swing");
    assert!(
        output.status.success(),
        "Swing command failed: stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Swing Analysis Report"),
        "Missing swing report header"
    );
}

#[test]
fn test_visualize_command() {
    let temp_dir = tempdir().unwrap();
    // Create a minimal diagnostics CSV
    let csv_content = "Hour,Zone_Temps,Mass_Temps,Surface_Temps,Solar_Watts,Internal_Watts,HVAC_Watts,InterZone_Watts,Infiltration_Watts\n";
    let csv_content = csv_content.to_string()
        + "0,20.0;20.5,21.0;21.5,22.0;22.5,100.0;120.0,50.0;50.0,0.0;0.0,0.0;0.0,30.0;30.0\n"
        + "1,20.1;20.6,21.1;21.6,22.1;22.6,110.0;130.0,55.0;55.0,0.0;0.0,0.0;0.0,31.0;31.0\n";
    let csv_path = temp_dir.path().join("diag.csv");
    std::fs::write(&csv_path, csv_content).unwrap();

    let output = Command::new(fluxion_bin())
        .arg("visualize")
        .arg("--input")
        .arg(&csv_path)
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to execute fluxion visualize");
    assert!(
        output.status.success(),
        "Visualize command failed: stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let html_path = temp_dir.path().join("diag.html");
    assert!(html_path.exists(), "Visualization HTML not found");
    let html_content = std::fs::read_to_string(&html_path).unwrap();
    assert!(html_content.contains("plotly"), "HTML missing Plotly");
}

#[test]
fn test_animate_command() {
    let temp_dir = tempdir().unwrap();
    let csv_content = "Hour,Zone_Temps,Mass_Temps,Surface_Temps,Solar_Watts,Internal_Watts,HVAC_Watts,InterZone_Watts,Infiltration_Watts\n";
    let csv_content = csv_content.to_string()
        + "0,20.0;20.5,21.0;21.5,22.0;22.5,100.0;120.0,50.0;50.0,0.0;0.0,0.0;0.0,30.0;30.0\n"
        + "1,20.1;20.6,21.1;21.6,22.1;22.6,110.0;130.0,55.0;55.0,0.0;0.0,0.0;0.0,31.0;31.0\n";
    let csv_path = temp_dir.path().join("diag.csv");
    std::fs::write(&csv_path, csv_content).unwrap();

    let output = Command::new(fluxion_bin())
        .arg("animate")
        .arg("--input")
        .arg(&csv_path)
        .current_dir(temp_dir.path())
        .output()
        .expect("Failed to execute fluxion animate");
    assert!(
        output.status.success(),
        "Animate command failed: stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let html_path = temp_dir.path().join("diag.html");
    assert!(html_path.exists(), "Animation HTML not found");
    let html_content = std::fs::read_to_string(&html_path).unwrap();
    assert!(html_content.contains("playBtn"), "HTML missing play button");
    assert!(
        html_content.contains("pauseBtn"),
        "HTML missing pause button"
    );
    assert!(html_content.contains("scrubber"), "HTML missing scrubber");
}

#[test]
fn test_references_update_command() {
    let output = Command::new(fluxion_bin())
        .arg("references")
        .arg("update")
        .output()
        .expect("Failed to execute fluxion references update");
    assert!(output.status.success(), "References update command failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Reference data is valid"),
        "Expected successful validation message, got: {}",
        stdout
    );
    assert!(
        stdout.contains("Version:"),
        "Expected version line in output"
    );
}
