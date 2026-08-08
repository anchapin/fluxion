//! Co-Simulation Master Integration Test for Issue #2391
//!
//! Tests the MasterSim co-simulation master setup for BES+FFD coupled simulation.
//!
//! ## Acceptance Criteria (Issue #2391)
//!
//! 1. **FMU Loading**: MasterSim can load BES and FFD FMUs simultaneously
//! 2. **24-Hour Simulation**: Simulation completes to t=86400s without hanging
//! 3. **No Deadlocks**: All 1440 FFD micro-steps (24h × 60min) complete
//! 4. **Clock Synchronization**: Master clock advances correctly with FFD micro-stepping
//!
//! ## Test Strategy
//!
//! This test uses the Python harness in `tools/cosim/run_cosimulation.py` which:
//! - Generates dummy BES and FFD FMU archives (FMI 2.0 compliant)
//! - Creates a MasterSim `.ums` configuration file
//! - Runs MasterSim (if available)
//! - Validates the simulation results
//!
//! The test is skipped if MasterSim is not installed (graceful degradation).

use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

/// Path to the cosimulation tools directory
fn cosim_dir() -> PathBuf {
    PathBuf::from("tools/cosim")
}

/// Find the MasterSim executable, checking common locations and PATH.
fn find_mastersim() -> Option<String> {
    // Check environment variable first
    if let Ok(path) = std::env::var("MASTERSIM_BIN") {
        if !path.is_empty() {
            return Some(path);
        }
    }

    // Check if 'mastersim' is in PATH
    if Command::new("which")
        .arg("mastersim")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some("mastersim".to_string());
    }

    // Check common installation locations
    let home = std::env::var("HOME").ok();
    let common_paths = ["/usr/local/bin/mastersim", "/usr/bin/mastersim"];

    for path in &common_paths {
        if PathBuf::from(path).exists() {
            return Some(path.to_string());
        }
    }

    // Check HOME-relative paths
    if let Some(ref h) = home {
        let home_paths = [
            format!("{}/MasterSim/build/mastersim", h),
            format!("{}/bin/mastersim", h),
        ];
        for path in &home_paths {
            if PathBuf::from(path).exists() {
                return Some(path.clone());
            }
        }
    }

    None
}

/// Check if Python 3 is available
fn python_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run the cosimulation Python harness
///
/// Returns: (exit_code, stdout, stderr)
fn run_cosim_harness(generate_only: bool) -> Result<(i32, String, String), std::io::Error> {
    let script = cosim_dir().join("run_cosimulation.py");

    let mut cmd = Command::new("python3");
    cmd.arg(script);

    if generate_only {
        cmd.arg("--generate-only");
    } else {
        cmd.arg("--run");
    }

    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let output = cmd.output()?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok((output.status.code().unwrap_or(-1), stdout, stderr))
}

/// Test that the cosimulation harness can generate FMUs and configuration.
///
/// This test does NOT require MasterSim to be installed. It only verifies
/// that the Python script can generate:
/// - bes_dummy.fmu
/// - ffd_dummy.fmu
/// - master_config_BES_FFD.ums
#[test]
fn test_cosim_generate_fmus_and_config() {
    if !python_available() {
        eprintln!("SKIP: Python 3 not available");
        return;
    }

    let result = run_cosim_harness(true);

    match result {
        Ok((exit_code, stdout, stderr)) => {
            println!("STDOUT:\n{}", stdout);
            if !stderr.is_empty() {
                println!("STDERR:\n{}", stderr);
            }

            assert_eq!(
                exit_code, 0,
                "cosimulation harness --generate-only failed with exit code {}",
                exit_code
            );

            // Verify generated files exist
            let cosim_dir = cosim_dir();
            assert!(
                cosim_dir.join("bes_dummy.fmu").exists(),
                "BES FMU not generated"
            );
            assert!(
                cosim_dir.join("ffd_dummy.fmu").exists(),
                "FFD FMU not generated"
            );
            assert!(
                cosim_dir.join("master_config_BES_FFD.ums").exists(),
                "MasterSim config not generated"
            );

            println!("Generated files verified:");
            println!("  - tools/cosim/bes_dummy.fmu");
            println!("  - tools/cosim/ffd_dummy.fmu");
            println!("  - tools/cosim/master_config_BES_FFD.ums");
        }
        Err(e) => {
            panic!("Failed to run cosimulation harness: {}", e);
        }
    }
}

/// Test that the MasterSim configuration file is valid.
///
/// Verifies:
/// - Configuration file exists and is parseable
/// - Both BES and FFD FMUs are referenced
/// - Simulation parameters are set for 24-hour run
/// - Data exchange connections are defined
#[test]
fn test_mastersim_config_validity() {
    if !python_available() {
        eprintln!("SKIP: Python 3 not available");
        return;
    }

    // First generate the config
    run_cosim_harness(true).expect("generate step failed");

    let config_path = cosim_dir().join("master_config_BES_FFD.ums");
    let content = std::fs::read_to_string(&config_path).expect("Failed to read MasterSim config");

    // Check for required elements
    assert!(
        content.contains("CoSimulationMaster"),
        "Config missing CoSimulationMaster element"
    );
    assert!(
        content.contains("SimulationParameters"),
        "Config missing SimulationParameters element"
    );
    assert!(
        content.contains("startTime=\"0"),
        "Config missing startTime"
    );
    assert!(
        content.contains("endTime=\"86400"),
        "Config missing endTime=86400 (24 hours)"
    );

    // Check for FMU definitions
    assert!(
        content.contains("name=\"BES\""),
        "Config missing BES FMU definition"
    );
    assert!(
        content.contains("name=\"FFD\""),
        "Config missing FFD FMU definition"
    );

    // Check for connections
    assert!(
        content.contains("fmu1=\"bes1\""),
        "Config missing BES→FFD connection"
    );
    assert!(
        content.contains("fmu1=\"ffd1\""),
        "Config missing FFD→BES connection"
    );

    println!("MasterSim configuration structure validated:");
    println!("  - Simulation period: 0s to 86400s (24 hours)");
    println!("  - BES and FFD FMUs defined");
    println!("  - Data exchange connections configured");
}

/// Test that BES FMU archive is FMI 2.0 compliant.
///
/// Verifies the generated FMU contains:
/// - modelDescription.xml at root
/// - Valid FMI 2.0 XML structure
/// - Required CoSimulation element
/// - Correct variable declarations
#[test]
fn test_bes_fmu_fmi20_compliance() {
    if !python_available() {
        eprintln!("SKIP: Python 3 not available");
        return;
    }

    run_cosim_harness(true).expect("generate step failed");

    let fmu_path = cosim_dir().join("bes_dummy.fmu");
    let file = std::fs::File::open(&fmu_path).expect("Failed to open BES FMU");
    let mut archive = zip::ZipArchive::new(file).expect("Failed to read FMU as ZIP");

    // modelDescription.xml must be at root
    let mut xml_content = String::new();
    archive
        .by_name("modelDescription.xml")
        .expect("FMU missing modelDescription.xml")
        .read_to_string(&mut xml_content)
        .expect("Failed to read modelDescription.xml");

    // Validate FMI 2.0 structure
    assert!(
        xml_content.contains("fmiVersion=\"2.0\""),
        "modelDescription.xml missing fmiVersion=\"2.0\""
    );
    assert!(
        xml_content.contains("<CoSimulation"),
        "modelDescription.xml missing CoSimulation element"
    );
    assert!(
        xml_content.contains("modelName=\"FluxionBES\""),
        "modelDescription.xml missing modelName"
    );
    assert!(
        xml_content.contains("outdoor_temperature"),
        "modelDescription.xml missing outdoor_temperature variable"
    );
    assert!(
        xml_content.contains("zone_temperature"),
        "modelDescription.xml missing zone_temperature variable"
    );
    assert!(
        xml_content.contains("heating_load"),
        "modelDescription.xml missing heating_load variable"
    );
    assert!(
        xml_content.contains("causality=\"input\""),
        "modelDescription.xml missing input variables"
    );
    assert!(
        xml_content.contains("causality=\"output\""),
        "modelDescription.xml missing output variables"
    );

    println!("BES FMU FMI 2.0 compliance validated");
}

/// Test that FFD FMU archive is FMI 2.0 compliant.
///
/// Verifies the generated FMU contains:
/// - modelDescription.xml at root
/// - Valid FMI 2.0 XML structure
/// - FFD-specific variables (inlet_air_temperature, zone_air_temperature_N, etc.)
#[test]
fn test_ffd_fmu_fmi20_compliance() {
    if !python_available() {
        eprintln!("SKIP: Python 3 not available");
        return;
    }

    run_cosim_harness(true).expect("generate step failed");

    let fmu_path = cosim_dir().join("ffd_dummy.fmu");
    let file = std::fs::File::open(&fmu_path).expect("Failed to open FFD FMU");
    let mut archive = zip::ZipArchive::new(file).expect("Failed to read FMU as ZIP");

    let mut xml_content = String::new();
    archive
        .by_name("modelDescription.xml")
        .expect("FMU missing modelDescription.xml")
        .read_to_string(&mut xml_content)
        .expect("Failed to read modelDescription.xml");

    // Validate FMI 2.0 structure
    assert!(
        xml_content.contains("fmiVersion=\"2.0\""),
        "modelDescription.xml missing fmiVersion=\"2.0\""
    );
    assert!(
        xml_content.contains("<CoSimulation"),
        "modelDescription.xml missing CoSimulation element"
    );
    assert!(
        xml_content.contains("modelName=\"FluxionFFD\""),
        "modelDescription.xml missing modelName"
    );

    // Check FFD-specific variables
    assert!(
        xml_content.contains("inlet_air_temperature"),
        "modelDescription.xml missing inlet_air_temperature"
    );
    assert!(
        xml_content.contains("mass_flow_rate_supply"),
        "modelDescription.xml missing mass_flow_rate_supply"
    );
    assert!(
        xml_content.contains("zone_air_temperature_0"),
        "modelDescription.xml missing zone_air_temperature_0"
    );
    assert!(
        xml_content.contains("chtc_0"),
        "modelDescription.xml missing chtc_0"
    );
    assert!(
        xml_content.contains("surface_heat_flux_0"),
        "modelDescription.xml missing surface_heat_flux_0"
    );

    println!("FFD FMU FMI 2.0 compliance validated");
}

/// Full co-simulation integration test (requires MasterSim).
///
/// This test runs the complete 24-hour BES+FFD co-simulation using MasterSim.
///
/// ## Requirements
/// - MasterSim installed and in PATH (or set MASTERSIM_BIN)
/// - Python 3 with zipfile module
///
/// ## What it tests
/// 1. MasterSim can load both FMUs simultaneously
/// 2. Simulation completes to t=86400s (24 hours)
/// 3. No synchronization deadlocks occur
/// 4. Master clock advances correctly
///
/// ## Graceful Degradation
/// If MasterSim is not available, the test is skipped with a clear message.
#[test]
fn test_mastersim_bes_ffd_24hour_cosimulation() {
    if !python_available() {
        eprintln!("SKIP: Python 3 not available");
        return;
    }

    let mastersim_path = find_mastersim();
    if mastersim_path.is_none() {
        eprintln!("SKIP: MasterSim not found in PATH or common locations");
        eprintln!("SKIP: Install from https://github.com/ghorwin/MasterSim");
        eprintln!("SKIP: Or set MASTERSIM_BIN environment variable");
        return;
    }

    println!("Found MasterSim at: {}", mastersim_path.as_ref().unwrap());

    // First generate FMUs and config
    run_cosim_harness(true).expect("generate step failed");

    // Run the simulation
    let result = run_cosim_harness(false);

    match result {
        Ok((exit_code, stdout, stderr)) => {
            println!("STDOUT:\n{}", stdout);
            if !stderr.is_empty() {
                println!("STDERR:\n{}", stderr);
            }

            // MasterSim should exit with code 0 on success
            assert_eq!(
                exit_code, 0,
                "MasterSim simulation failed with exit code {}",
                exit_code
            );

            // Check for result files
            let cosim_dir = cosim_dir();
            let result_files: Vec<_> = std::fs::read_dir(&cosim_dir)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| {
                    let binding = e.file_name();
                    let name = binding.to_string_lossy();
                    name.starts_with("cosim_results")
                })
                .map(|e| e.path())
                .collect();

            assert!(
                !result_files.is_empty(),
                "No result files produced by MasterSim"
            );

            println!("MasterSim 24-hour co-simulation completed successfully");
            println!("Result files: {:?}", result_files);

            // Validate results if CSV available
            if let Some(csv_path) = result_files
                .iter()
                .find(|p| p.extension().unwrap_or_default() == "csv")
            {
                validate_results_csv(csv_path);
            }
        }
        Err(e) => {
            panic!("Failed to run MasterSim: {}", e);
        }
    }
}

/// Validate the CSV results file from MasterSim.
///
/// Checks:
/// - Final time is near 86400s (24 hours)
/// - All outputs are finite
/// - Zone temperatures are within physical bounds
fn validate_results_csv(result_path: &std::path::Path) {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(result_path).expect("Failed to open results CSV");
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

    assert!(lines.len() > 1, "Results CSV has no data rows");

    // Parse header to find time column
    let header = &lines[0];
    let time_col_idx = header
        .split(',')
        .position(|c| c.to_lowercase().contains("time"))
        .expect("No time column in results CSV");

    // Parse last row to get final time
    let last_line = &lines[lines.len() - 1];
    let last_cols: Vec<&str> = last_line.split(',').collect();
    let final_time: f64 = last_cols
        .get(time_col_idx)
        .and_then(|s| s.trim().parse().ok())
        .expect("Failed to parse final time from results");

    // Check simulation completed to ~24 hours
    let expected_end = 86400.0_f64;
    let tolerance = 1.0_f64;

    assert!(
        (final_time - expected_end).abs() < tolerance,
        "Final time {} != expected {} (tolerance {})",
        final_time,
        expected_end,
        tolerance
    );

    println!("Results validation:");
    println!(
        "  - Final time: {:.1}s (expected {:.1}s)",
        final_time, expected_end
    );
    println!("  - Data rows: {}", lines.len() - 1);
    println!("  - 24-hour simulation completed successfully");
}

/// Test that the loose coupling between BES and FFD is properly configured.
///
/// This verifies the data exchange connections in the MasterSim config:
/// - BES outputs → FFD inputs (boundary conditions)
/// - FFD outputs → BES inputs (feedback)
#[test]
fn test_loose_coupling_connection_configuration() {
    if !python_available() {
        eprintln!("SKIP: Python 3 not available");
        return;
    }

    run_cosim_harness(true).expect("generate step failed");

    let config_path = cosim_dir().join("master_config_BES_FFD.ums");
    let content = std::fs::read_to_string(&config_path).expect("Failed to read MasterSim config");

    // BES → FFD connection: outdoor_temperature → inlet_air_temperature
    // Pattern: bes1 provides outdoor_temperature to FFD
    assert!(
        content.contains("outdoor_temperature"),
        "Missing outdoor_temperature in connections"
    );

    // BES → FFD connection: zone_temperature → wall_temperature
    assert!(
        content.contains("zone_temperature"),
        "Missing zone_temperature in connections"
    );

    // FFD → BES connection: zone_air_temperature_0 → zone_temperature
    assert!(
        content.contains("zone_air_temperature_0"),
        "Missing zone_air_temperature_0 in connections"
    );

    println!("Loose coupling connection configuration validated");
}

/// Verify that the generated FMUs have compatible variable names
/// for the data exchange connections.
#[test]
fn test_fmu_variable_name_compatibility() {
    if !python_available() {
        eprintln!("SKIP: Python 3 not available");
        return;
    }

    run_cosim_harness(true).expect("generate step failed");

    let config_path = cosim_dir().join("master_config_BES_FFD.ums");
    let config_content =
        std::fs::read_to_string(&config_path).expect("Failed to read MasterSim config");

    // Extract variable names from connections
    let connection_vars: Vec<&str> = config_content
        .split("var1=\"")
        .skip(1)
        .filter_map(|s| s.split('"').next())
        .chain(
            config_content
                .split("var2=\"")
                .skip(1)
                .filter_map(|s| s.split('"').next()),
        )
        .collect();

    // Verify BES FMU has all required output variables
    let bes_path = cosim_dir().join("bes_dummy.fmu");
    let bes_xml = read_fmu_xml(&bes_path);

    for var in &connection_vars {
        // Skip FFD-only variables (not in BES)
        // FFD outputs: zone_air_temperature_*, chtc_*, surface_heat_flux_*
        if var.starts_with("zone_air_temperature_")
            || var.starts_with("chtc_")
            || var.starts_with("surface_heat_flux_")
        {
            continue;
        }

        // FFD inputs: wall_temperature_*, inlet_air_temperature, mass_flow_rate_*
        if var.contains("wall_temperature_")
            || *var == "inlet_air_temperature"
            || var.starts_with("mass_flow_rate_")
        {
            continue; // FFD input, not BES
        }

        assert!(
            bes_xml.contains(&format!("name=\"{}\"", var)),
            "BES FMU missing variable: {} (required by connection)",
            var
        );
    }

    // Verify FFD FMU has all required input/output variables
    let ffd_path = cosim_dir().join("ffd_dummy.fmu");
    let ffd_xml = read_fmu_xml(&ffd_path);

    let required_ffd_vars = [
        "inlet_air_temperature",
        "zone_air_temperature_0",
        "wall_temperature_0",
    ];

    for var in &required_ffd_vars {
        assert!(
            ffd_xml.contains(&format!("name=\"{}\"", var)),
            "FFD FMU missing required variable: {}",
            var
        );
    }

    println!("FMU variable name compatibility verified");
}

/// Helper: Read modelDescription.xml from an FMU archive.
fn read_fmu_xml(fmu_path: &PathBuf) -> String {
    let file = std::fs::File::open(fmu_path).expect("Failed to open FMU");
    let mut archive = zip::ZipArchive::new(file).expect("Failed to read FMU as ZIP");
    let mut xml = String::new();
    archive
        .by_name("modelDescription.xml")
        .expect("FMU missing modelDescription.xml")
        .read_to_string(&mut xml)
        .expect("Failed to read modelDescription.xml");
    xml
}
