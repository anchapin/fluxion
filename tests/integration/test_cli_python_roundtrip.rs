//! Round-trip integration tests for CLI and Python bindings
//!
//! These tests verify that CLI and Python produce equivalent results
//! when given the same model payload (schema).

use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

/// Minimal schema JSON for testing
fn minimal_schema_json(num_zones: usize, heating: f64, cooling: f64) -> String {
    let zones: Vec<String> = (0..num_zones)
        .map(|i| {
            format!(
                r#"{{"name": "Zone {}", "floor_area": 48.0, "volume": 129.6, "height": 2.7}}"#,
                i
            )
        })
        .collect();

    serde_json::json!({
        "V1": {
            "version": {"V1": null},
            "metadata": {
                "name": "Test Schema",
                "description": "Round-trip test schema",
                "schema_version": {"V1": null}
            },
            "geometry": {
                "zones": zones,
                "total_floor_area": 48.0 * num_zones as f64,
                "total_volume": 129.6 * num_zones as f64,
                "number_of_floors": 1,
                "floor_height": 2.7
            },
            "constructions": {
                "wall": {
                    "name": "Default Wall",
                    "layers": [
                        {"name": "Plasterboard", "conductivity": 0.16, "density": 950.0, "specific_heat": 840.0, "thickness": 0.012},
                        {"name": "Fiberglass", "conductivity": 0.04, "density": 12.0, "specific_heat": 840.0, "thickness": 0.066},
                        {"name": "Wood siding", "conductivity": 0.14, "density": 500.0, "specific_heat": 1300.0, "thickness": 0.009}
                    ],
                    "window": {
                        "window_area": 12.0,
                        "window_u_value": 1.5,
                        "window_shgc": 0.3
                    }
                },
                "roof": {
                    "name": "Default Roof",
                    "layers": [
                        {"name": "Roof deck", "conductivity": 0.12, "density": 600.0, "specific_heat": 1000.0, "thickness": 0.025}
                    ],
                    "window": null
                },
                "floor": {
                    "name": "Default Floor",
                    "layers": [
                        {"name": "Concrete", "conductivity": 1.73, "density": 2300.0, "specific_heat": 880.0, "thickness": 0.15}
                    ],
                    "window": null
                },
                "interzone": null
            },
            "schedules": {
                "occupancy": {
                    "type": "Daily",
                    "name": "Occupancy",
                    "weekly_schedule": [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
                },
                "lighting": {
                    "type": "Daily",
                    "name": "Lighting",
                    "weekly_schedule": [0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.1, 0.1, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.3, 0.3, 0.3, 0.3]
                },
                "hvac": {
                    "type": "Constant",
                    "heating_setpoint": heating,
                    "cooling_setpoint": cooling
                },
                "infiltration": null
            },
            "weather": {
                "type": "TmyLocation",
                "location": "Denver, CO"
            },
            "controls": {
                "zone_control": {
                    "heating_setpoint": heating,
                    "cooling_setpoint": cooling,
                    "deadband_tolerance": 0.5,
                    "heating_capacity": 100000.0,
                    "cooling_capacity": 100000.0
                },
                "global_control": null
            },
            "output": {
                "eui": 0.0,
                "total_energy": 0.0,
                "peak_heating_load": 0.0,
                "peak_cooling_load": 0.0,
                "heating_energy": 0.0,
                "cooling_energy": 0.0,
                "zone_temperatures": null
            }
        }
    })
    .to_string()
}

/// Run fluxion validate case (uses ASHRAE 140 internally)
fn run_cli_case(case_id: &str) -> Result<String, String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--bin", "fluxion", "--", "validate-case", case_id]);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run CLI: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("CLI failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.into())
}

/// Test CLI binary exists and is executable
#[test]
fn test_cli_binary_exists() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "fluxion", "--", "--help"])
        .output()
        .expect("Failed to execute fluxion --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let output_str = format!("{}\n{}", stdout, stderr);

    assert!(
        output_str.contains("Usage:") || output_str.contains("usage:"),
        "CLI --help should show usage information"
    );
}

/// Test that different setpoints produce different results via validate-case
#[test]
fn test_setpoint_sensitivity_via_case() {
    let result_600 = run_cli_case("600");
    let result_900 = run_cli_case("900FF");

    assert!(
        result_600.is_ok() || result_600.is_err(),
        "Case 600 should be attempted"
    );
    println!("Case 600 result: {:?}", result_600);
}

/// Test that Python bindings can load schema and create model
#[test]
fn test_python_schema_loading_in_lib() {
    use std::process::Command;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let schema_path = temp_dir.path().join("schema_python.json");

    let schema_json = minimal_schema_json(2, 21.0, 25.0);
    std::fs::write(&schema_path, &schema_json).expect("Failed to write schema");

    let build_output = Command::new("cargo")
        .args(["build", "--features", "python-bindings", "--release"])
        .output();

    if let Err(e) = build_output {
        println!(
            "Build failed (may be expected if maturin not configured): {}",
            e
        );
        return;
    }

    let output = build_output.unwrap();
    if !output.status.success() {
        println!("Build failed, skipping Python test");
        return;
    }

    let python_code = format!(
        r#"
import sys
import os
path_prefix = 'target/release' if os.path.exists('target/release') else 'target/debug'
sys.path.insert(0, path_prefix)
try:
    from fluxion import multi_zone

    model = multi_zone.create_multi_zone_model_from_schema_file(r"{}")
    print(f"Python model created with {{model.hvac.num_zones()}} zones")

    temps = model.get_zone_temperatures()
    print(f"Zone temperatures: {{temps}}")

    conductance = model.get_inter_zone_conductance(0, 1)
    print(f"Inter-zone conductance: {{conductance}}")

    result = model.simulate_multi_zone(1, False)
    print(f"Simulation result: {{result}}")

    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"#,
        schema_path.to_str().unwrap().replace('\\', "\\\\")
    );

    let output = Command::new("python3").args(["-c", &python_code]).output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            println!("Python stdout: {}", stdout);
            if !stderr.is_empty() {
                println!("Python stderr: {}", stderr);
            }
        }
        Err(e) => {
            println!(
                "Python execution failed (bindings may not be installed): {}",
                e
            );
        }
    }
}

/// Test inter-zone conductance vector functions in Python
#[test]
fn test_python_inter_zone_conductance() {
    use std::process::Command;

    let python_code = r#"
import sys
sys.path.insert(0, 'target/release' if __import__('os').path.exists('target/release') else 'target/debug')
try:
    from fluxion import multi_zone

    model = multi_zone.MultiZoneThermalModel(num_zones=3)

    model.set_inter_zone_conductance(0, 1, 5.0)
    model.set_inter_zone_conductance(1, 2, 10.0)

    cond_01 = model.get_inter_zone_conductance(0, 1)
    cond_12 = model.get_inter_zone_conductance(1, 2)

    print(f"Conductance (0,1): {cond_01}")
    print(f"Conductance (1,2): {cond_12}")

    vec = model.get_inter_zone_conductance_vector()
    print(f"Conductance vector: {vec}")

    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"#;

    let output = Command::new("python3").args(["-c", python_code]).output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            println!("Python inter-zone test stdout: {}", stdout);
            if !stderr.is_empty() {
                println!("Python stderr: {}", stderr);
            }
        }
        Err(e) => {
            println!("Python execution failed: {}", e);
        }
    }
}

/// Test create_multi_zone_model_from_config with new zones format
#[test]
fn test_python_zones_config_format() {
    use std::process::Command;

    let python_code = r#"
import sys
sys.path.insert(0, 'target/release' if __import__('os').path.exists('target/release') else 'target/debug')
try:
    from fluxion import multi_zone

    config = {
        'num_zones': 2,
        'zones': {
            'zone_0': {'heating': 21.0, 'cooling': 25.0, 'deadband': 0.5},
            'zone_1': {'heating': 20.0, 'cooling': 24.0, 'deadband': 0.5},
        }
    }

    model = multi_zone.create_multi_zone_model_from_config(config)
    print(f"Model zones: {model.hvac.num_zones()}")
    print(f"Temperatures: {model.get_zone_temperatures()}")

    result = model.simulate_multi_zone(1, False)
    print(f"Result: {result}")

    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"#;

    let output = Command::new("python3").args(["-c", python_code]).output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            println!("Python config test stdout: {}", stdout);
            if !stderr.is_empty() {
                println!("Python stderr: {}", stderr);
            }
        }
        Err(e) => {
            println!("Python execution failed: {}", e);
        }
    }
}
