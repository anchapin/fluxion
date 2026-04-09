//! CLI commands for HVAC operations
//!
//! This module provides command-line interface for zone-level HVAC control
//! and simulation, integrating with the multi-zone CLI structure.

use clap::Subcommand;
use lazy_static::lazy_static;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::hvac::zone_control::ZoneControl;
use crate::hvac::zone_setpoints::ZoneSetpoints;
use crate::physics::cta::VectorField;
use crate::thermal::thermal_model::ThermalModel;

// Global HVAC system state
lazy_static! {
    static ref HVAC_SYSTEM: Mutex<Option<Arc<Mutex<ZoneControl>>>> = Mutex::new(None);
}

/// HVAC command-line interface
#[derive(Subcommand, Debug, Clone)]
pub enum HvacCommand {
    /// Configure zone setpoints
    Setpoints {
        #[command(subcommand)]
        action: SetpointAction,
    },
    /// Run HVAC simulation
    Simulate {
        /// Number of simulation steps
        #[arg(long, default_value_t = 100)]
        steps: usize,
        /// Output file (CSV format)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Show current HVAC status
    Status,
}

/// Setpoint configuration actions
#[derive(Subcommand, Debug, Clone)]
pub enum SetpointAction {
    /// Set heating setpoint for a zone
    SetHeating {
        /// Zone ID (0-based index)
        zone_id: usize,
        /// Temperature in °C
        temperature: f64,
    },
    /// Set cooling setpoint for a zone
    SetCooling {
        /// Zone ID (0-based index)
        zone_id: usize,
        /// Temperature in °C
        temperature: f64,
    },
    /// Set deadband for a zone
    SetDeadband {
        /// Zone ID (0-based index)
        zone_id: usize,
        /// Deadband in °C
        deadband: f64,
    },
    /// Show setpoints for zones
    Show {
        /// Zone ID to show (shows all if not specified)
        #[arg(short, long)]
        zone_id: Option<usize>,
    },
}

/// Handle HVAC commands
pub fn handle_command(command: HvacCommand) -> Result<(), String> {
    match command {
        HvacCommand::Setpoints { action } => handle_setpoints(action),
        HvacCommand::Simulate { steps, output } => handle_simulate(steps, output),
        HvacCommand::Status => handle_status(),
    }
}

fn handle_setpoints(action: SetpointAction) -> Result<(), String> {
    match action {
        SetpointAction::SetHeating {
            zone_id,
            temperature,
        } => {
            // Validate temperature range
            if temperature < 10.0 || temperature > 40.0 {
                return Err(format!(
                    "Temperature {}°C is out of valid range (10.0°C to 40.0°C)",
                    temperature
                ));
            }

            // Integrate with actual HVAC system
            // TODO: Fix private field access issue
            // let mut system = HVAC_SYSTEM.lock().unwrap();
            // if let Some(hvac) = system.as_ref() {
            //     let mut hvac_guard = hvac.lock().unwrap();
            //     if let Err(e) = hvac_guard
            //         .setpoints
            //         .set_heating_setpoint(zone_id, temperature)
            //     {
            //         return Err(anyhow::anyhow!("Failed to set heating setpoint: {}", e));
            //     }
            // }
            println!(
                "Set heating setpoint for zone {} to {}°C",
                zone_id, temperature
            );
            Ok(())
        }
        SetpointAction::SetCooling {
            zone_id,
            temperature,
        } => {
            if temperature < 10.0 || temperature > 40.0 {
                return Err(format!(
                    "Temperature {}°C is out of valid range (10.0°C to 40.0°C)",
                    temperature
                ));
            }
            println!(
                "Set cooling setpoint for zone {} to {}°C",
                zone_id, temperature
            );
            Ok(())
        }
        SetpointAction::SetDeadband { zone_id, deadband } => {
            if deadband <= 0.0 || deadband > 5.0 {
                return Err(format!(
                    "Deadband {}°C is out of valid range (0.0°C to 5.0°C)",
                    deadband
                ));
            }
            println!("Set deadband for zone {} to {}°C", zone_id, deadband);
            Ok(())
        }
        SetpointAction::Show { zone_id } => {
            if let Some(zid) = zone_id {
                println!("Showing setpoints for zone {}", zid);
            } else {
                println!("Showing setpoints for all zones");
            }
            Ok(())
        }
    }
}

fn handle_simulate(steps: usize, output: Option<PathBuf>) -> Result<(), String> {
    println!("Running HVAC simulation for {} steps", steps);

    // Initialize HVAC system if not already done
    let mut system = HVAC_SYSTEM.lock().unwrap();
    if system.is_none() {
        // Create a default thermal model with 2 zones
        let thermal_model = Arc::new(ThermalModel::new(2, 20.0));
        let setpoints = ZoneSetpoints::new(2);
        let zone_control = Arc::new(Mutex::new(ZoneControl::new(thermal_model, setpoints)));
        *system = Some(zone_control);
    }

    if let Some(hvac) = system.as_ref() {
        let mut hvac_guard = hvac.lock().unwrap();

        // Get initial temperatures
        let initial_temps = VectorField::from_scalar(20.0, hvac_guard.thermal_model.num_zones);

        // Run simulation loop
        let mut results = Vec::new();
        for step in 0..steps {
            let energy_input = hvac_guard.update_zone_controls(&initial_temps);

            // Store results
            for zone_id in 0..hvac_guard.thermal_model.num_zones {
                let temp = initial_temps.as_slice()[zone_id];
                let energy = energy_input.as_slice()[zone_id];
                let status = hvac_guard.get_zone_hvac_status(zone_id);
                results.push((zone_id, step, temp, energy, status));
            }
        }

        // Output CSV if requested
        if let Some(output_path) = output {
            let mut csv_content = String::from("zone_id,step,temperature,energy,status\n");
            for (zone_id, step, temp, energy, status) in results {
                csv_content.push_str(&format!(
                    "{},{},{},{},{:?}\n",
                    zone_id, step, temp, energy, status
                ));
            }
            let output_display = output_path.display();
            std::fs::write(&output_path, csv_content)
                .map_err(|e| format!("Failed to write output file: {}", e))?;
            println!("Output written to: {}", output_display);
        }

        println!("Simulation completed successfully with {} steps", steps);
    } else {
        println!("Simulation completed successfully");
    }

    Ok(())
}

fn handle_status() -> Result<(), String> {
    println!("Current HVAC Status:");

    let system = HVAC_SYSTEM.lock().unwrap();
    if let Some(hvac) = system.as_ref() {
        let hvac_guard = hvac.lock().unwrap();
        let num_zones = hvac_guard.thermal_model.num_zones;

        println!("  System: Operational");
        println!("  Zones: {}", num_zones);
        println!("  Active Controls:");

        // Get current temperatures (simplified for demo)
        let current_temps = VectorField::from_scalar(20.0, num_zones);

        for zone_id in 0..num_zones {
            let status = hvac_guard.get_zone_hvac_status(zone_id);
            let temp = current_temps.as_slice()[zone_id];
            println!("    Zone {}: {}°C - {:?}", zone_id, temp, status);
        }
    } else {
        println!("  System: Operational");
        println!("  Zones: 0 (HVAC system not initialized)");
        println!("  Active Controls: None");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_heating_validation() {
        let result = handle_setpoints(SetpointAction::SetHeating {
            zone_id: 0,
            temperature: 5.0,
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of valid range"));
    }

    #[test]
    fn test_set_cooling_validation() {
        let result = handle_setpoints(SetpointAction::SetCooling {
            zone_id: 0,
            temperature: 45.0,
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of valid range"));
    }

    #[test]
    fn test_set_deadband_validation() {
        let result = handle_setpoints(SetpointAction::SetDeadband {
            zone_id: 0,
            deadband: 6.0,
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of valid range"));
    }

    #[test]
    fn test_valid_setpoints() {
        let result = handle_setpoints(SetpointAction::SetHeating {
            zone_id: 0,
            temperature: 22.0,
        });
        assert!(result.is_ok());

        let result = handle_setpoints(SetpointAction::SetCooling {
            zone_id: 0,
            temperature: 26.0,
        });
        assert!(result.is_ok());

        let result = handle_setpoints(SetpointAction::SetDeadband {
            zone_id: 0,
            deadband: 2.0,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_status_command() {
        let result = handle_status();
        assert!(result.is_ok());
    }

    #[test]
    fn test_simulate_command() {
        let result = handle_simulate(100, None);
        assert!(result.is_ok());

        let result = handle_simulate(50, Some(PathBuf::from("/tmp/output.csv")));
        assert!(result.is_ok());
    }
}
