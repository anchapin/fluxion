//! Configurable diagnostic logging with hourly temperature, load, and energy tracking.
//!
//! This module provides structured diagnostics that can be attached to a ThermalModel
//! to collect detailed simulation data for debugging and analysis. Diagnostics are
//! controlled via the `RUST_LOG` environment variable (trace, debug, info, warn, error).
//!
//! # Usage
//!
//! ```
//! use fluxion::sim::engine::ThermalModel;
//! use fluxion::validation::diagnostics::SimulationDiagnostics;
//!
//! let mut model: ThermalModel<VectorField> = ...;
//! let mut diag = SimulationDiagnostics::new(model.num_zones, 8760);
//! model.set_diagnostics(Some(diag));
//!
//! // Run simulation...
//!
//! let diag = model.get_diagnostics().unwrap();
//! diag.print_summary();
//! diag.export_csv("output/diagnostics.csv").unwrap();
//! ```

use crate::physics::cta::ContinuousTensor;
use crate::sim::engine::ThermalModel;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

/// Collected diagnostic data for a single simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationDiagnostics {
    /// Timestamps (hour indices)
    pub hours: Vec<usize>,
    /// Zone temperatures (°C) - indexed by [timestep][zone]
    pub zone_temps: Vec<Vec<f64>>,
    /// Mass temperatures (°C)
    pub mass_temps: Vec<Vec<f64>>,
    /// Surface temperatures (°C) - interior surfaces (estimated)
    pub surface_temps: Vec<Vec<f64>>,
    /// Load breakdown per timestep (Watts)
    pub loads: LoadBreakdown,
    /// Cumulative energy accumulation (kWh)
    pub cumulative_energy: EnergyAccumulation,
}

/// Breakdown of thermal loads at each timestep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBreakdown {
    /// Solar gains per zone (Watts)
    pub solar: Vec<Vec<f64>>,
    /// Internal gains per zone (Watts)
    pub internal: Vec<Vec<f64>>,
    /// HVAC output per zone (Watts, positive=heating, negative=cooling)
    pub hvac: Vec<Vec<f64>>,
    /// Inter-zone transfer per zone (Watts, positive=gain from adjacent zone)
    pub inter_zone: Vec<Vec<f64>>,
    /// Infiltration heat loss per zone (Watts)
    pub infiltration: Vec<Vec<f64>>,
}

/// Energy accumulation over simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyAccumulation {
    /// Cumulative heating energy per zone (kWh)
    pub heating_kwh: Vec<f64>,
    /// Cumulative cooling energy per zone (kWh)
    pub cooling_kwh: Vec<f64>,
    /// Total energy per zone (kWh)
    pub total_kwh: Vec<f64>,
}

impl SimulationDiagnostics {
    /// Creates a new diagnostics collector.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones
    /// * `num_timesteps` - Expected number of timesteps (e.g., 8760 for 1 year)
    pub fn new(num_zones: usize, num_timesteps: usize) -> Self {
        Self {
            hours: Vec::with_capacity(num_timesteps),
            zone_temps: Vec::with_capacity(num_timesteps),
            mass_temps: Vec::with_capacity(num_timesteps),
            surface_temps: Vec::with_capacity(num_timesteps),
            loads: LoadBreakdown {
                solar: Vec::with_capacity(num_timesteps),
                internal: Vec::with_capacity(num_timesteps),
                hvac: Vec::with_capacity(num_timesteps),
                inter_zone: Vec::with_capacity(num_timesteps),
                infiltration: Vec::with_capacity(num_timesteps),
            },
            cumulative_energy: EnergyAccumulation {
                heating_kwh: vec![0.0; num_zones],
                cooling_kwh: vec![0.0; num_zones],
                total_kwh: vec![0.0; num_zones],
            },
        }
    }

    /// Exports all collected diagnostic data to a CSV file.
    ///
    /// The CSV includes hourly data with columns: hour, zone_temps, mass_temps, surface_temps,
    /// solar, internal, hvac, inter_zone, infiltration. Multiple zones are represented as
    /// comma-separated values within a column.
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Exporting diagnostics CSV to {:?}", path.as_ref());
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header
        writeln!(
            writer,
            "Hour,Zone_Temps,Mass_Temps,Surface_Temps,Solar_Watts,Internal_Watts,HVAC_Watts,InterZone_Watts,Infiltration_Watts"
        )?;

        // Data rows
        for i in 0..self.hours.len() {
            let hour = self.hours[i];
            let zone_temps_str = self
                .zone_temps
                .get(i)
                .map(|v| {
                    v.iter()
                        .map(|t| format!("{:.2}", t))
                        .collect::<Vec<_>>()
                        .join(";")
                })
                .unwrap_or_default();
            let mass_temps_str = self
                .mass_temps
                .get(i)
                .map(|v| {
                    v.iter()
                        .map(|t| format!("{:.2}", t))
                        .collect::<Vec<_>>()
                        .join(";")
                })
                .unwrap_or_default();
            let surface_temps_str = self
                .surface_temps
                .get(i)
                .map(|v| {
                    v.iter()
                        .map(|t| format!("{:.2}", t))
                        .collect::<Vec<_>>()
                        .join(";")
                })
                .unwrap_or_default();
            let solar_str = self
                .loads
                .solar
                .get(i)
                .map(|v| {
                    v.iter()
                        .map(|w| format!("{:.2}", w))
                        .collect::<Vec<_>>()
                        .join(";")
                })
                .unwrap_or_default();
            let internal_str = self
                .loads
                .internal
                .get(i)
                .map(|v| {
                    v.iter()
                        .map(|w| format!("{:.2}", w))
                        .collect::<Vec<_>>()
                        .join(";")
                })
                .unwrap_or_default();
            let hvac_str = self
                .loads
                .hvac
                .get(i)
                .map(|v| {
                    v.iter()
                        .map(|w| format!("{:.2}", w))
                        .collect::<Vec<_>>()
                        .join(";")
                })
                .unwrap_or_default();
            let inter_zone_str = self
                .loads
                .inter_zone
                .get(i)
                .map(|v| {
                    v.iter()
                        .map(|w| format!("{:.2}", w))
                        .collect::<Vec<_>>()
                        .join(";")
                })
                .unwrap_or_default();
            let infiltration_str = self
                .loads
                .infiltration
                .get(i)
                .map(|v| {
                    v.iter()
                        .map(|w| format!("{:.2}", w))
                        .collect::<Vec<_>>()
                        .join(";")
                })
                .unwrap_or_default();

            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{}",
                hour,
                zone_temps_str,
                mass_temps_str,
                surface_temps_str,
                solar_str,
                internal_str,
                hvac_str,
                inter_zone_str,
                infiltration_str
            )?;
        }

        writer.flush()?;
        debug!("CSV export completed");
        Ok(())
    }

    /// Prints a summary of the diagnostic data to the console at INFO level.
    pub fn print_summary(&self) {
        info!("=== Simulation Diagnostics Summary ===");
        info!("Total hours recorded: {}", self.hours.len());
        if !self.zone_temps.is_empty() {
            let first = &self.zone_temps[0];
            let last = &self.zone_temps.last().unwrap();
            info!(
                "Zone temperature range: first={:.2}°C, last={:.2}°C",
                first[0], last[0]
            );
        }
        info!("Cumulative energy per zone:");
        for (zone_idx, ((heating, cooling), total)) in self
            .cumulative_energy
            .heating_kwh
            .iter()
            .zip(self.cumulative_energy.cooling_kwh.iter())
            .zip(self.cumulative_energy.total_kwh.iter())
            .enumerate()
        {
            info!(
                "  Zone {}: Heating={:.2} kWh, Cooling={:.2} kWh, Total={:.2} kWh",
                zone_idx, heating, cooling, total
            );
        }
        info!("---------------------------------------");
    }
}

impl Default for SimulationDiagnostics {
    fn default() -> Self {
        Self::new(1, 8760)
    }
}

impl SimulationDiagnostics {
    /// Records data for a single timestep from the given model.
    /// This should be called at the end of step_physics.
    pub fn record_timestep<T: ContinuousTensor<f64> + AsRef<[f64]>>(
        &mut self,
        hour: usize,
        model: &ThermalModel<T>,
    ) {
        trace!("Recording diagnostics for hour {}", hour);
        let num_zones = model.num_zones;
        self.hours.push(hour);

        // Zone temperatures
        let zone_temps: Vec<f64> = model.temperatures.as_ref().to_vec();
        self.zone_temps.push(zone_temps.clone());

        // Mass temperatures
        let mass_temps: Vec<f64> = model.mass_temperatures.as_ref().to_vec();
        self.mass_temps.push(mass_temps.clone());

        // Surface temperatures: simple average placeholder
        let mut surface_est = Vec::with_capacity(num_zones);
        for i in 0..num_zones {
            let tm = mass_temps.get(i).copied().unwrap_or(20.0);
            let ti = zone_temps.get(i).copied().unwrap_or(20.0);
            surface_est.push((tm + ti) / 2.0);
        }
        self.surface_temps.push(surface_est);

        // Loads in Watts
        let zone_areas: Vec<f64> = model.zone_area.as_ref().to_vec();

        // Solar gains (W)
        let solar_watts: Vec<f64> = model
            .solar_gains
            .as_ref()
            .iter()
            .zip(zone_areas.iter())
            .map(|(s, a)| s * a)
            .collect();
        self.loads.solar.push(solar_watts);

        // Internal gains (W)
        let internal_watts: Vec<f64> = model
            .loads
            .as_ref()
            .iter()
            .zip(zone_areas.iter())
            .map(|(l, a)| l * a)
            .collect();
        self.loads.internal.push(internal_watts);

        // HVAC per-zone power (W) - from the temporary buffer
        let hvac_vec = if let Some(ref hvac_tensor) = model.current_hvac_output {
            hvac_tensor.as_ref().to_vec()
        } else {
            vec![0.0; num_zones]
        };
        self.loads.hvac.push(hvac_vec.clone());

        // Inter-zone transfer: placeholder zeros
        let zero_vec = vec![0.0; num_zones];
        self.loads.inter_zone.push(zero_vec);

        // Infiltration (W): approximate using ACH and zone volume, outdoor temp unknown (use 0)
        let infiltration_ach: Vec<f64> = model.infiltration_rate.as_ref().to_vec();
        let ceiling_heights: Vec<f64> = model.ceiling_height.as_ref().to_vec();
        let mut infiltration_watts = Vec::with_capacity(num_zones);
        for i in 0..num_zones {
            let ach = infiltration_ach.get(i).copied().unwrap_or(0.0);
            let floor_area = zone_areas.get(i).copied().unwrap_or(0.0);
            let height = ceiling_heights.get(i).copied().unwrap_or(2.5);
            let volume = floor_area * height;
            let rho = 1.2; // kg/m³
            let cp = 1005.0; // J/kg·K
            let t_zone = zone_temps.get(i).copied().unwrap_or(20.0);
            let delta_t = (0.0 - t_zone).abs(); // outdoor temp unknown
            let watts = (ach / 3600.0) * volume * rho * cp * delta_t;
            infiltration_watts.push(watts);
        }
        self.loads.infiltration.push(infiltration_watts);

        // Update cumulative energy per zone (kWh)
        for i in 0..num_zones {
            let hvac_power = hvac_vec.get(i).copied().unwrap_or(0.0);
            let increment = hvac_power / 1000.0; // kWh for 1 hour
            if increment > 0.0 {
                self.cumulative_energy.heating_kwh[i] += increment;
            } else if increment < 0.0 {
                self.cumulative_energy.cooling_kwh[i] += -increment;
            }
            self.cumulative_energy.total_kwh[i] =
                self.cumulative_energy.heating_kwh[i] + self.cumulative_energy.cooling_kwh[i];
        }

        trace!("Recorded hour {}: {} zones", hour, num_zones);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_diagnostics_new() {
        let diag = SimulationDiagnostics::new(2, 100);
        assert_eq!(diag.hours.len(), 0);
        assert_eq!(diag.zone_temps.len(), 0);
        assert_eq!(diag.mass_temps.len(), 0);
        assert_eq!(diag.surface_temps.len(), 0);
        assert_eq!(diag.loads.solar.len(), 0);
        assert_eq!(diag.loads.internal.len(), 0);
        assert_eq!(diag.loads.hvac.len(), 0);
        assert_eq!(diag.loads.inter_zone.len(), 0);
        assert_eq!(diag.loads.infiltration.len(), 0);
        assert_eq!(diag.cumulative_energy.heating_kwh, vec![0.0; 2]);
        assert_eq!(diag.cumulative_energy.cooling_kwh, vec![0.0; 2]);
        assert_eq!(diag.cumulative_energy.total_kwh, vec![0.0; 2]);
    }

    #[test]
    fn test_simulation_diagnostics_default() {
        let diag = SimulationDiagnostics::default();
        assert_eq!(diag.hours.capacity(), 8760);
        assert_eq!(diag.cumulative_energy.heating_kwh.len(), 1);
    }

    #[test]
    fn test_load_breakdown_clone() {
        let load = LoadBreakdown {
            solar: vec![vec![100.0, 200.0]],
            internal: vec![vec![50.0, 60.0]],
            hvac: vec![vec![300.0, 0.0]],
            inter_zone: vec![vec![10.0, -10.0]],
            infiltration: vec![vec![20.0, 25.0]],
        };
        let cloned = load.clone();
        assert_eq!(cloned.solar[0][0], 100.0);
        assert_eq!(cloned.infiltration[0][1], 25.0);
    }

    #[test]
    fn test_energy_accumulation_clone() {
        let energy = EnergyAccumulation {
            heating_kwh: vec![1.5, 2.0],
            cooling_kwh: vec![0.8, 1.2],
            total_kwh: vec![2.3, 3.2],
        };
        let cloned = energy.clone();
        assert_eq!(cloned.heating_kwh[0], 1.5);
        assert_eq!(cloned.total_kwh[1], 3.2);
    }

    #[test]
    fn test_simulation_diagnostics_clone() {
        let mut diag = SimulationDiagnostics::new(1, 10);
        diag.hours.push(0);
        diag.zone_temps.push(vec![20.0]);
        diag.mass_temps.push(vec![19.0]);
        diag.surface_temps.push(vec![19.5]);
        diag.loads.solar.push(vec![100.0]);
        diag.loads.internal.push(vec![50.0]);
        diag.loads.hvac.push(vec![200.0]);
        diag.loads.inter_zone.push(vec![0.0]);
        diag.loads.infiltration.push(vec![30.0]);

        let cloned = diag.clone();
        assert_eq!(cloned.hours[0], 0);
        assert_eq!(cloned.zone_temps[0][0], 20.0);
        assert_eq!(cloned.loads.solar[0][0], 100.0);
    }

    #[test]
    fn test_simulation_diagnostics_export_csv_single_zone() {
        let mut diag = SimulationDiagnostics::new(1, 10);

        // Manually populate data
        for i in 0..5 {
            diag.hours.push(i);
            diag.zone_temps.push(vec![20.0 + i as f64]);
            diag.mass_temps.push(vec![19.0 + i as f64]);
            diag.surface_temps.push(vec![19.5 + i as f64]);
            diag.loads.solar.push(vec![100.0 + i as f64]);
            diag.loads.internal.push(vec![50.0]);
            diag.loads.hvac.push(vec![200.0]);
            diag.loads.inter_zone.push(vec![0.0]);
            diag.loads.infiltration.push(vec![30.0]);
        }

        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join(format!("fluxion_diag_test_{}.csv", std::process::id()));

        let result = diag.export_csv(&csv_path);
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Header + 5 data rows
        assert_eq!(lines.len(), 6);
        assert!(lines[0].contains("Hour"));
        assert!(lines[0].contains("Zone_Temps"));
        assert!(lines[0].contains("Solar_Watts"));

        // Check data format
        assert!(lines[1].contains("0,"));
        assert!(lines[1].contains("20.00"));

        let _ = std::fs::remove_file(&csv_path);
    }

    #[test]
    fn test_simulation_diagnostics_export_csv_multi_zone() {
        let mut diag = SimulationDiagnostics::new(2, 10);

        for i in 0..3 {
            diag.hours.push(i);
            diag.zone_temps.push(vec![20.0, 18.0]);
            diag.mass_temps.push(vec![19.0, 17.0]);
            diag.surface_temps.push(vec![19.5, 17.5]);
            diag.loads.solar.push(vec![100.0, 80.0]);
            diag.loads.internal.push(vec![50.0, 40.0]);
            diag.loads.hvac.push(vec![200.0, 150.0]);
            diag.loads.inter_zone.push(vec![5.0, -5.0]);
            diag.loads.infiltration.push(vec![30.0, 25.0]);
        }

        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join(format!("fluxion_diag_mz_{}.csv", std::process::id()));

        let result = diag.export_csv(&csv_path);
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines.len(), 4); // Header + 3 rows
                                    // Multi-zone values should be semicolon-separated
        assert!(lines[1].contains("20.00;18.00"));

        let _ = std::fs::remove_file(&csv_path);
    }

    #[test]
    fn test_simulation_diagnostics_export_csv_empty() {
        let diag = SimulationDiagnostics::new(1, 10);

        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join(format!("fluxion_diag_empty_{}.csv", std::process::id()));

        let result = diag.export_csv(&csv_path);
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1); // Header only
        assert!(lines[0].contains("Hour"));

        let _ = std::fs::remove_file(&csv_path);
    }

    #[test]
    fn test_simulation_diagnostics_print_summary() {
        let mut diag = SimulationDiagnostics::new(2, 10);

        for i in 0..5 {
            diag.hours.push(i);
            diag.zone_temps.push(vec![20.0 + i as f64, 18.0 + i as f64]);
            diag.mass_temps.push(vec![19.0, 17.0]);
            diag.surface_temps.push(vec![19.5, 17.5]);
            diag.loads.solar.push(vec![100.0, 80.0]);
            diag.loads.internal.push(vec![50.0, 40.0]);
            diag.loads.hvac.push(vec![200.0, 150.0]);
            diag.loads.inter_zone.push(vec![0.0, 0.0]);
            diag.loads.infiltration.push(vec![30.0, 25.0]);
        }

        diag.cumulative_energy.heating_kwh = vec![1.5, 1.2];
        diag.cumulative_energy.cooling_kwh = vec![0.5, 0.3];
        diag.cumulative_energy.total_kwh = vec![2.0, 1.5];

        // Should not panic
        diag.print_summary();
    }

    #[test]
    fn test_simulation_diagnostics_print_summary_empty() {
        let diag = SimulationDiagnostics::new(1, 10);
        // Should handle empty data gracefully
        diag.print_summary();
    }

    #[test]
    fn test_simulation_diagnostics_serialization() {
        let mut diag = SimulationDiagnostics::new(1, 10);
        diag.hours.push(0);
        diag.zone_temps.push(vec![20.0]);
        diag.mass_temps.push(vec![19.0]);
        diag.surface_temps.push(vec![19.5]);
        diag.loads.solar.push(vec![100.0]);
        diag.loads.internal.push(vec![50.0]);
        diag.loads.hvac.push(vec![200.0]);
        diag.loads.inter_zone.push(vec![0.0]);
        diag.loads.infiltration.push(vec![30.0]);

        let json = serde_json::to_string(&diag).unwrap();
        let deserialized: SimulationDiagnostics = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.hours[0], 0);
        assert_eq!(deserialized.zone_temps[0][0], 20.0);
        assert_eq!(deserialized.loads.solar[0][0], 100.0);
    }

    #[test]
    fn test_simulation_diagnostics_export_csv_with_empty_loads() {
        let mut diag = SimulationDiagnostics::new(1, 10);
        diag.hours.push(0);
        diag.zone_temps.push(vec![20.0]);
        diag.mass_temps.push(vec![19.0]);
        diag.surface_temps.push(vec![19.5]);
        diag.loads.solar.push(vec![]);
        diag.loads.internal.push(vec![]);
        diag.loads.hvac.push(vec![]);
        diag.loads.inter_zone.push(vec![]);
        diag.loads.infiltration.push(vec![]);

        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join(format!(
            "fluxion_diag_empty_loads_{}.csv",
            std::process::id()
        ));
        let result = diag.export_csv(&csv_path);
        assert!(result.is_ok());
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let _ = std::fs::remove_file(&csv_path);
    }

    #[test]
    fn test_simulation_diagnostics_export_csv_missing_timestep() {
        let diag = SimulationDiagnostics::new(1, 10);
        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join(format!("fluxion_diag_missing_{}.csv", std::process::id()));
        let result = diag.export_csv(&csv_path);
        assert!(result.is_ok());
        let content = std::fs::read_to_string(&csv_path).unwrap();
        assert_eq!(content.lines().count(), 1);
        let _ = std::fs::remove_file(&csv_path);
    }

    #[test]
    fn test_simulation_diagnostics_new_capacity() {
        let diag = SimulationDiagnostics::new(3, 500);
        assert_eq!(diag.hours.capacity(), 500);
        assert_eq!(diag.zone_temps.capacity(), 500);
        assert_eq!(diag.cumulative_energy.heating_kwh.len(), 3);
        assert_eq!(diag.cumulative_energy.cooling_kwh.len(), 3);
        assert_eq!(diag.cumulative_energy.total_kwh.len(), 3);
    }

    #[test]
    fn test_load_breakdown_default() {
        let load = LoadBreakdown {
            solar: vec![],
            internal: vec![],
            hvac: vec![],
            inter_zone: vec![],
            infiltration: vec![],
        };
        assert!(load.solar.is_empty());
        assert!(load.hvac.is_empty());
    }

    #[test]
    fn test_energy_accumulation_default() {
        let energy = EnergyAccumulation {
            heating_kwh: vec![],
            cooling_kwh: vec![],
            total_kwh: vec![],
        };
        assert!(energy.heating_kwh.is_empty());
    }

    #[test]
    fn test_record_timestep() {
        use crate::physics::cta::VectorField;

        let mut model = ThermalModel::new(1);
        model.temperatures = VectorField::new(vec![22.0]);
        model.mass_temperatures = VectorField::new(vec![21.0]);
        model.zone_area = VectorField::new(vec![50.0]);
        model.solar_gains = VectorField::new(vec![10.0]);
        model.loads = VectorField::new(vec![5.0]);
        model.current_hvac_output = Some(VectorField::new(vec![1000.0]));
        model.infiltration_rate = VectorField::new(vec![0.5]);
        model.ceiling_height = VectorField::new(vec![2.5]);

        let mut diag = SimulationDiagnostics::new(1, 10);
        diag.record_timestep(0, &model);

        assert_eq!(diag.hours.len(), 1);
        assert_eq!(diag.zone_temps[0][0], 22.0);
        assert_eq!(diag.mass_temps[0][0], 21.0);
        assert_eq!(diag.loads.solar[0][0], 500.0); // 10.0 * 50.0
        assert_eq!(diag.loads.internal[0][0], 250.0); // 5.0 * 50.0
        assert_eq!(diag.loads.hvac[0][0], 1000.0);
        assert!(diag.cumulative_energy.heating_kwh[0] > 0.0);
    }
}
