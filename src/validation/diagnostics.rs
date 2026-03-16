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
