//! Extension trait for recording thermal-simulation diagnostics.
//!
//! The [`SimulationDiagnosticsExt`] trait provides the impure
//! `record_timestep` (needs `&ThermalModel<T>`) and `export_csv`
//! (needs `std::fs`) methods for
//! [`fluxion_core::diagnostics::SimulationDiagnostics`].
//!
//! Placement note (issue #4172): the trait lives in the `sim` layer —
//! not in `fluxion::validation::diagnostics` — because its methods
//! require `crate::sim` types. Homing it here means `src/sim/**` call
//! sites import from `crate::sim`, so no `crate::validation::*`
//! reference appears under `src/sim/**` and the `sim -> validation`
//! cycle baseline genuinely drops. `fluxion::validation::diagnostics`
//! re-exports the trait for compatibility.

use crate::physics::cta::ContinuousTensor;
use crate::sim::engine::ThermalModel;
use fluxion_core::diagnostics::SimulationDiagnostics as BaseSimulationDiagnostics;
use log::{debug, trace};
use std::convert::AsRef;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

pub trait SimulationDiagnosticsExt {
    /// Records data for a single timestep from the given model.
    /// This should be called at the end of step_physics.
    fn record_timestep<T: ContinuousTensor<f64> + AsRef<[f64]>>(
        &mut self,
        hour: usize,
        model: &ThermalModel<T>,
        outdoor_temp: f64,
        ground_temp: f64,
    );
    /// Exports all collected diagnostic data to a CSV file.
    fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>>;
}

impl SimulationDiagnosticsExt for BaseSimulationDiagnostics {
    fn record_timestep<T: ContinuousTensor<f64> + AsRef<[f64]>>(
        &mut self,
        hour: usize,
        model: &ThermalModel<T>,
        outdoor_temp: f64,
        ground_temp: f64,
    ) {
        trace!("Recording diagnostics for hour {}", hour);
        let num_zones = model.hvac.num_zones;
        self.hours.push(hour);

        // Outdoor and Ground temperatures
        self.outdoor_temps.push(outdoor_temp);
        self.ground_temps.push(ground_temp);

        // Zone temperatures
        let zone_temps: Vec<f64> = model.setpoints.temperatures.as_ref().to_vec();
        self.zone_temps.push(zone_temps.clone());

        // Mass temperatures
        let mass_temps: Vec<f64> = model.mass.mass_temperatures.as_ref().to_vec();
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
        let zone_areas: Vec<f64> = model.setpoints.zone_area.as_ref().to_vec();

        // Solar gains (W)
        let solar_watts: Vec<f64> = model
            .solar
            .solar_gains
            .as_ref()
            .iter()
            .zip(zone_areas.iter())
            .map(|(s, a)| s * a)
            .collect();
        self.loads.solar.push(solar_watts);

        // Internal gains (W)
        let internal_watts: Vec<f64> = model
            .setpoints
            .loads
            .as_ref()
            .iter()
            .zip(zone_areas.iter())
            .map(|(l, a)| l * a)
            .collect();
        self.loads.internal.push(internal_watts);

        // HVAC per-zone power (W) - from the temporary buffer
        let hvac_vec = if let Some(ref hvac_tensor) = model.hvac.current_hvac_output {
            hvac_tensor.as_ref().to_vec()
        } else {
            vec![0.0; num_zones]
        };
        self.loads.hvac.push(hvac_vec.clone());

        // Inter-zone transfer: placeholder zeros
        let zero_vec = vec![0.0; num_zones];
        self.loads.inter_zone.push(zero_vec);

        // Infiltration (W): approximate using ACH and zone volume, outdoor temp unknown (use 0)
        let infiltration_ach: Vec<f64> = model.setpoints.infiltration_rate.as_ref().to_vec();
        let ceiling_heights: Vec<f64> = model.setpoints.ceiling_height.as_ref().to_vec();
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

        // Envelope conduction (W): Q = h_tr_em * (T_outdoor - T_mass)
        // h_tr_em is the exterior-to-mass conductance
        let h_tr_em_vec: Vec<f64> = model.conduction.h_tr_em.as_ref().to_vec();
        let mass_temps: Vec<f64> = model.mass.mass_temperatures.as_ref().to_vec();
        let mut conduction_watts = Vec::with_capacity(num_zones);
        for i in 0..num_zones {
            let h_tr_em = h_tr_em_vec.get(i).copied().unwrap_or(0.0);
            let t_mass = mass_temps.get(i).copied().unwrap_or(20.0);
            let delta_t = outdoor_temp - t_mass;
            let watts = h_tr_em * delta_t;
            conduction_watts.push(watts);
        }
        self.loads.conduction.push(conduction_watts);

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
    /// Exports all collected diagnostic data to a CSV file.
    ///
    /// The CSV includes hourly data with columns: hour, zone_temps, mass_temps, surface_temps,
    /// solar, internal, hvac, inter_zone, infiltration. Multiple zones are represented as
    /// comma-separated values within a column.
    fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        debug!("Exporting diagnostics CSV to {:?}", path.as_ref());
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header
        writeln!(
            writer,
            "Hour,Outdoor_Temp,Ground_Temp,Zone_Temps,Mass_Temps,Surface_Temps,Solar_Watts,Internal_Watts,HVAC_Watts,InterZone_Watts,Infiltration_Watts"
        )?;

        // Data rows
        for i in 0..self.hours.len() {
            let hour = self.hours[i];
            let outdoor_temp = self.outdoor_temps.get(i).copied().unwrap_or(0.0);
            let ground_temp = self.ground_temps.get(i).copied().unwrap_or(0.0);
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
            let conduction_str = self
                .loads
                .conduction
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
                "{},{:.2},{:.2},{},{},{},{},{},{},{},{},{}",
                hour,
                outdoor_temp,
                ground_temp,
                zone_temps_str,
                mass_temps_str,
                surface_temps_str,
                solar_str,
                internal_str,
                hvac_str,
                inter_zone_str,
                infiltration_str,
                conduction_str
            )?;
        }

        writer.flush()?;
        debug!("CSV export completed");
        Ok(())
    }
}
