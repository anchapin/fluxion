//! Validation telemetry accumulator for the thermal simulation engine.
//!
//! Hoisted from `fluxion::validation::diagnostics` (issue #4172) to break
//! the name collision with [`fluxion_core::error::SimulationDiagnostics`].
//! Only the pure data structs and `new`/`print_summary` live here — the
//! `record_timestep` (needs `&ThermalModel<T>`) and `export_csv` (std::fs::io)
//! methods remain in `fluxion::validation::diagnostics` as an extension trait.
//!
//! [`fluxion_core::error::SimulationDiagnostics`]: crate::error

use log::info;
use serde::{Deserialize, Serialize};

/// Collected diagnostic data for a single simulation run.
///
/// Telemetry accumulator populated by [`record_timestep`] calls during the
/// simulation loop.
///
/// [`record_timestep`]: fluxion::validation::diagnostics::SimulationDiagnosticsExt::record_timestep
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
    /// Outdoor temperatures (°C)
    pub outdoor_temps: Vec<f64>,
    /// Ground temperatures (°C)
    pub ground_temps: Vec<f64>,
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
    /// Envelope conduction per zone (Watts, positive=heat loss to exterior)
    pub conduction: Vec<Vec<f64>>,
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
            outdoor_temps: Vec::with_capacity(num_timesteps),
            ground_temps: Vec::with_capacity(num_timesteps),
            loads: LoadBreakdown {
                solar: Vec::with_capacity(num_timesteps),
                internal: Vec::with_capacity(num_timesteps),
                hvac: Vec::with_capacity(num_timesteps),
                inter_zone: Vec::with_capacity(num_timesteps),
                infiltration: Vec::with_capacity(num_timesteps),
                conduction: Vec::with_capacity(num_timesteps),
            },
            cumulative_energy: EnergyAccumulation {
                heating_kwh: vec![0.0; num_zones],
                cooling_kwh: vec![0.0; num_zones],
                total_kwh: vec![0.0; num_zones],
            },
        }
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
            conduction: vec![vec![30.0, 35.0]],
        };
        let cloned = load.clone();
        assert_eq!(cloned.solar[0][0], 100.0);
        assert_eq!(cloned.infiltration[0][1], 25.0);
        assert_eq!(cloned.conduction[0][0], 30.0);
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
        diag.loads.conduction.push(vec![40.0]);

        let cloned = diag.clone();
        assert_eq!(cloned.hours[0], 0);
        assert_eq!(cloned.zone_temps[0][0], 20.0);
        assert_eq!(cloned.loads.solar[0][0], 100.0);
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
            diag.loads.conduction.push(vec![35.0, 30.0]);
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
        diag.loads.conduction.push(vec![40.0]);

        let json = serde_json::to_string(&diag).unwrap();
        let deserialized: SimulationDiagnostics = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.hours[0], 0);
        assert_eq!(deserialized.zone_temps[0][0], 20.0);
        assert_eq!(deserialized.loads.solar[0][0], 100.0);
    }

    #[test]
    fn test_load_breakdown_default() {
        let load = LoadBreakdown {
            solar: vec![],
            internal: vec![],
            hvac: vec![],
            inter_zone: vec![],
            infiltration: vec![],
            conduction: vec![],
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
    fn test_simulation_diagnostics_new_capacity() {
        let diag = SimulationDiagnostics::new(3, 500);
        assert_eq!(diag.hours.capacity(), 500);
        assert_eq!(diag.zone_temps.capacity(), 500);
        assert_eq!(diag.cumulative_energy.heating_kwh.len(), 3);
        assert_eq!(diag.cumulative_energy.cooling_kwh.len(), 3);
        assert_eq!(diag.cumulative_energy.total_kwh.len(), 3);
    }

    /// Regression test for issue #4172: verify that the validation telemetry
    /// `SimulationDiagnostics` (from `fluxion_core::diagnostics`) is serde-
    /// distinguishable from `fluxion_core::error::SimulationDiagnostics`
    /// (now `DivergenceDiagnostics`). Both types share the name but have
    /// incompatible field sets; they must not deserialize into each other.
    #[test]
    fn test_simulation_diagnostics_distinguishable_from_divergence_diagnostics() {
        use crate::error::DivergenceDiagnostics;

        // Serialize the telemetry accumulator type
        let telemetry_diag = SimulationDiagnostics::new(1, 1);
        let telemetry_json = serde_json::to_string(&telemetry_diag).unwrap();

        // Serialize a divergence diagnostics payload
        let divergence_diag = DivergenceDiagnostics {
            failing_timestep: 0,
            failing_zone: None,
            max_residual_pct: 0.0,
            last_known_good_timestep: 0,
        };
        let divergence_json = serde_json::to_string(&divergence_diag).unwrap();

        // Their field sets must differ (telemetry has "hours", divergence has "failing_timestep")
        let telemetry_value: serde_json::Value = serde_json::from_str(&telemetry_json).unwrap();
        let telemetry_fields: std::collections::HashSet<_> =
            telemetry_value.as_object().unwrap().keys().collect();

        let divergence_value: serde_json::Value = serde_json::from_str(&divergence_json).unwrap();
        let divergence_fields: std::collections::HashSet<_> =
            divergence_value.as_object().unwrap().keys().collect();

        assert_ne!(
            telemetry_fields, divergence_fields,
            "telemetry and divergence SimulationDiagnostics must have different field sets"
        );

        // Cross-deserialization must fail (they have incompatible schemas)
        let result_telemetry: Result<DivergenceDiagnostics, _> =
            serde_json::from_str(&telemetry_json);
        let result_divergence: Result<SimulationDiagnostics, _> =
            serde_json::from_str(&divergence_json);

        assert!(
            result_telemetry.is_err(),
            "telemetry JSON must not deserialize into DivergenceDiagnostics"
        );
        assert!(
            result_divergence.is_err(),
            "divergence JSON must not deserialize into SimulationDiagnostics"
        );
    }
}
