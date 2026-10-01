//! Configurable diagnostic logging with hourly temperature, load, and energy tracking.
//!
//! This module provides structured diagnostics that can be attached to a ThermalModel
//! to collect detailed simulation data for debugging and analysis. Diagnostics are
//! controlled via the `RUST_LOG` environment variable (trace, debug, info, warn, error).
//!
//! The core data structs ([`SimulationDiagnostics`], [`LoadBreakdown`],
//! [`EnergyAccumulation`]) are hoisted to [`fluxion_core::diagnostics`] (issue #4172)
//! to resolve the name collision with [`fluxion_core::error::SimulationDiagnostics`]
//! (now renamed to [`fluxion_core::error::DivergenceDiagnostics`]). The
//! IO-dependent methods (`record_timestep` requiring `&ThermalModel<T>` and
//! `export_csv` requiring std::fs) live in the [`SimulationDiagnosticsExt`]
//! extension trait, homed in [`crate::sim::diagnostics_ext`] (re-exported
//! here) so `src/sim/**` call sites introduce no `crate::validation::*`
//! references — the `sim -> validation` cycle baseline genuinely drops.
//!
//! # Usage
//!
//! ```rust,no_run
//! use fluxion::sim::engine::ThermalModel;
//! use fluxion::physics::cta::VectorField;
//! use fluxion::validation::diagnostics::SimulationDiagnostics;
//! use fluxion::validation::diagnostics::SimulationDiagnosticsExt;
//!
//! // Construct a model (single-zone example; see ThermalModel docs for the
//! // full from_spec / from_spec_with_selector constructors).
//! let mut model: ThermalModel<VectorField> = ThermalModel::<VectorField>::new(1);
//! let num_zones = model.hvac.num_zones;
//! let mut diag = SimulationDiagnostics::new(num_zones, 8760);
//! model.set_diagnostics(Some(diag));
//!
//! // Run simulation...
//!
//! let diag = model.get_diagnostics().unwrap();
//! diag.print_summary();
//! let _ = diag.export_csv("output/diagnostics.csv");
//! ```

// Re-export the hoisted data structs from fluxion-core (issue #4172).
pub use fluxion_core::diagnostics::EnergyAccumulation;
pub use fluxion_core::diagnostics::LoadBreakdown;
pub use fluxion_core::diagnostics::SimulationDiagnostics;

// Re-export the sim-layer extension trait (issue #4172). The trait
// lives in `crate::sim::diagnostics_ext` so `src/sim/**` call sites do
// not introduce `crate::validation::*` references (cycle guard).
pub use crate::sim::diagnostics_ext::SimulationDiagnosticsExt;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::engine::ThermalModel;

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
            diag.loads.conduction.push(vec![40.0]);
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
            diag.loads.conduction.push(vec![35.0, 30.0]);
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
        diag.loads.conduction.push(vec![]);

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
    fn test_record_timestep() {
        use crate::physics::cta::VectorField;

        let mut model = ThermalModel::new(1);
        model.setpoints.temperatures = VectorField::new(vec![22.0]);
        model.mass.mass_temperatures = VectorField::new(vec![21.0]);
        model.setpoints.zone_area = VectorField::new(vec![50.0]);
        model.solar.solar_gains = VectorField::new(vec![10.0]);
        model.setpoints.loads = VectorField::new(vec![5.0]);
        model.hvac.current_hvac_output = Some(VectorField::new(vec![1000.0]));
        model.setpoints.infiltration_rate = VectorField::new(vec![0.5]);
        model.setpoints.ceiling_height = VectorField::new(vec![2.5]);
        model.conduction.h_tr_em = VectorField::new(vec![10.0]);

        let mut diag = SimulationDiagnostics::new(1, 10);
        SimulationDiagnosticsExt::record_timestep(&mut diag, 0, &model, -5.0, 10.0);

        assert_eq!(diag.hours.len(), 1);
        assert_eq!(diag.outdoor_temps[0], -5.0);
        assert_eq!(diag.ground_temps[0], 10.0);
        assert_eq!(diag.zone_temps[0][0], 22.0);
        assert_eq!(diag.mass_temps[0][0], 21.0);
        assert_eq!(diag.loads.solar[0][0], 500.0); // 10.0 * 50.0
        assert_eq!(diag.loads.internal[0][0], 250.0); // 5.0 * 50.0
        assert_eq!(diag.loads.hvac[0][0], 1000.0);
        // Conduction: h_tr_em * (outdoor_temp - mass_temp) = 10.0 * (-5.0 - 21.0) = -260.0 W
        assert_eq!(diag.loads.conduction[0][0], -260.0);
        assert!(diag.cumulative_energy.heating_kwh[0] > 0.0);
    }
}
