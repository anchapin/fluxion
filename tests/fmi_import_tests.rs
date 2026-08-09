// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Integration tests for FMI 2.0 FMU import (`FmiMode::Import`) — issue #1708.
//!
//! These tests exercise the full export → re-import → co-simulation-master
//! round-trip:
//!
//! 1. [`FmiExporter`] writes a `.fmu` archive.
//! 2. [`FmiImporter`] (or the free [`import_fmu`] function) reads it back and
//!    rebuilds a [`ThermalModel`] with the correct zone count.
//! 3. [`FmuCoSimulationMaster`] drives the re-imported model one `doStep` at
//!    a time, and the output matches a directly-constructed `ThermalModel`
//!    driven with the same weather within 0.1 % (acceptance criterion #2).

use fluxion::interop::fmi::{
    import_fmu, FmiConfig, FmiExporter, FmiImporter, FmuCoSimulationMaster, FmuInputs,
    ZoneVariables,
};
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use std::path::Path;

/// Synthesize a simple 24-step "cold day" weather trace (Kelvin).
fn cold_day_weather() -> Vec<f64> {
    // Outdoor temperature oscillating between -5 °C and +5 °C around a
    // 268.15 K mean, in Kelvin.
    (0..24)
        .map(|h| 268.15 + 5.0 * (((h as f64) / 24.0) * std::f64::consts::TAU).sin())
        .collect()
}

#[test]
fn import_fmu_three_zone_has_correct_zone_count() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("fluxion_three_zone.fmu");
    FmiExporter::new()
        .with_zones(vec![
            ZoneVariables::new("zone"),
            ZoneVariables::new("bedroom"),
            ZoneVariables::new("kitchen"),
        ])
        .export_fmu(&out)
        .expect("export_fmu");

    // Acceptance criterion #1: import_fmu produces a ThermalModel with N
    // zones matching the exported FMU.
    let model: ThermalModel<VectorField> = import_fmu(&out).expect("import_fmu");
    assert_eq!(model.num_zones, 3);
}

#[test]
fn import_fmu_preserves_metadata_and_timestep() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("ts600.fmu");

    let mut cfg = FmiConfig::default();
    cfg.communication_timestep = 600.0;
    cfg.model_name = "RoundTripModel".to_string();
    FmiExporter::with_config(cfg)
        .unwrap()
        .export_fmu(&out)
        .expect("export_fmu");

    let fmu = FmiImporter::new().import(&out).expect("import");
    let desc = &fmu.description;
    assert_eq!(desc.model_name, "RoundTripModel");
    assert_eq!(desc.fmi_version, "2.0");
    assert_eq!(fmu.zone_count(), 1);
    assert!((fmu.communication_timestep() - 600.0).abs() < 1e-9);
    assert_eq!(fmu.thermal_model().num_zones, 1);
}

#[test]
fn reimport_round_trip_matches_direct_model_within_tolerance() {
    // Acceptance criterion #2: an exported FMU re-imported and simulated
    // must match a directly-constructed ThermalModel within 0.1 %.
    let n_steps = 24usize;
    let dt = 3600.0_f64;
    let weather = cold_day_weather();

    // Direct (reference) model.
    let mut direct = ThermalModel::<VectorField>::new(1);

    // Export → re-import path.
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("roundtrip.fmu");
    FmiExporter::new().export_fmu(&out).expect("export");
    let fmu = FmiImporter::new().import(&out).expect("import");
    let mut master = FmuCoSimulationMaster::from_imported(fmu);

    let mut max_rel_err = 0.0_f64;
    for t in 0..n_steps {
        let outdoor_k = weather[t];
        let outdoor_c = outdoor_k - 273.15;

        direct.step_physics(t, outdoor_c, dt);
        let direct_zone_k = direct.temperatures.as_ref()[0] + 273.15;

        let imported_out = master.do_step(
            FmuInputs {
                outdoor_temperature: outdoor_k,
                ..Default::default()
            },
            Some(dt),
        );

        // Issue #2459: do_step now returns one FmuOutputs per zone.
        assert_eq!(imported_out.len(), 1, "single-zone FMU ⇒ 1 output");
        let denom = direct_zone_k.abs().max(1e-9);
        let rel_err = (imported_out[0].zone_temperature - direct_zone_k).abs() / denom;
        max_rel_err = max_rel_err.max(rel_err);
    }

    // 0.1 % tolerance per acceptance criterion #2.
    assert!(
        max_rel_err < 1e-3,
        "re-import round-trip rel error {max_rel_err:.3e} exceeds 0.1 %"
    );
}

#[test]
fn do_step_advances_time_and_reports_outputs() {
    // Acceptance criterion #3: fmi2DoStep calls ThermalModel::step_physics
    // with correct weather inputs per timestep.
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("dostep.fmu");
    FmiExporter::new().export_fmu(&out).expect("export");
    let fmu = FmiImporter::new().import(&out).expect("import");
    let mut master = FmuCoSimulationMaster::from_imported(fmu);

    assert_eq!(master.current_time(), 0.0);

    let o = master.do_step(
        FmuInputs {
            outdoor_temperature: 263.15, // -10 °C
            ..Default::default()
        },
        Some(3600.0),
    );

    assert!((master.current_time() - 3600.0).abs() < 1e-9);
    assert_eq!(o.len(), 1, "single-zone FMU ⇒ 1 output");
    let zone_out = &o[0];
    assert!(zone_out.zone_temperature.is_finite());
    assert!(zone_out.zone_temperature > 200.0 && zone_out.zone_temperature < 320.0);
    assert!(zone_out.heating_load >= 0.0);
    assert!(zone_out.cooling_load >= 0.0);
}

#[test]
fn import_rejects_nonexistent_path() {
    let res = import_fmu(Path::new("/no/such/missing.fmu"));
    assert!(res.is_err());
}
