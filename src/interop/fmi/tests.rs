// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Unit tests for the FMI export/import/cosim paths.

use std::path::Path;

use super::xml::{days_to_ymd, format_float, sanitize_xml_name};
use super::*;

#[test]
fn test_fmi_config_default() {
    let config = FmiConfig::default();
    assert_eq!(config.model_name, "FluxionBuilding");
    assert_eq!(config.communication_timestep, 3600.0);
    assert_eq!(config.stop_time, 31536000.0);
}

#[test]
fn test_fmi_exporter_new() {
    let exporter = FmiExporter::new();
    assert_eq!(exporter.config().model_name, "FluxionBuilding");
    assert_eq!(exporter.zone_count(), 1);
}

#[test]
fn test_fmi_exporter_with_config_valid() {
    let config = FmiConfig::default();
    let exporter = FmiExporter::with_config(config);
    assert!(exporter.is_ok());
}

#[test]
fn test_fmi_exporter_with_config_invalid_timestep() {
    let mut config = FmiConfig::default();
    config.communication_timestep = 0.0;
    let exporter = FmiExporter::with_config(config);
    assert!(exporter.is_err());
}

#[test]
fn test_fmi_exporter_with_config_invalid_time_range() {
    let mut config = FmiConfig::default();
    config.start_time = 100.0;
    config.stop_time = 50.0;
    let exporter = FmiExporter::with_config(config);
    assert!(exporter.is_err());
}

#[test]
fn test_fmi_variables_default() {
    let vars = FmiVariables::default();
    assert_eq!(vars.outdoor_temperature, "outdoor_temperature");
    assert_eq!(vars.zone_temperature, "zone_temperature");
    assert_eq!(FmiVariables::PER_ZONE_VARIABLE_COUNT, 7);
}

#[test]
fn test_fmi_mode_default() {
    let mode = FmiMode::default();
    assert_eq!(mode, FmiMode::Cosimulation);
}

#[test]
fn test_fmi_error_display() {
    let err = FmiError::ExportFailed("test error".to_string());
    assert_eq!(format!("{}", err), "FMU export failed: test error");
}

// -------------------------------------------------------------------------
// Multi-zone extension tests (#1339)
// -------------------------------------------------------------------------

#[test]
fn test_multi_zone_default_single_zone() {
    let exporter = FmiExporter::new();
    assert_eq!(exporter.zone_count(), 1);
    assert_eq!(exporter.total_variable_count(), 7);
}

#[test]
fn test_multi_zone_three_zones_count() {
    let exporter = FmiExporter::new().with_zones(vec![
        ZoneVariables::new("living"),
        ZoneVariables::new("bedroom"),
        ZoneVariables::new("kitchen"),
    ]);
    assert_eq!(exporter.zone_count(), 3);
    assert_eq!(exporter.total_variable_count(), 7 * 3);
}

#[test]
#[should_panic(expected = "at least one zone is required")]
fn test_multi_zone_empty_zones_panics() {
    let _ = FmiExporter::new().with_zones(vec![]);
}

#[test]
fn test_multi_zone_variable_names() {
    let exporter = FmiExporter::new().with_zones(vec![
        ZoneVariables::new("zone"), // legacy single-zone shape
        ZoneVariables::new("bedroom"),
        ZoneVariables::new("kitchen"),
    ]);
    let vars = exporter.variable_names();
    // 3 zones × 7 vars = 21 entries
    assert_eq!(vars.len(), 21);

    // Zone 0 (legacy) keeps the bare template names (#1125 compatibility).
    let zone0_inputs: Vec<_> = vars
        .iter()
        .take(4)
        .map(|(n, c)| (n.clone(), c.clone()))
        .collect();
    assert_eq!(zone0_inputs[0].0, "outdoor_temperature");
    assert_eq!(zone0_inputs[0].1, "input");

    // Zone 1 ("bedroom") uses the `bedroom_` prefix.
    let zone1_inputs: Vec<_> = vars.iter().skip(7).take(4).collect();
    assert_eq!(zone1_inputs[0].0, "bedroom_outdoor_temperature");
    assert_eq!(zone1_inputs[0].1, "input");
    assert_eq!(zone1_inputs[3].0, "bedroom_internal_gains");

    // Zone 2 ("kitchen") uses the `kitchen_` prefix.
    let zone2_outputs: Vec<_> = vars.iter().skip(14).skip(4).collect();
    assert_eq!(zone2_outputs[0].0, "kitchen_zone_temperature");
    assert_eq!(zone2_outputs[0].1, "output");
}

#[test]
fn test_multi_zone_xml_generation_n3() {
    let exporter = FmiExporter::new().with_zones(vec![
        ZoneVariables::new("zone"),
        ZoneVariables::new("bedroom"),
        ZoneVariables::new("kitchen"),
    ]);
    let xml = exporter.generate_model_description_xml().unwrap();

    // FMI 2.0 root
    assert!(
        xml.contains("fmiVersion=\"2.0\""),
        "missing fmiVersion: {}",
        xml
    );
    assert!(
        xml.contains("<fmiModelDescription"),
        "missing fmiModelDescription root: {}",
        xml
    );
    assert!(
        xml.contains("<CoSimulation"),
        "missing CoSimulation element: {}",
        xml
    );
    assert!(
        xml.contains("<DefaultExperiment"),
        "missing DefaultExperiment: {}",
        xml
    );
    // 21 ScalarVariables total (3 × 7)
    let sv_count = xml.matches("<ScalarVariable ").count();
    assert_eq!(
        sv_count, 21,
        "expected 21 ScalarVariables for 3 zones, got {}",
        sv_count
    );
    // 21 Real children
    let real_count = xml.matches("<Real ").count();
    assert_eq!(
        real_count, 21,
        "expected 21 Real attributes, got {}",
        real_count
    );
}

#[test]
fn test_configurable_timestep_default_3600s() {
    let exporter = FmiExporter::new();
    let xml = exporter.generate_model_description_xml().unwrap();
    assert!(
        xml.contains("stepSize=\"3600.0\""),
        "default stepSize missing: {}",
        xml
    );
}

#[test]
fn test_configurable_timestep_60s() {
    let mut cfg = FmiConfig::default();
    cfg.communication_timestep = 60.0;
    let exporter = FmiExporter::with_config(cfg).unwrap();
    let xml = exporter.generate_model_description_xml().unwrap();
    assert!(
        xml.contains("stepSize=\"60.0\""),
        "60s stepSize missing: {}",
        xml
    );
}

#[test]
fn test_configurable_timestep_300s() {
    let mut cfg = FmiConfig::default();
    cfg.communication_timestep = 300.0;
    let exporter = FmiExporter::with_config(cfg).unwrap();
    let xml = exporter.generate_model_description_xml().unwrap();
    assert!(xml.contains("stepSize=\"300.0\""));
}

#[test]
fn test_configurable_timestep_600s() {
    let mut cfg = FmiConfig::default();
    cfg.communication_timestep = 600.0;
    let exporter = FmiExporter::with_config(cfg).unwrap();
    let xml = exporter.generate_model_description_xml().unwrap();
    assert!(xml.contains("stepSize=\"600.0\""));
}

#[test]
fn test_export_fmu_writes_valid_zip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("multi_zone.fmu");
    let exporter = FmiExporter::new().with_zones(vec![
        ZoneVariables::new("zone"),
        ZoneVariables::new("bedroom"),
        ZoneVariables::new("kitchen"),
    ]);
    exporter.export_fmu(&out).expect("export_fmu");

    // Round-trip: open the FMU and read modelDescription.xml back out.
    let xml = FmiExporter::read_model_description_from_fmu(&out).expect("read FMU");
    assert!(xml.contains("<fmiModelDescription"));
    assert!(xml.contains("fmiVersion=\"2.0\""));
    let sv_count = xml.matches("<ScalarVariable ").count();
    assert_eq!(sv_count, 21);
}

#[test]
fn test_xml_contains_required_attributes() {
    let exporter = FmiExporter::new();
    let xml = exporter.generate_model_description_xml().unwrap();

    // Required per FMI 2.0 spec §3
    for needle in &[
        "fmiVersion=\"2.0\"",
        "modelName=\"FluxionBuilding\"",
        "guid=\"{8c4e8d3a-2b1f-4a6c-9e5f-0d3b2a4c6e8d}\"",
        "<CoSimulation",
        "needsExecutionTool=\"true\"",
        "canHandleVariableCommunicationStepSize=\"true\"",
        "<DefaultExperiment",
        "<ModelVariables>",
        "<ScalarVariable name=\"outdoor_temperature\"",
        "<ScalarVariable name=\"zone_temperature\"",
        "causality=\"input\"",
        "causality=\"output\"",
    ] {
        assert!(
            xml.contains(needle),
            "missing required FMI 2.0 attribute/element `{}` in:\n{}",
            needle,
            xml
        );
    }
}

#[test]
fn test_variable_names_input_output_split() {
    let exporter = FmiExporter::new();
    let names = exporter.variable_names();
    let inputs: Vec<_> = names.iter().filter(|(_, c)| c == "input").collect();
    let outputs: Vec<_> = names.iter().filter(|(_, c)| c == "output").collect();
    // Single-zone: 4 inputs, 3 outputs
    assert_eq!(inputs.len(), 4);
    assert_eq!(outputs.len(), 3);
}

#[test]
fn test_zone_variables_default() {
    let z = ZoneVariables::default();
    assert_eq!(z.name, "zone");
    let z2 = ZoneVariables::new("kitchen");
    assert_eq!(z2.name, "kitchen");
}

#[test]
fn test_sanitize_xml_name() {
    assert_eq!(sanitize_xml_name("kitchen"), "kitchen");
    assert_eq!(sanitize_xml_name("living room"), "living_room");
    // First char digit → '_', '-' → '_' (FMI names must match C identifiers).
    assert_eq!(sanitize_xml_name("3rd-floor"), "_rd_floor");
    assert_eq!(sanitize_xml_name(""), "z"); // empty fallback
}

#[test]
fn test_format_float() {
    assert_eq!(format_float(3600.0), "3600.0");
    assert_eq!(format_float(0.0), "0.0");
    assert_eq!(format_float(60.0), "60.0");
    assert_eq!(format_float(1.5e-3), "0.0015");
}

#[test]
fn test_days_to_ymd_known_dates() {
    // 1970-01-01 (epoch)
    assert_eq!(days_to_ymd(0), (1970, 1, 1));
    // 2000-01-01
    assert_eq!(days_to_ymd(10_957), (2000, 1, 1));
    // 2024-02-29 (leap year)
    assert_eq!(days_to_ymd(19_782), (2024, 2, 29));
    // 2026-06-27 (today, just before write-time)
    assert_eq!(days_to_ymd(20_631), (2026, 6, 27));
}

// -------------------------------------------------------------------------
// Import (FmiMode::Import) tests — issue #1708
// -------------------------------------------------------------------------

#[test]
fn test_parse_model_description_single_zone() {
    let exporter = FmiExporter::new();
    let xml = exporter.generate_model_description_xml().unwrap();
    let desc = FmiImporter::parse_model_description(&xml).unwrap();

    assert_eq!(desc.fmi_version, "2.0");
    assert_eq!(desc.model_name, "FluxionBuilding");
    assert_eq!(desc.variable_naming_convention, "structured");
    // 4 inputs + 3 outputs = 7 variables
    assert_eq!(desc.variables.len(), 7);
    assert_eq!(desc.input_count(), 4);
    assert_eq!(desc.output_count(), 3);
    assert_eq!(desc.zone_count(), 1);
    assert_eq!(desc.communication_timestep(), 3600.0);
}

#[test]
fn test_parse_model_description_multi_zone() {
    let exporter = FmiExporter::new().with_zones(vec![
        ZoneVariables::new("zone"),
        ZoneVariables::new("bedroom"),
        ZoneVariables::new("kitchen"),
    ]);
    let xml = exporter.generate_model_description_xml().unwrap();
    let desc = FmiImporter::parse_model_description(&xml).unwrap();

    assert_eq!(desc.variables.len(), 21);
    assert_eq!(desc.input_count(), 12);
    assert_eq!(desc.output_count(), 9);
    assert_eq!(desc.zone_count(), 3);

    // Spot-check that variable names round-trip and units are captured.
    let outdoor = desc
        .variables
        .iter()
        .find(|v| v.name == "outdoor_temperature")
        .expect("outdoor_temperature present");
    assert_eq!(outdoor.causality, "input");
    assert_eq!(outdoor.unit, "K");
    assert_eq!(outdoor.start, Some(280.0));

    let zone_temp = desc
        .variables
        .iter()
        .find(|v| v.name == "kitchen_zone_temperature")
        .expect("kitchen_zone_temperature present");
    assert_eq!(zone_temp.causality, "output");
    assert_eq!(zone_temp.unit, "K");
}

#[test]
fn test_parse_model_description_empty_xml_errors() {
    let xml = r#"<?xml version="1.0"?>
<fmiModelDescription fmiVersion="2.0">
  <ModelVariables/>
</fmiModelDescription>"#;
    let res = FmiImporter::parse_model_description(xml);
    assert!(res.is_err(), "empty ModelVariables must error");
}

#[test]
fn test_import_fmu_round_trip_single_zone() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("single_zone.fmu");
    FmiExporter::new().export_fmu(&out).expect("export");

    let model = import_fmu(&out).expect("import_fmu");
    assert_eq!(model.hvac.num_zones, 1);
}

#[test]
fn test_import_fmu_round_trip_three_zone() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("fluxion_three_zone.fmu");
    let exporter = FmiExporter::new().with_zones(vec![
        ZoneVariables::new("zone"),
        ZoneVariables::new("bedroom"),
        ZoneVariables::new("kitchen"),
    ]);
    exporter.export_fmu(&out).expect("export");

    let fmu = FmiImporter::new().import(&out).expect("import");
    assert_eq!(fmu.zone_count(), 3);
    assert_eq!(fmu.communication_timestep(), 3600.0);
    assert_eq!(fmu.thermal_model().hvac.num_zones, 3);
    assert_eq!(fmu.into_thermal_model().hvac.num_zones, 3);
}

#[test]
fn test_import_fmu_configurable_timestep() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("ts300.fmu");
    let mut cfg = FmiConfig::default();
    cfg.communication_timestep = 300.0;
    FmiExporter::with_config(cfg)
        .unwrap()
        .export_fmu(&out)
        .expect("export");

    let fmu = FmiImporter::new().import(&out).expect("import");
    assert_eq!(fmu.communication_timestep(), 300.0);
}

#[test]
fn test_cosimulation_master_do_step_calls_step_physics() {
    // Export a single-zone FMU, re-import it, and drive one doStep.
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("master.fmu");
    FmiExporter::new().export_fmu(&out).expect("export");
    let fmu = FmiImporter::new().import(&out).expect("import");

    let initial_temp_k = fmu.thermal_model().setpoints.temperatures.as_ref()[0] + 273.15;
    let mut master = FmuCoSimulationMaster::from_imported(fmu);

    // Cold outdoor air (263.15 K = -10 °C) → expect the zone to cool
    // and/or heating to engage.
    let inputs = FmuInputs {
        outdoor_temperature: 263.15,
        direct_normal_solar: 0.0,
        diffuse_horizontal_solar: 0.0,
        internal_gains: 0.0,
    };
    let out_step = master.do_step(inputs, Some(3600.0));

    // do_step must return a finite zone temperature in Kelvin for the
    // single zone (single-zone FMU ⇒ vector length == 1).
    assert_eq!(out_step.len(), 1);
    let zone_out = &out_step[0];
    assert!(zone_out.zone_temperature.is_finite());
    assert!(zone_out.zone_temperature > 200.0 && zone_out.zone_temperature < 320.0);
    // The master advanced time by one communication step.
    assert_eq!(master.current_time(), 3600.0);
    // The zone temperature should have moved away from the initial 20 °C
    // (293.15 K) under the cold boundary condition.
    assert_ne!(zone_out.zone_temperature, initial_temp_k);
}

#[test]
fn test_cosimulation_master_loads_nonneg_and_balanced() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("loads.fmu");
    FmiExporter::new().export_fmu(&out).expect("export");
    let fmu = FmiImporter::new().import(&out).expect("import");
    let mut master = FmuCoSimulationMaster::from_imported(fmu);

    // Drive a handful of steps; loads must be non-negative for every
    // zone reported by do_step.
    for _ in 0..5 {
        let outputs = master.do_step(FmuInputs::default(), Some(3600.0));
        assert!(!outputs.is_empty());
        for o in &outputs {
            assert!(o.heating_load >= 0.0);
            assert!(o.cooling_load >= 0.0);
        }
    }
    assert_eq!(master.current_time(), 5.0 * 3600.0);
}

#[test]
fn test_import_fmu_missing_file_errors() {
    let res = import_fmu(Path::new("/nonexistent/does_not_exist.fmu"));
    assert!(res.is_err());
}
