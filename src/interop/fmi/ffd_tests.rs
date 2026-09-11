// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Tests for the FFD FMU C-API and exporter.

use super::*;

#[test]
fn test_ffd_fmu_config_default() {
    let config = FfdFmuConfig::default();
    assert_eq!(config.model_name, "FluxionFFD");
    assert_eq!(config.communication_timestep, 60.0);
}

#[test]
fn test_ffd_fmu_config_validate_ok() {
    let config = FfdFmuConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_ffd_fmu_config_validate_bad_timestep() {
    let mut config = FfdFmuConfig::default();
    config.communication_timestep = 0.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_ffd_fmu_config_validate_bad_time_range() {
    let mut config = FfdFmuConfig::default();
    config.start_time = 100.0;
    config.stop_time = 50.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_ffd_fmu_config_validate_bad_num_surfaces() {
    let mut config = FfdFmuConfig::default();
    config.num_surfaces = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_ffd_fmu_exporter_new() {
    let exporter = FfdFmuExporter::new();
    assert_eq!(exporter.config().model_name, "FluxionFFD");
    assert_eq!(exporter.input_count(), 3 + FFD_MAX_SURFACES);
    assert_eq!(
        exporter.output_count(),
        FFD_STRATIFICATION_LEVELS + 2 * FFD_MAX_SURFACES
    );
}

#[test]
fn test_ffd_fmu_exporter_with_config() {
    let config = FfdFmuConfig::default();
    let exporter = FfdFmuExporter::with_config(config);
    assert!(exporter.is_ok());
}

#[test]
fn test_ffd_fmu_exporter_with_config_invalid() {
    let mut config = FfdFmuConfig::default();
    config.communication_timestep = 0.0;
    let exporter = FfdFmuExporter::with_config(config);
    assert!(exporter.is_err());
}

#[test]
fn test_ffd_fmu_xml_generation() {
    let exporter = FfdFmuExporter::new();
    let xml = exporter.generate_model_description_xml().unwrap();

    assert!(xml.contains("fmiVersion=\"2.0\""));
    assert!(xml.contains("<fmiModelDescription"));
    assert!(xml.contains("<CoSimulation"));
    assert!(xml.contains("<DefaultExperiment"));
    assert!(xml.contains("inlet_air_temperature"));
    assert!(xml.contains("mass_flow_rate_supply"));
    assert!(xml.contains("zone_air_temperature_0"));
    assert!(xml.contains("chtc_0"));
    assert!(xml.contains("surface_heat_flux_0"));
}

#[test]
fn test_ffd_fmu_xml_has_required_attributes() {
    let exporter = FfdFmuExporter::new();
    let xml = exporter.generate_model_description_xml().unwrap();

    for needle in &[
        "fmiVersion=\"2.0\"",
        "modelName=\"FluxionFFD\"",
        "<CoSimulation",
        "needsExecutionTool=\"true\"",
        "canHandleVariableCommunicationStepSize=\"true\"",
        "<DefaultExperiment",
        "<ModelVariables>",
        "<ScalarVariable name=\"inlet_air_temperature\"",
        "causality=\"input\"",
        "causality=\"output\"",
    ] {
        assert!(
            xml.contains(needle),
            "missing required attribute `{}` in:\n{}",
            needle,
            xml
        );
    }
}

#[test]
fn test_ffd_fmu_export_fmu_writes_valid_zip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("ffd.fmu");
    let exporter = FfdFmuExporter::new();
    exporter.export_fmu(&out).expect("export_fmu");

    let file = std::fs::File::open(&out).expect("open FMU");
    let mut zip = zip::ZipArchive::new(file).expect("read FMU");
    assert!(zip.by_name("modelDescription.xml").is_ok());
}

#[test]
fn test_ffd_fmu_capi_set_real() {
    let mut api = FfdFmuCApi::new(60.0);

    api.set_real(1, 295.15).unwrap();
    assert_eq!(api.state().inputs.inlet_air_temperature, 295.15);

    api.set_real(2, 0.5).unwrap();
    assert_eq!(api.state().inputs.mass_flow_rate_supply, 0.5);

    api.set_real(4, 290.15).unwrap();
    assert_eq!(api.state().inputs.wall_temperatures[0], 290.15);

    assert!(api.set_real(999, 100.0).is_err());
}

#[test]
fn test_ffd_fmu_capi_get_real() {
    let api = FfdFmuCApi::new(60.0);

    // num_inputs = 3 (inlet, supply, exhaust) + 6 (wall temps) = 9
    // Output vrs start at 10 (num_inputs + 1)
    let num_inputs = 3 + FFD_MAX_SURFACES;
    // vr = num_inputs = 9 is still an input, should error
    assert!(api.get_real(num_inputs as u32).is_err());
    // vr = num_inputs + 1 = 10 is the first output, should be Ok
    let result = api.get_real(num_inputs as u32 + 1);
    assert!(result.is_ok());
}

#[test]
fn test_ffd_fmu_capi_do_step() {
    let mut api = FfdFmuCApi::new(60.0);
    api.state_mut().initialised = true;

    api.state_mut().inputs.inlet_air_temperature = 295.15;
    api.state_mut().inputs.mass_flow_rate_supply = 0.3;
    api.state_mut().inputs.wall_temperatures = [293.15; FFD_MAX_SURFACES];

    api.do_step(60.0).unwrap();

    assert_eq!(api.state().current_time, 60.0);
    assert_eq!(api.state().timestep, 1);

    for temp in api.state().outputs.zone_air_temperatures {
        assert!(temp > 200.0 && temp < 350.0);
    }
}

#[test]
fn test_ffd_fmu_capi_do_step_not_initialised() {
    let mut api = FfdFmuCApi::new(60.0);
    assert!(api.do_step(60.0).is_err());
}

#[test]
fn test_ffd_fmu_capi_reset() {
    let mut api = FfdFmuCApi::new(60.0);
    api.state_mut().initialised = true;
    api.state_mut().current_time = 3600.0;
    api.state_mut().timestep = 60;

    api.reset();

    assert_eq!(api.state().current_time, 0.0);
    assert_eq!(api.state().timestep, 0);
    assert!(!api.state().initialised);
}

#[test]
fn test_ffd_fmu_inputs_default() {
    let inputs = FfdFmuInputs::default();
    assert_eq!(inputs.inlet_air_temperature, 293.15);
    assert_eq!(inputs.mass_flow_rate_supply, 0.0);
    assert_eq!(inputs.mass_flow_rate_exhaust, 0.0);
    for t in inputs.wall_temperatures {
        assert_eq!(t, 293.15);
    }
}

#[test]
fn test_ffd_fmu_outputs_default() {
    let outputs = FfdFmuOutputs::default();
    for temp in outputs.zone_air_temperatures {
        assert_eq!(temp, 0.0);
    }
    for chtc in outputs.chtc {
        assert_eq!(chtc, 0.0);
    }
    for flux in outputs.surface_heat_fluxes {
        assert_eq!(flux, 0.0);
    }
}

#[test]
fn test_ffd_fmu_variable_names() {
    let vars = FfdFmuVariables::default();
    let inputs = vars.input_names();
    assert_eq!(inputs.len(), 3 + FFD_MAX_SURFACES);
    assert_eq!(inputs[0], "inlet_air_temperature");
    assert_eq!(inputs[1], "mass_flow_rate_supply");

    let outputs = vars.output_names(6, 4);
    assert_eq!(outputs.len(), 4 + 12);
    assert_eq!(outputs[0], "zone_air_temperature_0");
    assert_eq!(outputs[4], "chtc_0");
    assert_eq!(outputs[10], "surface_heat_flux_0");
}
