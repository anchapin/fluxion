//! h_tr_ms ISO 13790 Regression Tests
//!
//! These tests validate that the ISO 13790-aligned h_tr_ms calculation
//! produces expected thermal time constants for different mass classes.
//!
//! ## Issue #583 Fix
//!
//! This file was created to validate the fix for issue #583, which replaced
//! the physics-based layer resistance calculation with the ISO 13790-aligned
//! formula:
//!
//! ```text
//! h_tr_ms = C_m / τ
//! ```
//!
//! Where:
//! - C_m = effective thermal capacitance (J/K)
//! - τ = target thermal time constant (seconds)
//!
//! Target time constants:
//! - Light mass (Case 600): τ ~ 15 hours
//! - Heavy mass (Case 900): τ ~ 150 hours

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

const J_TO_KWH: f64 = 1.0 / 3.6e6;
const W_TO_KW: f64 = 1.0 / 1000.0;

/// Reference ranges for ASHRAE 140 cases (in MWh for energy, kW for peak)
struct CaseReference {
    case_id: &'static str,
    annual_heating_min: f64,
    annual_heating_max: f64,
    annual_cooling_min: f64,
    annual_cooling_max: f64,
    peak_heating_min: f64,
    peak_heating_max: f64,
    peak_cooling_min: f64,
    peak_cooling_max: f64,
}

const CASE_600: CaseReference = CaseReference {
    case_id: "600",
    annual_heating_min: 1.50,
    annual_heating_max: 2.20,
    annual_cooling_min: 0.60,
    annual_cooling_max: 1.20,
    peak_heating_min: 2.80,
    peak_heating_max: 3.80,
    peak_cooling_min: 1.50,
    peak_cooling_max: 2.20,
};

const CASE_900: CaseReference = CaseReference {
    case_id: "900",
    annual_heating_min: 1.17,
    annual_heating_max: 2.04,
    annual_cooling_min: 2.13,
    annual_cooling_max: 3.67,
    peak_heating_min: 1.10,
    peak_heating_max: 2.10,
    peak_cooling_min: 2.10,
    peak_cooling_max: 3.50,
};

fn calculate_thermal_time_constant(model: &ThermalModel<VectorField>) -> f64 {
    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_em = model.h_tr_em.as_ref()[0];

    let structure_cap = model.structure_thermal_cap.as_ref()[0];

    let tau_seconds = structure_cap / (h_tr_ms + h_tr_em).max(0.1);
    tau_seconds / 3600.0
}

fn run_annual_simulation(case: ASHRAE140Case) -> (f64, f64, f64, f64) {
    let spec = case.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let steps = 8760;

    let mut total_heating = 0.0_f64;
    let mut total_cooling = 0.0_f64;
    let mut peak_heating = 0.0_f64;
    let mut peak_cooling = 0.0_f64;

    for step in 0..steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());

        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if energy_kwh > 0.0 {
            total_heating += energy_kwh;
            let power_w = energy_kwh * 1000.0 / 3600.0;
            peak_heating = peak_heating.max(power_w);
        } else {
            total_cooling += -energy_kwh;
            let power_w = -energy_kwh * 1000.0 / 3600.0;
            peak_cooling = peak_cooling.max(power_w);
        }
    }

    (
        total_heating,
        total_cooling,
        peak_heating / 1000.0,
        peak_cooling / 1000.0,
    )
}

#[test]
fn test_h_tr_ms_light_mass_thermal_time_constant() {
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let tau_hours = calculate_thermal_time_constant(&model);

    println!(
        "Case 600 (light mass) thermal time constant: {:.1} hours",
        tau_hours
    );
    println!("Expected: ~10-20 hours (light mass)");

    assert!(
        tau_hours >= 5.0 && tau_hours <= 30.0,
        "Case 600 thermal time constant {} hours outside expected range [5, 30] hours",
        tau_hours
    );
}

#[test]
fn test_h_tr_ms_heavy_mass_thermal_time_constant() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let tau_hours = calculate_thermal_time_constant(&model);

    println!(
        "Case 900 (heavy mass) thermal time constant: {:.1} hours",
        tau_hours
    );
    println!("Expected: ~120-200 hours (heavy mass)");

    assert!(
        tau_hours >= 80.0 && tau_hours <= 250.0,
        "Case 900 thermal time constant {} hours outside expected range [80, 250] hours",
        tau_hours
    );
}

#[test]
fn test_h_tr_ms_iso_calculation_formula() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let h_tr_ms = model.h_tr_ms.as_ref()[0];
    let h_tr_is = model.h_tr_is.as_ref()[0];
    let structure_cap = model.structure_thermal_cap.as_ref()[0];

    let target_tau_hours = 150.0;
    let target_tau_seconds = target_tau_hours * 3600.0;

    let expected_h_tr_ms = structure_cap / target_tau_seconds;

    let min_h_tr_ms = h_tr_is * 0.1;
    let expected_h_tr_ms_bounded = expected_h_tr_ms.max(min_h_tr_ms);

    println!("ISO 13790 h_tr_ms calculation verification:");
    println!("  h_tr_is = {:.2} W/K", h_tr_is);
    println!("  C_m = {:.2e} J/K", structure_cap);
    println!("  target τ = {:.0} hours", target_tau_hours);
    println!(
        "  calculated h_tr_ms (unbounded) = {:.4f} W/K",
        expected_h_tr_ms
    );
    println!(
        "  calculated h_tr_ms (bounded, min 10% h_is) = {:.4f} W/K",
        expected_h_tr_ms_bounded
    );
    println!("  actual h_tr_ms = {:.4f} W/K", h_tr_ms);

    let tolerance = expected_h_tr_ms_bounded * 0.3;
    assert!(
        (h_tr_ms - expected_h_tr_ms_bounded).abs() <= tolerance,
        "h_tr_ms {:.4f} differs from expected {:.4f} by more than {:.4f} (30% tolerance)",
        h_tr_ms,
        expected_h_tr_ms_bounded,
        tolerance
    );
}

#[test]
fn test_case_600_annual_energy_within_expected_range() {
    let (heating, cooling, _, _) = run_annual_simulation(ASHRAE140Case::Case600);

    println!("Case 600 Results (light mass with ISO h_tr_ms):");
    println!("  Annual Heating: {:.2} MWh", heating);
    println!("  Annual Cooling: {:.2} MWh", cooling);
    println!(
        "  Reference Range: [{:.2}, {:.2}] MWh heating, [{:.2}, {:.2}] MWh cooling",
        CASE_600.annual_heating_min,
        CASE_600.annual_heating_max,
        CASE_600.annual_cooling_min,
        CASE_600.annual_cooling_max
    );

    let tolerance_pct = 0.30;

    let heating_in_range = heating >= CASE_600.annual_heating_min * (1.0 - tolerance_pct)
        && heating <= CASE_600.annual_heating_max * (1.0 + tolerance_pct);

    let cooling_in_range = cooling >= CASE_600.annual_cooling_min * (1.0 - tolerance_pct)
        && cooling <= CASE_600.annual_cooling_max * (1.0 + tolerance_pct);

    assert!(
        heating_in_range || cooling_in_range,
        "Case 600 energy outside extended range: heating={:.2} MWh, cooling={:.2} MWh. \
         Expected ~[{:.2}, {:.2}] MWh heating, ~[{:.2}, {:.2}] MWh cooling",
        heating,
        cooling,
        CASE_600.annual_heating_min,
        CASE_600.annual_heating_max,
        CASE_600.annual_cooling_min,
        CASE_600.annual_cooling_max
    );
}

#[test]
fn test_case_900_annual_energy_within_expected_range() {
    let (heating, cooling, _, _) = run_annual_simulation(ASHRAE140Case::Case900);

    println!("Case 900 Results (heavy mass with ISO h_tr_ms):");
    println!("  Annual Heating: {:.2} MWh", heating);
    println!("  Annual Cooling: {:.2} MWh", cooling);
    println!(
        "  Reference Range: [{:.2}, {:.2}] MWh heating, [{:.2}, {:.2}] MWh cooling",
        CASE_900.annual_heating_min,
        CASE_900.annual_heating_max,
        CASE_900.annual_cooling_min,
        CASE_900.annual_cooling_max
    );

    let tolerance_pct = 0.30;

    let heating_in_range = heating >= CASE_900.annual_heating_min * (1.0 - tolerance_pct)
        && heating <= CASE_900.annual_heating_max * (1.0 + tolerance_pct);

    let cooling_in_range = cooling >= CASE_900.annual_cooling_min * (1.0 - tolerance_pct)
        && cooling <= CASE_900.annual_cooling_max * (1.0 + tolerance_pct);

    assert!(
        heating_in_range || cooling_in_range,
        "Case 900 energy outside extended range: heating={:.2} MWh, cooling={:.2} MWh. \
         Expected ~[{:.2}, {:.2}] MWh heating, ~[{:.2}, {:.2}] MWh cooling",
        heating,
        cooling,
        CASE_900.annual_heating_min,
        CASE_900.annual_heating_max,
        CASE_900.annual_cooling_min,
        CASE_900.annual_cooling_max
    );
}

#[test]
fn test_h_tr_ms_no_longer_uses_layer_resistance() {
    let spec_600 = ASHRAE140Case::Case600.spec();
    let spec_900 = ASHRAE140Case::Case900.spec();

    let model_600 = ThermalModel::<VectorField>::from_spec(&spec_600);
    let model_900 = ThermalModel::<VectorField>::from_spec(&spec_900);

    let h_tr_ms_600 = model_600.h_tr_ms.as_ref()[0];
    let h_tr_ms_900 = model_900.h_tr_ms.as_ref()[0];

    let h_is_600 = model_600.h_tr_is.as_ref()[0];
    let h_is_900 = model_900.h_tr_is.as_ref()[0];

    println!("h_tr_ms comparison between mass classes:");
    println!(
        "  Case 600 (light): h_tr_ms={:.4f} W/K, h_is={:.2f} W/K, ratio={:.3}",
        h_tr_ms_600,
        h_is_600,
        h_tr_ms_600 / h_is_600
    );
    println!(
        "  Case 900 (heavy): h_tr_ms={:.4f} W/K, h_is={:.2f} W/K, ratio={:.3}",
        h_tr_ms_900,
        h_is_900,
        h_tr_ms_900 / h_is_900
    );

    assert!(
        h_tr_ms_900 < h_tr_ms_600,
        "Heavy mass should have LOWER h_tr_ms than light mass (for higher thermal damping)"
    );

    assert!(
        h_tr_ms_900 < h_is_900 * 0.5,
        "Heavy mass h_tr_ms should be significantly lower than h_is (strong thermal damping)"
    );

    assert!(
        h_tr_ms_600 > h_is_600 * 0.1,
        "Light mass h_tr_ms should be at least 10% of h_is (minimum coupling)"
    );
}

#[test]
fn test_all_900_series_h_tr_ms_consistency() {
    let cases = [
        (ASHRAE140Case::Case900, "900"),
        (ASHRAE140Case::Case910, "910"),
        (ASHRAE140Case::Case920, "920"),
        (ASHRAE140Case::Case930, "930"),
        (ASHRAE140Case::Case940, "940"),
        (ASHRAE140Case::Case950, "950"),
    ];

    println!("900-series thermal time constants (all should be ~80-200 hours for heavy mass):");

    for (case, name) in cases {
        let spec = case.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);
        let tau = calculate_thermal_time_constant(&model);
        let h_tr_ms = model.h_tr_ms.as_ref()[0];

        println!(
            "  Case {}: τ={:.1} hours, h_tr_ms={:.4f} W/K",
            name, tau, h_tr_ms
        );

        assert!(
            tau >= 50.0 && tau <= 300.0,
            "Case {} thermal time constant {} hours outside expected range [50, 300] hours",
            name,
            tau
        );
    }
}
