//! Phase 7A TDD Integration Tests: SOLAR-01 - Peak Cooling Under-Prediction
//!
//! This module implements the RED-GREEN-REFACTOR TDD cycle for addressing
//! peak cooling load underprediction by 40-80% across ASHRAE 140 cases.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;

/// Test: Case 600 Peak Cooling - RED Test
///
/// This test compares Fluxion peak cooling against EnergyPlus reference.
/// Expected to FAIL initially (RED state) showing underprediction.
#[test]
#[ignore = "solar distribution calculation issue — ref: #1216"]
fn test_case_600_peak_cooling_red() {
    // EnergyPlus reference for Case 600 (from ASHRAE 140 documentation)
    // Expected peak cooling: ~4.80 kW
    // Current Fluxion (pre-fix): ~2.5-3.0 kW (40-50% underprediction)
    let expected_peak_kw = 4.80;
    let tolerance_pct = 0.15; // ASHRAE tolerance: ±15%

    // Create Case 600 model
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec(&spec);

    // Load Denver TMY weather
    let _weather = DenverTmyWeather::new();

    // Run simulation for one year
    let surrogates = SurrogateManager::new().expect("Failed to create surrogate manager");
    let _eui = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    // Extract peak cooling from simulation
    let peak_cooling_kw = model.get_peak_cooling_power_kw();

    // Diagnostic output
    println!("\n=== Phase 7A TDD: Case 600 Peak Cooling ===");
    println!("EnergyPlus reference: {:.2} kW", expected_peak_kw);
    println!("Fluxion result: {:.2} kW", peak_cooling_kw);
    let deviation_pct = (peak_cooling_kw - expected_peak_kw).abs() / expected_peak_kw;
    println!("Deviation: {:.1}%", deviation_pct * 100.0);
    println!("ASHRAE tolerance: ±{:.0}%", tolerance_pct * 100.0);

    // RED TEST: Should fail initially due to underprediction
    assert!(
        deviation_pct <= tolerance_pct,
        "RED FAIL: Peak cooling underprediction detected! Expected: {:.2} kW Got: {:.2} kW Error: {:.1}% underprediction",
        expected_peak_kw, peak_cooling_kw, deviation_pct * 100.0
    );
}

/// Test: Solar Gain Distribution Analysis
///
/// Analyzes how solar gains are distributed between air and thermal mass.
/// This helps diagnose why peak cooling is underpredicted.
#[test]
#[ignore = "solar distribution calculation issue — ref: #1216"]
fn test_solar_gain_distribution_analysis() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec(&spec);
    let _weather = DenverTmyWeather::new();

    // Run simulation for a single day to analyze solar distribution
    let surrogates = SurrogateManager::new().expect("Failed to create surrogate manager");
    let _eui = model.solve_timesteps(24, &surrogates, false, None, None, None);

    println!("\n=== Solar Gain Distribution Analysis ===");
    println!(
        "Solar distribution to air fraction: {:.3}",
        model.solar_distribution_to_air
    );
    println!(
        "Solar beam to mass fraction: {:.3}",
        model.solar_beam_to_mass_fraction
    );
    println!("Convective fraction: {:.3}", model.convective_fraction);

    // Analysis:
    // - If solar_distribution_to_air is 0, all solar goes directly to thermal mass
    // - This may cause delays in peak cooling as mass must heat up first
    // - Some solar should go directly to air for immediate cooling load

    // Expected: solar_distribution_to_air > 0 for proper peak cooling response
    // Current behavior: likely 0.0, causing underprediction
    assert!(
        model.solar_distribution_to_air > 0.0,
        "Solar distribution to air is zero - all solar gains go to thermal mass. This delays peak cooling response. Expected: solar_distribution_to_air > 0.0 for immediate air heating. Actual: {:.3}",
        model.solar_distribution_to_air
    );
}

/// Test: High-Mass Case 900 Peak Cooling
///
/// High-mass cases show even larger underprediction (up to 80%).
/// This helps diagnose mass-related solar coupling issues.
#[test]
#[ignore = "solar distribution calculation issue — ref: #1216"]
fn test_case_900_peak_cooling_red() {
    // EnergyPlus reference for Case 900 (high-mass)
    // Expected peak cooling: ~3.5-4.0 kW
    let expected_peak_kw = 3.80;
    let tolerance_pct = 0.15;

    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::from_spec(&spec);

    let _weather = DenverTmyWeather::new();
    let surrogates = SurrogateManager::new().expect("Failed to create surrogate manager");
    let _eui = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    let peak_cooling_kw = model.get_peak_cooling_power_kw();

    println!("\n=== Phase 7A TDD: Case 900 Peak Cooling (High Mass) ===");
    println!("EnergyPlus reference: {:.2} kW", expected_peak_kw);
    println!("Fluxion result: {:.2} kW", peak_cooling_kw);
    let deviation_pct = (peak_cooling_kw - expected_peak_kw).abs() / expected_peak_kw;
    println!("Deviation: {:.1}%", deviation_pct * 100.0);

    // RED TEST: Should fail for high-mass case
    assert!(
        deviation_pct <= tolerance_pct,
        "RED FAIL: High-mass peak cooling underprediction detected! Expected: {:.2} kW Got: {:.2} kW Error: {:.1}% underprediction",
        expected_peak_kw, peak_cooling_kw, deviation_pct * 100.0
    );
}
