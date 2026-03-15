//! Thermal mass energy accounting validation tests
//!
//! These tests validate that the physics engine correctly conserves energy
//! according to the first law of thermodynamics for both high-mass
//! (900-series) and low-mass (600-series) ASHRAE 140 cases.
//!
//! ## Energy Balance Principle
//!
//! At each timestep, the following equation must hold:
//!
//! ```text
//! Σenergy_in = Σenergy_out + Δmass_energy
//! ```
//!
//! Where:
//! - `energy_in`: HVAC energy + solar gains + infiltration (external inputs)
//! - `energy_out`: HVAC demand (energy removed/rejected to maintain setpoints)
//! - `mass_energy_change`: Cm × ΔTm (thermal capacitance × temperature change)
//!
//! ## Test Cases
//!
//! - **High-mass cases (900-series)**: Case 900, 920, 930, 940, 950, 960
//! - **Low-mass cases (600-series)**: Case 600, 610, 620, 630, 640, 650
//!
//! ## Validation Criteria
//!
//! Energy balance is valid if cumulative error < 0.01% of total energy flow.
//! This indicates the physics engine correctly conserves energy, even if annual
//! energy predictions are inaccurate (which would indicate a fundamental 5R1C
//! limitation, not a bug).

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::thermal_mass_energy_accounting::{
    calculate_mass_energy, validate_energy_balance_over_year,
};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Test Case 900 (high-mass) energy accounting.
///
/// Case 900 is the high-mass version of Case 600 with thick concrete walls
/// and floors providing significant thermal mass. This test validates that the
/// physics engine correctly conserves energy for high-mass buildings.
#[test]
fn test_case_900_energy_accounting() {
    println!("\n=== Testing Case 900 (high-mass) Energy Accounting ===");

    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run energy balance validation
    let report = validate_energy_balance_over_year(&mut model);

    // Print diagnostic output
    println!("  Cumulative Error: {:.6e} J", report.cumulative_error);
    println!("  Error Percentage: {:.6}%", report.error_pct);
    println!(
        "  Status: {}",
        if report.is_valid { "PASSED" } else { "FAILED" }
    );
    println!("  Energy In Total: {:.6e} J", report.energy_in_total);
    println!("  Energy Out Total: {:.6e} J", report.energy_out_total);
    println!("  Hourly Errors: {} timesteps", report.hourly_errors.len());

    // Assert energy balance is valid (error < 0.01%)
    assert!(
        report.is_valid,
        "Case 900 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 900 energy accounting: {:.6}% error (status: PASSED)",
        report.error_pct
    );
}

/// Test Case 600 (low-mass) energy accounting.
///
/// Case 600 is the baseline low-mass case. This test validates that the
/// physics engine correctly conserves energy for low-mass buildings.
#[test]
fn test_case_600_energy_accounting() {
    println!("\n=== Testing Case 600 (low-mass) Energy Accounting ===");

    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Run energy balance validation
    let report = validate_energy_balance_over_year(&mut model);

    // Print diagnostic output
    println!("  Cumulative Error: {:.6e} J", report.cumulative_error);
    println!("  Error Percentage: {:.6}%", report.error_pct);
    println!(
        "  Status: {}",
        if report.is_valid { "PASSED" } else { "FAILED" }
    );
    println!("  Energy In Total: {:.6e} J", report.energy_in_total);
    println!("  Energy Out Total: {:.6e} J", report.energy_out_total);
    println!("  Hourly Errors: {} timesteps", report.hourly_errors.len());

    // Assert energy balance is valid (error < 0.01%)
    assert!(
        report.is_valid,
        "Case 600 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 600 energy accounting: {:.6}% error (status: PASSED)",
        report.error_pct
    );
}

/// Test Case 920 energy accounting.
///
/// Case 920 is high-mass with east/west windows.
#[test]
fn test_case_920_energy_accounting() {
    println!("\n=== Testing Case 920 Energy Accounting ===");

    let spec = ASHRAE140Case::Case920.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 920 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 920 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 930 energy accounting.
///
/// Case 930 is high-mass with thermostat setback.
#[test]
fn test_case_930_energy_accounting() {
    println!("\n=== Testing Case 930 Energy Accounting ===");

    let spec = ASHRAE140Case::Case930.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 930 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 930 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 940 energy accounting.
///
/// Case 940 is high-mass with overnight setback.
#[test]
fn test_case_940_energy_accounting() {
    println!("\n=== Testing Case 940 Energy Accounting ===");

    let spec = ASHRAE140Case::Case940.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 940 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 940 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 950 energy accounting.
///
/// Case 950 is high-mass with night ventilation.
#[test]
fn test_case_950_energy_accounting() {
    println!("\n=== Testing Case 950 Energy Accounting ===");

    let spec = ASHRAE140Case::Case950.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 950 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 950 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 960 energy accounting.
///
/// Case 960 uses COP correction (heating_efficiency=0.9, cooling_cop=3.0).
#[test]
fn test_case_960_energy_accounting() {
    println!("\n=== Testing Case 960 Energy Accounting ===");

    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 960 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 960 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 610 energy accounting.
///
/// Case 610 is the free-floating version of Case 600.
#[test]
fn test_case_610_energy_accounting() {
    println!("\n=== Testing Case 610 Energy Accounting ===");

    let spec = ASHRAE140Case::Case610.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 610 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 610 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 620 energy accounting.
///
/// Case 620 has higher insulation values.
#[test]
fn test_case_620_energy_accounting() {
    println!("\n=== Testing Case 620 Energy Accounting ===");

    let spec = ASHRAE140Case::Case620.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 620 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 620 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 630 energy accounting.
///
/// Case 630 has modified setpoints.
#[test]
fn test_case_630_energy_accounting() {
    println!("\n=== Testing Case 630 Energy Accounting ===");

    let spec = ASHRAE140Case::Case630.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 630 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 630 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 640 energy accounting.
///
/// Case 640 has higher solar absorptance.
#[test]
fn test_case_640_energy_accounting() {
    println!("\n=== Testing Case 640 Energy Accounting ===");

    let spec = ASHRAE140Case::Case640.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 640 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 640 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Test Case 650 energy accounting.
///
/// Case 650 has modified window-to-wall ratio.
#[test]
fn test_case_650_energy_accounting() {
    println!("\n=== Testing Case 650 Energy Accounting ===");

    let spec = ASHRAE140Case::Case650.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    let report = validate_energy_balance_over_year(&mut model);

    println!("  Error Percentage: {:.6}%", report.error_pct);

    assert!(
        report.is_valid,
        "Case 650 energy balance FAILED: {:.6}% error (threshold: 0.01%)",
        report.error_pct
    );

    println!(
        "✅ Case 650 energy accounting: {:.6}% error",
        report.error_pct
    );
}

/// Parameterized test for all 900-series cases.
///
/// This test validates energy accounting for all high-mass cases in the
/// 900-series to ensure the physics engine correctly conserves energy
/// for buildings with significant thermal mass.
#[test]
fn test_all_900_series_energy_accounting() {
    println!("\n=== Testing All 900-Series Cases Energy Accounting ===");

    let cases = [
        ("900", ASHRAE140Case::Case900),
        ("920", ASHRAE140Case::Case920),
        ("930", ASHRAE140Case::Case930),
        ("940", ASHRAE140Case::Case940),
        ("950", ASHRAE140Case::Case950),
        ("960", ASHRAE140Case::Case960),
    ];

    for (case_id, case_enum) in cases {
        println!("\n  Testing Case {}...", case_id);

        let spec = case_enum.spec();
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);

        let report = validate_energy_balance_over_year(&mut model);

        println!("    Error: {:.6}% (threshold: 0.01%)", report.error_pct);

        assert!(
            report.is_valid,
            "Case {} energy balance FAILED: {:.6}% error (threshold: 0.01%)",
            case_id, report.error_pct
        );

        println!("    ✅ PASSED");
    }

    println!("\n✅ All 900-series cases passed energy accounting validation");
}

/// Parameterized test for all 600-series cases.
///
/// This test validates energy accounting for all low-mass cases in the
/// 600-series to ensure the physics engine correctly conserves energy
/// for baseline building configurations.
#[test]
fn test_all_600_series_energy_accounting() {
    println!("\n=== Testing All 600-Series Cases Energy Accounting ===");

    let cases = [
        ("600", ASHRAE140Case::Case600),
        ("610", ASHRAE140Case::Case610),
        ("620", ASHRAE140Case::Case620),
        ("630", ASHRAE140Case::Case630),
        ("640", ASHRAE140Case::Case640),
        ("650", ASHRAE140Case::Case650),
    ];

    for (case_id, case_enum) in cases {
        println!("\n  Testing Case {}...", case_id);

        let spec = case_enum.spec();
        let mut model = ThermalModel::<VectorField>::from_spec(&spec);

        let report = validate_energy_balance_over_year(&mut model);

        println!("    Error: {:.6}% (threshold: 0.01%)", report.error_pct);

        assert!(
            report.is_valid,
            "Case {} energy balance FAILED: {:.6}% error (threshold: 0.01%)",
            case_id, report.error_pct
        );

        println!("    ✅ PASSED");
    }

    println!("\n✅ All 600-series cases passed energy accounting validation");
}
