//! Phase 7A TDD Tests: SOLAR-01 - Peak Cooling Under-Prediction
//!
//! Simple diagnostic test for solar gain distribution analysis.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;

/// Test: Solar Gain Distribution Analysis
///
/// Simple diagnostic to check solar distribution parameters.
/// Focus on understanding why peak cooling is underpredicted.
#[test]
fn test_solar_distribution_diagnostic() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let _weather = DenverTmyWeather::new();

    // Run simulation for a single day to analyze solar distribution
    let surrogates = SurrogateManager::new().expect("Failed to create surrogate manager");
    let _eui = model.solve_timesteps(24, &surrogates, false, None, None, None);

    println!("\n=== Solar Gain Distribution Diagnostic ===");
    println!(
        "Solar distribution to air fraction: {:.3}",
        model.solar_distribution_to_air
    );
    println!(
        "Solar beam to mass fraction: {:.3}",
        model.solar_beam_to_mass_fraction
    );
    println!("Convective fraction: {:.3}", model.convective_fraction);
    println!(
        "Peak cooling power: {:.2} kW",
        model.get_peak_cooling_power_kw()
    );

    // Analysis:
    // - If solar_distribution_to_air is 0, all solar goes directly to thermal mass
    // - This may cause delays in peak cooling as mass must heat up first
    // - Some solar should go directly to air for immediate cooling load

    // Expected: solar_distribution_to_air > 0 for proper peak cooling response
    // Current behavior: likely 0.0, causing underprediction

    // This is a diagnostic test - it will show current behavior
    // The actual fix needs to modify thermal model solar coupling
}
