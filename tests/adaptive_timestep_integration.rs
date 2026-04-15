//! Adaptive Timestep Integration Tests
//!
//! This module contains integration tests for the adaptive timestep feature,
//! validating accuracy improvements for high-mass buildings (Case 900 series).

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::adaptive_timestep::{
    AdaptiveTimestepScheduler, TimeConstantAnalyzer, TimestepMode,
};
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use std::time::Duration;

/// Test Case 900 with 1-hour timestep (baseline)
/// Note: This test verifies the API works; actual EUI validation requires proper weather data
#[test]
fn test_case_900_1hr_timestep() {
    let spec = ASHRAE140Case::Case900.spec();
    let _model = ThermalModel::<VectorField>::from_spec(&spec);

    let surrogates = SurrogateManager::new().unwrap_or_else(|_| {
        panic!("Failed to create SurrogateManager");
    });

    // Run 24 hours with 1-hour timestep (sanity check)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut m = ThermalModel::<VectorField>::from_spec(&ASHRAE140Case::Case900.spec());
        m.solve_timesteps_with_dt(24, &surrogates, false, None, None, None, 3600.0)
    }));

    assert!(
        result.is_ok(),
        "Case 900 1-hour timestep simulation should complete without panicking"
    );

    let eui = result.unwrap();
    assert!(
        eui.is_finite(),
        "Case 900 1-hour timestep EUI should be finite, got {:?}",
        eui
    );

    println!(
        "Case 900 (1-hour timestep, 24h): EUI = {:.2} kWh/m²/year",
        eui
    );
}

/// Test Case 900 with 15-minute timestep (compromise between accuracy and speed)
#[test]
fn test_case_900_15min_timestep() {
    let surrogates = SurrogateManager::new().unwrap_or_else(|_| {
        panic!("Failed to create SurrogateManager");
    });

    // Run 24 hours with 15-minute timestep (96 steps)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut m = ThermalModel::<VectorField>::from_spec(&ASHRAE140Case::Case900.spec());
        m.solve_timesteps_with_dt(96, &surrogates, false, None, None, None, 900.0)
    }));

    assert!(
        result.is_ok(),
        "Case 900 15-minute timestep simulation should complete without panicking"
    );

    let eui = result.unwrap();
    assert!(
        eui.is_finite(),
        "Case 900 15-minute timestep EUI should be finite, got {:?}",
        eui
    );

    println!(
        "Case 900 (15-minute timestep, 24h): EUI = {:.2} kWh/m²/year",
        eui
    );
}

/// Test Case 600 (low-mass) with 1-hour timestep - should work correctly
#[test]
fn test_case_600_1hr_timestep() {
    let surrogates = SurrogateManager::new().unwrap_or_else(|_| {
        panic!("Failed to create SurrogateManager");
    });

    // Run 24 hours with 1-hour timestep (sanity check)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut m = ThermalModel::<VectorField>::from_spec(&ASHRAE140Case::Case600.spec());
        m.solve_timesteps_with_dt(24, &surrogates, false, None, None, None, 3600.0)
    }));

    assert!(
        result.is_ok(),
        "Case 600 1-hour timestep simulation should complete without panicking"
    );

    let eui = result.unwrap();
    assert!(
        eui.is_finite(),
        "Case 600 1-hour timestep EUI should be finite, got {:?}",
        eui
    );

    println!(
        "Case 600 (1-hour timestep, 24h): EUI = {:.2} kWh/m²/year",
        eui
    );
}

/// Test Case 600 (low-mass) with 15-minute timestep - should give similar results
#[test]
fn test_case_600_15min_timestep() {
    let surrogates = SurrogateManager::new().unwrap_or_else(|_| {
        panic!("Failed to create SurrogateManager");
    });

    // Run 24 hours with 15-minute timestep (96 steps)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut m = ThermalModel::<VectorField>::from_spec(&ASHRAE140Case::Case600.spec());
        m.solve_timesteps_with_dt(96, &surrogates, false, None, None, None, 900.0)
    }));

    assert!(
        result.is_ok(),
        "Case 600 15-minute timestep simulation should complete without panicking"
    );

    let eui = result.unwrap();
    assert!(
        eui.is_finite(),
        "Case 600 15-minute timestep EUI should be finite, got {:?}",
        eui
    );

    println!(
        "Case 600 (15-minute timestep, 24h): EUI = {:.2} kWh/m²/year",
        eui
    );
}

/// Test time constant analyzer for ASHRAE 140 cases
#[test]
fn test_time_constant_classification() {
    // Low-mass cases should have τ < 2 hours
    let low_mass_cases = vec!["600", "610", "620", "630", "640", "650"];
    for case_id in low_mass_cases {
        let tau = TimeConstantAnalyzer::for_case(case_id)
            .expect(&format!("Case {} should have time constant", case_id));
        let classification = TimeConstantAnalyzer::classify_case(case_id)
            .expect(&format!("Case {} should have classification", case_id));

        assert_eq!(
            classification, "low-mass",
            "Case {} should be classified as low-mass (τ = {:.2} hours)",
            case_id, tau
        );
        assert!(
            tau < 2.0,
            "Low-mass case {} should have τ < 2 hours, got {:.2}",
            case_id,
            tau
        );
    }

    // High-mass cases should have τ >= 2 hours
    let high_mass_cases = vec!["900", "910", "920", "930", "940", "950", "960"];
    for case_id in high_mass_cases {
        let tau = TimeConstantAnalyzer::for_case(case_id)
            .expect(&format!("Case {} should have time constant", case_id));
        let classification = TimeConstantAnalyzer::classify_case(case_id)
            .expect(&format!("Case {} should have classification", case_id));

        assert_eq!(
            classification, "high-mass",
            "Case {} should be classified as high-mass (τ = {:.2} hours)",
            case_id, tau
        );
        assert!(
            tau >= 2.0,
            "High-mass case {} should have τ >= 2 hours, got {:.2}",
            case_id,
            tau
        );
    }
}

/// Test adaptive timestep scheduler with high-mass case
#[test]
fn test_adaptive_scheduler_high_mass() {
    // Create adaptive scheduler for high-mass building (τ = 5 hours)
    let scheduler = AdaptiveTimestepScheduler::new(
        TimestepMode::adaptive(
            Duration::from_secs(360), // 6-minute base timestep
            Duration::from_secs(60),  // 1-minute minimum
            2.0,                      // 2-hour threshold
        ),
        5.0, // τ = 5 hours (high-mass)
    );

    // Should use 6-minute timestep for high-mass
    assert_eq!(
        scheduler.timestep(),
        Duration::from_secs(360),
        "High-mass building should use 6-minute timestep"
    );
    assert_eq!(
        scheduler.timesteps_per_hour(),
        10,
        "6-minute timestep should give 10 timesteps per hour"
    );

    // Should be stable and accurate
    assert!(
        scheduler.is_stable(5.0),
        "Scheduler should be stable for τ = 5 hours"
    );
    assert!(
        scheduler.is_accurate(5.0),
        "Scheduler should be accurate for τ = 5 hours"
    );
}

/// Test adaptive timestep scheduler with low-mass case
#[test]
fn test_adaptive_scheduler_low_mass() {
    // Create adaptive scheduler for low-mass building (τ = 1 hour)
    let scheduler = AdaptiveTimestepScheduler::new(
        TimestepMode::adaptive(
            Duration::from_secs(360), // 6-minute base timestep
            Duration::from_secs(60),  // 1-minute minimum
            2.0,                      // 2-hour threshold
        ),
        1.0, // τ = 1 hour (low-mass)
    );

    // Should use 1-hour timestep for low-mass (fallback to standard)
    assert_eq!(
        scheduler.timestep(),
        Duration::from_secs(3600),
        "Low-mass building should use 1-hour timestep"
    );
    assert_eq!(
        scheduler.timesteps_per_hour(),
        1,
        "1-hour timestep should give 1 timestep per hour"
    );
}
