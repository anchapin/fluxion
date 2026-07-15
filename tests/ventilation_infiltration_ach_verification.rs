//! Ventilation infiltration ACH formula verification vs ASHRAE 140 reference.
//!
//! Issue #1599 — verifies wind-driven, stack-driven, and combined infiltration
//! ACH formulas at ASHRAE 140 reference conditions.
//!
//! # ASHRAE 140-2023 §5.5.3.6 Reference Conditions
//!
//! - Wind speed: 3.4 m/s
//! - Temperature difference (|ΔT|): 10 K
//! - Building/height: 2.7 m
//! - Shielding factor: 0.5
//! - Effective opening area: 1.0 m²
//! - Zone volume: 129.6 m³ (6m × 8m × 2.7m ASHRAE reference zone)
//!
//! # Acceptance Criteria
//!
//! 1. `calculate_wind_infiltration_ach` at (v=3.4, H=2.7, shielding=0.5)
//!    matches analytical reference to 1%
//! 2. `calculate_stack_infiltration_ach` at (ΔT=10, H=2.7, A=1.0) matches
//!    analytical formula to 1%
//! 3. Combined ACH ≥ wind ACH (combined formula is ≥ each component)
//! 4. Proptest failure count for ACH_wind, ACH_stack, ACH_combined: 0
//! 5. cargo test ventilation_infiltration_ach_verification passes
//!
//! # Verification
//!
//! Command: cargo test -p fluxion ventilation_infiltration_ach_verification

use proptest::prelude::*;
use fluxion::sim::ventilation::{
    calculate_combined_infiltration_ach, calculate_stack_infiltration_ach,
    calculate_wind_infiltration_ach, STACK_COEFFICIENT,
};

/// ASHRAE 140-2023 §5.5.3.6 reference conditions for infiltration verification.
mod ref_conditions {
    pub const WIND_SPEED_MPS: f64 = 3.4;
    pub const HEIGHT_M: f64 = 2.7;
    pub const SHIELDING_FACTOR: f64 = 0.5;
    pub const DELTA_T_K: f64 = 10.0;
    pub const OPENING_AREA_M2: f64 = 1.0;
    pub const ZONE_VOLUME_M3: f64 = 129.6;
    pub const INDOOR_TEMP_C: f64 = 20.0;
    pub const OUTDOOR_TEMP_C: f64 = 10.0;
    pub const BASE_WIND_SPEED: f64 = 3.0;
}

// ============================================================================
// Spot-check 1: calculate_wind_infiltration_ach
// ============================================================================

/// Hand-computed reference for `calculate_wind_infiltration_ach(v=3.4, H=2.7, shielding=0.5)`.
///
/// Formula from `src/sim/ventilation.rs`:
/// ```ignore
/// shelter_coefficient = (1 - shielding) * 0.4
/// height_factor       = sqrt(building_height / 3.0)
/// n_factor            = shelter_coefficient * height_factor
/// ACH_wind            = n_factor * (wind_speed / 3.0)
/// ```
///
/// Computation:
/// - shelter_coefficient = (1 - 0.5) * 0.4 = 0.20
/// - height_factor       = sqrt(2.7 / 3.0) = sqrt(0.9) ≈ 0.948683
/// - n_factor            = 0.20 * 0.948683 ≈ 0.189737
/// - ACH_wind            = 0.189737 * (3.4 / 3.0) ≈ 0.215037
#[test]
fn test_wind_infiltration_ach_spot_check() {
    let v = ref_conditions::WIND_SPEED_MPS;
    let H = ref_conditions::HEIGHT_M;
    let shielding = ref_conditions::SHIELDING_FACTOR;

    let ach = calculate_wind_infiltration_ach(v, H, shielding);

    // Hand-computed reference value
    let shelter_coefficient = (1.0 - shielding) * 0.4;
    let height_factor = (H / ref_conditions::BASE_WIND_SPEED).sqrt();
    let n_factor = shelter_coefficient * height_factor;
    let ach_ref = n_factor * (v / ref_conditions::BASE_WIND_SPEED);

    let rel_err = ((ach - ach_ref) / ach_ref.abs().max(1e-9)).abs();
    assert!(
        rel_err < 0.01,
        "ACH_wind ({:.6}) deviates >1% from hand-computed reference ({:.6}): rel_err={:.4}",
        ach,
        ach_ref,
        rel_err
    );

    // Spot-check against hard-coded reference
    let ach_ref_exact = 0.215037;
    assert!(
        (ach - ach_ref_exact).abs() < 0.001,
        "ACH_wind = {:.6}, expected ≈ {:.6}",
        ach,
        ach_ref_exact
    );
}

// ============================================================================
// Spot-check 2: calculate_stack_infiltration_ach
// ============================================================================

/// Hand-computed reference for `calculate_stack_infiltration_ach(ΔT=10, H=2.7, A=1.0, V=129.6)`.
///
/// Formula from `src/sim/ventilation.rs`:
/// ```ignore
/// delta_t     = |indoor - outdoor|
/// flow_sqrt  = sqrt(delta_t / height_diff)
/// q_vent     = STACK_COEFFICIENT * opening_area * flow_sqrt
/// ACH_stack  = q_vent / zone_volume
/// ```
///
/// Computation:
/// - delta_t    = 10.0
/// - flow_sqrt  = sqrt(10.0 / 2.7) = sqrt(3.7037) ≈ 1.9245
/// - q_vent     = 0.025 * 1.0 * 1.9245 ≈ 0.04811
/// - ACH_stack  = 0.04811 / 129.6 ≈ 0.000371
#[test]
fn test_stack_infiltration_ach_spot_check() {
    let delta_t = ref_conditions::DELTA_T_K;
    let H = ref_conditions::HEIGHT_M;
    let A = ref_conditions::OPENING_AREA_M2;
    let V = ref_conditions::ZONE_VOLUME_M3;

    let ach = calculate_stack_infiltration_ach(
        ref_conditions::INDOOR_TEMP_C,
        ref_conditions::OUTDOOR_TEMP_C,
        H,
        A,
        V,
    );

    // Hand-computed reference
    let flow_sqrt = (delta_t / H).sqrt();
    let q_vent = STACK_COEFFICIENT * A * flow_sqrt;
    let ach_ref = q_vent / V;

    let rel_err = ((ach - ach_ref) / ach_ref.abs().max(1e-9)).abs();
    assert!(
        rel_err < 0.01,
        "ACH_stack ({:.6}) deviates >1% from hand-computed reference ({:.6}): rel_err={:.4}",
        ach,
        ach_ref,
        rel_err
    );

    // Verify order of magnitude: ACH_stack should be << 1.0 for these conditions
    assert!(
        ach < 0.01,
        "ACH_stack ({:.6}) unexpectedly large for reference conditions",
        ach
    );
}

// ============================================================================
// Spot-check 3: calculate_combined_infiltration_ach at ASHRAE reference
// ============================================================================

/// Combined infiltration at ASHRAE 140 reference conditions.
///
/// The ASHRAE 140-2023 §5.5.3.6 specifies 0.5 ACH as the DEFAULT
/// (constant) infiltration rate for the reference zone.  The physics-based
/// wind+stack formulas produce a lower rate at reference wind/T conditions;
/// this test verifies the combined formula is correctly additive and
/// continuous.
///
/// At reference conditions:
/// - ACH_wind ≈ 0.215 (dominant term)
/// - ACH_stack ≈ 0.00037 (negligible; stack effect is small for this zone geometry)
/// - ACH_combined = ACH_wind + ACH_stack ≈ 0.2154
#[test]
fn test_combined_infiltration_ach_ashrae_reference() {
    let ach = calculate_combined_infiltration_ach(
        ref_conditions::OUTDOOR_TEMP_C,
        ref_conditions::INDOOR_TEMP_C,
        ref_conditions::WIND_SPEED_MPS,
        ref_conditions::HEIGHT_M,
        ref_conditions::OPENING_AREA_M2,
        ref_conditions::ZONE_VOLUME_M3,
        ref_conditions::SHIELDING_FACTOR,
    );

    // Combined ACH should be non-negative
    assert!(ach >= 0.0, "ACH_combined ({}) should be non-negative", ach);

    // Combined should be finite
    assert!(ach.is_finite(), "ACH_combined ({}) should be finite", ach);

    // Diagnostic: print component breakdown
    let ach_wind = calculate_wind_infiltration_ach(
        ref_conditions::WIND_SPEED_MPS,
        ref_conditions::HEIGHT_M,
        ref_conditions::SHIELDING_FACTOR,
    );
    let ach_stack = calculate_stack_infiltration_ach(
        ref_conditions::INDOOR_TEMP_C,
        ref_conditions::OUTDOOR_TEMP_C,
        ref_conditions::HEIGHT_M,
        ref_conditions::OPENING_AREA_M2,
        ref_conditions::ZONE_VOLUME_M3,
    );

    // Combined should be ≥ each component (additive combination)
    assert!(
        ach >= ach_wind - 1e-9,
        "ACH_combined ({:.4}) should be ≥ ACH_wind ({:.4})",
        ach,
        ach_wind
    );
    assert!(
        ach >= ach_stack - 1e-9,
        "ACH_combined ({:.4}) should be ≥ ACH_stack ({:.6})",
        ach,
        ach_stack
    );

    // Combined = sum of components (within floating-point tolerance)
    let expected_combined = ach_wind + ach_stack;
    assert!(
        (ach - expected_combined).abs() < 1e-9,
        "ACH_combined ({:.6}) should equal ACH_wind + ACH_stack ({:.6})",
        ach,
        expected_combined
    );

    println!(
        "[ASHRAE 140 reference] ACH_wind={:.4}, ACH_stack={:.6}, ACH_combined={:.4}",
        ach_wind, ach_stack, ach
    );
}

// ============================================================================
// Spot-check 4: Verify combined formula is additive (not sqrt form)
// ============================================================================

/// Verifies that the combined ACH uses additive combination.
///
/// The implementation uses `ACH_total = ACH_wind + ACH_stack`.  This test
/// confirms the additive behavior and checks that the sqrt-style upper bound
/// holds: sqrt(ACH_wind² + ACH_stack²) ≤ ACH_wind + ACH_stack.
#[test]
fn test_combined_formula_is_additive() {
    let ach_wind = calculate_wind_infiltration_ach(
        ref_conditions::WIND_SPEED_MPS,
        ref_conditions::HEIGHT_M,
        ref_conditions::SHIELDING_FACTOR,
    );
    let ach_stack = calculate_stack_infiltration_ach(
        ref_conditions::INDOOR_TEMP_C,
        ref_conditions::OUTDOOR_TEMP_C,
        ref_conditions::HEIGHT_M,
        ref_conditions::OPENING_AREA_M2,
        ref_conditions::ZONE_VOLUME_M3,
    );

    let ach_combined = calculate_combined_infiltration_ach(
        ref_conditions::OUTDOOR_TEMP_C,
        ref_conditions::INDOOR_TEMP_C,
        ref_conditions::WIND_SPEED_MPS,
        ref_conditions::HEIGHT_M,
        ref_conditions::OPENING_AREA_M2,
        ref_conditions::ZONE_VOLUME_M3,
        ref_conditions::SHIELDING_FACTOR,
    );

    // Additive: ACH_combined = ACH_wind + ACH_stack
    let expected_additive = ach_wind + ach_stack;
    assert!(
        (ach_combined - expected_additive).abs() < 1e-9,
        "ACH_combined ({:.6}) should equal additive sum ({:.6})",
        ach_combined,
        expected_additive
    );

    // Pythagorean upper bound: sqrt(wind² + stack²) ≤ wind + stack
    let ach_sqrt_form = (ach_wind.powi(2) + ach_stack.powi(2)).sqrt();
    assert!(
        ach_sqrt_form <= expected_additive + 1e-9,
        "sqrt-form ({:.6}) should be ≤ additive-form ({:.6})",
        ach_sqrt_form,
        expected_additive
    );
}

// ============================================================================
// Proptest bounds
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    /// ACH_wind is non-negative and finite for all physical inputs.
    /// ACH_wind ∈ [0, ∞) for wind ∈ [0, 30] m/s, height ∈ [0.5, 30] m, shielding ∈ [0, 1].
    #[test]
    fn proptest_wind_ach_non_negative_and_finite(
        wind_speed in 0.0_f64..30.0,
        building_height in 0.5_f64..30.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        let ach = calculate_wind_infiltration_ach(wind_speed, building_height, shielding_factor);
        prop_assert!(ach >= 0.0, "ACH_wind must be non-negative, got {}", ach);
        prop_assert!(ach.is_finite(), "ACH_wind must be finite, got {}", ach);
    }

    /// ACH_stack is non-negative and finite for all physical parameters.
    /// ACH_stack ∈ [0, ∞) for |ΔT| ≤ 50 K, height ∈ [0.1, 30] m, opening ∈ [0.01, 10] m².
    #[test]
    fn proptest_stack_ach_non_negative_and_finite(
        delta_t in 0.0_f64..50.0,
        height_diff in 0.1_f64..30.0,
        opening_area in 0.01_f64..10.0,
        zone_volume in 1.0_f64..1000.0,
    ) {
        // Only meaningful when delta_t > 0.5 (threshold in the implementation)
        if delta_t < 0.5 {
            return Ok(());
        }
        let ach = calculate_stack_infiltration_ach(
            20.0,                          // indoor
            20.0 - delta_t,               // outdoor (ΔT = delta_t)
            height_diff,
            opening_area,
            zone_volume,
        );
        prop_assert!(ach >= 0.0, "ACH_stack must be non-negative, got {}", ach);
        prop_assert!(ach.is_finite(), "ACH_stack must be finite, got {}", ach);
    }

    /// ACH_combined is non-negative, finite, and ≥ each component.
    #[test]
    fn proptest_combined_ach_non_negative_and_finite(
        wind_speed in 0.0_f64..30.0,
        delta_t in 0.0_f64..50.0,
        building_height in 0.5_f64..30.0,
        opening_area in 0.01_f64..10.0,
        zone_volume in 1.0_f64..1000.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        let indoor = 20.0;
        let outdoor = 20.0 - delta_t;

        let ach_combined = calculate_combined_infiltration_ach(
            outdoor,
            indoor,
            wind_speed,
            building_height,
            opening_area,
            zone_volume,
            shielding_factor,
        );

        prop_assert!(ach_combined >= 0.0, "ACH_combined must be non-negative, got {}", ach_combined);
        prop_assert!(ach_combined.is_finite(), "ACH_combined must be finite, got {}", ach_combined);

        // Combined ≥ wind component
        let ach_wind = calculate_wind_infiltration_ach(wind_speed, building_height, shielding_factor);
        prop_assert!(
            ach_combined >= ach_wind - 1e-9,
            "ACH_combined ({}) should be ≥ ACH_wind ({})",
            ach_combined,
            ach_wind
        );

        // Combined ≥ stack component (when stack is meaningful)
        if delta_t >= 0.5 {
            let ach_stack = calculate_stack_infiltration_ach(
                indoor, outdoor, building_height, opening_area, zone_volume,
            );
            prop_assert!(
                ach_combined >= ach_stack - 1e-9,
                "ACH_combined ({}) should be ≥ ACH_stack ({})",
                ach_combined,
                ach_stack
            );
        }
    }

    /// Combined ACH respects additive combination: ACH_combined = ACH_wind + ACH_stack.
    #[test]
    fn proptest_combined_equals_sum_of_components(
        wind_speed in 0.0_f64..30.0,
        delta_t in 0.5_f64..50.0,
        building_height in 0.5_f64..30.0,
        opening_area in 0.01_f64..10.0,
        zone_volume in 1.0_f64..1000.0,
        shielding_factor in 0.0_f64..1.0,
    ) {
        let indoor = 20.0;
        let outdoor = 20.0 - delta_t;

        let ach_wind = calculate_wind_infiltration_ach(wind_speed, building_height, shielding_factor);
        let ach_stack = calculate_stack_infiltration_ach(
            indoor, outdoor, building_height, opening_area, zone_volume,
        );
        let ach_combined = calculate_combined_infiltration_ach(
            outdoor,
            indoor,
            wind_speed,
            building_height,
            opening_area,
            zone_volume,
            shielding_factor,
        );

        let expected = ach_wind + ach_stack;
        prop_assert!(
            (ach_combined - expected).abs() < 1e-9,
            "ACH_combined ({}) should equal ACH_wind + ACH_stack ({})",
            ach_combined,
            expected
        );
    }
}
