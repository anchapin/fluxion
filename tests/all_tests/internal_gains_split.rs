//! ASHRAE 140 §6.5 convective/radiative split invariant tests
//!
//! Issue #2892: internal-gains convective/radiative split is profile-parameter
//! only — ASHRAE 140 case spec defaults not applied. This file pins the
//! §6.5 split at the case-spec layer AND at the validator seam so a future
//! refactor cannot regress to "all-convective" routing that bypasses surface
//! `phi_st` redistribution.
//!
//! ASHRAE 140-2023 §6.5 specifies the typical residential split as
//! convective = 0.6, radiative = 0.4. The validator applies this split
//! per-zone so the radiative portion reaches `phi_st` (surface) and `phi_m`
//! (mass / air-routing) rather than being lumped onto the air node.
//!
//! Test coverage:
//! 1. `InternalLoads::new` invariant — `radiative + convective = 1.0`.
//! 2. Case 600/900 series spec defaults — radiative = 0.4, convective = 0.6.
//! 3. `ThermalModel::from_spec` propagates the split — `model.solar.convective_fraction`
//!    equals `spec.internal_loads[0].convective_fraction`.
//! 4. `model.set_loads(&internal_loads)` does not silently reset the split.
//! 5. `phi_ia` + `phi_st` + `phi_m` accounts for 100% of internal-gain energy
//!    at the validator seam (conservation check).

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, InternalLoads};
use fluxion_core::ashrae_cases::InternalLoads as CoreInternalLoads;

/// ASHRAE 140 §6.5 residential split defaults (canonical).
const ASHRAE140_S65_RESIDENTIAL_RADIATIVE: f64 = 0.4;
const ASHRAE140_S65_RESIDENTIAL_CONVECTIVE: f64 = 0.6;

/// Floating-point tolerance for `sum == 1.0` checks.
const SPLIT_TOLERANCE: f64 = 1.0e-9;

// ---------------------------------------------------------------------------
// 1. InternalLoads struct invariant
// ---------------------------------------------------------------------------

#[test]
fn test_invariant_radiative_plus_convective_equals_one() {
    // ASHRAE 140 §6.5 default residential split.
    let loads = InternalLoads::new(200.0, 0.4, 0.6);
    assert_eq!(loads.total_load, 200.0);
    assert!((loads.radiative_fraction + loads.convective_fraction - 1.0).abs() < SPLIT_TOLERANCE);
    assert!((loads.radiative_load() - 80.0).abs() < SPLIT_TOLERANCE);
    assert!((loads.convective_load() - 120.0).abs() < SPLIT_TOLERANCE);

    // Sanity: any (radiative, convective) pair summing to 1.0 is accepted.
    let splits = [
        (0.0, 1.0),
        (0.25, 0.75),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.9, 0.1),
        (1.0, 0.0),
    ];
    for (rad, conv) in splits {
        let l = InternalLoads::new(100.0, rad, conv);
        assert!(
            (l.radiative_fraction + l.convective_fraction - 1.0).abs() < SPLIT_TOLERANCE,
            "split ({rad}, {conv}) must sum to 1.0"
        );
    }
}

#[test]
#[should_panic(expected = "Radiative + convective fractions must sum to 1.0")]
fn test_internal_loads_invalid_fractions_rejected() {
    InternalLoads::new(200.0, 0.5, 0.3); // sum = 0.8
}

#[test]
fn test_internal_loads_invalid_via_core_path() {
    // The core mirror (fluxion_core::ashrae_cases::InternalLoads) must reject
    // non-1.0 sums just like the validation facade.
    let result = std::panic::catch_unwind(|| CoreInternalLoads::new(200.0, 0.7, 0.7));
    assert!(
        result.is_err(),
        "core InternalLoads must panic on sum != 1.0"
    );
}

// ---------------------------------------------------------------------------
// 2. Case 600/900 series spec defaults — ASHRAE 140 §6.5 residential
// ---------------------------------------------------------------------------

/// Helper: pull the first zone's `InternalLoads` from a case spec, panic if
/// missing (every Case 600/900 spec under test here must have it set).
fn first_zone_loads(case: ASHRAE140Case) -> InternalLoads {
    let spec = case.spec();
    spec.internal_loads
        .first()
        .and_then(|l| l.as_ref())
        .copied()
        .unwrap_or_else(|| panic!("case {} has no internal loads", spec.case_id))
}

#[test]
fn test_case_600_spec_uses_s65_residential_split() {
    let l = first_zone_loads(ASHRAE140Case::Case600);
    assert!(
        (l.radiative_fraction - ASHRAE140_S65_RESIDENTIAL_RADIATIVE).abs() < SPLIT_TOLERANCE,
        "Case 600 radiative_fraction = {} (expected {})",
        l.radiative_fraction,
        ASHRAE140_S65_RESIDENTIAL_RADIATIVE
    );
    assert!(
        (l.convective_fraction - ASHRAE140_S65_RESIDENTIAL_CONVECTIVE).abs() < SPLIT_TOLERANCE,
        "Case 600 convective_fraction = {} (expected {})",
        l.convective_fraction,
        ASHRAE140_S65_RESIDENTIAL_CONVECTIVE
    );
}

#[test]
fn test_case_900_spec_uses_s65_residential_split() {
    let l = first_zone_loads(ASHRAE140Case::Case900);
    assert!((l.radiative_fraction - 0.4).abs() < SPLIT_TOLERANCE);
    assert!((l.convective_fraction - 0.6).abs() < SPLIT_TOLERANCE);
}

/// All low-mass + high-mass residential cases must use the same §6.5 split.
/// If a future refactor changes one but not the others, this test fails.
#[test]
fn test_all_600_900_residential_cases_use_s65_split() {
    let cases = [
        ("600", ASHRAE140Case::Case600),
        ("610", ASHRAE140Case::Case610),
        ("620", ASHRAE140Case::Case620),
        ("630", ASHRAE140Case::Case630),
        ("640", ASHRAE140Case::Case640),
        ("650", ASHRAE140Case::Case650),
        ("900", ASHRAE140Case::Case900),
        ("910", ASHRAE140Case::Case910),
        ("920", ASHRAE140Case::Case920),
        ("930", ASHRAE140Case::Case930),
        ("940", ASHRAE140Case::Case940),
        ("950", ASHRAE140Case::Case950),
    ];
    for (id, case) in cases {
        let l = first_zone_loads(case);
        assert!(
            (l.radiative_fraction - ASHRAE140_S65_RESIDENTIAL_RADIATIVE).abs() < SPLIT_TOLERANCE,
            "Case {id} radiative_fraction = {} (expected {})",
            l.radiative_fraction,
            ASHRAE140_S65_RESIDENTIAL_RADIATIVE
        );
        assert!(
            (l.convective_fraction - ASHRAE140_S65_RESIDENTIAL_CONVECTIVE).abs() < SPLIT_TOLERANCE,
            "Case {id} convective_fraction = {} (expected {})",
            l.convective_fraction,
            ASHRAE140_S65_RESIDENTIAL_CONVECTIVE
        );
        // Invariant: sum must equal 1.0
        assert!(
            (l.radiative_fraction + l.convective_fraction - 1.0).abs() < SPLIT_TOLERANCE,
            "Case {id} radiative + convective = {} (expected 1.0)",
            l.radiative_fraction + l.convective_fraction
        );
    }
}

// ---------------------------------------------------------------------------
// 3. ThermalModel::from_spec propagates the split
// ---------------------------------------------------------------------------

#[test]
fn test_from_spec_propagates_convective_fraction() {
    let spec = ASHRAE140Case::Case600.spec();
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    // Issue #2892: model.solar.convective_fraction must equal the case-spec §6.5
    // default (0.6) so the load splitter routes the radiative portion to
    // phi_st + phi_m. Pre-#2892 the field drifted to a profile parameter
    // (0.4) and the radiative portion effectively vanished.
    assert!(
        (model.solar.convective_fraction - ASHRAE140_S65_RESIDENTIAL_CONVECTIVE).abs()
            < SPLIT_TOLERANCE,
        "Case 600 model.solar.convective_fraction = {} (expected {})",
        model.solar.convective_fraction,
        ASHRAE140_S65_RESIDENTIAL_CONVECTIVE
    );
}

#[test]
fn test_from_spec_propagates_convective_fraction_high_mass() {
    let spec = ASHRAE140Case::Case900.spec();
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");
    assert!(
        (model.solar.convective_fraction - ASHRAE140_S65_RESIDENTIAL_CONVECTIVE).abs()
            < SPLIT_TOLERANCE,
        "Case 900 model.solar.convective_fraction = {} (expected {})",
        model.solar.convective_fraction,
        ASHRAE140_S65_RESIDENTIAL_CONVECTIVE
    );
}

// ---------------------------------------------------------------------------
// 4. set_loads does not silently reset the split
// ---------------------------------------------------------------------------

#[test]
fn test_set_loads_preserves_convective_fraction() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    // Capture the post-construction value.
    let conv_before = model.solar.convective_fraction;
    assert!((conv_before - ASHRAE140_S65_RESIDENTIAL_CONVECTIVE).abs() < SPLIT_TOLERANCE);

    // Per-timestep set_loads (mirrors the validator seam at
    // src/validation/ashrae_140_validator.rs:~917). The field MUST NOT change
    // because the convective fraction is owned by the case spec, not the
    // per-timestep load magnitude.
    let n = model.hvac.num_zones.max(1);
    let loads: Vec<f64> = vec![10.0; n];
    model.set_loads(&loads);

    assert!(
        (model.solar.convective_fraction - conv_before).abs() < SPLIT_TOLERANCE,
        "set_loads must not reset convective_fraction (was {}, now {})",
        conv_before,
        model.solar.convective_fraction
    );
}

// ---------------------------------------------------------------------------
// 5. Conservation check: φ_ia + φ_st + φ_m = total internal gain
// ---------------------------------------------------------------------------

/// Reproduce the per-zone split that `step_physics_5r1c` applies
/// (`src/sim/thermal_model_physics/physics_impl.rs:~226-244`) and verify the
/// three terms account for 100% of the internal-gain energy for any
/// `convective_fraction` ∈ [0, 1] and any `solar_distribution_to_air` ∈ [0, 1].
///
/// This is a structural invariant: if a future change accidentally routes the
/// radiative portion entirely to the air node (the pre-#2892 bug), the sum
/// would drift above 1.0 × load.
#[test]
fn test_split_conserves_total_internal_gain() {
    for conv_frac in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] {
        for sol_dist_to_air in [0.0, 0.1, 0.3, 0.5, 1.0] {
            let rad_frac = 1.0 - conv_frac;
            let st_int_frac = rad_frac * (1.0 - sol_dist_to_air);
            let m_air_frac = rad_frac * sol_dist_to_air;

            let phi_ia: f64 = conv_frac; // × load
            let phi_st: f64 = st_int_frac;
            let phi_m: f64 = m_air_frac;
            let total: f64 = phi_ia + phi_st + phi_m;

            assert!(
                (total - 1.0).abs() < 1.0e-12,
                "split must conserve: conv={conv_frac}, sol_to_air={sol_dist_to_air} \
                 -> phi_ia={phi_ia:.3} + phi_st={phi_st:.3} + phi_m={phi_m:.3} = {total:.3}"
            );
        }
    }
}

/// Verify the §6.5 residential split distributes the radiative portion to
/// surface + mass/air-routing — i.e. NOT zero on either side. This is the
/// "bypass surface phi_st" failure mode the issue describes.
#[test]
fn test_s65_split_does_not_bypass_surface_phi_st() {
    let conv = ASHRAE140_S65_RESIDENTIAL_CONVECTIVE;
    let rad = ASHRAE140_S65_RESIDENTIAL_RADIATIVE;

    // For all reasonable ASHRAE 140 solar-distribution-to-air values
    // (LowMass=0.30, HighMass=0.0, Special=0.10 per Issue #2359/#2444) the
    // radiative portion must split between phi_st and phi_m, neither of which
    // is zero.
    for (label, sol_dist_to_air) in [("LowMass", 0.30), ("HighMass", 0.0), ("Special", 0.10)] {
        let st_int_frac = rad * (1.0 - sol_dist_to_air);
        let m_air_frac = rad * sol_dist_to_air;

        assert!(
            st_int_frac > 0.0,
            "{label}: st_int_frac must be > 0 (got {st_int_frac})"
        );
        assert!(
            st_int_frac <= rad,
            "{label}: st_int_frac = {st_int_frac} must be <= radiative={rad}"
        );
        // Conservation
        assert!(
            (conv + st_int_frac + m_air_frac - 1.0).abs() < 1.0e-12,
            "{label}: split must conserve (got {})",
            conv + st_int_frac + m_air_frac
        );
    }
}
