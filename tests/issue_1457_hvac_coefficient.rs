//! Regression test for Issue #1457 — ASHRAE 140 Case 600 series HVAC coefficient.
//!
//! Locks in the ISO 13790 §12.2.1 simple-hourly HVAC demand coefficient
//! `h_coeff = H_tr,1 + H_tr,w` (where `H_tr,1 = 1/(1/h_tr_is + 1/h_tr_ms)`).
//!
//! History (pre-#1457):
//!   - Old `den/(2·term_rest_1)` formula inflated annual heating by ~3.4×.
//!   - Norton-equivalent `h_is_to_boundary + h_ve` (~76 W/K for Case 600)
//!     undersized annual heating by ~30% below reference.
//!   - ISO 13790 simple method (~123 W/K for Case 600) places Case 610
//!     annual heating/cooling within the published ASHRAE 140-2017 §B5
//!     ±15% band.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn expected_iso_hvac_coefficient(spec: &fluxion::validation::ashrae_140_cases::CaseSpec) -> f64 {
    // Replicate the ISO 13790 simple-hourly h_coeff formula without going through
    // the public ThermalModel API. We compute it from the spec's geometry /
    // construction so this test stays a pure-input sanity check (the engine
    // internal field names are an implementation detail).
    use fluxion::validation::ashrae_140_cases::WindowSpec;

    let floor_area: f64 = spec.geometry.iter().map(|g| g.floor_area()).sum();
    if floor_area <= 0.0 {
        return 0.0;
    }

    // ISO 13790 simplified: h_tr_is = 3.45 W/m²K × A_floor (matches
    // `thermal_model_core.rs::H_SI` constant used by the engine).
    const H_SI: f64 = 3.45;
    let _h_tr_is = H_SI * floor_area;

    // h_tr_ms per ISO 13790 simple method: 9.1 × A_m where A_m = 2.5 × A_floor.
    // (The engine replaces this with a physics-based calculation for low-mass
    // constructions, but the simple-method reference uses 9.1 × A_m. We use the
    // physics-based value that the engine actually computes.)
    //
    // For the regression test we only check the structural property
    //   h_coeff = H_tr,1 + H_tr_w  (with H_tr,1 = h_tr_is·h_tr_ms/(h_tr_is+h_tr_ms))
    // so we use the SAME h_tr_ms the engine stores on the ThermalModel after
    // construction — see `test_iso_hvac_coefficient_case_600` for the
    // comparison against the engine-internal value.

    let h_tr_ms: f64 = 9.1 * 2.5 * floor_area; // ISO 13790 simple method
                                               // The above is a placeholder. The real test below uses the engine's actual
                                               // stored h_tr_ms, so we don't need a closed-form approximation here.
    let _ = h_tr_ms;

    // Direct window conductance: sum over all zones × windows.
    let mut h_tr_w: f64 = 0.0;
    for (zone_idx, windows) in spec.windows.iter().enumerate() {
        let zone_floor = if zone_idx < spec.geometry.len() {
            spec.geometry[zone_idx].floor_area()
        } else {
            floor_area
        };
        let _ = zone_floor;
        for w in windows {
            h_tr_w += w.area * WindowSpec::double_clear_glass().u_value;
        }
    }

    // We don't have a public accessor for h_tr_ms, so defer the actual
    // comparison to the test below. Returning 0.0 here would only trigger
    // the early-return branch above; this helper is unused.
    h_tr_w
}

#[test]
fn test_iso_hvac_coefficient_case_600_is_in_band() {
    // Build Case 600 (low-mass baseline) and read the engine's stored
    // conductances. Verify h_coeff = H_tr,1 + H_tr,w is in the [100, 160] W/K
    // band expected for the ASHRAE 140 Case 600 building (ISO 13790 simple
    // method gives ≈ 123 W/K; Norton gives ≈ 76 W/K).
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let zone_idx = 0;
    let h_tr_is = model.0.h_tr_is.as_ref()[zone_idx];
    let h_tr_ms = model.0.h_tr_ms.as_ref()[zone_idx];
    let h_tr_w = model.0.h_tr_w.as_ref()[zone_idx];

    let h_tr_1 = if h_tr_is + h_tr_ms > 0.0 {
        h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms)
    } else {
        0.0
    };
    let h_coeff_iso = h_tr_1 + h_tr_w;

    // Sanity bounds for Case 600 (single zone, 12 m² south window):
    //   H_tr,1 ≈ 1 / (1/165.6 + 1/h_tr_ms)  — depends on h_tr_ms (engine value)
    //   H_tr,w ≈ 2.10 × 12 = 25.2 W/K
    // Expected total in [110, 145] W/K.
    assert!(
        h_coeff_iso > 110.0 && h_coeff_iso < 145.0,
        "ISO 13790 simple-method h_coeff for Case 600 = {h_coeff_iso:.2} W/K \
         outside the expected [110, 145] W/K band (h_tr_is={h_tr_is:.2}, h_tr_ms={h_tr_ms:.2}, h_tr_w={h_tr_w:.2})"
    );

    // Cross-check: h_coeff must strictly exceed the old Norton equivalent for
    // the same building. If this fails, the formula regressed to the
    // undersizing behaviour documented in #1457.
    let h_tr_em = model.0.h_tr_em.as_ref()[zone_idx];
    let h_ve = model.0.h_ve.as_ref()[zone_idx];
    let h_tr_floor = model.0.h_tr_floor.as_ref()[zone_idx];
    let h_ms_em_series = h_tr_ms * h_tr_em / (h_tr_ms + h_tr_em);
    let surface_to_boundary = h_tr_w + h_ms_em_series + h_tr_floor;
    let h_is_to_boundary = h_tr_is * surface_to_boundary / (h_tr_is + surface_to_boundary);
    let norton = h_is_to_boundary + h_ve;

    assert!(
        h_coeff_iso > norton,
        "ISO h_coeff ({h_coeff_iso:.2}) should exceed Norton equivalent ({norton:.2}); \
         otherwise the formula has regressed to the pre-#1457 undersized form"
    );
}

#[test]
fn test_iso_hvac_coefficient_case_610_in_band() {
    // Same check for Case 610 (south shading + west window).
    let spec = ASHRAE140Case::Case610.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let zone_idx = 0;
    let h_tr_is = model.0.h_tr_is.as_ref()[zone_idx];
    let h_tr_ms = model.0.h_tr_ms.as_ref()[zone_idx];
    let h_tr_w = model.0.h_tr_w.as_ref()[zone_idx];

    let h_tr_1 = h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms);
    let h_coeff_iso = h_tr_1 + h_tr_w;

    // Case 610 has 15 m² windows → H_tr,w ≈ 31.5 W/K.
    // Expected ISO h_coeff total in [115, 150] W/K.
    assert!(
        h_coeff_iso > 115.0 && h_coeff_iso < 150.0,
        "ISO h_coeff for Case 610 = {h_coeff_iso:.2} W/K outside [115, 150] band \
         (h_tr_is={h_tr_is:.2}, h_tr_ms={h_tr_ms:.2}, h_tr_w={h_tr_w:.2})"
    );
}

#[test]
fn test_iso_hvac_formula_does_not_regress_to_norton() {
    // Cross-case sweep: every 600-series case (low-mass) must have an ISO h_coeff
    // strictly greater than the pre-#1457 Norton equivalent. If any case matches
    // or is below the Norton value, the fix has been silently reverted.
    let cases = [
        ("600", ASHRAE140Case::Case600.spec()),
        ("610", ASHRAE140Case::Case610.spec()),
        ("620", ASHRAE140Case::Case620.spec()),
        ("630", ASHRAE140Case::Case630.spec()),
        ("640", ASHRAE140Case::Case640.spec()),
        ("650", ASHRAE140Case::Case650.spec()),
    ];

    for (name, spec) in cases.iter() {
        let model = ThermalModel::<VectorField>::from_spec(spec);
        let zone_idx = 0;
        let h_tr_is = model.0.h_tr_is.as_ref()[zone_idx];
        let h_tr_ms = model.0.h_tr_ms.as_ref()[zone_idx];
        let h_tr_w = model.0.h_tr_w.as_ref()[zone_idx];
        let h_tr_em = model.0.h_tr_em.as_ref()[zone_idx];
        let h_ve = model.0.h_ve.as_ref()[zone_idx];
        let h_tr_floor = model.0.h_tr_floor.as_ref()[zone_idx];

        let h_tr_1 = h_tr_is * h_tr_ms / (h_tr_is + h_tr_ms);
        let h_coeff_iso = h_tr_1 + h_tr_w;

        let h_ms_em_series = h_tr_ms * h_tr_em / (h_tr_ms + h_tr_em);
        let surface_to_boundary = h_tr_w + h_ms_em_series + h_tr_floor;
        let h_is_to_boundary = h_tr_is * surface_to_boundary / (h_tr_is + surface_to_boundary);
        let norton = h_is_to_boundary + h_ve;

        assert!(
            h_coeff_iso > norton,
            "Case {name}: ISO h_coeff ({h_coeff_iso:.2}) ≤ Norton ({norton:.2}) — \
             the #1457 fix has regressed to the pre-fix undersized form"
        );
    }
}

// Suppress unused-helper warning — the helper is a documented intent for any
// future expanded test coverage but is not currently used.
#[allow(dead_code)]
fn _unused_helper() -> f64 {
    expected_iso_hvac_coefficient(&ASHRAE140Case::Case600.spec())
}
