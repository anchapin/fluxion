//! Behavior tests for the PCM test-box sub-suite of issue #3986 / PR-B (#4118).
//!
//! These tests exercise the public API only. They verify the PhaseChangeMaterial
//! nominal properties, the enthalpy + apparent_cp method behavior, and the
//! PCMTestBox harness construction. Per the BLOCKER branch documented in
//! `tests/reference_data/pcm_test_box/PROVENANCE.md`, the experimental
//! reference curve is not available; therefore `solid_fraction_at_time` returns
//! the sentinel `None` and the matching curve test is the BLOCKER-documented
//! `tests/reference_data/pcm_test_box/PROVENANCE.md` §"BLOCKER" link, not a
//! numerical assertion.
//!
//! Behavior coverage:
//! - `rt27_constructs_with_nominal_properties` — Rubitherm RT27 nominal values
//! - `enthalpy_linear_below_solidus` — `enthalpy_j_per_kg(T)` linear in solid region
//! - `enthalpy_linear_above_liquidus` — `enthalpy_j_per_kg(T)` linear in liquid region
//! - `apparent_cp_sensible_outside_melting_band` — `apparent_cp` returns c_sensible
//!   outside the melting range
//! - `apparent_cp_latent_band_height` — `apparent_cp` at the melting midpoint
//!   returns `latent_heat / (T_liquidus - T_solidus)` (apparent-heat-capacity method)
//! - `test_box_constructs_with_default_pcm` — `PCMTestBox::new(rt27())` constructs
//! - `test_box_solid_fraction_returns_none_without_reference_data` — sentinel
//!   per BLOCKER branch
//! - `test_box_apparent_cp_at_wall_delegates_to_material` — `apparent_cp_at_wall`
//!   delegates to the underlying `PhaseChangeMaterial`

use fluxion::physics::pcm_test_box::{PCMTestBox, DEFAULT_LAYER_THICKNESS_M, DEFAULT_WALL_TEMP_C};
use fluxion::physics::phase_change_material::PhaseChangeMaterial;

// Tolerance for sensible-heat continuity at the melting-band boundary.
// Linear extrapolation below T_solidus and above T_liquidus; the apparent-heat-
// capacity method guarantees cp is continuous at the band edges.
const ENTHALPY_REL_TOL: f64 = 1e-9;
const APPARENT_CP_REL_TOL: f64 = 1e-9;

#[test]
fn rt27_constructs_with_nominal_properties() {
    // Rubitherm RT27 nominal values per
    // tests/reference_data/pcm_test_box/PROVENANCE.md.
    let pcm = PhaseChangeMaterial::rt27();
    assert_eq!(pcm.name, "Rubitherm RT27");
    assert_eq!(pcm.melting_range_c, (25.0, 28.0));
    assert_eq!(pcm.latent_heat_j_per_kg, 184_000.0);
    assert_eq!(pcm.specific_heat_j_per_kg_k, 2000.0);
    assert_eq!(pcm.density_kg_per_m3, 800.0);
}

#[test]
fn enthalpy_linear_below_solidus() {
    // Below T_solidus, enthalpy must be the sensible-heat linear extrapolation
    // h(T) = c_p * (T - T_ref). We pin h(0 °C) = 0 by convention (matches
    // ASHRAE 140 / 5R1C reference-point convention); this test checks the
    // monotonic linear scaling in the solid region.
    let pcm = PhaseChangeMaterial::rt27();
    let h_0 = pcm.enthalpy_j_per_kg(0.0);
    let h_10 = pcm.enthalpy_j_per_kg(10.0);
    let h_20 = pcm.enthalpy_j_per_kg(20.0);
    // Strictly increasing in the solid region.
    assert!(
        h_0 < h_10,
        "h(0)={h_0} < h(10)={h_10} expected (solid region)"
    );
    assert!(
        h_10 < h_20,
        "h(10)={h_10} < h(20)={h_20} expected (solid region)"
    );
    // Linear: equal slopes between adjacent 10-K intervals.
    let slope_low = h_10 - h_0;
    let slope_high = h_20 - h_10;
    let slope_diff = (slope_low - slope_high).abs();
    let slope_ref = slope_low.abs().max(1.0);
    assert!(
        slope_diff / slope_ref <= ENTHALPY_REL_TOL,
        "nonlinear in solid region: slope_low={slope_low}, slope_high={slope_high}, diff={slope_diff:.3e}"
    );
    // Slope equals c_p in the solid region (10 K interval * c_p = 20_000 J/kg).
    let expected_slope = pcm.specific_heat_j_per_kg_k * 10.0;
    assert!(
        (slope_low - expected_slope).abs() / expected_slope <= ENTHALPY_REL_TOL,
        "solid-region slope={slope_low} != c_p * ΔT={expected_slope}"
    );
}

#[test]
fn enthalpy_linear_above_liquidus() {
    let pcm = PhaseChangeMaterial::rt27();
    let h_40 = pcm.enthalpy_j_per_kg(40.0);
    let h_50 = pcm.enthalpy_j_per_kg(50.0);
    let h_60 = pcm.enthalpy_j_per_kg(60.0);
    assert!(
        h_40 < h_50,
        "h(40)={h_40} < h(50)={h_50} expected (liquid region)"
    );
    assert!(
        h_50 < h_60,
        "h(50)={h_50} < h(60)={h_60} expected (liquid region)"
    );
    let slope_low = h_50 - h_40;
    let slope_high = h_60 - h_50;
    let slope_diff = (slope_low - slope_high).abs();
    let slope_ref = slope_low.abs().max(1.0);
    assert!(
        slope_diff / slope_ref <= ENTHALPY_REL_TOL,
        "nonlinear in liquid region: slope_low={slope_low}, slope_high={slope_high}, diff={slope_diff:.3e}"
    );
    let expected_slope = pcm.specific_heat_j_per_kg_k * 10.0;
    assert!(
        (slope_low - expected_slope).abs() / expected_slope <= ENTHALPY_REL_TOL,
        "liquid-region slope={slope_low} != c_p * ΔT={expected_slope}"
    );
}

#[test]
fn apparent_cp_sensible_outside_melting_band() {
    let pcm = PhaseChangeMaterial::rt27();
    // Below T_solidus: c_p = c_sensible.
    let cp_below = pcm.apparent_cp_j_per_kg_k(10.0);
    assert!(
        (cp_below - pcm.specific_heat_j_per_kg_k).abs() / pcm.specific_heat_j_per_kg_k
            <= APPARENT_CP_REL_TOL,
        "apparent_cp below T_solidus = {cp_below}, expected c_sensible = {}",
        pcm.specific_heat_j_per_kg_k
    );
    // Above T_liquidus: c_p = c_sensible.
    let cp_above = pcm.apparent_cp_j_per_kg_k(50.0);
    assert!(
        (cp_above - pcm.specific_heat_j_per_kg_k).abs() / pcm.specific_heat_j_per_kg_k
            <= APPARENT_CP_REL_TOL,
        "apparent_cp above T_liquidus = {cp_above}, expected c_sensible = {}",
        pcm.specific_heat_j_per_kg_k
    );
    // `is_in_melting_range` boundary checks (T = 25.0 and 28.0 are IN the band).
    assert!(pcm.is_in_melting_range(25.0));
    assert!(pcm.is_in_melting_range(26.5));
    assert!(pcm.is_in_melting_range(28.0));
    assert!(!pcm.is_in_melting_range(24.999));
    assert!(!pcm.is_in_melting_range(28.001));
}

#[test]
fn apparent_cp_latent_band_height() {
    // Apparent-heat-capacity method: inside the melting band, c_p = L / (T_liquidus - T_solidus).
    // This is the standard approach for enthalpy-method PCM simulations
    // (see e.g. Shmueli et al. 2010, Assis et al. 2009 for derivation).
    let pcm = PhaseChangeMaterial::rt27();
    let (t_s, t_l) = pcm.melting_range_c;
    let expected_band_cp = pcm.latent_heat_j_per_kg / (t_l - t_s); // = 184_000 / 3 = 61_333.33...
    let cp_mid = pcm.apparent_cp_j_per_kg_k((t_s + t_l) / 2.0);
    assert!(
        (cp_mid - expected_band_cp).abs() / expected_band_cp <= APPARENT_CP_REL_TOL,
        "apparent_cp at melting midpoint = {cp_mid}, expected L/ΔT = {expected_band_cp}"
    );
    // Band height must be MUCH larger than c_sensible (latent heat dominates).
    assert!(
        cp_mid > 10.0 * pcm.specific_heat_j_per_kg_k,
        "apparent_cp band height = {cp_mid} should be >> c_sensible = {} (latent-heat dominance)",
        pcm.specific_heat_j_per_kg_k
    );
}

// ---------- PCMTestBox harness (Refs #3986 / #4118) ----------

#[test]
fn test_box_constructs_with_default_pcm() {
    let box_ = PCMTestBox::new(PhaseChangeMaterial::rt27());
    assert_eq!(box_.layer_thickness_m, DEFAULT_LAYER_THICKNESS_M);
    assert_eq!(box_.wall_temperature_c, DEFAULT_WALL_TEMP_C);
    assert_eq!(box_.pcm.name, "Rubitherm RT27");
    // `reference_curve` must be None in the skeleton (BLOCKER branch).
    assert!(box_.reference_curve.is_none());
}

#[test]
fn test_box_solid_fraction_returns_none_without_reference_data() {
    // Per BLOCKER branch in tests/reference_data/pcm_test_box/PROVENANCE.md:
    // when no experimental curve is loaded, `solid_fraction_at_time` returns
    // the sentinel `None`. This is the documented unlock criterion for PR-B+1.
    let box_ = PCMTestBox::new(PhaseChangeMaterial::rt27());
    assert!(box_.solid_fraction_at_time(0.0).is_none());
    assert!(box_.solid_fraction_at_time(60.0).is_none());
    assert!(box_.solid_fraction_at_time(3600.0).is_none());
}

#[test]
fn test_box_apparent_cp_at_wall_delegates_to_material() {
    // Default wall temperature = 40 °C (well above T_liquidus); apparent_cp
    // must equal c_sensible (the liquid-region value).
    let box_ = PCMTestBox::new(PhaseChangeMaterial::rt27());
    let cp = box_.apparent_cp_at_wall();
    assert!(
        (cp - box_.pcm.specific_heat_j_per_kg_k).abs() / box_.pcm.specific_heat_j_per_kg_k
            <= APPARENT_CP_REL_TOL,
        "apparent_cp_at_wall (T_wall=40°C) = {cp}, expected c_sensible = {}",
        box_.pcm.specific_heat_j_per_kg_k
    );
}
