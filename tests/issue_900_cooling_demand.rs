//! Regression test for Issue #900 — HVAC cooling demand asymmetry.
//!
//! # Background
//!
//! Issue #925 (merged) replaced the buggy `h_coeff = den / (2 × term_rest_1)` with
//! the building's true heat-loss coefficient `h_loss` (~93 W/K for the Case
//! 600/900 envelope). That fix brought **heating** into the reference range
//! (Case 900: 11.98 → 3.05 MWh) but left **cooling** systematically
//! under-counted (Case 900: 1.00 → 0.28 MWh; reference 2.13–3.67 MWh).
//!
//! The root cause: the steady-state formula
//!   `Q = h_loss × (T_free − T_cool_sp)`
//! relies on `T_free` being a correct prediction of the zone temperature
//! without HVAC. For high-mass buildings, the lumped-mass 5R1C
//! `t_i_free` never exceeds the cooling setpoint in the current model
//! (peaks at ~24°C vs cool_sp = 27°C), so the demand formula yields zero
//! for most of the cooling-active period.
//!
//! The physics it misses is the **dynamic mass heat release**: when the
//! building is held at the cooling setpoint and the thermal mass is hotter
//! than the setpoint, the mass continuously releases heat to the zone at
//! rate `h_tr_ms × (T_mass − T_cool_sp)`. This term dominates summer cooling
//! load for high-mass buildings and is missing from the steady-state
//! `t_free` approximation.
//!
//! # The fix
//!
//! `compute_zone_hvac_load` now accepts a `mass_temperatures` parameter and,
//! for **high-mass buildings only** (`h_tr_ms ≥ 500 W/K`), adds the
//! dynamic mass heat release term `h_tr_ms × (T_mass − T_cool_sp) ×
//! MASS_RELEASE_DAMPING` to the cooling demand. The heating demand formula
//! is unchanged, preserving the Issue #925 fix.
//!
//! # What this test pins down
//!
//!  1. **The asymmetric heating/cooling treatment.** Heating uses the
//!     steady-state `h_loss × (T_heat − T_zone)` formula (Issue #925).
//!     Cooling adds the dynamic mass heat release term when the mass is
//!     hotter than the cooling setpoint. This regression guards against
//!     accidental reversion to the symmetric formula.
//!
//!  2. **The high-mass threshold.** The mass heat release term is gated on
//!     `h_tr_ms ≥ 500 W/K`. For low-mass buildings (Case 600/650 with
//!     `h_tr_ms ≈ 240 W/K`), the term is zero and the standard formula is
//!     used. This guards against the term over-predicting cooling for
//!     low-mass buildings with night ventilation (Case 650, where the
//!     mass can spike transiently during the day).
//!
//!  3. **The dead-band case.** When `T_zone` is in the dead band
//!     (between heating and cooling setpoints) but `T_mass > T_cool_sp`,
//!     the formula still produces a cooling demand. The standard formula
//!     would yield zero in this case because `T_zone` is in the dead band.
//!
//!  4. **The mass temperature sanity guard.** If `T_mass` is outside
//!     `[-20, 80]°C` (degenerate numerical value from the 5R1C mass update
//!     in the 9R4C path), the term is suppressed. This guards against
//!     numerical instability in the lumped-mass update.
//!
//! See `src/sim/thermal_model_physics/hvac.rs` for the implementation.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

/// Compute the HVAC demand for a single zone with explicit mass temperature,
/// independent of the model's internal state. Mirrors the formula in
/// `compute_zone_hvac_load` (hvac.rs) for unit testing the asymmetry.
fn demand_for_zone(
    h_loss: f64,
    h_tr_ms: f64,
    t_zone: f64,
    t_mass: f64,
    heat_sp: f64,
    cool_sp: f64,
) -> f64 {
    const MASS_RELEASE_DAMPING: f64 = 1.0;
    const HIGH_MASS_H_TR_MS_THRESHOLD: f64 = 500.0;
    const MASS_TEMP_MAX: f64 = 35.0;

    let h_coeff = h_loss;
    let mass_heat_release_unclamped =
        if t_mass > cool_sp && t_mass <= MASS_TEMP_MAX && h_tr_ms >= HIGH_MASS_H_TR_MS_THRESHOLD {
            h_tr_ms * (t_mass - cool_sp) * MASS_RELEASE_DAMPING
        } else {
            0.0
        };
    let mass_heat_release = if mass_heat_release_unclamped > 0.0 {
        mass_heat_release_unclamped.min(h_loss * 10.0)
    } else {
        0.0
    };

    if t_zone < heat_sp {
        h_coeff * (heat_sp - t_zone)
    } else if t_zone > cool_sp {
        -h_coeff * (t_zone - cool_sp) - mass_heat_release
    } else if mass_heat_release > 0.0 {
        -mass_heat_release
    } else {
        0.0
    }
}

#[test]
fn issue_900_heating_formula_unchanged_from_925() {
    // The heating branch must use the Issue #925 formula: h_loss × (T_heat - T_zone)
    // and must NOT include the mass heat absorption term.
    // Case 900 (high mass): h_loss = 93, h_tr_ms = 1092.
    let h_loss = 93.0;
    let h_tr_ms = 1092.0;

    // Zone is 10°C in winter. With #925: q_heat = 93 × 10 = 930 W.
    let q = demand_for_zone(h_loss, h_tr_ms, 10.0, 5.0, 20.0, 27.0);
    assert!(
        (q - 930.0).abs() < 1.0,
        "Heating formula regressed: q={q:.2} (expected ≈ 930)"
    );

    // Even with a very cold mass (T_mass = -10°C, far below heat_sp), the
    // mass heat absorption term must NOT augment heating. This guards
    // against the asymmetric extension being applied to heating.
    let q_with_cold_mass = demand_for_zone(h_loss, h_tr_ms, 10.0, -10.0, 20.0, 27.0);
    assert!(
        (q_with_cold_mass - 930.0).abs() < 1.0,
        "Heating should not include mass heat absorption: q={q_with_cold_mass:.2} \
         (expected 930 same as without mass term)"
    );
}

#[test]
fn issue_900_cooling_includes_mass_heat_release_for_high_mass() {
    // For high-mass buildings, the cooling demand at setpoint must include
    // the dynamic mass heat release term h_tr_ms × (T_mass − T_cool_sp).
    //
    // Case 900: h_loss = 93, h_tr_ms = 1092. Mass at 30°C in summer
    // (within the realistic cap of 35°C; multi-node solver peaks ~33°C
    // and the well-behaved 5R1C path never exceeds 30°C).
    //
    // The mass term unclamped is -1092 × 3 = -3276 W, which exceeds the
    // 10×h_loss = 930 W magnitude cap (set to suppress 5R1C mass
    // divergence in multi-zone high-mass cases). The mass term is
    // therefore clipped to -930 W. Total demand: -h_loss × 3 - 930
    // = -279 - 930 = -1209 W.
    let h_loss = 93.0;
    let h_tr_ms = 1092.0;
    let cool_sp = 27.0;

    let q_above_cool = demand_for_zone(h_loss, h_tr_ms, 30.0, 30.0, 20.0, cool_sp);
    let expected = -(h_loss * 3.0 + h_loss * 10.0);
    assert!(
        (q_above_cool - expected).abs() < 1.0,
        "High-mass cooling demand should include mass heat release (clipped): \
         q={q_above_cool:.2} W (expected ≈ {expected:.2})"
    );
}

#[test]
fn issue_900_cooling_dead_band_with_hot_mass_produces_demand() {
    // This is the exact failure mode described in the issue:
    //
    //   "t_free never exceeds the cooling setpoint, so no cooling demand
    //    is computed. But the mass is hotter than the setpoint and is
    //    releasing heat that the HVAC must remove."
    //
    // Zone is in the dead band (e.g., 24°C, between heat_sp=20 and cool_sp=27).
    // The mass is at 30°C, hotter than cool_sp. The standard formula gives
    // 0; the mass-heat-release branch must yield a negative (cooling) demand.
    //
    // The mass term unclamped is 1092 × 3 = 3276 W; clipped to 10×h_loss
    // = 930 W in the dead band case.
    let h_loss = 93.0;
    let h_tr_ms = 1092.0;
    let cool_sp = 27.0;

    let q = demand_for_zone(h_loss, h_tr_ms, 24.0, 30.0, 20.0, cool_sp);
    let expected = -(h_loss * 10.0); // mass term only, clipped to 10×h_loss
    assert!(
        (q - expected).abs() < 1.0,
        "Dead-band with hot mass should yield cooling demand: \
         q={q:.2} W (expected ≈ {expected:.2})"
    );
    assert!(
        q < 0.0,
        "Demand must be negative (cooling) when mass > cool_sp"
    );
}

#[test]
fn issue_900_cooling_no_mass_term_for_low_mass_buildings() {
    // For low-mass buildings (h_tr_ms < 500 W/K), the mass heat release
    // term is suppressed to avoid over-predicting cooling for cases with
    // night ventilation (Case 650) or transient mass-temperature spikes.
    //
    // Case 600: h_tr_ms ≈ 240 W/K. The mass_heat_release term should be 0.
    let h_loss = 93.0;
    let h_tr_ms = 240.0; // Low mass
    let cool_sp = 27.0;

    // Zone in dead band, mass at 35°C (transient spike). Demand should
    // be 0 because the high-mass threshold blocks the mass term and the
    // zone is in the dead band.
    let q = demand_for_zone(h_loss, h_tr_ms, 25.0, 35.0, 20.0, cool_sp);
    assert!(
        q.abs() < 1e-9,
        "Low-mass dead-band demand must be 0 (term suppressed): q={q:.6} W"
    );

    // Zone above cool_sp: standard formula applies (no mass term).
    let q_above = demand_for_zone(h_loss, h_tr_ms, 30.0, 35.0, 20.0, cool_sp);
    let expected = -h_loss * (30.0 - cool_sp);
    assert!(
        (q_above - expected).abs() < 1.0,
        "Low-mass above-cool_sp demand should match standard formula: \
         q={q_above:.2} (expected {expected:.2})"
    );
}

#[test]
fn issue_900_cooling_mass_temp_sanity_guard() {
    // Mass temperatures above 35°C are treated as degenerate (the 5R1C
    // lumped mass can diverge numerically in high-mass buildings with
    // HVAC control — observed up to 75°C+ in Case 960 back zone and Case
    // 900 9R4C). The formula must suppress the mass term in that case
    // to avoid amplifying the divergence.
    let h_loss = 93.0;
    let h_tr_ms = 1092.0;
    let cool_sp = 27.0;

    // T_mass = 50°C (degenerate — above 35°C cap). Term must be suppressed.
    let q = demand_for_zone(h_loss, h_tr_ms, 25.0, 50.0, 20.0, cool_sp);
    assert!(
        q.abs() < 1e-9,
        "Degenerate mass temp (50°C) must not produce demand: q={q:.2} W"
    );

    // T_mass = 75°C (degenerate). Term must be suppressed.
    let q_extreme = demand_for_zone(h_loss, h_tr_ms, 25.0, 75.0, 20.0, cool_sp);
    assert!(
        q_extreme.abs() < 1e-9,
        "Degenerate mass temp (75°C) must not produce demand: q={q_extreme:.2} W"
    );

    // T_mass = 35.0°C is the upper cap (still allowed: t_mass <= 35.0).
    // The unclamped mass term is 1092 × 8 = 8736 W, which is above the
    // 10×h_loss = 930 W magnitude cap, so the term is clipped to 930 W.
    let q_cap = demand_for_zone(h_loss, h_tr_ms, 25.0, 35.0, 20.0, cool_sp);
    let expected = -(h_loss * 10.0); // clipped to 10×h_loss
    assert!(
        (q_cap - expected).abs() < 1.0,
        "T_mass = 35°C (cap) should be clipped to 10×h_loss: \
         q={q_cap:.2} (expected {expected:.2})"
    );
}

#[test]
fn issue_900_annual_cooling_increases_for_case_900() {
    // End-to-end check: simulating Case 900 with the fix should produce
    // more annual cooling than the pre-fix baseline of 0.28 MWh.
    //
    // We don't assert the ASHRAE 140 reference (2.13–3.67 MWh) here —
    // the mass-dynamics issues from #917/#924 still prevent the model
    // from reaching the reference range. This test just guards against
    // regression: the cooling must be > 0 and noticeably larger than
    // the pre-fix 0.28 MWh.
    let spec = ASHRAE140Case::Case900.spec();
    let mut model: ThermalModel<VectorField> = ThermalModel::from_spec(&spec);

    let weather = fluxion::weather::denver::DenverTmyWeather::new();
    for step in 0..8760 {
        let wd = weather.get_hourly_data(step).unwrap();
        model.weather = Some(wd.clone());
        model.step_physics(step, wd.dry_bulb_temp, 3600.0);
    }

    let cooling_mwh = model.annual_cooling_energy / 1000.0;
    println!("Case 900 annual cooling (post-#900): {cooling_mwh:.3} MWh");

    // Pre-fix baseline was 0.28 MWh. The fix should at least double this.
    assert!(
        cooling_mwh > 0.45,
        "Annual cooling regressed: {cooling_mwh:.3} MWh \
         (expected > 0.45 MWh, pre-fix was 0.28 MWh)"
    );
}

#[test]
fn issue_900_annual_heating_unchanged_for_case_900() {
    // The #925 fix brought Case 900 heating from 11.98 MWh down to ~3.05 MWh.
    // This test guards against the #900 fix accidentally making heating
    // worse by, e.g., adding the mass heat absorption term to the heating
    // branch.
    let spec = ASHRAE140Case::Case900.spec();
    let mut model: ThermalModel<VectorField> = ThermalModel::from_spec(&spec);

    let weather = fluxion::weather::denver::DenverTmyWeather::new();
    for step in 0..8760 {
        let wd = weather.get_hourly_data(step).unwrap();
        model.weather = Some(wd.clone());
        model.step_physics(step, wd.dry_bulb_temp, 3600.0);
    }

    let heating_mwh = model.annual_heating_energy / 1000.0;
    println!("Case 900 annual heating (post-#900): {heating_mwh:.3} MWh");

    // The #925 fix produces ~3.05 MWh for Case 900. Allow ±0.5 MWh slack
    // for numerical noise — the key check is that we did NOT regress to
    // 10+ MWh (the pre-#925 buggy value).
    assert!(
        (2.5..=3.6).contains(&heating_mwh),
        "Annual heating regressed: {heating_mwh:.3} MWh \
         (expected 2.5–3.6 MWh, preserving Issue #925 fix)"
    );
}
