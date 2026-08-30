//! Regression guard for Issue #2455 — Case 900FF night minimum regression.
//!
//! Background
//! ----------
//! `docs/investigations/ISSUE_1168_ROOT_CAUSE.md` documents the 9R4C high-mass
//! free-floating night minimum as "~0.6 °C warm" vs the ASHRAE 140 band
//! [-6.40, -1.60] °C, a small acceptable residual attributed to the air node
//! lacking a direct longwave-to-sky radiative path and the ground-coupled
//! floor node retaining heat. In 2026 the night minimum regressed to
//! -12.70 °C — 6.30 °C below the lower band edge — opposite the original
//! residual.
//!
//! Root cause
//! ----------
//! Per ISO 13790 Annex C, the half-insulation rule excludes layers exterior
//! to the dominant insulation from the effective thermal capacitance κ.
//! For the ASHRAE 140 Case 900 wall (`wood_siding + foam + concrete_block`,
//! interior → exterior) the half-insulation rule reduces the wall capacitance
//! to just the wood siding + foam layers (~12 kJ/m²K per area), dropping the
//! per-surface time constant to τ ≈ C_wall / (h_tr_em + h_tr_ms) ≈ 1–2 h.
//! The wall mass then tracks the outdoor dry-bulb within a few hours, and
//! during Denver TMY3 multi-day cold snaps the free-floating air node falls
//! below the band.
//!
//! Fix
//! ---
//! For HighMass construction (`ConstructionType::HighMass`), use the FULL
//! per-layer thermal capacitance (`thermal_capacitance_per_area()`) for the
//! wall layer instead of the half-insulation rule (`iso_13790_effective_
//! capacitance_per_area()`). This matches ISO 13790 §12.2.3 + Annex C, which
//! require the full envelope capacitance for "heavy" / "very heavy" classes.
//! The roof and floor constructions already keep their heavy mass under the
//! half-insulation rule (the heavy concrete / timber layers are interior to
//! the foam insulation per ASHRAE 140 Case 900), so the fix is wall-only.
//!
//! Test path
//! ---------
//! The validator uses `EpwWeatherSource::from_file(USA_CO_Denver-Stapleton...)`
//! rather than the synthetic `DenverTmyWeather` used by most other tests, so
//! this test loads the same EPW file and exercises the production
//! `step_physics` path with `model.setpoints.heating_setpoint = -999.0` /
//! `cooling_setpoint = 999.0` (free-floating mode). The captured night
//! minimum is logged; an assertion confirms it is within 1.6 °C of the lower
//! reference band edge (-8.0 °C = -6.40 - 1.6) — the issue body's proposed
//! `T_min > -8.0 °C` guard.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::sim::warmup::{run_warmup, WarmupConfig};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::epw::EpwWeatherSource;
use fluxion::weather::WeatherSource;

const EPW_PATH: &str = "assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw";

const REF_LOWER_EDGE: f64 = -6.40;
const REF_UPPER_EDGE: f64 = -1.60;
const REGRESSION_TOLERANCE: f64 = 1.6;

#[test]
fn test_case_900ff_night_minimum_within_reference_band() {
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    // Free-floating mode — matches the ASHRAE 140 validator path
    model.setpoints.heating_setpoint = -999.0;
    model.setpoints.cooling_setpoint = 999.0;
    model.hvac.hvac_heating_capacity = 0.0;
    model.hvac.hvac_cooling_capacity = 0.0;

    let weather = EpwWeatherSource::from_file(EPW_PATH)
        .expect("Failed to load Denver TMY EPW file required by this test");

    // Match the canonical ASHRAE 140 validator path: 14-day fixed warm-up
    // (ASHRAE 140 §B2 periodic-steady-state preconditioning).
    run_warmup(&mut model, &weather, &WarmupConfig::default());

    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;
    let mut step_of_min: usize = 0;
    let mut min_outdoor_at_step: f64 = f64::INFINITY;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.solar.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if let Some(&zone_temp) = model.setpoints.temperatures.as_slice().first() {
            if zone_temp < min_temp {
                min_temp = zone_temp;
                step_of_min = step;
                min_outdoor_at_step = weather_data.dry_bulb_temp;
            }
            if zone_temp > max_temp {
                max_temp = zone_temp;
            }
        }
    }

    println!("\n=== Case 900FF Bisect (Issue #2455) ===");
    println!("Thermal model  : {:?}", model.hvac.thermal_model_type);
    println!("Night minimum  : {min_temp:.2} °C (at step {step_of_min})");
    println!("Outdoor at min : {min_outdoor_at_step:.2} °C");
    println!("Max temperature: {max_temp:.2} °C");
    println!("Reference band : [{REF_LOWER_EDGE:.2}, {REF_UPPER_EDGE:.2}] °C");

    let coldest = min_temp;
    let below_lower = REF_LOWER_EDGE - coldest;
    let tolerance_gap = below_lower - REGRESSION_TOLERANCE;
    if tolerance_gap > 0.0 {
        eprintln!(
            "REGRESSION: night minimum is {tolerance_gap:.2} °C below the \
             tolerance (lower band edge {REF_LOWER_EDGE:.2} °C - \
             REGRESSION_TOLERANCE {REGRESSION_TOLERANCE:.1} °C)"
        );
    } else {
        println!("OK: night minimum is within {REGRESSION_TOLERANCE:.1} °C of the lower band edge");
    }

    assert!(
        coldest > REF_LOWER_EDGE - REGRESSION_TOLERANCE,
        "Case 900FF night minimum {coldest:.2} °C is more than \
         {REGRESSION_TOLERANCE:.1} °C below the lower reference band edge \
         ({REF_LOWER_EDGE:.2} °C); reference band is [{REF_LOWER_EDGE:.2}, \
         {REF_UPPER_EDGE:.2}] °C. This is the Issue #2455 regression \
         signature — see docs/investigations/ISSUE_1168_ROOT_CAUSE.md \
         for the underlying 9R4C air-mass coupling analysis.",
    );
}
