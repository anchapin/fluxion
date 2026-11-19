//! COP site/electric metering contracts — loop round 160 (scoping option R).
//!
//! Premise, verified 2026-11-19 at develop tip `577058a3`
//! (docs/investigations/2026-11-18-pyo3-030-and-cop-scoping.md §Decision
//! points 3-4): ISO 13790 §10.4.5 (FDIS text, ETH host) computes the zone
//! energy need independently of generation efficiency — system energy use
//! lives downstream of the zone. Option R meters delivered thermal energy
//! into site electric input energy at the metering sites:
//!
//!     input = delivered / COP
//!
//! accumulated alongside the thermal counters, zone-inert by construction.
//! These contracts pin the consumption exact (input = delivered / COP, with
//! power-of-two COPs giving bit-exact round-trips) and zone-inert (thermal
//! counters byte-identical under any COP value, valid or not).
//!
//! Metering-site enumeration (r160, the r112 lesson): production
//! accumulators live at
//!   - `thermal_model_physics/zone_balance.rs` (5R1C, sensible + latent),
//!   - `thermal_model_physics/step_dispatcher.rs` (gauge single-zone and
//!     multi-zone arms; compiled only under the `gauge-solver` feature),
//!   - `thermal_model_physics/physics_impl/step_9r4c.rs` (9R4C sensible,
//!     then the latent join).
//! The deprecated `DeprecatedMultiNodeHvacRunner` accumulators are the
//! ADR-002 legacy loop and are deliberately NOT metered (no default route
//! reaches them; recorded in the round doc).

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;
use fluxion::weather::denver::DenverTmyWeather;

/// Step `model` for `hours` hourly timesteps on the Denver TMY series,
/// starting at `start_hour`, mirroring the case-runner harness
/// (`tests/all_tests/case_900_quick_check.rs`).
fn step_case_model_from(model: &mut ThermalModel<VectorField>, start_hour: usize, hours: usize) {
    let weather = DenverTmyWeather::new();
    for step in start_hour..start_hour + hours {
        if let Ok(weather_data) = weather.get_hourly_data(step) {
            model.solar.weather = Some(weather_data.clone());
            let _ = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        }
    }
}

fn step_case_model(model: &mut ThermalModel<VectorField>, hours: usize) {
    step_case_model_from(model, 0, hours);
}

fn case600_model() -> ThermalModel<VectorField> {
    let spec = ASHRAE140Case::Case600.spec();
    ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector must initialize")
}

/// The plant COP defaults are the repo's own `HVACSystem` literals
/// (src/validation/ashrae140/mod.rs:3.0 / :3.2), transcribed from the
/// engine, not from textbook values.
#[test]
fn plant_cop_defaults_are_engine_literals() {
    let model = ThermalModel::<VectorField>::new(1);
    assert_eq!(model.hvac.cop_heating, 3.0);
    assert_eq!(model.hvac.cop_cooling, 3.2);
}

/// Input = delivered / COP exactly. With power-of-two COPs the per-step
/// division is an exact exponent shift, so the accumulated input equals
/// the accumulated thermal divided by the same constant bit-for-bit —
/// including the 9R4C latent join, whose delivered total is accumulated
/// in two streams (sensible, then latent).
#[test]
fn input_equals_delivered_divided_by_cop_case600() {
    let mut model = case600_model();
    model.hvac.cop_heating = 4.0;
    model.hvac.cop_cooling = 8.0;
    step_case_model(&mut model, 24 * 14);

    let thermal_h = model.hvac.annual_heating_energy;
    let thermal_c = model.hvac.annual_cooling_energy;
    assert!(thermal_h > 0.0, "Case600 January days must demand heating");
    assert_eq!(
        model.get_heating_input_energy_kwh(),
        model.hvac.annual_heating_input_energy,
        "accessor reads the metered field"
    );
    // Exact: each is a sum of terms divided by a power of two.
    assert_eq!(
        model.hvac.annual_heating_input_energy * 4.0,
        thermal_h,
        "heating input must be delivered/COP bit-exact"
    );
    assert_eq!(
        model.hvac.annual_cooling_input_energy * 8.0,
        thermal_c,
        "cooling input must be delivered/COP bit-exact"
    );
}

/// Default plant COPs (3.0 / 3.2 engine literals) meter the same delivered
/// energy; the round-trip closes to double-precision rounding only.
#[test]
fn default_cops_meter_delivered_energy_case600() {
    let mut model = case600_model();
    step_case_model(&mut model, 24 * 14);

    let thermal_h = model.hvac.annual_heating_energy;
    let thermal_c = model.hvac.annual_cooling_energy;
    assert!(thermal_h > 0.0);
    let h = model.hvac.annual_heating_input_energy;
    let c = model.hvac.annual_cooling_input_energy;
    assert!(
        (h * 3.0 - thermal_h).abs() <= thermal_h.abs() * 1e-12,
        "heating {h} kWh_e vs delivered {thermal_h} kWh at COP 3.0"
    );
    assert!(
        (c * 3.2 - thermal_c).abs() <= thermal_c.abs() * 1e-12,
        "cooling {c} kWh_e vs delivered {thermal_c} kWh at COP 3.2"
    );
}

/// Zone-inert: the COP values must not move a single thermal figure. Two
/// identical Case600 models, one carrying COP 2.0/2.0, step identically;
/// every thermal counter matches bit-for-bit while the electric inputs
/// differ. This is the battery byte-identity premise in miniature.
#[test]
fn metering_is_zone_inert_across_cops() {
    let mut a = case600_model();
    let mut b = case600_model();
    b.hvac.cop_heating = 2.0;
    b.hvac.cop_cooling = 2.0;
    step_case_model(&mut a, 24 * 14);
    step_case_model(&mut b, 24 * 14);

    assert_eq!(
        a.hvac.annual_heating_energy, b.hvac.annual_heating_energy,
        "delivered heating must be COP-independent bit-for-bit"
    );
    assert_eq!(
        a.hvac.annual_cooling_energy, b.hvac.annual_cooling_energy,
        "delivered cooling must be COP-independent bit-for-bit"
    );
    assert!(
        (a.hvac.annual_heating_input_energy - b.hvac.annual_heating_input_energy).abs()
            > a.hvac.annual_heating_input_energy * 1e-6,
        "electric inputs must differ across COPs"
    );
}

/// A non-finite or non-positive COP consumes nothing: the electric counters
/// stay at exactly 0.0 (never NaN), the thermal counters are untouched, and
/// the zone advance cannot tell the difference (twin comparison).
#[test]
fn invalid_cop_consumes_nothing_and_stays_finite() {
    for cop in [0.0, -3.0, f64::NAN, f64::INFINITY] {
        let mut model = case600_model();
        model.hvac.cop_heating = cop;
        model.hvac.cop_cooling = cop;
        step_case_model(&mut model, 24 * 14);

        assert_eq!(
            model.hvac.annual_heating_input_energy, 0.0,
            "cop {cop} must consume nothing"
        );
        assert_eq!(model.hvac.annual_cooling_input_energy, 0.0);

        let mut twin = case600_model();
        step_case_model(&mut twin, 24 * 14);
        assert_eq!(
            model.hvac.annual_heating_energy, twin.hvac.annual_heating_energy,
            "an invalid COP must not perturb delivered heating"
        );
        assert_eq!(
            model.hvac.annual_cooling_energy, twin.hvac.annual_cooling_energy,
            "an invalid COP must not perturb delivered cooling"
        );
    }
}

/// Free-float zones demand nothing, so they meter nothing: both electric
/// counters stay at exactly 0.0. Capacity zeroing mirrors the warm-up /
/// free-float validator path (`src/sim/warmup.rs`).
#[test]
fn free_float_meters_no_input() {
    let mut model = case600_model();
    model.hvac.free_float = true;
    model.hvac.hvac_heating_capacity = 0.0;
    model.hvac.hvac_cooling_capacity = 0.0;
    step_case_model(&mut model, 24 * 14);

    assert_eq!(model.hvac.annual_heating_energy, 0.0);
    assert_eq!(model.hvac.annual_cooling_energy, 0.0);
    assert_eq!(model.hvac.annual_heating_input_energy, 0.0);
    assert_eq!(model.hvac.annual_cooling_input_energy, 0.0);
}

/// Resetting the energy tracking clears the electric input counters but is
/// set-once plant metadata preservation for the COPs themselves: a configured
/// plant COP survives a reset (it is capacity-class metadata, not an
/// accumulator).
#[test]
fn reset_clears_inputs_and_keeps_cops() {
    let mut model = case600_model();
    model.hvac.cop_heating = 5.0;
    model.hvac.cop_cooling = 6.0;
    step_case_model(&mut model, 24 * 2);
    assert!(model.hvac.annual_heating_energy > 0.0);
    assert!(model.hvac.annual_heating_input_energy > 0.0);

    model.reset_heating_cooling_energy();
    assert_eq!(model.hvac.annual_heating_energy, 0.0);
    assert_eq!(model.hvac.annual_cooling_energy, 0.0);
    assert_eq!(model.hvac.annual_heating_input_energy, 0.0);
    assert_eq!(model.hvac.annual_cooling_input_energy, 0.0);
    assert_eq!(model.hvac.cop_heating, 5.0, "plant COP survives reset");
    assert_eq!(model.hvac.cop_cooling, 6.0, "plant COP survives reset");
}

/// The 9R4C production path meters too: high-mass Case900 auto-promotes to
/// the multi-node solver in the default build, and its cooling total
/// (sensible + the moisture-balance latent join) converts at the plant COP.
/// Denver's January window carries the heating assertion; a July window
/// carries cooling (the high-mass case does not cool in January).
#[test]
fn case900_9r4c_path_meters_sensible_plus_latent_cooling() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut winter = ThermalModel::<VectorField>::from_spec_with_selector(
        &spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");
    winter.hvac.cop_heating = 4.0;
    winter.hvac.cop_cooling = 8.0;
    step_case_model(&mut winter, 24 * 14);

    let thermal_h = winter.hvac.annual_heating_energy;
    let thermal_c = winter.hvac.annual_cooling_energy;
    assert!(thermal_h > 0.0, "Case900 January days must demand heating");
    assert_eq!(
        winter.hvac.annual_heating_input_energy * 4.0, thermal_h,
        "9R4C heating input bit-exact vs delivered"
    );
    assert_eq!(thermal_c, 0.0, "Case900 must not cool in January");
    assert_eq!(winter.hvac.annual_cooling_input_energy, 0.0);

    // July window: the cooling carrier (sensible + latent join).
    let spec = ASHRAE140Case::Case900.spec();
    let mut summer = ThermalModel::<VectorField>::from_spec_with_selector(
        &spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");
    summer.hvac.cop_heating = 4.0;
    summer.hvac.cop_cooling = 8.0;
    step_case_model_from(&mut summer, 24 * 183, 24 * 14);

    let thermal_c = summer.hvac.annual_cooling_energy;
    assert!(
        thermal_c > 0.0,
        "Case900 July days must demand cooling, got {thermal_c}"
    );
    assert_eq!(
        summer.hvac.annual_cooling_input_energy * 8.0, thermal_c,
        "9R4C cooling input bit-exact vs the sensible+latent total"
    );
}
