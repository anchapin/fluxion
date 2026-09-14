//! Regression harness for issue #2674 / #2747 — pre-existing default-schema
//! simulation divergence at timestep 91, now FIXED.
//!
//! Historical context (the bug, prior to the #2747 fix):
//!
//! `run_simulation` over the default `SimulationSchemaV1` used to return
//! `SimulationFailed("simulation diverged at timestep 91 in zone zone_0")`
//! because `run_simulation` (`src/api/server.rs`) constructed the engine
//! model via `ThermalModel::new(num_zones)` and set only the heating/cooling
//! setpoints, leaving the placeholder `thermal_capacitance = 1.0 J/K` and
//! `air_thermal_capacitance = 0.0` in place. `select_integration_method`
//! (`src/sim/thermal_integration.rs`) picked Explicit-Euler for `C_m <= 500`,
//! and the Explicit-Euler mass update `Tm_new = Tm_old + (q_net / C_m) · dt`
//! with `C_m = 1.0` and `dt = 3600 s` amplified any flux imbalance by ~3600
//! per step — an exponential blow-up that reached `inf`/`NaN` at hourly
//! index 91 (`last_known_good_timestep = 90`).
//!
//! The fix (#2747): `run_simulation` and `/v1/simulate/stream` now build the
//! model via `build_model_from_schema(schema)`, which mirrors
//! `ThermalModel::from_spec` for the simpler `SimulationSchemaV1` shape:
//! per-zone geometry, construction-layer U-values, ISO 13790 §7.2 thermal
//! capacitance C_m = wall_cap + roof_cap + floor_cap, ISO 13790 Eq. 64
//! envelope conductances (h_tr_em, h_tr_ms, h_tr_me), and HVAC setpoints /
//! schedules. The 6 `#[ignore]`'d API tests in `tests/api_integration_tests.rs`
//! and `tests/api_concurrent_throughput.rs` are un-ignored, and this file
//! asserts the simulation is now stable through step 91+.
//!
//! Run with:
//!   cargo test --profile ci --test issue_2674_repro -- --nocapture

use fluxion::api::schema::{
    ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaMetadata, SchemaVersion,
    SimulationOutput, SimulationSchemaV1, WeatherData,
};
use fluxion::api::server::run_simulation;
use fluxion::sim::thermal_selector::ThermalSelector;

fn default_schema_v1() -> SimulationSchemaV1 {
    SimulationSchemaV1 {
        version: SchemaVersion::V1,
        metadata: SchemaMetadata::default(),
        geometry: Geometry::default(),
        constructions: ConstructionSet::default(),
        schedules: ScheduleSet::default(),
        weather: WeatherData::default(),
        controls: ControlSet::default(),
        output: SimulationOutput::default(),
    }
}

/// Pin that the default-schema simulation now SUCCEEDS (no divergence).
///
/// Prior to #2747 this test asserted the timestep-91 divergence. With the
/// schema→physics wiring in `build_model_from_schema` (`src/api/server.rs`)
/// the simulation runs to completion with finite, physically-sane output.
/// If this test starts FAILING (the divergence returns), the schema→physics
/// wiring has regressed — see `build_model_from_schema` doc-comment for the
/// full field list that must be populated.
#[test]
fn pins_default_schema_succeeds_through_step_91() {
    let schema = default_schema_v1();
    let output = run_simulation(
        &schema,
        1,
        false,
        ThermalSelector::default(),
        "issue_2674_fix",
    )
    .expect(
        "default-schema simulation must succeed after the #2747 fix; \
         if it now diverges again, the schema→physics wiring in \
         `build_model_from_schema` has regressed",
    );

    // Sanity checks on the output — physically-sane ranges for a 48 m²
    // heating-dominated zone with the default envelope construction.
    // These bounds are deliberately loose: this test pins the *divergence-
    // is-fixed* invariant, not EnergyPlus-comparable accuracy. The latter
    // is the responsibility of the ASHRAE 140 validation suite.
    assert!(
        output.eui.is_finite() && output.eui > 0.0,
        "EUI must be finite and positive, got {}",
        output.eui
    );
    assert!(
        output.heating_energy > 0.0,
        "expected non-zero heating energy for a heating-dominated default zone, got {}",
        output.heating_energy
    );
    assert!(
        output.peak_heating_load > 0.0,
        "expected non-zero peak heating load, got {}",
        output.peak_heating_load
    );

    // Hourly temperature trace must be finite across ALL 8760 timesteps.
    // This is the direct inversion of the original divergence assertion
    // (which saw `inf`/`NaN` from timestep 91 onward).
    let hourly = output
        .hourly_zone_temperatures
        .as_ref()
        .and_then(|z| z.first())
        .expect("hourly_zone_temperatures[0] must be populated");
    assert_eq!(
        hourly.len(),
        8760,
        "expected a full 8760-step hourly trace, got {}",
        hourly.len()
    );
    let first_non_finite = hourly.iter().position(|&v| !v.is_finite());
    assert!(
        first_non_finite.is_none(),
        "divergence recurred: first non-finite temperature at index {:?} \
         (was None before #2674, must stay None after #2747)",
        first_non_finite
    );

    // Physical-sanity band: zone temperatures must stay within a sane
    // envelope. The default schema has heating=20°C / cooling=24°C with
    // 100 kW HVAC capacity, so the zone should never wander more than a
    // few degrees outside [15, 30]°C even with the synthetic sinusoidal
    // outdoor-air driver used by `solve_timesteps` when no inline weather
    // is supplied. ±100°C is a deliberately loose upper bound that still
    // catches the ±1e5 °C garbage the partial-fix prototype produced.
    let (min_t, max_t) = hourly
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    assert!(
        min_t > -100.0 && max_t < 100.0,
        "zone temperature out of physical-sanity band: [{min_t:.2}, {max_t:.2}] °C"
    );
}

/// Capture the now-stable per-zone trace so the magnitude/shape of the
/// fixed simulation is on record (not just "it didn't diverge"). This is
/// a no-op assertion test — it only exists so the `println!` output is
/// captured in `--nocapture` CI logs for future regression analysis.
#[test]
fn captures_stable_temperature_trace() {
    let schema = default_schema_v1();
    let output = run_simulation(
        &schema,
        1,
        false,
        ThermalSelector::default(),
        "issue_2674_trace",
    )
    .expect("in-process sim must succeed");

    let zone0: Vec<f64> = output
        .hourly_zone_temperatures
        .as_ref()
        .and_then(|t| t.first().cloned())
        .unwrap_or_default();

    let (min_t, max_t) = zone0
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    let mean_t: f64 = zone0.iter().sum::<f64>() / zone0.len().max(1) as f64;

    println!(
        "issue #2747 stable trace: len={} min={min_t:.2}°C max={max_t:.2}°C \
         mean={mean_t:.2}°C sample[t=0..6]={:?}",
        zone0.len(),
        &zone0[..zone0.len().min(6)],
    );

    // Qualitative shape — the trace must be finite across the whole year
    // and stay in a physically-sane diurnal band. The old Explicit-Euler
    // blow-up hit ±1e300 before reaching inf/NaN at step 91; the fixed
    // run oscillates inside the HVAC deadband with small excursions.
    assert_eq!(zone0.len(), 8760, "expected full-year trace");
    assert!(zone0.iter().all(|&v| v.is_finite()));
    assert!(
        min_t > -100.0 && max_t < 100.0,
        "physical-sanity band violated: [{min_t}, {max_t}]"
    );
    // Heating-dominated default zone → mean temperature tracks toward the
    // heating setpoint (20°C) within a few degrees.
    assert!(
        (15.0..=25.0).contains(&mean_t),
        "mean zone temperature {mean_t:.2}°C outside expected [15, 25]°C band"
    );
}
