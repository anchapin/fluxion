//! Regression harness for issue #2674 — pre-existing default-schema
//! simulation divergence at timestep 91.
//!
//! `run_simulation` over the default `SimulationSchemaV1` returns
//! `SimulationFailed("simulation diverged at timestep 91 in zone zone_0")`
//! because `run_simulation` (src/api/server.rs) constructs the engine model
//! via `ThermalModel::new(num_zones)` and sets only the heating/cooling
//! setpoints, leaving the placeholder `thermal_capacitance = 1.0 J/K` and
//! `air_thermal_capacitance = 0.0` in place. `select_integration_method`
//! (`src/sim/thermal_integration.rs`) picks Explicit-Euler for `C_m <= 500`,
//! and the Explicit-Euler mass update `Tm_new = Tm_old + (q_net / C_m) · dt`
//! with `C_m = 1.0` and `dt = 3600 s` amplifies any flux imbalance by ~3600
//! per step — an exponential blow-up that reaches `inf`/`NaN` at hourly
//! index 91 (`last_known_good_timestep = 90`).
//!
//! This file deliberately **asserts the divergence** so the gap stays
//! tracked in CI. The five `/v1/simulate`-driven tests in
//! `tests/api_integration_tests.rs` and `concurrent_throughput_smoke` in
//! `tests/api_concurrent_throughput.rs` are `#[ignore]`'d against this
//! same root cause (see docs/KNOWN_ISSUES.md LIMIT-07). When a real fix
//! lands — wiring the schema geometry/construction into the model's
//! thermal capacitances and conductances (an `from_spec`-equivalent for
//! `SimulationSchemaV1`) — this test will fail because the divergence
//! will no longer occur; that is the signal to remove the `#[ignore]`s
//! and flip the assertions below.
//!
//! Run with:
//!   cargo test --profile ci --test issue_2674_repro -- --nocapture

use fluxion::api::schema::{
    ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaMetadata, SchemaVersion,
    SimulationOutput, SimulationSchemaV1, WeatherData,
};
use fluxion::api::server::{run_simulation, ApiError};

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

/// Pin the exact divergence signature produced by the default schema today.
///
/// If this test starts FAILING it means the timestep-91 divergence has been
/// fixed — remove the `#[ignore]`'d tests in `api_integration_tests.rs` and
/// `api_concurrent_throughput.rs` and update docs/KNOWN_ISSUES.md LIMIT-07.
#[test]
fn pins_default_schema_diverges_at_timestep_91() {
    let schema = default_schema_v1();
    let err = run_simulation(&schema, 1, false, "issue_2674").expect_err(
        "default-schema simulation must still diverge at timestep 91; \
         if it now succeeds, the bug is fixed — see module docs",
    );

    let ApiError::SimulationFailed(msg, diag) = err else {
        panic!("expected SimulationFailed, got {err:?}");
    };
    assert!(
        msg.contains("diverged at timestep 91"),
        "unexpected divergence message: {msg}"
    );
    assert!(
        msg.contains("zone_0"),
        "divergence must be attributed to zone_0: {msg}"
    );

    let diag = diag.expect("divergence must carry SimulationDiagnostics");
    assert_eq!(diag.failing_timestep, 91, "failing_timestep drift");
    assert_eq!(diag.failing_zone.as_deref(), Some("zone_0"));
    assert_eq!(diag.last_known_good_timestep, 90, "last_known_good drift");
}

/// Capture the diverging per-zone trace so the magnitude/shape of the
/// explosion is on record (not just the timestep). This is a no-op
/// assertion — it only exists so the `println!` output is captured in
/// `--nocapture` CI logs for future root-cause work.
#[test]
fn captures_diverging_temperature_trace() {
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;

    let schema = default_schema_v1();
    let num_zones = schema.geometry.zones.len().max(1);
    let heating = schema.controls.zone_control.heating_setpoint;
    let cooling = schema.controls.zone_control.cooling_setpoint;

    // Mirror run_simulation's (un-initialized) model setup exactly.
    let mut model = ThermalModel::<VectorField>::new(num_zones);
    for z in 0..model.num_zones {
        model.heating_setpoints.as_mut_slice()[z] = heating;
        model.cooling_setpoints.as_mut_slice()[z] = cooling;
    }
    let surrogates = fluxion::ai::surrogate::SurrogateManager::new().unwrap();
    let _ = model.solve_timesteps(8760, &surrogates, false, None, None, None);

    let zone0 = model
        .get_hourly_temperatures()
        .and_then(|t| t.into_iter().next())
        .unwrap_or_default();
    let first_bad = zone0.iter().position(|&v| !v.is_finite());
    let max_abs_finite = zone0
        .iter()
        .take_while(|&&v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);

    println!(
        "issue #2674 trace: len={} first_non_finite={first_bad:?} \
         max_abs_over_finite_prefix={max_abs_finite:.3e} \
         sample[t=0..6]={:?}",
        zone0.len(),
        &zone0[..zone0.len().min(6)],
    );

    // The diagnostic-side test pins the exact timestep; here we only
    // assert the qualitative explosion shape so this stays robust to
    // minor upstream changes while still failing if the divergence
    // disappears (which is the event we want to detect).
    assert_eq!(first_bad, Some(91), "divergence first-finite-break moved");
    assert!(
        max_abs_finite > 1e10,
        "expected explicit-Euler blow-up magnitude before step 91, got {max_abs_finite:.3e}"
    );
}
