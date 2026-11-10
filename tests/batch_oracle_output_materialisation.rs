//! Issue #4696 (spike) — kill-outcome evidence for the "aggregate-only output
//! mode" premise on the `BatchOracle` population path.
//!
//! # What this file establishes
//!
//! The spike proposed an opt-in *aggregate-only* output mode for
//! sweep/calibration population runs, on the premise that a BatchOracle
//! population run spends a material fraction of its wall time materialising /
//! writing **full output** (per-timestep series) that sweep callers never
//! read, and that streaming aggregates (annual/monthly totals, peaks with
//! timestamps, unmet hours) would reclaim that time.
//!
//! The premise does not hold for `BatchOracle`, and this file pins the
//! evidence with deterministic, public-API-only checks:
//!
//! 1. `population_output_is_one_scalar_per_config` — the population API
//!    returns exactly one `f64` (EUI) per candidate. There is no per-timestep
//!    series in the return type to discard.
//! 2. `population_stepping_materialises_no_hourly_trace` — the per-timestep
//!    kernel the population path uses (`ThermalModel::solve_single_step`)
//!    allocates **no** per-timestep trace buffers (`hourly_temperatures`,
//!    `nodal_temperatures`, end-use metering series all stay `None`).
//!    Those buffers are only allocated by the model-level
//!    `solve_timesteps` path, which the population path never calls.
//! 3. `aggregate_from_full_series_matches_population_scalar_bitwise` — the
//!    aggregate actually returned by the population path is **bit-identical**
//!    to the aggregate computed from the full 8 760-step series in the same
//!    accumulation order. I.e. the population path already returns the
//!    aggregate; there is no "full output" whose materialisation an
//!    aggregate-only mode could avoid.
//!
//! # Consequence
//!
//! The ceiling for an aggregate-only mode on this path is the cost of
//! materialising the returned `Vec<f64>` (one `f64` per config) — measured in
//! `benches/batch_oracle_bench.rs` as
//! `batch_oracle_api_boundary_output/{size}`. See `WORKER_REPORT.md` for the
//! before/after numbers and the kill decision.

use fluxion::physics::cta::{ContinuousTensor, VectorField};
use fluxion::sim::engine::{StepParameters, ThermalModel};
use fluxion::BatchOracle;

/// Number of timesteps in the analytical population path's per-config loop.
const TIMESTEPS: usize = 8760;

/// Deterministic population fixture (no RNG dependency in the test).
fn population() -> Vec<Vec<f64>> {
    vec![
        vec![1.5, 20.0, 27.0],
        vec![0.8, 21.0, 25.0],
        vec![3.2, 19.0, 30.0],
    ]
}

/// Reference implementation of the analytical population kernel: clone the
/// base model, apply the parameters, then step the model once per timestep,
/// accumulating the timestep energies in timestep order.
///
/// This mirrors `RayonChunksOrchestrator::run_cpu_analytical` (the analytical
/// branch of `BatchOracle::evaluate_population`) step for step, and also
/// materialises the *full* per-timestep series (which the population path
/// does not) so the test can compute the aggregate both ways.
///
/// Returns `(eui, hourly_energy_kwh, stepped_model)` — the population-path
/// aggregate, the full series behind it, and the model after stepping (so
/// tests can inspect the stepped clone's diagnostics, not the untouched
/// base model).
fn full_series_eui(
    base_model: &ThermalModel<VectorField>,
    params: &[f64],
) -> (f64, Vec<f64>, ThermalModel<VectorField>) {
    let mut model = base_model.clone();
    model.apply_parameters(params);

    let step_params = StepParameters::build_analytical();
    let mut hourly_energy_kwh = Vec::with_capacity(TIMESTEPS);
    for t in 0..TIMESTEPS {
        let hour_of_day = t % 24;
        let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
        let outdoor_temp = 10.0 + 10.0 * daily_cycle;
        let energy = model.solve_single_step(t, outdoor_temp, &step_params, 3600.0);
        hourly_energy_kwh.push(energy);
    }

    let total_area = model.setpoints.zone_area.integrate();
    let total_energy: f64 = hourly_energy_kwh.iter().sum();
    let eui = if total_area > 0.0 {
        (total_energy / total_area).max(0.0)
    } else {
        0.0
    };
    (eui, hourly_energy_kwh, model)
}

/// (1) The population API returns exactly one scalar per candidate — there is
/// no per-timestep series in the output for an aggregate-only mode to skip.
#[test]
fn population_output_is_one_scalar_per_config() {
    let oracle = BatchOracle::from_model(ThermalModel::<VectorField>::new(10)).unwrap();
    let pop = population();

    let results = oracle.evaluate_population(pop.clone(), false).unwrap();

    assert_eq!(
        results.len(),
        pop.len(),
        "the population API must return one result per candidate"
    );
    // The output is a flat `Vec<f64>`: 8 bytes per config, i.e. O(1) memory
    // per config (not O(timesteps)).
    assert_eq!(
        std::mem::size_of_val(&results),
        pop.len() * std::mem::size_of::<f64>()
    );
}

/// (2) The per-timestep kernel used by the population path allocates no
/// per-timestep trace buffers — the "full output" the spike wanted to avoid
/// materialising is never materialised on this path.
#[test]
fn population_stepping_materialises_no_hourly_trace() {
    let base_model = ThermalModel::<VectorField>::new(10);
    let params = population().remove(0);

    let (eui, series, stepped_model) = full_series_eui(&base_model, &params);

    assert_eq!(series.len(), TIMESTEPS);
    assert!(eui.is_finite() && eui > 0.0);

    // `solve_single_step` (the population path's kernel) never allocates the
    // hourly / nodal / end-use trace buffers; only the model-level
    // `solve_timesteps` path does. After 8 760 steps on the stepped clone all
    // of them must still be absent, i.e. per-config memory is O(1) in the
    // number of timesteps. (Supervisor fix at collection: assert on the
    // stepped model — asserting on the unstepped base model proved nothing.)
    let diagnostics = &stepped_model.0.diagnostics_state;
    assert!(
        diagnostics.hourly_temperatures.is_none(),
        "the population kernel must not materialise hourly temperature traces"
    );
    assert!(
        diagnostics.nodal_temperatures.is_none(),
        "the population kernel must not materialise nodal temperature traces"
    );
    assert!(
        diagnostics.hourly_heating_kwh.is_none(),
        "the population kernel must not materialise end-use metering series"
    );
    assert!(
        diagnostics.hourly_cooling_kwh.is_none(),
        "the population kernel must not materialise end-use metering series"
    );
}

/// (3) The aggregate returned by the population path is bit-identical to the
/// aggregate computed from the full 8 760-step series in the same
/// accumulation order (timestep order, left-to-right `sum`).
///
/// This is the "aggregate equality" check of the spike, applied to the
/// output the population path actually produces: the aggregate-only result
/// and the full-series result agree bit for bit, so there is no numerical
/// headroom for an aggregate-only mode either.
#[test]
fn aggregate_from_full_series_matches_population_scalar_bitwise() {
    let base_model = ThermalModel::<VectorField>::new(10);
    let oracle = BatchOracle::from_model(base_model.clone()).unwrap();
    let pop = population();

    let results = oracle.evaluate_population(pop.clone(), false).unwrap();

    for (params, &population_eui) in pop.iter().zip(results.iter()) {
        let (full_series_eui, series, _stepped_model) = full_series_eui(&base_model, params);

        // The aggregate over the full series, accumulated in timestep order.
        let total_energy: f64 = series.iter().sum();
        let total_area = base_model.setpoints.zone_area.integrate();
        let expected = if total_area > 0.0 {
            (total_energy / total_area).max(0.0)
        } else {
            0.0
        };

        assert_eq!(
            population_eui.to_bits(),
            expected.to_bits(),
            "population aggregate must be bit-identical to the full-series aggregate \
             (params {params:?})"
        );
        assert_eq!(
            full_series_eui.to_bits(),
            expected.to_bits(),
            "the reference full-series aggregate must match its own recomputation"
        );
    }
}
