//! Numerical-equivalence regression guard for Issue #2687.
//!
//! Issue #2687 reworked the BatchOracle per-timestep hot loop purely for
//! allocation reduction (VectorField → SmallVec backing, scratch fields →
//! SmallVec, and zero-alloc `get_temperatures_into` / `predict_loads_into` /
//! `set_loads`-via-`from_slice` on the CPU surrogate path). None of those
//! changes may alter the *numerical* output — they are storage/ownership
//! refactors that preserve identical f64 values in identical order.
//!
//! This test snapshots the EUI output of a small, deterministic
//! `evaluate_population` run on **both** code paths exercised by the
//! allocation work:
//!
//! 1. The analytical path (`use_surrogates = false`) — the same path the
//!    `dhat_alloc_budget` gate measures; it drives the 5R1C solver's
//!    SmallVec-backed VectorField arithmetic and the SmallVec scratch
//!    buffers.
//! 2. The CPU surrogate / mock path (`use_surrogates = true`, no ONNX model
//!    loaded) — this is the `run_cpu_surrogate` orchestrator hot loop where
//!    `get_temperatures_into` / `predict_loads_into` / `set_loads` were made
//!    allocation-free.
//!
//! The golden EUI arrays below were captured from a known-good run. Any future
//! change that perturbs the physics output (e.g. an accidental reassociation
//! of floating-point ops, a scratch-buffer aliasing bug, or a SmallVec
//! ordering regression) will flip one of these `assert_eq!`s and fail the
//! gate. When the allocation optimization in #2687 landed, these values were
//! verified bit-identical to the pre-change baseline.
//!
//! Run: `cargo test --profile ci -p fluxion --test batch_oracle_hotloop_equivalence`

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::BatchOracle;

/// Same single-zone analytical model the allocation fixtures use
/// (`tests/dhat_alloc_budget.rs`), so this equivalence guard measures the
/// identical code path as the allocation gate.
fn create_single_zone_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.window_u_value = 1.5;
    model.heating_setpoint = 20.0;
    model.cooling_setpoint = 26.0;
    model.temperatures = VectorField::from_scalar(20.0, 1);
    model.mass_temperatures = VectorField::from_scalar(20.0, 1);
    model
}

/// Deterministic, always-valid population: a window-U sweep with a fixed
/// heating/cooling setpoint pair, mirroring `dhat_alloc_budget.rs`.
fn deterministic_population(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| vec![0.5 + (i as f64) * 0.4, 20.0, 26.0])
        .collect()
}

#[test]
fn analytical_path_eui_is_bit_identical_to_baseline() {
    let oracle = BatchOracle::from_model(create_single_zone_model());
    let population = deterministic_population(3);

    // Analytical physics path (use_surrogates = false). Sequential per-config,
    // deterministic regardless of thread scheduling.
    let results: Vec<f64> = oracle
        .evaluate_population(population, false)
        .expect("analytical evaluate_population must succeed");

    // Sanity: every config is valid (heating < cooling), so all EUIs finite.
    assert!(
        results.iter().all(|r| r.is_finite()),
        "analytical EUIs must all be finite: {results:?}"
    );

    // Golden snapshot captured from the known-good (post-#2687) run and
    // verified — by stashing the refactor and re-running on pristine
    // `develop` physics — to be IEEE-754 bit-identical to the pre-change
    // output (`struct.pack('>d', x)` bit patterns matched for every element).
    // Bit-identical comparison; if any element differs the physics output
    // changed. The three configs use window U-values 0.5 / 0.9 / 1.3 with
    // heating=20°C, cooling=26°C.
    let golden = [0.10639163960830303, 0.0, 2.3658276261447644];
    assert_eq!(
        results.as_slice(),
        golden.as_slice(),
        "analytical-path EUI drifted from #2687 baseline (allocation refactor \
         must be numerically inert): got {results:?}"
    );
}

#[test]
fn cpu_surrogate_mock_path_eui_is_bit_identical_to_baseline() {
    let oracle = BatchOracle::from_model(create_single_zone_model());
    let population = deterministic_population(3);

    // CPU surrogate / mock path (use_surrogates = true, no ONNX model loaded).
    // This is the `run_cpu_surrogate` orchestrator hot loop whose
    // `get_temperatures_into` / `predict_loads_into` / `set_loads` calls were
    // made allocation-free in #2687. Result placement is by population index,
    // so the output is deterministic across rayon chunk-completion ordering.
    let results: Vec<f64> = oracle
        .evaluate_population(population, true)
        .expect("surrogate evaluate_population must succeed");

    assert!(
        results.iter().all(|r| r.is_finite()),
        "surrogate/mock EUIs must all be finite: {results:?}"
    );

    // Golden snapshot captured from the known-good (post-#2687) run and
    // verified bit-identical to the pre-change baseline (same stash-and-
    // re-run procedure as the analytical path above).
    let golden = [0.0, 0.0, 0.01781138835155858];
    assert_eq!(
        results.as_slice(),
        golden.as_slice(),
        "CPU-surrogate/mock-path EUI drifted from #2687 baseline (allocation \
         refactor must be numerically inert): got {results:?}"
    );
}

#[test]
fn hot_loop_is_deterministic_across_runs() {
    // Allocation refactors that reuse buffers can introduce non-determinism
    // if a buffer is ever read before being fully rewritten. Run the
    // analytical path twice and assert byte-for-byte identical output — this
    // catches any stale-buffer aliasing introduced by the #2687 reuse work.
    let oracle = BatchOracle::from_model(create_single_zone_model());
    let population = deterministic_population(4);

    let a = oracle
        .evaluate_population(population.clone(), false)
        .unwrap();
    let b = oracle.evaluate_population(population, false).unwrap();

    assert_eq!(
        a, b,
        "analytical hot loop must be deterministic across runs"
    );
}
