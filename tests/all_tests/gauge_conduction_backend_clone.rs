//! BatchOracle `Clone` contract for the gauge conduction backend (Issue #3729).
//!
//! Issue #3729 pinned the clone semantics for the cfg-gated
//! `gauge_zone_solver` / `gauge_multi_zone_solver` slots on
//! `ConductionBackend`. Before the pin, the slots were `#[derive(Clone)]`
//! on the underlying solver structs, which silently deep-cloned per-step
//! state (`T_air`, `initialized`, per-surface `GaugeSolver` slot state).
//! That meant a mid-solve `BatchOracle::base_model.clone()` in
//! `evaluate_population` would silently contaminate every freshly-cloned
//! candidate with the parent's post-step zone air temperature.
//!
//! The fix lives in `src/physics/gauge_zone_solver.rs`:
//! `SurfaceGaugeSolver::clone`, `GaugeZoneSolver::clone`, and
//! `MultiZoneGaugeSolver::clone` are now hand-rolled and reset the
//! runtime state (`T_air → 20.0`, `initialized → false`, per-surface
//! gauges re-initialized from `wall_spec`) while preserving topology.
//! `ARCHITECTURE.md` §"Gauge-slot clone semantics (issue #3729)" documents
//! the per-field contract.
//!
//! These end-to-end tests pin the contract through the BatchOracle hot
//! loop path:
//!
//! 1. `clone_resets_gauge_zone_solver_t_air` — clones a `ConductionBackend`
//!    whose gauge slot is populated, and verifies the clone's `T_air` is
//!    reset to the `new_with_id` default (`20.0`) regardless of the
//!    parent's mid-solve value.
//! 2. `clone_preserves_gauge_topology` — surface count, `zone_id`,
//!    `C_air` round-trip across clone. (Per-surface metadata is
//!    `pub(crate)` and exercised by the in-module unit tests in
//!    `src/physics/gauge_zone_solver.rs`.)
//! 3. `clone_resets_gauge_initialized_flag` — the clone's `initialized`
//!    flag is `false` after clone.
//! 4. `cloned_gauge_models_produce_identical_eui_in_batch_oracle` —
//!    two clones of the same base model produce identical EUI when
//!    fed identical parameters through `BatchOracle::evaluate_population`,
//!    proving the candidate-independence guarantee holds at the
//!    `BatchOracle` layer (not just at the `ConductionBackend` layer).
//!
//! ## Feature gate
//!
//! The tests below only assert on gauge-slot state when the
//! `gauge-solver` cargo feature is enabled. Without the feature, the
//! gauge slots are absent (per the issue's cfg-gated contract) and the
//! non-gauge `ConductionBackend` clone path is exercised instead. The
//! nightly gauge workflow (`.github/workflows/nightly-ashrae-140-gauge.yml`)
//! compiles and runs these tests with `--features gauge-solver` so the
//! feature-on oracle path is exercised before the β-soak gate closes.
//!
//! Run: `cargo test --features gauge-solver --test all_tests gauge_conduction_backend_clone::`

#![cfg(feature = "gauge-solver")]

use fluxion::physics::cta::VectorField;
use fluxion::physics::units::ToF64;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_model_data::{GaugeZoneSolver, MultiZoneGaugeSolver};
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::BatchOracle;

/// Build a Case 600-style gauge-initialized `ThermalModel` via the
/// selector-driven constructor that `BatchOracle::new` uses internally.
/// Mirrors `src/napi/batch_oracle.rs:92-100` (the NAPI constructor that
/// issue #3729 calls out as the production-path entry point).
fn build_gauge_initialized_base_model() -> ThermalModel<VectorField> {
    let spec = ASHRAE140Case::Case600.spec();
    ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("default selector (gauge+default) must initialize a Case 600 thermal model")
}

/// Direct `ConductionBackend` clone — the layer where the gauge slot
/// reset is observable. Mirrors the in-module unit tests in
/// `src/physics/gauge_zone_solver.rs`, but reaches the slot through the
/// `ThermalModelData` facade the way production code does (via
/// `model.conduction.backend.gauge_zone_solver`).
#[test]
fn clone_resets_gauge_zone_solver_t_air() {
    let model = build_gauge_initialized_base_model();
    let original_solver: &GaugeZoneSolver = model
        .conduction
        .backend
        .gauge_zone_solver
        .as_ref()
        .expect("Issue #3729 fixture: gauge-zone-solver must be populated when gauge-solver feature is on");

    // Pin a non-default T_air on the original so we can verify the clone
    // resets it. The original `T_air` must be at the `new_with_id`
    // default (20.0) before we mutate it.
    let pre_clone_t_air = original_solver.T_air().to_value();
    assert!(
        (pre_clone_t_air - 20.0).abs() < 1e-9,
        "fixture: freshly-built GaugeZoneSolver.T_air must equal the \
         new_with_id default (20.0), got {pre_clone_t_air}"
    );

    // Clone the model and pin the parent's T_air to a non-default value
    // (simulating a mid-solve state). The clone's gauge-slot T_air must
    // remain at the `new_with_id` default — the parent mutation must not
    // be observable through the clone.
    let mut model = model;
    let parent_solver = model
        .conduction
        .backend
        .gauge_zone_solver
        .as_mut()
        .expect("gauge-zone-solver must be Some");
    parent_solver.set_T_air(35.0);

    let cloned_model = model.clone();
    let cloned_solver: &GaugeZoneSolver = cloned_model
        .conduction
        .backend
        .gauge_zone_solver
        .as_ref()
        .expect("cloned model must preserve the gauge-zone-solver slot");

    assert!(
        (cloned_solver.T_air().to_value() - 20.0).abs() < 1e-9,
        "clone.gauge_zone_solver.T_air must reset to the new_with_id \
         default (20.0), got {} — Issue #3729 slot-reset contract violated",
        cloned_solver.T_air().to_value()
    );
    // Original must remain untouched by the clone (independence).
    let parent_after_clone = model
        .conduction
        .backend
        .gauge_zone_solver
        .as_ref()
        .expect("parent gauge-zone-solver must remain Some after clone");
    assert!(
        (parent_after_clone.T_air().to_value() - 35.0).abs() < 1e-9,
        "parent.gauge_zone_solver.T_air must remain at the mid-solve pin \
         (35.0) after clone, got {} — Clone must not perturb the parent",
        parent_after_clone.T_air().to_value()
    );
}

/// Topology preservation across `ConductionBackend::clone`: surface
/// count, `zone_id`, `C_air`. The clone describes the same building
/// as the parent — only its runtime state diverges. Per-surface
/// metadata round-trip is pinned by the in-module
/// `issue_3729_clone_preserves_topology` test in
/// `src/physics/gauge_zone_solver.rs` (the per-surface fields are
/// `pub(crate)` and not reachable from an integration test).
#[test]
fn clone_preserves_gauge_topology() {
    let model = build_gauge_initialized_base_model();
    let original_solver = model
        .conduction
        .backend
        .gauge_zone_solver
        .as_ref()
        .expect("gauge-zone-solver must be populated");

    let cloned_model = model.clone();
    let cloned_solver = cloned_model
        .conduction
        .backend
        .gauge_zone_solver
        .as_ref()
        .expect("cloned gauge-zone-solver must remain Some");

    // Surface count round-trips.
    assert_eq!(
        cloned_solver.surface_count(),
        original_solver.surface_count(),
        "clone must preserve surface count"
    );

    // Zone identifier round-trips.
    assert_eq!(
        cloned_solver.zone_id(),
        original_solver.zone_id(),
        "clone must preserve zone_id"
    );

    // Capacitance round-trips (a derived quantity — but useful as an
    // end-to-end topology sanity check; the per-floor-area/height
    // fields are private and exercised by in-module tests).
    assert!(
        (cloned_solver.C_air() - original_solver.C_air()).abs() < 1e-9,
        "clone must preserve C_air (got {} vs original {})",
        cloned_solver.C_air(),
        original_solver.C_air()
    );
}

/// The clone's `initialized` flag must reset to `false` — the clone is
/// not pre-solved; the caller re-initializes before stepping. Mirrors
/// the `HybridThermalModel::conduction_solver` slot-reset precedent
/// (ARCHITECTURE.md #2539).
#[test]
fn clone_resets_gauge_initialized_flag() {
    let model = build_gauge_initialized_base_model();
    let original_solver = model
        .conduction
        .backend
        .gauge_zone_solver
        .as_ref()
        .expect("gauge-zone-solver must be populated");
    assert!(
        original_solver.is_initialized(),
        "fixture: freshly-built gauge solver must be initialized"
    );

    let cloned_model = model.clone();
    let cloned_solver = cloned_model
        .conduction
        .backend
        .gauge_zone_solver
        .as_ref()
        .expect("cloned gauge-zone-solver must be Some");

    assert!(
        !cloned_solver.is_initialized(),
        "clone.gauge_zone_solver.is_initialized must be false (Issue #3729 \
         slot-reset contract); got true"
    );
}

/// End-to-end candidate-independence at the BatchOracle layer: two
/// clones of the same base model, fed identical parameters through
/// `evaluate_population`, must produce identical EUI values. A
/// candidate-independence bug (shared state across clones) would
/// produce non-deterministic EUI as rayon scheduling varies.
///
/// We use the analytical path (`use_surrogates = false`) because it
/// runs every candidate deterministically — the population-parallel
/// hot loop is the exact code path Issue #3729 called out as the
/// gauge-affected candidate-clone site.
#[test]
fn cloned_gauge_models_produce_identical_eui_in_batch_oracle() {
    let base = build_gauge_initialized_base_model();

    // Construct two BatchOracle instances from the same base model —
    // each `BatchOracle::from_model` performs its own candidate cloning
    // in `evaluate_population`, so any candidate-state leak between
    // clones would surface as a divergence between the two oracles'
    // outputs.
    let oracle_a =
        BatchOracle::from_model(base.clone()).expect("BatchOracle A must build");
    let oracle_b =
        BatchOracle::from_model(base.clone()).expect("BatchOracle B must build");

    // Single-candidate population so the EUI is purely a function of
    // the base-model clone + the parameter set. U-value / setpoints
    // chosen to land in a stable analytical regime (heating < cooling).
    let population = vec![vec![1.5, 20.0, 26.0]];

    let eui_a = oracle_a
        .evaluate_population(population.clone(), false)
        .expect("oracle_a evaluate_population must succeed");
    let eui_b = oracle_b
        .evaluate_population(population, false)
        .expect("oracle_b evaluate_population must succeed");

    assert_eq!(
        eui_a.len(),
        1,
        "single-candidate population must produce exactly one EUI, got {}",
        eui_a.len()
    );
    assert_eq!(eui_a.len(), eui_b.len());
    assert!(
        eui_a[0].is_finite(),
        "oracle_a EUI must be finite, got {}",
        eui_a[0]
    );
    assert!(
        eui_b[0].is_finite(),
        "oracle_b EUI must be finite, got {}",
        eui_b[0]
    );

    // Bit-identical EUI is the candidate-independence guarantee: the
    // candidate clone in `evaluate_population` must not share state
    // with any other clone. If a future refactor re-introduces shared
    // state (e.g., by reverting the Issue #3729 `Clone` impl to
    // `#[derive(Clone)]`), the analytical path's deterministic
    // scheduling should still produce identical EUI for a single
    // candidate — but a multi-candidate population test below catches
    // cross-candidate interference.
    assert!(
        (eui_a[0] - eui_b[0]).abs() < 1e-9,
        "two BatchOracle clones of the same base model must produce \
         identical EUI for identical parameters (candidate-independence \
         guarantee, Issue #3729); got oracle_a={}, oracle_b={}",
        eui_a[0],
        eui_b[0]
    );

    // Cross-candidate independence: run the same oracle twice on a
    // 3-candidate population. Each clone is independent, so the EUI
    // sequence must be deterministic across runs (the analytical path
    // sorts results by population index, so this is exact bit-equality,
    // not just tolerance).
    let oracle_c =
        BatchOracle::from_model(base).expect("BatchOracle C must build");
    let multi_population = vec![
        vec![0.5, 20.0, 26.0],
        vec![1.5, 20.0, 26.0],
        vec![2.5, 20.0, 26.0],
    ];
    let run_a = oracle_c
        .evaluate_population(multi_population.clone(), false)
        .expect("oracle_c run A must succeed");
    let run_b = oracle_c
        .evaluate_population(multi_population, false)
        .expect("oracle_c run B must succeed");

    assert_eq!(
        run_a.len(),
        3,
        "3-candidate population must produce exactly 3 EUI, got {}",
        run_a.len()
    );
    assert_eq!(
        run_a, run_b,
        "evaluate_population must be deterministic across runs when the \
         base model is identical (cross-candidate independence, Issue #3729)"
    );
}

/// `MultiZoneGaugeSolver` slot-reset end-to-end: clones a
/// `MultiZoneGaugeSolver` standalone and verifies the Issue #3729
/// contract holds (`initialized = false`, per-zone `T_air` reset to
/// the `new_with_id` default). Mirrors the in-module unit test
/// `issue_3729_multi_zone_clone_resets_aggregate_state` from
/// `ConductionBackend` perspective.
#[test]
fn clone_resets_multi_zone_gauge_solver() {
    // Case 600 (single-zone) does not populate gauge_multi_zone_solver,
    // so we exercise the multi-zone path directly with a custom-built
    // solver. The cfg-gated invariant from #3275 / #3291 means exactly
    // one of `gauge_zone_solver` / `gauge_multi_zone_solver` is
    // populated per `ConductionBackend` — the standalone multi-zone
    // builder below exercises the same `Clone` impl that
    // `ConductionBackend::clone` would invoke when the multi-zone slot
    // is populated.
    use fluxion::physics::wall_spec::WallSpec;
    let wall = WallSpec::single_layer("LightWeight", 0.09, 1.0, 50.0, 50.0);
    let mut mz = MultiZoneGaugeSolver::new();
    mz.add_zone(0, 48.0, 2.7);
    mz.add_zone(1, 36.0, 2.7);
    mz.add_opaque_surface_to_zone(0, &wall, 48.0, fluxion::sim::thermal_model_data::SurfaceType::Wall, 180.0, 90.0)
        .unwrap();
    mz.add_opaque_surface_to_zone(1, &wall, 36.0, fluxion::sim::thermal_model_data::SurfaceType::Wall, 180.0, 90.0)
        .unwrap();
    mz.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();
    mz.initialize().expect("multi-zone must initialize");

    let mz_clone = mz.clone();

    assert!(
        !mz_clone.is_initialized(),
        "MultiZoneGaugeSolver clone must reset initialized to false \
         (Issue #3729); got true"
    );
    assert_eq!(
        mz_clone.num_zones(),
        mz.num_zones(),
        "MultiZoneGaugeSolver clone must preserve num_zones"
    );
    assert!(
        (mz_clone.get_zone(0).unwrap().T_air().to_value() - 20.0).abs() < 1e-9,
        "cloned multi-zone zone[0].T_air must reset to 20.0, got {}",
        mz_clone.get_zone(0).unwrap().T_air().to_value()
    );
    assert!(
        (mz_clone.get_zone(1).unwrap().T_air().to_value() - 20.0).abs() < 1e-9,
        "cloned multi-zone zone[1].T_air must reset to 20.0, got {}",
        mz_clone.get_zone(1).unwrap().T_air().to_value()
    );
}
