//! Issue #2523 — regression + determinism test for the HybridThermalModel
//! per-timestep tracing migration.
//!
//! Before the fix, the default hybrid routing path emitted `tracing::info!`
//! on **every** timestep (5 branch sites in `src/sim/thermal_model.rs`).
//! With a production subscriber installed at `INFO` level that produced up
//! to 5 × 8 760 = 43 800 structured-log events per simulation — and 43.8 M
//! per 1 000-config `BatchOracle::evaluate_population`.
//!
//! This test:
//!   1. Installs a counting `tracing` subscriber at `INFO` level (the
//!      production-realistic default) for the duration of the solve — scoped
//!      via `with_default` so it cannot leak into other tests.
//!   2. Runs a full annual `HybridThermalModel::solve_timesteps` (8 760 steps,
//!      default hybrid routing which fires the surrogate-load and
//!      physics-conduction branches every step).
//!   3. Asserts **zero** events reached the `INFO`-level subscriber — proving
//!      the per-timestep diagnostics are now `trace!` (8.76 M → 0 at the
//!      production log level).
//!   4. Re-runs the solve on a fresh clone and asserts byte-identical EUI —
//!      the migration is side-effect-free on the computation, so the
//!      determinism gate (#1351) is preserved.
//!
//! Note on `tracing` semantics: `tracing::trace!` callsites are statically
//! disabled when the active subscriber's max-level hint is `INFO`, so they
//! perform zero dispatch work and cannot perturb the numerical result.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::sim::thermal_model::{HybridRouting, HybridThermalModel, ThermalModelTrait};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry;

/// Counting layer: increments a shared counter once per event that reaches it.
struct CountingLayer {
    count: std::sync::Arc<AtomicU64>,
}

impl<S> Layer<S> for CountingLayer
where
    S: tracing::Subscriber,
{
    fn on_event(&self, _event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

/// Run `f` under a scoped `INFO`-level subscriber that counts every recorded
/// event. Returns the event count observed during the run. Using
/// `with_default` (thread-local dispatch) keeps the subscriber local to this
/// test and never disturbs the global default other tests rely on.
fn run_under_info_subscriber<F: FnOnce()>(f: F) -> u64 {
    let counter = std::sync::Arc::new(AtomicU64::new(0));
    let layer = CountingLayer {
        count: counter.clone(),
    };
    let subscriber = registry().with(LevelFilter::INFO).with(layer);
    tracing::subscriber::with_default(subscriber, f);
    counter.load(Ordering::Relaxed)
}

#[test]
fn hybrid_hotloop_emits_zero_info_events_after_2523() {
    let spec = ASHRAE140Case::Case600.spec();
    // Default routing: `use_surrogate_loads = true`, everything else physics.
    // This fires the surrogate-load branch (when the mock manager returns Ok)
    // and the physics-conduction branch on every one of the 8 760 steps —
    // precisely the calls that were `info!` before #2523.
    let mut model = HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::default());
    let surrogates = SurrogateManager::new().expect("mock SurrogateManager");

    let recorded = run_under_info_subscriber(|| {
        let _eui = model.solve_timesteps(8760, &surrogates, true);
    });

    // The entire annual solve must not emit a single event at INFO level.
    // Before #2523 this would have been thousands of `info!` events.
    assert_eq!(
        recorded, 0,
        "per-timestep diagnostics must be trace! (filtered at INFO); \
         found {recorded} INFO-level events during an 8760-step hybrid solve"
    );
}

#[test]
fn hybrid_hotloop_determinism_unaffected_by_trace_migration() {
    // The trace! migration is purely observational: tracing is side-effect-free
    // on the computation. Two solves with identical inputs must yield identical
    // EUI, preserving the determinism gate (#1351).
    let spec = ASHRAE140Case::Case600.spec();
    let surrogates = SurrogateManager::new().expect("mock SurrogateManager");

    let mut model_a = HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::default());
    let mut model_b = HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::default());

    let eui_a = model_a.solve_timesteps(8760, &surrogates, true);
    let eui_b = model_b.solve_timesteps(8760, &surrogates, true);

    assert!(eui_a.is_finite(), "EUI must be finite");
    // bit-equality: no nondeterminism introduced by the logging change.
    assert_eq!(
        eui_a.to_bits(),
        eui_b.to_bits(),
        "determinism violation: EUI changed across identical solves ({} != {})",
        eui_a,
        eui_b
    );
}

#[test]
fn hybrid_hotloop_all_surrogate_routing_also_emits_zero_info() {
    // Exercise the surrogate-conduction and surrogate-ventilation branches too
    // (all_surrogate routing) to confirm every migrated call site is trace!.
    let spec = ASHRAE140Case::Case600.spec();
    let mut model =
        HybridThermalModel::from_spec_with_routing(&spec, HybridRouting::all_surrogate());
    let surrogates = SurrogateManager::new().expect("mock SurrogateManager");

    let recorded = run_under_info_subscriber(|| {
        // A shorter run keeps the all-surrogate path fast; the per-timestep
        // assertion holds for any step count.
        let _eui = model.solve_timesteps(240, &surrogates, true);
    });

    assert_eq!(
        recorded, 0,
        "all-surrogate routing must also emit zero INFO events (240-step run)"
    );
}
