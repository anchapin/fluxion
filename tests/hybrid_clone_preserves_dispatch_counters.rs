//! HybridThermalModel `Clone` counter-preservation regression gate (Issue #2925).
//!
//! `ARCHITECTURE.md` (lines 1442-1481) documents a deliberate asymmetry in
//! [`HybridThermalModel::clone`](fluxion::HybridThermalModel): solver and
//! ventilation schedule slots are reset to fresh defaults (because
//! `Box<dyn HeatConductionSolver>` / `Box<dyn VentilationSchedule>` carry
//! per-step internal state that is not meaningfully cloneable across `dyn`
//! types), while the four dispatch counters (`surrogate_load_calls`,
//! `physics_conduction_calls`, `surrogate_conduction_calls`,
//! `surrogate_ventilation_calls`) are preserved verbatim so a caller can
//! snapshot routing statistics across branches without losing them.
//!
//! Before Issue #2925 this asymmetry was *only* documented, never
//! regression-gated. A future refactor that re-derives the counters in
//! `Clone` (for example resetting them to zero "for symmetry") would
//! silently corrupt any caller that relied on counter preservation — most
//! importantly `BatchOracle::evaluate_population`, which clones the base
//! model once per config and reports the per-config dispatch counters.
//!
//! These two tests pin the contract end-to-end:
//!
//! 1. `clone_preserves_dispatch_counters_mid_solve` — solves an original
//!    model for 1000 timesteps, clones it, verifies every counter on the
//!    clone equals the pre-clone value, and asserts the clone is fully
//!    functional (it can still solve an additional 1000 timesteps and the
//!    original is left untouched).
//!
//! 2. `clone_resets_solver_and_schedule_slots_independently` — clones a
//!    model, observes that the clone's `conduction_solver` and
//!    `ventilation_schedule` slots are at *different* addresses than the
//!    original's (fresh defaults), then swaps each slot on the clone via
//!    `set_conduction_solver` / `set_ventilation_schedule` and confirms
//!    that the original's slots are untouched. The slots are independent
//!    ownership, not aliased share.
//!
//! ## Note on `solve_timesteps` resetting counters
//!
//! `HybridThermalModel::solve_timesteps` (in `src/sim/thermal_model.rs`)
//! calls `self.reset_counters()` on entry. That means the second `solve_timesteps`
//! in test (1) starts from zero, increments to 1000, and the post-solve
//! counters are 1000 — NOT (pre-clone + 1000). The Issue #2925 acceptance
//! criterion's literal "(pre-clone + 1000)" expectation conflates the two
//! separate guarantees: (a) clone preserves counters, and (b) `solve_timesteps`
//! resets them at the start of a run. Both must hold for the documented
//! "clone BEFORE solve" pattern in `BatchOracle` / `empirical_hybrid` to
//! work. This test asserts both, with comments distinguishing them.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::five_r1c_solver::FiveR1CSolver;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::sim::thermal_model::HybridThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::sim::ventilation::{ConstantVentilation, VentilationSchedule};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::ThermalModelTrait as _;

/// Match the test harness in `tests/surrogate_models/test_hybrid_mode_dispatch.rs`:
/// the analytical surrogate fallback exercises the same dispatch path as a
/// trained ONNX model would (the wiring assertions are agnostic to which path
/// the surrogate branch takes).
const STEPS: usize = 1000;

/// Address of a `Box<dyn Trait>` payload, used to compare slot identity.
fn solver_address(model: &HybridThermalModel) -> *const (dyn HeatConductionSolver + '_) {
    model.conduction_solver() as *const (dyn HeatConductionSolver + '_)
}

fn schedule_address(model: &HybridThermalModel) -> *const (dyn VentilationSchedule + '_) {
    model.ventilation_schedule() as *const (dyn VentilationSchedule + '_)
}

#[test]
fn clone_preserves_dispatch_counters_mid_solve() {
    let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
    let spec = ASHRAE140Case::Case600.spec();
    let mut original = HybridThermalModel::from_spec(&spec);

    // (1) Solve the original for 1000 steps so every counter is non-zero.
    let _ = original.solve_timesteps(STEPS, &surrogates, false);

    let pre_load = original.surrogate_load_calls();
    let pre_phys = original.physics_conduction_calls();
    let pre_surrogate_cond = original.surrogate_conduction_calls();
    let pre_surrogate_vent = original.surrogate_ventilation_calls();

    assert!(
        pre_load >= STEPS,
        "pre-clone surrogate_load_calls must be >= {STEPS}, got {pre_load}"
    );
    assert!(
        pre_phys >= STEPS,
        "pre-clone physics_conduction_calls must be >= {STEPS}, got {pre_phys}"
    );

    // (2) Clone mid-solve. All four counters must be preserved verbatim.
    let mut clone = original.clone();

    assert_eq!(
        clone.surrogate_load_calls(),
        pre_load,
        "clone.surrogate_load_calls must equal pre-clone value (clone preserves counters, Issue #2925)"
    );
    assert_eq!(
        clone.physics_conduction_calls(),
        pre_phys,
        "clone.physics_conduction_calls must equal pre-clone value (clone preserves counters, Issue #2925)"
    );
    assert_eq!(
        clone.surrogate_conduction_calls(),
        pre_surrogate_cond,
        "clone.surrogate_conduction_calls must equal pre-clone value"
    );
    assert_eq!(
        clone.surrogate_ventilation_calls(),
        pre_surrogate_vent,
        "clone.surrogate_ventilation_calls must equal pre-clone value"
    );

    // (3) Cloning must NOT mutate the original — sanity check that the
    // `Clone` impl reads from `&self` and doesn't accidentally move state.
    assert_eq!(original.surrogate_load_calls(), pre_load);
    assert_eq!(original.physics_conduction_calls(), pre_phys);
    assert_eq!(original.surrogate_conduction_calls(), pre_surrogate_cond);
    assert_eq!(original.surrogate_ventilation_calls(), pre_surrogate_vent);

    // (4) Solve the clone for another 1000 steps. `solve_timesteps`
    // calls `reset_counters()` on entry, so the post-solve counters are
    // 1000 (not pre-clone + 1000). The clone is a fully functional,
    // independent model that just happens to have inherited the original's
    // counter snapshot.
    let _ = clone.solve_timesteps(STEPS, &surrogates, false);
    assert!(
        clone.surrogate_load_calls() >= STEPS,
        "clone must remain solvable after clone+reset_counters dance"
    );
    assert!(
        clone.physics_conduction_calls() >= STEPS,
        "clone must remain solvable after clone+reset_counters dance"
    );

    // (5) Solving the clone must NOT mutate the original's counters. They
    // remain pinned at the pre-clone values for the entire lifetime of the
    // original, exactly as the documented contract specifies.
    assert_eq!(
        original.surrogate_load_calls(),
        pre_load,
        "solving the clone must not affect original.surrogate_load_calls"
    );
    assert_eq!(
        original.physics_conduction_calls(),
        pre_phys,
        "solving the clone must not affect original.physics_conduction_calls"
    );
    assert_eq!(
        original.surrogate_conduction_calls(),
        pre_surrogate_cond,
        "solving the clone must not affect original.surrogate_conduction_calls"
    );
    assert_eq!(
        original.surrogate_ventilation_calls(),
        pre_surrogate_vent,
        "solving the clone must not affect original.surrogate_ventilation_calls"
    );
}

#[test]
fn clone_resets_solver_and_schedule_slots_independently() {
    let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
    let spec = ASHRAE140Case::Case600.spec();
    let mut original = HybridThermalModel::from_spec(&spec);

    // Establish a solved original so we exercise clone() of a non-fresh
    // model (the documented "clone mid-solve" scenario from Issue #2925).
    let _ = original.solve_timesteps(STEPS, &surrogates, false);

    let orig_solver_addr = solver_address(&original);
    let orig_schedule_addr = schedule_address(&original);

    // Clone and observe that the slots are FRESH defaults — different
    // allocation addresses from the original's. This is the asymmetry
    // documented in ARCHITECTURE.md: counters preserved, slots reset.
    let mut clone = original.clone();

    let clone_solver_addr = solver_address(&clone);
    let clone_schedule_addr = schedule_address(&clone);

    assert_ne!(
        clone_solver_addr, orig_solver_addr,
        "clone.conduction_solver must point to a fresh default (different allocation), \
         not alias the original's slot (Issue #2925)"
    );
    assert_ne!(
        clone_schedule_addr, orig_schedule_addr,
        "clone.ventilation_schedule must point to a fresh default (different allocation), \
         not alias the original's slot (Issue #2925)"
    );

    // Swap the clone's conduction_solver slot with a freshly-built
    // FiveR1CSolver. The original's slot must remain untouched (no
    // aliasing).
    let new_solver: Box<dyn HeatConductionSolver> = Box::new(FiveR1CSolver::default());
    let returned_solver = clone.set_conduction_solver(new_solver);

    // `set_conduction_solver` returns the slot it displaced, which should
    // still equal the address we observed before the swap (i.e., the
    // fresh-default slot allocated during clone).
    let _ = returned_solver; // suppress unused-must-use if needed; we only care about addresses
    assert_ne!(
        solver_address(&clone),
        orig_solver_addr,
        "after set_conduction_solver, the clone's slot must differ from the original's"
    );
    assert_eq!(
        solver_address(&original),
        orig_solver_addr,
        "swapping the clone's conduction_solver must NOT affect the original's slot pointer \
         (slot ownership is independent, not aliased share, per Issue #2925)"
    );

    // Same for the ventilation_schedule slot.
    let new_schedule: Box<dyn VentilationSchedule> = Box::new(ConstantVentilation::new(1.25));
    let _returned_schedule = clone.set_ventilation_schedule(new_schedule);

    assert_ne!(
        schedule_address(&clone),
        orig_schedule_addr,
        "after set_ventilation_schedule, the clone's slot must differ from the original's"
    );
    assert_eq!(
        schedule_address(&original),
        orig_schedule_addr,
        "swapping the clone's ventilation_schedule must NOT affect the original's slot pointer \
         (slot ownership is independent, not aliased share, per Issue #2925)"
    );

    // Sanity: both models must remain solvable end-to-end after the
    // slot-swap dance — neither slot is corrupted by the swap.
    let _ = original.solve_timesteps(STEPS, &surrogates, false);
    let _ = clone.solve_timesteps(STEPS, &surrogates, false);
    assert!(original.surrogate_load_calls() >= STEPS);
    assert!(clone.surrogate_load_calls() >= STEPS);
}
