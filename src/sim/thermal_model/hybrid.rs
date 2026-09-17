//! Hybrid thermal-model family — Issue #3789 module split.
//!
//! Decomposed out of `thermal_model.rs` so the parent module can stay
//! under the Issue #3457 module-size ratchet. Hosts:
//!
//! - [`HybridRouting`] — per-subsystem routing flags (Issue #1431)
//! - [`MetricsSnapshot`] — dispatch-counter snapshot (Issue #1608)
//! - [`HybridThermalModel`] — concrete per-component dispatching model
//!   (Issue #1431) with `Box<dyn HeatConductionSolver>` and
//!   `Box<dyn VentilationSchedule>` slots (Issue #2457), pre-allocated
//!   scratch buffers (Issues #2860 / #2921), and the OOD-aware fallback
//!   behaviour from Issue #1892.
//! - [`default_conduction_solver`] / [`default_ventilation_schedule`] —
//!   the no-op-swap defaults installed in [`HybridThermalModel::new`].

use fluxion_twin::TwinCorrection;

use super::comfort::compute_pmv_ppd_and_adaptive;
use super::{ThermalModelMode, ThermalModelTrait, ZoneComfortMetrics};
use crate::ai::surrogate::SurrogateManager;
use crate::sim::engine::ThermalModel;
use crate::sim::thermal_model_data::{
    lightweight_wall_spec, ContinuousTensor, FiveR1CSolver, FromF64, HeatConductionSolver,
    HeatTransferCoefficient, Temperature, Time, ToF64, VectorField,
};
use crate::sim::thermal_selector::ThermalSelector;
use crate::sim::ventilation::{ConstantVentilation, VentilationSchedule};
// Issue #2523: per-timestep HybridThermalModel diagnostics were emitted
// at `info!` level, producing up to 8.76M (5 branches × 8760 steps × 1
// config) structured-log invocations per BatchOracle population even when
// the level filter discarded them. They are now `trace!` — available
// under verbose tracing (`RUST_LOG=trace`) but zero-cost at the default
// INFO/WARN release filter. This is consistent with the `debug-physics`
// hot-loop gating pattern (#1967): per-timestep diagnostics must never
// pay formatting/dispatch cost in the production binary.
use tracing::trace;

/// Per-subsystem routing policy for [`HybridThermalModel`] (Issue #1431).
///
/// Each flag selects whether the corresponding subsystem consults the
/// [`SurrogateManager`] (true) or stays on the analytical/physics path
/// (false). Subsystems are deliberately fine-grained so callers can route
/// only the high-value / low-risk subsystem to ML while leaving safety-
/// critical subsystems on physics (per the Phase-3 validation envelope
/// in `ARCHITECTURE.md` §Validation Strategy).
///
/// # OOD-aware routing (Issue #1892)
///
/// When `use_ood_fallback` is `true`, the hybrid model performs an OOD
/// check before each surrogate inference call. If the input vector falls
/// outside the stored training bounds, the model transparently reroutes
/// to the analytical physics solver and emits an `OodInputWarning` for
/// each out-of-bounds feature. This prevents the surrogate from silently
/// extrapolating on inputs it was never trained on (e.g. extreme weather
/// from untrusted EPW data or unphysical internal gains).
///
/// # Default policy
///
/// [`HybridThermalModel`] is constructed with a default policy that routes
/// **load prediction only** to the surrogate and keeps conduction,
/// ventilation, and HVAC on physics. This is the highest-value / lowest-
/// risk split per Issue #1431 acceptance criteria.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridRouting {
    /// Route conduction (5R1C / 9R4C thermal network solve) to the surrogate.
    pub use_surrogate_conduction: bool,
    /// Route ventilation heat transfer (h_ve) to the surrogate.
    pub use_surrogate_ventilation: bool,
    /// Route internal/external load prediction to the surrogate.
    pub use_surrogate_loads: bool,
    /// Route HVAC power demand to the surrogate.
    pub use_surrogate_hvac: bool,
    /// When `true`, check inputs against training bounds before surrogate
    /// inference and fall back to the physics solver when OOD is detected
    /// (Issue #1892). When `false` (default), no OOD check is performed.
    pub use_ood_fallback: bool,
}

impl Default for HybridRouting {
    fn default() -> Self {
        Self {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: false,
            use_surrogate_loads: true,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        }
    }
}

impl HybridRouting {
    /// All subsystems on physics (equivalent to `ThermalModelMode::Physics`).
    pub const fn all_physics() -> Self {
        Self {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        }
    }

    /// All subsystems on surrogate (equivalent to `ThermalModelMode::Surrogate`).
    pub const fn all_surrogate() -> Self {
        Self {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: true,
            use_surrogate_loads: true,
            use_surrogate_hvac: true,
            use_ood_fallback: false,
        }
    }

    /// OOD-aware routing: surrogate load prediction with automatic physics
    /// fallback when inputs fall outside training bounds (Issue #1892).
    /// All other subsystems remain on physics. Use this for safety-critical
    /// deployments where the surrogate may receive untrusted EPW data.
    pub const fn ood_fallback() -> Self {
        Self {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: false,
            use_surrogate_loads: true,
            use_surrogate_hvac: false,
            use_ood_fallback: true,
        }
    }
}

/// Structured snapshot of [`HybridThermalModel`] dispatch counters (Issue #1608).
///
/// Returned by [`HybridThermalModel::metrics`]. Callers can inspect the
/// counters and routing configuration without accessing the inner model.
#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    /// Number of times the surrogate load-prediction branch fired.
    pub surrogate_load_calls: usize,
    /// Number of times the physics conduction solver was called.
    ///
    /// Renamed from `physics_step_calls` in Issue #2457: now reflects ONLY
    /// the analytical physics conduction path and does NOT increment when
    /// `use_surrogate_conduction` reroutes conduction to the
    /// `Box<dyn HeatConductionSolver>` slot. External consumers should
    /// rely on this counter for any "did the physics solver fire?"
    /// assertion; the surrogate counter is the inverse.
    pub physics_conduction_calls: usize,
    /// Number of times the surrogate conduction branch fired (Issue #1702).
    pub surrogate_conduction_calls: usize,
    /// Number of times the surrogate ventilation branch fired (Issue #1702).
    pub surrogate_ventilation_calls: usize,
    /// Current execution mode.
    pub mode: ThermalModelMode,
    /// Number of thermal zones.
    pub num_zones: usize,
    /// Active routing policy.
    pub routing: HybridRouting,
}

impl MetricsSnapshot {
    /// Returns `true` when no dispatch branch has fired yet.
    pub fn is_zero(&self) -> bool {
        self.surrogate_load_calls == 0
            && self.physics_conduction_calls == 0
            && self.surrogate_conduction_calls == 0
            && self.surrogate_ventilation_calls == 0
    }
}

/// Per-component hybrid thermal model (Issue #1431).
///
/// `HybridThermalModel` is the concrete implementation behind
/// [`ThermalModelMode::Hybrid`]. Unlike [`UnifiedThermalModel`], which
/// silently downgrades `Hybrid` to `Physics` (the legacy bug fixed by
/// this issue), `HybridThermalModel` actually dispatches per-component:
/// for every subsystem named in [`HybridRouting`] whose flag is `true`,
/// the corresponding surrogate path is taken; otherwise the analytical /
/// physics path is taken.
///
/// The default policy is [`HybridRouting::default`] (loads → surrogate,
/// everything else → physics), which is the highest-value + lowest-risk
/// split called out in Issue #1431's acceptance criteria.
///
/// `HybridThermalModel` is `Clone`-by-design (AGENTS.md, "Module
/// Boundaries") so report generators such as the
/// `validation::empirical_hybrid` harness (Issue #1846) can run a fresh
/// hybrid solve on a cloned model without disturbing the caller's
/// instance.
///
/// # Clone asymmetry (Issue #2539)
///
/// The hand-rolled `impl Clone for HybridThermalModel` (see below) has an
/// asymmetric split: solver/schedule slots are reset to fresh defaults,
/// while the routing counters are preserved verbatim. This is intentional
/// and documented as part of the swap-point contract in `ARCHITECTURE.md`
/// §"Thermal Model Trait Hierarchy" → "Clone semantics & BatchOracle
/// parallelism contract". Contract summary for callers:
///
/// 1. **Clone BEFORE `solve_timesteps`** — every in-tree caller
///    (`BatchOracle::evaluate_population`, `empirical_hybrid`) does this,
///    so the preserved counters are zero and the reset solver slots agree
///    with them.
/// 2. **Cloning AFTER `solve_timesteps` yields counters that do not
///    correspond to the clone's fresh solver state.** Call
///    `reset_counters()` on the clone before re-solving, or your
///    published routing counters will describe the *previous* run.
/// 3. **Custom `conduction_solver` / `ventilation_schedule` slots do not
///    round-trip** — they are replaced with defaults on clone. Re-install
///    via `set_conduction_solver` / `set_ventilation_schedule` on the clone.
pub struct HybridThermalModel {
    inner: ThermalModel<VectorField>,
    routing: HybridRouting,
    /// Per-subsystem routing slots (Issue #2457).
    ///
    /// `Box<dyn HeatConductionSolver>` — replaces the legacy
    /// `use_surrogate_conduction` counter-only stub. Initially holds a
    /// [`FiveR1CSolver::default()`] so the dispatch is a no-op swap for
    /// the default routing; a future commit can swap in an ONNX-trained
    /// conduction surrogate by replacing this field via
    /// [`HybridThermalModel::set_conduction_solver`].
    ///
    /// `Box<dyn VentilationSchedule>` — replaces the legacy
    /// `use_surrogate_ventilation` counter-only stub. Initially holds a
    /// [`ConstantVentilation::new(0.5)`]; replaceable via
    /// [`HybridThermalModel::set_ventilation_schedule`].
    ///
    /// The slots exist regardless of the routing flags so that toggling
    /// a flag at runtime only changes the dispatch path, not the slot's
    /// lifecycle. See `ARCHITECTURE.md` §Thermal Model Trait Hierarchy.
    conduction_solver: Box<dyn HeatConductionSolver>,
    ventilation_schedule: Box<dyn VentilationSchedule>,
    /// Number of times the surrogate load predictor was consulted.
    /// Tracked independently of the inner model's instrumentation so
    /// callers (and tests) can verify the surrogate branch actually fired.
    surrogate_load_calls: usize,
    /// Number of times the physics conduction solver was called.
    ///
    /// Renamed from `physics_step_calls` in Issue #2457: this counter is
    /// now incremented ONLY when the analytical physics path actually
    /// fires. When `use_surrogate_conduction` is `true` the dispatcher
    /// routes through `conduction_solver.step(...)` and this counter
    /// stays at zero — the regression test
    /// `hybrid_conduction_flag_routes_through_slot_not_physics` guards
    /// the no-op anti-pattern closed by Issue #1702.
    physics_conduction_calls: usize,
    /// Number of times the surrogate conduction branch fired.
    /// Incremented when `routing.use_surrogate_conduction` is `true`
    /// (Issue #1702, wired by Issue #2457).
    surrogate_conduction_calls: usize,
    /// Number of times the surrogate ventilation branch fired.
    /// Incremented when `routing.use_surrogate_ventilation` is `true`
    /// (Issue #1702, wired by Issue #2457).
    surrogate_ventilation_calls: usize,
    /// Reuse buffer for [`SurrogateManager::predict_loads_into`] (Issue #2921).
    ///
    /// Pre-allocated once in [`HybridThermalModel::new`] / `from_spec` /
    /// `from_spec_with_routing`, kept across timesteps via
    /// `Vec::clear()` (which preserves capacity), and reset on
    /// [`HybridThermalModel::reset_counters`]. After the first timestep
    /// it holds `num_zones` `f64` slots so the per-step surrogate-load
    /// hot loop performs **zero** heap allocation — replacing the
    /// per-step `Vec<f64>` that `predict_loads_with_fallback` returned
    /// from each `predict_loads_onnx_impl` success path (Issue #2860).
    surrogate_load_scratch: Vec<f64>,
    /// Reuse buffer for hourly zone temperature snapshots (Issue #2860).
    ///
    /// Pre-allocated to `num_zones` inner Vecs in
    /// [`HybridThermalModel::new`] / `from_spec` / `from_spec_with_routing`.
    /// Each `solve_timesteps` call clears the inner Vecs (preserving
    /// capacity) and grows them lazily to `steps` capacity on the first
    /// call where `steps` exceeds the previous capacity. After warm-up
    /// the per-step `push` stays zero-alloc — replacing:
    ///
    /// 1. the per-call `Some(vec![Vec::with_capacity(steps); num_zones])`
    ///    allocation at the top of `solve_timesteps` (Issue #1846) that
    ///    fired on every solve (8 760 × `pop_size` per annual sweep).
    /// 2. the per-step `self.inner.temperatures.as_ref().to_vec()` copy
    ///    that broke the borrow conflict between
    ///    `self.inner.temperatures` (read) and
    ///    `self.inner.diagnostics_state.hourly_temperatures` (write).
    ///    The new write path borrows `self.inner.temperatures`
    ///    immutably and `self.hourly_buf` mutably — non-overlapping
    ///    borrows — so the snapshot copy is gone.
    ///
    /// Cloned shallowly on [`Clone`] (the inner Vecs start empty, so each
    /// clone gets a fresh outer Vec sized to `num_zones`; the first solve
    /// on the clone grows capacity the same way as a fresh construction).
    hourly_buf: Vec<Vec<f64>>,
}

impl Clone for HybridThermalModel {
    fn clone(&self) -> Self {
        // Solver / schedule slots are reset to fresh defaults on clone.
        // The `validation::empirical_hybrid` harness (Issue #1846)
        // clones models BEFORE solving them, so per-step solver state
        // never needs to round-trip across clones. Counters are
        // preserved (the caller can `reset_counters()` if they want a
        // clean slate before solving). The surrogate-load scratch buffer
        // is cloned verbatim — its capacity is preserved so the first
        // solve on the clone stays zero-alloc.
        Self {
            inner: self.inner.clone(),
            routing: self.routing,
            conduction_solver: default_conduction_solver(),
            ventilation_schedule: default_ventilation_schedule(),
            surrogate_load_calls: self.surrogate_load_calls,
            physics_conduction_calls: self.physics_conduction_calls,
            surrogate_conduction_calls: self.surrogate_conduction_calls,
            surrogate_ventilation_calls: self.surrogate_ventilation_calls,
            surrogate_load_scratch: self.surrogate_load_scratch.clone(),
            // Issue #2860: clones get a fresh outer Vec sized to
            // `num_zones` with empty inner Vecs. The first
            // `solve_timesteps` call on the clone grows inner Vec
            // capacity the same way as a fresh construction (cheap
            // one-time cost per clone, zero per-step overhead).
            // We deliberately do NOT `self.hourly_buf.clone()` —
            // cloning the populated buffer would deep-copy the f64
            // values only to drop them on the next `clear()`.
            hourly_buf: (0..self.inner.hvac.num_zones).map(|_| Vec::new()).collect(),
        }
    }
}

/// Build the default conduction-solver slot (Issue #2457).
///
/// Returns a [`FiveR1CSolver`] pre-initialized with the lightweight wall
/// spec (`lightweight_wall_spec()`), so the dispatcher's first call to
/// `step()` succeeds. Without `initialize()`, `FiveR1CSolver::step()`
/// returns `SolverError::InvalidConfig` and the dispatcher would fall
/// back to the physics path — defeating the no-op-swap property called
/// for in the issue.
///
/// The lightweight wall is a representative low-mass construction
/// (wood stud + fiberglass + plasterboard, ASHRAE 140 Case 600FF-style).
/// It is a placeholder: a future commit (per Issue #1896's
/// output-side residual guard) trains a wall-system ONNX surrogate and
/// plugs it in via `HybridThermalModel::set_conduction_solver`.
pub(crate) fn default_conduction_solver() -> Box<dyn HeatConductionSolver> {
    let mut solver = FiveR1CSolver::default();
    // Initialize with a representative wall. Errors here indicate a
    // bug in the wall spec — propagate via a `Box<dyn HeatConductionSolver>`
    // wrapper that returns `InvalidConfig` from `step()`.
    if let Err(e) = solver.initialize(&lightweight_wall_spec()) {
        log::warn!(
            "HybridThermalModel: default conduction solver initialize failed ({}); \
             the dispatcher will fall back to the analytical physics path until \
             the slot is replaced via `set_conduction_solver`.",
            e
        );
    }
    Box::new(solver)
}

/// Build the default ventilation-schedule slot (Issue #2457).
///
/// Returns a [`ConstantVentilation`] of 0.5 ACH — the ASHRAE 140
/// default-infiltration value for the Case 900 / 920 / 940 / 950 / 960
/// reference models (see `WeatherDependentVentilation` doc-comment).
/// A future commit can swap in a weather-aware schedule via
/// `HybridThermalModel::set_ventilation_schedule`.
pub(crate) fn default_ventilation_schedule() -> Box<dyn VentilationSchedule> {
    Box::new(ConstantVentilation::new(0.5))
}

impl HybridThermalModel {
    /// Build a fresh `HybridThermalModel` with the supplied routing policy.
    pub fn new(num_zones: usize, routing: HybridRouting) -> Self {
        Self {
            inner: ThermalModel::new(num_zones),
            routing,
            conduction_solver: default_conduction_solver(),
            ventilation_schedule: default_ventilation_schedule(),
            surrogate_load_calls: 0,
            physics_conduction_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
            // Issue #2921: pre-allocate the surrogate-load scratch buffer to
            // `num_zones` slots so the first `predict_loads_into` call does
            // not need to grow. Empty Vec is fine here — the first call will
            // `clear()` then `extend_from_slice` into a grown Vec; subsequent
            // calls reuse the existing capacity.
            surrogate_load_scratch: Vec::with_capacity(num_zones),
            // Issue #2860: pre-allocate the hourly snapshot buffer's outer
            // Vec to `num_zones` empty inner Vecs. Inner Vec capacity is
            // grown lazily to the requested `steps` on the first
            // `solve_timesteps` call; subsequent calls reuse that capacity
            // via `clear()`.
            hourly_buf: (0..num_zones).map(|_| Vec::new()).collect(),
        }
    }

    /// Build from an ASHRAE 140 case specification with the default policy.
    pub fn from_spec(spec: &crate::validation::ashrae_140_cases::CaseSpec) -> Self {
        Self {
            inner: ThermalModel::from_spec_with_selector(spec, &ThermalSelector::default())
                .expect("default selector must initialize"),
            routing: HybridRouting::default(),
            conduction_solver: default_conduction_solver(),
            ventilation_schedule: default_ventilation_schedule(),
            surrogate_load_calls: 0,
            physics_conduction_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
            // Issue #2921: same zero-alloc rationale as `new`.
            surrogate_load_scratch: Vec::with_capacity(spec.num_zones),
            // Issue #2860: same zero-alloc rationale as `new`.
            hourly_buf: (0..spec.num_zones).map(|_| Vec::new()).collect(),
        }
    }

    /// Build from an ASHRAE 140 case specification with a caller-supplied
    /// routing policy.
    pub fn from_spec_with_routing(
        spec: &crate::validation::ashrae_140_cases::CaseSpec,
        routing: HybridRouting,
    ) -> Self {
        Self {
            inner: ThermalModel::from_spec_with_selector(spec, &ThermalSelector::default())
                .expect("default selector must initialize"),
            routing,
            conduction_solver: default_conduction_solver(),
            ventilation_schedule: default_ventilation_schedule(),
            surrogate_load_calls: 0,
            physics_conduction_calls: 0,
            surrogate_conduction_calls: 0,
            surrogate_ventilation_calls: 0,
            // Issue #2921: same zero-alloc rationale as `new`.
            surrogate_load_scratch: Vec::with_capacity(spec.num_zones),
            // Issue #2860: same zero-alloc rationale as `new`.
            hourly_buf: (0..spec.num_zones).map(|_| Vec::new()).collect(),
        }
    }

    /// Replace the routing policy in place. Counters are preserved.
    pub fn set_routing(&mut self, routing: HybridRouting) {
        self.routing = routing;
    }

    /// Current routing policy.
    pub fn routing(&self) -> HybridRouting {
        self.routing
    }

    /// Swap the conduction solver slot (Issue #2457).
    ///
    /// Replaces the `Box<dyn HeatConductionSolver>` consulted by the
    /// dispatcher when `routing.use_surrogate_conduction` is `true`.
    /// A future ONNX-trained conduction surrogate (per Issue #1896's
    /// output-side residual guard) can be plugged in here without
    /// touching the dispatcher.
    pub fn set_conduction_solver(
        &mut self,
        solver: Box<dyn HeatConductionSolver>,
    ) -> Box<dyn HeatConductionSolver> {
        std::mem::replace(&mut self.conduction_solver, solver)
    }

    /// Swap the ventilation schedule slot (Issue #2457).
    ///
    /// Replaces the `Box<dyn VentilationSchedule>` consulted by the
    /// dispatcher when `routing.use_surrogate_ventilation` is `true`.
    pub fn set_ventilation_schedule(
        &mut self,
        schedule: Box<dyn VentilationSchedule>,
    ) -> Box<dyn VentilationSchedule> {
        std::mem::replace(&mut self.ventilation_schedule, schedule)
    }

    /// Borrow the conduction solver slot (Issue #2457). Read-only.
    pub fn conduction_solver(&self) -> &dyn HeatConductionSolver {
        self.conduction_solver.as_ref()
    }

    /// Borrow the ventilation schedule slot (Issue #2457). Read-only.
    pub fn ventilation_schedule(&self) -> &dyn VentilationSchedule {
        self.ventilation_schedule.as_ref()
    }

    /// Number of times the surrogate load-prediction branch fired in the
    /// most recent (or cumulative) solve. Useful for wiring tests.
    pub fn surrogate_load_calls(&self) -> usize {
        self.surrogate_load_calls
    }

    /// Number of times the physics conduction solver fired in the most
    /// recent (or cumulative) solve. Useful for wiring tests.
    ///
    /// Renamed from `physics_step_calls` in Issue #2457: this counter is
    /// incremented ONLY when the analytical physics conduction path
    /// actually runs. When `use_surrogate_conduction` is `true` the
    /// dispatcher routes to the surrogate slot and this counter stays at
    /// zero.
    pub fn physics_conduction_calls(&self) -> usize {
        self.physics_conduction_calls
    }

    /// Number of times the surrogate conduction branch fired in the most
    /// recent (or cumulative) solve. Useful for wiring tests (Issue #1702).
    pub fn surrogate_conduction_calls(&self) -> usize {
        self.surrogate_conduction_calls
    }

    /// Number of times the surrogate ventilation branch fired in the most
    /// recent (or cumulative) solve. Useful for wiring tests (Issue #1702).
    pub fn surrogate_ventilation_calls(&self) -> usize {
        self.surrogate_ventilation_calls
    }

    /// Reset all routing counters to zero.
    pub fn reset_counters(&mut self) {
        self.surrogate_load_calls = 0;
        self.physics_conduction_calls = 0;
        self.surrogate_conduction_calls = 0;
        self.surrogate_ventilation_calls = 0;
        // Issue #2921: clear (NOT deallocate) the surrogate-load scratch
        // buffer so the next `solve_timesteps` call starts from a clean
        // state but the pre-allocated capacity is preserved — the
        // `predict_loads_into` hot path stays zero-alloc on every solve.
        self.surrogate_load_scratch.clear();
    }

    /// Get the full hourly zone temperature profiles from the last simulation.
    ///
    /// Must be called after `solve_timesteps`. Returns `None` if the
    /// simulation has not been run or if the inner model did not capture
    /// hourly temperatures (e.g. zero-step solve).
    ///
    /// Mirrors [`PhysicsThermalModel::get_hourly_temperatures`] so the
    /// hybrid report (`validation::empirical_hybrid`, Issue #1846) can
    /// compare hybrid temperatures against FLEXLAB measurements on the
    /// same per-timestep grid as the physics model.
    ///
    /// Issue #2860: now reads from the pre-allocated `hourly_buf`
    /// (reused across solves) instead of `self.inner.diagnostics_state
    /// .hourly_temperatures`, which `solve_timesteps` now leaves at
    /// `None` to avoid a per-call `Some(Vec<Vec<f64>>)` allocation.
    /// The returned `Vec<Vec<f64>>` is a clone of the live buffer —
    /// allocation cost is one block per outer Vec plus one block per
    /// non-empty inner Vec, paid only at the call site that reads the
    /// temperatures (typically once per run, not per timestep).
    pub fn get_hourly_temperatures(&self) -> Option<Vec<Vec<f64>>> {
        if self.hourly_buf.is_empty() {
            None
        } else {
            Some(self.hourly_buf.clone())
        }
    }

    /// Returns a structured snapshot of the current dispatch counters,
    /// routing mode, and zone count (Issue #1608).
    pub fn metrics(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            surrogate_load_calls: self.surrogate_load_calls,
            physics_conduction_calls: self.physics_conduction_calls,
            surrogate_conduction_calls: self.surrogate_conduction_calls,
            surrogate_ventilation_calls: self.surrogate_ventilation_calls,
            mode: ThermalModelMode::Hybrid,
            num_zones: self.inner.hvac.num_zones,
            routing: self.routing,
        }
    }
}

impl ThermalModelTrait for HybridThermalModel {
    fn num_zones(&self) -> usize {
        self.inner.hvac.num_zones
    }

    fn get_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    fn set_temperatures(&mut self, temperatures: &[f64]) {
        self.inner.setpoints.temperatures =
            crate::physics::cta::VectorField::new(temperatures.to_vec());
    }

    fn mode(&self) -> ThermalModelMode {
        ThermalModelMode::Hybrid
    }

    fn set_mode(&mut self, _mode: ThermalModelMode) {
        // Hybrid mode is intrinsic to this struct; ignore reassignment.
        // Callers that want a different mode should construct the
        // appropriate concrete model (Physics / Surrogate / Unified).
    }

    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        _use_surrogates: bool,
    ) -> f64 {
        self.reset_counters();

        // The hybrid dispatcher walks each subsystem independently.
        // Each branch is a thin wrapper around the existing
        // `ThermalModel::solve_timesteps` entry point; we only swap the
        // boolean that flips between
        // `SurrogateManager::predict_loads_with_fallback` and the
        // analytical `calc_analytical_loads` path inside
        // `solve_single_step` (see src/sim/thermal_model_iterative.rs:41).
        let use_surrogate_loads = self.routing.use_surrogate_loads;
        let use_surrogate_conduction = self.routing.use_surrogate_conduction;
        let use_surrogate_ventilation = self.routing.use_surrogate_ventilation;
        let use_ood_fallback = self.routing.use_ood_fallback;

        // Issue #1846 — initialize hourly zone temperature storage before
        // the timestep loop. Mirrors the physics-model behaviour in
        // `thermal_model_physics::solver_core::solve_timesteps` so
        // `get_hourly_temperatures()` returns the same shape for hybrid
        // and physics models, enabling apples-to-apples MAE comparison
        // in the empirical_hybrid harness.
        //
        // Issue #2860 — reuse the pre-allocated `hourly_buf` instead of
        // allocating a fresh `Vec<Vec<f64>>` every call. We:
        //   1. Defensively resize the outer Vec to `num_zones` (cheap
        //      no-op when the outer is already sized correctly — the
        //      typical case for the perf-test clones, which construct
        //      and clone `hourly_buf` at `num_zones`).
        //   2. Clear each inner Vec (preserves capacity, drops length).
        //   3. Reserve `steps` capacity if a new solve requests more
        //      steps than the buffer has — first-time cost only;
        //      subsequent calls with the same `steps` are zero-alloc.
        //   4. Set `diagnostics_state.hourly_temperatures = None` so the
        //      legacy accessor path no longer leaks a fresh Vec every
        //      call. `HybridThermalModel::get_hourly_temperatures()` now
        //      reads from `hourly_buf` directly (see the public accessor
        //      below).
        self.hourly_buf
            .resize_with(self.inner.hvac.num_zones, Vec::new);
        for inner in self.hourly_buf.iter_mut() {
            inner.clear();
            if inner.capacity() < steps {
                inner.reserve(steps - inner.capacity());
            }
        }
        self.inner.diagnostics_state.hourly_temperatures = None;

        // Issue #2457: `use_surrogate_conduction` and `use_surrogate_ventilation`
        // now route through the corresponding `Box<dyn Trait>` slot. When
        // `true`, the dispatcher consults the slot (and skips the legacy
        // `step_physics` path for conduction) — replacing the counter-only
        // stub from Issue #1702 that the regression test
        // `hybrid_conduction_flag_routes_through_slot_not_physics` guards
        // against. The slots are initially populated with physics-grade
        // solvers (FiveR1CSolver / ConstantVentilation) so the dispatch is
        // a no-op swap for the default routing; a future commit can plug in
        // a real ONNX-trained conduction surrogate via
        // `HybridThermalModel::set_conduction_solver`.
        //
        // Issue #2507: hoist the total zone envelope area out of the
        // timestep loop. The surrogate-conduction branch converts the
        // per-surface `HeatFlux` [W/m²] returned by the slot into a
        // zone-level energy term [kWh] via `q * A * dt / 3.6e6`; the area
        // is invariant across the run, so we resolve the immutable borrow
        // once here (returns a copied `f64`) to avoid re-borrowing
        // `self.inner` inside the `&mut self` closure body.
        let zone_area_m2 = self.inner.setpoints.zone_area.integrate();
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                // Branch 1: surrogate load prediction (only if policy says so).
                if use_surrogate_loads {
                    // Issue #1892: OOD-aware routing — check bounds before surrogate inference.
                    if use_ood_fallback {
                        let ood_result =
                            surrogates.validate_input_bounds(self.inner.setpoints.temperatures.as_ref());
                        if ood_result.is_ood {
                            // OOD detected — emit warnings and reroute to physics solver.
                            ood_result.log_warnings();
                            log::warn!(
                                "HybridThermalModel[OOD]: timestep {} input vector is out-of-distribution; rerouting to analytical physics solver",
                                t
                            );
                            self.inner.calc_analytical_loads(t, true, 3600.0);
                            // Do NOT increment surrogate_load_calls — this was a physics call.
                        } else {
                            // In-distribution — proceed with surrogate inference.
                            // Issue #2921: zero-alloc `predict_loads_into` writes
                            // the prediction into the pre-allocated
                            // `surrogate_load_scratch` buffer instead of returning
                            // a fresh `Vec<f64>` each step. `predict_loads_into`
                            // never errors (it silently falls back to the 1.2 mock
                            // on ONNX failure, matching `predict_loads` semantics),
                            // so the `Err` arm of the previous match disappears.
                            // The result is installed into `self.inner.setpoints.loads` via
                            // `VectorField::from_slice`, which stores inline in the
                            // SmallVec for ≤ 4 zones (no heap alloc) — covering the
                            // 1-zone and small-multi-zone regimes that drive the
                            // absolute-perf-gate harness.
                            surrogates.predict_loads_into(
                                self.inner.setpoints.temperatures.as_ref(),
                                &mut self.surrogate_load_scratch,
                            );
                            self.inner.setpoints.loads = crate::physics::cta::VectorField::from_slice(
                                &self.surrogate_load_scratch,
                            );
                            self.surrogate_load_calls += 1;
                            trace!(
                                hybrid.surrogate_load_calls = self.surrogate_load_calls,
                                hybrid.timestep = t,
                                "surrogate load branch fired"
                            );
                        }
                    } else {
                        // Standard path: no OOD check, direct surrogate call.
                        // Issue #2921: same zero-alloc `predict_loads_into` swap
                        // as the OOD-enabled branch above. The `Err` arm goes
                        // away — `predict_loads_into` always succeeds (with a
                        // mock fallback on ONNX failure).
                        surrogates.predict_loads_into(
                            self.inner.setpoints.temperatures.as_ref(),
                            &mut self.surrogate_load_scratch,
                        );
                        self.inner.setpoints.loads = crate::physics::cta::VectorField::from_slice(
                            &self.surrogate_load_scratch,
                        );
                        self.surrogate_load_calls += 1;
                        trace!(
                            hybrid.surrogate_load_calls = self.surrogate_load_calls,
                            hybrid.timestep = t,
                            "surrogate load branch fired"
                        );
                    }
                } else {
                    // Branch 2: analytical (physics) load prediction.
                    self.inner.calc_analytical_loads(t, true, 3600.0);
                }

                let hour_of_day = t % 24;
                let daily_cycle =
                    (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                let outdoor_temp = 10.0 + 10.0 * daily_cycle;

                // Issue #2457: Branch 3 — surrogate conduction dispatch.
                //
                // When `use_surrogate_conduction` is `true`, the dispatcher
                // consults `conduction_solver.step(...)` instead of the
                // legacy `self.inner.step_physics(...)` call. This is the
                // wiring Issue #1702 left as a follow-up. The slot initially
                // holds a FiveR1CSolver::default(); an uninitialized solver
                // returns `SolverError::InvalidConfig` from `step()` and we
                // fall back to the analytical path so the energy remains
                // finite. A future ONNX surrogate plugs in via
                // `set_conduction_solver(...)`.
                let mut conduction_used_surrogate = false;
                // Issue #2507: the slot returns a `HeatFlux` [W/m²]
                // (positive = heat flowing into the zone) that MUST be
                // fed into the zone energy balance — not discarded.
                // Captured here and converted to kWh in Branch 5 below.
                let mut surrogate_conduction_flux_wm2: f64 = 0.0;
                if use_surrogate_conduction {
                    let zone_temp = self
                        .inner
                        .setpoints
                        .temperatures
                        .as_ref()
                        .first()
                        .copied()
                        .unwrap_or(20.0);
                    let t_sol_air = outdoor_temp;
                    match self.conduction_solver.step(
                        Time::from_value(3600.0),
                        Temperature::from_value(zone_temp),
                        Temperature::from_value(t_sol_air),
                        HeatTransferCoefficient::from_value(8.0),
                        HeatTransferCoefficient::from_value(25.0),
                    ) {
                        Ok(flux) => {
                            // Issue #2507: capture the returned
                            // `HeatFlux` instead of discarding it. The
                            // value (W/m², positive = into zone) is fed
                            // into the zone energy balance in Branch 5.
                            surrogate_conduction_flux_wm2 = flux.to_value();
                            conduction_used_surrogate = true;
                            self.surrogate_conduction_calls += 1;
                            trace!(
                                hybrid.surrogate_conduction_calls =
                                    self.surrogate_conduction_calls,
                                hybrid.timestep = t,
                                "surrogate conduction branch fired"
                            );
                        }
                        Err(e) => {
                            log::warn!(
                                "HybridThermalModel: surrogate conduction step failed ({}); \
                                 falling back to analytical physics at timestep {}",
                                e,
                                t
                            );
                            // Fall through to Branch 5 below.
                        }
                    }
                }

                // Issue #2457: Branch 4 — surrogate ventilation dispatch.
                //
                // When `use_surrogate_ventilation` is `true`, the dispatcher
                // consults `ventilation_schedule.get_ach(...)` so the
                // schedule is actually exercised (not just a counter bump).
                // Plumbing the returned ACH into the zone h_ve balance is
                // the next step (future PR); for now the slot records the
                // call and `step_physics` below still uses its internal
                // ventilation when conduction stays on physics. When
                // `use_surrogate_conduction` is also `true`, Branch 5 is
                // skipped entirely and the slot is the sole route.
                if use_surrogate_ventilation {
                    let zone_temp = self
                        .inner
                        .setpoints
                        .temperatures
                        .as_ref()
                        .first()
                        .copied()
                        .unwrap_or(20.0);
                    let _ach = self.ventilation_schedule.get_ach(
                        hour_of_day,
                        outdoor_temp,
                        zone_temp,
                        0.0, // wind speed not retained on inner; placeholder
                        0.0, // zone volume not retained on inner; placeholder
                    );
                    self.surrogate_ventilation_calls += 1;
                    trace!(
                        hybrid.surrogate_ventilation_calls =
                            self.surrogate_ventilation_calls,
                        hybrid.timestep = t,
                        "surrogate ventilation branch fired"
                    );
                }

                // Branch 5: physics conduction (5R1C / 9R4C thermal network).
                //
                // The dispatcher picks 5R1C / 9R4C based on the model's
                // construction. The `physics_conduction_calls` counter
                // (renamed from `physics_step_calls` in Issue #2457)
                // increments ONLY when the analytical path actually fires —
                // when `use_surrogate_conduction` rerouted conduction to the
                // slot, Branch 5 is skipped entirely so the counter stays
                // at zero. This is the behaviour the regression test
                // `hybrid_conduction_flag_routes_through_slot_not_physics`
                // asserts (the bug Issue #2457 closes: previously the
                // physics path fired in parallel with the surrogate counter
                // bump, paying the full physics cost plus overhead).
                let energy = if conduction_used_surrogate {
                    // Issue #2507: feed the surrogate-conduction
                    // `HeatFlux` into the zone energy balance. The slot
                    // returns a per-surface flux q [W/m²] (positive =
                    // heat into the zone); convert to the per-timestep
                    // zone energy [kWh] exactly as `step_physics` does
                    // (watts × seconds / 3.6e6):
                    //
                    //   E = q × A × dt / 3.6e6
                    //
                    // where A is the total zone envelope area [m²] and
                    // dt = 3600 s. This replaces the hard-coded `0.0`
                    // placeholder that silently produced wrong annual
                    // energy whenever `use_surrogate_conduction` was
                    // enabled. The sign convention is preserved: a
                    // positive flux (heat gain) is a positive energy
                    // term, matching the conduction heat gain term in
                    // the physics path's zone balance.
                    surrogate_conduction_flux_wm2 * zone_area_m2 * 3600.0 / 3.6e6
                } else {
                    let energy = self.inner.step_physics(t, outdoor_temp, 3600.0);
                    self.physics_conduction_calls += 1;
                    trace!(
                        hybrid.physics_conduction_calls = self.physics_conduction_calls,
                        hybrid.timestep = t,
                        "physics conduction branch fired"
                    );
                    energy
                };

                // Issue #1846 — capture zone temperatures after each timestep
                // so `get_hourly_temperatures()` returns the full per-timestep
                // profile for the empirical_hybrid harness (FLEXLAB MAE report).
                //
                // Issue #2860 — write directly into the pre-allocated
                // `hourly_buf` instead of copying temperatures into a fresh
                // `Vec<f64>` snapshot first. The two borrows are now on
                // distinct `self` paths (`self.inner.setpoints.temperatures`
                // is read immutably via the nested `setpoints` field; `self.hourly_buf`
                // is written mutably), so no borrow conflict forces a snapshot copy.
                // The inner Vecs have capacity ≥ `steps` after warm-up, so the
                // `push` is zero-alloc on the steady-state hot path.
                let temps = self.inner.setpoints.temperatures.as_ref();
                for (zone_idx, &temp) in temps.iter().enumerate() {
                    if let Some(inner) = self.hourly_buf.get_mut(zone_idx) {
                        inner.push(temp);
                    }
                }

                energy
            })
            .sum();

        let total_area = self.inner.setpoints.zone_area.integrate();
        if total_area > 0.0 {
            total_energy_kwh / total_area
        } else {
            0.0
        }
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        self.inner.apply_parameters(params);
    }

    fn zone_area(&self) -> f64 {
        self.inner.setpoints.zone_area.integrate()
    }

    fn heating_setpoint(&self) -> f64 {
        self.inner.setpoints.heating_setpoint
    }

    fn cooling_setpoint(&self) -> f64 {
        self.inner.setpoints.cooling_setpoint
    }

    fn hvac_power_demand(&self, timestep: usize, _outdoor_temp: f64) -> f64 {
        let temps = self.inner.setpoints.temperatures.as_ref();
        if temps.is_empty() {
            return 0.0;
        }
        let t = temps[0];
        let heating_sp = self.inner.setpoints.heating_schedule.value(timestep % 24);
        let cooling_sp = self.inner.setpoints.cooling_schedule.value(timestep % 24);

        if t < heating_sp {
            (heating_sp - t) * 100.0
        } else if t > cooling_sp {
            -(t - cooling_sp) * 100.0
        } else {
            0.0
        }
    }

    fn is_valid(&self) -> bool {
        self.inner.hvac.num_zones > 0 && self.zone_area() > 0.0
    }

    fn get_comfort_metrics(&self) -> Vec<ZoneComfortMetrics> {
        self.inner
            .get_temperatures()
            .iter()
            .map(|&t| compute_pmv_ppd_and_adaptive(t, 0.5, 0.1, 1.0, 0.5))
            .collect()
    }

    fn set_twin_correction(&mut self, correction: &TwinCorrection) {
        self.inner.set_twin_correction(correction);
    }
}
