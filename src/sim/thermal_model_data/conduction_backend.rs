//! Conduction backend state — all concrete solver state for the thermal model.
//!
//! Extracted from `ThermalModelData` (Issue #2767) so swap-point traits can
//! reason about a single boxed backend instead of a struct that owns every
//! concrete solver simultaneously. The custom `Clone` copies only
//! CTF/coefficients state (matching the pre-refactor `ThermalModelData::clone`
//! behaviour) — the heavy `Vec<ImplicitFDSolver>`, `Vec<MultiNodeSolver>`, and
//! `SolverManager` are dropped and re-initialised by `prepare_solvers` on the
//! first timestep after clone.
//!
//! ## Gauge-slot clone semantics (Issue #3729)
//!
//! The cfg-gated `gauge_zone_solver` / `gauge_multi_zone_solver` slots
//! follow the slot-reset contract documented for
//! [`HybridThermalModel::conduction_solver`](crate::sim::thermal_model::HybridThermalModel)
//! in `ARCHITECTURE.md` §"Clone semantics & BatchOracle parallelism contract"
//! (Issue #2539). Per the per-zone hand-rolled `Clone` impls in
//! [`crate::physics::gauge_zone_solver`]:
//!
//! - **Topology preserved**: `zone_id`, `floor_area`, `zone_volume`, `C_air`,
//!   `num_surfaces`, surface metadata (`area_m2`, `surface_type`, azimuth,
//!   tilt, `wall_spec`), couplings, inter-zone conductance. Two clones
//!   describe identical buildings; only their runtime states diverge.
//! - **Runtime state reset**: `T_air` resets to the `new_with_id` default
//!   (20.0), `initialized` resets to `false`. Per-surface `GaugeSolver`
//!   state is reset transitively by `SurfaceGaugeSolver::clone`, which
//!   re-initializes the slot from `wall_spec` when available (the common
//!   `add_opaque_surface` path).
//! - **Candidates are independent**: mutating one clone's `T_air` or
//!   stepping one clone does not perturb the other's. Pinned by
//!   `tests/all_tests/gauge_conduction_backend_clone.rs` (end-to-end with
//!   `BatchOracle`) and by the in-module unit tests in
//!   `src/physics/gauge_zone_solver.rs`.
//!
//! The `BatchOracle::evaluate_population` hot loop relies on this contract:
//! it clones `base_model` once per candidate and then solves each clone
//! independently via the per-timestep dispatch (which routes through the
//! gauge slot when `ThermalSelector::default()` is selected). Without the
//! slot reset, a mid-solve `T_air` from the parent would silently
//! contaminate every freshly-cloned candidate.

use super::{
    CTFCoefficients, CTFSolver, CtfZoneCouplingSolver, ImplicitFDSolver, MultiNodeSolver,
    SolverManager,
};
#[cfg(feature = "gauge-solver")]
use super::{GaugeZoneSolver, MultiZoneGaugeSolver};

pub struct ConductionBackend {
    // --- CTF (Conduction Transfer Function) ---
    pub ctf_coefficients: Option<CTFCoefficients>,
    pub ctf_solvers: Vec<CTFSolver>,
    pub ctf_enabled: bool,
    pub ctf_timestep: f64,
    pub ctf_zone_coupling_solver: Option<CtfZoneCouplingSolver>,
    pub ctf_primary: bool,
    // --- FD (Finite Difference) ---
    pub fd_solvers: Vec<ImplicitFDSolver>,
    pub fd_enabled: bool,
    pub fd_timestep: f64,
    // --- Multi-node (9R4C) ---
    pub multi_node_solvers: Vec<MultiNodeSolver>,
    // --- Unified solver manager ---
    pub solver_manager: Option<SolverManager>,
    // --- Gauge-zone solver (experimental, feature-gated, always None per #2686) ---
    #[cfg(feature = "gauge-solver")]
    pub gauge_zone_solver: Option<GaugeZoneSolver>,
    // --- Gauge multi-zone solver (experimental, feature-gated, added #3273) ---
    #[cfg(feature = "gauge-solver")]
    pub gauge_multi_zone_solver: Option<MultiZoneGaugeSolver>,
}

impl Clone for ConductionBackend {
    fn clone(&self) -> Self {
        Self {
            ctf_coefficients: self.ctf_coefficients.clone(),
            ctf_solvers: self.ctf_solvers.clone(),
            ctf_enabled: self.ctf_enabled,
            ctf_timestep: self.ctf_timestep,
            ctf_zone_coupling_solver: self.ctf_zone_coupling_solver.clone(),
            ctf_primary: self.ctf_primary,
            // Heavy Vecs — dropped on clone (re-initialised by prepare_solvers).
            fd_solvers: Vec::new(),
            fd_enabled: self.fd_enabled,
            fd_timestep: self.fd_timestep,
            multi_node_solvers: Vec::new(),
            solver_manager: None,
            #[cfg(feature = "gauge-solver")]
            gauge_zone_solver: self.gauge_zone_solver.clone(),
            #[cfg(feature = "gauge-solver")]
            gauge_multi_zone_solver: self.gauge_multi_zone_solver.clone(),
        }
    }
}

impl Default for ConductionBackend {
    fn default() -> Self {
        Self {
            ctf_coefficients: None,
            ctf_solvers: Vec::new(),
            ctf_enabled: false,
            ctf_timestep: 3600.0,
            ctf_zone_coupling_solver: None,
            ctf_primary: false,
            fd_solvers: Vec::new(),
            fd_enabled: false,
            fd_timestep: 3600.0,
            multi_node_solvers: Vec::new(),
            solver_manager: None,
            #[cfg(feature = "gauge-solver")]
            gauge_zone_solver: None,
            #[cfg(feature = "gauge-solver")]
            gauge_multi_zone_solver: None,
        }
    }
}
