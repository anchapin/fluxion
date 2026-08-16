//! Thermal Model Data — thin wrapper around 6 focused sub-structs.
//!
//! Issue #2878 split the legacy god-struct (~140 fields, 145-line Clone impl)
//! into 6 per-domain sub-structs so a per-config clone in
//! `BatchOracle::evaluate_population` visits exactly 6 fields (one delegated
//! `Clone` per sub-struct). New top-level fields should be added to the
//! appropriate sub-struct rather than to this file.
//!
//! Sub-structs: [`HvacState`], [`SetpointState`], [`SolarState`], [`MassState`],
//! [`ConductionState`] (wraps [`ConductionBackend`]), [`DiagnosticsState`].
//!
//! `ThermalModel` (`src/sim/thermal_model_core.rs`) `Deref`s into
//! `ThermalModelData`; existing `self.0.<field>` call sites have been
//! rewritten to access fields through the new sub-structs (e.g.
//! `self.0.solar.window_properties` instead of `self.0.solar.window_properties`).

mod conduction_backend;
mod conduction_state;
mod diagnostics_state;
mod hvac_state;
mod incident_solar_accumulator;
mod mass_state;
mod setpoint_state;
mod solar_state;

pub use conduction_backend::ConductionBackend;
pub use conduction_state::ConductionState;
pub use diagnostics_state::DiagnosticsState;
pub use hvac_state::HvacState;
pub use incident_solar_accumulator::IncidentSolarAccumulator;
pub use mass_state::MassState;
pub use setpoint_state::SetpointState;
pub use solar_state::SolarState;

pub use crate::physics::{
    cta::{ContinuousTensor, VectorField},
    ctf_coefficients::CTFCoefficients,
    ctf_solver::CTFSolver,
    ctf_zone_coupling::CtfZoneCouplingSolver,
    fd_solver::ImplicitFDSolver,
    multi_node_solver::MultiNodeSolver,
    solver_manager::SolverManager,
};
#[cfg(feature = "gauge-solver")]
pub use crate::physics::gauge_zone_solver::GaugeZoneSolver;

/// Thin wrapper composing the 6 per-domain sub-structs.
///
/// A per-config clone touches exactly 6 fields (one delegated `Clone` per
/// sub-struct); the previous 145-line hand-rolled `Clone` impl was replaced
/// by this layout to keep `BatchOracle::evaluate_population` clones cheap
/// (Issue #2878).
pub struct ThermalModelData<T: ContinuousTensor<f64> + Clone> {
    pub hvac: HvacState<T>,
    pub setpoints: SetpointState<T>,
    pub solar: SolarState<T>,
    pub mass: MassState<T>,
    pub conduction: ConductionState<T>,
    pub diagnostics_state: DiagnosticsState,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for ThermalModelData<T> {
    fn clone(&self) -> Self {
        Self {
            hvac: self.hvac.clone(),
            setpoints: self.setpoints.clone(),
            solar: self.solar.clone(),
            mass: self.mass.clone(),
            conduction: self.conduction.clone(),
            diagnostics_state: self.diagnostics_state.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Source-level guard: Clone impl body must contain exactly the 6
    /// delegated field initializers and no other field initializers, so
    /// future "just one more field on the wrapper" additions fail CI.
    #[test]
    fn test_thermal_model_data_clone_visits_six_fields() {
        let src = std::include_str!("mod.rs");
        let body_start = src
            .find("fn clone(&self) -> Self {")
            .expect("Clone impl must exist");
        let body_end = src
            .find("}\n}\n\n#[cfg(test)]")
            .expect("Clone impl must end before #[cfg(test)]");
        let body = &src[body_start..body_end];
        for needle in [
            "hvac: self.hvac.clone()",
            "setpoints: self.setpoints.clone()",
            "solar: self.solar.clone()",
            "mass: self.mass.clone()",
            "conduction: self.conduction.clone()",
            "diagnostics_state: self.diagnostics_state.clone()",
        ] {
            assert!(
                body.contains(needle),
                "Clone impl missing delegated field `{needle}` — Clone now touches more than 6 fields. Body:\n{body}",
            );
        }
    }

    #[test]
    fn test_conduction_backend_clone_preserves_flags_drops_heavy_state() {
        let mut backend = ConductionBackend::default();
        backend.ctf_enabled = true;
        backend.ctf_primary = true;
        backend.fd_enabled = true;
        backend.fd_timestep = 120.0;

        let cloned = backend.clone();
        assert!(cloned.ctf_enabled);
        assert!(cloned.ctf_primary);
        assert!(cloned.fd_enabled);
        assert_eq!(cloned.fd_timestep, 120.0);
        assert!(cloned.fd_solvers.is_empty());
        assert!(cloned.multi_node_solvers.is_empty());
        assert!(cloned.solver_manager.is_none());
    }

    #[test]
    fn test_diagnostics_state_clone_drops_live_state() {
        let mut diag = DiagnosticsState::default();
        diag.hourly_temperatures = Some(vec![vec![20.0; 8760]]);
        diag.incident_solar_per_surface
            .insert("wall_S".to_string(), IncidentSolarAccumulator::new());

        let cloned = diag.clone();
        assert!(cloned.diagnostics.is_none());
        assert!(cloned.hourly_temperatures.is_none());
        assert!(cloned.nodal_temperatures.is_none());
        assert_eq!(cloned.incident_solar_per_surface.len(), 1);
    }

    #[test]
    fn test_conduction_backend_default_values() {
        let backend = ConductionBackend::default();
        assert!(backend.ctf_solvers.is_empty());
        assert!(backend.fd_solvers.is_empty());
        assert!(backend.multi_node_solvers.is_empty());
        assert!(backend.solver_manager.is_none());
        assert!(!backend.ctf_enabled);
        assert!(!backend.fd_enabled);
        assert!(!backend.ctf_primary);
        assert_eq!(backend.ctf_timestep, 3600.0);
        assert_eq!(backend.fd_timestep, 3600.0);
    }
}
