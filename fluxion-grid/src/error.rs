use thiserror::Error;

#[derive(Debug, Error)]
pub enum GridModelError {
    #[error("voltage {voltage:.3} pu is outside valid range [0.5, 1.5]")]
    VoltageOutOfRange { voltage: f64 },

    #[error("coupler thermal mass is zero — cannot apply voltage adjustment")]
    ZeroThermalMass,

    #[error("negative COP adjustment factor {factor:.4} would increase COP")]
    NegativeAdjustment { factor: f64 },
}

/// Errors returned by the [`PowerFlowSolver`](crate::power_flow::PowerFlowSolver).
///
/// These distinguish the failure modes of Newton-Raphson iteration so callers
/// can decide whether to retry, reconfigure, or abort.
#[derive(Debug, Error)]
pub enum GridSolveError {
    /// The solver did not reach the requested tolerance within `max_iterations`.
    #[error(
        "power flow did not converge in {max_iterations} iterations (residual {residual:.3e} pu)"
    )]
    NonConvergence {
        /// Iteration budget that was exhausted.
        max_iterations: u32,
        /// Infinity-norm of the power mismatch at the last iteration.
        residual: f64,
    },

    /// The Jacobian became singular (non-invertible) during iteration, so the
    /// linear update step could not be solved. Usually indicates an ill-conditioned
    /// or islanded network.
    #[error("singular Jacobian matrix encountered during Newton-Raphson iteration {iteration}")]
    SingularJacobian {
        /// One-based iteration index when the singularity occurred.
        iteration: u32,
    },

    /// The network has no slack (swing) bus, which is required to anchor the
    /// voltage-angle reference.
    #[error("no slack bus defined in the network — exactly one slack bus is required")]
    NoSlackBus,

    /// A transmission line references a bus id that is not present in the bus map.
    #[error("transmission line references unknown bus (from={from}, to={to})")]
    UnknownBus {
        /// `from` bus identifier (as a string for diagnostics).
        from: String,
        /// `to` bus identifier.
        to: String,
    },

    /// The network is too small to run a meaningful power flow.
    #[error("network must have at least 2 buses, found {0}")]
    TooFewBuses(usize),

    /// A computed voltage became non-finite (NaN or Inf), indicating divergence.
    #[error("voltage magnitude became non-finite ({voltage}) during iteration {iteration}")]
    NonFiniteVoltage {
        /// The offending voltage value.
        voltage: f64,
        /// One-based iteration index.
        iteration: u32,
    },

    /// A building in the HVAC state list has no corresponding bus in the grid.
    #[error("building {building_id} has no bus in the grid mapping — skipping")]
    MissingBuildingBus {
        /// The UUID of the building that was not found.
        building_id: uuid::Uuid,
    },
}
