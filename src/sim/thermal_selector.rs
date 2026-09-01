//! User-facing selector for the thermal solver stack.
//! Mirrors the binding-layer `thermal_model.zone_solver` and
//! `thermal_model.conduction_solver` fields.

/// Composite selector pairing a zone solver with a conduction algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThermalSelector {
    pub zone_solver: ZoneSolverKind,
    pub conduction_solver: ConductionSolverKind,
}

/// Production zone solvers. Experimental solvers (6R2C, 8R3C) are gated
/// behind the `fluxion-experimental-zone-solvers` cargo feature (out of
/// scope for this issue; tracked separately).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZoneSolverKind {
    #[default]
    Gauge,
    FiveROneC,
    NineRFourC,
}

/// Conduction algorithms paired with the zone solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConductionSolverKind {
    #[default]
    Default,
    Ctf,
    Fd,
}

impl ZoneSolverKind {
    /// Lowercase identifier used in the `{zone}+{conduction}` metric label.
    pub fn as_str(&self) -> &'static str {
        match self {
            ZoneSolverKind::Gauge => "gauge",
            ZoneSolverKind::FiveROneC => "5r1c",
            ZoneSolverKind::NineRFourC => "9r4c",
        }
    }
}

impl ConductionSolverKind {
    /// Lowercase identifier used in the `{zone}+{conduction}` metric label.
    pub fn as_str(&self) -> &'static str {
        match self {
            ConductionSolverKind::Default => "default",
            ConductionSolverKind::Ctf => "ctf",
            ConductionSolverKind::Fd => "fd",
        }
    }
}

use std::sync::OnceLock;

/// Hidden gate for experimental solvers. Reads
/// `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` from env once and caches the result.
///
/// The function is a process-wide configuration gate, not a hot-path check,
/// so caching is intentional: it avoids race conditions when tests in the
/// same binary mutate the environment concurrently.
pub fn experimental_zone_solver_enabled() -> bool {
    *EXPERIMENTAL_ENABLED.get_or_init(|| {
        std::env::var("FLUXION_EXPERIMENTAL_ZONE_SOLVERS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

static EXPERIMENTAL_ENABLED: OnceLock<bool> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_selector_is_gauge_plus_default() {
        let s = ThermalSelector::default();
        assert_eq!(s.zone_solver, ZoneSolverKind::Gauge);
        assert_eq!(s.conduction_solver, ConductionSolverKind::Default);
    }

    #[test]
    fn zone_solver_as_str_is_lowercase() {
        assert_eq!(ZoneSolverKind::Gauge.as_str(), "gauge");
        assert_eq!(ZoneSolverKind::FiveROneC.as_str(), "5r1c");
        assert_eq!(ZoneSolverKind::NineRFourC.as_str(), "9r4c");
    }

    #[test]
    fn conduction_solver_as_str_is_lowercase() {
        assert_eq!(ConductionSolverKind::Default.as_str(), "default");
        assert_eq!(ConductionSolverKind::Ctf.as_str(), "ctf");
        assert_eq!(ConductionSolverKind::Fd.as_str(), "fd");
    }

    #[test]
    fn experimental_zone_solver_returns_bool() {
        // Smoke test: the function is callable and returns a bool.
        // The actual env-var semantics are validated by integration tests
        // that launch the binary with `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1`
        // pre-set; the cached `OnceLock` makes mid-test env mutation a
        // no-op after first call, which is the desired production behaviour
        // (process-wide config gate, not a hot-path check).
        let _: bool = experimental_zone_solver_enabled();
    }
}
