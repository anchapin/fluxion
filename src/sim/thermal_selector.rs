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

// =============================================================================
// String parsing for the binding layers (Issue #3281 / #3282 / #3283).
//
// The REST `SimulateOptions`, the CLI `--zone-solver` / `--conduction-solver`
// flags, and the Python / Node binding kwargs all carry the solver choice as
// a free-form string. Every surface funnels through the two `parse_*`
// functions below so the accepted vocabulary and the experimental-gate
// wording stay identical across all four layers.
//
// Error style: plain `String` messages (the callers wrap them in the
// layer-appropriate error type — `ApiError::InvalidRequest` → HTTP 400,
// `anyhow!` → non-zero CLI exit, `PyValueError` / NAPI `Error` for the
// bindings).
// =============================================================================

/// Recognised-but-experimental zone-solver identifiers. These have no
/// [`ZoneSolverKind`] variant yet (they need the `fluxion-experimental
/// -zone-solvers` cargo feature, tracked for PR4 of issue #3291), but they
/// are matched *before* the unknown-value error so the rejection message can
/// point at the experimental gate instead of the vocabulary list.
const EXPERIMENTAL_ZONE_SOLVERS: [&str; 2] = ["6r2c", "8r3c"];

/// Parse a user-facing zone-solver string into a [`ZoneSolverKind`].
///
/// Accepted (case-insensitive): `"gauge"`, `"5r1c"`, `"9r4c"`.
///
/// Experimental identifiers `"6r2c"` / `"8r3c"` are always rejected, with a
/// message that names [`experimental_zone_solver_enabled`]'s env var. Even
/// with `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` set they stay rejected until
/// the cargo feature exists (fail-closed: the env var widens no doors that
/// the build cannot back).
///
/// # Errors
/// Returns a human-readable message for unknown or experimental values.
pub fn parse_zone_solver(s: &str) -> Result<ZoneSolverKind, String> {
    let normalized = s.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "gauge" => Ok(ZoneSolverKind::Gauge),
        "5r1c" => Ok(ZoneSolverKind::FiveROneC),
        "9r4c" => Ok(ZoneSolverKind::NineRFourC),
        other if EXPERIMENTAL_ZONE_SOLVERS.contains(&other) => {
            if experimental_zone_solver_enabled() {
                Err(format!(
                    "experimental zone solver '{other}' requires the \
                     `fluxion-experimental-zone-solvers` cargo feature, which is \
                     not part of this build (tracked by issue #3291, PR4)"
                ))
            } else {
                Err(format!(
                    "experimental zone solver '{other}' requires \
                     FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1 to be set (and even then \
                     it stays unavailable until the `fluxion-experimental-zone-solvers` \
                     cargo feature ships; issue #3291)"
                ))
            }
        }
        _ => Err(format!(
            "unknown zone_solver '{s}' (expected one of: gauge, 5r1c, 9r4c)"
        )),
    }
}

/// Parse a user-facing conduction-solver string into a [`ConductionSolverKind`].
///
/// Accepted (case-insensitive): `"default"`, `"ctf"`, `"fd"`.
///
/// # Errors
/// Returns a human-readable message for unknown values.
pub fn parse_conduction_solver(s: &str) -> Result<ConductionSolverKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "default" => Ok(ConductionSolverKind::Default),
        "ctf" => Ok(ConductionSolverKind::Ctf),
        "fd" => Ok(ConductionSolverKind::Fd),
        _ => Err(format!(
            "unknown conduction_solver '{s}' (expected one of: default, ctf, fd)"
        )),
    }
}

/// `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` — the hidden env gate shared by all
/// binding layers (Issue #3282). See [`experimental_zone_solver_enabled`].
static EXPERIMENTAL_ENABLED: OnceLock<bool> = OnceLock::new();

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

    // ---- Issue #3281/#3282/#3283 — shared string parsers -------------------

    #[test]
    fn parse_zone_solver_accepts_known_values_case_insensitive() {
        assert_eq!(parse_zone_solver("gauge"), Ok(ZoneSolverKind::Gauge));
        assert_eq!(parse_zone_solver("5r1c"), Ok(ZoneSolverKind::FiveROneC));
        assert_eq!(parse_zone_solver("9r4c"), Ok(ZoneSolverKind::NineRFourC));
        // Case- and whitespace-insensitive.
        assert_eq!(parse_zone_solver(" Gauge "), Ok(ZoneSolverKind::Gauge));
        assert_eq!(parse_zone_solver("5R1C"), Ok(ZoneSolverKind::FiveROneC));
        assert_eq!(parse_zone_solver("9R4C"), Ok(ZoneSolverKind::NineRFourC));
    }

    #[test]
    fn parse_zone_solver_rejects_unknown_values() {
        let err = parse_zone_solver("fast_solver").unwrap_err();
        assert!(err.contains("unknown zone_solver"), "got: {err}");
        assert!(err.contains("gauge"), "must list the vocabulary: {err}");
    }

    #[test]
    fn parse_zone_solver_rejects_experimental_values() {
        // Both experimental identifiers must be rejected regardless of the
        // env gate state: without the env var the gate message fires; with
        // it, the missing-cargo-feature message fires. This test cannot
        // control which branch `experimental_zone_solver_enabled()` takes
        // (the OnceLock may already be initialised by a sibling test), so
        // assert only the invariant both branches share: Err, and the
        // message names the identifier.
        for value in ["6r2c", "8r3c"] {
            let err = parse_zone_solver(value).unwrap_err();
            assert!(
                err.contains(value),
                "experimental rejection must name '{value}': {err}"
            );
            assert!(
                err.contains("experimental"),
                "rejection must be flagged experimental: {err}"
            );
        }
    }

    #[test]
    fn parse_conduction_solver_accepts_known_values_case_insensitive() {
        assert_eq!(
            parse_conduction_solver("default"),
            Ok(ConductionSolverKind::Default)
        );
        assert_eq!(
            parse_conduction_solver("ctf"),
            Ok(ConductionSolverKind::Ctf)
        );
        assert_eq!(parse_conduction_solver("fd"), Ok(ConductionSolverKind::Fd));
        assert_eq!(
            parse_conduction_solver(" CTF "),
            Ok(ConductionSolverKind::Ctf)
        );
    }

    #[test]
    fn parse_conduction_solver_rejects_unknown_values() {
        let err = parse_conduction_solver("quantum").unwrap_err();
        assert!(err.contains("unknown conduction_solver"), "got: {err}");
        assert!(err.contains("ctf"), "must list the vocabulary: {err}");
    }
}
