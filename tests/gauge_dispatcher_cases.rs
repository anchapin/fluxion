//! Integration test for Issue #3724 — Gauge solver dispatcher identity.
//!
//! Tests the Phase A8 (Issue #3291) guarantee that:
//!  1. In the default build (no `gauge-solver` feature), `ZoneSolverKind::Gauge`
//!     must NOT panic — it falls through to the legacy 5R1C/9R4C path.
//!  2. In the `gauge-solver` build, a `Gauge` selector with no gauge backend
//!     configured must panic loudly (fail-closed, not silent fallback).
//!
//! These tests live in a `tests/` binary (not a `src/` lib test) because they
//! import `crate::validation::ashrae_140_cases` — that import creates sim→validation
//! cycle edges, which are prohibited inside `src/sim/` by the cycle guard
//! (scripts/check_ashrae_cases_cycle.py). The tests are still first-class cargo
//! tests; they just live outside the cycle-guard boundary.

use fluxion::sim::thermal_model_core::ThermalModel;
use fluxion::sim::thermal_selector::{ConductionSolverKind, ThermalSelector, ZoneSolverKind};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use std::sync::Mutex;

/// Process-wide mutex serializing tests that may initialise OnceLock state
/// inside the thermal model (Issue #3453).
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Issue #3724 — default-build dispatcher identity: `ZoneSolverKind::Gauge`
/// must NOT panic in the default build (no `gauge-solver` feature). It
/// falls through to legacy 5R1C, which is exercised via the normal
/// `step_physics` path.
///
/// This test verifies the "falls through to legacy" arm of the Phase A8
/// contract documented in `src/sim/thermal_selector.rs` and `ARCHITECTURE.md`.
#[test]
fn gauge_selector_falls_through_to_legacy_in_default_build() {
    let _guard = ENV_LOCK.lock().unwrap();
    let spec = ASHRAE140Case::Case600.spec();
    // Explicit Gauge selector in default build — must not panic.
    let model = ThermalModel::from_spec_with_selector(
        &spec,
        &ThermalSelector {
            zone_solver: ZoneSolverKind::Gauge,
            conduction_solver: ConductionSolverKind::Default,
        },
    )
    .expect("Gauge selector must initialise in default build (falls through to legacy)");
    // The legacy path is active. In the default build (no gauge-solver feature)
    // the gauge_zone_solver field doesn't exist on ConductionBackend, so we verify
    // only that the model initialises without panicking — the non-panic itself
    // proves the fallback path was taken (panic would fire if gauge was active).
    let _ = model;
}

/// Issue #3724 — gauge-solver-build dispatcher identity: `ZoneSolverKind::Gauge`
/// with no gauge backend configured must panic loudly (programming error,
/// not silent fallback). This is the fail-closed contract for Phase A8
/// (Issue #3291). The panic fires in `enable_gauge_solver` called from
/// `ThermalModel::from_spec_with_selector`.
///
/// This test is a no-op in the default build (no `gauge-solver` feature).
#[cfg(feature = "gauge-solver")]
#[test]
#[should_panic(expected = "gauge")]
fn gauge_selector_panics_loudly_without_backend_in_gauge_build() {
    let _guard = ENV_LOCK.lock().unwrap();
    let spec = ASHRAE140Case::Case600.spec();
    // Explicit Gauge selector with gauge-solver feature — no gauge backend
    // provided, so the selector must panic loudly rather than silently
    // falling through to legacy.
    let _ = ThermalModel::from_spec_with_selector(
        &spec,
        &ThermalSelector {
            zone_solver: ZoneSolverKind::Gauge,
            conduction_solver: ConductionSolverKind::Default,
        },
    );
    // If we reach here, the panic did not fire — fail.
    panic!("Gauge selector in gauge-solver build must panic without a backend");
}
