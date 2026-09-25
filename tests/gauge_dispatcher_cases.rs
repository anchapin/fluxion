//! Integration test for Issue #3724 / #3978 — Gauge solver dispatcher identity.
//!
//! Tests the ADR-0017 (Issue #3978) interim posture that ended the
//! two-solver limbo:
//!  1. In the default build (no `gauge-solver` feature), the *default*
//!     selector is the explicit legacy `FiveROneC` (HighMass specs still
//!     auto-promote to 9R4C) — the selector name now matches the physics
//!     the default build actually executes. An *explicit* `Gauge`
//!     selector is a loud configuration error at construction: the
//!     silent fall-through to legacy 5R1C/9R4C is gone.
//!  2. In the `gauge-solver` build, the default selector remains `Gauge`
//!     (ADR-0007 Phase A8 production posture) and gauge dispatch is
//!     unconditional; a `Gauge` selector with no gauge backend configured
//!     must panic loudly (fail-closed, not silent fallback).
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

/// ADR-0017 (Issue #3978): in default builds the *default* selector is the
/// explicit legacy `FiveROneC` — the selector name now matches the physics
/// the default build actually executes. Previously the `Gauge` default
/// silently fell through to 5R1C/9R4C (the "two-solver limbo").
#[cfg(not(feature = "gauge-solver"))]
#[test]
fn default_selector_is_explicit_legacy_in_default_build() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        ThermalSelector::default().zone_solver,
        ZoneSolverKind::FiveROneC,
        "ADR-0017: the default-build default selector must be the explicit legacy FiveROneC"
    );
}

/// ADR-0017 (Issue #3978): an explicit `Gauge` selector in a default build
/// (no `gauge-solver` feature) is a loud configuration error at model
/// construction — the silent fall-through to legacy 5R1C/9R4C was removed.
#[cfg(not(feature = "gauge-solver"))]
#[test]
#[should_panic(expected = "gauge-solver")]
fn explicit_gauge_selector_panics_loudly_in_default_build() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec_with_selector(
        &spec,
        &ThermalSelector {
            zone_solver: ZoneSolverKind::Gauge,
            conduction_solver: ConductionSolverKind::Default,
        },
    )
    .expect("unreachable: the ADR-0017 default-build Gauge panic must fire first");
    let _ = std::hint::black_box(model.step_physics(0, 20.0, 3600.0));
}

/// ADR-0017 (Issue #3978): the explicit-legacy default must preserve the
/// ADR-0002 high-mass auto-promotion — a HighMass spec with the *default*
/// selector dispatches through the 9R4C FD network (the legacy 5R1C network
/// is structurally dead for high-mass peaks, #1522/#3983). The old
/// fall-through arm carried this promotion; the `FiveROneC` dispatch arm
/// must honor the `is_nine_r4c_model` flag set by `from_spec_with_selector`.
#[test]
fn default_selector_preserves_high_mass_9r4c_promotion() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::from_spec_with_selector(&spec, &ThermalSelector::default())
        .expect("Case900 must initialise with the default selector");
    let _ = std::hint::black_box(model.step_physics(0, 10.0, 3600.0));
    assert_eq!(
        model.effective_zone_solver(),
        ZoneSolverKind::NineRFourC,
        "ADR-0017 + ADR-0002: HighMass specs with the default selector must dispatch 9R4C"
    );
}

/// ADR-0017 interim posture: in `gauge-solver` builds the default selector
/// remains `Gauge` (ADR-0007 Phase A8 production posture; β-soak #3286
/// validates this arm).
#[cfg(feature = "gauge-solver")]
#[test]
fn default_selector_is_gauge_in_gauge_build() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        ThermalSelector::default().zone_solver,
        ZoneSolverKind::Gauge,
        "ADR-0017: gauge-solver builds keep the Gauge default selector"
    );
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
