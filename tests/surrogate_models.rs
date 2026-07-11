//! Isolation tests for HybridThermalModel per-component routing (Issue #1431).
//!
//! `tests/surrogate_models/test_hybrid_mode_dispatch.rs` (the nested module
//! `hybrid_mode_dispatch` declared below) is the canonical harness for
//! verifying that `ThermalModelMode::Hybrid` actually dispatches some
//! subsystems to the surrogate while keeping others on physics. Each
//! integration-test file at the top of `tests/` becomes its own test
//! binary, so we use `#[path]` to pull in the nested file without
//! breaking the directory layout that `tests/surrogate_models/golden/`
//! and `tests/surrogate_models/registry.json` rely on.

#[path = "surrogate_models/test_hybrid_mode_dispatch.rs"]
mod hybrid_mode_dispatch;
