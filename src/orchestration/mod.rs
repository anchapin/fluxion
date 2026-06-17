//! Orchestration layer: formal decision catalogue and tracing hooks for the TDQS
//! harness (Issue #708 / PR #771).
//!
//! This module exposes [`decision_types::OrchestrationDecisionKind`] and
//! [`decision_types::OrchestrationDecision`] for use by both the engine (tracing
//! spans) and the benchmark harness (`benches/orchestration_decisions/`).

pub mod decision_types;

pub use decision_types::{OrchestrationDecision, OrchestrationDecisionKind};
