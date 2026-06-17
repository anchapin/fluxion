//! Formal orchestration decision types for the TDQS harness (Issue #708 / PR #771).
//!
//! Each [`OrchestrationDecisionKind`] variant corresponds to one category of implicit
//! engine decision that the TDQS harness labels and scores.  The tracing spans emitted
//! at each decision site use the same `decision_type` key so the recorder in
//! `benches/orchestration_decisions/decision_recorder.rs` can correlate spans to labels.
//!
//! # Decision sites
//! | Variant | Tracing site |
//! |---------|-------------|
//! | `SolverSelection` | `src/physics/method_selector.rs::ThermalMethodSelector::select_method` |
//! | `AdaptiveTimestep` | `src/sim/adaptive_timestep.rs::TimestepMode::get_timestep` |
//! | `SurrogateRouting` | `src/ai/surrogate.rs` (stub — wired when batch-oracle lands) |
//! | `ConstraintWarning` | `src/sim/thermal_model_core.rs::ThermalModel::new_with_validation` |
//! | `HvacHorizon` | `src/sim/thermal_model_core.rs::ThermalModel::new_with_validation` |

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Category of an engine orchestration decision.
///
/// Used as the `decision_type` field in every `tracing::info!` span emitted at a
/// decision site.  The TDQS harness reads these fields from the structured log stream
/// and matches them against the 195 ASHRAE 140 ground-truth labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationDecisionKind {
    /// Selection of thermal solver (5R1C, CTF, or FD) for a wall assembly.
    ///
    /// Ground-truth: for ASHRAE 140 900-series the correct choice is FD (Issue #726).
    /// Current engine defaults to CTF for τ ≥ 2 h, causing TDQS = 0.667 for this slot.
    SolverSelection,

    /// Adaptive-timestep trigger: whether to switch from the default 1-hour step to a
    /// sub-hourly step (e.g. 6 min) based on the building's thermal time constant.
    AdaptiveTimestep,

    /// Routing between the physics engine and a surrogate model.
    ///
    /// Stub — no surrogate routing exists yet; hook will be wired in batch_oracle once
    /// the ML & Surrogate Modeling Engineer merges the ONNX inference path.
    SurrogateRouting,

    /// Pre-simulation constraint / parameter validation decision.
    ///
    /// Emitted at `ThermalModel::new_with_validation` once all conductance and setpoint
    /// checks have run.  `chosen` is `"passed"` or the name of the first failing field.
    ConstraintWarning,

    /// HVAC prediction-horizon selection.
    ///
    /// Currently fixed at 24 h; the `chosen` field will reflect "24h_fixed" until an
    /// adaptive horizon is implemented.
    HvacHorizon,
}

impl OrchestrationDecisionKind {
    /// Canonical kebab-case string used as the `decision_type` tracing field value.
    ///
    /// Must match the label strings in the TDQS harness's `decision_recorder.rs`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SolverSelection => "solver_selection",
            Self::AdaptiveTimestep => "adaptive_timestep",
            Self::SurrogateRouting => "surrogate_routing",
            Self::ConstraintWarning => "constraint_warning",
            Self::HvacHorizon => "hvac_horizon",
        }
    }
}

impl std::fmt::Display for OrchestrationDecisionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single orchestration decision record emitted by the engine.
///
/// Created at each decision site and used by the TDQS harness to catalogue engine
/// decisions and compute the Traceable Decision Quality Score.
///
/// # Example
/// ```rust
/// use fluxion::orchestration::decision_types::{OrchestrationDecision, OrchestrationDecisionKind};
/// use serde_json::json;
///
/// let d = OrchestrationDecision::new(
///     OrchestrationDecisionKind::SolverSelection,
///     "ctf",
///     json!({ "tau_hours": 3.5, "threshold_hours": 2.0 }),
/// );
/// assert_eq!(d.kind.as_str(), "solver_selection");
/// assert_eq!(d.chosen, "ctf");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationDecision {
    /// Which category of decision this is.
    pub kind: OrchestrationDecisionKind,
    /// The option the engine chose (e.g. `"ctf"`, `"adaptive_6min"`, `"physics"`).
    pub chosen: String,
    /// Relevant feature values used to make the decision (τ, density, threshold, …).
    pub features: Value,
}

impl OrchestrationDecision {
    /// Construct a decision record with explicit feature data.
    pub fn new(
        kind: OrchestrationDecisionKind,
        chosen: impl Into<String>,
        features: Value,
    ) -> Self {
        Self {
            kind,
            chosen: chosen.into(),
            features,
        }
    }

    /// Convenience: construct with an empty feature map.
    pub fn simple(kind: OrchestrationDecisionKind, chosen: impl Into<String>) -> Self {
        Self::new(kind, chosen, Value::Object(serde_json::Map::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_kind_as_str() {
        assert_eq!(
            OrchestrationDecisionKind::SolverSelection.as_str(),
            "solver_selection"
        );
        assert_eq!(
            OrchestrationDecisionKind::AdaptiveTimestep.as_str(),
            "adaptive_timestep"
        );
        assert_eq!(
            OrchestrationDecisionKind::SurrogateRouting.as_str(),
            "surrogate_routing"
        );
        assert_eq!(
            OrchestrationDecisionKind::ConstraintWarning.as_str(),
            "constraint_warning"
        );
        assert_eq!(
            OrchestrationDecisionKind::HvacHorizon.as_str(),
            "hvac_horizon"
        );
    }

    #[test]
    fn test_kind_display() {
        assert_eq!(
            format!("{}", OrchestrationDecisionKind::SolverSelection),
            "solver_selection"
        );
    }

    #[test]
    fn test_decision_new() {
        let d = OrchestrationDecision::new(
            OrchestrationDecisionKind::SolverSelection,
            "ctf",
            json!({ "tau_hours": 3.5 }),
        );
        assert_eq!(d.kind, OrchestrationDecisionKind::SolverSelection);
        assert_eq!(d.chosen, "ctf");
        assert_eq!(d.features["tau_hours"], 3.5);
    }

    #[test]
    fn test_decision_simple() {
        let d = OrchestrationDecision::simple(OrchestrationDecisionKind::HvacHorizon, "24h_fixed");
        assert!(d.features.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_kind_serde_roundtrip() {
        let kind = OrchestrationDecisionKind::AdaptiveTimestep;
        let json = serde_json::to_string(&kind).unwrap();
        let back: OrchestrationDecisionKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);
    }

    #[test]
    fn test_decision_serde_roundtrip() {
        let d = OrchestrationDecision::new(
            OrchestrationDecisionKind::ConstraintWarning,
            "passed",
            json!({ "h_tr_em": 0.4, "hvac_setpoint": 20.0 }),
        );
        let json = serde_json::to_string(&d).unwrap();
        let back: OrchestrationDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind, OrchestrationDecisionKind::ConstraintWarning);
        assert_eq!(back.chosen, "passed");
    }
}
