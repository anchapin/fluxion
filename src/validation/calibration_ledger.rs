//! Calibration-factor audit ledger for ASHRAE 140 blind validation (Issue #2516).
//!
//! Goal 3 phases A/B/C of the v1.3 blind-validation plan require a machine-readable
//! audit trail of every calibration factor, multiplier, tolerance adjustment, and
//! hardcoded reference value present in the validation pipeline. Phase E
//! SUSTAIN-01..02 depend on a per-case history of which multipliers/tolerances are
//! active in any given build so that blind runs can be reproduced and audited.
//!
//! This module is the single source of truth for that trail. Every calibration
//! factor in `case_195_calibration.rs` and `adaptive_calibration.rs` is enumerated
//! in the static [`LEDGER`] table. A drift gate ([`calibration_ledger_is_complete`])
//! enforces that no factor is added without a corresponding ledger entry.
//!
//! # Drift-gate convention
//!
//! Every site that defines or uses a calibration factor **must** carry a trailing
//! `// LEDGER: <ID>` comment (where `<ID>` is one of the [`CAL_*` constants](LEDGER)).
//! The drift test scans `case_195_calibration.rs` and `adaptive_calibration.rs` for
//! these markers and asserts each `<ID>` resolves via [`lookup`]. If a contributor
//! adds a new calibration factor and marks it `// LEDGER: NEW_ID` but forgets to add
//! a `CalibrationFactor` entry to [`LEDGER`], the test fails.
//!
//! When removing a factor, delete its `// LEDGER:` marker and its ledger entry in
//! the same change — the removed entry's `removal_issue` / `removed_in_version`
//! fields record the audit provenance (or the entry is simply dropped if it no
//! longer affects any code path).

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

/// A single calibration factor recorded in the audit ledger.
///
/// A "calibration factor" is any empirical constant that biases validation
/// results: a hardcoded default material property, a multiplier applied to a
/// gain term, a convergence tolerance, a trigger threshold, or a reference
/// value. Purely physical bounds (e.g. `clamp(0.0, 2.0)` safety limits) are
/// **not** calibration factors and are intentionally excluded.
///
/// Owned `String` fields (rather than `&'static str`) so the struct is fully
/// `Serialize`/`Deserialize` and can round-trip through JSON for the blind-mode
/// audit stream consumed by Phase E SUSTAIN-01..02. The static [`LEDGER`] table
/// is built once via `once_cell` and lends `&'static` references.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationFactor {
    /// Stable identifier, also referenced at the call site via `// LEDGER: <id>`.
    pub id: String,
    /// Human-readable location: `file::symbol` where the factor is defined/used.
    pub location: String,
    /// The numeric value of the factor.
    pub value: f64,
    /// What the value is based on (physical reference, empirical fit, etc.).
    pub reference: String,
    /// GitHub issue number slated to remove this factor, if any (`None` = active).
    pub removal_issue: Option<u32>,
    /// Version in which this factor is/was removed, if any (`None` = active).
    pub removed_in_version: Option<String>,
}

impl CalibrationFactor {
    /// Whether this factor is still active in the current build.
    pub fn is_active(&self) -> bool {
        self.removal_issue.is_none() && self.removed_in_version.is_none()
    }
}

// ---------------------------------------------------------------------------
// Calibration-factor ID constants. Reference these at call sites so that a
// typo in a `// LEDGER:` marker surfaces as a compile-time-adjacent drift
// failure rather than a silent mismatch.
// ---------------------------------------------------------------------------

// -- Case 195 calibration defaults (empirical material properties) ----------
pub const CAL_CASE195_THERMAL_CONDUCTIVITY: &str = "CAL_CASE195_THERMAL_CONDUCTIVITY";
pub const CAL_CASE195_SPECIFIC_HEAT: &str = "CAL_CASE195_SPECIFIC_HEAT";
pub const CAL_CASE195_DENSITY: &str = "CAL_CASE195_DENSITY";
pub const CAL_CASE195_INFILTRATION_RATE: &str = "CAL_CASE195_INFILTRATION_RATE";

// -- Adaptive calibration defaults (CalibrationState) ----------------------
pub const CAL_ADAPTIVE_THERMAL_CONDUCTIVITY: &str = "CAL_ADAPTIVE_THERMAL_CONDUCTIVITY";
pub const CAL_ADAPTIVE_SPECIFIC_HEAT: &str = "CAL_ADAPTIVE_SPECIFIC_HEAT";
pub const CAL_ADAPTIVE_DENSITY: &str = "CAL_ADAPTIVE_DENSITY";
pub const CAL_ADAPTIVE_INFILTRATION_RATE: &str = "CAL_ADAPTIVE_INFILTRATION_RATE";
pub const CAL_ADAPTIVE_INTERNAL_GAIN_MULT: &str = "CAL_ADAPTIVE_INTERNAL_GAIN_MULT";
pub const CAL_ADAPTIVE_SOLAR_GAIN_MULT: &str = "CAL_ADAPTIVE_SOLAR_GAIN_MULT";

// -- Convergence / learning parameters -------------------------------------
pub const CAL_ADAPTIVE_MAX_ITERATIONS: &str = "CAL_ADAPTIVE_MAX_ITERATIONS";
pub const CAL_ADAPTIVE_CONVERGENCE_TOLERANCE: &str = "CAL_ADAPTIVE_CONVERGENCE_TOLERANCE";
pub const CAL_ADAPTIVE_LEARNING_RATE: &str = "CAL_ADAPTIVE_LEARNING_RATE";

// -- Trigger thresholds ----------------------------------------------------
pub const CAL_ADAPTIVE_TEMP_ANOMALY_THRESHOLD: &str = "CAL_ADAPTIVE_TEMP_ANOMALY_THRESHOLD";
pub const CAL_ADAPTIVE_BIAS_CHANGE_THRESHOLD: &str = "CAL_ADAPTIVE_BIAS_CHANGE_THRESHOLD";
pub const CAL_ADAPTIVE_OCCUPANCY_SHIFT_THRESHOLD: &str = "CAL_ADAPTIVE_OCCUPANCY_SHIFT_THRESHOLD";

// -- Bias-classification thresholds ----------------------------------------
pub const CAL_ADAPTIVE_UNIVERSAL_BIAS_RATIO: &str = "CAL_ADAPTIVE_UNIVERSAL_BIAS_RATIO";
pub const CAL_ADAPTIVE_SEASONAL_CORR_THRESHOLD: &str = "CAL_ADAPTIVE_SEASONAL_CORR_THRESHOLD";
pub const CAL_ADAPTIVE_MIXED_BIAS_RATIO: &str = "CAL_ADAPTIVE_MIXED_BIAS_RATIO";
pub const CAL_ADAPTIVE_MIXED_CORR_THRESHOLD: &str = "CAL_ADAPTIVE_MIXED_CORR_THRESHOLD";

// -- Parameter-adjustment weights (UniversalBias) --------------------------
pub const CAL_ADAPTIVE_W_UNIVERSAL_TC: &str = "CAL_ADAPTIVE_W_UNIVERSAL_TC";
pub const CAL_ADAPTIVE_W_UNIVERSAL_INFIL: &str = "CAL_ADAPTIVE_W_UNIVERSAL_INFIL";
pub const CAL_ADAPTIVE_W_UNIVERSAL_IGAIN: &str = "CAL_ADAPTIVE_W_UNIVERSAL_IGAIN";

// -- Parameter-adjustment weights (SeasonalBias) ---------------------------
pub const CAL_ADAPTIVE_W_SEASONAL_INFIL: &str = "CAL_ADAPTIVE_W_SEASONAL_INFIL";
pub const CAL_ADAPTIVE_W_SEASONAL_SGAIN: &str = "CAL_ADAPTIVE_W_SEASONAL_SGAIN";

// -- Parameter-adjustment weights (MixedBias) ------------------------------
pub const CAL_ADAPTIVE_W_MIXED_TC: &str = "CAL_ADAPTIVE_W_MIXED_TC";
pub const CAL_ADAPTIVE_W_MIXED_INFIL: &str = "CAL_ADAPTIVE_W_MIXED_INFIL";
pub const CAL_ADAPTIVE_W_MIXED_IGAIN: &str = "CAL_ADAPTIVE_W_MIXED_IGAIN";
pub const CAL_ADAPTIVE_W_MIXED_SGAIN: &str = "CAL_ADAPTIVE_W_MIXED_SGAIN";

// -- Energy-simulation correction ------------------------------------------
pub const CAL_ADAPTIVE_INFIL_CORRECTION_FACTOR: &str = "CAL_ADAPTIVE_INFIL_CORRECTION_FACTOR";

// -- Target threshold ------------------------------------------------------
pub const CAL_ADAPTIVE_TARGET_ERROR_PCT: &str = "CAL_ADAPTIVE_TARGET_ERROR_PCT";

/// The audit ledger enumerating every calibration factor in the validation
/// pipeline.
///
/// **Adding a factor**: append a `mk(...)` entry here and reference its `id`
/// with a `// LEDGER: <id>` comment at the call site. The
/// [`calibration_ledger_is_complete`] drift gate enforces the pairing.
///
/// Built once via `once_cell` (owned `String` fields so the struct is fully
/// serde round-trippable); lends `&'static` references for [`lookup`].
pub static LEDGER: Lazy<Vec<CalibrationFactor>> = Lazy::new(|| {
    vec![
        // -- Case 195 empirical material-property defaults ---------------------
        mk(CAL_CASE195_THERMAL_CONDUCTIVITY,
            "case_195_calibration::CalibrationParameters::default",
            0.16,
            "Case 195 empirical fit (W/m·K); flagged for blind-mode review per TODO-BLIND-VALIDATION"),
        mk(CAL_CASE195_SPECIFIC_HEAT,
            "case_195_calibration::CalibrationParameters::default",
            840.0,
            "Case 195 empirical fit (J/kg·K); flagged for blind-mode review"),
        mk(CAL_CASE195_DENSITY,
            "case_195_calibration::CalibrationParameters::default",
            2400.0,
            "Case 195 empirical fit (kg/m³); flagged for blind-mode review"),
        mk(CAL_CASE195_INFILTRATION_RATE,
            "case_195_calibration::CalibrationParameters::default",
            0.5,
            "Case 195 empirical fit (ACH); flagged for blind-mode review"),
        // -- Adaptive CalibrationState defaults ---------------------------------
        mk(CAL_ADAPTIVE_THERMAL_CONDUCTIVITY,
            "adaptive_calibration::CalibrationState::default",
            0.16,
            "Mirrors CAL_CASE195_THERMAL_CONDUCTIVITY; empirical default (W/m·K)"),
        mk(CAL_ADAPTIVE_SPECIFIC_HEAT,
            "adaptive_calibration::CalibrationState::default",
            840.0,
            "Mirrors CAL_CASE195_SPECIFIC_HEAT; empirical default (J/kg·K)"),
        mk(CAL_ADAPTIVE_DENSITY,
            "adaptive_calibration::CalibrationState::default",
            2400.0,
            "Mirrors CAL_CASE195_DENSITY; empirical default (kg/m³)"),
        mk(CAL_ADAPTIVE_INFILTRATION_RATE,
            "adaptive_calibration::CalibrationState::default",
            0.5,
            "Mirrors CAL_CASE195_INFILTRATION_RATE; empirical default (ACH)"),
        mk(CAL_ADAPTIVE_INTERNAL_GAIN_MULT,
            "adaptive_calibration::CalibrationState::default",
            1.0,
            "Unit gain multiplier (no correction); baseline reference"),
        mk(CAL_ADAPTIVE_SOLAR_GAIN_MULT,
            "adaptive_calibration::CalibrationState::default",
            1.0,
            "Unit gain multiplier (no correction); baseline reference"),
        // -- Convergence / learning parameters ----------------------------------
        mk(CAL_ADAPTIVE_MAX_ITERATIONS,
            "adaptive_calibration::AdaptiveHourlyCalibrator::new",
            50.0,
            "Calibration-loop iteration cap; empirical convergence budget"),
        mk(CAL_ADAPTIVE_CONVERGENCE_TOLERANCE,
            "adaptive_calibration::AdaptiveHourlyCalibrator::new",
            0.01,
            "1% percentage-error convergence tolerance"),
        mk(CAL_ADAPTIVE_LEARNING_RATE,
            "adaptive_calibration::AdaptiveHourlyCalibrator::new",
            0.1,
            "Gradient-step learning rate for parameter adjustment"),
        // -- Trigger thresholds --------------------------------------------------
        mk(CAL_ADAPTIVE_TEMP_ANOMALY_THRESHOLD,
            "adaptive_calibration::TriggerDetector::new",
            5.0,
            "°C deviation from weekly-mean outdoor temp triggering weather-anomaly recalibration"),
        mk(CAL_ADAPTIVE_BIAS_CHANGE_THRESHOLD,
            "adaptive_calibration::TriggerDetector::new",
            0.15,
            "15% relative change in mean bias triggering bias-pattern-change recalibration"),
        mk(CAL_ADAPTIVE_OCCUPANCY_SHIFT_THRESHOLD,
            "adaptive_calibration::TriggerDetector::detect_triggers",
            0.3,
            "Absolute occupancy-level delta triggering occupancy-shift recalibration"),
        // -- Bias-classification thresholds -------------------------------------
        mk(CAL_ADAPTIVE_UNIVERSAL_BIAS_RATIO,
            "adaptive_calibration::SmartMeterPatternAnalyzer::classify_bias_pattern",
            0.1,
            "std_dev < |mean_bias| × this  ⇒  UniversalBias classification"),
        mk(CAL_ADAPTIVE_SEASONAL_CORR_THRESHOLD,
            "adaptive_calibration::SmartMeterPatternAnalyzer::classify_bias_pattern",
            0.5,
            "seasonal_correlation > this  ⇒  SeasonalBias classification"),
        mk(CAL_ADAPTIVE_MIXED_BIAS_RATIO,
            "adaptive_calibration::SmartMeterPatternAnalyzer::classify_bias_pattern",
            0.3,
            "std_dev > |mean_bias| × this (with correlation) ⇒ MixedBias classification"),
        mk(CAL_ADAPTIVE_MIXED_CORR_THRESHOLD,
            "adaptive_calibration::SmartMeterPatternAnalyzer::classify_bias_pattern",
            0.3,
            "seasonal_correlation > this (with high std_dev) ⇒ MixedBias classification"),
        // -- Parameter-adjustment weights (UniversalBias) -----------------------
        mk(CAL_ADAPTIVE_W_UNIVERSAL_TC,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[UniversalBias]",
            0.1,
            "thermal_conductivity adjustment weight under UniversalBias"),
        mk(CAL_ADAPTIVE_W_UNIVERSAL_INFIL,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[UniversalBias]",
            0.3,
            "infiltration_rate adjustment weight under UniversalBias"),
        mk(CAL_ADAPTIVE_W_UNIVERSAL_IGAIN,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[UniversalBias]",
            0.6,
            "internal_gain_multiplier adjustment weight under UniversalBias"),
        // -- Parameter-adjustment weights (SeasonalBias) ------------------------
        mk(CAL_ADAPTIVE_W_SEASONAL_INFIL,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[SeasonalBias]",
            0.2,
            "infiltration_rate adjustment weight under SeasonalBias"),
        mk(CAL_ADAPTIVE_W_SEASONAL_SGAIN,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[SeasonalBias]",
            0.8,
            "solar_gain_multiplier adjustment weight under SeasonalBias"),
        // -- Parameter-adjustment weights (MixedBias) ---------------------------
        mk(CAL_ADAPTIVE_W_MIXED_TC,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[MixedBias]",
            0.2,
            "thermal_conductivity adjustment weight under MixedBias"),
        mk(CAL_ADAPTIVE_W_MIXED_INFIL,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[MixedBias]",
            0.3,
            "infiltration_rate adjustment weight under MixedBias"),
        mk(CAL_ADAPTIVE_W_MIXED_IGAIN,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[MixedBias]",
            0.3,
            "internal_gain_multiplier adjustment weight under MixedBias"),
        mk(CAL_ADAPTIVE_W_MIXED_SGAIN,
            "adaptive_calibration::AdaptiveHourlyCalibrator::select_parameters[MixedBias]",
            0.2,
            "solar_gain_multiplier adjustment weight under MixedBias"),
        // -- Energy-simulation correction ---------------------------------------
        mk(CAL_ADAPTIVE_INFIL_CORRECTION_FACTOR,
            "adaptive_calibration::AdaptiveHourlyCalibrator::simulate_energy",
            0.1,
            "Infiltration sensitivity coefficient: correction = 1 + (rate − 0.5) × this"),
        // -- Target threshold ---------------------------------------------------
        mk(CAL_ADAPTIVE_TARGET_ERROR_PCT,
            "adaptive_calibration::AdaptiveCalibrationResult::from_calibrator",
            10.0,
            "Annual-energy target: |final_error_pct| < this ⇒ target_met (research: <10%)"),
    ]
});

/// Constructor helper: builds an active (not-yet-removed) [`CalibrationFactor`]
/// from `&'static str` inputs, owning the strings so the struct is serde-safe.
fn mk(
    id: &'static str,
    location: &'static str,
    value: f64,
    reference: &'static str,
) -> CalibrationFactor {
    CalibrationFactor {
        id: id.into(),
        location: location.into(),
        value,
        reference: reference.into(),
        removal_issue: None,
        removed_in_version: None,
    }
}

/// Number of factors currently recorded in the ledger.
pub fn ledger_len() -> usize {
    LEDGER.len()
}

/// Look up a calibration factor by its stable identifier.
pub fn lookup(id: &str) -> Option<&'static CalibrationFactor> {
    LEDGER.iter().find(|f| f.id == id)
}

/// Emit the active ledger factors as structured `tracing::info!` records so the
/// audit trail is captured in the blind-mode log stream.
///
/// Called by the validator when a run is in [`Blind` mode][crate::validation::ashrae_140_validator::ValidationMode::Blind].
/// Each factor is logged as a structured field set so downstream JSON log
/// consumers (Phase E SUSTAIN-01..02) can reconstruct the per-case factor
/// history without parsing free text.
pub fn emit_blind_mode_audit() {
    tracing::info!(
        target: "fluxion::validation::calibration_ledger",
        factor_count = LEDGER.len(),
        "calibration audit ledger: beginning blind-mode emission"
    );
    for factor in LEDGER.iter().filter(|f| f.is_active()) {
        tracing::info!(
            target: "fluxion::validation::calibration_ledger",
            id = factor.id,
            location = factor.location,
            value = factor.value,
            reference = factor.reference,
            "calibration factor active in this build"
        );
    }
    tracing::info!(
        target: "fluxion::validation::calibration_ledger",
        "calibration audit ledger: blind-mode emission complete"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `id` constant must resolve through `lookup`.
    #[test]
    fn lookup_resolves_all_id_constants() {
        let ids = [
            CAL_CASE195_THERMAL_CONDUCTIVITY,
            CAL_CASE195_SPECIFIC_HEAT,
            CAL_CASE195_DENSITY,
            CAL_CASE195_INFILTRATION_RATE,
            CAL_ADAPTIVE_THERMAL_CONDUCTIVITY,
            CAL_ADAPTIVE_SPECIFIC_HEAT,
            CAL_ADAPTIVE_DENSITY,
            CAL_ADAPTIVE_INFILTRATION_RATE,
            CAL_ADAPTIVE_INTERNAL_GAIN_MULT,
            CAL_ADAPTIVE_SOLAR_GAIN_MULT,
            CAL_ADAPTIVE_MAX_ITERATIONS,
            CAL_ADAPTIVE_CONVERGENCE_TOLERANCE,
            CAL_ADAPTIVE_LEARNING_RATE,
            CAL_ADAPTIVE_TEMP_ANOMALY_THRESHOLD,
            CAL_ADAPTIVE_BIAS_CHANGE_THRESHOLD,
            CAL_ADAPTIVE_OCCUPANCY_SHIFT_THRESHOLD,
            CAL_ADAPTIVE_UNIVERSAL_BIAS_RATIO,
            CAL_ADAPTIVE_SEASONAL_CORR_THRESHOLD,
            CAL_ADAPTIVE_MIXED_BIAS_RATIO,
            CAL_ADAPTIVE_MIXED_CORR_THRESHOLD,
            CAL_ADAPTIVE_W_UNIVERSAL_TC,
            CAL_ADAPTIVE_W_UNIVERSAL_INFIL,
            CAL_ADAPTIVE_W_UNIVERSAL_IGAIN,
            CAL_ADAPTIVE_W_SEASONAL_INFIL,
            CAL_ADAPTIVE_W_SEASONAL_SGAIN,
            CAL_ADAPTIVE_W_MIXED_TC,
            CAL_ADAPTIVE_W_MIXED_INFIL,
            CAL_ADAPTIVE_W_MIXED_IGAIN,
            CAL_ADAPTIVE_W_MIXED_SGAIN,
            CAL_ADAPTIVE_INFIL_CORRECTION_FACTOR,
            CAL_ADAPTIVE_TARGET_ERROR_PCT,
        ];
        for id in ids {
            assert!(lookup(id).is_some(), "ledger missing entry for {id}");
        }
    }

    /// Ledger IDs must be unique.
    #[test]
    fn ledger_ids_are_unique() {
        let mut ids: Vec<String> = LEDGER.iter().map(|f| f.id.clone()).collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            total,
            "duplicate calibration-factor IDs in LEDGER"
        );
    }

    #[test]
    fn ledger_has_expected_factor_count() {
        // 4 (case195) + 6 (adaptive defaults) + 3 (convergence) + 3 (triggers)
        // + 4 (bias classification) + 3 (universal weights) + 2 (seasonal weights)
        // + 4 (mixed weights) + 1 (infil correction) + 1 (target) = 31
        assert_eq!(
            ledger_len(),
            31,
            "factor count changed — update this test + docs"
        );
    }

    #[test]
    fn all_factors_active_by_default() {
        for f in LEDGER.iter() {
            assert!(f.is_active(), "{} marked removed but still in code", f.id);
        }
    }

    #[test]
    fn ledger_round_trips_serde_json() {
        let json = serde_json::to_string(&*LEDGER).expect("serialize");
        let parsed: Vec<CalibrationFactor> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.len(), LEDGER.len());
        assert_eq!(parsed[0], LEDGER[0]);
    }

    /// Drift gate (Issue #2516 acceptance criterion): every `// LEDGER: <id>`
    /// marker in the two tracked calibration source files must resolve to a
    /// ledger entry. If a contributor adds a calibration factor with a marker
    /// but omits the [`LEDGER`] entry, this test fails.
    ///
    /// Conversely, if a `// LEDGER:` marker references an unknown id the test
    /// also fails (catching typos).
    #[test]
    fn calibration_ledger_is_complete() {
        let files: &[(&str, &str)] = &[
            (
                "case_195_calibration.rs",
                include_str!("case_195_calibration.rs"),
            ),
            (
                "adaptive_calibration.rs",
                include_str!("adaptive_calibration.rs"),
            ),
        ];

        let mut referenced: Vec<(&str, String)> = Vec::new();
        for (file_name, src) in files {
            for (lineno, line) in src.lines().enumerate() {
                if let Some(idx) = line.find("// LEDGER:") {
                    let after = &line[idx + "// LEDGER:".len()..];
                    let id = after.split_whitespace().next().unwrap_or("");
                    assert!(
                        !id.is_empty(),
                        "{file_name}:{lineno}: empty `// LEDGER:` marker"
                    );
                    referenced.push((file_name, id.to_string()));
                }
            }
        }

        assert!(
            !referenced.is_empty(),
            "no `// LEDGER:` markers found — the convention must be applied to all factor sites"
        );

        let mut failures: Vec<String> = Vec::new();
        for (file_name, id) in &referenced {
            if lookup(id).is_none() {
                failures.push(format!(
                    "{file_name}: `// LEDGER: {id}` has no ledger entry"
                ));
            }
        }
        if !failures.is_empty() {
            panic!(
                "calibration ledger is incomplete ({} unmatched marker(s)):\n  - {}\n\
                 Add a `CalibrationFactor` entry to `LEDGER` for each missing id, \
                 or fix the marker typo.",
                failures.len(),
                failures.join("\n  - ")
            );
        }
    }

    /// Every active ledger entry must be referenced by at least one
    /// `// LEDGER:` marker in the tracked files — guards against stale entries
    /// left behind when a factor is removed from code but not the ledger.
    #[test]
    fn every_ledger_entry_is_referenced_in_code() {
        let files: &[(&str, &str)] = &[
            (
                "case_195_calibration.rs",
                include_str!("case_195_calibration.rs"),
            ),
            (
                "adaptive_calibration.rs",
                include_str!("adaptive_calibration.rs"),
            ),
        ];
        let mut referenced_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (_file, src) in files {
            for line in src.lines() {
                if let Some(idx) = line.find("// LEDGER:") {
                    let after = &line[idx + "// LEDGER:".len()..];
                    if let Some(id) = after.split_whitespace().next() {
                        referenced_ids.insert(id);
                    }
                }
            }
        }
        for factor in LEDGER.iter() {
            assert!(
                referenced_ids.contains(factor.id.as_str()),
                "ledger entry `{}` has no `// LEDGER:` marker in tracked files — \
                 either add the marker at its call site or remove the stale entry",
                factor.id
            );
        }
    }
}
