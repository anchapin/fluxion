//! Heuristic diagnostic copilot for ASHRAE 140 `SystematicIssue::Unknown` failures.
//!
//! When the 11-rule decision tree in [`issue_classifier`](super::issue_classifier)
//! cannot classify a failed metric, this module provides structured diagnostic
//! hypotheses based on the case identifier, metric type, and deviation
//! characteristics. These hypotheses are drawn from documented LIMIT-* patterns
//! in [`docs/KNOWN_ISSUES.md`] and from systematic deviation signatures observed
//! across the ASHRAE 140 case suite.
//!
//! # Design contract
//!
//! * **Diagnostic only** — this module suggests hypotheses, not conclusions. It
//!   supplements human investigation; it does not replace the decision tree.
//! * **No parameter tuning** — heuristics are based on structural signatures and
//!   documented known issues; they never suggest tuning a constant to close a gap.
//! * **Traceable** — every hypothesis carries a `source` field citing the KNOWN_ISSUES.md
//!   section or the deviation-pattern evidence that motivated it.

use crate::validation::issue_classifier::{
    case_mass, deviation_direction, deviation_magnitude, metric_axis, CaseMass, DeviationDirection,
    MetricAxis, ENERGY_SYSTEMATIC_THRESHOLD_PCT,
};
use crate::validation::report::{MetricType, ValidationResult};
use crate::validation::reporter::SystematicIssue;

/// A diagnostic hypothesis for an [`SystematicIssue::Unknown`] failure.
///
/// Each hypothesis names a probable issue category, provides a free-text rationale
/// explaining *why* this hypothesis fits the observed pattern, and cites the
/// [`docs/KNOWN_ISSUES.md`] section (or deviation-pattern source) that motivates it.
#[derive(Debug, Clone)]
pub struct DiagnosticHypothesis {
    /// Ranked likelihood (lower = more likely).
    pub rank: usize,
    /// Suggested systematic issue category.
    pub issue: SystematicIssue,
    /// Free-text rationale connecting the observed metrics to this hypothesis.
    pub rationale: String,
    /// KNOWN_ISSUES.md section or deviation-pattern source citation.
    pub source: &'static str,
}

impl DiagnosticHypothesis {
    fn new(
        rank: usize,
        issue: SystematicIssue,
        rationale: &'static str,
        source: &'static str,
    ) -> Self {
        Self {
            rank,
            issue,
            rationale: rationale.to_string(),
            source,
        }
    }
}

/// Checks whether a result is classified as Unknown by the main classifier.
pub fn is_unknown(result: &ValidationResult) -> bool {
    let mass = case_mass(&result.case_id);
    let axis = match metric_axis(&result.metric) {
        Some(axis) => axis,
        None => return true,
    };
    let direction = deviation_direction(result);
    let magnitude = deviation_magnitude(result);

    // Rule 1 — Case 960 annual energy
    if result.case_id == "960" && axis == MetricAxis::Energy {
        return false;
    }
    // Rule 2
    if axis == MetricAxis::Energy
        && mass == CaseMass::HighMass
        && direction == DeviationDirection::Over
        && magnitude >= ENERGY_SYSTEMATIC_THRESHOLD_PCT
    {
        return false;
    }
    // Rule 3
    if axis == MetricAxis::Energy
        && mass == CaseMass::LowMass
        && direction == DeviationDirection::Under
        && magnitude >= ENERGY_SYSTEMATIC_THRESHOLD_PCT
    {
        return false;
    }
    // Rule 4
    if result.metric == MetricType::PeakCooling && direction == DeviationDirection::Under {
        return false;
    }
    // Rule 5
    if result.metric == MetricType::PeakCooling
        && direction == DeviationDirection::Over
        && mass == CaseMass::HighMass
    {
        return false;
    }
    // Rule 6
    if axis == MetricAxis::FreeFloat && mass == CaseMass::HighMass {
        return false;
    }
    // Rule 7
    if axis == MetricAxis::FreeFloat && mass == CaseMass::LowMass {
        return false;
    }
    // Rule 8
    if result.metric == MetricType::PeakHeating && mass == CaseMass::HighMass {
        return false;
    }
    // Rule 9
    if result.metric == MetricType::PeakHeating && mass == CaseMass::LowMass {
        return false;
    }
    // Rule 10
    if axis == MetricAxis::Energy && mass == CaseMass::Special {
        return false;
    }
    // Rule 11
    if result.metric == MetricType::PeakCooling
        && direction == DeviationDirection::Over
        && mass == CaseMass::LowMass
    {
        return false;
    }

    true
}

/// Returns ordered diagnostic hypotheses for an [`SystematicIssue::Unknown`] failure.
///
/// The hypotheses are ranked by likelihood based on the deviation pattern,
/// case series membership, and documented LIMIT-* signatures in KNOWN_ISSUES.md.
/// Returns an empty slice when no hypotheses can be confidently generated.
pub fn diagnose(result: &ValidationResult) -> Vec<DiagnosticHypothesis> {
    if !is_unknown(result) {
        return Vec::new();
    }

    let mass = case_mass(&result.case_id);
    let axis = match metric_axis(&result.metric) {
        Some(axis) => axis,
        None => return Vec::new(),
    };
    let direction = deviation_direction(result);
    let magnitude = deviation_magnitude(result);
    let abs_pct = magnitude;

    let mut hypotheses: Vec<DiagnosticHypothesis> = Vec::new();

    // ---- Case-series specific heuristics ----

    // Case 600/610/620/630/640/650 series (LowMass)
    if result.case_id.starts_with('6') || result.case_id == "195" {
        match axis {
            MetricAxis::Energy => {
                // LowMass energy but not caught by Rule 3 (Under+>=30%)
                // Check for over-prediction patterns similar to LIMIT-30
                if direction == DeviationDirection::Over {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::SolarGains,
                        "Low-mass annual energy over-prediction — possible solar distribution routing issue. \
                         LIMIT-30 (Issue #3797) identifies Case 600 solar distribution to air fraction as the \
                         dominant deviation source for this series.",
                        "KNOWN_ISSUES.md §LIMIT-30 (Issue #3797)",
                    ));
                }
                // Under + magnitude below threshold — might be thermal mass coupling
                if direction == DeviationDirection::Under
                    && abs_pct < ENERGY_SYSTEMATIC_THRESHOLD_PCT
                {
                    hypotheses.push(DiagnosticHypothesis::new(
                        2,
                        SystematicIssue::ThermalMass,
                        "Low-mass annual energy under-prediction with magnitude below the 30% systematic \
                         threshold — possible thermal coupling asymmetry. The 5R1C/9R4C mass-node \
                         routing may under-estimate solar buffering in lightweight constructions.",
                        "Deviation pattern + LIMIT-05 family (Issue #3072)",
                    ));
                }
            }
            MetricAxis::Peak => {
                // Peak metrics not caught by rules 4/9/11
                if result.metric == MetricType::PeakCooling && direction == DeviationDirection::Over
                {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::HvacLoad,
                        "Low-mass peak cooling over-prediction — LIMIT-16 (Issue #3059) documents \
                         the single-lumped thermal-mass node signature (dt/τ ≈ 3.6) driving peak OVER \
                         in this series.",
                        "KNOWN_ISSUES.md §LIMIT-16 (Issue #3059)",
                    ));
                }
                if result.metric == MetricType::PeakHeating
                    && direction == DeviationDirection::Under
                {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::HvacLoad,
                        "Low-mass peak heating under-prediction — possible HVAC control timing or \
                         setpoint/deadband issue. Examine the hourly heating profile for \
                         premature setback recovery.",
                        "Deviation pattern + LIMIT-05 family",
                    ));
                }
            }
            MetricAxis::FreeFloat => {
                if direction == DeviationDirection::Under {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::SolarGains,
                        "Low-mass free-float minimum under-prediction — the night-vent or sky-radiation \
                         boundary may be overwhelming the lightweight structure. LIMIT-17 (Issue #3058) \
                         documents the ~570 W/K h_ve_night forcing on envelope mass.",
                        "KNOWN_ISSUES.md §LIMIT-17 (Issue #3058)",
                    ));
                }
                if direction == DeviationDirection::Over {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::ThermalMass,
                        "Low-mass free-float maximum over-prediction — insufficient thermal damping. \
                         Lightweight constructions should show faster equilibration; investigate \
                         whether mass-node coupling is over-counting.",
                        "Deviation pattern",
                    ));
                }
            }
        }
    }

    // Case 900/910/920/930/940/950 series (HighMass)
    if result.case_id.starts_with('9') && result.case_id != "960" {
        match axis {
            MetricAxis::Energy => {
                // HighMass energy but not caught by Rule 2 (Over+>=30%)
                if direction == DeviationDirection::Under {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::SolarGains,
                        "High-mass annual energy under-prediction — possible solar lag or thermal mass \
                         damping issue. Heavy constructions buffer solar gains over longer timescales; \
                         investigate whether the lag filter is under-counting.",
                        "LIMIT-05 family (Issue #3072)",
                    ));
                }
                if direction == DeviationDirection::Over
                    && abs_pct < ENERGY_SYSTEMATIC_THRESHOLD_PCT
                {
                    hypotheses.push(DiagnosticHypothesis::new(
                        2,
                        SystematicIssue::ModelLimitation,
                        "High-mass annual energy slight over-prediction below the 30% systematic threshold — \
                         may be a 5R1C model limitation rather than a bug. Monitor for regression.",
                        "Rule 2 boundary (Issue #1423)",
                    ));
                }
            }
            MetricAxis::Peak => {
                if result.metric == MetricType::PeakCooling
                    && direction == DeviationDirection::Under
                {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::SolarGains,
                        "High-mass peak cooling under-prediction — solar gains may be damped too \
                         aggressively by the thermal mass filter. The τ_lag = √(τ_air × τ_mass) \
                         lag correction may be over-counting.",
                        "LIMIT-05 family + LIMIT-21 Phase 9 (Issue #3916)",
                    ));
                }
                if result.metric == MetricType::PeakHeating
                    && direction == DeviationDirection::Under
                {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::ThermalMass,
                        "High-mass peak heating under-prediction — thermal mass dynamics may be \
                         damping peak demand too aggressively. Consider the air-trajectory \
                         equilibration rate (dt/τ ≈ 3.6 for Gauge).",
                        "LIMIT-21 (Issue #3297) + LIMIT-05 family",
                    ));
                }
                if result.metric == MetricType::PeakHeating && direction == DeviationDirection::Over
                {
                    hypotheses.push(DiagnosticHypothesis::new(
                        2,
                        SystematicIssue::ThermalMass,
                        "High-mass peak heating over-prediction — possible overnight setback \
                         recovery overshoot. LIMIT-12 (Issue #3062) documents similar \
                         patterns in Case 940 annual heating.",
                        "KNOWN_ISSUES.md §LIMIT-12 (Issue #3062)",
                    ));
                }
            }
            MetricAxis::FreeFloat => {
                if direction == DeviationDirection::Under {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::ThermalMass,
                        "High-mass free-float minimum under-prediction — night-sky radiative forcing \
                         may not be coupling correctly to the thermal mass. LIMIT-17 (Issue #3058) \
                         documents h_ve_night overwhelming the exterior film correction by ~8×.",
                        "KNOWN_ISSUES.md §LIMIT-17 (Issue #3058)",
                    ));
                }
                if direction == DeviationDirection::Over {
                    hypotheses.push(DiagnosticHypothesis::new(
                        1,
                        SystematicIssue::SolarGains,
                        "High-mass free-float maximum over-prediction — possible insufficient damping \
                         of peak solar gains. Investigate whether solar distribution routing is \
                         bypassing the mass node.",
                        "LIMIT-05 family (Issue #3072)",
                    ));
                }
            }
        }
    }

    // Case 960 (special inter-zone)
    if result.case_id == "960" {
        // Rule 1 should catch Energy, but Peak/FreeFloat may fall through
        if axis == MetricAxis::Peak {
            hypotheses.push(DiagnosticHypothesis::new(
                1,
                SystematicIssue::InterZoneTransfer,
                "Case 960 peak metric not classified by Rule 1 — possible sunspace/back-zone \
                 coupling asymmetry. LIMIT-14 (Issue #3061) tracks the annual cooling OVER; \
                 peak may exhibit similar inter-zone dynamics.",
                "KNOWN_ISSUES.md §LIMIT-14 (Issue #3061)",
            ));
        }
        if axis == MetricAxis::FreeFloat {
            hypotheses.push(DiagnosticHypothesis::new(
                1,
                SystematicIssue::InterZoneTransfer,
                "Case 960 free-float not classified — inter-zone conductance matrix may \
                 not be equilibrating correctly between sunspace and back zone.",
                "LIMIT-14 (Issue #3061) + LIMIT-18 (Issue #3104)",
            ));
        }
    }

    // Case 970 (multi-zone)
    if result.case_id == "970" {
        hypotheses.push(DiagnosticHypothesis::new(
            1,
            SystematicIssue::InterZoneTransfer,
            "Case 970 is a 5-zone multi-zone topology per ASHRAE 140-2017 §B6.7. \
             LIMIT-23 (Issue #3552) documents simultaneous heating+cooling OVER — consistent with \
             5R1C/9R4C air-mass distribution amplifying solar over-charge in the coupling matrix.",
            "KNOWN_ISSUES.md §LIMIT-23 (Issue #3552)",
        ));
    }

    // Case 800/810 series
    if result.case_id.starts_with("80") {
        hypotheses.push(DiagnosticHypothesis::new(
            1,
            SystematicIssue::WeatherData,
            "Case 800/810 series — verify weather data source matches ASHRAE 140 DRYCOLD.TM2 \
             specification. Case 195 weather methodology issue (LIMIT-15, Issue #3060) suggests \
             weather-file selection is a known source of systematic deviation.",
            "KNOWN_ISSUES.md §LIMIT-15 (Issue #3060)",
        ));
    }

    // ---- Generic deviation-pattern heuristics ----

    // Very large deviations (>=50%) that didn't match any rule
    if abs_pct >= 50.0 && mass == CaseMass::Special {
        hypotheses.push(DiagnosticHypothesis::new(
            1,
            SystematicIssue::WeatherData,
            "Non-standard case with >=50% deviation — large deviations on unrecognised case \
             identifiers often indicate a weather data or case construction specification mismatch. \
             Verify the case spec matches the ASHRAE 140-2023 Annex B definition.",
            "Deviation magnitude + LIMIT-15 (Issue #3060)",
        ));
    }

    // Small-magnitude in-range failures (flagged despite being near midpoint)
    if direction == DeviationDirection::InRange {
        hypotheses.push(DiagnosticHypothesis::new(
            1,
            SystematicIssue::ModelLimitation,
            "Result flagged as failed despite being inside the reference range — this is an \
             in-range-but-flagged pattern. The ASHRAE 140 tolerance band (ref * 0.95 to ref * 1.05) \
             may be too tight for this metric's natural variability, or the reference data \
             source may differ.",
            "ValidationResult status = InRange + ASHRAE 140-2023 §8 tolerance bands",
        ));
    }

    // IncidentSolar (not handled by the decision tree)
    if matches!(result.metric, MetricType::IncidentSolar { .. }) {
        hypotheses.push(DiagnosticHypothesis::new(
            1,
            SystematicIssue::SolarGains,
            "IncidentSolar metrics are not part of the BESTEST pass/fail set per ASHRAE 140-2023 \
             §8.2.3 — they are informational outputs. Per-metric distribution routing should be \
             validated against analytical flux calculations (see LIMIT-30 Issue #3797 B1a audit).",
            "issue_classifier.rs + KNOWN_ISSUES.md §LIMIT-30 (Issue #3797)",
        ));
    }

    // Sort by rank (lowest first = most likely) and return
    hypotheses.sort_by_key(|a| a.rank);
    hypotheses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::report::ValidationResult;

    fn unknown_result(
        case_id: &str,
        metric: MetricType,
        fluxion: f64,
        ref_min: f64,
        ref_max: f64,
    ) -> ValidationResult {
        ValidationResult::new(case_id, metric, fluxion, ref_min, ref_max)
    }

    #[test]
    fn test_is_unknown_for_lowmass_energy_over() {
        // LowMass + Energy + Over should be caught by rule 3 only if Under
        let r = unknown_result("600", MetricType::AnnualCooling, 10.0, 5.0, 7.0);
        assert!(
            is_unknown(&r),
            "LowMass Energy Over should be Unknown (no rule matches Over)"
        );
    }

    #[test]
    fn test_is_unknown_for_highmass_energy_under() {
        // HighMass + Energy + Under should be Unknown (rule 2 is for Over)
        let r = unknown_result("900", MetricType::AnnualHeating, 1.0, 2.0, 4.0);
        assert!(is_unknown(&r), "HighMass Energy Under should be Unknown");
    }

    #[test]
    fn test_diagnose_lowmass_peak_cooling_over() {
        let r = unknown_result("600", MetricType::PeakCooling, 9.0, 5.0, 7.0);
        let hypotheses = diagnose(&r);
        assert!(!hypotheses.is_empty());
        assert_eq!(hypotheses[0].issue, SystematicIssue::HvacLoad);
    }

    #[test]
    fn test_diagnose_case_960_peak() {
        let r = unknown_result("960", MetricType::PeakHeating, 9.0, 5.0, 7.0);
        let hypotheses = diagnose(&r);
        assert!(!hypotheses.is_empty());
        assert_eq!(hypotheses[0].issue, SystematicIssue::InterZoneTransfer);
    }

    #[test]
    fn test_diagnose_case_970() {
        let r = unknown_result("970", MetricType::AnnualCooling, 25.0, 7.0, 10.0);
        let hypotheses = diagnose(&r);
        assert!(!hypotheses.is_empty());
        assert_eq!(hypotheses[0].issue, SystematicIssue::InterZoneTransfer);
    }

    #[test]
    fn test_diagnose_incident_solar() {
        let r = unknown_result(
            "600",
            MetricType::IncidentSolar {
                surface_id: "roof".into(),
                orientation: crate::validation::ashrae_140_cases::Orientation::Horizontal,
            },
            1200.0,
            800.0,
            1000.0,
        );
        let hypotheses = diagnose(&r);
        assert!(!hypotheses.is_empty());
        assert_eq!(hypotheses[0].issue, SystematicIssue::SolarGains);
    }

    #[test]
    fn test_diagnose_non_unknown_returns_empty() {
        // A result that IS classified by the tree should return empty
        let r = unknown_result("960", MetricType::AnnualCooling, 5.0, 1.6, 2.8);
        assert!(!is_unknown(&r));
        assert!(diagnose(&r).is_empty());
    }

    #[test]
    fn test_diagnose_case_800() {
        let r = unknown_result("800", MetricType::AnnualHeating, 10.0, 5.0, 7.0);
        let hypotheses = diagnose(&r);
        assert!(!hypotheses.is_empty());
        assert_eq!(hypotheses[0].issue, SystematicIssue::WeatherData);
    }
}
