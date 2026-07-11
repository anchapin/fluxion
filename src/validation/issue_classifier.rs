//! Data-driven systematic-issue classifier for ASHRAE 140 validation failures.
//!
//! Replaces the hardcoded `(case_id, metric)` heuristic that previously lived in
//! `reporter.rs::classify_issue` (issue #1423). The old heuristic dispatched only
//! 4 narrow patterns and left 32 of 50 failed metrics as `SystematicIssue::Unknown`.
//!
//! This module computes four objective features directly from the
//! [`ValidationResult`] and routes the failure through a fixed, parameter-free
//! decision tree:
//!
//! | Feature              | Source                                        | Values                    |
//! |----------------------|-----------------------------------------------|---------------------------|
//! | `case_mass`          | `case_id` (proxy for `CaseSpec.construction`) | `LowMass`/`HighMass`/`Special` |
//! | `metric_axis`        | `MetricType`                                  | `Energy`/`Peak`/`FreeFloat`   |
//! | `deviation_direction`| `fluxion_value` vs `[ref_min, ref_max]`       | `Over`/`Under`/`InRange`      |
//! | `deviation_magnitude`| `percent_error.abs()`                         | `f64` (percent)               |
//!
//! # Design contract
//!
//! * **No parameter tuning** — every threshold is fixed (only `30.0%`, taken from
//!   the issue spec). Adjusting it to make a particular case pass would violate
//!   AGENTS.md ("No parameter tuning ... fix the underlying math").
//! * **Deterministic** — pure function of the result; same input always yields
//!   the same output (locked by `classifier_property_test`).
//! * **Stable** — the decision tree is static data, not a fitted model, so it
//!   cannot drift between runs.
//!
//! # Decision tree
//!
//! ```text
//! 1. case 960 + Energy                  -> InterZoneTransfer   (issue #273)
//! 2. Energy + HighMass + Over + |err|>=30% -> ModelLimitation  (5R1C limit)
//! 3. Energy + LowMass + Under + |err|>=30% -> SolarGains
//! 4. PeakCooling + Under                -> SolarGains
//! 5. PeakCooling + Over + HighMass      -> ThermalMass
//! 6. FreeFloat + HighMass               -> ThermalMass
//! 7. FreeFloat + LowMass                -> SolarGains
//! 8. PeakHeating + HighMass             -> ThermalMass
//! 9. PeakHeating + LowMass              -> HvacLoad
//! 10. Energy + Special                  -> HvacLoad
//! 11. PeakCooling + Over + LowMass      -> HvacLoad
//! else                                  -> Unknown (genuinely small bucket)
//! ```

use crate::validation::report::{MetricType, ValidationResult};
use crate::validation::reporter::SystematicIssue;

/// Construction-mass family inferred from the ASHRAE 140 case identifier.
///
/// This is a lightweight proxy for `CaseSpec.construction_type`; the classifier
/// only has access to the [`ValidationResult`] (which carries `case_id`, not the
/// full case spec), so the mass family is recovered from the well-known BESTEST
/// case numbering:
///
/// * `900`–`999` (incl. `900FF`, `950FF`, `960`) -> `HighMass`
/// * `195`, `600`–`699` (incl. `600FF`, `650FF`) -> `LowMass`
/// * anything else                                -> `Special`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseMass {
    /// Lightweight constructions (Cases 195, 600–650 and their free-float twins).
    LowMass,
    /// Heavyweight constructions (Cases 900–960 and their free-float twins).
    HighMass,
    /// Unrecognised or non-standard case identifiers.
    Special,
}

/// Coarse metric family used by the decision tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricAxis {
    /// Annual energy totals (`AnnualHeating`, `AnnualCooling`).
    Energy,
    /// Peak loads (`PeakHeating`, `PeakCooling`).
    Peak,
    /// Free-floating temperatures (`MinFreeFloat`, `MaxFreeFloat`).
    FreeFloat,
}

/// Direction of the simulation value relative to the reference range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviationDirection {
    /// `fluxion_value > ref_max`.
    Over,
    /// `fluxion_value < ref_min`.
    Under,
    /// Inside `[ref_min, ref_max]` (only set when a result is flagged failed
    /// despite sitting inside the band, e.g. via a tightened tolerance).
    InRange,
}

/// Threshold (percent) above which an energy deviation is considered systematic.
///
/// Fixed by the issue #1423 specification; **do not tune** — see module docs.
pub const ENERGY_SYSTEMATIC_THRESHOLD_PCT: f64 = 30.0;

/// Infers the construction-mass family from an ASHRAE 140 case identifier.
pub fn case_mass(case_id: &str) -> CaseMass {
    if case_id.starts_with('9') {
        CaseMass::HighMass
    } else if case_id == "195" || case_id.starts_with('6') {
        CaseMass::LowMass
    } else {
        CaseMass::Special
    }
}

/// Maps a [`MetricType`] to its coarse axis family.
///
/// `IncidentSolar` (per-surface irradiance) is not part of the BESTEST pass/fail
/// set and maps to `None`; such failures stay in the `Unknown` bucket.
pub fn metric_axis(metric: &MetricType) -> Option<MetricAxis> {
    match metric {
        MetricType::AnnualHeating | MetricType::AnnualCooling => Some(MetricAxis::Energy),
        MetricType::PeakHeating | MetricType::PeakCooling => Some(MetricAxis::Peak),
        MetricType::MinFreeFloat | MetricType::MaxFreeFloat => Some(MetricAxis::FreeFloat),
        MetricType::IncidentSolar { .. } => None,
    }
}

/// Direction of `fluxion_value` relative to the reference range `[ref_min, ref_max]`.
pub fn deviation_direction(result: &ValidationResult) -> DeviationDirection {
    if result.fluxion_value > result.ref_max {
        DeviationDirection::Over
    } else if result.fluxion_value < result.ref_min {
        DeviationDirection::Under
    } else {
        DeviationDirection::InRange
    }
}

/// Absolute percent deviation of the simulation value from the reference midpoint.
pub fn deviation_magnitude(result: &ValidationResult) -> f64 {
    result.percent_error.abs()
}

/// Classifies a single failed [`ValidationResult`] into a [`SystematicIssue`].
///
/// Pure, deterministic, and parameter-free. See the [module docs](self) for the
/// full decision tree and design contract.
pub fn classify(result: &ValidationResult) -> SystematicIssue {
    let mass = case_mass(&result.case_id);
    let axis = match metric_axis(&result.metric) {
        Some(axis) => axis,
        // IncidentSolar and any future non-BESTEST metric: not classifiable here.
        None => return SystematicIssue::Unknown,
    };
    let direction = deviation_direction(result);
    let magnitude = deviation_magnitude(result);

    // Rule 1 — Case 960 annual energy: inter-zone heat transfer (issue #273).
    if result.case_id == "960" && axis == MetricAxis::Energy {
        return SystematicIssue::InterZoneTransfer;
    }

    // Rule 2 — High-mass energy over-prediction: 5R1C model limitation.
    if axis == MetricAxis::Energy
        && mass == CaseMass::HighMass
        && direction == DeviationDirection::Over
        && magnitude >= ENERGY_SYSTEMATIC_THRESHOLD_PCT
    {
        return SystematicIssue::ModelLimitation;
    }

    // Rule 3 — Low-mass energy under-prediction: solar gains dominate.
    if axis == MetricAxis::Energy
        && mass == CaseMass::LowMass
        && direction == DeviationDirection::Under
        && magnitude >= ENERGY_SYSTEMATIC_THRESHOLD_PCT
    {
        return SystematicIssue::SolarGains;
    }

    // Rule 4 — Peak cooling under-prediction: solar gains.
    if result.metric == MetricType::PeakCooling && direction == DeviationDirection::Under {
        return SystematicIssue::SolarGains;
    }

    // Rule 5 — Peak cooling over-prediction in high-mass: thermal mass dynamics.
    if result.metric == MetricType::PeakCooling
        && direction == DeviationDirection::Over
        && mass == CaseMass::HighMass
    {
        return SystematicIssue::ThermalMass;
    }

    // Rule 6 — Free-floating temperatures in high-mass: thermal mass dynamics.
    if axis == MetricAxis::FreeFloat && mass == CaseMass::HighMass {
        return SystematicIssue::ThermalMass;
    }

    // Rule 7 — Free-floating temperatures in low-mass: solar-driven excursions.
    if axis == MetricAxis::FreeFloat && mass == CaseMass::LowMass {
        return SystematicIssue::SolarGains;
    }

    // Rule 8 — Peak heating in high-mass: thermal mass dynamics.
    if result.metric == MetricType::PeakHeating && mass == CaseMass::HighMass {
        return SystematicIssue::ThermalMass;
    }

    // Rule 9 — Peak heating in low-mass: HVAC load calculation.
    if result.metric == MetricType::PeakHeating && mass == CaseMass::LowMass {
        return SystematicIssue::HvacLoad;
    }

    // Rule 10 — Energy in special/non-standard cases: HVAC load calculation.
    if axis == MetricAxis::Energy && mass == CaseMass::Special {
        return SystematicIssue::HvacLoad;
    }

    // Rule 11 — Peak cooling over-prediction in low-mass: HVAC load calculation.
    if result.metric == MetricType::PeakCooling
        && direction == DeviationDirection::Over
        && mass == CaseMass::LowMass
    {
        return SystematicIssue::HvacLoad;
    }

    // Genuinely small bucket: unrecognised case/metric combinations or
    // in-range-but-failed results that the tree does not model.
    SystematicIssue::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::report::{MetricType, ValidationResult};

    /// Helper: build a failed result (fluxion well outside the ref band).
    fn failed(
        case_id: &str,
        metric: MetricType,
        fluxion: f64,
        ref_min: f64,
        ref_max: f64,
    ) -> ValidationResult {
        ValidationResult::new(case_id, metric, fluxion, ref_min, ref_max)
    }

    // ------------------------------------------------------------------------
    // Feature extractors
    // ------------------------------------------------------------------------

    #[test]
    fn test_case_mass_inference() {
        assert_eq!(case_mass("195"), CaseMass::LowMass);
        for c in &["600", "610", "620", "630", "640", "650", "600FF", "650FF"] {
            assert_eq!(case_mass(c), CaseMass::LowMass, "case {}", c);
        }
        for c in &["900", "910", "920", "930", "940", "950", "960", "900FF", "950FF"] {
            assert_eq!(case_mass(c), CaseMass::HighMass, "case {}", c);
        }
        assert_eq!(case_mass("XXX"), CaseMass::Special);
        assert_eq!(case_mass("1234"), CaseMass::Special);
    }

    #[test]
    fn test_metric_axis_mapping() {
        assert_eq!(metric_axis(&MetricType::AnnualHeating), Some(MetricAxis::Energy));
        assert_eq!(metric_axis(&MetricType::AnnualCooling), Some(MetricAxis::Energy));
        assert_eq!(metric_axis(&MetricType::PeakHeating), Some(MetricAxis::Peak));
        assert_eq!(metric_axis(&MetricType::PeakCooling), Some(MetricAxis::Peak));
        assert_eq!(metric_axis(&MetricType::MinFreeFloat), Some(MetricAxis::FreeFloat));
        assert_eq!(metric_axis(&MetricType::MaxFreeFloat), Some(MetricAxis::FreeFloat));
        assert_eq!(
            metric_axis(&MetricType::IncidentSolar {
                surface_id: "roof".into(),
                orientation: crate::validation::ashrae_140_cases::Orientation::Horizontal,
            }),
            None,
            "IncidentSolar is not part of the BESTEST pass/fail set"
        );
    }

    #[test]
    fn test_deviation_direction() {
        assert_eq!(
            deviation_direction(&failed("600", MetricType::AnnualCooling, 8.0, 5.0, 7.0)),
            DeviationDirection::Over
        );
        assert_eq!(
            deviation_direction(&failed("600", MetricType::AnnualCooling, 3.0, 5.0, 7.0)),
            DeviationDirection::Under
        );
        assert_eq!(
            deviation_direction(&failed("600", MetricType::AnnualCooling, 6.0, 5.0, 7.0)),
            DeviationDirection::InRange
        );
    }

    // ------------------------------------------------------------------------
    // Decision tree — per-rule coverage
    // ------------------------------------------------------------------------

    #[test]
    fn test_rule1_case_960_energy_interzone() {
        // Over the band, but rule 1 fires first for any 960 energy metric.
        let r = failed("960", MetricType::AnnualCooling, 5.0, 1.6, 2.8);
        assert_eq!(classify(&r), SystematicIssue::InterZoneTransfer);
        let r = failed("960", MetricType::AnnualHeating, 0.5, 1.6, 2.8);
        assert_eq!(classify(&r), SystematicIssue::InterZoneTransfer);
    }

    #[test]
    fn test_rule2_highmass_energy_over_model_limitation() {
        for c in &["900", "910", "920", "930", "940", "950", "900FF", "950FF"] {
            let r = failed(c, MetricType::AnnualHeating, 5.0, 1.17, 2.04);
            assert_eq!(
                classify(&r),
                SystematicIssue::ModelLimitation,
                "case {}",
                c
            );
        }
    }

    #[test]
    fn test_rule2_below_threshold_falls_through() {
        // High-mass energy over but < 30% -> not ModelLimitation.
        // ref_mid=6.0, fluxion=6.5 -> ~8.3% deviation.
        let r = failed("900", MetricType::AnnualHeating, 6.5, 5.0, 7.0);
        // Over + HighMass + Energy but mag<30 -> falls through to Unknown
        // (no other rule matches Energy+HighMass non-960).
        assert_eq!(classify(&r), SystematicIssue::Unknown);
    }

    #[test]
    fn test_rule3_lowmass_energy_under_solar_gains() {
        for c in &["600", "610", "620", "630", "640", "650", "195"] {
            let r = failed(c, MetricType::AnnualCooling, 3.0, 5.0, 7.0);
            assert_eq!(classify(&r), SystematicIssue::SolarGains, "case {}", c);
        }
    }

    #[test]
    fn test_rule4_peak_cooling_under_solar_gains() {
        for c in &["600", "610", "900", "920", "960", "195"] {
            let r = failed(c, MetricType::PeakCooling, 3.0, 5.0, 7.0);
            assert_eq!(classify(&r), SystematicIssue::SolarGains, "case {}", c);
        }
    }

    #[test]
    fn test_rule5_peak_cooling_over_highmass_thermal_mass() {
        for c in &["900", "920", "950", "960"] {
            let r = failed(c, MetricType::PeakCooling, 9.0, 5.0, 7.0);
            assert_eq!(classify(&r), SystematicIssue::ThermalMass, "case {}", c);
        }
    }

    #[test]
    fn test_rule6_freefloat_highmass_thermal_mass() {
        let r = failed("900FF", MetricType::MinFreeFloat, 30.0, 40.0, 50.0);
        assert_eq!(classify(&r), SystematicIssue::ThermalMass);
        let r = failed("950FF", MetricType::MaxFreeFloat, 60.0, 40.0, 50.0);
        assert_eq!(classify(&r), SystematicIssue::ThermalMass);
    }

    #[test]
    fn test_rule7_freefloat_lowmass_solar_gains() {
        let r = failed("600FF", MetricType::MaxFreeFloat, 60.0, 40.0, 50.0);
        assert_eq!(classify(&r), SystematicIssue::SolarGains);
        let r = failed("650FF", MetricType::MinFreeFloat, 30.0, 40.0, 50.0);
        assert_eq!(classify(&r), SystematicIssue::SolarGains);
    }

    #[test]
    fn test_rule8_peak_heating_highmass_thermal_mass() {
        for c in &["900", "920", "940", "960"] {
            let r = failed(c, MetricType::PeakHeating, 9.0, 5.0, 7.0);
            assert_eq!(classify(&r), SystematicIssue::ThermalMass, "case {}", c);
        }
    }

    #[test]
    fn test_rule9_peak_heating_lowmass_hvac_load() {
        for c in &["600", "610", "195"] {
            let r = failed(c, MetricType::PeakHeating, 9.0, 5.0, 7.0);
            assert_eq!(classify(&r), SystematicIssue::HvacLoad, "case {}", c);
        }
    }

    #[test]
    fn test_rule10_energy_special_hvac_load() {
        let r = failed("XXX", MetricType::AnnualCooling, 9.0, 5.0, 7.0);
        assert_eq!(classify(&r), SystematicIssue::HvacLoad);
    }

    #[test]
    fn test_rule11_peak_cooling_over_lowmass_hvac_load() {
        let r = failed("600", MetricType::PeakCooling, 9.0, 5.0, 7.0);
        assert_eq!(classify(&r), SystematicIssue::HvacLoad);
    }

    #[test]
    fn test_unknown_for_unrecognised_peak_heating() {
        // Special mass + PeakHeating: no rule matches -> Unknown.
        let r = failed("XXX", MetricType::PeakHeating, 9.0, 5.0, 7.0);
        assert_eq!(classify(&r), SystematicIssue::Unknown);
    }

    #[test]
    fn test_unknown_for_incident_solar() {
        let r = failed(
            "600",
            MetricType::IncidentSolar {
                surface_id: "roof".into(),
                orientation: crate::validation::ashrae_140_cases::Orientation::Horizontal,
            },
            9.0,
            5.0,
            7.0,
        );
        assert_eq!(classify(&r), SystematicIssue::Unknown);
    }

    // ------------------------------------------------------------------------
    // Determinism + stability (property test mirrored in tests/test_reporter.rs)
    // ------------------------------------------------------------------------

    #[test]
    fn test_classify_is_deterministic() {
        let r = failed("920", MetricType::AnnualCooling, 5.0, 1.17, 2.04);
        let a = classify(&r);
        for _ in 0..50 {
            assert_eq!(classify(&r), a, "classifier must be deterministic");
        }
    }
}
