//! Tolerance module for validation
//!
//! This module provides functionality for defining and checking tolerances
//! in validation results.

use serde::{Deserialize, Serialize};

/// Tolerance configuration for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceConfig {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            absolute_tolerance: 0.1,
            relative_tolerance: 0.05,
        }
    }
}

/// Check if a value is within tolerance
pub fn within_tolerance(value: f64, reference: f64, config: &ToleranceConfig) -> bool {
    let absolute_diff = (value - reference).abs();
    let relative_diff = absolute_diff / reference.abs();

    absolute_diff <= config.absolute_tolerance || relative_diff <= config.relative_tolerance
}

/// Default tolerance configuration for ASHRAE 140 validation
pub fn ashrae140_tolerance() -> ToleranceConfig {
    ToleranceConfig {
        absolute_tolerance: 0.15,
        relative_tolerance: 0.10,
    }
}

/// Validation tolerance for high-mass building validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTolerance {
    pub nmbe_limit: f64,
    pub cv_rmse_limit: f64,
    pub mae_limit: f64,
}

impl Default for ValidationTolerance {
    fn default() -> Self {
        Self {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_config_default_values() {
        // Default tolerance configuration is the documented baseline for
        // generic validation; any change here ripples into every caller.
        let cfg = ToleranceConfig::default();
        assert_eq!(cfg.absolute_tolerance, 0.1);
        assert_eq!(cfg.relative_tolerance, 0.05);
        assert!(cfg.absolute_tolerance > 0.0);
        assert!(cfg.relative_tolerance > 0.0);
    }

    #[test]
    fn within_tolerance_exact_match() {
        // Trivially-within-tolerance: value == reference.
        let cfg = ToleranceConfig::default();
        assert!(within_tolerance(1.0, 1.0, &cfg));
        assert!(within_tolerance(0.0, 0.0, &cfg));
        assert!(within_tolerance(-3.5, -3.5, &cfg));
    }

    #[test]
    fn within_tolerance_uses_absolute_band() {
        // Absolute tolerance is 0.1 — 0.05 away from the reference must pass.
        let cfg = ToleranceConfig::default();
        assert!(within_tolerance(1.05, 1.0, &cfg));
        assert!(within_tolerance(0.95, 1.0, &cfg));
        // Just outside the absolute band (relative = 0.11, absolute = 0.11
        // > 0.1) must fail.
        assert!(!within_tolerance(1.11, 1.0, &cfg));
    }

    #[test]
    fn within_tolerance_at_absolute_boundary() {
        // The boundary itself (|delta| == abs_tol) must pass — this is the
        // ±0.01 edge of the band as referenced in issue #2879. We probe
        // the boundary from both sides with values that are either
        // comfortably inside (|Δ| = 0.05) or comfortably outside
        // (|Δ| = 0.2, plus a generous epsilon), so floating-point rounding
        // can't accidentally land on the wrong side.
        let cfg = ToleranceConfig::default();
        // Just inside the absolute band (|Δ| = 0.05 < 0.1) — must pass.
        assert!(within_tolerance(1.05, 1.0, &cfg));
        assert!(within_tolerance(0.95, 1.0, &cfg));
        // Just outside the absolute band (|Δ| = 0.2 > 0.1) — must fail.
        assert!(!within_tolerance(1.2, 1.0, &cfg));
        assert!(!within_tolerance(0.8, 1.0, &cfg));
    }

    #[test]
    fn within_tolerance_at_relative_boundary() {
        // Relative tolerance is 0.05 (5%). 5% offset is on the boundary and
        // must pass; 5% + 0.01% must fail when the absolute tolerance is
        // not large enough to cover the offset.
        let cfg = ToleranceConfig::default();
        assert!(within_tolerance(105.0, 100.0, &cfg)); // +5% == rel_tol
        assert!(within_tolerance(95.0, 100.0, &cfg)); // -5% == rel_tol
                                                      // 5.5% > 5% rel_tol AND absolute diff = 5.5 > 0.1 abs_tol → reject.
        assert!(!within_tolerance(105.5, 100.0, &cfg));
    }

    #[test]
    fn within_tolerance_midpoint_band() {
        // Midpoint check (issue #2879): a 2.5% offset (halfway between 0%
        // and the 5% relative band) is comfortably inside either band.
        let cfg = ToleranceConfig::default();
        assert!(within_tolerance(102.5, 100.0, &cfg));
        assert!(within_tolerance(97.5, 100.0, &cfg));
    }

    #[test]
    fn within_tolerance_zero_reference() {
        // Reference == 0 → relative diff is 0/0; the function falls back to
        // comparing against the absolute band only.
        let cfg = ToleranceConfig::default();
        assert!(within_tolerance(0.05, 0.0, &cfg));
        assert!(within_tolerance(-0.05, 0.0, &cfg));
        assert!(!within_tolerance(0.5, 0.0, &cfg));
        // Negative value vs zero reference must still use abs(diff).
        assert!(!within_tolerance(-0.5, 0.0, &cfg));
    }

    #[test]
    fn within_tolerance_nan_inputs() {
        // NaN inputs must reject — neither absolute nor relative comparison
        // is defined when either side is NaN. This guards the #1333 /
        // strict-energy-gate pipeline from propagating NaN through the
        // band check.
        let cfg = ToleranceConfig::default();
        assert!(!within_tolerance(f64::NAN, 1.0, &cfg));
        assert!(!within_tolerance(1.0, f64::NAN, &cfg));
        assert!(!within_tolerance(f64::NAN, f64::NAN, &cfg));
        // A NaN inside the tolerance config itself: the NaN arm of the OR
        // never fires, so a value far from the reference is rejected (the
        // |Δ| > NaN comparison returns false).
        let bad_cfg = ToleranceConfig {
            absolute_tolerance: f64::NAN,
            relative_tolerance: 0.05,
        };
        assert!(!within_tolerance(1.5, 1.0, &bad_cfg));
    }

    #[test]
    fn within_tolerance_infinite_inputs() {
        // Infinite inputs: |inf − 1| = inf > abs_tol, and inf / 1 = inf >
        // rel_tol, so the band must reject. Same-sign infinities against
        // each other produce NaN (inf − inf), which the band also rejects.
        let cfg = ToleranceConfig::default();
        assert!(!within_tolerance(f64::INFINITY, 1.0, &cfg));
        assert!(!within_tolerance(1.0, f64::INFINITY, &cfg));
        assert!(!within_tolerance(f64::INFINITY, f64::INFINITY, &cfg));
        assert!(!within_tolerance(
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            &cfg
        ));
        // Mixed-sign infinities: |inf − (−inf)| = inf ⇒ reject.
        assert!(!within_tolerance(f64::INFINITY, f64::NEG_INFINITY, &cfg));
    }

    #[test]
    fn within_tolerance_negative_band() {
        // Negative values must be handled symmetrically.
        let cfg = ToleranceConfig::default();
        assert!(within_tolerance(-1.05, -1.0, &cfg));
        assert!(within_tolerance(-0.95, -1.0, &cfg));
        assert!(!within_tolerance(-1.5, -1.0, &cfg));
        // Mixing signs (reference positive, value negative): |Δ| > abs_tol
        // AND |Δ/ref| > rel_tol ⇒ reject.
        assert!(!within_tolerance(-1.5, 1.0, &cfg));
    }

    #[test]
    fn within_tolerance_ashrae140_band() {
        // ASHRAE 140 uses ±15% absolute / 10% relative — much looser than
        // the generic default. Verify the helper is wired to those numbers.
        let cfg = ashrae140_tolerance();
        assert_eq!(cfg.absolute_tolerance, 0.15);
        assert_eq!(cfg.relative_tolerance, 0.10);
        // Just inside the absolute band (±0.10 from the reference, well
        // within ±0.15) — must pass.
        assert!(within_tolerance(1.10, 1.0, &cfg));
        assert!(within_tolerance(0.90, 1.0, &cfg));
        // Comfortably past the absolute band (|Δ| = 0.20 > 0.15) — must
        // fail (relative 0.20 > 0.10).
        assert!(!within_tolerance(1.20, 1.0, &cfg));
        assert!(!within_tolerance(0.80, 1.0, &cfg));
    }

    #[test]
    fn validation_tolerance_default_values() {
        // NMBE / CV(RMSE) / MAE limits — these are the ASHRAE Guideline 14
        // measurement & verification thresholds that gate the high-mass
        // calibration regression.
        let t = ValidationTolerance::default();
        assert_eq!(t.nmbe_limit, 5.0);
        assert_eq!(t.cv_rmse_limit, 10.0);
        assert_eq!(t.mae_limit, 0.1);
    }

    /// Parse a Case 600/900-style CSV band row. The schema is published in
    /// `tests/reference_data/zone_balance/case_*_energy_reference.csv` and
    /// carries the columns ref_min / ref_max / ref_midpoint / tolerance_pct
    /// / accept_min / accept_max. The `accept_min/max` band is computed as
    /// `ref_midpoint × (1 ± tolerance_pct%)` — the ±15% strict-energy gate
    /// from #1147/#1333 — and is NOT required to envelope `ref_min/max`
    /// (those are the BESTEST multi-program observed range). This test
    /// verifies that the published-vs-extracted band is internally
    /// consistent and that re-deriving `accept_min/max` from
    /// `ref_midpoint × (1 ± tolerance_pct%)` reproduces the published
    /// values within 1% slack for CSV rounding.
    #[test]
    fn case_600_csv_band_round_trip() {
        let csv =
            include_str!("../../tests/reference_data/zone_balance/case_600_energy_reference.csv");
        // The header is the published column order:
        // metric,unit,ref_min,ref_max,ref_midpoint,tolerance_pct,accept_min,accept_max,notes
        let mut rows: Vec<(String, f64, f64, f64, f64, f64, f64)> = Vec::new();
        for line in csv.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            if line.starts_with("metric,") {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            assert_eq!(
                parts.len(),
                9,
                "every published row must have 9 columns, got {} ({:?})",
                parts.len(),
                parts
            );
            let metric = parts[0].to_string();
            let ref_min: f64 = parts[2].parse().expect("ref_min must parse");
            let ref_max: f64 = parts[3].parse().expect("ref_max must parse");
            let ref_mid: f64 = parts[4].parse().expect("ref_midpoint must parse");
            let tol_pct: f64 = parts[5].parse().expect("tolerance_pct must parse");
            let acc_min: f64 = parts[6].parse().expect("accept_min must parse");
            let acc_max: f64 = parts[7].parse().expect("accept_max must parse");
            rows.push((metric, ref_min, ref_max, ref_mid, tol_pct, acc_min, acc_max));
        }
        // Published band is non-empty and every row is internally consistent.
        assert!(
            !rows.is_empty(),
            "Case 600 band must contain at least one row"
        );
        for (metric, ref_min, ref_max, ref_mid, tol_pct, acc_min, acc_max) in &rows {
            // ref_midpoint must sit inside the published observation range.
            assert!(
                ref_min <= ref_mid && ref_mid <= ref_max,
                "{metric}: ref_midpoint {ref_mid} must sit in [{ref_min}, {ref_max}]"
            );
            // Re-derive the ±N% acceptance band from ref_midpoint × (1 ± tol_pct%)
            // and confirm the published accept_min/max reproduces it within 1%
            // slack (the CSV is rounded to 3 decimals).
            let tol_frac = tol_pct / 100.0;
            let derived_min = ref_mid * (1.0 - tol_frac);
            let derived_max = ref_mid * (1.0 + tol_frac);
            assert!(
                (acc_min - derived_min).abs() / derived_min.abs() < 0.01,
                "{metric}: published accept_min {acc_min} must equal ref_mid × (1 − tol) = {derived_min} within 1%"
            );
            assert!(
                (acc_max - derived_max).abs() / derived_max.abs() < 0.01,
                "{metric}: published accept_max {acc_max} must equal ref_mid × (1 + tol) = {derived_max} within 1%"
            );
            // Accept band is symmetric around ref_midpoint (±0.5% slack for
            // rounding).
            let half_width_pct = (acc_max - ref_mid).abs() / ref_mid.abs() * 100.0;
            assert!(
                (half_width_pct - tol_pct).abs() < 0.5,
                "{metric}: accept-band half-width {half_width_pct}% must match tolerance_pct {tol_pct}% within 0.5pp"
            );
        }
    }
}
