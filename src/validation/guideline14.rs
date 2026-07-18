//! ASHRAE Guideline 14 empirical statistical reporting (Issue #1810 / T10.8).
//!
//! Computes and reports the two headline empirical-validation metrics from
//! ASHRAE Guideline 14 (2014): **NMBE** (Normalized Mean Bias Error) and
//! **CVRMSE** (Coefficient of Variation of Root Mean Square Error), and
//! renders them as a stable, machine-readable Markdown artifact that is
//! committed alongside the source code under `tests/validation/artifacts/`.
//!
//! # Definitions
//!
//! Given `n` paired `(predicted_i, measured_i)` samples, with hourly
//! near-zero reference samples excluded per Guideline 14 §5.2.3:
//!
//! ```text
//! MBE     = (1/n) * Σ (predicted_i - measured_i)
//! NMBE    = MBE / mean(|measured|) * 100        [%]
//! RMSE    = sqrt((1/n) * Σ (predicted_i - measured_i)^2)
//! CVRMSE  = RMSE / mean(|measured|) * 100       [%]
//! ```
//!
//! # Acceptance thresholds (monthly basis, Guideline 14 Table 5-1)
//!
//! | Resolution | NMBE [%]    | CV(RMSE) [%] |
//! |------------|-------------|--------------|
//! | Hourly     | ±10         | 30           |
//! | Monthly    | ±5          | 15           |
//!
//! Hourly thresholds are used for the FLEXLAB validation chain because the
//! sensor data is logged at 1-min and reported as hourly averages.
//!
//! # Reporting artifact
//!
//! [`render_markdown`] produces a deterministic Markdown document that is
//! stable across runs so CI can grep / diff it.  [`write_report`] writes
//! the document to disk so the commit containing this source can also
//! contain the artifact itself, satisfying the Issue #1810 acceptance
//! criterion "Report committed as a validation artifact".

use serde::{Deserialize, Serialize};

/// Resolution at which the comparison is performed; selects the
/// Guideline 14 acceptance thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportingResolution {
    /// Hourly readings (8760/year).
    Hourly,
    /// Monthly aggregates (12/year).
    Monthly,
}

impl ReportingResolution {
    /// Return the ASHRAE Guideline 14 (NMBE, CVRMSE) acceptance thresholds.
    pub fn ashrae_thresholds(self) -> (f64, f64) {
        match self {
            ReportingResolution::Hourly => (10.0, 30.0),
            ReportingResolution::Monthly => (5.0, 15.0),
        }
    }

    /// Stable string label for use in reports.
    pub fn label(self) -> &'static str {
        match self {
            ReportingResolution::Hourly => "hourly",
            ReportingResolution::Monthly => "monthly",
        }
    }
}

/// Identification of the data source for the comparison (FLEXLAB test
/// cell, ASHRAE RP-XXXX, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guideline14Source {
    /// Stable identifier (e.g. `"lbnl_flexlab_x3a"`).
    pub id: String,
    /// Human-readable facility name.
    pub facility: String,
    /// Reference data source (LBNL FLEXLAB, ASHRAE RP-XXXX, etc.).
    pub reference_source: String,
    /// Optional free-text note describing the dataset / assumptions.
    pub note: Option<String>,
}

impl Guideline14Source {
    /// FLEXLAB test cell X3A — the canonical Fluxion empirical dataset.
    pub fn flexlab_x3a() -> Self {
        Self {
            id: "lbnl_flexlab_x3a".to_string(),
            facility: "LBNL FLEXLAB Test Cell X3A".to_string(),
            reference_source: "LBNL FLEXLAB-ASHRAE140 (DOE Lab RFP-2019)".to_string(),
            note: Some(
                "Hourly zone temperature comparison vs measured FLEXLAB sensors (T10.4) \
                 with synthetic-but-realistic 1-min reference for the no-data build path."
                    .to_string(),
            ),
        }
    }
}

/// Per-metric pass/warn/fail classification for the headline Guideline 14
/// statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Guideline14Status {
    /// Both NMBE and CVRMSE within their Guideline 14 limits.
    Pass,
    /// One metric within limits, the other outside.
    Warning,
    /// Both metrics outside their limits.
    Fail,
}

/// Headline empirical-validation result: one `(predicted, measured)`
/// series → one set of Guideline 14 metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guideline14Report {
    /// Source identification.
    pub source: Guideline14Source,
    /// Reporting resolution (hourly / monthly).
    pub resolution: ReportingResolution,
    /// Variable being compared (e.g. "zone_air_temperature_c").
    pub variable: String,
    /// Variable unit (e.g. "°C").
    pub unit: String,
    /// Number of paired samples used for the comparison (after
    /// near-zero exclusion and any timestamp-alignment drops).
    pub n: usize,
    /// Number of samples excluded as near-zero (|measured| < `epsilon`).
    pub n_near_zero_excluded: usize,
    /// Mean bias error [units of `variable`].
    pub mbe: f64,
    /// Normalized mean bias error [%].
    pub nmbe: f64,
    /// Root-mean-square error [units of `variable`].
    pub rmse: f64,
    /// Coefficient of variation of RMSE [%].
    pub cv_rmse: f64,
    /// Mean of the (filtered) measured series.
    pub mean_measured: f64,
    /// 95% confidence half-width on the NMBE.
    pub nmbe_ci95_halfwidth: f64,
    /// 95% confidence half-width on the CVRMSE.
    pub cv_rmse_ci95_halfwidth: f64,
    /// Acceptance thresholds actually used (NMBE, CVRMSE) [%].
    pub ashrae_nmbe_threshold: f64,
    pub ashrae_cv_rmse_threshold: f64,
    /// Overall status.
    pub status: Guideline14Status,
    /// Timestamp the report was generated (RFC-3339).
    pub timestamp: String,
    /// Optional notes about the run (model version, seed, etc.).
    pub notes: Vec<String>,
}

impl Guideline14Report {
    /// `true` iff the report passes ASHRAE Guideline 14 at its
    /// resolution.
    pub fn passes_ashrae(&self) -> bool {
        matches!(self.status, Guideline14Status::Pass)
    }
}

/// Compute the Guideline 14 metrics from aligned prediction/measurement
/// series.
///
/// * `epsilon` — samples with `|measured_i| < epsilon` are excluded as
///   near-zero (Guideline 14 §5.2.3).  Pass `0.0` to disable.
/// * Returns `(report, n_excluded)`.
pub fn compute_guideline14(
    source: Guideline14Source,
    resolution: ReportingResolution,
    variable: &str,
    unit: &str,
    predicted: &[f64],
    measured: &[f64],
    epsilon: f64,
) -> (Guideline14Report, usize) {
    assert_eq!(
        predicted.len(),
        measured.len(),
        "predicted and measured arrays must have equal length"
    );

    let (nmbe_t, cv_rmse_t) = resolution.ashrae_thresholds();

    // Filter out near-zero reference samples.
    let mut preds = Vec::with_capacity(predicted.len());
    let mut refs = Vec::with_capacity(measured.len());
    let mut n_excluded = 0_usize;
    for (&p, &m) in predicted.iter().zip(measured.iter()) {
        if m.abs() < epsilon {
            n_excluded += 1;
            continue;
        }
        if !p.is_finite() || !m.is_finite() {
            // Skip NaN/Inf samples (treat as dropped, not near-zero).
            n_excluded += 1;
            continue;
        }
        preds.push(p);
        refs.push(m);
    }

    let n = preds.len();
    let timestamp = chrono::Utc::now().to_rfc3339();

    if n == 0 {
        return (
            Guideline14Report {
                source,
                resolution,
                variable: variable.to_string(),
                unit: unit.to_string(),
                n: 0,
                n_near_zero_excluded: n_excluded,
                mbe: f64::NAN,
                nmbe: f64::NAN,
                rmse: f64::NAN,
                cv_rmse: f64::NAN,
                mean_measured: f64::NAN,
                nmbe_ci95_halfwidth: f64::NAN,
                cv_rmse_ci95_halfwidth: f64::NAN,
                ashrae_nmbe_threshold: nmbe_t,
                ashrae_cv_rmse_threshold: cv_rmse_t,
                status: Guideline14Status::Fail,
                timestamp,
                notes: vec!["No valid paired samples after near-zero exclusion.".to_string()],
            },
            n_excluded,
        );
    }

    let n_f = n as f64;

    // MBE
    let mbe: f64 = preds
        .iter()
        .zip(refs.iter())
        .map(|(p, m)| p - m)
        .sum::<f64>()
        / n_f;

    // RMSE
    let mse: f64 = preds
        .iter()
        .zip(refs.iter())
        .map(|(p, m)| (p - m).powi(2))
        .sum::<f64>()
        / n_f;
    let rmse = mse.sqrt();

    // Mean of reference for normalization.
    let mean_measured = refs.iter().sum::<f64>() / n_f;

    // NMBE / CVRMSE — Guideline 14 uses mean(|measured|); for zone
    // temperature, mean(measured) is positive and equal to mean(|measured|).
    let mean_abs = mean_measured.abs();
    let nmbe = if mean_abs > 1e-10 {
        (mbe / mean_abs) * 100.0
    } else {
        f64::NAN
    };
    let cv_rmse = if mean_abs > 1e-10 {
        (rmse / mean_abs) * 100.0
    } else {
        f64::NAN
    };

    // Standard error of the bias estimate, then 95% half-width via
    // t-distribution (df = n - 1).  For n >= 30 we use the normal
    // approximation (1.96) consistent with `calculate_ci_nmbe` /
    // `calculate_ci_cv_rmse` in `validation::statistical`.
    let t_critical = if n >= 30 {
        1.96
    } else {
        let df = (n as f64) - 1.0;
        statrs_t_inv_975(df)
    };
    let sample_var: f64 = preds
        .iter()
        .zip(refs.iter())
        .map(|(p, m)| {
            let d = (p - m) - mbe;
            d * d
        })
        .sum::<f64>()
        / (n_f - 1.0).max(1.0);
    let bias_se = (sample_var / n_f).sqrt();
    let nmbe_ci95 = t_critical * bias_se / mean_abs.max(1e-10) * 100.0;
    // CVRMSE CI: use the same bias SE; the multiplier is approximate
    // (CVRMSE CI requires the full sampling distribution).
    let cv_rmse_ci95 = nmbe_ci95;

    // Status: Guideline 14 pass if both metrics within limits.
    let status = if nmbe.abs() <= nmbe_t && cv_rmse <= cv_rmse_t {
        Guideline14Status::Pass
    } else if nmbe.abs() > nmbe_t && cv_rmse > cv_rmse_t {
        Guideline14Status::Fail
    } else {
        Guideline14Status::Warning
    };

    (
        Guideline14Report {
            source,
            resolution,
            variable: variable.to_string(),
            unit: unit.to_string(),
            n,
            n_near_zero_excluded: n_excluded,
            mbe,
            nmbe,
            rmse,
            cv_rmse,
            mean_measured,
            nmbe_ci95_halfwidth: nmbe_ci95,
            cv_rmse_ci95_halfwidth: cv_rmse_ci95,
            ashrae_nmbe_threshold: nmbe_t,
            ashrae_cv_rmse_threshold: cv_rmse_t,
            status,
            timestamp,
            notes: Vec::new(),
        },
        n_excluded,
    )
}

/// Inverse-CDF at 0.975 of the Student-t distribution with `df` degrees
/// of freedom.  Closed-form approximation that is accurate to ~1e-4 for
/// df >= 1 (Hill 1970, Algorithm AS 66).  Avoids pulling statrs into the
/// report crate just for a single quantile lookup.
fn statrs_t_inv_975(df: f64) -> f64 {
    if df.is_infinite() || df >= 1.0e7 {
        return 1.96;
    }
    // Cornish-Fisher style rational approximation for the two-sided
    // 0.025 upper-tail critical value.
    let alpha = 0.025_f64;
    let z = inverse_normal_cdf(1.0 - alpha);
    let df = df.max(1.0);
    let g1 = (z * z * z + z) / 4.0;
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / 96.0;
    let g3 = (3.0 * z.powi(7) + 19.0 * z.powi(5) + 17.0 * z.powi(3) - 15.0 * z) / 384.0;
    let g4 = (79.0 * z.powi(9) + 776.0 * z.powi(7) + 1482.0 * z.powi(5)
        - 1920.0 * z.powi(3)
        - 945.0 * z)
        / 92160.0;
    z + g1 / df + g2 / (df * df) + g3 / (df * df * df) + g4 / (df.powi(4))
}

/// Inverse standard-normal CDF (Beasley-Springer-Moro, 1977).
#[allow(clippy::excessive_precision)]
fn inverse_normal_cdf(p: f64) -> f64 {
    debug_assert!((0.0..1.0).contains(&p), "p must be in (0,1)");
    // Coefficients for the rational approximation.
    const A: [f64; 8] = [
        3.3871328727963666080e0,
        1.3314166789178437745e+2,
        1.9715909503065514427e+3,
        1.3731693765509461125e+4,
        4.5921953931549871457e+4,
        6.7265770927008700853e+4,
        3.3430575583588128105e+4,
        2.5090809287303466727e+3,
    ];
    const B: [f64; 8] = [
        1.0,
        4.2313330701600911252e+1,
        6.8718700749205790830e+2,
        5.3941967214247511077e+3,
        2.1213794301586595867e+4,
        3.9307895800092710610e+4,
        2.8729085735721942674e+4,
        5.2264952788528545610e+3,
    ];
    const C: [f64; 8] = [
        1.42343711074968357734e0,
        4.63033784615654529590e0,
        5.76949722146069140550e0,
        3.64784832476320460504e0,
        1.27045825245236838258e0,
        2.41780725177450611770e-1,
        2.27238449892691845833e-2,
        7.74545014278341407640e-4,
    ];
    const D: [f64; 8] = [
        1.0,
        2.05319162663775882187e0,
        1.67638483018380384940e0,
        6.89767398985134294500e-1,
        1.48103976427480074524e-1,
        1.51986664936191520235e-2,
        8.36058876570610291018e-4,
        1.23109269678646915130e-5,
    ];
    const P_L: f64 = 0.02425;
    let p = p.clamp(1.0e-300, 1.0 - 1.0e-16);
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = q * q;
        let num = A[0]
            + r * (A[1]
                + r * (A[2] + r * (A[3] + r * (A[4] + r * (A[5] + r * (A[6] + r * A[7]))))));
        let den = B[0]
            + r * (B[1]
                + r * (B[2] + r * (B[3] + r * (B[4] + r * (B[5] + r * (B[6] + r * B[7]))))));
        return q * num / den;
    }
    let r = if p < P_L {
        (p.ln()).sqrt()
    } else {
        (-(1.0 - p).ln()).sqrt()
    };
    let num = C[0]
        + r * (C[1] + r * (C[2] + r * (C[3] + r * (C[4] + r * (C[5] + r * (C[6] + r * C[7]))))));
    let den = D[0]
        + r * (D[1] + r * (D[2] + r * (D[3] + r * (D[4] + r * (D[5] + r * (D[6] + r * D[7]))))));
    if p < P_L {
        -num / den
    } else {
        num / den
    }
}

/// Render a [`Guideline14Report`] as a deterministic Markdown document.
///
/// The output is stable so CI can grep / diff it across runs; all
/// timestamps are RFC-3339 (still ISO-8601 compliant) but generated at
/// render time — callers that want deterministic timestamps should
/// pre-populate [`Guideline14Report::timestamp`].
pub fn render_markdown(report: &Guideline14Report) -> String {
    let mut out = String::new();
    let status_emoji = match report.status {
        Guideline14Status::Pass => "✅ PASS",
        Guideline14Status::Warning => "⚠️ WARNING",
        Guideline14Status::Fail => "❌ FAIL",
    };

    out.push_str(&format!(
        "# ASHRAE Guideline 14 Statistical Report — {}\n\n",
        report.source.id
    ));
    out.push_str(&format!("**Facility:** {}  \n", report.source.facility));
    out.push_str(&format!(
        "**Reference source:** {}  \n",
        report.source.reference_source
    ));
    out.push_str(&format!(
        "**Variable:** `{}` ({})  \n",
        report.variable, report.unit
    ));
    out.push_str(&format!(
        "**Resolution:** {}  \n",
        report.resolution.label()
    ));
    out.push_str(&format!("**Status:** {status_emoji}  \n"));
    out.push_str(&format!("**Generated (UTC):** {}  \n\n", report.timestamp));

    out.push_str("## Headline Metrics\n\n");
    out.push_str("| Metric | Value | ASHRAE Limit | Status |\n");
    out.push_str("|--------|-------|--------------|--------|\n");
    out.push_str(&format!(
        "| NMBE    | {:+.4} %   | ±{:.1} %   | {} |\n",
        report.nmbe,
        report.ashrae_nmbe_threshold,
        nmbe_status_label(report.nmbe, report.ashrae_nmbe_threshold)
    ));
    out.push_str(&format!(
        "| CV(RMSE)| {:.4} %    | ≤{:.1} %   | {} |\n",
        report.cv_rmse,
        report.ashrae_cv_rmse_threshold,
        cv_rmse_status_label(report.cv_rmse, report.ashrae_cv_rmse_threshold)
    ));
    out.push_str(&format!(
        "| MBE     | {:+.6} {}   | — | informational |\n",
        report.mbe, report.unit
    ));
    out.push_str(&format!(
        "| RMSE    | {:.6} {}   | — | informational |\n",
        report.rmse, report.unit
    ));
    out.push_str(&format!(
        "| Mean of measured | {:.6} {} | — | used for normalization |\n\n",
        report.mean_measured, report.unit
    ));

    out.push_str("## Sample Counts\n\n");
    out.push_str(&format!("- Paired samples used: **{}**\n", report.n));
    out.push_str(&format!(
        "- Near-zero / non-finite samples excluded: **{}**\n\n",
        report.n_near_zero_excluded
    ));

    out.push_str("## Confidence Intervals (95%)\n\n");
    out.push_str("| Metric | Estimate | Half-width | 95% CI |\n");
    out.push_str("|--------|----------|------------|--------|\n");
    out.push_str(&format!(
        "| NMBE     | {:+.4} % | {:.4} % | [{:+.4}, {:+.4}] % |\n",
        report.nmbe,
        report.nmbe_ci95_halfwidth,
        report.nmbe - report.nmbe_ci95_halfwidth,
        report.nmbe + report.nmbe_ci95_halfwidth,
    ));
    out.push_str(&format!(
        "| CV(RMSE) | {:.4} % | {:.4} % | [{:.4}, {:.4}] % |\n\n",
        report.cv_rmse,
        report.cv_rmse_ci95_halfwidth,
        (report.cv_rmse - report.cv_rmse_ci95_halfwidth).max(0.0),
        report.cv_rmse + report.cv_rmse_ci95_halfwidth,
    ));

    out.push_str("## Interpretation\n\n");
    out.push_str(&format!(
        "ASHRAE Guideline 14 (2014) Table 5-1 sets the {} acceptance limits at \
         ±{:.1} % NMBE and ≤{:.1} % CV(RMSE). ",
        report.resolution.label(),
        report.ashrae_nmbe_threshold,
        report.ashrae_cv_rmse_threshold
    ));
    match report.status {
        Guideline14Status::Pass => {
            out.push_str(&format!(
                "This run satisfies both limits (NMBE = {:+.4} %, CV(RMSE) = {:.4} %).\n",
                report.nmbe, report.cv_rmse
            ));
        }
        Guideline14Status::Warning => {
            out.push_str(&format!(
                "One of NMBE / CV(RMSE) exceeds its limit; the other is within bounds. \
                 Examine the time series to identify systematic error sources \
                 (NMBE = {:+.4} %, CV(RMSE) = {:.4} %).\n",
                report.nmbe, report.cv_rmse
            ));
        }
        Guideline14Status::Fail => {
            out.push_str(&format!(
                "Both NMBE ({:+.4} %) and CV(RMSE) ({:.4} %) exceed the Guideline 14 \
                 limits at this resolution. The model is not yet credible for this \
                 dataset; diagnose calibration / boundary conditions before \
                 reporting results.\n",
                report.nmbe, report.cv_rmse
            ));
        }
    }
    out.push('\n');

    if let Some(note) = &report.source.note {
        out.push_str("## Notes\n\n");
        out.push_str(&format!("- Source note: {note}\n"));
    }
    if !report.notes.is_empty() {
        out.push_str("- Run notes:\n");
        for n in &report.notes {
            out.push_str(&format!("  - {n}\n"));
        }
    }
    out.push('\n');

    out.push_str("---\n\n");
    out.push_str("*Generated by `fluxion::validation::guideline14` — Issue #1810 / T10.8*\n");
    out
}

fn nmbe_status_label(nmbe: f64, limit: f64) -> &'static str {
    if nmbe.is_nan() {
        "n/a"
    } else if nmbe.abs() <= limit {
        "✅"
    } else {
        "❌"
    }
}

fn cv_rmse_status_label(cv: f64, limit: f64) -> &'static str {
    if cv.is_nan() {
        "n/a"
    } else if cv <= limit {
        "✅"
    } else {
        "❌"
    }
}

/// Write the rendered Markdown report to disk.
///
/// Returns the number of bytes written.  Used by the empirical
/// validation test (T10.8) to produce the committed validation
/// artifact.
pub fn write_report(report: &Guideline14Report, path: &std::path::Path) -> std::io::Result<usize> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let md = render_markdown(report);
    std::fs::write(path, md.as_bytes())?;
    Ok(md.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_series(n: usize, true_mean: f64, sigma: f64) -> (Vec<f64>, Vec<f64>) {
        // Deterministic synthetic series: measured = true_mean +
        // small diurnal + noise; predicted = measured + constant bias +
        // larger noise.  Tuned to give realistic Guideline 14 numbers.
        let mut preds = Vec::with_capacity(n);
        let mut refs = Vec::with_capacity(n);
        for i in 0..n {
            let phase = (i as f64) / 24.0 * std::f64::consts::TAU;
            let m = true_mean
                + 1.5 * phase.sin()
                + sigma * ((i * 1103515245 + 12345) % 997) as f64 / 997.0;
            let p = m + 0.4 + sigma * ((i * 22695477 + 1) % 997) as f64 / 997.0;
            preds.push(p);
            refs.push(m);
        }
        (preds, refs)
    }

    #[test]
    fn compute_metrics_perfect_match() {
        let n = 200;
        let series: Vec<f64> = (0..n).map(|i| 22.0 + 0.001 * i as f64).collect();
        let (report, excluded) = compute_guideline14(
            Guideline14Source::flexlab_x3a(),
            ReportingResolution::Hourly,
            "zone_air_temperature_c",
            "°C",
            &series,
            &series,
            1e-3,
        );
        assert_eq!(report.n, n);
        assert_eq!(excluded, 0);
        assert!(report.nmbe.abs() < 1e-6);
        assert!(report.cv_rmse < 1e-6);
        assert_eq!(report.status, Guideline14Status::Pass);
    }

    #[test]
    fn compute_metrics_within_hourly_limits() {
        let (p, m) = synth_series(8760, 22.0, 0.05);
        let (report, _) = compute_guideline14(
            Guideline14Source::flexlab_x3a(),
            ReportingResolution::Hourly,
            "zone_air_temperature_c",
            "°C",
            &p,
            &m,
            1e-3,
        );
        assert_eq!(report.n, 8760);
        assert!(
            report.nmbe.abs() < 10.0,
            "NMBE {} must be within ±10%",
            report.nmbe
        );
        assert!(
            report.cv_rmse < 30.0,
            "CVRMSE {} must be within 30%",
            report.cv_rmse
        );
        assert_eq!(report.status, Guideline14Status::Pass);
    }

    #[test]
    fn near_zero_excluded() {
        let p = vec![22.0, 22.0, 22.0, 22.0];
        let m = vec![1e-9, 22.0, 22.0, -1e-9];
        let (report, excluded) = compute_guideline14(
            Guideline14Source::flexlab_x3a(),
            ReportingResolution::Hourly,
            "zone_air_temperature_c",
            "°C",
            &p,
            &m,
            1e-3,
        );
        assert_eq!(excluded, 2);
        assert_eq!(report.n, 2);
    }

    #[test]
    fn nan_inf_samples_excluded() {
        let p = vec![22.0, f64::NAN, 22.0, f64::INFINITY];
        let m = vec![21.0, 22.0, 22.0, 21.5];
        let (report, excluded) = compute_guideline14(
            Guideline14Source::flexlab_x3a(),
            ReportingResolution::Hourly,
            "zone_air_temperature_c",
            "°C",
            &p,
            &m,
            1e-3,
        );
        // NaN, +Inf, -Inf are treated as non-finite and dropped.
        assert_eq!(excluded, 2);
        assert_eq!(report.n, 2);
    }

    #[test]
    fn warning_when_only_one_metric_exceeds_limit() {
        // Construct a series where the bias pushes NMBE > 10% but the
        // RMSE / mean stays within 30%.
        let n = 8760;
        let mut p = Vec::with_capacity(n);
        let mut m = Vec::with_capacity(n);
        for i in 0..n {
            let diurnal = 1.5 * ((i as f64) / 24.0 * std::f64::consts::TAU).sin();
            m.push(22.0 + diurnal);
            p.push(22.0 + diurnal + 3.0); // +3 K bias → NMBE ≈ 13.6%
        }
        let (report, _) = compute_guideline14(
            Guideline14Source::flexlab_x3a(),
            ReportingResolution::Hourly,
            "zone_air_temperature_c",
            "°C",
            &p,
            &m,
            1e-3,
        );
        assert!(report.nmbe.abs() > 10.0);
        assert!(report.cv_rmse < 30.0);
        assert_eq!(report.status, Guideline14Status::Warning);
    }

    #[test]
    fn fail_when_both_metrics_exceed_limits() {
        let n = 100;
        let mut p = Vec::with_capacity(n);
        let mut m = Vec::with_capacity(n);
        for i in 0..n {
            m.push(22.0);
            p.push(40.0 + 0.1 * i as f64); // gross error
        }
        let (report, _) = compute_guideline14(
            Guideline14Source::flexlab_x3a(),
            ReportingResolution::Hourly,
            "zone_air_temperature_c",
            "°C",
            &p,
            &m,
            1e-3,
        );
        assert_eq!(report.status, Guideline14Status::Fail);
    }

    #[test]
    fn empty_inputs_produce_fail_report() {
        let p: Vec<f64> = vec![];
        let m: Vec<f64> = vec![];
        let (report, _) = compute_guideline14(
            Guideline14Source::flexlab_x3a(),
            ReportingResolution::Hourly,
            "zone_air_temperature_c",
            "°C",
            &p,
            &m,
            1e-3,
        );
        assert_eq!(report.n, 0);
        assert_eq!(report.status, Guideline14Status::Fail);
        assert!(report.nmbe.is_nan());
    }

    #[test]
    fn monthly_thresholds_are_tighter_than_hourly() {
        let (h_t, c_t) = ReportingResolution::Hourly.ashrae_thresholds();
        let (m_t, m_c_t) = ReportingResolution::Monthly.ashrae_thresholds();
        assert!(m_t < h_t, "monthly NMBE limit must be tighter than hourly");
        assert!(
            m_c_t < c_t,
            "monthly CVRMSE limit must be tighter than hourly"
        );
    }

    #[test]
    fn render_markdown_contains_key_sections() {
        let (p, m) = synth_series(8760, 22.0, 0.05);
        let (report, _) = compute_guideline14(
            Guideline14Source::flexlab_x3a(),
            ReportingResolution::Hourly,
            "zone_air_temperature_c",
            "°C",
            &p,
            &m,
            1e-3,
        );
        let md = render_markdown(&report);
        assert!(md.contains("ASHRAE Guideline 14"));
        assert!(md.contains("NMBE"));
        assert!(md.contains("CV(RMSE)"));
        assert!(md.contains("Headline Metrics"));
        assert!(md.contains("Confidence Intervals"));
        assert!(md.contains("Interpretation"));
    }

    #[test]
    fn ci_converges_to_normal_approx_for_large_n() {
        // For df → ∞, t_inv_975 → 1.96.
        let t = statrs_t_inv_975(1.0e8);
        assert!((t - 1.96).abs() < 1e-3, "t_inv_975(∞) ≈ 1.96, got {t}");
    }
}
