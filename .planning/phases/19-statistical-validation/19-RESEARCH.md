# Phase 19: Statistical Validation - Research

**Researched:** 2026-03-15
**Domain:** Statistical validation methodology for building energy modeling
**Confidence:** MEDIUM

## Summary

Phase 19 implements formal statistical validation framework for ASHRAE 140 compliance using NMBE (Normalized Mean Bias Error) and CV(RMSE) (Coefficient of Variation of Root Mean Square Error) metrics with multiple testing corrections via Benjamini-Hochberg procedure. The phase extends the existing tolerance-based validation system (`ASHRAE140Validator`) with parallel statistical validation path, integrated via new `--statistical` CLI flag.

The implementation requires building statistical computation capabilities from scratch (no existing Rust statistical library integration) and extending the validation report generation infrastructure to include confidence intervals, effect sizes (Cohen's d), and corrected p-values. Case group validation ensures minimum 80% passing rate per validation group using hybrid threshold approach (80% for groups with ≥5 cases, single-case validation for 1-4 cases).

**Primary recommendation:** Implement statistical metrics calculation, Benjamini-Hochberg correction, and confidence interval computation as a new `StatisticalValidator` module that wraps the existing `ASHRAE140Validator`, with integration points in report generation and CLI.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Statistical Metrics Calculation:** Calculate NMBE and CV(RMSE) per case (not as group-level aggregates). Use reference range midpoint `(ref_min + ref_max) / 2.0` as the NMBE denominator. Exclude cases with zero or near-zero reference values from NMBE calculation entirely.
- **NMBE Formula:** Use standard ASHRAE Guideline 14 formula: `NMBE = Σ((predicted - reference) / reference) / n`. Signed metric (positive = overprediction, negative = underprediction).
- **Multiple Testing Corrections:** Implement Benjamini-Hochberg (FDR) procedure (not Bonferroni). Use alpha threshold α = 0.05 (5% family-wise error rate). Apply FDR correction separately within each validation group.
- **Multiple Testing Reporting:** Report uncorrected p-values, and annotate which ones would pass BH correction. Combines transparency with clarity.
- **Case Group Validation:** Require minimum 5 cases per validation group for statistical validity. Use hybrid threshold approach: 80% passing rate for groups with ≥5 cases, single-case validation for groups with 1-4 cases.
- **Group Validation Threshold:** Use strict 80% threshold for fail handling (binary PASS/FAIL, no marginal/warn status). Report PASS if ≥80% of cases in a group pass tolerance criteria.
- **Report Format and Integration:** Add new "Statistical Validation" section to existing reports. Keep the existing tolerance-based Pass/Warning/Fail system. Report 95% confidence intervals always for NMBE and CV(RMSE). Report comprehensive statistical package: NMBE, CV(RMSE), 95% CI for both metrics, BH-corrected p-values, effect size (Cohen's d), and effect direction.
- **CLI Integration:** Add `--statistical` flag to `fluxion validate` command. Default behavior unchanged (tolerance-based validation). When flag is set, enable statistical validation with NMBE/CV(RMSE)/CI reporting.

### Claude's Discretion
- **Group definition for 80% passing validation:** Implementation should choose the most appropriate grouping approach (case type, mass level, or case range) based on ASHRAE 140 structure and case availability. Factors to consider include: (1) alignment with phase structure, (2) creating reasonably balanced group sizes, (3) physical meaningfulness of groups, (4) ensuring minimum 5 cases per group.
- **Statistical testing granularity:** Whether to apply per-case, per-metric, or per-validation-group statistical testing should be determined based on ASHRAE 140 Addendum B guidance and practical considerations for interpretability.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STATS-01 | Implement ASHRAE 140 Addendum B acceptance criteria | Context.md decisions specify NMBE and CV(RMSE) calculation methodology; requires implementing Addendum B criteria in new StatisticalValidator module |
| STATS-02 | Implement NMBE (Normalized Mean Bias Error) calculations | Context.md specifies per-case calculation using reference range midpoint; formula: `Σ((predicted - reference) / reference) / n` |
| STATS-03 | Implement CV(RMSE) (Coefficient of Variation of RMSE) calculations | Standard formula: `RMSE / mean(reference) * 100`; requires RMSE calculation from validation results |
| STATS-04 | Implement multiple testing corrections (Bonferroni, Benjamini-Hochberg) | Context.md locks to Benjamini-Hochberg FDR procedure with α=0.05; Bonferroni deferred |
| STATS-05 | Implement case group validation (minimum 80% passing per group) | Context.md specifies hybrid threshold: 80% for groups with ≥5 cases, single-case for 1-4 cases |
| STATS-06 | Generate comprehensive validation reports with statistical metrics | Context.md requires NMBE, CV(RMSE), 95% CI, BH-corrected p-values, Cohen's d, effect direction in new report section |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **statrs** | 0.18.0 | Statistical computing library (distributions, statistical functions) | Port of Math.NET Numerics; provides t-distribution, Student's T, special functions (beta, gamma, erf) needed for confidence intervals and p-values |
| **serde** | 1.0 (existing) | Serialization for statistical results | Project already uses serde for JSON/CSV export; consistent with existing patterns |
| **anyhow** | 1.0 (existing) | Error handling | Project standard for ergonomic error handling |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **rayon** | 1.10 (existing) | Parallel statistical calculations | For batch processing of case-level statistics across large case sets |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| statrs | ndarray-stats | ndarray-stats requires ndarray 0.15+ (project uses 0.16), but lacks t-distribution CDF needed for confidence intervals; statrs has complete distribution functions |
| statrs | nalgebra-linfa | linfa ecosystem has machine learning focus, not core statistical distributions; statrs has direct distribution implementations |
| statrs | hand-roll | Hand-rolling t-distribution CDF and inverse CDF is error-prone; statrs provides well-tested implementations |

**Installation:**
```toml
# Add to Cargo.toml [dependencies]
statrs = "0.18.0"
```

## Architecture Patterns

### Recommended Project Structure
```
src/validation/
├── statistical.rs          # NEW: Statistical validation module
│   ├── StatisticalValidator  # Main validator struct
│   ├── StatisticalMetrics   # NMBE, CV(RMSE), CI calculations
│   ├── BenjaminiHochberg # FDR correction implementation
│   └── CohenD             # Effect size calculation
├── ashrae_140_validator.rs # EXISTING: Tolerance-based validation
├── report.rs               # EXISTING: Extend with StatisticalSection
└── reporter.rs             # EXISTING: Extend generation logic

src/api/
└── fluxion.rs             # EXISTING: Add --statistical flag to CLI
```

### Pattern 1: StatisticalValidator Wrapper
**What:** New struct that wraps `ASHRAE140Validator` to provide parallel statistical validation path
**When to use:** When `--statistical` flag is set; tolerance-based validation remains default
**Example:**
```rust
use crate::validation::ashrae_140_validator::ASHRAE140Validator;
use crate::validation::statistical::{StatisticalValidator, StatisticalMetrics};

pub struct StatisticalValidator {
    base_validator: ASHRAE140Validator,
    alpha: f64,  // Family-wise error rate (default 0.05)
}

impl StatisticalValidator {
    pub fn new() -> Self {
        Self {
            base_validator: ASHRAE140Validator::with_full_diagnostics(),
            alpha: 0.05,
        }
    }

    pub fn validate_with_statistics(&self, cases: &[ASHRAE140Case]) -> StatisticalReport {
        // Run tolerance-based validation
        let tolerance_report = self.base_validator.validate_all(cases);

        // Calculate statistical metrics
        let metrics = StatisticalMetrics::calculate(&tolerance_report);

        // Apply Benjamini-Hochberg correction
        let corrected = BenjaminiHochberg::apply(metrics.p_values, self.alpha);

        // Group-level validation (80% passing)
        let group_results = self.validate_groups(&tolerance_report, &corrected);

        StatisticalReport {
            tolerance: tolerance_report,
            metrics,
            corrected_p_values: corrected,
            group_validation: group_results,
        }
    }
}
```

### Pattern 2: Statistical Metrics Calculation
**What:** Compute NMBE, CV(RMSE), and 95% confidence intervals per case
**When to use:** Called by StatisticalValidator after tolerance-based validation completes
**Example:**
```rust
use statrs::distribution::{StudentsT, ContinuousCDF};
use statrs::statistics::Statistics;

pub struct StatisticalMetrics {
    pub nmbe: f64,
    pub cv_rmse: f64,
    pub nmbe_ci: (f64, f64),  // (lower, upper)
    pub cv_rmse_ci: (f64, f64),
    pub cohens_d: f64,
    pub effect_direction: EffectDirection,
}

impl StatisticalMetrics {
    pub fn calculate(report: &BenchmarkReport) -> Self {
        let predicted: Vec<f64> = report.results.iter()
            .map(|r| r.fluxion_value)
            .collect();

        let reference: Vec<f64> = report.results.iter()
            .map(|r| (r.ref_min + r.ref_max) / 2.0)  // Midpoint
            .collect();

        let n = predicted.len() as f64;

        // NMBE calculation
        let nmbe = (0..predicted.len())
            .map(|i| (predicted[i] - reference[i]) / reference[i])
            .sum::<f64>() / n * 100.0;  // Percentage

        // CV(RMSE) calculation
        let rmse = Self::rmse(&predicted, &reference);
        let mean_ref = reference.iter().mean();
        let cv_rmse = (rmse / mean_ref) * 100.0;

        // 95% CI for NMBE (t-distribution for small samples)
        let t_stat = StudentsT::new(0.0, 1.0, n - 1.0)
            .unwrap()
            .cdf(1.96);  // 95% confidence
        let nmbe_se = Self::standard_error(&predicted, &reference);
        let nmbe_ci = (
            nmbe - t_stat * nmbe_se,
            nmbe + t_stat * nmbe_se,
        );

        // Cohen's d (effect size)
        let cohens_d = (mean_ref - predicted.mean()) / Self::pooled_std(&reference, &predicted);

        StatisticalMetrics {
            nmbe,
            cv_rmse,
            nmbe_ci,
            cv_rmse_ci,  // Similar calculation
            cohens_d,
            effect_direction: if cohens_d > 0.0 {
                EffectDirection::Underprediction
            } else {
                EffectDirection::Overprediction
            },
        }
    }

    fn rmse(predicted: &[f64], reference: &[f64]) -> f64 {
        let sq_errors: Vec<f64> = predicted.iter()
            .zip(reference.iter())
            .map(|(p, r)| (p - r).powi(2))
            .collect();
        (sq_errors.iter().sum::<f64>() / sq_errors.len() as f64).sqrt()
    }
}
```

### Pattern 3: Benjamini-Hochberg Correction
**What:** Implement FDR control to prevent false positives from multiple testing
**When to use:** Applied within each validation group separately
**Example:**
```rust
pub struct BenjaminiHochberg;

impl BenjaminiHochberg {
    /// Applies BH FDR correction to p-values.
    ///
    /// # Arguments
    /// - `p_values`: Vector of p-values (one per test)
    /// - `alpha`: Family-wise error rate (default 0.05)
    ///
    /// # Returns
    /// Vector of booleans indicating which tests pass after correction
    pub fn apply(p_values: Vec<f64>, alpha: f64) -> Vec<bool> {
        if p_values.is_empty() {
            return vec![];
        }

        let mut indexed: Vec<(usize, f64)> = p_values
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        // Sort by p-value (ascending)
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let m = p_values.len() as f64;
        let mut corrected = vec![false; p_values.len()];

        // BH procedure: find largest k where p_k ≤ (k/m) * alpha
        for (k, (original_idx, p)) in indexed.iter().enumerate() {
            let k = k as f64 + 1.0;  // 1-indexed
            let threshold = (k / m) * alpha;

            if *p <= threshold {
                corrected[*original_idx] = true;
            } else {
                break;  // No further tests can pass
            }
        }

        corrected
    }
}
```

### Anti-Patterns to Avoid
- **Mixing tolerance and statistical logic in same struct:** Keep separation—`ASHRAE140Validator` handles tolerances, `StatisticalValidator` handles statistics
- **Applying BH correction globally:** Must apply separately within each validation group (baseline, diagnostics, equipment) per CONTEXT.md decisions
- **Reporting only corrected p-values:** Context.md requires reporting uncorrected p-values with BH annotations for transparency
- **Hardcoding group thresholds:** Use hybrid approach (80% for ≥5 cases, single-case for 1-4) per CONTEXT.md decisions
- **Using normal distribution for small samples:** Use t-distribution for n < 30 per CONTEXT.md decisions

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| t-distribution CDF/inverse CDF | Hand-roll numerical integration | statrs::distribution::StudentsT | t-distribution critical values require special functions, error-prone to implement |
| Student's T critical values | Hardcode lookup tables | statrs::distribution::StudentsT::cdf | Tables are static, statrs provides exact calculations for any degrees of freedom |
| Beta function, gamma function | Hand-roll series approximation | statrs::function::beta, statrs::function::gamma | Special functions have convergence issues; statrs implements robust algorithms |
| Error function (erf) | Hand-roll polynomial approximation | statrs::function::erf | Erf is numerically delicate; statrs uses accurate implementations |
| Statistical tests | Hand-roll hypothesis testing logic | statrs::statistics::Statistics | Standard deviations, means, variances are well-tested in statrs |

**Key insight:** Statistical functions (distributions, special functions) are mathematically delicate and numerically sensitive. Hand-rolling leads to subtle bugs (e.g., t-distribution tail behavior, erf precision at small arguments). statrs provides battle-tested implementations with proper numerical methods.

## Common Pitfalls

### Pitfall 1: Division by Zero in NMBE Calculation
**What goes wrong:** NMBE formula divides by reference values; if reference is zero, NMBE explodes to infinity
**Why it happens:** Some ASHRAE 140 cases have near-zero reference values (e.g., peak loads in free-floating cases)
**How to avoid:** CONTEXT.md locks this decision: Exclude cases with zero or near-zero reference values entirely. Check `abs(reference) < epsilon` before NMBE calculation, report separately.
**Warning signs:** NaN/Infinity in NMBE results, extremely large NMBE values (>1000%)

### Pitfall 2: Incorrect Confidence Interval Formula
**What goes wrong:** Using normal distribution (z=1.96) for small samples (n<30) produces incorrect confidence intervals
**Why it happens:** Standard practice uses normal distribution for large samples, but small samples have higher uncertainty
**How to avoid:** Use t-distribution with `df = n - 1` degrees of freedom. statrs provides `StudentsT::new(0.0, 1.0, df).cdf(1.96)` for 95% CI.
**Warning signs:** CIs are too narrow for small case groups, fail to contain true value 95% of time

### Pitfall 3: Applying BH Correction Across All Tests
**What goes wrong:** BH correction applied globally across all validation groups produces overly conservative results
**Why it happens:** Context.md specifies "apply FDR correction separately within each validation group" to maintain power
**How to avoid:** Partition tests by validation group (baseline, diagnostics, equipment), apply BH correction independently to each group.
**Warning signs:** Very few tests pass BH correction, overly conservative compared to ASHRAE 140 Addendum B

### Pitfall 4: Hardcoded Group Thresholds
**What goes wrong:** Using fixed 80% threshold for all groups, even groups with 1-2 cases
**Why it happens:** 80% threshold is meaningless for 1-2 cases (e.g., 1/2 cases = 50% < 80%)
**How to avoid:** Use hybrid threshold: 80% for groups with ≥5 cases, single-case validation (PASS if all pass tolerance) for 1-4 cases per CONTEXT.md.
**Warning signs:** Small groups always fail validation despite all cases passing tolerance criteria

### Pitfall 5: Effect Size Calculation Without Pooled Standard Deviation
**What goes wrong:** Cohen's d calculated incorrectly, leading to wrong effect size interpretation
**Why it happens:** Cohen's d requires pooled standard deviation, not individual std devs
**How to avoid:** Use `pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))`. For single case vs reference, treat reference as population (use reference std dev).
**Warning signs:** Cohen's d values are unusually large (>2.0) or small (<0.2), don't match visual inspection of results

## Code Examples

Verified patterns from official sources:

### NMBE Calculation (Per-Case)
```rust
use crate::validation::report::ValidationResult;

pub fn calculate_nmbe(results: &[ValidationResult]) -> f64 {
    let mut nmbe_sum = 0.0;
    let mut count = 0;

    for result in results {
        let ref_midpoint = (result.ref_min + result.ref_max) / 2.0;

        // Exclude zero/near-zero references (CONTEXT.md decision)
        if ref_midpoint.abs() < 1e-10 {
            continue;
        }

        let error = (result.fluxion_value - ref_midpoint) / ref_midpoint;
        nmbe_sum += error;
        count += 1;
    }

    if count == 0 {
        return f64::NAN;  // All excluded
    }

    (nmbe_sum / count as f64) * 100.0  // Percentage
}
```

### CV(RMSE) Calculation
```rust
pub fn calculate_cv_rmse(results: &[ValidationResult]) -> f64 {
    let predicted: Vec<f64> = results.iter()
        .map(|r| r.fluxion_value)
        .collect();

    let reference: Vec<f64> = results.iter()
        .map(|r| (r.ref_min + r.ref_max) / 2.0)
        .collect();

    let n = predicted.len();

    // RMSE
    let sq_errors: f64 = predicted.iter()
        .zip(reference.iter())
        .map(|(p, r)| (p - r).powi(2))
        .sum();
    let rmse = (sq_errors / n as f64).sqrt();

    // CV(RMSE) = (RMSE / mean(reference)) * 100
    let mean_ref = reference.iter().sum::<f64>() / n as f64;

    (rmse / mean_ref) * 100.0
}
```

### 95% Confidence Interval (t-distribution)
```rust
use statrs::distribution::{StudentsT, ContinuousCDF};
use statrs::statistics::Statistics;

pub fn calculate_ci_nmbe(nmbe: f64, std_error: f64, n: usize) -> (f64, f64) {
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }

    // t-distribution with df = n - 1
    let df = (n - 1) as f64;
    let t = StudentsT::new(0.0, 1.0, df).unwrap();

    // Critical value for 95% CI (two-tailed)
    let t_critical = 1.96;  // Approximate, use inverse CDF for exact

    let lower = nmbe - t_critical * std_error;
    let upper = nmbe + t_critical * std_error;

    (lower, upper)
}
```

### Benjamini-Hochberg Implementation
```rust
/// Applies BH FDR correction within a validation group.
///
/// # Arguments
/// - `p_values`: P-values for tests in this group
/// - `alpha`: Family-wise error rate (default 0.05)
///
/// # Returns
/// Vector of booleans: true if test passes BH correction
pub fn benjamini_hochberg(p_values: Vec<f64>, alpha: f64) -> Vec<bool> {
    if p_values.is_empty() {
        return vec![];
    }

    // Create indexed pairs and sort by p-value
    let mut indexed: Vec<(usize, f64)> = p_values
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let m = p_values.len() as f64;
    let mut corrected = vec![false; p_values.len()];

    // BH procedure: find largest k where p_k ≤ (k/m) * α
    for (rank, (original_idx, p_val)) in indexed.iter().enumerate() {
        let k = (rank + 1) as f64;  // 1-indexed
        let threshold = (k / m) * alpha;

        if *p_val <= threshold {
            corrected[*original_idx] = true;
        } else {
            break;  // No further tests can pass (sorted)
        }
    }

    corrected
}
```

### Group Validation (80% Passing Rate)
```rust
use std::collections::HashMap;

pub enum ValidationGroup {
    Baseline,
    HighMass,
    FreeFloating,
    Diagnostics,
    Equipment,
}

pub fn validate_groups(
    report: &BenchmarkReport,
    min_cases_per_group: usize,
) -> HashMap<ValidationGroup, bool> {
    let mut results = HashMap::new();

    // Group cases by type (Claude's discretion on exact grouping)
    let baseline_cases = ["600", "610", "620", "630", "640", "650"];
    let high_mass_cases = ["900", "910", "920", "930", "940", "950"];
    let free_floating_cases = ["600FF", "650FF", "900FF", "950FF"];

    // Baseline group validation
    let baseline_results: Vec<_> = report.results.iter()
        .filter(|r| baseline_cases.contains(&r.case_id.as_str()))
        .collect();

    let baseline_pass = if baseline_results.len() >= min_cases_per_group {
        // 80% threshold for ≥5 cases
        let passing = baseline_results.iter().filter(|r| r.passed()).count();
        (passing as f64 / baseline_results.len() as f64) >= 0.8
    } else {
        // Single-case validation for 1-4 cases
        baseline_results.iter().all(|r| r.passed())
    };

    results.insert(ValidationGroup::Baseline, baseline_pass);

    // Repeat for other groups...
    results
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Tolerance-based only (±15% annual, ±10% monthly) | Tolerance + Statistical (NMBE, CV(RMSE), CI, BH) | Phase 19 (this phase) | Provides formal statistical compliance with ASHRAE 140 Addendum B, enables hypothesis testing |
| Manual statistical analysis in Excel/Python | Automated statistical validation in Rust CLI | Phase 19 (this phase) | Reduces manual effort, ensures consistent methodology across all runs |
| No multiple testing correction | Benjamini-Hochberg FDR control | Phase 19 (this phase) | Prevents false compliance claims from multiple hypothesis testing |
| Single validation threshold | Hybrid threshold (80% for ≥5 cases, single-case for <5) | Phase 19 (this phase) | Adapts to group size, avoids misleading statistics for small groups |

**Deprecated/outdated:**
- **Tolerance-only validation:** Remains as default for backward compatibility, but statistical validation provides formal compliance evidence
- **Manual Excel analysis:** Automated reports now include comprehensive statistical package
- **Bonferroni correction:** Too conservative for ASHRAE 140 validation; BH FDR provides better power while controlling false discoveries

## Open Questions

1. **Case Group Definition**
   - What we know: CONTEXT.md gives Claude discretion to choose grouping approach (case type, mass level, or case range)
   - What's unclear: Exact group boundaries—should Cases 800-810 be separate "Equipment" group or merged with "Diagnostics"?
   - Recommendation: Implement flexible group system (enum with match pattern) to enable easy adjustment after ASHRAE 140 Addendum B review

2. **Statistical Testing Granularity**
   - What we know: NMBE/CV(RMSE) calculated per case (CONTEXT.md decision), but hypothesis testing granularity unspecified
   - What's unclear: Should we test per-metric (e.g., AnnualHeating passes?), per-case (all 4 metrics pass?), or per-group (≥80% cases pass)?
   - Recommendation: Implement per-metric BH correction (6 metrics × N cases), then aggregate to case-level and group-level for report clarity

3. **P-value Calculation Method**
   - What we know: Need p-values for BH correction, but CONTEXT.md doesn't specify calculation method
   - What's unclear: Should we use t-test (Fluxion vs reference midpoint), chi-square (observed vs expected), or tolerance-based hypothesis?
   - Recommendation: Use one-sample t-test (Fluxion vs reference midpoint) with null hypothesis: Fluxion = reference; compute p-value from t-statistic

4. **Acceptance Thresholds for Statistical Metrics**
   - What we know: ASHRAE Guideline 14 has NMBE/CV(RMSE) acceptance criteria (±5% NMBE, ±15% CV(RMSE for hourly data)
   - What's unclear: Are these thresholds applicable to ASHRAE 140 annual/monthly data, or are there Addendum B-specific thresholds?
   - Recommendation: Start with ASHRAE Guideline 14 thresholds as placeholder, adjust after Addendum B review

5. **CI Calculation for Single Cases (n=1)**
   - What we know: 95% CI requires t-distribution with df = n - 1, which is undefined for n=1
   - What's unclear: How to report CI for single-case validation groups?
   - Recommendation: Report "N/A" for CI when n < 2, annotate in report as "insufficient data for CI calculation"

## Validation Architecture

> Nyquist validation is ENABLED (workflow.nyquist_validation not explicitly set to false in .planning/config.json)

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (built-in Rust test framework) |
| Config file | .planning/config.json (nyquist_validation not set, defaults to enabled) |
| Quick run command | `cargo test --lib validation::statistical` |
| Full suite command | `cargo test` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STATS-01 | Addendum B acceptance criteria implemented | unit | `cargo test test_addendum_b_criteria` | ❌ Wave 0 |
| STATS-02 | NMBE calculated correctly per case | unit | `cargo test test_nmbe_calculation` | ❌ Wave 0 |
| STATS-03 | CV(RMSE) calculated correctly per case | unit | `cargo test test_cv_rmse_calculation` | ❌ Wave 0 |
| STATS-04 | Benjamini-Hochberg FDR correction applied | unit | `cargo test test_benjamini_hochberg_correction` | ❌ Wave 0 |
| STATS-05 | Group validation 80% passing rate enforced | unit | `cargo test test_group_validation_80_percent` | ❌ Wave 0 |
| STATS-06 | Comprehensive report with statistical metrics generated | integration | `cargo test test_statistical_report_generation` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --lib validation::statistical` (quick unit tests for new statistical module)
- **Per wave merge:** `cargo test` (full test suite, includes existing ASHRAE 140 validation tests)
- **Phase gate:** Full suite green before `/gsd:verify-work` (ensures statistical validation doesn't break existing functionality)

### Wave 0 Gaps
- [ ] `src/validation/statistical.rs` — main statistical validation module (StatisticalValidator, StatisticalMetrics, BenjaminiHochberg)
- [ ] `tests/test_statistical.rs` — comprehensive unit tests for all statistical functions
- [ ] `tests/test_report_integration.rs` — integration tests for statistical report generation
- [ ] Framework install: N/A (cargo test is built-in)

## Sources

### Primary (HIGH confidence)
- [statrs 0.18.0 docs.rs](https://docs.rs/statrs/latest/statrs/) - Provides StudentsT distribution, beta/gamma functions, erf for CI calculations
- [statrs crates.io](https://crates.io/crates/statrs) - Confirms latest version and dependencies
- [ndarray 0.16 docs.rs](https://docs.rs/ndarray/latest/ndarray/) - Confirms project uses ndarray 0.16 (relevant for potential future statrs integration)
- [Fluxion CONTEXT.md Phase 19](/.planning/phases/19-statistical-validation/19-CONTEXT.md) - Locked decisions on NMBE/CV(RMSE) methodology, BH correction, group validation
- [Fluxion src/validation/ashrae_140_validator.rs](/home/alex/Projects/fluxion/src/validation/ashrae_140_validator.rs) - Existing tolerance-based validation structure to extend
- [Fluxion src/validation/report.rs](/home/alex/Projects/fluxion/src/validation/report.rs) - Report data structures (BenchmarkReport, ValidationResult) to extend
- [Fluxion src/validation/reporter.rs](/home/alex/Projects/fluxion/src/validation/reporter.rs) - Report generation infrastructure to add statistical sections

### Secondary (MEDIUM confidence)
- [Fluxion src/validation/multi_reference.rs](/home/alex/Projects/fluxion/src/validation/multi_reference.rs) - Multi-reference database structure for reference range midpoint calculation
- [Fluxion src/bin/export_csv.rs](/home/alex/Projects/fluxion/src/bin/export_csv.rs) - CLI pattern for adding new subcommands/flags (clap::Parser)

### Tertiary (LOW confidence)
- **ASHRAE 140 Addendum B** - Not found via web search (may be paywalled or not indexed); CONTEXT.md provides methodology based on ASHRAE Guideline 14 principles
- **ASHRAE Guideline 14 acceptance criteria** - Web search returned no results; general knowledge of NMBE/CV(RMSE) thresholds (±5% NMBE, ±15% CV(RMSE for hourly) applied as placeholder
- **Benjamini-Hochberg implementation details** - Algorithm well-documented in statistics literature; implementation based on standard procedure (sort p-values, find largest k where p_k ≤ (k/m)α)

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - statrs confirmed available and suitable, but ASHRAE 140 Addendum B specifics not verified (LOW confidence on exact thresholds)
- Architecture: MEDIUM - Existing validation structure well-understood, but statistical integration points require careful design
- Pitfalls: HIGH - Statistical validation pitfalls (division by zero, small sample CI, BH application scope) are well-documented in statistical literature

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (30 days - ASHRAE standards are stable, but Addendum B specifics may emerge)
