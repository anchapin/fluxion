# Phase 19: Statistical Validation - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement formal statistical validation framework with ASHRAE 140 Addendum B compliance. This phase delivers:

- NMBE (Normalized Mean Bias Error) and CV(RMSE) (Coefficient of Variation of RMSE) calculations
- Multiple testing corrections using Benjamini-Hochberg (FDR) procedure to prevent false compliance claims
- Case group validation ensuring minimum 80% passing rate per validation group
- Comprehensive validation reports with 95% confidence intervals, BH-corrected p-values, effect sizes (Cohen's d), and compliance determinations
- Integration with existing ASHRAE140Validator tolerance-based system via CLI flag

The scope is statistical validation methodology — not new physics modeling, HVAC systems, or validation cases (those are delivered in prior phases).

</domain>

<decisions>
## Implementation Decisions

### Statistical Metrics Calculation
- Calculate NMBE and CV(RMSE) per case (not as group-level aggregates). This provides per-case diagnostics and identifies outliers. More computationally intensive but most informative.
- Use reference range midpoint `(ref_min + ref_max) / 2.0` as the NMBE denominator. This handles ASHRAE 140 reference ranges correctly (e.g., [1.17, 2.04] MWh → 1.605 MWh).
- Exclude cases with zero or near-zero reference values from NMBE calculation entirely. Report them separately with a note about why they were excluded. Most conservative but clearest approach.
- Use standard ASHRAE Guideline 14 NMBE formula: `NMBE = Σ((predicted - reference) / reference) / n`. Signed metric (positive = overprediction, negative = underprediction).

### Multiple Testing Corrections
- Implement Benjamini-Hochberg (FDR) procedure (not Bonferroni). Controls false discovery rate while maintaining higher power. Less conservative than Bonferroni, more suitable for larger test sets. This is the default for ASHRAE 140 Addendum B guidance.
- Use alpha threshold α = 0.05 (5% family-wise error rate). This is the recommended threshold for ASHRAE 140 compliance. Controls for 5% chance of any false positive across all tests.
- Apply FDR correction separately within each validation group (baseline, diagnostics, equipment). Treats each group as independent family of tests.
- Report uncorrected p-values, and annotate which ones would pass BH correction. Combines transparency with clarity. Shows correction decisions without doubling p-value columns.

### Case Group Validation
- **Group definition:** Claude's discretion on grouping approach (case type, mass level, or case range) to be determined during implementation based on ASHRAE 140 structure and case availability.
- Require minimum 5 cases per validation group for statistical validity. Too small groups (1-2 cases) can't provide meaningful 80% passing rates. 5 is a reasonable lower bound for statistical inference.
- Use hybrid threshold approach: apply 80% passing rate validation for groups with ≥5 cases, use single-case validation for groups with 1-4 cases. Adapts to group size while avoiding misleading statistics.
- Use strict 80% threshold for fail handling (binary PASS/FAIL, no marginal/warn status). Report PASS if ≥80% of cases in a group pass tolerance criteria, otherwise FAIL. Simple and clear for compliance determination.

### Report Format and Integration
- Add new "Statistical Validation" section to existing reports. Keep the existing tolerance-based Pass/Warning/Fail system. Users see both statistical metrics and compliance status. Most comprehensive approach.
- Report 95% confidence intervals always for NMBE and CV(RMSE). Uses t-distribution for small samples (n<30) or normal distribution for larger samples. Provides uncertainty quantification and is standard in scientific reporting.
- Report comprehensive statistical package: NMBE, CV(RMSE), 95% CI for both metrics, BH-corrected p-values, effect size (Cohen's d), and effect direction (over/under prediction). Most thorough for scientific analysis.
- Add `--statistical` flag to `fluxion validate` command. Default behavior unchanged (tolerance-based validation). When flag is set, enable statistical validation with NMBE/CV(RMSE)/CI reporting. Opt-in approach, no breaking changes.

### Claude's Discretion

- **Group definition for 80% passing validation:** Implementation should choose the most appropriate grouping approach (case type, mass level, or case range) based on ASHRAE 140 structure and case availability. Factors to consider include: (1) alignment with phase structure, (2) creating reasonably balanced group sizes, (3) physical meaningfulness of groups, (4) ensuring minimum 5 cases per group.
- **Statistical testing granularity:** Whether to apply per-case, per-metric, or per-validation-group statistical testing should be determined based on ASHRAE 140 Addendum B guidance and practical considerations for interpretability.

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for statistical validation in ASHRAE 140 compliance.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- **ASHRAE140Validator** (`src/validation/ashrae_140_validator.rs`) - Current tolerance-based validation with Pass/Warning/Fail determination. Has diagnostic configuration and diagnostic collector infrastructure that can be extended for statistical metrics.
- **MetricType** enum (`src/validation/report.rs`) - Defines validation metrics (AnnualHeating, AnnualCooling, PeakHeating, PeakCooling, MinFreeFloat, MaxFreeFloat). Should be extended with new statistical metric variants.
- **ValidationReportGenerator** (`src/validation/reporter.rs`) - Report generation infrastructure that produces Markdown reports. Can be extended to add statistical validation sections.
- **BenchmarkReport** struct (`src/validation/report.rs`) - Contains validation results data structure. Should be extended to include statistical metrics.

### Established Patterns
- **Validation data structures:** Uses structs with serde serialization for JSON/CSV export. Statistical metrics should follow this pattern.
- **Report generation:** Reports built by appending strings, then writing to files. Statistical validation sections should follow this Markdown formatting approach.
- **CLI pattern:** `fluxion validate` subcommand loads case data, runs validation, generates reports. Adding `--statistical` flag follows existing CLI pattern from Phase 7 (multi-reference integration).
- **Tolerance-based validation:** Current system uses `compute_status(value, ref_min, ref_max)` for Pass/Warning/Fail. Statistical validation will add a parallel determination path (NMBE/CV(RMSE) + hypothesis testing).

### Integration Points
- **ASHRAE140Validator:** Statistical metrics should be calculated as an extension to the existing validation workflow. Add methods to `ASHRAE140Validator` or create a new `StatisticalValidator` struct that wraps or extends the current validator.
- **Report generation:** Statistical validation results should be integrated into `ValidationReportGenerator` to appear in the same Markdown reports (e.g., add "## Statistical Validation" section to `ASHRAE140_RESULTS.md`).
- **CLI interface:** Add `--statistical` flag to the `validate` subcommand. Should check environment or config for default behavior.
- **MetricType enum:** Extend with new statistical metrics if needed for report display, or keep statistical metrics separate from tolerance-based metrics.

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 19-statistical-validation*
*Context gathered: 2026-03-15*
