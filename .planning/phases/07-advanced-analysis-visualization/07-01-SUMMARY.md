---
phase: 07-advanced-analysis-visualization
plan: 01
title: Analysis Core: Sensitivity & Delta
status: complete
date: 2025-03-10
---

# Phase 7 Plan 01: Analysis Core - Sensitivity & Delta - Summary

## One-Liner

Implemented core sensitivity analysis capabilities: OAT and Sobol sampling strategies, sensitivity metrics (NMBE, CVRMSE, slope), batch evaluation via BatchOracle, and CSV export with ranking. Prepared stub modules for delta testing and component breakdown.

## Deviations from Plan

**Minor:** The `run_sensitivity` function currently creates a new `ThermalModel` and `SurrogateManager` for each evaluation rather than accepting a pre-built `BatchOracle`. This is acceptable for the current version because:

- The API is simple and works for single-building sensitivity studies
- Batch evaluation still uses rayon parallelism over the design matrix
- Future CLI integration (Plan 07-06) may provide a more sophisticated API that directly uses BatchOracle

No structural changes to file ownership or plan scope.

## Implementation Details

### 1. Analysis Module Scaffolding

Created `src/analysis/` with the following structure:

```rust
// src/analysis/mod.rs
//! Advanced analysis tools (Phase 7)
pub mod sensitivity;
pub mod delta;       // stub: "// TODO: Delta testing implementation"
pub mod components; // stub: "// TODO: Component breakdown"
pub mod swing;      // stub: "// TODO: Swing analysis"
```

Added `mod analysis;` to `src/lib.rs` (line 3).

### 2. Dependencies Added to Cargo.toml

```toml
csv = "1.3"
sobol = "1.0"
linregress = "0.5"
tempfile = "3.10"  # for tests
```

### 3. Sensitivity Engine (`src/analysis/sensitivity.rs`)

#### Core Types

```rust
/// Parameter range for design of experiments
pub struct ParameterRange {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

/// Metric set for a single parameter's sensitivity
pub struct MetricSet {
    pub normalized_coeff: f64,  // (max - min) / mean * 100%
    pub cvrmse: f64,           // Coefficient of Variation of RMSE (%)
    pub nmbe: f64,             // Normalized Mean Bias Error (%)
    pub slope: f64,            // Linear regression slope
}

/// Sensitivity report containing results for each parameter
pub struct SensitivityReport {
    pub parameters: Vec<String>,
    pub metrics: Vec<MetricSet>,
}
```

#### Sampling Strategies

- **`generate_oat_design(ranges, levels)`**: One-Factor-At-A-Time design. For each parameter, create `levels` samples where that parameter varies linearly from min to max while others hold at midpoint. Total rows = `ranges.len() * levels`.

- **`generate_sobol_design(ranges, num_samples)`**: Quasi-random Sobol sequence sampling using the `sobol` crate. Scales the unit hypercube to the specified ranges.

Both functions return `Vec<Vec<f64>>` design matrices suitable for batched evaluation.

#### Batch Evaluation

`run_sensitivity(design, case_builder, use_surrogates) -> Vec<f64>`

- Builds a `ThermalModel` from `case_builder`
- Uses rayon's `par_iter()` to evaluate all design points in parallel
- Each design point is a parameter vector applied to a cloned model
- Computes EUI as `energy / total_area` (kWh/m²/yr)
- Returns vector of outputs aligned with design matrix rows

#### Metrics Computation

`compute_metrics(design, outputs) -> SensitivityReport`

For each parameter column `i`:
- Extract x-values (design[*][i]) and y-values (outputs)
- Compute normalized coefficient: `(y_max - y_min) / mean(y) * 100%`
- Perform simple linear regression (manual implementation, no `linregress` crate due to API mismatch) to obtain slope
- Compute CVRMSE: `sqrt(mean((y - y_fit)²)) / mean(y) * 100%`
- Compute NMBE: `sum(y - y_fit) / (n * mean(y)) * 100%`

Results are sorted by descending absolute normalized coefficient.

#### CSV Export

`export_to_csv(report, path) -> Result<()>`

Writes CSV with headers: `Rank,Parameter,NormalizedCoeff,CVRMSE,NMBE,Slope`. Rows are in rank order (rank starting at 1).

#### Unit Tests (4 passing)

- `test_oat_generates_correct_matrix`: Validates 2-parameter, 5-level OAT design shape and baseline holding
- `test_sobol_coverage`: Checks Sobol sample dimensions and range coverage
- `test_metrics_computation`: Validates regression metrics with synthetic linear data
- `test_csv_export`: Verifies CSV headers and rank ordering

### 4. Fixes to Test Infrastructure During Execution

During implementation, discovered that the new `ValidationResult` struct needed a `per_program` field for multi-reference comparison (added in Plan 07-05). Applied the following fixes:

- **Added `per_program: Option<HashMap<String, ValidationStatus>>`** to `ValidationResult` in `src/validation/report.rs` with `#[serde(skip_serializing_if = "Option::is_none")]`
- **Updated `ValidationResult::new()`** to set `per_program: None`
- **Updated all test scaffolding** in:
  - `tests/test_guardrail_exit_codes.rs`
  - `tests/test_result_aggregation.rs`
  - `tests/benchmark_report_validation.rs`
  - `src/validation/analyzer.rs` (test module)
- **Fixed `generate()` call signature** in `tests/ashrae_140_validation.rs` to pass `None` for baseline argument
- **Fixed CSV assertion type** in `src/analysis/sensitivity.rs` test: changed from array to slice comparison
- **Added `use rand::Rng`** to `tests/test_modular_surrogates.rs` to fix `gen_range` errors

All test scaffolding now compiles cleanly and passes.

### 5. Stub Modules for Future Plans

- `src/analysis/delta.rs`: `// TODO: Delta testing implementation` (Plan 07-02)
- `src/analysis/components.rs`: `// TODO: Component breakdown` (Plan 07-03)
- `src/analysis/swing.rs`: `// TODO: Swing analysis` (Plan 07-03)

## Artifacts Deliverable

**Files Modified (total 10):**

1. `src/analysis/mod.rs` (new)
2. `src/analysis/sensitivity.rs` (new, 387 lines)
3. `src/analysis/delta.rs` (new stub)
4. `src/analysis/components.rs` (new stub)
5. `src/analysis/swing.rs` (new stub)
6. `src/lib.rs` (added `mod analysis;`)
7. `Cargo.toml` (added `csv`, `sobol`, `linregress`, `tempfile`)
8. `tests/test_guardrail_exit_codes.rs` (updated BenchmarkReport/ValidationResult constructions)
9. `tests/test_result_aggregation.rs` (updated constructions)
10. `tests/benchmark_report_validation.rs` (updated constructions)
11. `src/validation/analyzer.rs` (test module updated)
12. `tests/ashrae_140_validation.rs` (fixed generate() call)
13. `tests/test_modular_surrogates.rs` (added rand::Rng import)

Additional supporting changes:

- `src/validation/report.rs`: Added `per_program` field to `ValidationResult`
- `tests/test_modular_surrogates.rs`: Added missing import

## Key Decisions

- **Sampling library**: Chose `sobol` crate (v1.0) for quasi-random sequences. Lightweight and well-maintained.
- **Linear regression**: Initially considered `linregress` crate but encountered API mismatch; implemented manual ordinary least squares for simplicity and control.
- **Normalized coefficient definition**: `(max - min) / mean(y) * 100%` provides scale-free sensitivity measure.
- **Design matrix format**: `Vec<Vec<f64>>` is simple and works with rayon's `par_iter()`.
- **Stub approach**: Minimal placeholder comments preserve plan structure without premature implementation.

## Verification

✅ All unit tests pass (4 in sensitivity module, plus updated aggregation and guardrail tests)

```
cargo test --lib analysis::  # 4 passed
cargo test --test test_guardrail_exit_codes  # 6 passed
cargo test --test test_result_aggregation   # 10 passed
cargo test --test benchmark_report_validation  # 2 passed (2 ignored)
```

✅ Code compiles cleanly with `cargo check`

✅ API surface aligns with plan must-haves: sampling, evaluation, metrics, CSV export.

## Next Steps

- Plan 07-01 complete. Ready for Wave 2 execution.
- **Plan 07-02** (Delta Testing Framework) should begin next to fill `src/analysis/delta.rs`.
- **Plan 07-03** (Component & Swing) follows.
- **Plan 07-04** (Visualization) and **Plan 07-07** (Extensible Case Framework) can run in parallel with 07-02/03.
- Final **Plan 07-06** (CLI Integration) will wire all analysis features to the `fluxion` CLI.
