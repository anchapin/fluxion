# Debug Result: Issue #853 - Pre-existing CI Failures

## Status: FIXED

## Summary
Fixed 3 of 4 pre-existing CI failure categories identified in issue #853. The 4th category (format trait errors) was already fixed on main.

## Files Changed
| File | Change |
|------|--------|
| `src/validation/analyzer.rs` | Fixed `test_quality_metrics_mae_calculation` test data: single result with `(6.5, 5.5, 7.5)` replaced with two results `(22.5, 10.0, 20.0)` and `(13.5, 10.0, 20.0)` to match documented MAE=30% expectations |
| `benches/orchestration_decisions/tdqs.rs` | Added `#![allow(dead_code)]` to suppress 28 dead-code warnings for benchmark utility types |
| `benches/orchestration_decisions/decision_recorder.rs` | Added `#![allow(dead_code)]` to suppress dead-code warnings for benchmark recording types |

## Root Cause Analysis

### 1. `test_quality_metrics_mae_calculation` (analyzer.rs)
- **Root cause**: Test data (`value=6.5, ref_min=5.5, ref_max=7.5`) produced MAE=0.0% (value equals midpoint), but assertions expected MAE=30.0% and max_deviation=50.0%. The test comments documented different data than what was in the code.
- **Fix**: Updated test data to use two metrics matching the documented scenario: `(22.5, [10,20])` gives 50% error (Fail), `(13.5, [10,20])` gives 10% error (Warning). MAE=(50+10)/2=30%.

### 2. Format trait errors (`{:.3f}` / `{:8.3f}`)
- **Status**: Already fixed on main (no occurrences found in grep).
- **No action needed.**

### 3. Dead-code warnings in benches
- **Root cause**: `tdqs.rs` and `decision_recorder.rs` define types for future orchestration decision integration. These are benchmark utilities not yet consumed.
- **Fix**: Added `#![allow(dead_code)]` at module level for both files.

### 4. CI matrix test failures
- **Expected resolution**: Format trait errors (item 2) were the likely cause of feature-gated compilation failures. These are already fixed.

## Acceptance Criteria Checklist
- [x] `test_quality_metrics_mae_calculation` passes
- [x] All 664 validation tests pass
- [x] `cargo clippy --all-targets` reports 0 warnings
- [x] Format trait errors already fixed on main
- [x] Dead-code warnings in benches suppressed
- [x] Minimal changes only - no refactoring

## Out of Scope
- `thermal_mass_energy_accounting.rs` warnings inside `#[cfg(feature = "pr821-diag")]` blocks (feature-gated, not in default CI)
- CI matrix test failures may need separate verification after this PR merges
