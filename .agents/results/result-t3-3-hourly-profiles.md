# T3.3: Hourly Free-Float Temperature Profile Storage

**Status**: COMPLETE
**Issues**: #763, #749-G5

## Summary

Full hourly zone temperature profiles (8760 entries) are now stored for all four free-floating (FF) cases (600FF, 650FF, 900FF, 950FF), with min/max/mean statistics derived directly from the stored profile. A `MeanFreeFloat` metric type was added to the reporting infrastructure.

## What Was Already in Place

The codebase had significant existing infrastructure from Issue #827:

- `FreeFloatValidationResult.hourly_temperatures: Option<Vec<f64>>` — populated only for FF cases
- `CaseResults.hourly_temperatures: Option<Vec<f64>>` — same pattern in the validation suite
- `ThermalModelData.hourly_temperatures: Option<Vec<Vec<f64>>>` — sim-level storage (Issue #763)
- `TemperatureProfile` struct in `diagnostic.rs` with `hourly_temps`, `min_temp`, `max_temp`, `avg_temp`
- Hourly recording logic in all four simulation methods (`simulate_case`, `simulate_case_with_ideal_control`, `simulate_case_with_diagnostics`, `simulate_case_with_diagnostics_collector`)
- Tests for 600FF, 650FF, 900FF

## Changes Made

### 1. Added `free_float_mean_temp` to `FreeFloatValidationResult`
**File**: `src/validation/ashrae_140_validator.rs`
- Added `pub free_float_mean_temp: f64` field to the struct
- Updated construction in `ASHRAE140Validator::validate_ashrae_140()` to derive min/max/mean from the hourly profile (not from incremental counters), ensuring consistency between summary statistics and the full profile
- Non-FF cases get `free_float_mean_temp: 0.0`

### 2. Added `MeanFreeFloat` metric type
**File**: `src/validation/report.rs`
- Added `MeanFreeFloat` variant to `MetricType` enum
- Added to `Ord` implementation (after `MaxFreeFloat`, before `IncidentSolar`)
- Added display name: "Mean Free-Floating Temperature (°C)"
- Added to `units()` match alongside `MinFreeFloat`/`MaxFreeFloat`
- Added `mean_free_float_min`/`mean_free_float_max` fields to `BenchmarkData`
- Added `MeanFreeFloat` to `get_range()` and benchmark aggregation logic
- Updated unit tests to verify `MeanFreeFloat` range lookup

### 3. Updated `BenchmarkData` across all construction sites
**File**: `src/validation/benchmark.rs`
- Added `mean_free_float_min: 0.0, mean_free_float_max: 0.0` to all 21 `BenchmarkData` struct initializations
- Reference ranges for mean temperature will be populated once reference data becomes available

### 4. Added 950FF test and mean verification
**File**: `tests/validation/hourly_ff_profile.rs`
- Added `hourly_profile_populated_for_950ff` test (Case950FF)
- Added mean temperature assertion to `assert_ff_profile_shape` helper
- Added `free_float_mean_temp == 0.0` assertion for non-FF case test
- Updated module docstring to reflect Issue #763 coverage and all four FF cases

## Files Changed

| File | Change |
|------|--------|
| `src/validation/ashrae_140_validator.rs` | Added `free_float_mean_temp` field; derived min/max/mean from profile |
| `src/validation/report.rs` | Added `MeanFreeFloat` metric type; extended `BenchmarkData` |
| `src/validation/benchmark.rs` | Added `mean_free_float_min/max` to all `BenchmarkData` constructions |
| `tests/validation/hourly_ff_profile.rs` | Added 950FF test; mean temp assertions; updated docs |

## Test Results

```
cargo test --test validation_hourly_ff_profile: 5 passed
  - hourly_profile_populated_for_600ff ✓
  - hourly_profile_populated_for_650ff ✓
  - hourly_profile_populated_for_900ff ✓
  - hourly_profile_populated_for_950ff ✓
  - hourly_profile_none_for_non_ff_case ✓

cargo test --lib report: 167 passed
cargo test --lib benchmark: 59 passed
```

## Acceptance Criteria Checklist

- [x] Full hourly zone temperature profiles stored for 600FF
- [x] Full hourly zone temperature profiles stored for 650FF
- [x] Full hourly zone temperature profiles stored for 900FF
- [x] Full hourly zone temperature profiles stored for 950FF
- [x] Min temperature derived from stored profile
- [x] Max temperature derived from stored profile
- [x] Mean temperature derived from stored profile
- [x] No allocation for non-FF cases (Option stays None)
- [x] ~70 KB per FF case (Vec<f64> with 8760 entries)
- [x] Tests pass for all four FF cases
- [x] MeanFreeFloat metric type available for validation reports

## Architecture Notes

The profile-derived approach ensures min/max/mean are always consistent with the actual hourly data. Previously, min/max were tracked via incremental `f64::min/f64::max` calls during simulation, which could theoretically diverge from the profile if recording logic differed. Now the final values are computed from the stored profile in a single pass, eliminating any possibility of inconsistency.
