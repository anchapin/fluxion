# T3.1: Add Peak Load Timestamps to ValidationResult

**Status**: DONE
**Issue**: #761, #749-G3
**Agent**: backend-engineer

## Summary

Added peak load hour tracking to the thermal model and wired it through to ASHRAE 140 validation reports. Peak heating and cooling timestamps are now captured during simulation and included in `ValidationReport`.

## Files Changed

### 1. `src/sim/thermal_model_data.rs`
- Added `peak_power_heating_hour: usize` field (hour index 0-8759 of peak heating)
- Added `peak_power_cooling_hour: usize` field (hour index 0-8759 of peak cooling)
- Added fields to clone implementation

### 2. `src/sim/thermal_model_core.rs`
- Added `get_peak_heating_hour()` getter
- Added `get_peak_cooling_hour()` getter
- Updated `reset_peak_power()` to reset hour fields
- Added initialization of new fields in model constructor

### 3. `src/sim/thermal_model_physics.rs`
- Updated 3 peak power tracking locations (5R1C, equipment, 6R2C models) to record hour when peak occurs
- Changed from `.max()` to conditional assignment with hour tracking

### 4. `src/validation/ashrae_140_validator.rs`
- Added `peak_heating_time: Option<String>` to `ValidationReport` struct
- Added `peak_cooling_time: Option<String>` to `ValidationReport` struct
- Updated `simulate_case_with_diagnostics()` to use model's peak hour getters for `PeakTiming`
- Updated `validate_case_960()` to compute and include peak timestamps
- Updated `validate_case_with_diagnostics()` to compute and include peak timestamps
- Added `test_peak_hour_tracking_in_model` test
- Added `test_peak_timing_hour_to_datetime` test

## Technical Details

### Peak Hour Tracking
- The model tracks peak power via `peak_power_heating`/`peak_power_cooling` in Watts
- Previously, the hour of peak occurrence was lost (initialized to 0, never updated)
- Now, when a new peak is detected during `step_physics()`, the current `timestep` is recorded alongside the power value

### Timestamp Formatting
- Uses existing `PeakTiming::hour_to_datetime()` which converts hour index to "Mon DD HH:00" format
- Examples: hour 0 = "Jan 1 00:00", hour 4380 = "Jul 2 12:00", hour 8759 = "Dec 31 23:00"

### Backward Compatibility
- New fields use `Option<String>` in `ValidationReport` for backward compatibility
- New model fields default to 0 (consistent with previous behavior)
- `PeakTiming` struct unchanged (already had hour fields, just was always 0)

## Acceptance Criteria Checklist

- [x] Peak load hour tracked in thermal model (peak_power_heating_hour, peak_power_cooling_hour)
- [x] Peak timing correctly flows from simulation to validation diagnostics
- [x] ValidationReport includes human-readable timestamps (peak_heating_time, peak_cooling_time)
- [x] validate_case_960 includes peak timestamps
- [x] validate_case_with_diagnostics includes peak timestamps
- [x] Backward compatible (Option<> types, default values)
- [x] All existing tests pass (8 validator, 50 thermal model, 12 peak)
- [x] New tests added (test_peak_hour_tracking_in_model, test_peak_timing_hour_to_datetime)
