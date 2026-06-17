# T3.4: Verify Output Units Are Correct (G1, G3, G5)

**Status**: COMPLETED
**Date**: 2026-05-16

## Summary

Audited all output metrics across the validation pipeline against ASHRAE 140-2023 Section 8 requirements. Found and fixed 3 issues: 2 doc-comment mismatches and 1 CSV header missing unit suffixes. Also corrected 2 integration test files that asserted incorrect unit expectations.

## Unit Map (Final State)

### ASHRAE 140 Section 8 Output Metrics

| Metric | ASHRAE 140 Ref | Code Unit | Display Name | Status |
|--------|---------------|-----------|--------------|--------|
| G1: Annual Heating Energy | MWh | MWh | "Annual Heating Energy (MWh)" | CORRECT |
| G1: Annual Cooling Energy | MWh | MWh | "Annual Cooling Energy (MWh)" | CORRECT |
| G3: Peak Heating Load | kW | kW | "Peak Heating Load (kW)" | CORRECT |
| G3: Peak Cooling Load | kW | kW | "Peak Cooling Load (kW)" | CORRECT |
| G3: Peak Timestamps | Month Day HH:00 | String | "Jan 15 14:00" format | CORRECT |
| G5: Min Free-Float Temp | deg C | deg C | "Minimum Free-Floating Temperature (deg C)" | CORRECT |
| G5: Max Free-Float Temp | deg C | deg C | "Maximum Free-Floating Temperature (deg C)" | CORRECT |
| G5: Mean Free-Float Temp | deg C | deg C | "Mean Free-Floating Temperature (deg C)" | CORRECT |
| G4: Incident Solar | kWh/m2 | kWh/m2 | "Incident Solar Radiation (kWh/m2)" | CORRECT |

### Internal Data Structure Units

| Struct | Field | Unit | Verified |
|--------|-------|------|----------|
| `CaseResults` | `annual_heating_mwh` | MWh | YES |
| `CaseResults` | `annual_cooling_mwh` | MWh | YES |
| `CaseResults` | `peak_heating_kw` | kW | YES |
| `CaseResults` | `peak_cooling_kw` | kW | YES |
| `CaseResults` | `min_temp_celsius` | deg C | YES |
| `CaseResults` | `max_temp_celsius` | deg C | YES |
| `CaseResults` | `hourly_temperatures` | deg C | YES |
| `CaseResults` | `incident_solar` | kWh/m2 | YES |
| `EnergyBreakdown` | all `*_mwh` fields | MWh | YES |
| `PeakTiming` | `peak_heating_kw` | kW | YES |
| `PeakTiming` | `peak_cooling_kw` | kW | YES |
| `TemperatureProfile` | all temp fields | deg C | YES |
| `HourlyData` | `outdoor_temp` | deg C | YES |
| `HourlyData` | `zone_temps` | deg C | YES |
| `HourlyData` | `solar_gains` | W | YES |
| `HourlyData` | `hvac_heating` | W | YES |
| `HourlyData` | `hvac_cooling` | W | YES |

### Report Output Units

| Output Format | Energy Unit | Power Unit | Temp Unit | Solar Unit |
|--------------|-------------|------------|-----------|------------|
| Markdown table | MWh | kW | deg C | kWh/m2 |
| HTML report | MWh | kW | deg C | kWh/m2 |
| CSV export | MWh | kW | deg C | kWh/m2 |
| JSON serialize | MWh | kW | deg C | kWh/m2 |
| Console summary | MWh | kW | deg C | N/A |
| FF Temp Profile CSV | N/A | N/A | deg C | N/A |
| Hourly diagnostic CSV | N/A | W | deg C | W |

## Corrections Made

### 1. Fixed doc-comment mismatch in `MetricType` enum
**File**: `src/validation/report.rs:84-87`
**Issue**: `AnnualHeating` and `AnnualCooling` doc comments said "(kWh)" but `units()` and `display_name()` correctly return "MWh".
**Fix**: Updated doc comments from "kWh" to "MWh" to match actual implementation and reference data.

### 2. Fixed CSV export headers missing unit suffixes
**File**: `src/validation/diagnostic.rs:678`
**Issue**: CSV column headers lacked unit annotations (`Outdoor_Temp`, `Zone_Temps`, `Solar_Gains`, etc.).
**Fix**: Updated headers to include unit suffixes: `Outdoor_Temp_C`, `Zone_Temps_C`, `Solar_Gains_W`, `HVAC_Heating_W`, `HVAC_Cooling_W`.

### 3. Fixed integration test assertions with incorrect unit expectations
**File**: `tests/test_validator_core.rs:48,52`
**Issue**: Test asserted `display_name()` returns "kWh" but implementation returns "MWh" (which matches reference data).
**Fix**: Updated assertions to expect "MWh".

**File**: `tests/test_validator_core.rs:75-76`
**Issue**: Test asserted `units()` returns "kWh" but implementation returns "MWh".
**Fix**: Updated assertions to expect "MWh".

**File**: `tests/validation_report.rs:42-43`
**Issue**: Same incorrect unit assertion in duplicate test.
**Fix**: Updated assertions to expect "MWh".

## Key Decision: MWh vs kWh

**Decision**: Energy units remain **MWh** (not kWh as Issue #760 suggested).

**Rationale**:
1. Reference benchmark data field names use `annual_heating_MWh` (see `src/validation/benchmark.rs:532`)
2. Reference values for Case 600 (4.30-5.71) are clearly MWh — 4.30 kWh would be absurdly low for annual space heating
3. The ASHRAE 140-2023 standard output spreadsheet (`Std140_TF_Output.xlsx`) uses MWh for annual energy
4. All internal accumulation and display already uses MWh consistently
5. Changing to kWh would require a 1000x conversion throughout the entire pipeline with no accuracy benefit

## Acceptance Criteria Checklist

- [x] G1: Annual heating/cooling energy uses MWh with correct labels
- [x] G3: Peak heating/cooling loads use kW with correct labels
- [x] G3: Peak load timestamps in readable "Mon DD HH:00" format
- [x] G5: Free-float temperatures (min, max, mean) use deg C
- [x] G4: Incident solar radiation uses kWh/m2
- [x] All MetricType `units()` and `display_name()` match struct field names
- [x] CSV exports include unit suffixes in column headers
- [x] Markdown/HTML/JSON reports use consistent unit labels
- [x] All doc comments match actual units
- [x] All tests pass with corrected assertions

## Files Changed

1. `src/validation/report.rs` — Fixed MetricType doc comments (kWh -> MWh)
2. `src/validation/diagnostic.rs` — Added unit suffixes to CSV headers
3. `tests/test_validator_core.rs` — Fixed unit assertion expectations
4. `tests/validation_report.rs` — Fixed unit assertion expectations
