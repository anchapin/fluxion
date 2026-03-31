# Session 74: Test-Driven Development with EnergyPlus Integration

## Executive Summary

This session continued the test-driven development process to improve physics engine accuracy, focusing on:
1. **Fixed critical free-floating case setpoint configuration** - Case 600FF now properly configured
2. **Validated unit test infrastructure** - All 12 step_physics unit tests passing (100%)
3. **Identified remaining ASHRAE 140 Case 900 accuracy issues** - 8 tests failing, requiring further tuning

## Session 73 Inherited Issues (Resolved)

| Test | Issue | Status |
|------|-------|--------|
| `test_free_floating_stability_case_600ff` | Setpoints not configured for free-floating | ✅ FIXED |

## Key Accomplishments

### 1. Fixed Free-Floating Case Setpoint Configuration

**Problem:** The test `test_free_floating_stability_case_600ff` was failing because `ThermalModel::from_spec()` didn't set extreme setpoints (-999°C heating, 999°C cooling) for free-floating cases.

**Root Cause:** The `from_spec()` function was using the HVAC schedule's setpoints directly (20°C/27°C) instead of extreme values that effectively disable HVAC.

**Solution:** Modified `src/sim/engine.rs` in `from_spec()`:
```rust
// SESSION 73: For free-floating cases, set extreme setpoints to disable HVAC
// This matches the behavior in ashrae_140_validator.rs
if spec.is_free_floating() {
    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;
    model.heating_schedule = DailySchedule::constant(-999.0);
    model.cooling_schedule = DailySchedule::constant(999.0);
} else {
    model.heating_setpoint = hvac.heating_setpoint;
    model.cooling_setpoint = hvac.cooling_setpoint;
}
```

**Result:** All 12 unit tests now passing (100% pass rate)

### 2. Unit Test Results

```
running 12 tests
test test_energy_accumulation_consistency ... ok
test test_hvac_heating_mode_detection ... ok
test test_hvac_cooling_mode_detection ... ok
test test_hvac_deadband ... ok
test test_step_physics_finite_case_600 ... ok
test test_step_physics_finite_case_900 ... ok
test test_step_physics_reasonable_range_case_600 ... ok
test test_step_physics_reasonable_range_case_900 ... ok
test test_free_floating_stability_case_600ff ... ok
test test_free_floating_stability_case_900ff ... ok
test test_temperature_stability_case_600 ... ok
test test_temperature_stability_case_900 ... ok

test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### 3. ASHRAE 140 Case 900 Validation Status

```
running 15 tests
test result: FAILED. 7 passed; 8 failed; 1 ignored
```

**Failing Tests:**
- `test_case_900_annual_heating_within_reference_range` - Heating energy outside ASHRAE range
- `test_case_900_annual_cooling_within_reference_range` - Cooling energy outside ASHRAE range
- `test_case_900_annual_cooling_energy_with_correction` - COP correction issue
- `test_case_900_peak_heating_within_reference_range` - Peak heating outside range
- `test_case_900_thermal_mass_energy_balance` - Energy balance issue
- `test_case_900ff_min_temperature_within_reference_range` - Free-floating min temp
- `test_case_900ff_max_temperature_within_reference_range` - Free-floating max temp
- `test_case_900ff_temperature_swing_reduction` - Only 10% reduction (expected ~19.6%)

## EnergyPlus Integration Resources

The project has extensive EnergyPlus integration infrastructure:

### Python Tools (`tools/` directory)
- `ep_oracle.py` - EnergyPlus oracle for reference data generation
- `generate_energyplus_idf.py` - IDF file generation for ASHRAE 140 cases
- `run_energyplus_simulations.py` - Batch EnergyPlus simulation runner
- `extract_energyplus_components.py` - Extract component-level data from EnergyPlus
- `compare_case_900.py` - Compare Fluxion vs EnergyPlus results
- `mcp_client.py` - MCP client for OpenStudio integration
- `create_case_900_with_mcp.py` - Create Case 900 model using MCP tools

### Reference Data
- `energyplus_workflow_results_ashrae140.csv` - 184KB of EnergyPlus simulation results
- `extract_ep_reference.py` - Extract reference values from EnergyPlus output

### MCP Integration
- `list_mcp_tools.py` - Lists available MCP tools
- `geometry_extraction.py` - VLM-based geometry extraction pipeline

## Remaining Work

### Priority 1: Fix Case 900 Annual Energy Accuracy
- Heating energy: Currently outside 1.17-2.04 MWh reference range
- Cooling energy: Currently outside 2.13-3.67 MWh reference range
- Root cause: Likely solar gain distribution and thermal mass coupling

### Priority 2: Fix Free-Floating Temperature Ranges
- Temperature swing reduction only 10% (expected ~19.6%)
- Min/max temperatures outside reference ranges
- Root cause: Thermal capacitance and conductance tuning needed

### Priority 3: Leverage EnergyPlus for Calibration
- Use `ep_oracle.py` to generate hourly reference data
- Compare component-level heat flows (conduction, solar, ventilation)
- Calibrate thermal mass parameters against EnergyPlus

## Test-Driven Development Approach

### Phase 1: Unit Tests (Complete)
- ✅ All 12 step_physics unit tests passing
- ✅ Temperature stability verified
- ✅ Free-floating cases properly configured

### Phase 2: Integration Tests (In Progress)
- ⚠️ Case 900 annual energy tests failing
- ⚠️ Free-floating temperature tests failing
- Need to calibrate against EnergyPlus reference data

### Phase 3: EnergyPlus Comparison Tests (Pending)
- Create hourly comparison tests using `ep_oracle.py`
- Validate heat balance components
- Calibrate thermal mass parameters

## Files Modified

1. `src/sim/engine.rs`:
   - Lines ~1100-1110: Added free-floating setpoint handling in `from_spec()`

## Success Criteria

| Criterion | Target | Current | Status |
|-----------|--------|---------|--------|
| Unit test pass rate | 100% | 100% (12/12) | ✅ PASS |
| Case 900 heating energy | 1.17-2.04 MWh | Outside range | ❌ FAIL |
| Case 900 cooling energy | 2.13-3.67 MWh | Outside range | ❌ FAIL |
| Free-floating temp swing reduction | ~19.6% | 10% | ❌ FAIL |
| Overall ASHRAE 140 pass rate | ≥90% | In progress | ⏳ PENDING |

## Next Steps

1. **Immediate:** Run EnergyPlus oracle to generate hourly reference data for Case 900
2. **Short-term:** Calibrate thermal mass coupling parameters against EnergyPlus
3. **Medium-term:** Implement EnergyPlus comparison test suite
4. **Long-term:** Achieve ≥90% ASHRAE 140 validation pass rate
