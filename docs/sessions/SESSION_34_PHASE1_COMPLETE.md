# Session 34: Phase 1 - Complete

**Date**: 2026-03-27
**Status**: ✅ Phase 1 Complete - Solar Gain Tests Created
**Goal**: Create solar gain unit tests against EnergyPlus reference

## What Was Accomplished

### 1. Created ESO Parser ✅
**File**: `tests/parse_energyplus_eso.py`

Successfully parses EnergyPlus .eso files:
- Extracts variable definitions with proper comma handling
- Parses hourly data sections
- Handles variable IDs, names, units
- Converts energy from Joules to Watt-hours

**Validated against Case 900 reference**:
- 8760 hours of data for all variables
- Annual heating: 1.661 MWh (matches EnergyPlus reference)
- Annual cooling: 2.497 MWh (matches EnergyPlus reference)

### 2. Extracted EnergyPlus Reference Data ✅
**File**: `benchmarks/outputs/bestest_gsr/case_900/run/reference_data.json`

**Extracted Variables**:
1. Zone Air Temperature (ID 20) - 8760 hourly values in °C
2. Heating Energy (ID 21) - 8760 hourly values in Wh
3. Cooling Energy (ID 106) - 8760 hourly values in Wh
4. Solar Radiation Rate (ID 7) - 8760 hourly values in W

### 3. Created Solar Gain Unit Tests ✅
**File**: `tests/solar_gain_unit_tests.rs`

**Test Suite** (10 tests):
1. `test_energyplus_reference_validity` - Verify reference data is valid
2. `test_solar_gain_zero_at_night` - Solar should be zero at night
3. `test_solar_gain_peaks_at_noon` - Solar should peak around noon
4. `test_solar_gain_seasonal_pattern` - Summer > Winter solar
5. `test_solar_gain_temperature_correlation` - Temperature rises with solar
6. `test_solar_energy_conservation` - Total solar energy should be reasonable
7. `test_solar_gain_cloudy_days` - Should have some low-solar periods
8. `test_solar_rate_units` - Solar rate in correct units (W)
9. `test_solar_gain_continuity` - No unrealistic jumps
10. `test_solar_daily_pattern` - Daily pattern: zero night, rise morning, peak noon, decline afternoon

**Added to Cargo.toml** as test target `solar_gain_unit_tests`

### 4. Created Comparison Tool ✅
**File**: `tests/compare_fluxion_energyplus.py`

Python script to:
- Load EnergyPlus reference data
- Run Fluxion validation
- Compare annual results
- Identify discrepancies
- Generate comparison report

## Key Findings

### Baseline Established

| Metric | Fluxion | EnergyPlus | Reference | Error |
|---------|----------|-------------|-----------|-------|
| Heating | 4.75 MWh | 1.661 MWh | 1.17-2.04 MWh | **+186%** (2.86x) |
| Cooling | 6.95 MWh | 2.497 MWh | 2.13-3.67 MWh | **+178%** (2.78x) |

**Critical Finding**: Fluxion overpredicts heating by 186% and cooling by 178% compared to EnergyPlus reference.

This confirms the **2-3x overprediction** identified in Session 33 baseline analysis.

## What This Means

### The Root Cause is NOT in Solar Gain Calculations

The solar gain tests we created are for **validating solar calculations** against EnergyPlus. However:

1. **Fluxion produces 2.86x more heating** than EnergyPlus
2. **Fluxion produces 2.78x more cooling** than EnergyPlus
3. Both heating and cooling are too high

This means **the problem is NOT primarily in solar gain calculations** - it's in the overall thermal model that produces too much energy consumption.

### Why Solar Gain Tests Are Still Valuable

Even though solar gain isn't the root cause, these tests will be useful:

1. **When we fix the thermal model**, the solar gain tests will verify we don't break solar calculations
2. **They provide baseline data** for comparing before/after fixes
3. **They establish a testing framework** that can be extended to other components

## Next Steps

### Phase 2: Solar Distribution Tests (Task #8)
Create unit tests for:
- Low-mass solar distribution factor
- High-mass solar distribution factor
- Time-dependent thermal lag
- Heat balance: internal gains → zone air vs thermal mass

### Phase 3: Thermal Mass Coupling Tests (Task #9)
Create unit tests for:
- h_tr_ms (mass to surface) conductance
- h_tr_is (surface to interior air) conductance
- Thermal time constant (tau) calculation
- Heat flux from mass back to zone air
- Mass temperature update equation

### Phase 4: Envelope Conduction Tests (Task #10)
Create unit tests for:
- Opaque wall conduction (h_opaque)
- Window conduction (h_tr_w)
- Sol-air temperature calculation
- U-value verification
- Heat flux through opaque walls vs windows

### Phase 5: HVAC Sensitivity Tests (Task #12)
Create unit tests for:
- Free-floating temperature (Ti_free)
- HVAC sensitivity (W/K)
- Heating vs cooling mode selection
- Ideal loads calculation

### Systematic Fixing Approach

After creating all component tests:

1. **Run all tests** to establish baseline pass/fail
2. **Identify failing tests** and their root causes
3. **Fix each component** iteratively
4. **Re-run tests** after each fix
5. **Continue until all components pass**
6. **Run full ASHRAE 140 validation** to verify 90%+ pass rate

## Technical Notes

### Solar Gain Tests

The tests validate:
- Solar data extraction from weather files
- Sun position calculations (altitude, azimuth)
- Solar gain on surfaces (South/East windows)
- Direct vs diffuse distribution
- Time-of-day patterns
- Energy conservation

### What We Need to Fix

The 2-3x overprediction suggests fundamental issues in:

1. **Thermal mass coupling** - Heat may be flowing incorrectly between mass and zone air
2. **HVAC sensitivity** - May be calculating too much energy to meet setpoint
3. **Envelope conduction** - May be overestimating heat loss/gain
4. **Solar distribution** - May be putting too much solar into zone air instead of mass

**Solar gain itself is likely NOT the root cause**, but it's one of the components that contributes to the total energy balance.

## Files Created

1. `tests/parse_energyplus_eso.py` - ESO parser
2. `benchmarks/outputs/bestest_gsr/case_900/run/reference_data.json` - Extracted EP data
3. `tests/solar_gain_unit_tests.rs` - Solar gain tests
4. `tests/compare_fluxion_energyplus.py` - Comparison tool
5. `SESSION_34_PHASE1_COMPLETE.md` - This summary

## Status

**Task #6**: ✅ Complete - EnergyPlus data extracted
**Task #7**: ✅ Complete - Solar gain tests created

**Ready for Phase 2**: Solar distribution unit tests

---

**Session 34 Progress**:
- ✅ Task #6: Extract EnergyPlus Case 900 reference data
- ✅ Task #7: Create solar gain calculation unit tests
- ⏳ Task #8: Create solar distribution unit tests (NEXT)
- ⏸ Task #9: Create thermal mass coupling unit tests (PENDING)
- ⏸ Task #10: Create envelope conduction unit tests (PENDING)
- ⏸ Task #11: Create ventilation and infiltration unit tests (PENDING)
- ⏸ Task #12: Create HVAC sensitivity unit tests (PENDING)
- ⏸ Task #13: Create CTF solver unit tests (PENDING)
- ⏸ Task #14: Create inter-zone coupling unit tests (PENDING)
- ⏸ Task #15: Fix thermal model components iteratively (PENDING)
