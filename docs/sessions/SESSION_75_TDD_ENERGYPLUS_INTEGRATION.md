# Session 75: Test-Driven Development with EnergyPlus Integration

## Executive Summary

This session established a comprehensive EnergyPlus comparison test framework and identified a **critical systemic bug** causing massive heating overprediction (~44 MWh vs ~2 MWh reference) across all high-mass cases.

## Critical Issue Discovered: Systemic Heating Overprediction

### Problem Statement
All high-mass cases (900-series) show heating energy of ~44-45 MWh, which is **20-60x higher** than ASHRAE 140 reference values.

| Case | Fluxion Heating (MWh) | Reference (MWh) | Error Factor |
|------|----------------------|-----------------|--------------|
| 900  | 44.44                | 1.17-2.04       | **22x**      |
| 910  | 44.89                | 1.51-2.28       | **20x**      |
| 920  | 44.75                | 3.26-4.30       | **11x**      |
| 930  | 45.40                | 4.14-5.34       | **9x**       |
| 940  | 34.60                | 0.79-1.41       | **25x**      |
| 960  | 44.79                | 5.00-15.00      | **3-9x**     |
| 195  | 45.45                | 3.50-6.00       | **7-13x**    |

### Root Cause Hypothesis
The consistent ~44-45 MWh value across different cases suggests:
1. **Energy accumulation bug**: Energy may be accumulating incorrectly in `step_physics()`
2. **CTF solver issue**: The CTF solver may be producing incorrect heat flux values
3. **Unit conversion error**: kWh vs Joules confusion in energy tracking

## Test Infrastructure Created

### New Test File: `tests/energyplus_comparison_tests.rs`

Created comprehensive EnergyPlus comparison test suite with:

1. **EnergyPlus Reference Data Structure**
   - `EnergyPlusReference` struct with case-specific reference values
   - Reference data for all 600-series and 900-series cases
   - Tolerance settings (±15% for energy, ±20% for peaks)

2. **Simulation Helper Functions**
   - `simulate_annual()`: Run full year simulation and return results
   - `SimulationResults` struct: Heating, cooling, peaks, temperatures

3. **Test Cases**
   - `test_case_900_annual_energy_vs_energyplus`: Annual energy comparison
   - `test_case_900ff_temperatures_vs_energyplus`: Free-floating temperatures
   - `test_case_900_peak_loads_vs_energyplus`: Peak load comparison
   - `test_thermal_mass_temperature_swing_reduction`: Thermal mass effect
   - `test_900_series_comprehensive_comparison`: Full 900-series validation

### Test Results (Current State)

```
running 7 tests
test test_case_900_hourly_energy_comparison ... ignored
test test_case_900_hourly_temperature_comparison ... ignored
test test_900_series_comprehensive_comparison ... ignored
test test_case_900_annual_energy_vs_energyplus ... FAILED
test test_case_900_peak_loads_vs_energyplus ... FAILED
test test_case_900ff_temperatures_vs_energyplus ... FAILED
test test_thermal_mass_temperature_swing_reduction ... FAILED

test result: FAILED. 0 passed; 4 failed; 3 ignored; 0 measured
```

### Key Findings

1. **Heating Energy**: 44.44 MWh vs 1.66 MWh reference (2577% error)
2. **Cooling Energy**: 1.45 MWh vs 2.49 MWh reference (42% error)
3. **Temperature Swing Reduction**: 10.0% vs expected ~19.6%
4. **Free-Floating Temps**: Min=-10.70°C, Max=32.13°C (outside ASHRAE ranges)

## Existing Test Infrastructure Status

### Unit Tests (✅ Passing)
```
running 12 tests
test test_hvac_cooling_mode_detection ... ok
test test_hvac_deadband ... ok
test test_energy_accumulation_consistency ... ok
test test_hvac_heating_mode_detection ... ok
test test_step_physics_finite_case_600 ... ok
test test_step_physics_reasonable_range_case_600 ... ok
test test_step_physics_finite_case_900 ... ok
test test_step_physics_reasonable_range_case_900 ... ok
test test_free_floating_stability_case_600ff ... ok
test test_free_floating_stability_case_900ff ... ok
test test_temperature_stability_case_600 ... ok
test test_temperature_stability_case_900 ... ok

test result: ok. 12 passed; 0 failed
```

### ASHRAE 140 Case 900 Tests (⚠️ Partial)
```
running 16 tests
test result: FAILED. 7 passed; 8 failed; 1 ignored
```

## EnergyPlus Integration Resources

### Python Tools Available
- `tools/ep_oracle.py`: EnergyPlus oracle for reference data generation
- `tools/mcp_client.py`: MCP client for OpenStudio integration
- `tools/compare_case_900.py`: Comparison tool for Case 900
- `tools/generate_energyplus_idf.py`: IDF file generation

### Reference Data
- `tests/energyplus_data/energyplus_workflow_results_ashrae140.csv`: 184KB of EnergyPlus results
- `tests/energyplus_data/case_900_baseline_results.json`: Detailed Case 900 results

## Recommended Next Steps

### Priority 1: Debug Systemic Heating Overprediction (CRITICAL)
1. Add debug output to `step_physics()` to trace energy values
2. Compare CTF solver output with expected heat flux
3. Verify energy accumulation logic in simulation loop
4. Check for unit conversion errors (kWh vs J vs W)

### Priority 2: Fix Thermal Mass Temperature Swing
- Current: 10% reduction vs expected ~19.6%
- Investigate thermal capacitance and coupling conductances
- Consider adjusting h_tr_em and h_tr_ms parameters

### Priority 3: Validate Free-Floating Temperatures
- Min temp: -10.70°C (should be -6.4 to -1.6°C)
- Max temp: 32.13°C (should be 41.8 to 46.4°C)
- Thermal mass buffering not capturing full effect

### Priority 4: Enhance EnergyPlus Comparison Tests
- Extract hourly data from EnergyPlus SQL output
- Implement hour-by-hour temperature comparison
- Add RMSE and max error metrics

## Files Modified

1. `tests/energyplus_comparison_tests.rs` (NEW)
   - Comprehensive EnergyPlus comparison test suite
   - Reference data for all ASHRAE 140 cases
   - Simulation helpers and validation tests

## Success Criteria (Current vs Target)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Unit test pass rate | 100% (12/12) | 100% | ✅ PASS |
| Case 900 heating error | 2577% | <15% | ❌ FAIL |
| Case 900 cooling error | 42% | <15% | ❌ FAIL |
| Temperature swing reduction | 10% | ~19.6% | ❌ FAIL |
| Free-floating min temp | -10.70°C | -6.4 to -1.6°C | ❌ FAIL |
| Free-floating max temp | 32.13°C | 41.8 to 46.4°C | ❌ FAIL |

## Lessons Learned

1. **Unit tests pass but integration tests fail**: The step_physics unit tests pass because they only check for finite values and reasonable ranges, not absolute accuracy.

2. **Systemic bug pattern**: The consistent ~44-45 MWh across different cases suggests a bug in the energy tracking or CTF solver, not a case-specific issue.

3. **EnergyPlus integration is critical**: Without EnergyPlus reference data, this systemic bug might have gone unnoticed.

4. **TDD approach working**: The test-first approach successfully identified the gap between expected and actual behavior.

## Appendix: Debug Output Analysis

The validator output shows:
```
[Solver] Case 900: Enabled CTF solver for high-mass construction (3 layers, U=0.556 W/m²K, τ=73.3h)
```

The CTF solver is being enabled correctly. The issue is likely in:
1. How CTF heat flux is calculated
2. How the heat flux is applied to the zone energy balance
3. How energy is accumulated over time
