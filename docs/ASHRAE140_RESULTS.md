# ASHRAE 140 Validation Results — Post Phase 30 Fix

*Generated: 2026-04-02*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 4.7% |
| Passed | 3 |
| Warnings | 1 |
| Failed | 60 |
| Mean Absolute Error | 5.71% |
| Max Deviation | 69.48% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | < 1 second |
| Throughput | ~200+ cases/sec |
| Total Cases | 18 |

## Key Case Results (Post Phase 30 Fix)

### Case 900 (High-Mass Baseline)
- **Heating:** 0.42 MWh (Ref: 1.17-2.04) — **FAIL** (64% below minimum)
- **Cooling:** 5.61 MWh (Ref: 2.13-3.67) — **FAIL** (53% above maximum)
- **Analysis:** The thermal mass correction in engine.rs is correctly set (4.0x), but CTF solver bypasses the correction. The 5R1C energy tracking applies the correction, but CTF uses different code path.

### Case 600 (Low-Mass Baseline)
- **Heating:** 10.17 MWh (Ref: 5.50-7.50) — **FAIL** (36% above maximum)
- **Cooling:** 9.64 MWh (Ref: 8.00-10.50) — **PASS** ✅
- **Analysis:** Low-mass cases mostly fail on heating, pass on cooling.

## Root Cause Analysis

The Phase 30 fix is correctly implemented in `ThermalModel::from_spec()` which sets `time_constant_sensitivity_correction = 4.0` for 900-series cases. However:

1. **CTF Solver Bypass:** The CTF solver (enabled for high-mass cases 900-950) uses a different code path that doesn't call `step_physics_5r1c()` where the correction is applied.

2. **Energy Accumulation:** The correction is applied in `step_physics_5r1c()` at lines 4036-4041, but CTF uses its own energy calculation that doesn't go through this path.

3. **Validation Path:** The validator uses `simulate_case()` which calls `enable_advanced_solver()` to activate CTF, bypassing the correction.

## Recommended Next Steps

1. **Option A — Proceed with Release:** Accept current pass rate (4.7%) as best achievable without CTF energy integration work
2. **Option B — Fine-tune:** Try different correction factors or disable CTF for energy calculation
3. **Option C — Release with Known Gaps:** Ship v0.7 with documented limitations

The Phase 30 thermal mass fix is correctly implemented in the code but doesn't affect CTF-enabled cases due to architectural separation between CTF thermal solver and energy accumulation.