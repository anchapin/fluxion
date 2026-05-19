# Result: Issue #866 — Verify and Enable Multi-Node HVAC Energy Override

**Status**: PARTIAL SUCCESS — energy override enabled, 1 of 5 metrics now passes
**PR**: https://github.com/anchapin/fluxion/pull/870
**Branch**: `fix/866-enable-energy-override`

## Summary

The `step_physics_9r4c` method was missing annual energy accumulation and peak power tracking that `step_physics_5r1c` already performs. Added both, plus fixed a return-value unit mismatch (was Joules, now kWh).

## Files Changed

- `src/sim/thermal_model_physics.rs` — +38 lines, -1 line

## Changes Made

1. **Energy accumulation** (lines ~2596-2627): Added heating/cooling energy accumulation mirroring `step_physics_5r1c` pattern. Uses `hvac_output` from `hvac_demand_from_ideal_loads` to split into heating (positive) and cooling (negative) components, accumulates as kWh.

2. **Peak power tracking** (lines ~2619-2626): Added `peak_power_heating` and `peak_power_cooling` tracking from `hvac_power_watts`.

3. **Return value fix** (line ~2646): Changed `hvac_power_watts * dt` (Joules) to `hvac_power_watts * dt / 3.6e6` (kWh) to match the public API contract and all other physics paths.

## Acceptance Criteria

- [x] Energy override re-enabled in `step_physics_9r4c`
- [x] Annual heating within 1.17–2.04 MWh → **1.91 MWh** ✅
- [ ] Annual cooling within 2.13–3.67 MWh → **0.04 MWh** ❌ (pre-existing)
- [ ] Peak heating within 1.10–2.10 kW → **0.42 kW** ❌ (pre-existing)
- [ ] Peak cooling within 1.50–3.50 kW → **0.15 kW** ❌ (pre-existing)
- [ ] 900FF max temperature within 41.8–46.4°C → **27.73°C** ❌ (pre-existing)
- [x] Full lib test suite passes (2451 passed, 2 ignored)

## Pre-existing Issues (Out of Scope)

All 4 remaining failures are **pre-existing** on `main` (verified by testing without my changes). Root cause: the 9R4C thermal model computes zone temperatures from the 5R1C free-floating formula using pre-solver mass temperatures, then updates mass temperatures from the multi-node solver *afterwards*. The solver's thermal feedback never reaches the zone air temperature, causing:
- Insufficient summer temperatures → low cooling demand
- Low peak power values
- 900FF max temp far below reference range

Fixing this requires refactoring the 9R4C step to feed multi-node solver temperatures back into the zone temperature calculation, which is a separate issue.

## Net Improvement

- **Before** (main): 9 pass, 8 fail on `ashrae_140_case_900`
- **After** (this PR): 10 pass, 7 fail — annual heating now passes
- No regressions in any other test suite
