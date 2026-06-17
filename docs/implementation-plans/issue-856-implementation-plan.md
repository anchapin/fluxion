# Issue #856: Full Multi-Node HVAC Energy Simulation — Implementation Plan

**Epic**: #856 — Full Multi-Node HVAC Energy Simulation for Case 900
**Date**: 2026-06-10
**Author**: Backend Agent
**Status**: PLAN (Phase 4+ remaining work)

---

## Executive Summary

Phase 3 (#871–#875) and Phase 3b (#876) are closed but **9 of 19 Case 900 tests still fail**. The multi-node solver is wired in as a side-car but zone temperature computation still depends on 5R1C for the free-float temperature, and the HVAC coefficient formula produces results outside reference ranges. The remaining work falls into four phases:

| Phase | Description | Key Issues | Effort |
|-------|-------------|------------|--------|
| 4A | Fix HVAC demand coefficient calibration | #893, #895 | 1–2 days |
| 4B | Fix free-floating temperature calibration | #904, #917, #918, #919, #920 | 2–3 days |
| 4C | Fix mass dynamics (phase lag, h_tr_ms) | #896, #921 | 1–2 days |
| 4D | Regression cleanup (600 series) | #903 | 1 day |
| **Total** | | | **5–8 days** |

---

## Current Metric Status

| Metric | Current | Reference Range | Gap |
|--------|---------|----------------|-----|
| Annual heating | ~4.6 MWh | 1.17–2.04 MWh | ❌ 2.2× too high |
| Annual cooling | 1.00 MWh | 2.13–3.67 MWh | ❌ 2× too low |
| Peak heating | 3.74 kW | 1.10–2.10 kW | ❌ 1.7× too high |
| Peak cooling | 1.31 kW | 2.10–3.50 kW | ❌ ~2× too low |
| 900FF max temp | 32.50°C | 41.8–46.4°C | ❌ 9°C too low |
| 900FF min temp | –13.82°C | –6.4 to –1.6°C | ❌ too cold |

**Test results**: 10 passed, 9 failed in `case_900` filter. No regressions on 600-series tests.

---

## Architecture Analysis

### What Works

1. **Per-surface conduction solver** (#857) — independent surface temperature tracking via backward Euler ✅
2. **Multi-node solver** (#858) — 9R4C backward Euler with wall/roof/floor/internal mass nodes ✅
3. **Sol-air temperature integration** (#863) — sol-air temps feed into per-surface exterior boundaries ✅
4. **Solar gain distribution** (#864) — gains routed to air/surface/mass nodes per ISO 13790 §C.1-C.3 ✅
5. **Warm-up period** (#865) — 14-day warm-up prevents phantom transient heating ✅
6. **Crank-Nicolson mass update** (#876) — ISO 13790 §C.4 formulation committed but not yet producing correct metrics ✅

### What's Broken

The system has two interacting problems:

**Problem A — HVAC demand coefficient is too high**
The `h_coeff = den / (2 * term_rest_1)` formula from the 5R1C network produces ~1250 W/K, while the HVAC demand at setpoint should be ~40 W/K (governed by the building's actual heat loss conductance). This overcounts heating by ~2× and undercounts cooling by ~2×.

**Problem B — Free-floating temperature is wrong**
900FF max is 32.50°C (should be 41.8–46.4°C). The multi-node solver's zone air temperature computation suppresses the peak by ~9°C, likely because:
- The mass temperature feedback loop still exists (multi-node temps fed back into 5R1C mass_temperatures)
- The H_tr_3 coupling conductance (~40 W/K) is 30× smaller than h_tr_ms (~1300 W/K), causing mass to respond too slowly to solar gains
- Solar gains may not be reaching the surface/mass nodes with sufficient magnitude

---

## Phase 4A: Fix HVAC Demand Coefficient

### Root Cause

**Issue #895**: The HVAC coefficient `h_coeff = den / (2 * term_rest_1)` is derived from the 5R1C network's total building conductance. For heavy-mass buildings, this includes the mass-to-surface path (`h_tr_ms × A_m ≈ 1300 W/K`), which is NOT the HVAC load. The actual HVAC load is the air-side conductance (`h_tr_is + h_ve ≈ 1270 W/K`), but even this overcounts because the 5R1C's `den / (2 * term_rest_1)` formula double-counts the series path.

### Required Changes

1. **#893 — HVAC coefficient formula fix**
   - File: `src/sim/thermal_model_physics.rs` (~L2640)
   - Replace `h_coeff = den / (2 * term_rest_1)` with a formula that uses the air-side conductance only
   - The correct coefficient for HVAC demand is: `Q_hvac = (h_tr_is + h_ve) × (T_set − T_free)`
   - This is the 5R1C air node energy balance: `0 = h_tr_is × (T_s − T_air) + h_ve × (T_out − T_air) + Q_hvac`
   - With `T_s` solved from the surface node: `T_s = (h_tr_ms × T_m + h_tr_is × T_air) / (h_tr_ms + h_tr_is)`
   - Simplified: `Q_hvac = (h_tr_is + h_ve) × (T_set − T_free)` where `T_free` is the free-floating air temperature
   - **Risk**: Medium — affects all 9R4C HVAC tests
   - **Effort**: Small (2–3 hours)

2. **#895 — HVAC sensitivity calibration**
   - The sensitivity-based approach (`dQ/dT`) was tried and rejected (overcounts by ~4×)
   - Instead, verify the new coefficient against EnergyPlus output for Case 900
   - Add unit test: `test_hvac_coefficient_matches_energyplus` with ±10% tolerance
   - **Risk**: Low (validation only)
   - **Effort**: Small (1–2 hours)

### Acceptance Criteria

- [ ] Annual heating: 1.17–2.04 MWh
- [ ] Annual cooling: 2.13–3.67 MWh
- [ ] Peak heating: 1.10–2.10 kW
- [ ] Peak cooling: 1.50–3.50 kW
- [ ] No regression on Cases 600/610/620/650

---

## Phase 4B: Fix Free-Floating Temperature

### Root Cause

**Issue #904**: 900FF max is 32.50°C (should be 41.8–46.4°C). This is the "smoking gun" — free-floating has no HVAC, so the error is purely in the thermal network.

The root cause is multi-factorial:

1. **#917 — Identical max temps for 600FF and 900FF**: Both produce ~32°C max, suggesting the multi-node solver's zone air temperature is dominating and the 5R1C free-float is being overridden. The multi-node solver computes `T_air = (h_tr_is × T_s + h_ve × T_out + phi_ia) / (h_tr_is + h_ve)`, but `T_s` (surface temperature) is being set from the previous zone temperature minus 0.5°C, not from the actual mass node temperatures.

2. **#918 — Ventilation term dominates**: `h_ve × T_out` in the air balance dominates `h_tr_is × T_s`, pulling the free-float temperature toward outdoor conditions. For Case 900, `h_ve = 21.7 W/K` while `h_tr_is = 1251 W/K`, so the surface term should dominate. But if `T_s` is wrong (set from `t_zone_prev - 0.5`), the balance is incorrect.

3. **#919 — 900FF has LARGER swing than 600FF**: This is physically wrong. Heavy mass should dampen the swing. The issue is that the multi-node solver's mass temperatures are being fed back into the 5R1C's `mass_temperatures`, corrupting the 5R1C computation.

4. **#920 — Solar distribution**: ASHRAE 140 sets `solar_distribution_to_air = 0.0`, meaning all solar goes to mass. But the multi-node solver may not be receiving the solar gains correctly.

### Required Changes

1. **#917 — Fix surface temperature computation in multi-node solver**
   - File: `src/sim/thermal_model_physics.rs` (~L2530)
   - Replace `let t_surface = t_zone_prev - 0.5` with conductance-weighted mass temperatures:
     ```rust
     let h_ms_w = solver.mass.wall.h_tr_ms;
     let h_ms_r = solver.mass.roof.h_tr_ms;
     let h_ms_f = solver.mass.floor.h_tr_ms;
     let h_ms_total = h_ms_w + h_ms_r + h_ms_f;
     let t_surface = (h_ms_w * solver.wall_temperature()
         + h_ms_r * solver.roof_temperature()
         + h_ms_f * solver.floor_temperature()) / h_ms_total;
     ```
   - This ensures the solver's surface temperature reflects actual mass node temperatures
   - **Risk**: Medium — may require re-tuning of HVAC coefficient
   - **Effort**: Small (1 hour)

2. **#918 — Decouple multi-node from 5R1C mass_temperatures**
   - File: `src/sim/thermal_model_physics.rs` (~L2470)
   - Remove the feedback loop where multi-node mass temperatures overwrite `self.0.mass_temperatures`
   - The 5R1C should own `mass_temperatures` exclusively; the multi-node solver maintains its own internal state
   - Add a comment explaining the decoupling rationale
   - **Risk**: Low (already partially done in #872)
   - **Effort**: Small (30 min)

3. **#919 — Verify 900FF swing reduction**
   - After fixing #917 and #918, verify that 900FF swing is smaller than 600FF
   - Add diagnostic test: `test_900ff_swing_smaller_than_600ff`
   - **Risk**: Low (diagnostic only)
   - **Effort**: Small (30 min)

4. **#920 — Investigate solar gain path**
   - File: `src/sim/thermal_model_physics.rs` (gain distribution section ~L2310)
   - Verify that solar gains reach the multi-node solver's mass nodes
   - Check that `phi_st` and `phi_m` are correctly computed and passed to `step_with_gains()`
   - Add diagnostic logging for solar gain magnitudes at each node
   - **Risk**: Low (diagnostic, no code change)
   - **Effort**: Small (1–2 hours)

### Acceptance Criteria

- [ ] 900FF max temp: 41.8–46.4°C
- [ ] 900FF min temp: –6.4 to –1.6°C
- [ ] 900FF swing < 600FF swing
- [ ] No regression on Cases 600/610/620/650

---

## Phase 4C: Fix Mass Dynamics

### Root Cause

**Issue #896**: The mass temperature phase lag is incorrect. The 900FF maximum should occur several hours after peak solar (thermal lag from mass), but the current model shows peak temperature coinciding with peak solar, suggesting the mass is not properly storing and releasing heat.

**Issue #921**: The `h_tr_ms` calibration may be wrong. ISO 13790 Annex D calibrates `h_tr_ms = 9.1 × A_m` across hundreds of detailed simulations. Our per-surface decomposition uses different values, creating an impedance mismatch.

### Required Changes

1. **#896 — Fix phase lag**
   - The Crank-Nicolson update (#876) should have improved phase lag, but the current 32.50°C max suggests the mass is still not storing enough heat
   - Verify that `Cm` (thermal capacitance) values are correct for Case 900 construction
   - Check that the backward Euler step in the multi-node solver uses the correct `dt` (3600s for hourly)
   - Add diagnostic: plot mass temperature vs. time to verify phase lag
   - **Risk**: Medium (may require capacitance tuning)
   - **Effort**: Medium (2–3 hours)

2. **#921 — Verify h_tr_ms calibration**
   - File: `src/sim/thermal_model_solvers.rs` (where h_tr_ms is computed)
   - ISO 13790 §D.3: `h_tr_ms = H_ms = 9.1 × A_m` where `A_m = 0.5 × A_t` for heavy construction
   - Verify our per-surface h_tr_ms values sum to the expected total
   - If values are wrong, correct the computation
   - **Risk**: Low (validation)
   - **Effort**: Small (1 hour)

### Acceptance Criteria

- [ ] 900FF peak occurs 2–4 hours after peak solar (thermal lag)
- [ ] h_tr_ms values match ISO 13790 Annex D calibration
- [ ] Mass temperature time constant ≈ 500h (per ISO 13790)

---

## Phase 4D: Regression Cleanup

### Issue #903: 600 Series Pre-existing Failures

22 pre-existing test failures in the 600 series. These are NOT caused by the multi-node changes (600 series uses `step_physics_5r1c`), but they need to be resolved for the overall test suite to pass.

### Required Changes

1. **#903 — Fix 600 series failures**
   - Audit all 22 failing tests to determine root causes
   - Likely causes: incorrect construction parameters, missing test fixtures, outdated reference values
   - Fix each failure with minimal code changes
   - **Risk**: Low (isolated to 600 series)
   - **Effort**: Medium (1–2 days)

---

## Dependency Graph

```
Phase 4A (#893, #895) ──────────┐
                                  ├──→ Final Validation (#861) → Close Epic #856
Phase 4B (#917, #918, #919, #920) ┤
                                  │
Phase 4C (#896, #921) ──────────┘
                                  │
Phase 4D (#903) ─────────────────┘
```

**Critical path**: Phase 4A + 4B (both must pass for metrics to be in range)
**Parallel work**: Phase 4C and 4D can run in parallel with 4A/4B

---

## Execution Plan

### Week 1

| Day | Task | Issue | Effort |
|-----|------|-------|--------|
| Mon | Fix HVAC coefficient formula | #893 | 3h |
| Mon | Fix surface temp in multi-node solver | #917 | 1h |
| Tue | Decouple multi-node from 5R1C mass | #918 | 1h |
| Tue | Run Case 900 tests, iterate | #861 | 3h |
| Wed | Verify h_tr_ms calibration | #921 | 1h |
| Wed | Fix phase lag | #896 | 3h |
| Thu | Solar gain path investigation | #920 | 2h |
| Thu | 900FF swing verification | #919 | 1h |
| Fri | HVAC sensitivity calibration | #895 | 2h |
| Fri | Run full validation suite | #861 | 3h |

### Week 2 (if needed)

| Day | Task | Issue | Effort |
|-----|------|-------|--------|
| Mon–Tue | Fix 600 series regressions | #903 | 1–2 days |
| Wed | Final validation, close epic | #861, #856 | 4h |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| HVAC coefficient fix breaks 600 series | Medium | High | Run 600 tests after each change; guard with feature flag |
| h_tr_ms recalibration cascades | Low | Medium | Validate incrementally; keep old values as fallback |
| Solar gain path fix causes instability | Low | Medium | Use backward Euler (unconditionally stable); add denominator guard |
| 600 series regressions require large refactor | Medium | Low | Isolate to `step_physics_5r1c` path; do not modify shared code |
| Crank-Nicolson produces negative mass temps | Low | High | Existing fallback handles negative denominator; add bounds check |

---

## Files to Modify

| File | Changes | Issues |
|------|---------|--------|
| `src/sim/thermal_model_physics.rs` | HVAC coefficient, surface temp, decoupling | #893, #917, #918 |
| `src/physics/multi_node_solver.rs` | Surface temp getter, gain injection | #917, #920 |
| `src/sim/thermal_model_solvers.rs` | h_tr_ms verification | #921 |
| `src/sim/thermal_integration.rs` | Crank-Nicolson phi_m_tot (already added) | #876 |
| `tests/ashrae_140_case_900.rs` | Update test expectations | #861 |
| `tests/ashrae_140_free_floating.rs` | Update 900FF expectations | #904 |

---

## References

- ISO 13790:2008 Annex C (§C.1-C.13) — Crank-Nicolson mass update
- ISO 13790:2008 Annex D (§D.3) — h_tr_ms calibration
- ASHRAE 140-2017 §B2 — Case 900 reference metrics
- RC_BuildingSimulator (ETH Zurich) — H_tr_3 formulation
- `docs/implementation-plans/ashrae-140-case-900-fix-plan.md` — Root cause analysis
- `docs/implementation-plans/issue-876-iso13790-crank-nicolson.md` — CN implementation plan

---

## Success Criteria (Epic #856 Closure)

All of the following must pass:

- [ ] Annual heating: 1.17–2.04 MWh
- [ ] Annual cooling: 2.13–3.67 MWh
- [ ] Peak heating: 1.10–2.10 kW
- [ ] Peak cooling: 1.50–3.50 kW
- [ ] 900FF max temp: 41.8–46.4°C
- [ ] 900FF min temp: –6.4 to –1.6°C
- [ ] 900FF swing < 600FF swing
- [ ] All 19 Case 900 tests pass
- [ ] No regressions on Cases 600/610/620/650
- [ ] Full test suite passes (2457+ lib tests)
- [ ] `MultiNodeHvacRunner` deprecated
