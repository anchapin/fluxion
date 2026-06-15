# Phase 3: Multi-Node as Primary Engine — Implementation Plan

**Status**: READY FOR IMPLEMENTATION
**Date**: 2026-05-17
**Epic**: #856 — Full Multi-Node HVAC Energy Simulation for Case 900
**Fix Plan**: `docs/implementation-plans/ashrae-140-case-900-fix-plan.md`

## Overview

Phase 3 makes the multi-node solver the **primary thermal engine** for the 9R4C path, replacing the 5R1C sensitivity formula for zone temperature and HVAC demand. This is the "Option C — Coupled Multi-Node Air Balance" approach from the BEM engineer analysis.

## Current Metric Status

| Metric | Value | Reference Range | Gap |
|--------|-------|----------------|-----|
| Annual heating | 1.91 MWh | 1.17–2.04 MWh | ✅ PASS |
| Annual cooling | 0.04 MWh | 2.13–3.67 MWh | 50x too low |
| Peak heating | 0.42 kW | 1.10–2.10 kW | 3x too low |
| Peak cooling | 0.15 kW | 1.50–3.50 kW | 10x too low |
| 900FF max temp | 27.73°C | 41.8–46.4°C | 15°C too low |

## Issue Dependency Graph

```
#871 ──┐                    ┌──→ #872 ──→ #874 ──┐
(API)  ├────────────────────┤                      ├──→ #875 (Validate)
        │                    └──→ #873 ────────────┘
        └── (can start immediately)
```

### Parallel Tracks

- **Track A**: #871 → #872 → #874 (critical path)
- **Track B**: #871 → #873 (independent of #872, can run in parallel)
- **Final Gate**: #875 depends on all 4

### Recommended Execution Order

1. **#871** (Small-Med, ~2-3h) — Add API methods to `MultiNodeSolver`
2. **#872** (Medium, ~3-4h) — Replace 5R1C zone temp with multi-node (critical)
3. **#873** (Small, ~1-2h) — Wire gains into backward Euler (can parallel with #872)
4. **#874** (Small, ~1h) — Fix mass temp feedback (after #872)
5. **#875** (Medium, ~3-5h) — Validate, iterate, deprecate dead code

**Total estimated effort**: ~10-15 hours

## Issue Summary

### #871 — Add Multi-Node Air Balance API
**File**: `src/physics/multi_node_solver.rs`
**Risk**: Zero (additive only)
**What**: Add `compute_zone_air_temperature()`, `compute_hvac_demand()`, `step_with_gains()` methods + unit tests
**Effort**: Small-Medium (~2-3 hours)

### #872 — Replace 5R1C Zone Temperature with Multi-Node Air Balance
**File**: `src/sim/thermal_model_physics.rs`
**Risk**: Medium (only affects 9R4C path)
**What**: Replace 5R1C `t_i_free` with `solver.compute_zone_air_temperature()`, replace HVAC demand with `solver.compute_hvac_demand()`, replace hardcoded `t_surface = t_zone - 0.5` with conductance-weighted surface temp
**Effort**: Medium (~3-4 hours)
**Expected impact**: 900FF max should jump from 27.73°C toward 35-40°C; cooling should increase from 0.04 to 1-2 MWh

### #873 — Wire Gain Distribution into Multi-Node Solver
**File**: `src/sim/thermal_model_physics.rs`
**Risk**: Low
**What**: Call `solver.step_with_gains()` instead of `solver.step()`, distributing `phi_st` proportional to `h_tr_ms` and `phi_m` to internal mass
**Effort**: Small (~1-2 hours)
**Expected impact**: Surface temperatures increase from solar absorption, further raising cooling demand

### #874 — Fix Mass Temperature Feedback
**File**: `src/sim/thermal_model_physics.rs`
**Risk**: Low
**What**: Replace `(wall + roof + floor + internal) / 4` with conductance-weighted average
**Effort**: Small (~1 hour)
**Expected impact**: Cold floor node no longer drags down average by 25%

### #875 — Validate and Gate
**Files**: Multiple (test files, `multi_node_hvac_runner.rs`)
**Risk**: Medium (may require iteration)
**What**: Run full ASHRAE 140 validation, iterate if metrics outside range, deprecate `MultiNodeHvacRunner`
**Effort**: Medium (~3-5 hours)

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Case 600/610/620/650 regression | These use `step_physics_5r1c` — zero risk from 9R4C changes |
| Annual heating regression | Winter dominated by `h_ve × T_out` — structurally insensitive to mass changes |
| 900FF still too low | Check per-surface mass temps vs EnergyPlus BESTEST; tune h_ext if needed |
| Numerical instability | Backward Euler unconditionally stable; add `denom < 1e-6` fallback |
| Gain distribution wrong | phi_st proportional to h_tr_ms is standard ISO 13790 §C.4 approach |

## Success Criteria

All 5 Case 900 metrics within reference ranges:
- [ ] Annual heating: 1.17–2.04 MWh
- [ ] Annual cooling: 2.13–3.67 MWh
- [ ] Peak heating: 1.10–2.10 kW
- [ ] Peak cooling: 1.50–3.50 kW
- [ ] 900FF max temp: 41.8–46.4°C
- [ ] No regressions on Cases 600/610/620/650
- [ ] Full test suite passes
