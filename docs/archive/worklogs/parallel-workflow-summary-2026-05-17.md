# Parallel Issue Workflow Summary — 2026-05-17

## Top 5 Issues Ranked

| Rank | Issue | Score | Labels | Status |
|------|-------|-------|--------|--------|
| 1 | #848 Case 900 annual cooling | 81.9 | bug, ashrae-140 | Pre-existing failure |
| 2 | #849 Case 900 peak cooling | 81.9 | bug, ashrae-140 | Pre-existing failure |
| 3 | #850 Case 900 peak heating | 81.9 | bug, ashrae-140 | Pre-existing failure |
| 4 | #872 Replace 5R1C with multi-node | 45.1 | enhancement | Partial fix (cleanup) |
| 5 | #871 Air balance API | 40.1 | enhancement | Already implemented |

## Thread Results

### Thread 1: #872 — HVAC Formula Fix (9R4C Path)
- **Status**: Partial — sensitivity removal cleanup committed
- **What changed**: Removed `derived_sensitivity` field and its computation from `update_optimization_cache()`. Cleaned up all references in `thermal_model_physics.rs`, `thermal_model_solvers.rs`, `thermal_model_core.rs`, `thermal_model_data.rs`.
- **HVAC formula**: Attempted `Q = h_tr_is × (T_set - T_free)` replacement. Self-consistent and mathematically correct, but **caused regression** (heating 6.88→89.33 MWh) because multi-node solver corrupts `t_i_free` values (900FF max=27.73°C vs reference 41.8–46.4°C).
- **Lesson**: The h_loss formula is correct in isolation but requires the multi-node solver (#872 full) to provide correct free-floating temperatures first.

### Thread 2: #871 — Multi-Node Air Balance API
- **Status**: Already implemented ✅
- All three methods exist in `src/physics/multi_node_solver.rs`:
  - `compute_zone_air_temperature()` (L194) — enhanced with 3 params
  - `compute_hvac_demand()` (L238) — exact match to issue spec
  - `set_surface_temperature()` (L403) — direct assignment
- 18/18 solver tests pass

### Thread 3: #848/#849/#850 — Case 900 Validation
- **Status**: All pre-existing failures confirmed
- Case 900 test results: 10 passed / 7 failed / 1 ignored (same as baseline)
- 600-series: 4 passed / 22 failed (same as baseline, pre-existing)
- No new regressions introduced

### Thread 4: #851 — 600-Series Heating Overestimate
- **Status**: Root cause identified ✅
- **Finding**: Same `ideal_loads` formula bug — HVAC capacity 57.6× undersized for low-mass buildings
- The fix for #872 (h_loss formula) applies equally to 5R1C path used by 600-series
- Report: `docs/investigations/issue-851-600-series-heating.md`

### Thread 5: #859 — Per-Surface Gain Distribution
- **Status**: 3 critical gaps documented ✅
- **Gap 1**: `A_m/A_t` ratio never used — code uses fixed fractions instead
- **Gap 2**: Solar gains aggregated per-zone, per-surface breakdown discarded
- **Gap 3**: Opaque solar bypasses A_m/A_t split
- **4-phase plan** proposed with feature flag for safe rollout
- Report: `docs/analysis/issue-859-gain-distribution.md`

## Critical QA Finding

The QA thread caught a regression when applying the h_loss formula in isolation:
- **Before**: 6.88 MWh heating (baseline, wrong but stable)
- **After**: 89.33 MWh heating (h_tr_is formula, mathematically correct but garbage-in-garbage-out)

**Root cause**: The 9R4C multi-node solver feedback loop corrupts `t_i_free`:
- 900FF max temperature: 27.73°C (reference: 41.8–46.4°C)
- The free-floating temperature is ~15°C too low
- With h_tr_is formula, HVAC perfectly tracks setpoint on the corrupted t_free
- Result: HVAC fires almost every hour, accumulating massive energy

**Resolution**: Reverted to ideal_loads formula for now. The fix requires:
1. First fix multi-node solver feedback (#872 full implementation)
2. Then apply h_loss formula (which is mathematically proven correct)

## Dependency Graph

```
#859 (gain distribution) ─── independent ───→ can proceed anytime
#851 (600-series heat)   ─── same root cause as #872 ───→ fixed together
#871 (air balance API)   ─── already done ✅
#872 (HVAC formula)      ─── needs multi-node solver fix FIRST
                              └→ #873 (per-node gains) + #874 (solver improvements)
#848/#849/#850           ─── all fixed by #872 chain
```

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `src/sim/thermal_model_physics.rs` | Removed sensitivity references, cleaned up comments | +40/-65 |
| `src/sim/thermal_model_solvers.rs` | Removed derived_sensitivity computation | +3/-11 |
| `src/sim/thermal_model_core.rs` | Removed derived_sensitivity field | -1 |
| `src/sim/thermal_model_data.rs` | Removed derived_sensitivity field | -2 |

## Files Created (Research/Analysis)

| File | Description |
|------|-------------|
| `docs/investigations/issue-851-600-series-heating.md` | Root cause analysis for 600-series heating overestimate |
| `docs/analysis/issue-859-gain-distribution.md` | Per-surface gain distribution gap analysis |
| `docs/implementation-plans/issue-872-hvac-formula-fix.md` | Pre-existing implementation plan (unchanged) |

## Build & Test Status

- `cargo check`: ✅ Clean
- `cargo test --lib`: ✅ 2457 passed, 2 ignored
- Case 900 tests: 10 passed / 7 failed (baseline: same)
- 600-series tests: 4 passed / 22 failed (baseline: same)
- **No regressions introduced**
