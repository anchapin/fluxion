# Result — Issue #1522: Case 600 strict CI

**Status**: PARTIAL — option (a) air-node capacitance investigated and found INFEASIBLE at 1 h timestep. Structural improvements shipped (air_thermal_capacitance field + Cm correction); 1 of 14 failing tests flipped green (14/13 vs baseline 13/14). 13 tests remain failing — option (c) GaugeSolver or sub-hour air-node sub-stepping required to close them.

## Summary

Issue #1522 tasked closing 14 of 27 failing tests in `tests/ashrae_140_case_600_series.rs` via option (a): "restore a real capacitance on the air node." Investigation found that the air-node ODE time constant (τ_air ≈ 0.28 h) is much smaller than the 1-hour simulation timestep, so any air-node damping either has negligible effect (exact exponential: 1.6% carry-over) or over-damps peak_heating (implicit Euler: 22% carry-over, pushes peak_heating from −24% to −40%). The root cause is solar_distribution_to_air = 0.7 over-injecting solar into the air node (free-float peak ~62°C vs EnergyPlus ~50°C), which cannot be fixed by air-node capacitance alone.

## Files Changed

| File | Lines | Purpose |
|------|-------|---------|
| `src/sim/thermal_model_data.rs` | +14 | Add `air_thermal_capacitance: T` field + Clone impl |
| `src/sim/thermal_model_core.rs` | +20 | Populate field in from_spec; remove air_cap from Cm; struct init |
| `src/sim/thermal_model_physics/physics_impl.rs` | +45 | Air-node ODE scaffolding (disabled, documented why) |
| `docs/KNOWN_ISSUES.md` | +58 | LIMIT-05 UPDATE (#1522 investigation) |

## Acceptance Criteria Checklist

- [x] No regressions to 2716 lib tests (was 2716, still 2716)
- [x] CTF step-response (#1417): 126 passed, 0 failed
- [x] Case 900 regression (#1420): 13/4/1 — pre-existing, unchanged
- [x] Free-float tests pass: 24/24
- [x] Setback/ventilation tests pass
- [ ] All 27 tests in ashrae_140_case_600_series pass (14 pass / 13 fail — option (a) cannot close the remaining 13)
- [ ] Annual Case 600 cooling within ±15% band
- [ ] Peak Case 600 cooling within band
- [ ] Free-float (600FF, 900FF) min temp in band

## Approach Chosen

**Option (a)** — investigated thoroughly, found INFEASIBLE at 1 h timestep.
**Option (b)** — probed (forced 9R4C routing for Case 600), found WORSE (11/16).
**Option (c)** — out of scope per task instructions.

## Structural Fix Applied

1. `air_thermal_capacitance` field added to `ThermalModelData` (C_air = ρ·cp·V per zone)
2. `air_cap` removed from mass-node `Cm` (was incorrectly lumped onto the slow mass node)
3. Air-node ODE scaffolding implemented but disabled (3 integration methods tested, all fail)

## Test Results

| Suite | Before | After |
|-------|--------|-------|
| ashrae_140_case_600_series | 13/14 | **14/13** (1 flipped: Case 620 annual_cooling) |
| lib tests | 2716 pass | 2716 pass (no regressions) |
| CTF step-response | 126 pass | 126 pass |
| Case 900 series | 13/4/1 | 13/4/1 (pre-existing failures unchanged) |
| Free-floating | 24 pass | 24 pass |
| Setback/ventilation | pass | pass |

## Blockers

13 of 14 failing tests cannot be closed without option (c) GaugeSolver revival or sub-hour air-node sub-stepping. The fundamental limitation is the 1-hour ASHRAE 140 timestep vs the 0.28-hour air-node time constant.
