# Result: Backend — Issue #864 Per-Surface Gain Distribution

## Status: DONE

## Summary

Connected per-surface solar and internal gain distribution to the 9R4C conduction solver mass nodes. The MultiNodeSolver now accepts gain terms per envelope surface, and `step_physics_9r4c` distributes opaque solar and radiative internal gains using `h_tr_em` conductances as area proxies.

## Files Changed

| File | Changes |
|------|---------|
| `src/physics/multi_node_solver.rs` | +67/-14: Added `step_with_gains()`, renamed `step_backward_euler` to accept 6 gain params, added gain terms to wall/roof/floor numerators, denom safety checks, 4 new tests |
| `src/sim/boundary.rs` | +133/-0: Added `SurfaceSolarGains` struct, `distribute_opaque_solar_gains()`, `distribute_radiative_gains()`, 7 distribution tests |
| `src/sim/thermal_model_physics.rs` | +44/-1: Added boundary import, replaced `solver.step(dt)` with `solver.step_with_gains(dt, ...)` including gain distribution from `opaque_solar_ref` and `loads_ref` |

## Commit

`1df707a` on branch `fix/864-per-surface-gains`

## Test Results

- **2438 passed** (including 4 new solver tests + 7 new distribution tests)
- **1 pre-existing failure**: `validation::analyzer::tests::test_quality_metrics_mae_calculation`
- **2 ignored** (pre-existing)

## Acceptance Criteria

- [x] `step_with_gains(dt, phi_m_wall, phi_m_roof, phi_m_floor, phi_st_wall, phi_st_roof, phi_st_floor)` added to MultiNodeSolver
- [x] `step(dt)` delegates to `step_with_gains(dt, 0, 0, 0, 0, 0, 0)` — backward compatible
- [x] `distribute_opaque_solar_gains()` with irradiance-weighted area distribution + area-only fallback
- [x] `distribute_radiative_gains()` distributing by area fraction
- [x] `step_physics_9r4c` wired to use gain distribution from opaque_solar_ref and loads_ref
- [x] All new code has unit tests
- [x] No regressions (2438 pass, same 1 pre-existing failure)
- [x] `cargo check` passes clean
- [ ] Git push blocked (SSL cert error) — manual push needed
- [ ] PR creation blocked (same SSL issue)

## Out-of-Scope Dependencies

- Issue #863 (per-surface exterior temps) changes are included in the same commit as they were already in the uncommitted working tree
- Git push requires SSL cert fix or manual push
