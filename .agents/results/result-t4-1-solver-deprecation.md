# T4.1: Deprecate SolverManager Legacy Methods

**Status**: DONE
**Issue**: #712
**Summary**: Removed 4 deprecated methods from `SolverManager`, consolidated to `step_all()` as the sole batch-stepping interface. Removed 7 tests that exercised deprecated API. All 2432 lib tests pass.

## Files Changed

- `src/physics/solver_manager.rs`
  - Removed `get_solver_mut()` (deprecated wrapper exposing registry internals)
  - Removed `get_solver()` (deprecated wrapper exposing registry internals)
  - Removed `step()` (deprecated single-wall convenience method)
  - Removed `energy_storage_rate()` (deprecated single-wall accessor)
  - Removed 7 deprecated tests: `test_solver_manager_step`, `test_solver_manager_get_solver_mut`, `test_solver_manager_get_solver`, `test_solver_manager_get_solver_not_found`, `test_solver_manager_energy_storage_rate`, `test_solver_manager_energy_storage_rate_not_found`, `test_solver_manager_step_invalid_wall`
  - Updated module doc example to show `step_all()` instead of `step()`

## Acceptance Criteria Checklist

- [x] All code uses `step_all()` — no callers of deprecated methods remain
- [x] Deprecated methods removed — `get_solver_mut`, `get_solver`, `step`, `energy_storage_rate` deleted
- [x] No external callers were affected — all deprecated method usage was internal tests only
- [x] Tests pass — 2432 passed, 0 failed, 2 ignored

## Notes

- `step_all()` internally uses `self.registry.get_solver_mut()` directly (the registry method, not the removed wrapper). No change to that path.
- `all_valid()` also uses `self.registry.get_solver()` directly — unaffected.
- External `#[allow(deprecated)]` in `adaptive_timestep.rs`, `test_automation.rs`, `cli_integration.rs` are for chrono's deprecated `from_timestamp_opt`, not SolverManager.
