# Result: Issue #863 — Wire Sol-Air Temperatures into Multi-Node HVAC Runner

**Status**: COMPLETE
**PR**: https://github.com/anchapin/fluxion/pull/869
**Branch**: fix/863-sol-air-temps

## Summary

Replaced uniform `outdoor_temp` as the exterior boundary for all multi-node solver envelope nodes with per-surface sol-air temperatures computed from weather data.

## Files Changed

1. **`src/physics/multi_node_solver.rs`** (+69 lines)
   - Added `SurfaceExteriorTemperatures` struct (wall, roof, floor boundary temps)
   - Added `set_surface_exterior_temperatures()` method
   - Modified `step_backward_euler()` to use per-surface temps when available
   - Added `exterior_temperatures: Option<SurfaceExteriorTemperatures>` field

2. **`src/sim/thermal_model_physics.rs`** (+70 lines)
   - Added imports: `SolAirTemperature`, `calculate_solar_position`, `calculate_surface_irradiance`, `SurfaceExteriorTemperatures`, `Orientation`
   - In `step_physics_9r4c()`: compute solar position from timestep, wall/roof irradiance, sol-air temps
   - Pass `SurfaceExteriorTemperatures` to solver via new method
   - Fallback to uniform `outdoor_temp` when no weather data

## Acceptance Criteria Checklist

- [x] Sol-air temperatures computed for wall and roof in `step_physics_9r4c`
- [x] Passed to solver via `set_surface_exterior_temperatures()` instead of raw `outdoor_temp`
- [x] `cargo test` passes (2430 passed, 1 pre-existing unrelated failure)
- [x] Committed and pushed to `fix/863-sol-air-temps`
- [x] PR created: #869

## Test Results

- `cargo check`: clean (no warnings from changed files)
- `cargo test --lib`: 2430 passed, 1 failed (pre-existing `analyzer::test_quality_metrics_mae_calculation`)
- Multi-node solver tests: 11/11 passed
- Thermal model tests: 50/50 passed
- Sky radiation tests: 74/74 passed
