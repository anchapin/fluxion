# T2.3: Implement Warm-up / Pre-conditioning Period

**Status**: ✅ COMPLETE
**Issue**: #744

## Summary

Implemented ASHRAE 140 §B2-compliant warm-up / pre-conditioning period for annual simulations. The warm-up runs before results are collected to ensure temperatures converge to periodic steady state from the default 20°C initial condition.

## Approach

### Architecture

Created a new `warmup` module (`src/sim/warmup.rs`) with:

1. **`WarmupConfig`** — Configurable warm-up settings:
   - `warmup_days`: Fixed-duration warm-up in days (default: 14 per ASHRAE 140 §B2)
   - `use_convergence`: Optional convergence-based warm-up (iterates full years until ΔT < threshold)
   - `convergence_threshold`: 0.01°C default (per ASHRAE 140 §B2 guidance)
   - `max_iterations`: 4 full-year cap for convergence mode
   - `enabled`: Toggle to disable warm-up (legacy behavior)
   - Builder pattern: `WarmupConfig::fixed_days(14)`, `WarmupConfig::convergence()`, etc.

2. **`run_warmup()`** — Core function that advances a `ThermalModel` through the warm-up period:
   - Uses `(hour % 8760)` for periodic weather data wrapping
   - Supports both fixed-duration and convergence-based approaches
   - Returns `WarmupResult` with timesteps, max ΔT, convergence status

3. **`WarmupResult`** — Diagnostic output from warm-up phase

### Integration Points

- Exported via `src/sim/engine.rs`: `WarmupConfig`, `WarmupResult`, `run_warmup`
- Applied to test helpers in `tests/ashrae_140_free_floating.rs`:
  - `simulate_free_float_case()` — Added 14-day warm-up before recording
  - `simulate_free_float_with_time_series()` — Added 14-day warm-up before recording
- Applied to `tests/ashrae_140_case_900.rs`:
  - `simulate_case_900ff()` — Added 14-day warm-up

### How It Works

1. Model initializes at 20°C (default)
2. Warm-up runs 14 days (336 hours) using Denver TMY weather data (wrapping from Jan 1)
3. During warm-up, `step_physics()` is called but results are discarded
4. After warm-up, the model's temperatures reflect realistic January conditions
5. The recording period then runs 8760 hours starting from the converged state

### Convergence Mode (Optional)

For cases requiring stricter convergence:
```rust
let config = WarmupConfig::convergence()
    .with_warmup_days(7)
    .with_convergence_threshold(0.01)
    .with_max_iterations(4);
```

This runs an initial 7-day warm-up, then iterates full years until max temperature change between consecutive years is < 0.01°C, up to 4 iterations.

## Files Changed

| File | Change |
|------|--------|
| `src/sim/warmup.rs` | **NEW** — Warm-up module (WarmupConfig, run_warmup, WarmupResult) |
| `src/sim/mod.rs` | Added `pub mod warmup` |
| `src/sim/engine.rs` | Added re-exports for WarmupConfig, WarmupResult, run_warmup |
| `tests/ashrae_140_free_floating.rs` | Updated `simulate_free_float_case` and `simulate_free_float_with_time_series` with warm-up; added `test_warmup_period_improves_steady_state` test |
| `tests/ashrae_140_case_900.rs` | Updated `simulate_case_900ff` with warm-up |

## Test Results

| Test Suite | Result |
|------------|--------|
| `sim::warmup` (7 unit tests) | ✅ 7/7 passed |
| `ashrae_140_free_floating` (14 tests) | ✅ 14/14 passed |
| `ashrae_140_case_900` (FF subset) | ✅ All FF tests passed |
| `free_floating_temperature_validation` | ✅ 3/3 passed |
| Full library compilation | ✅ Clean |

## Acceptance Criteria Checklist

- [x] Annual simulation runs a warm-up period per ASHRAE 140 §B2
- [x] Default warm-up is 14 days (configurable)
- [x] Results represent periodic steady state (warm-up results discarded)
- [x] Weather data wraps periodically (hour % 8760)
- [x] Warm-up is configurable (fixed-duration or convergence-based)
- [x] Warm-up can be disabled for legacy behavior
- [x] Free-floating test cases updated to use warm-up
- [x] All existing passing tests continue to pass
- [x] New test verifies warm-up effect on initial conditions
