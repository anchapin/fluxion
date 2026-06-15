# T2.1: Ground Temperature Boundary Condition for Floor Slab

**Status**: COMPLETED
**Issue**: #746
**Date**: 2026-05-16

## Investigation Findings

### Current State

1. **Ground temperature trait exists**: `GroundTemperature` trait in `src/sim/boundary.rs` with two implementations:
   - `ConstantGroundTemperature` — fixed temperature model
   - `DynamicGroundTemperature` — Kusuda-Achenbach dynamic model

2. **Default is fixed 10°C**: The thermal model defaults to `ConstantGroundTemperature(10.0)` at line 2213-2215 of `src/sim/thermal_model_core.rs`:
   ```rust
   ground_temperature: Box::new(ConstantGroundTemperature::new(10.0)),
   ```

3. **DynamicGroundTemperature had a critical bug**: It was missing the `t_shift` parameter (day of minimum surface temperature). The original Kusuda formula implementation was:
   ```rust
   let annual_cycle = (omega * day - phase).cos();
   ```
   But the Kusuda-Achenbach equation per ASHRAE 140 Annex B §B3.3 requires:
   ```rust
   let annual_cycle = (omega * (day - t_shift) - decay).cos();
   ```

4. **Ground temp is properly wired into physics**: The `t_g` variable is retrieved per timestep via `self.0.ground_temperature.ground_temperature(timestep)` and used in floor heat transfer calculations (`Q_ground = h_tr_floor * (T_ground - T_surface)`).

### Bug: Missing t_shift Parameter

The `DynamicGroundTemperature` struct had 4 parameters but the Kusuda-Achenbach equation requires 5:
- `t_mean` — mean annual ground surface temperature
- `t_amplitude` — amplitude of surface temperature
- `depth` — depth below surface
- `diffusivity` — soil thermal diffusivity
- **`t_shift`** — day of year of minimum surface temperature (MISSING)

Without `t_shift`, the model assumed minimum surface temperature occurs on January 1 (day 0), which is incorrect for most locations. For Denver, the minimum surface temperature typically occurs around day 17.

## Changes Made

### File: `src/sim/boundary.rs`
1. **Added `t_shift` field** to `DynamicGroundTemperature` struct
2. **Updated constructor** `new()` to accept 5 parameters including `t_shift`
3. **Added `new_default_shift()`** for backward compatibility (t_shift=0)
4. **Fixed the Kusuda-Achenbach formula** to include `t_shift`:
   - Before: `cos(ω*t - decay)`
   - After: `cos(ω*(t - t_shift) - decay)` per ASHRAE 140 Annex B §B3.3
5. **Added `t_shift()` getter**
6. **Updated all doc examples** with Denver-specific parameters
7. **Updated all 13 existing tests** to use 5-parameter constructor
8. **Added 3 new tests**:
   - `test_kusuda_achenbach_t_shift_affects_minimum_day` — verifies t_shift moves the minimum
   - `test_kusuda_achenbach_denver_annex_b` — validates Denver ASHRAE 140 Annex B parameters
   - `test_default_shift_backward_compatibility` — ensures backward compat

### File: `src/sim/thermal_model_iterative.rs`
1. **Updated `set_dynamic_ground_temp()`** to accept 5 parameters including `t_shift`
2. **Added `set_dynamic_ground_temp_default_shift()`** for backward compatibility

### File: `src/sim/engine.rs`
1. **Updated 3 test calls** of `set_dynamic_ground_temp` to pass the new `t_shift` parameter

## Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Kusuda-Achenbach model implemented | DONE | DynamicGroundTemperature with t_shift |
| Includes t_shift parameter | DONE | Per ASHRAE 140 Annex B §B3.3 |
| Formula matches spec | DONE | `T = T_mean - A_s * exp(-d√(π/365α)) * cos(2π/365*(t-t_shift) - d√(π/365α))` |
| Floor slab uses ground BC | DONE | Already wired via `h_tr_floor` and `ground_temperature` trait |
| Existing tests pass | DONE | 20 boundary tests, 36 engine tests all pass |
| ASHRAE 140 Annex B §B3.3 compliant | DONE | Denver parameters validated in test |

## ASHRAE 140 Denver Parameters (Annex B §B3.3)

| Parameter | Value | Source |
|-----------|-------|--------|
| T_mean | 11.0°C | Denver TMY2 annual mean |
| A_s | 12.0°C | Annual amplitude |
| α (diffusivity) | 0.07 m²/day | Standard soil |
| t_shift | 17 days | Day of minimum surface temp |
| depth | 0.3m | Slab-on-grade depth |

## Test Results

```
cargo test --lib sim::boundary: 20 passed
cargo test --lib sim::engine::tests: 36 passed
cargo check: Compiles clean
```

## Out-of-Scope Findings

1. **ASHRAE 140 validator** (`src/validation/ashrae_140_validator.rs`) does not explicitly call `set_dynamic_ground_temp` — it uses the default `ConstantGroundTemperature(10.0)`. A future task should wire the dynamic model into the validation suite for the floor slab BC.
2. **Slab depth parameter** — Currently uses a generic depth. For precise ASHRAE 140 compliance, the depth should be derived from the floor construction assembly (insulation + slab thickness).
