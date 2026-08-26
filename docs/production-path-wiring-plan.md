# Production-Path Wiring Plan: GaugeSolver as Default Thermal Solver

## Issue
[#3213](https://github.com/anchapin/fluxion/issues/3213): GaugeSolver production-path wiring is dead code blocking all LIMIT-* issues

## Problem Statement

The `#[cfg(feature = "gauge-solver")]` branch in `step_dispatcher.rs:66-130` is documented as "always dead" because:
1. `conduction_backend.gauge_zone_solver` is initialized to `None`
2. No code path sets it to `Some`
3. The block is also gated behind `#[cfg(feature = "gauge-solver")]` feature flag

ADR-0007 records the production-path switchover scope but it has not landed.

## Root Cause Analysis

### Immediate Dead Code Issue
```rust
// conduction_backend.rs
#[cfg(feature = "gauge-solver")]
pub gauge_zone_solver: Option<GaugeZoneSolver>,  // Always None

// step_dispatcher.rs
#[cfg(feature = "gauge-solver")]
if let Some(ref mut gauge_solver) = self.0.conduction.backend.gauge_zone_solver {
    // This block is unreachable because gauge_zone_solver is always None
}
```

### Architectural Gap
`GaugeZoneSolver.add_opaque_surface()` requires a `WallSpec` (layer properties: thickness, conductivity, density, specific heat), but the thermal model only stores `WallSurface` (area, orientation, window_area, U-value).

**Conversion is lossy:**
- `WallSurface.u_value` → Cannot uniquely determine layer structure
- A wall with U=0.5 W/m²K could be: 100mm insulation, OR 200mm concrete + 50mm insulation, OR...

## Required Changes

### Phase 1: Data Layer (Needed before GaugeSolver can work)
- [ ] Store `WallSpec` (or equivalent layer information) in `WallSurface` or a parallel structure
- [ ] Ensure layer information flows from `CaseSpec` → `ThermalModel` → `GaugeZoneSolver`
- [ ] Update `from_spec()` to capture construction layer data

### Phase 2: Initialization Path
- [ ] Create `enable_gauge_solver()` method following `enable_ctf()` / `enable_fd()` pattern
- [ ] Add surfaces to `GaugeZoneSolver` during model initialization
- [ ] Call `initialize()` on the solver before first timestep

### Phase 3: Dispatch Path
- [ ] Remove `#[cfg(feature = "gauge-solver")]` gate from step_dispatcher gauge block
- [ ] Make `gauge_zone_solver: Option<GaugeZoneSolver>` available in default build
- [ ] Route `step_physics_5r1c` / `step_physics_9r4c` to GaugeSolver
- [ ] Implement fallback to legacy solvers if GaugeSolver fails

### Phase 4: Validation
- [ ] All ASHRAE 140 cases pass without parameter tuning
- [ ] LIMIT-* issues flip from FAIL to PASS
- [ ] Energy conservation maintained

## Key Files

| File | Current State | Required Change |
|------|---------------|-----------------|
| `src/sim/thermal_model_data/conduction_backend.rs` | `gauge_zone_solver: None` always | Initialize properly |
| `src/sim/thermal_model_physics/step_dispatcher.rs` | Gauge block behind `#[cfg(feature = "gauge-solver")]` | Remove gate, wire to production |
| `src/sim/thermal_model_solvers.rs` | Has `enable_ctf()`, `enable_fd()` | Add `enable_gauge_solver()` |
| `src/sim/construction.rs` | `WallSurface` has no layer info | Store layer data |
| `src/physics/gauge_zone_solver.rs` | `add_opaque_surface(wall: &WallSpec, ...)` | Already correct interface |

## References

- ADR-0007: https://github.com/anchapin/fluxion/blob/main/docs/adr/0007-gauge-solver-structural-work.md
- Issue #3072: LIMIT-* cohort meta-issue
- Issue #1462: Phase 1b shadow-mode GaugeSolver (Closed)
- Issue #1465: Phase 3 GaugeSolver validation (Closed)

## Status

**This is a large architectural change. A single PR cannot complete this work.**

Created tracking issue: **#3213** (this issue)
