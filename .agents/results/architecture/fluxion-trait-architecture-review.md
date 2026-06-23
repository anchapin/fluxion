# Fluxion Trait Architecture Review

**Date**: 2026-06-22
**Reviewer**: Architecture Specialist
**Scope**: Trait design, module boundaries, ML surrogate swap points, ADR-002 dual-path issue

---

## Executive Summary

The trait hierarchy is **architecturally sound** for runtime polymorphic dispatch and ML surrogate swapping, but suffers from **three critical design flaws** that violate documented contracts and cause behavioral surprises:

1. **`VentilationSchedule::get_ach` ignores weather parameters** — the trait signature does not match the documented contract
2. **Architecture drift (ADR-002)** — Module 3 `HeatConductionSolver` is not the zone solver; Module 5 thermal network is
3. **9R4C vs 5R1C routing is opaque** — gated by string-matching on `case_id`, not a proper type-level selection

---

## 1. Trait Design Issues

### CRITICAL: `VentilationSchedule::get_ach` — Contract Violation

**File**: `src/sim/ventilation.rs:323-329`

```rust
impl VentilationSchedule for WeatherDependentVentilation {
    fn get_ach(&self, _hour: usize) -> f64 {
        self.base_ach  // ❌ Ignores ALL weather parameters!
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> { ... }
}
```

**Problem**: The ARCHITECTURE.md Module 4 contract states:
> **Input**: Outdoor temperature, Indoor temperature, Wind speed

But `get_ach(&self, hour: usize)` takes **no weather parameters**. The `WeatherDependentVentilation` struct has `get_ach_weather(outdoor_temp, indoor_temp, wind_speed, zone_volume)` which does the correct calculation, but this method is **not part of the trait**.

**Impact**: Any code using `dyn VentilationSchedule` will silently get `base_ach` regardless of conditions:

```rust
let vent: Box<dyn VentilationSchedule> = Box::new(WeatherDependentVentilation::new(...));
// This IGNORES outdoor temp, indoor temp, wind speed, zone_volume!
let ach = vent.get_ach(hour);  // Always returns base_ach
```

**Fix**: Change the trait signature to include weather context:

```rust
pub trait VentilationSchedule: Debug + Send + Sync {
    fn get_ach(&self, hour: usize) -> f64;
    // NEW: Weather-aware ACH calculation
    fn get_ach_with_weather(
        &self,
        hour: usize,
        outdoor_temp: f64,
        indoor_temp: f64,
        wind_speed: f64,
        zone_volume: f64,
    ) -> f64;
}
```

Or rename the current method to `get_base_ach()` and add the weather-aware version as `get_ach()`.

---

### CRITICAL: Architecture Drift — Module 3 vs Module 5 (ADR-002)

**Files**: `src/physics/solver_trait.rs` vs `src/sim/thermal_model_core.rs`

**Problem**: The ARCHITECTURE.md line 212 documents `HeatConductionSolver` as the interface for "5R1C, CTF, and FD methods" and implies this drives zone heat balance. **ADR-002 reveals this is false.**

| Path | Location | Drives zone balance? |
|------|----------|---------------------|
| `FiveR1CSolver` (Module 3) | `physics/five_r1c_solver.rs` | **No** — steady-state only |
| Zone-level ISO 13790 network | `thermal_model_core.rs` (Module 5) | **Yes** — `step_physics_5r1c/6r2c/8r3c/9r4c` |

The `FiveR1CSolver::step()` **ignores timestep, h_interior, h_exterior** — it computes only `Q = ΔT / R_total`. This is confirmed by the isolation test (`tests/conduction_5r1c_isolation.rs`).

**Impact**: Developers expecting Module 3 to be the zone solver will make incorrect architectural decisions.

**ADR-002 Note**: The ADR correctly identifies this as "architecture drift" and resolves it by documenting the two paths. But the code structure still implies Module 3 is primary.

**Fix**: Update ARCHITECTURE.md line 212 to explicitly state:
> The `HeatConductionSolver` trait (Module 3) provides **per-surface** conduction solvers. The **zone-level** thermal network lives in Module 5 and is NOT accessible through `HeatConductionSolver` — it is internal to `ThermalModel`.

---

### HIGH: Hidden 9R4C/5R1C Routing via String Matching

**File**: `src/sim/thermal_model_core.rs:1865-1869`

```rust
if (spec.case_id.starts_with("9") && spec.case_id != "960")
    || (spec.case_id.starts_with("6")
        && !["600FF", "650FF"].contains(&spec.case_id.as_str()))
{
    model.enable_9r4c_model();
}
```

**Problem**: The routing decision is based on string pattern matching on `case_id`, not a proper type-level selection. This is:

1. **Brittle** — if case naming changes, routing silently breaks
2. **Opaque** — it's impossible to determine which path is active without running the code
3. **Not testable** in isolation

**Fix**: The `ThermalModelType` enum already exists. Selection should use:
```rust
match self.thermal_model_type {
    ThermalModelType::FiveROneC => self.step_physics_5r1c(...),
    ThermalModelType::SixRTwoC => self.step_physics_6r2c(...),
    ThermalModelType::EightRThreeC => self.step_physics_8r3c(...),
    ThermalModelType::NineRFourC => self.step_physics_9r4c(...),
}
```

The `case_id` string matching in `from_spec` should set `thermal_model_type` once, not gate individual features.

---

### MODERATE: `ThermalModelTrait::solve_timesteps` — SurrogateManager Coupling

**File**: `src/sim/thermal_model.rs:60-65`

```rust
fn solve_timesteps(
    &mut self,
    steps: usize,
    surrogates: &SurrogateManager,  // ❌ Always passed, even when use_surrogates=false
    use_surrogates: bool,
) -> f64;
```

**Problem**: `SurrogateManager` is passed as a parameter even when `use_surrogates=false`. This:
- Creates coupling that shouldn't exist at this level
- Suggests the surrogate decision should be at a different layer

**Fix**: Consider whether surrogates should be swapped at the `HeatConductionSolver` or `SurfaceHeatFluxProvider` level instead of (or in addition to) the zone level.

---

### MODERATE: `SurfaceHeatFluxProvider` — Missing Film Coefficient Updates

**File**: `src/sim/surface_flux_provider.rs:31-55`

```rust
pub trait SurfaceHeatFluxProvider: Send + Sync {
    fn surface_heat_flux(&self, surface_idx: usize, T_zone: f64, T_outdoor: f64, dt_seconds: f64) -> f64;
    fn num_surfaces(&self) -> usize;
    fn name(&self) -> &str;
    // ❌ No way to update h_int/h_ext film coefficients at runtime
}
```

**Problem**: Film coefficients (`h_int`, `h_ext`) affect convection and change with surface orientation, wind speed, etc. The `PhysicsSurfaceFluxProvider` stores these but there's no trait method to update them.

**Fix**: Add to the trait:
```rust
fn set_film_coefficients(&mut self, surface_idx: usize, h_int: f64, h_ext: f64);
```

---

### LOW: `ThermalModelTrait::hvac_power_demand` — Scalar Output for Multi-Zone

**File**: `src/sim/thermal_model.rs:88`

```rust
fn hvac_power_demand(&self, timestep: usize, _outdoor_temp: f64) -> f64;
```

**Problem**: Returns a single `f64` but multi-zone models need per-zone power demands.

**Fix**: Return `Vec<f64>` or introduce a `ZonePowerDemand` type.

---

### LOW: `VentilationSchedule::ach_to_conductance` — Free Function vs Trait Method

**File**: `src/sim/ventilation.rs:254-255`

```rust
pub trait VentilationSchedule: Debug + Send + Sync {
    fn get_ach(&self, hour: usize) -> f64;
    fn ach_to_conductance(ach: f64, volume: f64, rho: f64, cp: f64) -> f64; // ❌ Static method
}
```

**Problem**: `ach_to_conductance` is a static utility function, not a method on `self`. Callers need both the trait object AND this separate function.

**Fix**: Either remove it from the trait (keep as free function) or make it an instance method with sensible defaults.

---

## 2. ML Surrogate Swap Point Analysis

### Current Architecture

According to ARCHITECTURE.md line 306:
> **ML Surrogate Path**: `SurrogateThermalModel` implements `ThermalModelTrait`

This means the swap point is at the **zone level** — the entire zone heat balance can be replaced by a neural network.

### Correct Swap Points (Based on Module Boundaries)

| Layer | Trait | Appropriate for Surrogate? |
|-------|-------|---------------------------|
| Per-surface conduction | `HeatConductionSolver` | ✅ Yes — deterministic, bounded I/O |
| Combined surface flux | `SurfaceHeatFluxProvider` | ✅ Yes — aggregates conduction + solar |
| Zone heat balance | `ThermalModelTrait` | ⚠️ Complex — multiple interacting physics |

### Issues

1. **Wrong abstraction level**: Swapping at `ThermalModelTrait` means the surrogate must replace ALL zone physics (conduction + solar + ventilation + internal gains + HVAC). This is:
   - Hard to train (needs full physics simulation data)
   - Hard to validate (can't isolate components)
   - Brittle (any building config change requires retraining)

2. **SurrogateManager coupling at zone level**: The `surrogates: &SurrogateManager` parameter in `solve_timesteps` couples the zone solver to the AI layer.

**Recommendation**: The per-surface swap at `HeatConductionSolver` and `SurfaceHeatFluxProvider` level is the **correct** approach for component-level ML surrogates. The zone-level swap via `UnifiedThermalModel` is appropriate only for whole-building surrogate models.

---

## 3. Module Boundary Problems

### `HeatConductionSolver` — Module 3 is Per-Surface, Not Zone-Level

The trait is correctly scoped for **per-surface** conduction calculation. The implementations (`FiveR1CSolver`, `CTFSolverWrapper`, `FDSolverWrapper`) operate on individual surfaces with independent boundary conditions.

**However**: The ARCHITECTURE.md documentation blurs this boundary by implying the 5R1C thermal network lives in Module 3.

### `ThermalModelTrait` — Zone-Level Solver is Internal

The zone heat balance solver (`step_physics_5r1c/6r2c/8r3c/9r4c`) is internal to `ThermalModel` and not exposed through any trait. This is a **hidden implementation detail** rather than a swappable interface.

**If** the intent is to make the zone solver swappable, a `ZoneSolverTrait` should be extracted.

---

## 4. Specific Code Locations Needing Fixes

| Issue | File | Lines | Priority |
|-------|------|-------|----------|
| `WeatherDependentVentilation::get_ach` ignores weather | `src/sim/ventilation.rs` | 323-329 | CRITICAL |
| ARCHITECTURE.md misleading Module 3 description | `ARCHITECTURE.md` | 212 | CRITICAL |
| 9R4C routing via `case_id` string matching | `src/sim/thermal_model_core.rs` | 1865-1869 | HIGH |
| SurrogateManager passed even when unused | `src/sim/thermal_model.rs` | 60-65 | MODERATE |
| Missing film coefficient update method | `src/sim/surface_flux_provider.rs` | 31-55 | MODERATE |
| `hvac_power_demand` returns scalar | `src/sim/thermal_model.rs` | 88 | LOW |
| `ach_to_conductance` in trait but static | `src/sim/ventilation.rs` | 254-255 | LOW |

---

## 5. Recommendations

### Immediate (Fix Current Behavior)

1. **Fix `WeatherDependentVentilation`** — the `get_ach` method must honor weather parameters or the struct should not implement `VentilationSchedule`

2. **Clarify ARCHITECTURE.md Module 3/5 boundary** — explicitly state that Module 3 is per-surface conduction, Module 5 is zone-level thermal network

3. **Replace `case_id` string matching with `ThermalModelType` enum** — make the 9R4C/5R1C selection explicit and type-checked

### Short-Term (Trait Improvements)

4. **Add weather parameters to `VentilationSchedule::get_ach`** — change signature to include `outdoor_temp`, `indoor_temp`, `wind_speed`, `zone_volume`

5. **Extract `ZoneSolverTrait`** if the intent is to make zone solver swappable at the trait level

6. **Decouple `SurrogateManager` from `solve_timesteps`** — consider whether surrogate selection should happen at a different layer

### Long-Term (Architecture)

7. **ML surrogates at component level** — prioritize `HeatConductionSolver` and `SurfaceHeatFluxProvider` swap points over zone-level swap

8. **Consider a `ThermalModelCoreTrait`** to abstract the 5R1C/6R2C/8R3C/9R4C implementations behind a common interface

---

## Validation Steps

To verify these issues:

1. **VentilationIssue**:
   ```rust
   let vent: Box<dyn VentilationSchedule> = Box::new(
       WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0)
   );
   // Get same ACH regardless of outdoor temp — bug!
   assert_eq!(vent.get_ach(12), vent.get_ach_weather(35.0, 25.0, 10.0, 100.0));
   ```

2. **9R4C Routing**: Add `fn thermal_model_type(&self) -> ThermalModelType` to `ThermalModelTrait` and assert it returns the expected type for Case 600 vs Case 900

3. **Surrogate Coupling**: Verify `SurrogateManager` is never accessed when `use_surrogates=false`

---

## Files Referenced

- `src/physics/solver_trait.rs` — HeatConductionSolver trait
- `src/sim/ventilation.rs` — VentilationSchedule trait
- `src/sim/thermal_model.rs` — ThermalModelTrait
- `src/sim/surface_flux_provider.rs` — SurfaceHeatFluxProvider trait
- `src/weather/mod.rs` — WeatherSource trait
- `src/sim/thermal_model_core.rs` — Zone-level thermal network (5R1C/6R2C/8R3C/9R4C)
- `src/sim/thermal_model_physics/step_dispatcher.rs` — Physics dispatch
- `docs/adr/0002-promote-9r4c-high-mass-default.md` — ADR-002
- `docs/adr/0003-5r1c-high-mass-limitations.md` — ADR-003
- `ARCHITECTURE.md` — Module contracts documentation
