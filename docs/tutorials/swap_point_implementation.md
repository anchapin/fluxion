# Developer Guide: Implementing Swap-Point Traits

This guide explains how to implement custom swap-point traits in Fluxion, allowing you to replace or extend the built-in thermal modeling components.

## Overview

Fluxion exposes three main **swap-point traits** — modular interfaces that let you substitute alternative implementations without modifying core simulation code:

| Swap Point | Trait | File | Purpose |
|---|---|---|---|
| Heat Conduction | `HeatConductionSolver` | `src/physics/solver_trait.rs` | Per-surface heat conduction solve |
| Ventilation | `VentilationSchedule` | `src/sim/ventilation.rs` | Air-change rate and infiltration |
| Thermal Model | `ThermalModelTrait` | `src/sim/thermal_model.rs` | Top-level zone thermal orchestration |

These three traits are the primary extension points documented in `ARCHITECTURE.md`. The builder-pattern API on `HybridThermalModel` provides the wiring methods.

---

## 1. HeatConductionSolver — Per-Surface Conduction

**Purpose:** Computes per-surface conductive heat transfer for the zone-level thermal network.

**Trait signature:**
```rust
pub trait HeatConductionSolver {
    fn solve(
        &self,
        t_int: ThermodynamicTemperature,
        t_ext: ThermodynamicTemperature,
        surface: &Surface,
        geometry: &SurfaceGeometry,
        input: &AmbientConditions,
    ) -> ConductionSolution;
}
```

**Key types:**
- `ConductionSolution` — returned heat flux [W/m²] and conduction conductance [W/m²K]
- `Surface` — surface properties from the building geometry (area, orientation, construction)
- `SurfaceGeometry` — derived geometric data (view factors, shading)
- `AmbientConditions` — interior/exterior air temperatures and radiation

**Existing implementations:**
- `FiveR1C` — ISO 13790 5R1C steady-state conduction (default)
- `NineR4C` — ISO 13790 9R4C response-factor conduction (high-mass buildings)

**Implementation steps:**

1. **Add your solver struct** in `src/physics/` or a separate crate:
   ```rust
   pub struct MyCustomConductionSolver {
       pub u_value: f64, // W/m²K
   }
   ```

2. **Implement the trait:**
   ```rust
   impl HeatConductionSolver for MyCustomConductionSolver {
       fn solve(
           &self,
           t_int: ThermodynamicTemperature,
           t_ext: ThermodynamicTemperature,
           surface: &Surface,
           _geometry: &SurfaceGeometry,
           _input: &AmbientConditions,
       ) -> ConductionSolution {
           let conductance = self.u_value * surface.area;
           let flux = conductance * (t_ext - t_int);
           ConductionSolution { flux, conductance }
       }
   }
   ```

3. **Wire it into HybridThermalModel:**
   ```rust
   let model = HybridThermalModel::new()
       .with_conduction_solver(MyCustomConductionSolver { u_value: 0.3 });
   ```

4. **Add tests** validating conduction against an analytical reference case.

5. **Register in SolverManager** if you want named lookup (optional).

---

## 2. VentilationSchedule — Air Change Rate

**Purpose:** Defines zone ventilation and infiltration rates over time.

**Trait signature:**
```rust
pub trait VentilationSchedule {
    fn get_ach(&self, timestep: TimeStep, conditions: &WeatherConditions) -> f64;
    fn get_infiltration(&self, timestep: TimeStep, conditions: &WeatherConditions) -> f64;
}
```

**Key conversion:** Use `ach_to_conductance(ach, zone_volume)` to convert ACH to thermal conductance [W/K]:
```rust
pub fn ach_to_conductance(ach: f64, volume: f64) -> f64 {
    let rho = 1.2;    // kg/m³
    let c_p = 1005.0; // J/kg·K
    (ach * volume * rho * c_p) / 3600.0
}
```

**Existing implementations:**
- `ConstantVentilation` — fixed ACH
- `ScheduledVentilation` — time-based ACH schedule
- `WeatherResponsiveVentilation` — ACH varies with outdoor temperature

**Implementation steps:**

1. **Create your schedule struct:**
   ```rust
   pub struct MyVentilationSchedule {
       pub base_ach: f64,
       pub night_flush_ach: f64,
       pub flush_hour: usize,
   }
   ```

2. **Implement the trait:**
   ```rust
   impl VentilationSchedule for MyVentilationSchedule {
       fn get_ach(&self, timestep: TimeStep, _conditions: &WeatherConditions) -> f64 {
           if timestep.hour() == self.flush_hour {
               self.night_flush_ach
           } else {
               self.base_ach
           }
       }
       fn get_infiltration(&self, timestep: TimeStep, conditions: &WeatherConditions) -> f64 {
           // Simple wind-based infiltration model
           self.base_ach * (1.0 + 0.1 * conditions.wind_speed)
       }
   }
   ```

3. **Wire it:**
   ```rust
   let model = HybridThermalModel::new()
       .with_ventilation_schedule(MyVentilationSchedule {
           base_ach: 0.5,
           night_flush_ach: 2.0,
           flush_hour: 2,
       });
   ```

---

## 3. ThermalModelTrait — Top-Level Zone Model

**Purpose:** The top-level trait that orchestrates all thermal subsystems (conduction, ventilation, HVAC, loads).

**Trait signature (core interface):**
```rust
pub trait ThermalModelTrait: Send + Sync {
    fn solve_timesteps(
        &self,
        parameters: &ThermalModelParameters,
        weather: &WeatherData,
        schedules: &ScheduleData,
        timesteps: usize,
    ) -> Result<ThermalModelResult, ThermalModelError>;

    fn set_conduction_solver(&mut self, solver: Box<dyn HeatConductionSolver>);
    fn set_ventilation_schedule(&mut self, schedule: Box<dyn VentilationSchedule>);
}
```

**Three execution modes** (`ThermalModelMode`):
- `Physics` — full analytical 5R1C/9R4C thermal network (default)
- `Surrogate` — neural-network inference via `SurrogateManager`
- `Hybrid` — per-subsystem routing via `HybridRouting`; default policy routes loads to surrogate, keeps conduction/ventilation/HVAC on physics

**Implementation steps:**

1. **Create your model struct:**
   ```rust
   pub struct MyThermalModel {
       conduction: Box<dyn HeatConductionSolver>,
       ventilation: Box<dyn VentilationSchedule>,
       zone_volume: f64,
   }
   ```

2. **Implement ThermalModelTrait:**
   ```rust
   impl ThermalModelTrait for MyThermalModel {
       fn solve_timesteps(
           &self,
           parameters: &ThermalModelParameters,
           weather: &WeatherData,
           schedules: &ScheduleData,
           timesteps: usize,
       ) -> Result<ThermalModelResult, ThermalModelError> {
           // Your implementation:
           // 1. For each timestep:
           //    a. Get ventilation ACH → conductance via ach_to_conductance
           //    b. Solve conduction for each surface via self.conduction.solve()
           //    c. Compute zone energy balance
           //    d. Update zone temperatures
           // 2. Return results
           todo!()
       }
   }
   ```

3. **Implement wiring methods** (required by the trait):
   ```rust
   fn set_conduction_solver(&mut self, solver: Box<dyn HeatConductionSolver>) {
       self.conduction = solver;
   }
   fn set_ventilation_schedule(&mut self, schedule: Box<dyn VentilationSchedule>) {
       self.ventilation = schedule;
   }
   ```

4. **Clone support:** `HybridThermalModel::clone()` is required before `solve_timesteps` — implement `Clone` or use `Arc`.

---

## Testing Swap-Point Implementations

### Unit Tests
Test each trait implementation in isolation:
```rust
#[test]
fn test_my_conduction_matches_analytical() {
    let solver = MyCustomConductionSolver { u_value: 0.5 };
    let conditions = AmbientConditions { ... };
    let surface = Surface { area: 10.0, .. };
    let sol = solver.solve(t_int, t_ext, &surface, &geometry, &conditions);
    let expected_flux = 0.5 * 10.0 * (t_ext - t_int).get::<degree_celsius>();
    assert!((sol.flux - expected_flux).abs() < 1e-6);
}
```

### Integration Tests
Wire your implementation into `HybridThermalModel` and validate against ASHRAE 140 reference outputs:
```rust
#[test]
fn test_ventilation_matches_ashrae_600ff() {
    let model = HybridThermalModel::new()
        .with_ventilation_schedule(MyVentilationSchedule { base_ach: 0.5 });
    let result = model.solve_timesteps(&params, &weather, &schedules, 8760).unwrap();
    assert_close_to_reference(&result.zone_temperatures, "Case600FF", 0.01);
}
```

### Regression Tests
Add entries to `tests/ashrae_140_free_floating.rs` to prevent regressions:
```rust
#[test]
fn test_case_600ff_with_my_ventilation() {
    // ... validate against ASHRAE 140 reference bands
    assert!(result.min_temp_celsius >= -20.2 && result.min_temp_celsius <= -17.8);
    assert!(result.max_temp_celsius >= 35.5 && result.max_temp_celsius <= 38.5);
}
```

---

## Reference: Existing Implementations in Codebase

| Trait | Implementation | File |
|---|---|---|
| `HeatConductionSolver` | `FiveR1CConductionSolver` | `fluxion-core/src/physics/` |
| `HeatConductionSolver` | `NineR4CConductionSolver` | `fluxion-core/src/physics/` |
| `VentilationSchedule` | `ConstantVentilationSchedule` | `src/sim/ventilation.rs` |
| `VentilationSchedule` | `ScheduledVentilationSchedule` | `src/sim/ventilation.rs` |
| `ThermalModelTrait` | `HybridThermalModel` | `src/sim/thermal_model.rs` |
| `ThermalModelTrait` | `SurrogateThermalModel` | `src/ai/surrogate.rs` |

---

## Common Pitfalls

1. **Confusing per-surface with zone-level:** `HeatConductionSolver` is per-surface; the zone-level 5R1C/9R4C network lives in `thermal_model_core.rs`.

2. **Missing unit conversion:** Always use `ThermodynamicTemperature::new::<degree_celsius>()` for temperature values. Don't mix raw `f64` Celsius with typed temperatures.

3. **Clone before solve:** `HybridThermalModel` requires `.clone()` before `solve_timesteps()` — the solve method takes `&self` but mutates internally. See issue #3158 for context.

4. **ASHRAE 140 validation:** Any change to swap-point behavior requires re-running the ASHRAE 140 test suite and updating reference baselines if tolerances are exceeded.

5. **No nested parallelism:** The `BatchOracle` parallelizes zone populations only. Do not add inner-loop Rayon parallelism within `solve_timesteps`.
