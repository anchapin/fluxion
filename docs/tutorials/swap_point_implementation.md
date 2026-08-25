# Implementing Swap-Point Traits in Fluxion

This guide covers the three modular swap-point traits that allow new contributors to implement custom thermal models, heat conduction solvers, and ventilation strategies without modifying core simulation logic.

## Table of Contents

1. [Overview of Swap-Point Traits](#overview-of-swap-point-traits)
2. [HeatConductionSolver: Per-Surface Thermal Solvers](#heatconductionsolver-per-surface-thermal-solvers)
3. [VentilationSchedule: Air Change Rate Strategies](#ventilationschedule-air-change-rate-strategies)
4. [ThermalModelTrait: Zone-Level Thermal Modeling](#thermalmodeltrait-zone-level-thermal-modeling)
5. [Wiring Custom Implementations into Simulation](#wiring-custom-implementations-into-simulation)
6. [Testing Guidance](#testing-guidance)
7. [Reference Examples in the Codebase](#reference-examples-in-the-codebase)

---

## Overview of Swap-Point Traits

Fluxion's modular architecture provides three trait-based swap points that enable runtime substitution of physics implementations:

| Trait | Location | Purpose | Default Implementation |
|-------|----------|---------|----------------------|
| `HeatConductionSolver` | `src/physics/solver_trait.rs` | Per-surface wall conduction | `FiveR1CSolver` |
| `VentilationSchedule` | `src/sim/ventilation.rs` | Air change rate (ACH) scheduling | `ConstantVentilation` |
| `ThermalModelTrait` | `src/sim/thermal_model.rs` | Zone-level thermal network | `PhysicsThermalModel` |

These traits follow the **Dependency Rule**: source code dependencies point inward from concrete implementations to trait abstractions, so swapping an implementation requires no changes to calling code.

### Trait Hierarchy in HybridThermalModel

`HybridThermalModel` (`src/sim/thermal_model.rs:814`) is the primary consumer of all three traits. It holds boxed trait objects:

```rust
pub struct HybridThermalModel {
    // ...
    conduction_solver: Box<dyn HeatConductionSolver>,
    ventilation_schedule: Box<dyn VentilationSchedule>,
    // ...
}
```

The `HybridRouting` struct (`src/sim/thermal_model.rs:669`) controls which subsystems use surrogate vs. physics paths.

---

## HeatConductionSolver: Per-Surface Thermal Solvers

**File:** `src/physics/solver_trait.rs`

### Trait Definition

```rust
pub trait HeatConductionSolver: Send + Sync {
    fn name(&self) -> &str;
    fn initialize(&mut self, wall: &WallSpec) -> Result<(), SolverError>;
    fn step(
        &mut self,
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        h_interior: HeatTransferCoefficient,
        h_exterior: HeatTransferCoefficient,
    ) -> Result<HeatFlux, SolverError>;
    fn energy_storage_rate(&self) -> f64;
    fn steady_state_flux(
        &self,
        T_interior: Temperature,
        T_exterior: Temperature,
    ) -> Result<HeatFlux, SolverError>;
    fn is_valid(&self) -> bool;
}
```

### Step vs. Steady-State Query

- **`step()`** — Advances the solver's internal thermal-mass state by one timestep. Mutates internal state. Use for transient simulation.
- **`steady_state_flux()`** — Pure query returning closed-form steady-state heat flux. Does NOT mutate state. Use for parity checks and ML-surrogate validation.

### Implementing HeatConductionSolver

```rust
use fluxion::physics::solver_trait::{HeatConductionSolver, SolverError};
use fluxion::physics::units::{FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time};
use fluxion::physics::wall_spec::WallSpec;

pub struct MyConductionSolver {
    valid: bool,
    storage_rate: f64,
    // Add your solver state here
}

impl HeatConductionSolver for MyConductionSolver {
    fn name(&self) -> &str {
        "MyConductionSolver"
    }

    fn initialize(&mut self, wall: &WallSpec) -> Result<(), SolverError> {
        // Validate wall construction
        if wall.total_r_value <= 0.0 {
            return Err(SolverError::ConstructionError(
                "Wall must have positive R-value".to_string(),
            ));
        }
        self.valid = true;
        Ok(())
    }

    fn step(
        &mut self,
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        h_interior: HeatTransferCoefficient,
        h_exterior: HeatTransferCoefficient,
    ) -> Result<HeatFlux, SolverError> {
        // Your transient heat conduction calculation
        // Return heat flux in W/m² (positive = heat flowing into zone)
        let flux = calculate_transient_flux(
            timestep,
            T_interior.to_value(),
            T_exterior.to_value(),
            h_interior.to_value(),
            h_exterior.to_value(),
        )?;
        self.storage_rate = calculate_storage_rate(/* ... */);
        Ok(HeatFlux::from_value(flux))
    }

    fn energy_storage_rate(&self) -> f64 {
        self.storage_rate
    }

    fn steady_state_flux(
        &self,
        T_interior: Temperature,
        T_exterior: Temperature,
    ) -> Result<HeatFlux, SolverError> {
        // Fourier's law: q = ΔT / R_total
        let r_total = /* your total resistance calculation */;
        let flux = (T_exterior.to_value() - T_interior.to_value()) / r_total;
        Ok(HeatFlux::from_value(flux))
    }

    fn is_valid(&self) -> bool {
        self.valid
    }
}
```

### Swapping in HybridThermalModel

```rust
use fluxion::sim::thermal_model::HybridThermalModel;
use fluxion::physics::solver_trait::HeatConductionSolver;

let mut model = HybridThermalModel::from_spec(&case_spec);

// Replace the default FiveR1CSolver with your custom solver
let old_solver = model.set_conduction_solver(Box::new(my_solver));
// old_solver is dropped; if you need it, use mem::replace
```

### Existing Implementations

| Implementation | File | Use Case |
|---------------|------|----------|
| `FiveR1CSolver` | `src/physics/five_r1c_solver.rs` | Low-mass buildings, ISO 13790 5R1C network |
| CTF wrapper | (planned) | High-mass constructions via Conduction Transfer Functions |
| FD fallback | (planned) | Complex multi-layer constructions |

---

## VentilationSchedule: Air Change Rate Strategies

**File:** `src/sim/ventilation.rs`

### Trait Definition

```rust
pub trait VentilationSchedule: Debug + Send + Sync {
    fn get_ach(
        &self,
        hour: usize,
        T_outdoor: f64,
        T_indoor: f64,
        wind_speed: f64,
        volume: f64,
    ) -> f64;

    fn clone_box(&self) -> Box<dyn VentilationSchedule>;
}
```

### ACH to Conductance Conversion

The `ach_to_conductance()` function converts air change rate to thermal conductance:

```rust
pub fn ach_to_conductance(ach: f64, volume: f64, rho: f64, cp: f64) -> ThermalConductance {
    ThermalConductance::from_value((ach * volume * rho * cp) / 3600.0)
}
```

Validation: For `ACH=0.5`, `V=129.6 m³`, fluxion yields ~21.71 W/K vs EnergyPlus 21.6 W/K (Δ < 0.5%). See Issue #918.

### Implementing VentilationSchedule

```rust
use fluxion::sim::ventilation::{VentilationSchedule, ach_to_conductance};
use fluxion::physics::units::ThermalConductance;

#[derive(Debug)]
pub struct MyVentilationSchedule {
    base_ach: f64,
    // Add your schedule parameters
}

impl MyVentilationSchedule {
    pub fn new(base_ach: f64) -> Self {
        Self { base_ach }
    }
}

impl VentilationSchedule for MyVentilationSchedule {
    fn get_ach(
        &self,
        hour: usize,
        T_outdoor: f64,
        T_indoor: f64,
        wind_speed: f64,
        volume: f64,
    ) -> f64 {
        // Your ventilation calculation
        // Return ACH in 1/h
        let weather_factor = /* calculate from T_outdoor, wind_speed */;
        (self.base_ach + weather_factor).max(0.0)
    }

    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(self.clone())
    }
}
```

### Existing Implementations

| Implementation | File | Use Case |
|---------------|------|----------|
| `ConstantVentilation` | `src/sim/ventilation.rs:244` | Fixed ACH regardless of conditions |
| `ScheduledVentilation` | `src/sim/ventilation.rs:273` | Time-based ACH changes (night ventilation) |
| `WeatherDependentVentilation` | `src/sim/ventilation.rs:343` | ACH varies with outdoor temperature and wind |
| `EarthTubeVentilation<S>` | `src/sim/ventilation.rs:562` | Decorator adding ground-air heat exchange |

### Swapping in HybridThermalModel

```rust
use fluxion::sim::thermal_model::HybridThermalModel;
use fluxion::sim::ventilation::{VentilationSchedule, ConstantVentilation};

let mut model = HybridThermalModel::from_spec(&case_spec);

// Replace with constant 0.5 ACH (ASHRAE 140 default)
let old_schedule = model.set_ventilation_schedule(Box::new(ConstantVentilation::new(0.5)));

// Or with custom schedule
let old_schedule = model.set_ventilation_schedule(Box::new(MyVentilationSchedule::new(0.3)));
```

---

## ThermalModelTrait: Zone-Level Thermal Modeling

**File:** `src/sim/thermal_model.rs`

### Trait Definition

```rust
pub trait ThermalModelTrait: Send + Sync {
    fn num_zones(&self) -> usize;
    fn get_temperatures(&self) -> Vec<f64>;
    fn set_temperatures(&mut self, temperatures: &[f64]);
    fn mode(&self) -> ThermalModelMode;
    fn set_mode(&mut self, mode: ThermalModelMode);
    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_surrogates: bool,
    ) -> f64;
    fn apply_parameters(&mut self, params: &[f64]);
    fn zone_area(&self) -> f64;
    fn heating_setpoint(&self) -> f64;
    fn cooling_setpoint(&self) -> f64;
    fn hvac_power_demand(&self, timestep: usize, outdoor_temp: f64) -> f64;
    fn is_valid(&self) -> bool;
    fn get_comfort_metrics(&self) -> Vec<ZoneComfortMetrics>;
    fn set_twin_correction(&mut self, correction: &TwinCorrection);
}
```

### Concrete Implementations

| Type | Mode | Notes |
|------|------|-------|
| `PhysicsThermalModel` | `ThermalModelMode::Physics` | Default; analytical 5R1C / 9R4C |
| `SurrogateThermalModel` | `ThermalModelMode::Surrogate` | ONNX inference with optional physics fallback |
| `HybridThermalModel` | `ThermalModelMode::Hybrid` | Per-component routing via `HybridRouting` |
| `UnifiedThermalModel` | Any | Runtime-switchable wrapper |

### HybridRouting Flags

```rust
pub struct HybridRouting {
    pub use_surrogate_conduction: bool,   // 5R1C / 9R4C thermal network solve
    pub use_surrogate_ventilation: bool,  // Ventilation h_ve
    pub use_surrogate_loads: bool,        // Internal / external load prediction
    pub use_surrogate_hvac: bool,         // HVAC power demand
    pub use_ood_fallback: bool,           // Out-of-distribution detection
}
```

Default policy (`HybridRouting::default()`): only load prediction runs on surrogate; all other subsystems remain on physics.

### Implementing ThermalModelTrait

```rust
use fluxion::sim::thermal_model::{ThermalModelTrait, ThermalModelMode, ZoneComfortMetrics};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion_twin::TwinCorrection;

pub struct MyThermalModel {
    num_zones: usize,
    mode: ThermalModelMode,
    // Add your model state
}

impl ThermalModelTrait for MyThermalModel {
    fn num_zones(&self) -> usize {
        self.num_zones
    }

    fn get_temperatures(&self) -> Vec<f64> {
        // Return current zone temperatures
        todo!()
    }

    fn set_temperatures(&mut self, temperatures: &[f64]) {
        todo!()
    }

    fn mode(&self) -> ThermalModelMode {
        self.mode
    }

    fn set_mode(&mut self, mode: ThermalModelMode) {
        self.mode = mode;
    }

    fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_surrogates: bool,
    ) -> f64 {
        // Your simulation loop
        // Return EUI in kWh/m²/year
        todo!()
    }

    fn apply_parameters(&mut self, params: &[f64]) {
        // params[0]: Window U-value (W/m²K, range: 0.5-3.0)
        // params[1]: Heating setpoint (°C, range: 15-25)
        // params[2]: Cooling setpoint (°C, range: 22-32)
        todo!()
    }

    fn zone_area(&self) -> f64 {
        todo!()
    }

    fn heating_setpoint(&self) -> f64 {
        todo!()
    }

    fn cooling_setpoint(&self) -> f64 {
        todo!()
    }

    fn hvac_power_demand(&self, timestep: usize, outdoor_temp: f64) -> f64 {
        todo!()
    }

    fn is_valid(&self) -> bool {
        self.num_zones > 0 && self.zone_area() > 0.0
    }

    fn get_comfort_metrics(&self) -> Vec<ZoneComfortMetrics> {
        todo!()
    }

    fn set_twin_correction(&mut self, correction: &TwinCorrection) {
        todo!()
    }
}
```

---

## Wiring Custom Implementations into Simulation

### Via HybridThermalModel (Recommended)

```rust
use fluxion::sim::thermal_model::{
    HybridThermalModel, HybridRouting, ThermalModelTrait,
};
use fluxion::sim::ventilation::ConstantVentilation;
use fluxion::physics::five_r1c_solver::FiveR1CSolver;
use fluxion::validation::ashrae_140_cases::CaseSpec;

// Build from ASHRAE 140 spec
let spec = CaseSpec::case600();
let mut model = HybridThermalModel::from_spec(&spec);

// Configure routing
model.set_routing(HybridRouting::ood_fallback()); // Surrogate loads + OOD detection

// Swap solvers
let custom_solver = FiveR1CSolver::new();
model.set_conduction_solver(Box::new(custom_solver));

// Swap ventilation
model.set_ventilation_schedule(Box::new(ConstantVentilation::new(0.5)));

// Run simulation
let surrogates = SurrogateManager::new();
let eui = model.solve_timesteps(8760, &surrogates, false);
```

### Via ThermalModelBuilder (Fluent API)

```rust
use fluxion::sim::thermal_model::{ThermalModelBuilder, ThermalModelMode};

let model = ThermalModelBuilder::new()
    .num_zones(1)
    .mode(ThermalModelMode::Physics)
    .with_conduction_solver(my_solver)
    .with_ventilation(ConstantVentilation::new(0.5))
    .build();
```

---

## Testing Guidance

### Unit Tests for HeatConductionSolver

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use fluxion::physics::units::{FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time};

    #[test]
    fn test_steady_state_flux() {
        let mut solver = MyConductionSolver::new();
        solver.initialize(&wall_spec).unwrap();

        let flux = solver
            .steady_state_flux(
                Temperature::from_value(20.0),
                Temperature::from_value(5.0),
            )
            .unwrap();

        // Analytical: q = ΔT / R = 15 / 2.0 = 7.5 W/m²
        assert!((flux.to_value() - 7.5).abs() < 0.01);
    }

    #[test]
    fn test_step_stateful() {
        let mut solver = MyConductionSolver::new();
        solver.initialize(&wall_spec).unwrap();

        // First call initializes internal state
        let flux1 = solver
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();

        // Second call should give different result (thermal mass effect)
        let flux2 = solver
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();

        // With thermal mass, flux2 ≠ flux1 (unless steady state reached)
        assert_ne!(flux1.to_value(), flux2.to_value());
    }

    #[test]
    fn test_parity_with_physics() {
        // Compare your solver against FiveR1CSolver on known test case
        let five_r1c = FiveR1CSolver::new();
        five_r1c.initialize(&wall_spec).unwrap();

        let my_solver = MyConductionSolver::new();
        my_solver.initialize(&wall_spec).unwrap();

        let T_int = Temperature::from_value(20.0);
        let T_ext = Temperature::from_value(5.0);

        // Steady-state should match exactly
        let five_r1c_flux = five_r1c.steady_state_flux(T_int, T_ext).unwrap();
        let my_flux = my_solver.steady_state_flux(T_int, T_ext).unwrap();

        assert!((five_r1c_flux.to_value() - my_flux.to_value()).abs() < 1e-6);
    }
}
```

### Unit Tests for VentilationSchedule

```rust
#[cfg(test)]
mod ventilation_tests {
    use super::*;
    use fluxion::sim::ventilation::{VentilationSchedule, ConstantVentilation};

    #[test]
    fn test_constant_ach() {
        let vent = ConstantVentilation::new(0.5);
        assert_eq!(vent.get_ach(0, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent.get_ach(12, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent.get_ach(23, 20.0, 22.0, 2.0, 100.0), 0.5);
    }

    #[test]
    fn test_night_ventilation() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 22, 6);
        // Fan ON from hour 22 to 23, 0 to 5
        assert_eq!(vent.get_ach(21, 20.0, 22.0, 2.0, 100.0), 0.3); // before start
        assert_eq!(vent.get_ach(22, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on
        assert_eq!(vent.get_ach(6, 20.0, 22.0, 2.0, 100.0), 0.3); // fan off
    }

    #[test]
    fn test_trait_object_boxing() {
        let vent1: Box<dyn VentilationSchedule> = Box::new(ConstantVentilation::new(0.5));
        let cloned = vent1.clone_box();
        assert_eq!(cloned.get_ach(12, 20.0, 22.0, 2.0, 100.0), 0.5);
    }
}
```

### Integration Tests for HybridThermalModel

```rust
#[cfg(test)]
mod hybrid_tests {
    use super::*;

    #[test]
    fn test_custom_solver_slot_routing() {
        let spec = CaseSpec::case600();
        let mut model = HybridThermalModel::from_spec(&spec);

        // Set routing to use custom solver
        let routing = HybridRouting {
            use_surrogate_conduction: true,
            ..Default::default()
        };
        model.set_routing(routing);

        // Install custom solver
        let custom_solver = MyConductionSolver::new();
        model.set_conduction_solver(Box::new(custom_solver));

        // Verify slot is installed
        assert_eq!(model.conduction_solver().name(), "MyConductionSolver");

        // Run simulation
        let surrogates = SurrogateManager::new();
        let eui = model.solve_timesteps(8760, &surrogates, false);

        // Verify routing counters
        let metrics = model.metrics();
        assert!(metrics.surrogate_conduction_calls > 0);
    }

    #[test]
    fn test_clone_preserves_counters() {
        let spec = CaseSpec::case600();
        let mut model = HybridThermalModel::from_spec(&spec);

        // Solve to accumulate counters
        let surrogates = SurrogateManager::new();
        model.solve_timesteps(8760, &surrogates, false);

        let original_calls = model.surrogate_load_calls();

        // Clone AFTER solve
        let mut cloned = model.clone();

        // Counters are preserved on clone
        assert_eq!(cloned.surrogate_load_calls(), original_calls);

        // Reset counters for fresh solve
        cloned.reset_counters();
        assert_eq!(cloned.surrogate_load_calls(), 0);
    }
}
```

### Regression Tests Against ASHRAE 140

Always validate custom implementations against ASHRAE 140 reference cases:

```rust
#[test]
fn test_case600_annual_energy() {
    let spec = CaseSpec::case600();
    let mut model = HybridThermalModel::from_spec(&spec);

    let surrogates = SurrogateManager::new();
    let eui = model.solve_timesteps(8760, &surrogates, false);

    // Case 600 reference: ~3.06 kWh/m²/year
    let reference = 3.06;
    let tolerance = 0.15; // 15% tolerance

    assert!((eui - reference).abs() / reference < tolerance,
        "EUI {:.2} outside {:.0}% tolerance of reference {:.2}", eui, tolerance * 100.0, reference);
}
```

---

## Reference Examples in the Codebase

| Example | Location | Traits Implemented |
|---------|----------|-------------------|
| `FiveR1CSolver` | `src/physics/five_r1c_solver.rs` | `HeatConductionSolver` |
| `ConstantVentilation` | `src/sim/ventilation.rs:244` | `VentilationSchedule` |
| `ScheduledVentilation` | `src/sim/ventilation.rs:273` | `VentilationSchedule` |
| `WeatherDependentVentilation` | `src/sim/ventilation.rs:343` | `VentilationSchedule` |
| `EarthTubeVentilation<S>` | `src/sim/ventilation.rs:562` | `VentilationSchedule` (decorator) |
| `PhysicsThermalModel` | `src/sim/thermal_model.rs:225` | `ThermalModelTrait` |
| `SurrogateThermalModel` | `src/sim/thermal_model.rs:371` | `ThermalModelTrait` |
| `HybridThermalModel` | `src/sim/thermal_model.rs:814` | `ThermalModelTrait` + trait consumer |

### Trait Object Pattern Tests

```rust
// HeatConductionSolver trait object
let solver: Box<dyn HeatConductionSolver> = Box::new(FiveR1CSolver::new());
assert_eq!(solver.name(), "FiveR1CSolver");
assert!(solver.is_valid());

// VentilationSchedule trait object
let vent: Box<dyn VentilationSchedule> = Box::new(ConstantVentilation::new(0.5));
assert_eq!(vent.get_ach(12, 20.0, 22.0, 2.0, 100.0), 0.5);
```

---

## See Also

- [`src/physics/solver_trait.rs`](../../src/physics/solver_trait.rs) — `HeatConductionSolver` trait and `SolverError`
- [`src/sim/ventilation.rs`](../../src/sim/ventilation.rs) — `VentilationSchedule` trait and ACH utilities
- [`src/sim/thermal_model.rs`](../../src/sim/thermal_model.rs) — `ThermalModelTrait`, `HybridThermalModel`, and `HybridRouting`
- [`src/physics/five_r1c_solver.rs`](../../src/physics/five_r1c_solver.rs) — Reference `HeatConductionSolver` implementation
- [`docs/ARCHITECTURE.md`](../../ARCHITECTURE.md) — Architecture overview and swap-point contracts
- [`docs/ASHRAE140_RESULTS.md`](../ASHRAE140_RESULTS.md) — Validation methodology and reference values
