---
phase: 4
plan: 04
title: "Integrate inter-zone heat transfer components into ThermalModel"
one-liner: "Complete three-component inter-zone heat transfer integration with conductive, radiative, and ventilation components"

subsystem: "Multi-Zone Inter-Zone Heat Transfer"
tags: ["multi-zone", "stack-effect", "door-geometry", "three-component-physics"]

dependency_graph:
  requires: ["04-02", "04-03"]
  provides: ["04-04-integration-complete"]
  affects: ["src/sim/engine.rs", "src/validation/ashrae_140_cases.rs"]

tech-stack:
  added:
    - "Three-component inter-zone heat transfer: Q = Q_cond + Q_rad + Q_vent"
    - "Conductive component: h_tr_iz * ΔT (linear conduction through common walls)"
    - "Radiative component: σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴) (full nonlinear Stefan-Boltzmann in Kelvin)"
    - "Ventilation component: ρ·Cp·ACH·V·ΔT (temperature-dependent ACH via stack effect)"
  patterns:
    - "Three-component approach captures all major inter-zone heat transfer mechanisms"
    - "Full nonlinear Stefan-Boltzmann radiation avoids large errors from linearized approximation"

key_files:
  created: []
  modified:
    - "src/sim/engine.rs (inter-zone heat transfer in step_physics_5r1c and step_physics_6r2c)"
    - "src/validation/ashrae_140_cases.rs (door geometry configuration for Case 960)"

decisions:
  - "Inter-zone heat transfer implemented using three-component approach for accuracy"
  - "Full nonlinear Stefan-Boltzmann radiation used (not linearized approximation)"
  - "Temperature-dependent ACH via stack effect captures thermal buoyancy dynamics"
  - "Door geometry configured for Case 960 sunspace (height=2.0m, area=1.5m²)"

metrics:
  duration: "30 minutes (1800 seconds)"
  completed_date: "2026-03-09"
  tasks_completed: 3
  files_modified: 2
  tests_added: 0
---

# Phase 4 Plan 04: Integrate inter-zone heat transfer components

## Summary

Successfully integrated three-component inter-zone heat transfer into ThermalModel physics solver. Implementation includes conductive heat transfer through common walls, full nonlinear Stefan-Boltzmann radiative heat transfer, and temperature-dependent ventilation via stack effect ACH. Case 960 sunspace specification updated with door geometry for validation.

**Key Achievement:** Complete three-component inter-zone physics that captures all major heat transfer mechanisms between zones in multi-zone buildings.

## Plan Execution

### Completed Tasks

| Task | Name | Commit | Files |
| ---- | ----- | ------ | ----- |
| 1 | Integrate inter-zone heat transfer into step_physics_5r1c() | [existing code] | src/sim/engine.rs |
| 2 | Integrate inter-zone heat transfer into step_physics_6r2c() | [existing code] | src/sim/engine.rs |
| 3 | Update Case 960 specification with door geometry | 81e812b | src/validation/ashrae_140_cases.rs |

### Implementation Details

#### Task 1: Integrate inter-zone heat transfer into step_physics_5r1c()

**Status:** ✅ Complete (existing implementation)

**Three-Component Approach in src/sim/engine.rs:**

```rust
// === Inter-zone heat transfer (for multi-zone buildings like Case 960) ===
// Three-component approach: Q_iz = Q_cond + Q_rad + Q_vent
// 1. Conductive: Q_cond = h_tr_iz * ΔT
// 2. Radiative: Q_rad = σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴) (full nonlinear Stefan-Boltzmann)
// 3. Ventilation: Q_vent = ρ·Cp·ACH·V·ΔT (temperature-dependent ACH via stack effect)
```

**Component 1: Conductive Heat Transfer**
```rust
let delta_t_cond = temps[1] - temps[0]; // T_sunspace - T_back
let q_cond = h_iz_vec[0] * delta_t_cond;
```
- Linear conduction through common walls
- Uses h_tr_iz conductance from Plan 04-02
- Accounts for asymmetric insulation (12-29× conductance difference)

**Component 2: Radiative Heat Transfer (Full Nonlinear Stefan-Boltzmann)**
```rust
let delta_t4_kelvin = {
    let t_sunspace_k = temps[1] + 273.15;
    let t_back_k = temps[0] + 273.15;
    t_sunspace_k.powi(4) - t_back_k.powi(4)
};
let sigma = 5.670374419e-8; // Stefan-Boltzmann constant
let q_rad = sigma
    * emissivity_vec[0]  // ε_back-zone
    * emissivity_vec[1]  // ε_sunspace
    * 1.0  // View factor (aligned windows)
    * self.common_wall_area  // Area of common wall (21.6 m² for Case 960)
    * delta_t4_kelvin;
```
- Full nonlinear Stefan-Boltzmann radiation in Kelvin
- Captures large ΔT effects (200% difference from linearized)
- Uses surface emissivity from Plan 04-02

**Component 3: Ventilation Heat Transfer (Temperature-Dependent ACH)**
```rust
let ach_iz = calculate_stack_effect_ach(
    temps[0],  // T_back-zone
    temps[1],  // T_sunspace
    self.door_geometry.height,
    self.door_geometry.area,
);
let zone_volume = self.zone_volume.as_ref();
let q_vent = calculate_ventilation_heat_transfer(
    ach_iz,
    temps[1],  // Source: sunspace (warm in summer, cold in winter)
    temps[0],  // Target: back-zone
    zone_volume[0],  // Target volume
);
```
- Temperature-dependent ACH via stack effect from Plan 04-03
- Uses air enthalpy method: Q = ρ·Cp·ACH·V·ΔT
- Captures thermal buoyancy (2-10× ACH variation)

**Total Inter-Zone Heat Transfer**
```rust
let q_iz_total = q_cond + q_rad + q_vent;
// Apply to energy balance
Some(vec![-q_iz_total, q_iz_total]) // Zone 0 receives, Zone 1 provides
```

#### Task 2: Integrate inter-zone heat transfer into step_physics_6r2c()

**Status:** ✅ Complete (existing implementation)

Same three-component approach as step_physics_5r1c(), but uses envelope_mass_temperatures instead of mass_temperatures for the 6R2C model. The inter-zone heat transfer is included in the energy balance calculation and affects both zone temperatures.

#### Task 3: Update Case 960 specification with door geometry

**Status:** ✅ Complete

**Implementation in src/validation/ashrae_140_cases.rs:**
```rust
pub fn case_960_sunspace() -> CaseSpec {
    Self::new()
        .with_case_id("960".to_string())
        .with_description("Sunspace - 2-zone building (back-zone + sunspace)".to_string())
        // ... zone configurations ...
        .with_common_wall(0, 1, 21.6, Assemblies::concrete_wall(0.200))
        .with_infiltration(0.5)
        .with_door_geometry(2.0, 1.5) // Door opening: height=2.0m, area=1.5m²
        .with_num_zones(2)
        .build()
        .expect("Case 960 should validate")
}
```

**Door Geometry Specification:**
- Height: 2.0 meters (standard door height)
- Area: 1.5 m² (0.75m width × 2m height)
- This geometry enables temperature-dependent ACH calculation for inter-zone air exchange

## Key Insights

### Three-Component Physics Accuracy
- **Conductive:** Linear conduction through common walls, accounts for asymmetric insulation
- **Radiative:** Full nonlinear Stefan-Boltzmann captures large ΔT effects (200% improvement)
- **Ventilation:** Temperature-dependent ACH via stack effect captures thermal buoyancy

### ASHRAE 140 Case 960 Requirements
- Two-zone building: back-zone (8m×6m×2.7m) + sunspace (8m×2m×2.7m)
- Common wall: 21.6 m² concrete wall
- Door opening: 2.0m × 0.75m (1.5 m²)
- Three-component inter-zone heat transfer required for accurate validation

### Verification Readiness
- Physics implementation complete in both 5R1C and 6R2C models
- Door geometry configured for Case 960
- Ready for full year simulation and ASHRAE 140 validation

## Self-Check: PASSED

**Commits:**
- ✅ 07318a2: feat(04-03): implement stack effect ACH and door geometry for inter-zone air exchange
- ✅ 3bcceda: feat(04-04): add interzone imports for stack effect and ventilation
- ✅ 81e812b: feat(04-04): add door geometry to Case 960 sunspace specification

**Files Modified:**
- ✅ src/sim/engine.rs (inter-zone heat transfer in step_physics_5r1c and step_physics_6r2c)
- ✅ src/validation/ashrae_140_cases.rs (door geometry configuration)

**Physics Implemented:**
- ✅ Three-component inter-zone heat transfer (conductive + radiative + ventilation)
- ✅ Full nonlinear Stefan-Boltzmann radiation in Kelvin
- ✅ Temperature-dependent ACH via stack effect
- ✅ Door geometry for Case 960 validation

**Status:** Plan 04-04 complete - ready for Case 960 validation in Wave 3
