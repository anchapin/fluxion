# HVAC Architecture Contract

**Wave:** Wave 1
**Estimate:** 1 week
**Owner:** HVAC Lead
**Repository focus:** `src/hvac/*`, `src/sim/components.rs`, `docs`
**Depends on:** None

## Overview

This document defines the architecture contract for the HVAC subsystem in Fluxion. The architecture establishes clear boundaries between five concern areas: **Loads**, **Equipment**, **Controls**, **Schedules**, and **Reporting**.

## Architecture Boundaries

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Simulation Engine                            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Schedules  │  │   Controls   │  │   Reporting  │              │
│  │  (setpoints) │──┼───(mode,    )│──┼──(energy,   )│              │
│  │              │  │   modulation)│  │   status)    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                 │                  ▲                       │
│         ▼                 ▼                  │                       │
│  ┌─────────────────────────────────────────────────────┐           │
│  │                     LOADS                           │           │
│  │  ZoneIdealLoads: Q = ρ·cp·V̇·ΔT (thermal demand)    │           │
│  └─────────────────────────────────────────────────────┘           │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────────────┐           │
│  │                    EQUIPMENT                         │           │
│  │  VariableCapacityEquipment: capacity, efficiency,   │           │
│  │  power (Chiller, Boiler, HeatPump, VAV, CAV)       │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. Zone Equipment Interface

### Purpose
Zone equipment represents the thermal energy conversion devices that meet zone loads.

### Core Trait: `VariableCapacityEquipment`

```rust
pub trait VariableCapacityEquipment: Send + Sync + Clone {
    fn calculate_capacity(&self, plr: f64, outdoor_temp: f64) -> f64;
    fn calculate_efficiency(&self, plr: f64, outdoor_temp: f64, mode: HVACMode) -> f64;
    fn calculate_power(&self, load: f64, outdoor_temp: f64, mode: HVACMode) -> f64;
    fn rated_capacity(&self) -> f64;
    fn rated_efficiency(&mode: HVACMode) -> f64;
    fn current_plr(&self) -> f64;
    fn update_state(&mut self, current_load: f64, outdoor_temp: f64, mode: HVACMode);
}
```

### Equipment Types

| Type | File | Description |
|------|------|-------------|
| `Chiller` | `sim/hvac/equipment.rs` | Cooling-only, uses polynomial COP curves |
| `Boiler` | `sim/hvac/equipment.rs` | Heating-only, AFUE efficiency model |
| `HeatPump` | `sim/hvac/equipment.rs` | Reversible, dual COP curves |
| `VAVTerminal` | `sim/hvac/mod.rs` | Variable air volume with reheat |
| `CAVSystem` | `sim/hvac/mod.rs` | Constant air volume |
| `IdealLoadsSystem` | `sim/hvac/ideal_loads.rs` | Infinite capacity (ASHRAE 140) |

### Output Interface

```rust
pub struct EquipmentOutput {
    pub capacity_watts: f64,      // Thermal capacity available
    pub power_kw: f64,            // Electrical consumption
    pub efficiency: f64,           // COP or efficiency ratio
    pub mode: HVACMode,           // Current operating mode
    pub part_load_ratio: f64,    // 0.0 to 1.0
}
```

## 2. Heating/Cooling Loops

### Purpose
Loops represent the distribution networks that deliver thermal energy from equipment to zones.

### Loop Types

| Loop | File | Purpose |
|------|------|---------|
| `HydronicLoop` | Future | Hot/chilled water circulation |
| `AirLoop` | `VAVTerminal`, `CAVSystem` | Forced air distribution |
| `IdealLoop` | `IdealLoadsSystem` | Direct zone connection |

### Interface: `ThermalLoop`

```rust
pub trait ThermalLoop {
    fn calculate_load(&self, zone_temp: f64, setpoint: f64, volume: f64, ach: f64) -> f64;
    fn deliver_load(&mut self, load: f64, equipment: &mut dyn VariableCapacityEquipment) -> EquipmentOutput;
    fn reset(&mut self);
}
```

## 3. Controllers

### Purpose
Controllers determine equipment operating mode and modulation based on zone conditions.

### Controller Types

| Controller | File | Description |
|------------|------|-------------|
| `ZoneControl` | `hvac/zone_control.rs` | Simple on/off with deadband |
| `PredictiveController` | `sim/hvac/control.rs` | Thermal inertia-aware control |

### Interface: `HVACController`

```rust
pub trait HVACController {
    fn compute_mode(&self, zone_temp: f64, heating_sp: f64, cooling_sp: f64) -> HVACMode;
    fn compute_modulation(&self, zone_temp: f64, setpoint: f64, mode: HVACMode) -> f64;
}
```

### `ZoneControl` Implementation

```rust
pub struct ZoneControl {
    pub thermal_model: Arc<ThermalModel>,
    setpoints: ZoneSetpoints,
    zone_status: VectorField,
}

impl ZoneControl {
    pub fn update_zone_controls(&mut self, current_temperatures: &VectorField) -> VectorField {
        // Returns energy input vector (Watts) per zone
    }

    pub fn get_zone_hvac_status(&self, zone_id: usize) -> HVACStatus {
        // Returns: Heating, Cooling, or Off
    }

    pub fn calculate_energy_input(&self, zone_id: usize, current_temp: f64, status: &HVACStatus) -> f64 {
        // Simple proportional: 1000W per °C difference
    }
}
```

### `PredictiveController` Implementation

```rust
pub struct PredictiveController {
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub deadband_tolerance: f64,
    pub thermal_inertia_gain: f64,
    pub temp_rate_gain: f64,
    pub previous_zone_temp: f64,
}

impl PredictiveController {
    pub fn calculate_modulation(
        &mut self,
        zone_temp: f64,
        mass_temp: f64,
        temp_rate: f64,
    ) -> (HVACMode, f64) {
        // Returns (mode, modulation_factor 0.0-1.0)
    }
}
```

## 4. Setpoints and Schedules

### Purpose
Schedules define time-varying setpoints for occupied/unoccupied periods.

### `ZoneSetpoints` Implementation

```rust
pub struct ZoneSetpoints {
    num_zones: usize,
    heating_setpoints: VectorField,  // °C
    cooling_setpoints: VectorField,  // °C
    deadbands: VectorField,          // °C
}

impl ZoneSetpoints {
    pub fn set_heating_setpoint(&mut self, zone_id: usize, temperature: f64) -> Result<(), String>;
    pub fn set_cooling_setpoint(&mut self, zone_id: usize, temperature: f64) -> Result<(), String>;
    pub fn set_deadband(&mut self, zone_id: usize, deadband: f64) -> Result<(), String>;
    pub fn get_heating_setpoint(&self, zone_id: usize) -> f64;
    pub fn get_cooling_setpoint(&self, zone_id: usize) -> f64;
    pub fn get_deadband(&self, zone_id: usize) -> f64;
    pub fn validate_setpoints(&self) -> Result<(), String>;
}
```

### Validation Rules

- Heating setpoint: 10°C to 40°C
- Cooling setpoint: 10°C to 40°C
- Deadband: 0°C to 5°C
- Heating setpoint must be below cooling setpoint
- Deadband cannot exceed setpoint difference

## 5. Reporting

### Purpose
Reporting captures energy consumption and operational status for analysis.

### Reporting Interface

```rust
pub struct HVACEnergyResult {
    pub thermal_load_watts: f64,   // Zone thermal demand
    pub electrical_kw: f64,         // Equipment consumption
    pub mode: HVACMode,             // Operating mode
}

pub struct EquipmentOutput {
    pub capacity_watts: f64,
    pub power_kw: f64,
    pub efficiency: f64,
    pub mode: HVACMode,
    pub part_load_ratio: f64,
}
```

### Metrics Captured

| Metric | Unit | Source |
|--------|------|--------|
| Zone thermal load | W | `ZoneIdealLoads` |
| Equipment power | kW | `VariableCapacityEquipment::calculate_power` |
| Efficiency | COP | `VariableCapacityEquipment::calculate_efficiency` |
| Operating mode | enum | `HVACMode` |
| Part-load ratio | 0-1 | `VariableCapacityEquipment::current_plr` |

## Module Organization

```
src/
├── hvac/                          # Primary HVAC interface
│   ├── mod.rs                     # Module exports
│   ├── zone_control.rs            # Zone-level control (public API)
│   └── zone_setpoints.rs          # Setpoint management
│
└── sim/
    └── hvac/                      # Advanced HVAC simulation
        ├── mod.rs                 # System types (VAV, CAV, HeatPump)
        ├── equipment.rs           # VariableCapacityEquipment trait + implementations
        ├── control.rs             # PredictiveController
        ├── efficiency_curves.rs   # AHRI polynomial curves
        ├── economizer.rs          # Free cooling logic
        ├── ideal_loads.rs         # ASHRAE 140 ideal loads
        ├── cycling.rs             # Equipment cycling tracker
        └── tests/                 # Equipment-specific tests
```

## Data Flow

```
Weather/BCS
    │
    ▼
┌─────────────────┐
│  ZoneSetpoints  │  (heating_sp, cooling_sp, deadband)
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                     ZoneControl                           │
│  ┌────────────────────────────────────────────────────┐   │
│  │  determine_hvac_status()                          │   │
│  │    if temp < heating_threshold → Heating           │   │
│  │    if temp > cooling_threshold → Cooling           │   │
│  │    else → Off                                      │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────┘
                         │ HVACStatus, EnergyInput
                         ▼
┌──────────────────────────────────────────────────────────┐
│                   VariableCapacityEquipment               │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────┐    │
│  │  Chiller   │  │   Boiler   │  │   HeatPump      │    │
│  │  calculate │  │  calculate │  │   calculate     │    │
│  │  _power()  │  │  _power()  │  │   _power()      │    │
│  └────────────┘  └────────────┘  └─────────────────┘    │
└────────────────────────┬─────────────────────────────────┘
                         │ Power_kW, Efficiency, Mode
                         ▼
┌──────────────────────────────────────────────────────────┐
│                   HVACEnergyResult                        │
│    thermal_load_watts | electrical_kw | mode            │
└──────────────────────────────────────────────────────────┘
```

## Integration Points

### With `ThermalModel` (5R1C Network)

```rust
pub struct ThermalModel {
    pub num_zones: usize,
    pub temperatures: VectorField,       // Zone air temperatures
    pub mass_temperatures: VectorField,  // Thermal mass temperatures
    pub loads: VectorField,              // Total thermal loads (Watts)
}
```

### With Weather Data

Controllers receive outdoor temperature for:
- Equipment capacity degradation
- Economizer free cooling判断
- Heat pump COP calculation

### With SurrogateManager

When `use_surrogates=true`, thermal loads may be predicted by neural networks instead of calculated analytically.

## Definition of Done Checklist

- [x] **Zone Equipment Interface**: `VariableCapacityEquipment` trait defined in `sim/hvac/equipment.rs`
- [x] **Equipment Implementations**: Chiller, Boiler, HeatPump, VAV, CAV in `sim/hvac/equipment.rs` and `sim/hvac/mod.rs`
- [x] **Zone Control Interface**: `ZoneControl` in `src/hvac/zone_control.rs`
- [x] **Predictive Control**: `PredictiveController` in `sim/hvac/control.rs`
- [x] **Setpoints**: `ZoneSetpoints` in `src/hvac/zone_setpoints.rs`
- [x] **Reporting**: `HVACEnergyResult`, `EquipmentOutput` structs defined
- [x] **Loops**: `IdealLoadsSystem` with zone/equipment separation in `sim/hvac/ideal_loads.rs`
- [x] **Economizer**: Free cooling logic in `sim/hvac/economizer.rs`
- [x] **Efficiency Curves**: AHRI polynomial curves in `sim/hvac/efficiency_curves.rs`
- [x] **Tests**: Unit tests for all components

## Future Extensions

- **HydronicLoop**: Hot/chilled water distribution networks
- **HeatRecovery**: Energy recovery ventilation
- **Multiple Zones per Equipment**: VAV air handling units
- **Advanced Schedules**: Time-series setpoint schedules with occupancy

## References

- ASHRAE 140 Standard for Building Energy Performance
- AHRI Directory of Certified Equipment Performance
- EnergyPlus Input Output Reference Manual
