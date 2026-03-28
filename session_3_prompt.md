# Physics-Based Refactoring - Session 3 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 2 Recap
- Created `src/sim/hvac/ideal_loads.rs` with:
  - `ZoneIdealLoads` struct (calculates what zone NEEDS at 100% efficiency)
  - `SimpleHVACEquipment` struct (converts thermal to electrical via COP/efficiency)
  - `IdealLoadsSystem` struct (combines both)
  - `HVACEnergyResult` struct (thermal_load_watts + electrical_kw)
- Added 23 unit tests demonstrating separation of concerns
- Default ASHRAE 140 values: cooling COP=3.0, heating efficiency=0.9

---

## Session 3 Task: Integrate Ideal Loads into ThermalModel

### Objective
Connect the Ideal Loads system to the ThermalModel to output electrical consumption in addition to thermal demand.

### Background
The current ThermalModel calculates HVAC demand in terms of thermal energy (Watts). The new IdealLoads architecture requires:
1. First calculate zone thermal load (what the zone NEEDS at 100% efficiency)
2. Then apply efficiency conversion to get electrical consumption (what equipment USES)

### Steps

#### Part A: Add IdealLoadsSystem to ThermalModel

1. Add `IdealLoadsSystem` as a field in `ThermalModel` (in `src/sim/engine.rs`):
```rust
// In ThermalModel struct definition
pub ideal_loads: IdealLoadsSystem,
```

2. Initialize in constructor:
```rust
// In ThermalModel::new() or similar
ideal_loads: IdealLoadsSystem::new(),
```

#### Part B: Modify HVAC Demand Calculation

1. Locate the current HVAC demand calculation in ThermalModel
   - Likely in `step_physics()` or a similar method

2. Modify to use IdealLoadsSystem:
```rust
// Instead of directly calculating HVAC energy, use the ideal loads system
let result = self.ideal_loads.calculate(
    zone_temp,
    self.heating_setpoint,
    self.cooling_setpoint,
);

// Use result.thermal_load_watts for thermal balance
// Use result.electrical_kw for energy reporting
```

3. Track both thermal and electrical values separately:
   - Keep thermal loads for internal physics balance
   - Report electrical consumption for validation comparison

#### Part C: Update Energy Reporting

1. Modify energy accumulation to use electrical values:
   - `annual_heating_energy` should track electrical kWh
   - `annual_cooling_energy` should track electrical kWh

2. Or add new fields to track both:
   - `annual_thermal_heating_kwh`: What zone needed
   - `annual_electrical_heating_kwh`: What equipment used

### Expected Architecture

```
ThermalModel
├── ideal_loads: IdealLoadsSystem
│   ├── zone_loads: ZoneIdealLoads      (what zone needs)
│   └── equipment: SimpleHVACEquipment   (COP=3.0, eff=0.9)
│
├── step_physics() 
│   └── Calls ideal_loads.calculate()
│
├── annual_electrical_heating_kwh      (from equipment)
└── annual_electrical_cooling_kwh       (from equipment)
```

### Deliverable
- Modified ThermalModel with IdealLoadsSystem integration
- Both thermal and electrical energy tracked separately

### Success Criteria
- [ ] Model compiles without errors
- [ ] Outputs both thermal (zone needs) and electrical (equipment uses) loads
- [ ] ASHRAE 140 standard values correctly applied (COP=3.0, efficiency=0.9)
- [ ] Existing tests still pass
- [ ] Demonstrates proper separation of ideal loads vs equipment

### Important Notes
- This creates the ARCHITECTURE - don't worry about passing ASHRAE 140 tests yet (that's Task 1.4)
- The thermal load should still be used for the zone heat balance (thermodynamics)
- Electrical consumption is what we compare against ASHRAE reference values
- Keep the thermal calculation path as the PRIMARY for physics, add electrical as a secondary output