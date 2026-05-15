# Physics-Based Refactoring - Session 2 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 1 Recap
- Created `docs/empirical_hacks_audit.md` documenting 6 corrections
- Identified 5 empirical hacks to remove (validation layer + engine layer)
- Identified 1 legitimate conversion to keep (thermal→electrical for Case 960)

---

## Session 2 Task: Create Zone Ideal Loads and Simple HVAC Equipment Structures

### Objective
Design and implement Rust data structures that separate zone thermal load calculation from equipment energy consumption. This is a critical architectural change that follows EnergyPlus terminology.

### Background
In standard BEM terminology (EnergyPlus):
- **Ideal Loads Air System**: Calculates the sensible and latent thermal energy required to meet a zone setpoint - assumes 100% efficiency and infinite capacity
- **Equipment/Plant Model**: Converts thermal load to electrical power via COP (Coefficient of Performance)

The current Fluxion code mixes these concerns. This task creates proper separation.

### Steps

#### Part A: Zone Ideal Loads (calculates physical heat extraction)

1. Create new module `src/hvac/ideal_loads.rs` (or add to existing HVAC module)

2. Define `ZoneIdealLoads` struct:
```rust
pub struct ZoneIdealLoads {
    pub sensible_cooling_watts: f64,
    pub sensible_heating_watts: f64,
    pub latent_cooling_watts: f64,
    pub latent_heating_watts: f64,
}
```

3. Implement methods:
   - `calculate_sensible_cooling_load(zone_temp: f64, cooling_setpoint: f64, supply_air_temp: f64) -> f64`
   - `calculate_sensible_heating_load(zone_temp: f64, heating_setpoint: f64, supply_air_temp: f64) -> f64`
   - These calculate the physical heat removal/addition at 100% efficiency

4. Key assumption: These represent the IDEAL loads - what the zone NEEDS, not what the equipment USES

#### Part B: Simple HVAC Equipment (converts thermal to electrical)

1. Define `SimpleHVACEquipment` struct:
```rust
pub struct SimpleHVACEquipment {
    /// Coefficient of Performance for cooling (default 3.0 for ASHRAE 140)
    pub cooling_cop: f64,
    /// Heating efficiency (default 0.9 for electric resistance)
    pub heating_efficiency: f64,
    pub equipment_name: String,
}

impl SimpleHVACEquipment {
    pub fn new() -> Self { ... }
    pub fn with_custom_cop(cop: f64, efficiency: f64) -> Self { ... }

    /// Convert thermal load to electrical consumption
    pub fn calculate_electrical_consumption(&self, thermal_load_watts: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Cooling => thermal_load_watts / self.cooling_cop,
            HVACMode::Heating => thermal_load_watts / self.heating_efficiency,
            HVACMode::None => 0.0,
        }
    }
}
```

2. Document ASHRAE 140 standard values:
   - Cooling COP: 3.0 (typical for heat pump)
   - Heating efficiency: 0.9 (electric resistance/furnace)

#### Part C: Integration

1. Create `IdealLoadsSystem` that combines both:
```rust
pub struct IdealLoadsSystem {
    pub zone_loads: ZoneIdealLoads,
    pub equipment: SimpleHVACEquipment,
}

impl IdealLoadsSystem {
    /// Calculate both thermal loads AND electrical consumption
    pub fn calculate(&mut self, zone_temp: f64, heating_sp: f64, cooling_sp: f64, mode: HVACMode) -> HVACEnergyResult {
        // Step 1: Calculate ideal thermal load
        // Step 2: Convert to electrical via equipment
    }
}
```

2. Add unit tests demonstrating:
   - Ideal loads calculated independently of equipment
   - Equipment correctly converts thermal to electrical
   - Multiple equipment types can be swapped

### Expected Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     IdealLoadsSystem                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌──────────────────────────┐  │
│  │   ZoneIdealLoads    │    │   SimpleHVACEquipment    │  │
│  │                     │    │                          │  │
│  │ sensible_cooling   │    │ cooling_cop = 3.0       │  │
│  │ sensible_heating   │    │ heating_efficiency = 0.9│  │
│  │ latent_cooling     │    │                          │  │
│  │ latent_heating     │    │ calculate_electrical()   │  │
│  └─────────────────────┘    └──────────────────────────┘  │
│            │                            │                   │
│            ▼                            ▼                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              HVACEnergyResult                          │ │
│  │  thermal_load_watts: 10000.0  // Zone needs             │ │
│  │  electrical_kw: 3.33         // Equipment uses (COP=3)│ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Deliverable
- New HVAC module at `src/hvac/ideal_loads.rs` (or extend existing)
- Clear separation between ideal loads and equipment
- Unit tests for both components

### Success Criteria
- [ ] Module compiles without errors
- [ ] Zone loads calculated separately from electrical consumption
- [ ] ASHRAE 140 standard COP/efficiency values applied (COP=3.0, efficiency=0.9)
- [ ] Unit tests pass for both components
- [ ] Demonstrates proper separation of concerns

### Important Notes
- This creates the ARCHITECTURE for proper HVAC modeling - don't connect to ThermalModel yet (that's Task 1.3)
- Use ASHRAE 140 standard values: cooling COP=3.0, heating efficiency=0.9
- The "ideal loads" concept means 100% efficiency - equipment efficiency is applied SEPARATELY
- Document any assumptions about supply air temperature and flow rates
