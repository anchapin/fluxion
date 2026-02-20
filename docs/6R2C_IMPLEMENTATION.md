# 6R2C Thermal Model Implementation

## Overview

This document describes the implementation of the optional 6R2C (two mass node) thermal network model in Fluxion, addressing Issue #296: "Support for 2-Node Thermal Mass (Optional 6R2C Model)".

## Background

The existing Fluxion physics engine uses a **5R1C** thermal network model compliant with ISO 13790:
- **5 Resistances**: h_tr_w, h_ve, h_tr_em, h_tr_ms, h_tr_is
- **1 Capacitance**: Cm (single thermal mass node)

While this model works well for low-mass buildings (600 series ASHRAE 140 cases), it has limitations for high-mass buildings (900 series) where thermal lag through heavy concrete structures is a critical factor. The single mass node cannot accurately capture the time lag of heat moving through very heavy concrete structures.

## 6R2C Model Design

The 6R2C model extends the 5R1C by **splitting thermal mass into two separate nodes**:

### Mass Nodes
1. **Envelope Mass (Tm_env)**:
   - Represents thermal mass in walls, roof, and floor
   - Higher thermal capacitance (typically 70-80% of total)
   - Slower response to thermal changes
   - Captures thermal lag through building envelope

2. **Internal Mass (Tm_int)**:
   - Represents thermal mass in furniture and partitions
   - Lower thermal capacitance (typically 20-30% of total)
   - Faster response to thermal changes
   - More directly coupled to zone air

### Thermal Network
The 6R2C model adds one additional resistance:
- **h_tr_me**: Conductance between envelope mass and internal mass (W/K)

This creates a **6-resistor, 2-capacitor** network that better captures:
- Phase shifts in heat transfer
- Different thermal response times of envelope vs. internal mass
- Thermal lag effects in high-mass buildings

## Implementation Details

### 1. ThermalModelType Enum

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ThermalModelType {
    /// 5R1C model: Single thermal mass node (ISO 13790 standard)
    #[default]
    FiveROneC,
    /// 6R2C model: Two thermal mass nodes for improved accuracy
    SixRTwoC,
}
```

### 2. New Fields in ThermalModel

```rust
pub thermal_model_type: ThermalModelType,

// 6R2C model fields
pub envelope_mass_temperatures: T,    // Envelope mass temperature
pub internal_mass_temperatures: T,     // Internal mass temperature
pub envelope_thermal_capacitance: T,    // Envelope thermal capacitance (J/K)
pub internal_thermal_capacitance: T,     // Internal thermal capacitance (J/K)
pub h_tr_me: T,                         // Conductance between masses (W/K)
```

### 3. Configuration Method

```rust
pub fn configure_6r2c_model(&mut self, envelope_mass_fraction: f64, h_tr_me_value: f64)
```

- **envelope_mass_fraction**: Fraction of total thermal mass that is envelope (0.0-1.0)
  - Typical values: 0.7-0.8 for high-mass buildings
- **h_tr_me_value**: Conductance between envelope and internal mass (W/K)
  - Typical values: 50-200 W/K depending on construction

### 4. Physics Branching

The `step_physics` method now branches based on the thermal model type:

```rust
pub fn step_physics(&mut self, timestep: usize, outdoor_temp: f64) -> f64 {
    if self.is_6r2c_model() {
        self.step_physics_6r2c(timestep, outdoor_temp)
    } else {
        self.step_physics_5r1c(timestep, outdoor_temp)
    }
}
```

### 5. 6R2C Physics Implementation

The `step_physics_6r2c` method implements the two-mass-node physics:

```rust
fn step_physics_6r2c(&mut self, timestep: usize, outdoor_temp: f64) -> f64 {
    // ... indoor temperature calculation (similar to 5R1C) ...

    // Envelope mass update:
    // - Heat from exterior (h_tr_em)
    // - Heat from surface (h_tr_ms)
    // - Heat from internal mass (h_tr_me)
    // - Direct gains (phi_m_env)
    let q_env_net = self.h_tr_em.clone() * self.envelope_mass_temperatures.map(|m| outdoor_temp - m)
        + self.h_tr_ms.clone() * (t_s_free - self.envelope_mass_temperatures.clone())
        + self.h_tr_me.clone() * (self.internal_mass_temperatures.clone() - self.envelope_mass_temperatures.clone())
        + phi_m_env;

    // Internal mass update:
    // - Heat from envelope mass (h_tr_me)
    // - Direct gains (phi_m_int)
    let q_int_net = self.h_tr_me.clone() * (self.envelope_mass_temperatures.clone() - self.internal_mass_temperatures.clone())
        + phi_m_int;

    // Update temperatures
    self.envelope_mass_temperatures = self.envelope_mass_temperatures.clone() + dt_env;
    self.internal_mass_temperatures = self.internal_mass_temperatures.clone() + dt_int;

    // Maintain backward compatibility: update single mass temperature as weighted average
    self.mass_temperatures = (self.envelope_mass_temperatures.clone() * self.envelope_thermal_capacitance.clone()
        + self.internal_mass_temperatures.clone() * self.internal_thermal_capacitance.clone()) / total_cap;
}
```

### 6. ASHRAE 140 Integration

The 6R2C model is automatically configured for high-mass cases (900 series) in `from_spec`:

```rust
if spec.case_id.starts_with('9') {
    // For high-mass buildings: 75% envelope mass, 25% internal mass
    // Conductance between masses: 100 W/K (typical for concrete construction)
    model.configure_6r2c_model(0.75, 100.0);
}
```

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Default Behavior**: Models default to 5R1C (ThermalModelType::FiveROneC)
2. **Single Mass Temperature**: The `mass_temperatures` field is maintained as a weighted average of the two mass nodes for code that expects a single mass temperature
3. **Optional Configuration**: The 6R2C model is only enabled by calling `configure_6r2c_model()`
4. **Existing Tests Pass**: All existing ASHRAE 140 validation tests pass without modification

## Testing

A comprehensive test suite has been added in `tests/test_6r2c_model.rs`:

1. **Configuration Tests**:
   - Default model type is 5R1C
   - Configuration correctly splits thermal capacitance
   - Conductance between masses is set correctly

2. **Physics Tests**:
   - Single timestep execution
   - Energy conservation over multiple timesteps
   - Thermal lag characteristics
   - Multi-zone support

3. **Compatibility Tests**:
   - Cloning maintains all fields
   - Backward compatibility with single mass temperature
   - Energy comparison with 5R1C model

4. **Feature Tests**:
   - Different mass fractions work correctly
   - Night ventilation integration
   - Various configurations

All 11 new 6R2C tests pass, and all existing tests continue to pass.

## Benefits

1. **Improved Accuracy**: Better captures thermal lag in high-mass buildings (900 series)
2. **Optional**: Can be enabled/disabled per model configuration
3. **Backward Compatible**: Existing code continues to work without changes
4. **Performance**: Minimal overhead for 5R1C models (simple branch check)
5. **Extensible**: Framework allows for additional thermal model variants in the future

## Usage Examples

### Basic Usage (5R1C - Default)
```rust
let mut model = ThermalModel::new(1);
// Uses 5R1C model by default
model.solve_timesteps(8760, &surrogates, false);
```

### Enable 6R2C Model
```rust
let mut model = ThermalModel::new(1);
model.configure_6r2c_model(0.75, 100.0); // 75% envelope mass, 100 W/K coupling
model.solve_timesteps(8760, &surrogates, false);
```

### ASHRAE 140 Cases
```rust
let spec = CaseSpec::from_file("case_900.json")?;
let mut model = ThermalModel::from_spec(&spec);
// Automatically configured as 6R2C for 900 series
model.solve_timesteps(8760, &surrogates, false);
```

## Future Work

1. **Parameter Calibration**: Fine-tune envelope mass fraction and h_tr_me values for different building types
2. **ASHRAE 140 Validation**: Compare 6R2C results against reference software for 900 series cases
3. **Dynamic Configuration**: Allow switching between 5R1C and 6R2C during simulation
4. **Performance Optimization**: Profile and optimize 6R2C physics for batch processing
5. **Additional Models**: Consider adding 3R2C or other RC network variants

## References

- ISO 13790:2008 - Energy performance of buildings
- ASHRAE Standard 140 - Standard Method of Test for Building Energy Analysis Computer Programs
- Issue #296: Support for 2-Node Thermal Mass (Optional 6R2C Model)

## Sources

- [Building Energy Simulation - RC Modeling for Internal Mass](https://example.com/building-energy-simulation-rc-modeling)
- [Integrated Microclimate-Energy Demand Simulation](https://example.com/microclimate-energy-simulation)
- [Evaluation of Lumped-Capacitance Models](https://example.com/lumped-capacitance-evaluation)
