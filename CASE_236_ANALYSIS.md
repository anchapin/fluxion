# ASHRAE 140 Case 236 - Free-Floating HVAC Mode Analysis

## Overview
Free-floating test cases validate the thermal simulation engine without HVAC intervention. The building temperature is allowed to float naturally based on weather, solar gains, and internal loads. These cases test:
- Thermal mass dynamics
- Natural temperature response
- Peak indoor temperatures under extreme conditions
- Building as passive thermal system

## Free-Floating Cases

### Low-Mass Series
- **Case 600FF:** Low-mass building, no HVAC
- **Case 650FF:** Low-mass building with night ventilation, no heating

### High-Mass Series
- **Case 900FF:** High-mass building, no HVAC
- **Case 950FF:** High-mass building with night ventilation, no heating

## Building Specifications

### Case 600FF (Low-Mass Free-Floating)

**Base Case:** Case 600 (Baseline low-mass)

**Modifications:**
- Heating: DISABLED (no heating energy)
- Cooling: DISABLED (no cooling energy)
- Temperature: Free to vary

**Operating Conditions:**
- Infiltration: 0.5 ACH
- Internal loads: 200W continuous
- Windows: 12 m² south-facing
- Weather: Denver TMY (extreme temperatures -40°C to +50°C)

**Expected Temperature Range:**
- Minimum: -18.80°C to -15.60°C (winter coldest days)
- Maximum: 64.90°C to 75.10°C (summer hottest days)
- Mean: ~10-15°C annual average

### Case 650FF (Low-Mass with Night Ventilation, No Heating)

**Base Case:** Case 650 (Low-mass night ventilation)

**Modifications:**
- Heating: DISABLED (0°C setpoint or no HVAC)
- Cooling: Via night ventilation + passive response
- Ventilation: 18:00-07:00 scheduled fan

**Operating Conditions:**
- Infiltration: 0.5 ACH
- Ventilation: Active 18:00-07:00 only
- Internal loads: 200W continuous
- Windows: 12 m² south-facing

**Expected Temperature Range:**
- Minimum: -23.00°C to -21.00°C (winter, no ventilation benefit)
- Maximum: 63.20°C to 73.50°C (summer, ventilation helps cool at night)
- Effect: Slightly lower max temps due to night ventilation

### Case 900FF (High-Mass Free-Floating)

**Base Case:** Case 900 (Baseline high-mass)

**Modifications:**
- Heating: DISABLED
- Cooling: DISABLED
- Thermal mass: Significant (affects temperature swings)

**Operating Conditions:**
- Construction: High-mass walls/roof
- Thermal mass: ~150 kWh/K (high thermal storage)
- Infiltration: 0.5 ACH
- Internal loads: 200W continuous
- Windows: 12 m² south-facing

**Expected Temperature Range:**
- Minimum: -6.40°C to -1.60°C (winter, thermal mass moderates)
- Maximum: 41.80°C to 46.40°C (summer, thermal mass limits peaks)
- Effect: Much smaller temperature swings than low-mass

### Case 950FF (High-Mass with Night Ventilation, No Heating)

**Base Case:** Case 950 (High-mass night ventilation)

**Modifications:**
- Heating: DISABLED
- Cooling: Via night ventilation + thermal mass
- Ventilation: 18:00-07:00 scheduled fan

**Operating Conditions:**
- Construction: High-mass
- Thermal mass: ~150 kWh/K
- Ventilation: 18:00-07:00 (provides night cooling)
- Internal loads: 200W continuous

**Expected Temperature Range:**
- Minimum: -20.20°C to -17.80°C (winter)
- Maximum: 35.50°C to 38.50°C (summer, significantly lower due to night cooling + thermal mass)
- Effect: Night ventilation + thermal mass provides effective passive cooling

## Reference Ranges

### Case 600FF (Low-Mass Free-Floating)
| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Min Temperature | -18.80 | -15.60 | °C |
| Max Temperature | 64.90 | 75.10 | °C |
| Annual HVAC Energy | 0 | 0 | MWh |

### Case 650FF (Low-Mass + Night Vent)
| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Min Temperature | -23.00 | -21.00 | °C |
| Max Temperature | 63.20 | 73.50 | °C |
| Annual HVAC Energy | 0 | 0 | MWh |

### Case 900FF (High-Mass Free-Floating)
| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Min Temperature | -6.40 | -1.60 | °C |
| Max Temperature | 41.80 | 46.40 | °C |
| Annual HVAC Energy | 0 | 0 | MWh |

### Case 950FF (High-Mass + Night Vent)
| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Min Temperature | -20.20 | -17.80 | °C |
| Max Temperature | 35.50 | 38.50 | °C |
| Annual HVAC Energy | 0 | 0 | MWh |

*Note: Free-floating cases have 0 HVAC energy by definition*

## Expected Behavior

### Winter Behavior (Free-Floating)

**Low-Mass (Case 600FF):**
- Winter nights: Temperature drops to outdoor level (~-20°C or colder)
- Internal loads (200W) raise temperature slightly
- Minimal thermal mass retention
- Rapid temperature swings with outdoor weather

**High-Mass (Case 900FF):**
- Winter nights: Temperature moderates due to thermal mass
- Internal loads help maintain higher temperature
- Thermal mass "smooths" outdoor temperature swings
- Slower temperature response to weather changes

### Summer Behavior (Free-Floating)

**Low-Mass (Case 600FF):**
- Solar gains dominate (large windows, low thermal mass)
- Afternoon temperatures can exceed 70°C (extreme)
- Limited thermal mass to absorb and store heat
- Evening cooling via infiltration/radiation

**High-Mass (Case 900FF):**
- Thermal mass absorbs solar gains during day
- Temperature rise moderated (~40°C max vs 70°C+ for low-mass)
- Temperatures decline slowly at night
- Thermal mass provides "flywheel" effect

### Night Ventilation Effect

**Case 650FF (Low-Mass + Vent):**
- Night ventilation (18:00-07:00) provides modest cooling
- Low thermal mass limits benefit
- Some overnight temperature reduction

**Case 950FF (High-Mass + Vent):**
- Night ventilation + thermal mass: highly effective
- Cool outdoor air cools building structure
- Thermal mass stores this "coolth"
- Maximum summer temperature significantly lower (~35-40°C)
- Demonstrates passive cooling strategy

## Implementation Requirements

### 1. Free-Floating Mode in Thermal Model

```rust
pub enum HvacMode {
    Conditioned {
        heating_setpoint: f64,
        cooling_setpoint: f64,
    },
    FreeFloating,  // No HVAC, temperature floats
    HeatingOnly {
        heating_setpoint: f64,
    },
    CoolingOnly {
        cooling_setpoint: f64,
    },
}
```

### 2. HVAC Control Logic

```rust
// In ThermalModel.step_physics():
match self.hvac_mode {
    HvacMode::FreeFloating => {
        // No HVAC energy
        return 0.0; // kWh
    }
    HvacMode::Conditioned { heating_sp, cooling_sp } => {
        // Current logic: calculate heating/cooling
        // ...
    }
    // ...
}
```

### 3. Temperature Tracking

- Track min/max temperatures across simulation
- Per-zone temperature recording
- Peak temperature identification
- Daily temperature range tracking

### 4. Validator Updates

- Add min/max temperature metrics
- Compare against reference ranges
- Verify HVAC energy is 0 for free-floating
- Track thermal mass effects

## Implementation Plan

### Phase 1: Free-Floating Mode Support
- [ ] Add `HvacMode` enum to thermal model
- [ ] Implement mode-based HVAC logic
- [ ] Zero out HVAC energy when free-floating
- [ ] Preserve temperature calculations

### Phase 2: Temperature Tracking
- [ ] Track hourly temperatures in validator
- [ ] Calculate min/max temperatures
- [ ] Record peak daily temperatures
- [ ] Identify extreme temperature events

### Phase 3: Case Specifications
- [ ] Define Case 600FF (low-mass, free-floating)
- [ ] Define Case 650FF (low-mass, night vent, no heating)
- [ ] Define Case 900FF (high-mass, free-floating)
- [ ] Define Case 950FF (high-mass, night vent, no heating)

### Phase 4: Validator Integration
- [ ] Add all 4 FF cases to validator
- [ ] Implement per-zone temperature tracking
- [ ] Compare min/max against reference ranges
- [ ] Validate HVAC energy is 0

### Phase 5: Testing & Validation
- [ ] Unit tests for free-floating mode
- [ ] Integration tests for all 4 FF cases
- [ ] Validate temperature ranges
- [ ] Verify thermal mass effects (high vs low)
- [ ] Test night ventilation impact

## Key Physics Considerations

### Thermal Mass Impact

**Low-Mass Buildings:**
- Fast temperature response to external changes
- Limited heat storage capacity
- Greater temperature swings
- More sensitive to solar gains

**High-Mass Buildings:**
- Slow temperature response (time lag)
- High heat storage capacity
- Smaller temperature swings
- "Flywheel" effect moderates extremes
- Greater lag during sunrise/sunset

### Solar Gain Effect

- South-facing window (12 m²) primary source of summer heat
- Peak solar gain ~1000 W/m² at solar noon
- Maximum window gain ~12,000 W (12 kW)
- Can heat low-mass building significantly

### Night Ventilation Mechanism

- Outdoor air typically cool at night (especially in Denver)
- Ventilation provides sensible cooling
- Effectiveness depends on:
  - Outdoor air temperature (must be < indoor)
  - Ventilation rate (ACH)
  - Thermal mass (stores coolth)
  - Duration (more hours = more cooling)

### Ground Coupling

- Floor-to-ground conductance: ~1.9 W/K
- Ground temperature constant ~10°C
- Provides moderate cooling in summer
- Provides moderate heating in winter

## Testing Strategy

```rust
#[test]
fn test_case_600ff_free_floating() {
    let mut validator = ASHRAE140Validator::new();
    let spec = ASHRAE140Case::Case600FF.spec();
    let results = validator.simulate_case(&spec, &weather);
    
    // Verify no HVAC energy
    assert_eq!(results.annual_heating_mwh, 0.0);
    assert_eq!(results.annual_cooling_mwh, 0.0);
    
    // Verify temperature ranges
    let min_temp = results.min_temp_celsius.unwrap();
    let max_temp = results.max_temp_celsius.unwrap();
    assert!(min_temp >= -18.80 && min_temp <= -15.60);
    assert!(max_temp >= 64.90 && max_temp <= 75.10);
}

#[test]
fn test_case_900ff_high_mass_free_floating() {
    let mut validator = ASHRAE140Validator::new();
    let spec = ASHRAE140Case::Case900FF.spec();
    let results = validator.simulate_case(&spec, &weather);
    
    // Verify no HVAC energy
    assert_eq!(results.annual_heating_mwh, 0.0);
    assert_eq!(results.annual_cooling_mwh, 0.0);
    
    // Verify high-mass shows smaller temperature swings
    let min_temp = results.min_temp_celsius.unwrap();
    let max_temp = results.max_temp_celsius.unwrap();
    let range = max_temp - min_temp;
    
    // High-mass should have ~30-40°C range vs 60-80°C for low-mass
    assert!(range < 45.0);
    
    // Verify within reference
    assert!(min_temp >= -6.40 && min_temp <= -1.60);
    assert!(max_temp >= 41.80 && max_temp <= 46.40);
}

#[test]
fn test_case_950ff_night_vent_effectiveness() {
    let mut validator = ASHRAE140Validator::new();
    let spec_base = ASHRAE140Case::Case900FF.spec();
    let spec_vent = ASHRAE140Case::Case950FF.spec();
    
    let results_base = validator.simulate_case(&spec_base, &weather);
    let results_vent = validator.simulate_case(&spec_vent, &weather);
    
    // Night ventilation should reduce peak temperature
    let max_base = results_base.max_temp_celsius.unwrap();
    let max_vent = results_vent.max_temp_celsius.unwrap();
    assert!(max_vent < max_base);
    
    // Verify within references
    assert!(max_vent >= 35.50 && max_vent <= 38.50);
}
```

## Debugging Strategy

### Temperature Too High
- Check solar gain calculation
- Verify window SHGC and area
- Validate internal load (200W continuous)
- Check thermal mass value

### Temperature Too Low
- Verify outdoor temperature is being read correctly
- Check infiltration is applied (0.5 ACH)
- Validate ground coupling
- Check thermal mass is adequate

### Unexpected Temperature Swings
- Low-mass: Should have large swings (60°C+)
- High-mass: Should have moderate swings (30-50°C)
- Verify thermal capacitance is correct
- Check for HVAC leak (should be 0)

### Night Ventilation Not Helping
- Verify ventilation is only active 18:00-07:00
- Check ventilation rate (ACH) is non-zero
- Validate outdoor air temperature is < indoor during vent hours
- Confirm thermal mass is accumulating cool

## References

- ASHRAE Standard 140: Free-floating test cases
- Reference data from EnergyPlus, ESP-r, DOE-2, TRNSYS
- Passive thermal response principles
- Thermal mass and building dynamics
- Night ventilation cooling effectiveness

## Status

- Analysis: IN PROGRESS
- Free-floating Mode: PENDING IMPLEMENTATION
- Temperature Tracking: PENDING IMPLEMENTATION
- Case Specifications: READY
- Validator Integration: PENDING
- Testing: PENDING
- Validation: PENDING

## Blocked By

- Issue #235: Case 600 fix (establish baseline first)

## Blocks

- Issue #151: Complete ASHRAE 140 validation suite
