# ASHRAE 140 Cases 237 - Thermostat Setback & Night Ventilation Analysis

## Overview
ASHRAE 140 includes test cases for two advanced HVAC control strategies:
1. **Thermostat Setback** (Cases 640, 940): Reduced heating setpoint overnight
2. **Night Ventilation** (Cases 650, 950): Passive cooling through scheduled ventilation

These cases test the simulator's ability to handle time-varying control setpoints and ventilation schedules.

## Case Categories

### Thermostat Setback Cases (640, 940)
- **Base Cases:** 600 (low-mass) and 900 (high-mass)
- **Modification:** Heating setpoint drops to 10°C overnight (23:00-07:00)
- **Daytime:** Normal heating setpoint of 20°C (07:00-23:00)
- **Purpose:** Test energy savings from setback strategy

### Night Ventilation Cases (650, 950)
- **Base Cases:** 600 (low-mass) and 900 (high-mass)
- **Modification:** Ventilation fan runs 18:00-07:00 during cool hours
- **Heating:** Disabled entirely (no heating energy)
- **Cooling:** Free cooling via ventilation during night, active cooling during day
- **Purpose:** Test passive cooling effectiveness

## Building Specifications

### Common Elements (All Cases)
- **Geometry:** 8m × 6m × 2.7m (single zone)
- **Windows:** South-facing 12 m² (same as baseline)
- **Infiltration:** 0.5 ACH
- **Internal Loads:** 200W continuous
- **Weather:** Denver TMY

### Case-Specific HVAC Control

#### Case 640 (Low-Mass, Setback)
```
Heating setpoint schedule:
  23:00 - 07:00: 10°C (night setback)
  07:00 - 23:00: 20°C (normal)
Cooling setpoint: 27°C (constant)
Infiltration: 0.5 ACH
```

#### Case 650 (Low-Mass, Night Ventilation)
```
Heating: DISABLED (0°C setpoint or NO_HVAC mode)
Cooling setpoint: 27°C (daytime)
Ventilation schedule:
  18:00 - 07:00: ON (provides cooling)
  07:00 - 18:00: OFF (daytime cooling via AC)
Ventilation rate: (To be determined - likely 3-5 ACH during night)
```

#### Case 940 (High-Mass, Setback)
```
Same as Case 640 but with high-mass construction
Heating setpoint schedule:
  23:00 - 07:00: 10°C (night setback)
  07:00 - 23:00: 20°C (normal)
Cooling setpoint: 27°C (constant)
```

#### Case 950 (High-Mass, Night Ventilation)
```
Same as Case 650 but with high-mass construction
Heating: DISABLED
Cooling setpoint: 27°C (daytime)
Ventilation schedule:
  18:00 - 07:00: ON (night cooling)
  07:00 - 18:00: OFF (daytime cooling via AC)
```

## Expected Behavior

### Thermostat Setback Physics (Cases 640, 940)

**Winter Night (Setback Active):**
1. Building cools to ~10°C during night (reduced heating)
2. Thermal mass stores cold energy
3. Less heating needed in morning (recovers slowly)
4. Overall heating energy reduced compared to baseline

**Winter Day (Normal Setpoint):**
1. Heating kicks in to raise temperature to 20°C
2. Takes time to reach setpoint (depends on thermal mass)
3. High-mass building takes longer to recover from setback

**Energy Savings:**
- Setback saves energy: fewer degree-hours above minimum
- High-mass saves more: thermal mass moderates temperature swings
- Low-mass less efficient: faster temperature drop means quicker recovery

### Night Ventilation Physics (Cases 650, 950)

**Summer Night (Ventilation Active):**
1. Outdoor air is cool (often <20°C at night in Denver)
2. Ventilation fan brings in outside air
3. Removes heat from building without AC
4. Building cools below daytime setpoint
5. Reduces daytime AC cooling load

**Summer Day (Ventilation Off):**
1. Ventilation fan stops
2. Building heats up from solar gains + internal loads
3. AC cools to 27°C (high setpoint to save energy)
4. Night ventilation means less AC cooling needed

**Energy Savings:**
- Night ventilation exploits cool outdoor air
- High-mass buildings benefit more (store coolth overnight)
- Low-mass buildings show minimal benefit
- Requires cool nights (suitable for Denver climate)

## Reference Ranges

### Case 640 (Low-Mass, Setback)
| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Annual Heating | 2.50 | 3.80 | MWh |
| Annual Cooling | 6.14 | 8.45 | MWh |
| Energy Savings | TBD | TBD | % vs Case 600 |

*Expected: ~15-20% heating reduction vs Case 600*

### Case 650 (Low-Mass, Night Vent)
| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Annual Heating | 0.00 | 0.00 | MWh |
| Annual Cooling | 2.50 | 4.50 | MWh |
| Energy Savings | TBD | TBD | % vs Case 600 |

*Expected: ~60% cooling reduction vs Case 600*

### Case 940 (High-Mass, Setback)
| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Annual Heating | 2.20 | 3.40 | MWh |
| Annual Cooling | 3.92 | 6.14 | MWh |
| Energy Savings | TBD | TBD | % vs Case 900 |

*Expected: ~20-25% heating reduction vs Case 900*

### Case 950 (High-Mass, Night Vent)
| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Annual Heating | 0.00 | 0.00 | MWh |
| Annual Cooling | 0.39 | 0.92 | MWh |
| Energy Savings | TBD | TBD | % vs Case 900 |

*Expected: ~80% cooling reduction vs Case 900*

## Implementation Requirements

### 1. HVAC Schedule Support
```rust
pub struct HvacSchedule {
    /// Heating setpoint (°C) by hour of day
    heating_setpoint_by_hour: [f64; 24],
    
    /// Cooling setpoint (°C) by hour of day
    cooling_setpoint_by_hour: [f64; 24],
    
    /// Whether heating is enabled
    heating_enabled: bool,
    
    /// Whether cooling is enabled
    cooling_enabled: bool,
}
```

### 2. Ventilation Schedule Support
```rust
pub struct VentilationSchedule {
    /// Whether ventilation is active by hour of day
    active_by_hour: [bool; 24],
    
    /// Ventilation rate when active (ACH)
    ventilation_rate_ach: f64,
    
    /// Type of ventilation (natural, fan, heat recovery, etc.)
    ventilation_type: VentilationType,
}
```

### 3. Thermal Model Extensions
- Support time-varying heating/cooling setpoints
- Handle disabled heating/cooling
- Model scheduled ventilation air exchange
- Track separate heating/cooling energy

### 4. Validator Updates
- Read HVAC schedule from case specification
- Apply setpoint at each timestep based on hour
- Apply ventilation schedule to infiltration rate
- Validate against per-zone energy tracking

## Implementation Plan

### Phase 1: Schedule Infrastructure
- [ ] Create `HvacSchedule` struct with hourly setpoints
- [ ] Create `VentilationSchedule` struct
- [ ] Extend `CaseSpec` to include schedules
- [ ] Add setpoint lookup functions

### Phase 2: Thermal Model Integration
- [ ] Modify `ThermalModel` to accept dynamic setpoints
- [ ] Implement setpoint override per timestep
- [ ] Add ventilation rate modulation
- [ ] Handle heating/cooling enable/disable flags

### Phase 3: Case Specifications
- [ ] Define Case 640 (setback schedule)
- [ ] Define Case 650 (ventilation + disabled heating)
- [ ] Define Case 940 (setback with high mass)
- [ ] Define Case 950 (ventilation with high mass)

### Phase 4: Validator Integration
- [ ] Add Cases 640/650/940/950 to validator
- [ ] Apply schedules during simulation loop
- [ ] Validate against reference ranges
- [ ] Debug discrepancies

### Phase 5: Testing & Validation
- [ ] Unit tests for schedule lookups
- [ ] Integration tests for each case
- [ ] Compare energy savings vs baseline
- [ ] Validate against ASHRAE reference data

## Key Physics Considerations

### Setback Strategy
1. **Temperature Response:** How quickly does building cool/warm?
2. **Thermal Mass Effect:** High-mass buildings benefit more
3. **Recovery Penalty:** Energy to reheat building in morning
4. **Infiltration Loss:** Continuous loss during setback period
5. **Equilibrium:** Building may not reach setpoint during night

### Night Ventilation Strategy
1. **Outdoor Air Temperature:** Must be below interior for cooling
2. **Coolth Storage:** High-mass buildings store overnight cool
3. **Ventilation Load:** Sensible cooling only (no latency control)
4. **Transition Time:** When to switch ventilation on/off
5. **Occupancy Interaction:** Ventilation during unoccupied (night)

## Testing Strategy

```rust
#[test]
fn test_case_640_setback() {
    let mut validator = ASHRAE140Validator::new();
    let spec = ASHRAE140Case::Case640.spec();
    let results = validator.simulate_case(&spec, &weather);
    
    // Verify heating is less than baseline (Case 600)
    assert!(results.annual_heating_mwh < 5.71);  // Less than Case 600 max
    assert!(results.annual_heating_mwh >= 2.50); // Within reference
    
    // Verify cooling similar to baseline
    assert!(results.annual_cooling_mwh >= 6.14 && results.annual_cooling_mwh <= 8.45);
}

#[test]
fn test_case_650_night_vent() {
    let mut validator = ASHRAE140Validator::new();
    let spec = ASHRAE140Case::Case650.spec();
    let results = validator.simulate_case(&spec, &weather);
    
    // Verify heating is disabled
    assert_eq!(results.annual_heating_mwh, 0.0);
    
    // Verify cooling is much less than baseline
    assert!(results.annual_cooling_mwh < 6.14);  // Much less than Case 600 min
    assert!(results.annual_cooling_mwh >= 2.50); // Within reference
}

#[test]
fn test_case_950_high_mass_night_vent() {
    let mut validator = ASHRAE140Validator::new();
    let spec = ASHRAE140Case::Case950.spec();
    let results = validator.simulate_case(&spec, &weather);
    
    // Verify heating is disabled
    assert_eq!(results.annual_heating_mwh, 0.0);
    
    // Verify cooling is minimal (high-mass stores coolth well)
    assert!(results.annual_cooling_mwh < 0.92);
    assert!(results.annual_cooling_mwh >= 0.39);
}
```

## Debugging Strategy

### Setback Issues
- Check setpoint is actually changing with hour
- Verify infiltration is still applied during setback
- Validate thermal mass response time
- Check for HVAC control logic errors

### Ventilation Issues
- Confirm ventilation is only active during scheduled hours
- Verify ventilation doesn't activate heating
- Check outdoor air temperature is driving cooling
- Validate ventilation rate (ACH) is correct

## References

- ASHRAE Standard 140: Cases 640, 650, 940, 950
- Reference data from EnergyPlus, ESP-r, DOE-2, TRNSYS
- Schedule-based HVAC control strategies
- Passive cooling and night ventilation principles

## Status

- Analysis: IN PROGRESS
- Schedule Infrastructure: PENDING
- Thermal Model Integration: PENDING
- Case Specifications: PENDING
- Validator Integration: PENDING
- Testing: PENDING
- Validation: PENDING

## Blocked By

- Issue #235: Case 600 fix
- Issue #236: Free-floating mode

## Blocks

- Issue #151: Complete ASHRAE 140 validation suite
