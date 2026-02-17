# ASHRAE 140 Case 960 - Sunspace (Multi-Zone) Analysis

## Overview
Case 960 is a multi-zone test case featuring an attached sunspace. It tests the simulator's ability to model:
- Two thermal zones with different HVAC control
- Inter-zone heat transfer through common walls
- Solar gains distribution across multiple zones

## Building Specification

### Geometry (Two-Zone Configuration)

#### Zone 0: Conditioned Back-Zone
- Dimensions: 8m (W) × 6m (D) × 2.7m (H)
- Floor area: 48 m²
- Volume: 129.6 m³
- HVAC: Conditioned (controlled)

#### Zone 1: Sunspace (Attached)
- Dimensions: 8m (W) × 3m (D) × 2.7m (H)
- Floor area: 24 m²
- Volume: 64.8 m³
- HVAC: Free-floating (unconditioned)
- Purpose: Captures solar gains and transfers heat to back-zone

### Interconnection
- **Common Wall:** 8m (W) × 2.7m (H) = 21.6 m²
- **Connection Type:** Door opening (modeled as conductance)
- **Heat Transfer:** Through both opaque surface and air mixing

### Construction (Low-Mass)
- Walls: Low-mass (U ≈ 0.514 W/m²K)
- Roof: Low-mass (U ≈ 0.318 W/m²K)
- Floor: Insulated (U ≈ 0.039 W/m²K)

### Windows
- **Back-zone:** South-facing 12 m² (same as Case 600)
- **Sunspace:** South-facing glazing (high solar gain)
  - Area: ~24 m² of transparent glazing (south-facing)
  - U-value: 3.0 W/m²K (same as back-zone)
  - SHGC: 0.789

### HVAC Control
- **Back-zone (Zone 0):**
  - Heating setpoint: 20°C
  - Cooling setpoint: 27°C
  - Full HVAC capacity (heating + cooling)
  
- **Sunspace (Zone 1):**
  - Free-floating (no HVAC)
  - Temperature tracks naturally
  - Can exceed control setpoints

### Operating Conditions
- **Infiltration:** 0.5 ACH (applied to both zones)
- **Internal loads:** 200W (applied to back-zone only)
- **Ground temperature:** Constant 10°C
- **Weather:** Denver TMY

## Expected Behavior

### Heat Flow Paths
1. **Solar Gains in Sunspace:**
   - High solar absorption in unshaded glazing
   - Zone 1 temperature can rise significantly
   - Heat transfer to back-zone through common wall

2. **Inter-Zone Heat Transfer:**
   - If Sunspace is warmer: heat flows to conditioned back-zone
   - HVAC in back-zone handles this additional load
   - Creates coupling between zones

3. **Back-Zone HVAC:**
   - Must satisfy both its own loads AND inter-zone transfer
   - Heating: Cold winter nights + infiltration
   - Cooling: Solar gains (direct + from sunspace)

### Seasonal Pattern
- **Winter:** Sunspace acts as buffer, reduces heating load
- **Summer:** Sunspace can overheat, increases cooling load
- **Spring/Fall:** Sunspace provides variable load depending on weather

## Reference Ranges

| Metric | Zone | Min | Max | Unit |
|--------|------|-----|-----|------|
| Annual Heating | 0 (Back) | 1.65 | 2.45 | MWh |
| Annual Cooling | 0 (Back) | 1.55 | 2.78 | MWh |
| Peak Heating | 0 (Back) | 1.0 | 1.5 | kW |
| Peak Cooling | 0 (Back) | 1.0 | 1.5 | kW |

*Note: HVAC energy is measured for back-zone only (Zone 0)*

## Multi-Zone Implementation Requirements

### 1. Thermal Model Extensions
```rust
// Multi-zone support in ThermalModel
struct InterZoneConnection {
    from_zone: usize,
    to_zone: usize,
    conductance: f64,  // W/K (common wall)
    area: f64,         // m²
}
```

### 2. Heat Transfer Modeling
- Common wall conductance: `U_wall * area`
- Air mixing rate: Model through door opening conductance
- Temperature gradient across common wall

### 3. Zone-Specific Control
- Independent HVAC setpoints per zone
- Free-floating mode for unconditioned zones
- Load accounting per zone

### 4. Solar Gain Distribution
- Distribute solar gains to respective zones
- Sunspace receives full solar benefit
- Back-zone limited to south-facing window

## Implementation Plan

### Phase 1: Thermal Model Enhancement
- [ ] Extend `ThermalModel` to support inter-zone connections
- [ ] Implement common wall heat transfer
- [ ] Add zone-specific HVAC control flags
- [ ] Test with 2-zone configuration

### Phase 2: Case Specification
- [ ] Define geometry for both zones
- [ ] Create inter-zone connection (common wall)
- [ ] Set zone-specific HVAC parameters
- [ ] Configure sunspace glazing

### Phase 3: Validator Integration
- [ ] Add Case 960 to validator's case list
- [ ] Track HVAC energy for back-zone only
- [ ] Calculate inter-zone temperatures
- [ ] Validate against reference ranges

### Phase 4: Testing & Validation
- [ ] Verify temperature response in both zones
- [ ] Validate heating/cooling split
- [ ] Compare against reference values
- [ ] Debug discrepancies

## Key Physics Considerations

1. **Common Wall Coupling:**
   - Direct conductive path between zones
   - Temperature difference drives heat flow
   - Affects both zone energy balances

2. **Sunspace as Thermal Buffer:**
   - High solar absorptance (mostly glazed)
   - Decouples outdoor weather from back-zone
   - Reduces both heating and cooling loads in back-zone

3. **Free-Floating Unconditioned Zone:**
   - Temperature not constrained by HVAC
   - Can exceed outdoor temperatures (in sun)
   - Affects inter-zone heat transfer rates

4. **Control Interaction:**
   - Back-zone HVAC reacts to combined loads (internal + inter-zone)
   - More complex than single-zone case
   - Energy balance must account for inter-zone flows

## Testing Strategy

```rust
#[test]
fn test_case_960_multizone() {
    let mut validator = ASHRAE140Validator::new();
    let spec = ASHRAE140Case::Case960.spec();
    let results = validator.simulate_case(&spec, &weather);
    
    // Verify back-zone (0) has both heating and cooling
    assert!(results.annual_heating_mwh > 0.0);
    assert!(results.annual_cooling_mwh > 0.0);
    
    // Verify within reference ranges
    assert!(results.annual_heating_mwh >= 1.65 && results.annual_heating_mwh <= 2.45);
    assert!(results.annual_cooling_mwh >= 1.55 && results.annual_cooling_mwh <= 2.78);
    
    // Verify sunspace exists and has no HVAC energy
    assert_eq!(results.hvac_energy_per_zone[1], 0.0); // Zone 1 is free-floating
}
```

## Debugging Strategy

### If loads are too high:
1. Check inter-zone conductance (may be overestimated)
2. Verify sunspace is truly free-floating (no HVAC)
3. Validate solar gain calculation in sunspace
4. Check thermal mass model

### If loads are too low:
1. Verify common wall area is correct
2. Check sunspace glazing SHGC
3. Validate inter-zone temperature difference
4. Verify infiltration in both zones

### If temperature is unrealistic:
1. Check zone-specific HVAC setpoints
2. Verify mass temperature updates per zone
3. Validate heat capacity allocation
4. Check for numerical instabilities

## References

- ASHRAE Standard 140, Test Case 960: Sunspace
- Reference data from EnergyPlus, ESP-r, TRNSYS, DOE-2
- Multi-zone thermal modeling principles
- Inter-zone coupling in energy simulation

## Implementation Dependencies

- Requires Phase 1 of multi-zone support in thermal engine
- May require modifications to `ThermalModel` architecture
- Affects validator's per-zone energy tracking

## Current Status

- Analysis: IN PROGRESS
- Thermal Model: PENDING ENHANCEMENT
- Case Specification: READY
- Validator Integration: PENDING
- Testing: PENDING
- Validation: PENDING

## Blocked By

- Issue #235: Case 600 fix (prioritize single-zone physics first)

## Blocks

- Issue #151: Complete full ASHRAE 140 validation suite
