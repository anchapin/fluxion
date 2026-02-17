# ASHRAE 140 Case 195 - Solid Conduction Analysis

## Overview
Case 195 is a conduction-only test case from ASHRAE 140. It isolates envelope heat transfer by eliminating windows, infiltration, and internal loads.

## Building Specification

### Geometry
- Dimensions: 8m (W) × 6m (D) × 2.7m (H)
- Floor area: 48 m²
- Volume: 129.6 m³
- Surface areas:
  - Walls: 75.6 m² (opaque only, no windows)
  - Roof: 48 m²
  - Floor: 48 m²

### Construction (Low-Mass)
- Walls: Low-mass wall (U ≈ 0.514 W/m²K)
- Roof: Low-mass roof (U ≈ 0.318 W/m²K)
- Floor: Insulated floor (U ≈ 0.039 W/m²K) to ground at 10°C

### HVAC Control
- **Heating setpoint:** 20°C
- **Cooling setpoint:** 20°C (bang-bang control)
- Efficiency: 100% (ideal)
- Both heating and cooling maintain fixed setpoint

### Operating Conditions
- **Infiltration:** 0.0 ACH (tightly sealed)
- **Internal loads:** 0 W (no people, equipment, or lights)
- **Windows:** None
- **Weather:** Denver TMY (typical meteorological year)
- **Ground temperature:** Constant 10°C

## Expected Behavior

This case tests **conduction-only heat transfer**:
1. Heat flows through opaque envelope surfaces (walls, roof, floor)
2. No solar gains (no windows)
3. No internal generation (no loads)
4. HVAC must maintain 20°C year-round

### Heating Season (Winter)
- Outdoor temps drop below 20°C
- Heat escapes through envelope
- HVAC adds heating energy to maintain setpoint

### Cooling Season (Summer)
- Outdoor temps exceed 20°C
- Heat enters through envelope
- HVAC removes cooling energy to maintain setpoint

## Reference Ranges

| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| Annual Heating | 3.50 | 4.50 | MWh |
| Annual Cooling | 1.50 | 2.50 | MWh |
| Peak Heating | 2.5 | 3.2 | kW |
| Peak Cooling | 1.5 | 2.2 | kW |

*Note: These are preliminary ranges based on ASHRAE 140 methodology*

## Implementation Plan

### Phase 1: Case Specification
- [x] Define geometry in `CaseSpec`
- [x] Define construction assemblies
- [x] Set HVAC parameters (20°C setpoint for both)
- [x] Disable windows and infiltration
- [x] Set internal loads to zero

### Phase 2: Simulation Integration
- [ ] Add Case 195 to validator's case list
- [ ] Configure thermal model for single-zone operation
- [ ] Verify loads are properly zeroed
- [ ] Verify windows are excluded

### Phase 3: Validation & Debugging
- [ ] Run simulation
- [ ] Compare against reference ranges
- [ ] Debug if results are outside acceptable range
- [ ] Validate peak load calculations

### Phase 4: Documentation
- [ ] Add test case documentation
- [ ] Document any physics assumptions
- [ ] Validate against ASHRAE 140 standard

## Key Physics Considerations

1. **Thermal Mass:** Low-mass construction responds quickly to temperature changes
2. **Ground Coupling:** Floor-to-ground conductance is small (U~0.039) but significant
3. **Envelope Heat Transfer:** U-values are dominant energy pathway
4. **No Lag Effects:** Absence of solar gains removes daily timing effects
5. **Bang-Bang Control:** Constant 20°C setpoint makes this purely conduction

## Testing Strategy

```rust
#[test]
fn test_case_195_heating_cooling() {
    let mut model = ASHRAE140Case::Case195.spec();
    let results = validator.simulate_case(&model, &weather);
    
    // Verify both heating and cooling are required
    assert!(results.annual_heating_mwh > 0.0);
    assert!(results.annual_cooling_mwh > 0.0);
    
    // Verify within reference ranges
    assert!(results.annual_heating_mwh >= 3.50 && results.annual_heating_mwh <= 4.50);
    assert!(results.annual_cooling_mwh >= 1.50 && results.annual_cooling_mwh <= 2.50);
}
```

## Common Failure Modes

| Issue | Symptom | Likely Cause |
|-------|---------|--------------|
| No cooling | Cooling = 0 MWh | Infiltration compensation masking cooling needs |
| Too much heating | Heating >> 4.50 | Ground coupling underestimated |
| Unrealistic peak | Peak >> 3.2 kW | HVAC capacity incorrectly modeled |
| Temperature drift | Zone temp ≠ 20°C | Setpoint control not working |

## References

- ASHRAE Standard 140: Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs
- Test Case Description: Solid Conduction (Case 195)
- Reference data from EnergyPlus, ESP-r, DOE-2, TRNSYS

## Status

- Analysis: IN PROGRESS
- Implementation: PENDING
- Testing: PENDING
- Validation: PENDING
