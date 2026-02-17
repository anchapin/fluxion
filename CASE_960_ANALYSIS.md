# Case 960 Sunspace Analysis & Implementation Plan

## Current Status

**Test Result**: FAILING  
**Current Values**: Heating=28.67 MWh, Cooling=36.25 MWh  
**Reference Range**: Heating=1.65-2.45 MWh, Cooling=1.55-2.78 MWh  
**Error**: ~1500% high on both metrics

## Case Description

Case 960 is a **2-zone building with an attached sunspace** designed to test:
- Inter-zone heat transfer through common walls
- Multi-zone thermal coupling
- Free-floating zones (unconditioned sunspace)
- Solar gains distribution across zones

### Geometry

```
Zone 0: Back-zone (conditioned, HVAC)
  - Dimensions: 8m W × 6m D × 2.7m H
  - Floor area: 48 m²
  - Windows: South-facing, area varies
  - HVAC setpoints: 20°C (heat) / 27°C (cool)

Zone 1: Sunspace (unconditioned, free-floating)
  - Dimensions: 8m W × 2m D × 2.7m H (3m in spec but building 2m model)
  - Floor area: 16 m²
  - Windows: South-facing, 6.0 m² (high solar gain)
  - HVAC: Disabled (free-floating)

Common Wall:
  - Dimension: 8m W × 2.7m H = 21.6 m²
  - Material: Concrete, 200mm thickness
  - Connects zones through door opening (allows inter-zone air exchange)
```

## Root Cause Analysis

The ~15x energy variance (36.25 vs 2.78 MWh) suggests one or more of:

1. **Inter-zone conductance is zero or incorrect**
   - Common wall conductance not properly calculated
   - Inter-zone air exchange not modeled
   - Missing door opening model

2. **Solar gains being double-counted**
   - Sunspace solar gains being applied to both zones
   - Window area being duplicated in calculations
   - Zone index mismatch in solar gain distribution

3. **Thermal mass effects**
   - High-mass construction (Case 960 uses high-mass assemblies)
   - Sunspace thermal response should be fast (glass envelope)
   - Back-zone thermal response should be slow (high-mass walls)

4. **Free-floating zone logic**
   - Sunspace should have zero HVAC energy
   - But loads from sunspace should affect back-zone via common wall
   - Current implementation may be ignoring zone 1 loads

## Implementation Steps

### Step 1: Verify Inter-Zone Conductance
- [ ] Check that common wall conductance is calculated correctly in `ThermalModel::from_spec()`
- [ ] Verify conductance units: W/K (must multiply U × Area)
- [ ] Test with simplified 2-zone debug case

### Step 2: Debug Solar Gain Distribution
- [ ] Log solar gains for each zone separately
- [ ] Verify sunspace only gets solar gains (zone 1 windows)
- [ ] Verify back-zone wall gains are correct
- [ ] Check for window area double-counting

### Step 3: Validate Free-Floating Logic
- [ ] Confirm sunspace HVAC is disabled (capacity = 0)
- [ ] Verify sunspace temperature is calculated
- [ ] Check that sunspace temperature affects back-zone via common wall
- [ ] Verify inter-zone temperature coupling in physics solver

### Step 4: High-Mass Construction Tuning
- [ ] Verify thermal mass calculation for high-mass walls
- [ ] Check if roof/floor thermal mass is being included
- [ ] Validate surface-air coupling in 5R1C model

## Expected Behavior

After fixes, Case 960 should show:
- **Sunspace**: Free-floating, temperature oscillates 15-35°C
- **Back-zone**: HVAC maintains 20-27°C deadband
- **Common wall**: Heat flows from warm sunspace to back-zone in winter (helps heating)
- **Inter-zone cooling**: Sunspace provides free cooling in summer
- **Result**: Total HVAC energy should be 1.65-2.45 (heating) + 1.55-2.78 (cooling)

## Testing Approach

1. Create a simplified 2-zone test case with known boundary conditions
2. Compare zone temperatures against EnergyPlus/ESP-r reference
3. Compare inter-zone heat flow against manual calculation
4. Gradually increase complexity (add solar, add thermal mass)
5. Validate against ASHRAE 140 Case 960 reference ranges

## References

- ASHRAE Standard 140-2023: Section on Case 960 (multi-zone sunspace)
- EnergyPlus Case 960 test (https://github.com/NREL/EnergyPlus/...)
- ESP-r Case 960 test suite

## Next Steps

1. Merge PR #246 (peak load fix) and PR #247 (Case 195 heating-only)
2. Create detailed test for inter-zone heat transfer
3. Add debug logging to identify which calculation is producing 15x error
4. Implement fixes incrementally with test validation
