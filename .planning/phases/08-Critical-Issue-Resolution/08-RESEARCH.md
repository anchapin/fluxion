# Phase 8 Research: Case 960 Cooling Failure

## Research Questions

1. **Why is annual cooling 4.53 MWh vs expected 1.0-3.5 MWh?**
2. **Why is sunspace colder than back-zone?** (Expected: sunspace warmer due to solar gains)
3. **Are solar gains being applied correctly to both zones?**
4. **Is inter-zone heat transfer direction and magnitude correct?**
5. **Does the model account for HVAC equipment efficiency (COP)?**
6. **What are the reference values from EnergyPlus/ESP-r for Case 960?**

## Current Hypotheses

### H1: Solar Gains Not Applied to Sunspace

The sunspace may have windows (6 m² south-facing) but solar gains calculation might be returning 0 due to:
- Window area not properly associated with Zone 1 surfaces
- Shading device erroneously blocking all solar
- Weather data nil for sunspace lat/long?
- Bug in `calculate_zone_solar_gain` for zone_idx > 0

**Test**: Diagnostic logging should show `solar_gain_watts > 0` for Zone 1 in summer. If zero, H1 likely.

### H2: Inter-Zone Conductance Sign Error

If the sunspace temperature is lower, heat flows from back-zone to sunspace, causing back-zone to lose heat and require more cooling. The code applies:
```rust
q_iz_total = q_cond + q_rad + q_vent;
Some(vec![-q_iz_total, q_iz_total])  // [back, sunspace]
```
This means back receives `-q_iz_total` (negative if q_iz_total > 0) and sunspace receives `+q_iz_total`. If q_iz_total > 0 (sunspace warmer), back gains heat (negative contribution). But in our case sunspace is colder, so q_iz_total would be negative, and back loses heat (positive contribution), increasing cooling load. But why is sunspace cold? That's the root.

### H3: Missing Cooling COP

EnergyPlus reports **electrical** energy. Our model returns **thermal** HVAC load. If we don't divide by COP (~3.0 for cooling), we would overpredict electrical consumption by 3x. But our cooling is only 30% high (4.53 vs 3.5 max), not 300% high. So COP alone may not explain it, but could be a contributing factor if we also have other errors.

### H4: Excessive Inter-Zone Conductance

If h_iz is too high, the cold sunspace could suck more heat from back-zone than intended. Check common wall conductance calculation:
- Area = 21.6 m²
- U-value from `concrete_wall(0.200)` - need to check what U this gives
- Expected h_iz = U * area ≈ ?

### H5: Infiltration or Ventilation Error

Night ventilation or infiltration might be overcooling the sunspace. Case 960 has 0.5 ACH for both zones. Stack effect ventilation through the door could be excessive. Check `calculate_stack_effect_ach` parameters.

## Information Needed

- [ ] Confirm `solar_gain_watts` values for Zone 1 during summer hours from diagnostic test
- [ ] Check `self.surfaces` construction for Zone 1: does it include the 6 m² south window?
- [ ] Get actual inter-zone conductance value from model (h_tr_iz[0])
- [ ] Compare temperature traces hour-by-hour for summer period: do we see sunspace warming during day?
- [ ] Find ASHRAE 140 reference documentation for Case 960: expected temperatures, solar gains, inter-zone heat transfer

## External Research (Web Search)

Need to find:
1. ASHRAE 140-2023 standard tables for Case 960 (may require purchase)
2. EnergyPlus validation results for Case 960 (publicly available?)
3. Academic papers describing sunspace thermal performance

**Search queries**:
- "ASHRAE 140 Case 960 results EnergyPlus"
- "sunspace multi-zone building energy simulation validation"
- "Case 960 annual cooling energy reference"

## Investigation Plan Priority

1. **First**: Run diagnostic test with summer week logging to get actual numbers
2. **Second**: Examine solar gains for Zone 1 - if zero, fix immediately
3. **Third**: Check inter-zone heat transfer components - verify magnitude and sign
4. **Fourth**: HVAC efficiency - apply COP division if needed
5. **Fifth**: Tune inter-zone conductance if physics is correct but calibration off

## Code Paths to Examine

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `calculate_zone_solar_gain` | src/sim/engine.rs | 2958-3112 | Solar gain per zone |
| `calc_analytical_loads` | src/sim/engine.rs | 3298-3362 | Load calculation entry |
| `step_physics_5r1c` | src/sim/engine.rs | 2009-2400+ | Main physics loop |
| Inter-zone Q | src/sim/engine.rs | 2104-2166 | Heat transfer between zones |
| `from_spec` (Case 960) | src/validation/ashrae_140_cases.rs | 1869-1895 | Spec builder |
| Window properties | src/validation/ashrae_140_cases.rs | around 1882-1887 | zone_window method |

## Known Constraints

- ASHRAE 140 reference ranges are **calibrated for 5R1C model** (comment in benchmark.rs)
- The reference ranges for Case 960 are wider (5-15 heating, 1-3.5 cooling) due to multi-zone complexity
- Must not break single-zone cases (600-950) when fixing multi-zone
- HVAC for Zone 1 must remain disabled (free-floating)

## Expected Outcome After Fix

| Metric | Before | After Target |
|--------|--------|--------------|
| Annual Cooling | 4.53 MWh | 1.5-3.0 MWh (mid-range ~2.2) |
| Sunspace Mean Temp | 18.02°C | >22°C (warmer than back-zone in summer) |
| Back-zone Temp | 22.82°C | 20-22°C (stable) |
| Inter-zone Q (summer) | negative (back→sunspace) | positive (sunspace→back) |
| Solar gains Zone 1 | ? (need data) | >50 W/m² on summer days |
