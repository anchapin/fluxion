## Issue Description

Case 900 cooling energy is **6.13 MWh** vs target **8.00-10.50 MWh** (−34% to −42% low).

Symptom: `tests/zone_balance_eplus_isolation.rs:845,908` are `#[ignore]`

## Root Cause Analysis

The 9R4C multi-node model architecture is correct:
- Wall node (R-values per layer)
- Roof node
- Floor node
- Internal mass node (furniture, partitions)

**But** coupling to the 5R1C air node appears to have an issue in the zone energy balance:

```rust
// Zone energy balance:
let Q_solar = self.solar_gains[i] * self.zone_areas[i];
let Q_conv = h_c * A_surf * (T_surf - T_air);
let Q_vent = self.ventilation_rates[i] * RHO_AIR * C_P_AIR * (T_outdoor - T_air);
let Q_int = self.internal_gains[i];
let Q_hvac = hvac_cooling - hvac_heating;
let dT_air = (Q_solar + Q_conv + Q_vent + Q_int + Q_hvac) / (M_air * C_P_AIR);
```

## Potential Issues

1. **9R4C node coupling coefficients** may not match ISO 13790 Annex C
2. **Missing thermal mass coupling** — wall/roof/floor nodes not properly coupled to air node
3. **Missing sky long-wave radiation** (can be 20-50 W/m² cooling at night)
4. **Internal mass node decoupling** — furniture not properly coupled to zone air

## Night Minimum ~0.6°C Warm

Multi-node solver systematically runs 0.6°C warmer than expected at night, suggesting:
- Missing long-wave radiation to sky
- Internal mass coupling too weak

## Related Issues

- #1203: 5R1C is steady-state only
- #1208: Architecture docs need update

## Files Affected

- `src/sim/thermal_model_core.rs`
- `src/sim/multi_node_thermal.rs`
- `src/sim/thermal_model_physics/physics_impl.rs`

## Acceptance Criteria

- [ ] Case 900 cooling energy: 8.00-10.50 MWh (currently 6.13 MWh)
- [ ] Zone air temperature within 0.5°C of E+ for ASHRAE 140 Case 900
- [ ] Night minimum within 0.5°C of E+ reference