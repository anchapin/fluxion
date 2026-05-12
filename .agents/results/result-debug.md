# Debug Result: ASHRAE 140 Case 600 Series Failures (Issue #714)

## Status: IN PROGRESS - Root Cause Identified

## Summary
- **17 test cases failing** in 600 series (Case 610, 620, 630, 640, 650, 600FF, 650FF)
- Root cause traced to **thermal physics implementation**, NOT correction factors
- Correction factors (5.2/1.74) were deliberately removed to achieve generic accuracy

## Key Finding: Model Type Mismatch

### Discovery
- **Case 650FF uses 5R1C model** (not 6R2C)
- **Case 900FF uses 6R2C model** (high-mass properly configured)

### Code Path Analysis
```rust
// src/sim/thermal_model_core.rs:1281-1287
if spec.case_id.starts_with("9") && spec.case_id != "960" {
    model.configure_6r2c_model(0.75, 100.0, None);  // Only 900 series!
}
```

The 6R2C model (which separates envelope mass from internal mass) is **only enabled for 900 series** cases. Low-mass 600 series cases use the default 5R1C model.

### Model Types Observed
| Case | Model Type | Min Temp | Reference Min |
|------|-----------|----------|---------------|
| 600FF | 5R1C | TBD | TBD |
| 650FF | 5R1C | **-32.90°C** | **-23.00 to -21.00°C** |
| 900FF | **6R2C** | ~-6.4°C | ~-6.4 to -1.6°C |

## Thermal Network Parameters (Case 650FF)

```
Cm = 3.46e6 J/K
h_tr_ms = 122 W/K
h_tr_me = 108 W/K
h_tr_em = 109 W/K
τ = 4.18 hours
```

## Night Ventilation Physics

### Case 650/650FF Specification
```rust
NightVentilation {
    fan_capacity: 1703.16 m³/h,  // Standard m³/h
    operating_hours: (18, 7),     // 18:00 to 07:00
    adds_heat: false,
}
```

### Ventilation Parameters
- **h_ve_vent = 570.56 W/K** (when active)
- **Ventilation time constant: τ_vent = Cm / h_ve_vent ≈ 1.2 hours**
- **Thermal mass time constant: τ_mass ≈ 4.18 hours**

**Key issue**: Ventilation time constant (1.2h) is MUCH faster than thermal mass (4.18h), meaning ventilation directly cools the zone air with minimal thermal mass buffering.

### Night Ventilation Implementation in t_i_free Calculation

**Location**: `src/sim/thermal_model_physics.rs:665-688`

```rust
let h_ext = if let Some(night_vent) = &self.0.night_ventilation {
    if night_vent.is_active_at_hour(hour_of_day) {
        let air_cap_vent = night_vent.fan_capacity * 1.2 * 1005.0;
        let h_ve_vent = air_cap_vent / 3600.0;

        // h_ext = derived_h_ext + h_ve_vent
        let mut new_h_ext = h_ext_base.clone();
        for x in new_h_ext.as_mut() {
            *x += h_ve_vent;
        }
        modified_h_ext = Some(new_h_ext);
    }
}
```

**Finding**: Night ventilation IS properly included in t_i_free calculation (zone air heat balance) via modified h_ext.

## Temperature Comparison: 650FF vs 600FF

At end of simulation (timesteps 8757-8759):

| Case | Min Temp (at 8757) | Max Temp |
|------|---------------------|----------|
| 600FF | **-5.59°C** | TBD |
| 650FF | **-12.09°C** | **80.46°C** |
| Reference 650FF | **-23.00 to -21.00°C** | 63.20 to 73.50°C |

**Finding**: 650FF is ~6.5°C warmer at min temp than 600FF at same timestep, but still too cold vs reference (-12°C vs -23°C to -21°C).

The night ventilation is having SOME effect (cools more than 600FF) but the MINIMUM temperature is still ~10°C too warm compared to reference. This suggests the night ventilation isn't aggressive enough to match reference.

## Files Changed During Debugging

1. **`src/sim/thermal_model_physics.rs`**:
   - Added `DIAG_MODEL_TYPE` at line ~568-576 (shows which model type is used)
   - Added `DIAG_ZONE_AIR_5R1C` at line ~1332 (zone air temp for low-mass FF cases)
   - Changed `println!` to `eprintln!` for night vent debug
   - Extended night vent debug to include 650FF/950FF cases

## Next Steps

1. **Verify ASHRAE 140 Case 650 specification details**:
   - Fan capacity in standard m³/h vs actual?
   - Operating hours (18:00-07:00) vs (20:00-06:00)?
   - Ventilation effectiveness or air change rate?

2. **Compare ventilation implementation with EnergyPlus/reference**:
   - Is 1703.16 m³/h correct for Case 650?
   - Should ventilation apply to thermal mass or just zone air?

3. **Investigate 6R2C model for low-mass cases**:
   - Should 650FF use 6R2C like 900FF?
   - What would be the appropriate mass distribution?

4. **Check h_vent_mass coupling factor**:
   - Currently using 30% (0.3) for mass coupling
   - Is this physically correct?

## Acceptance Criteria

- [ ] All 17 failing tests pass validation
- [ ] Case 650FF min temperature within reference (-23.00 to -21.00°C)
- [ ] Case 650FF max temperature within reference (63.20 to 73.50°C)
- [ ] Case 650FF peak cooling within reference (1.90-2.50 kW)
- [ ] No correction factors added (physics-based solution)
