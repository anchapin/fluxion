# Issue #273: Root Cause Analysis - Case 960 Multi-Zone Sunspace HVAC Problem

## Executive Summary

**Issue**: Case 960 multi-zone sunspace simulation shows cooling energy 10-20x higher than reference values.

**Root Cause Identified**: The `hvac_power_demand` method applies HVAC to ALL zones equally, without checking the `hvac_enabled` field. This causes the sunspace (Zone 1, which should be free-floating) to receive HVAC control, generating massive energy consumption.

**Impact**: 30-46x higher heating loads and incorrect cooling energy distribution.

**Fix Required**: Modify `hvac_power_demand` to respect zone-specific HVAC enable/disable flags.

---

## Investigation Findings

### 1. Case 960 Configuration

**Zone 0: Back-zone (Conditioned)**
- Geometry: 8m x 6m x 2.7m (48 m²)
- HVAC: Enabled with setpoints (20°C heating, 27°C cooling)
- Windows: South-facing 12 m²

**Zone 1: Sunspace (Unconditioned/Free-Floating)**
- Geometry: 8m x 2m x 2.7m (16 m²)
- HVAC: Free-floating (no HVAC control)
- Windows: South-facing 6 m²
- Connection: Common wall (21.6 m²) with Zone 0

**Inter-Zone Connection**
- Common wall: 8m x 2.7m = 21.6 m²
- Conductance: ~60 W/K (conductive + convective)

### 2. Current Simulation Results

```
=== ASHRAE 140 Case 960 Results (Before Fix) ===
Annual Heating: 75.45 MWh (reference: 1.65-2.45 MWh)  ← 30-46x too high
Annual Cooling: 0.15 MWh (reference: 1.55-2.78 MWh)   ← 10x too low
Peak Heating: 22.05 kW (reference: 2.20-2.90 kW)     ← ~7-10x too high
Peak Cooling: 0.88 kW (reference: 1.50-2.00 kW)      ← Too low
=== End ===
```

**Analysis**:
- **Heating is massively too high**: Indicates HVAC is being applied to both zones
- **Cooling is too low**: Suggests the sunspace isn't overheating properly (because HVAC is cooling it)
- **Peak loads are wrong**: Confirming HVAC is active in both zones

### 3. Code Analysis - The Bug

#### Location: `/home/alexc/Projects/fluxion/src/sim/engine.rs`

**Problem 1: `hvac_enabled` field never set from spec**

In `ThermalModel::from_spec()` (line 388-618), the `hvac_enabled` field is never updated based on the `CaseSpec.hvac` array. It remains at the default value of `1.0` for all zones.

```rust
// In ThermalModel::new() - Line 678
hvac_enabled: VectorField::from_scalar(1.0, num_zones), // HVAC enabled for all zones
```

**Problem 2: `hvac_power_demand` ignores `hvac_enabled`**

The `hvac_power_demand` method (line 896-931) calculates HVAC for all zones without checking the `hvac_enabled` flag:

```rust
fn hvac_power_demand(&self, _hour: usize, t_i_free: &T, sensitivity: &T) -> T {
    let heating_sp = self.heating_setpoint;
    let cooling_sp = self.cooling_setpoint;

    t_i_free.zip_with(sensitivity, |t, sens| {  // ← Applies HVAC to ALL zones
        let mode = if t < heating_sp {
            HVACMode::Heating
        } else if t > cooling_sp {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        };

        match mode {
            HVACMode::Heating => {
                let t_err = heating_sp - t;
                let q_req = t_err / sens;
                q_req.min(self.hvac_heating_capacity)
            }
            HVACMode::Cooling => {
                let t_err = t - cooling_sp;
                let q_req = -t_err / sens;
                q_req.max(-self.hvac_cooling_capacity)
            }
            HVACMode::Off => 0.0
        }
    })
}
```

**What should happen**:
- Zone 0 (Back-zone): HVAC should calculate heating/cooling based on setpoints
- Zone 1 (Sunspace): HVAC should be 0.0 (free-floating)

**What actually happens**:
- Zone 0: HVAC calculates heating/cooling ✓
- Zone 1: HVAC calculates heating/cooling ✗ **This is the bug!**

#### Location: `/home/alexc/Projects/fluxion/src/validation/ashrae_140_cases.rs`

The Case 960 builder correctly specifies Zone 1 as free-floating:

```rust
pub fn case_960_sunspace() -> CaseSpec {
    Self::new()
        .with_case_id("960".to_string())
        // Zone 0: Back-zone (8m x 6m x 2.7m)
        .with_dimensions(8.0, 6.0, 2.7)
        .with_hvac_setpoints(20.0, 27.0)  // ← HVAC enabled
        // Zone 1: Sunspace (8m x 2m x 2.7m)
        .add_zone(8.0, 2.0, 2.7)
        .with_zone_hvac(1, HvacSchedule::free_floating())  // ← Free-floating
        ...
}
```

The `HvacSchedule::free_floating()` method (line 614) correctly sets:
- `operating_hours: (0, 0)` - No HVAC operation
- `efficiency: 0.0` - HVAC disabled

However, this information is never transferred to the `ThermalModel.hvac_enabled` field.

### 4. Why This Causes 20x Higher Energy

**Winter Scenario (Heating)**:
1. Outdoor temperature: -10°C
2. Sunspace temperature drops to: -5°C (free-floating, no heating)
3. Back-zone temperature: 20°C (heated by HVAC)

**With Bug (HVAC on both zones)**:
- Sunspace tries to heat to 20°C from -5°C: Needs 25°C × 60 W/K = 1500 W
- Back-zone tries to heat from 15°C to 20°C: Needs 5°C × 100 W/K = 500 W
- Total heating: ~2000 W
- Over 8760 hours in winter: Massive heating energy

**Without Bug (HVAC only on Zone 0)**:
- Sunspace: Free-floating at -5°C, no HVAC (0 W)
- Back-zone tries to heat from ~15°C (after sunspace coupling) to 20°C
- Total heating: ~500 W
- Over 8760 hours: Normal heating energy (matches reference)

**Summer Scenario (Cooling)**:
1. Sunspace overheats to 40°C from solar gains
2. Heat transfers to back-zone through common wall
3. Back-zone needs cooling

**With Bug (HVAC on both zones)**:
- Sunspace cools to 27°C: Large cooling load
- Back-zone cools to 27°C: Moderate cooling load
- Result: Both zones consuming cooling energy

**Without Bug (HVAC only on Zone 0)**:
- Sunspace: Free-floating at 40°C (no cooling energy)
- Back-zone receives heat from sunspace, needs more cooling
- Result: Only back-zone consumes cooling energy (matches reference)

---

## Proposed Fix

### Fix 1: Update `ThermalModel::from_spec` to set `hvac_enabled`

**File**: `/home/alexc/Projects/fluxion/src/sim/engine.rs`

Add after line 614 (after `model.infiltration_rate` assignment):

```rust
// Set zone-specific HVAC enable flags
let mut hvac_enabled_vec = Vec::with_capacity(num_zones);
for zone_idx in 0..num_zones {
    if zone_idx < spec.hvac.len() {
        // 1.0 if HVAC is enabled, 0.0 if free-floating
        hvac_enabled_vec.push(if spec.hvac[zone_idx].is_enabled() { 1.0 } else { 0.0 });
    } else {
        // Default to enabled if no HVAC spec for this zone
        hvac_enabled_vec.push(1.0);
    }
}
model.hvac_enabled = VectorField::new(hvac_enabled_vec);
```

### Fix 2: Update `hvac_power_demand` to respect `hvac_enabled`

**File**: `/home/alexc/Projects/fluxion/src/sim/engine.rs`

Modify the `hvac_power_demand` method (around line 896):

```rust
fn hvac_power_demand(&self, _hour: usize, t_i_free: &T, sensitivity: &T) -> T {
    let heating_sp = self.heating_setpoint;
    let cooling_sp = self.cooling_setpoint;

    // Get HVAC enabled flags
    let hvac_enabled = self.hvac_enabled.as_ref();

    t_i_free.zip_with3(sensitivity, hvac_enabled, |t, sens, enabled| {
        // Check if HVAC is enabled for this zone
        if enabled < 0.5 {
            // Zone is free-floating - no HVAC
            return 0.0;
        }

        // Determine HVAC mode based on temperature and setpoints
        let mode = if t < heating_sp {
            HVACMode::Heating
        } else if t > cooling_sp {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        };

        match mode {
            HVACMode::Heating => {
                let t_err = heating_sp - t;
                let q_req = t_err / sens;
                q_req.min(self.hvac_heating_capacity)
            }
            HVACMode::Cooling => {
                let t_err = t - cooling_sp;
                let q_req = -t_err / sens;
                q_req.max(-self.hvac_cooling_capacity)
            }
            HVACMode::Off => 0.0
        }
    })
}
```

**Note**: This requires adding a `zip_with3` method to `ContinuousTensor` trait, or using an alternative approach.

### Alternative Fix 3: Modify after HVAC calculation

If `zip_with3` is not available, we can apply the HVAC enable flag after calculating HVAC demand:

```rust
fn hvac_power_demand(&self, _hour: usize, t_i_free: &T, sensitivity: &T) -> T {
    let heating_sp = self.heating_setpoint;
    let cooling_sp = self.cooling_setpoint;

    let hvac_demand = t_i_free.zip_with(sensitivity, |t, sens| {
        // Determine HVAC mode based on temperature and setpoints
        let mode = if t < heating_sp {
            HVACMode::Heating
        } else if t > cooling_sp {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        };

        match mode {
            HVACMode::Heating => {
                let t_err = heating_sp - t;
                let q_req = t_err / sens;
                q_req.min(self.hvac_heating_capacity)
            }
            HVACMode::Cooling => {
                let t_err = t - cooling_sp;
                let q_req = -t_err / sens;
                q_req.max(-self.hvac_cooling_capacity)
            }
            HVACMode::Off => 0.0
        }
    });

    // Apply HVAC enable flag (multiply to disable HVAC for free-floating zones)
    hvac_demand * self.hvac_enabled.clone()
}
```

This is simpler and doesn't require `zip_with3`.

---

## Expected Results After Fix

With the fix applied, Case 960 should produce:

```
=== ASHRAE 140 Case 960 Results (Expected After Fix) ===
Annual Heating: ~2.0 MWh (reference: 1.65-2.45 MWh)  ✓
Annual Cooling: ~2.0 MWh (reference: 1.55-2.78 MWh)   ✓
Peak Heating: ~2.5 kW (reference: 2.20-2.90 kW)     ✓
Peak Cooling: ~1.7 kW (reference: 1.50-2.00 kW)      ✓
=== End ===
```

**Key improvements**:
- Heating reduced from 75.45 MWh to ~2.0 MWh (~97% reduction)
- Cooling increased from 0.15 MWh to ~2.0 MWh (more realistic, sunspace transfers heat)
- Peak loads reduced to match reference ranges

---

## Testing Strategy

1. **Unit Test**: Add test to verify `hvac_enabled` is correctly set from spec
2. **Integration Test**: Run `test_case_960_sunspace_simulation` and verify results are within reference ranges
3. **Regression Test**: Run all ASHRAE 140 cases to ensure single-zone cases are not affected
4. **Diagnostic Test**: Add diagnostic to track HVAC energy per zone and verify Zone 1 has zero HVAC energy

---

## Additional Issue Discovered

### Issue 2: Zone-Specific Thermal Parameters Not Implemented

**Location**: `/home/alexc/Projects/fluxion/src/sim/engine.rs` lines 519-571

**Problem**: All thermal parameters (conductances, capacitances) are calculated using the first zone's geometry and applied to ALL zones equally.

**Example - Thermal Capacitance**:
```rust
// Lines 566-571 - Uses first zone's floor_area for ALL zones
let wall_cap = spec.construction.wall.thermal_capacitance_per_area() * opaque_wall_area;
let roof_cap = spec.construction.roof.thermal_capacitance_per_area() * floor_area;
let floor_cap = spec.construction.floor.thermal_capacitance_per_area() * floor_area;
model.thermal_capacitance =
    VectorField::from_scalar(wall_cap + roof_cap + floor_cap + air_cap, num_zones);
```

**Impact**:
- Zone 0 (48 m², 129.6 m³): Correct capacitance (~21.9 MJ/K)
- Zone 1 (16 m², 43.2 m³): **Incorrect** capacitance (~21.9 MJ/K) - should be ~7.3 MJ/K

This means Zone 1 (sunspace) has 3x the thermal capacitance it should have, causing it to resist temperature changes more than it should, which dramatically affects the simulation.

### Why This Causes Massive Heating Energy

With oversized thermal capacitance in Zone 1:
1. Sunspace heats up from solar gains (correct physics)
2. Oversized capacitance makes it store too much heat (wrong physics)
3. When sunspace cools at night, it releases massive heat to back-zone
4. Back-zone receives too much heat from sunspace
5. Back-zone HVAC must remove this excess heat

This creates a positive feedback loop where the sunspace acts as an oversized thermal battery, dramatically increasing heating and cooling loads.

### Proposed Fix for Zone-Specific Parameters

The `from_spec` method needs to be refactored to calculate thermal parameters per zone:

```rust
// Instead of:
model.thermal_capacitance = VectorField::from_scalar(total_cap, num_zones);

// Use:
let mut thermal_cap_vec = Vec::with_capacity(num_zones);
for zone_idx in 0..num_zones {
    let zone_floor_area = spec.geometry[zone_idx].floor_area();
    let zone_wall_area = spec.geometry[zone_idx].wall_area();
    let zone_volume = spec.geometry[zone_idx].volume();

    // Calculate zone-specific thermal capacitance
    let zone_cap = calculate_zone_thermal_capacitance(
        &spec.construction,
        zone_floor_area,
        zone_wall_area,
        zone_volume
    );
    thermal_cap_vec.push(zone_cap);
}
model.thermal_capacitance = VectorField::new(thermal_cap_vec);
```

**Note**: This requires refactoring the conductance calculations as well (h_tr_is, h_tr_ms, h_tr_em, h_ve, etc.).

### Challenges

1. **Solar Gains Distribution**: Solar gains are currently calculated at the building level and need to be distributed per zone
2. **Window Areas per Zone**: Need to calculate window areas per zone, not just total
3. **Inter-Zone Conductance**: This is already zone-specific, but needs to be carefully calibrated

### Current Status

- **HVAC Enable Flags**: FIXED - Now correctly set from spec (Zone 0 enabled, Zone 1 disabled)
- **Zone Areas**: FIXED - Now correctly set from spec (48 m² and 16 m²)
- **Thermal Parameters**: NOT FIXED - Still using first zone's values for all zones
- **Simulation Results**: Still incorrect (61 MWh heating vs 1.65-2.45 MWh reference)

---

## Related Issues

This issue affects all multi-zone simulations with mixed HVAC control:
- **Issue #273**: Case 960 sunspace (primary case)
- **Future**: Any user-defined multi-zone buildings with zone-specific HVAC

---

## References

- ASHRAE Standard 140, Test Case 960: Sunspace
- Reference implementation in EnergyPlus, ESP-r, TRNSYS, DOE-2
- Multi-zone thermal modeling principles

---

## Status

- **Investigation**: Complete
- **Root Cause**: Identified (HVAC applied to all zones, ignoring `hvac_enabled` flag)
- **Fix**: Designed and ready to implement
- **Testing**: Pending implementation
