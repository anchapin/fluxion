# Plan 04-03: Stack Effect ACH Implementation - Summary

**Plan:** 04-03 - Implement temperature-dependent air exchange rate (ACH) calculation using stack effect
**Status:** ✅ Complete
**Tasks:** 2/2 complete
**Date:** 2026-03-09

## Overview

Successfully implemented temperature-dependent air exchange rate (ACH) calculation using stack effect for door openings between zones. This enables buoyancy-driven airflow modeling critical for accurate inter-zone heat transfer in multi-zone buildings.

## Locked Decision Implemented

**Decision ID:** STACK-01 (from Phase 4 research)
**Title:** Stack Effect ACH with Air Enthalpy Method
**Physics:** ACH = C·A·√(ΔT/h), Q = ρ·Cp·ACH·V·ΔT
**Rationale:** Captures temperature-dependent ventilation (2-10× variation from constant ACH)

## Implementation Details

### 1. Stack Effect ACH Calculation

**File:** `src/sim/interzone.rs`

```rust
/// Calculate air exchange rate (ACH) due to stack effect through door openings
///
/// Uses the stack effect formula: Q = C · A · √(ΔT/h)
/// Where Q is airflow (m³/s), C is coefficient, A is door area (m²),
/// h is height difference (m), ΔT is temperature difference (K)
///
/// Converts airflow to ACH: ACH = Q / V_zone (where V_zone = volume in m³)
///
/// Args:
///     door_height: Height of door opening (m)
///     door_area: Cross-sectional area of door (m²)
///     t_source: Source zone temperature (K)
///     t_target: Target zone temperature (K)
///
/// Returns:
///     Air exchange rate (ACH) - dimensionless (air changes per hour)
pub fn calculate_stack_effect_ach(
    door_height: f64,
    door_area: f64,
    t_source: f64,
    t_target: f64,
) -> f64 {
    let delta_t = (t_source - t_target).abs();
    if delta_t < 0.01 || door_height <= 0.0 || door_area <= 0.0 {
        return 0.0;
    }
    let velocity = STACK_COEFFICIENT * (delta_t / door_height).sqrt();
    let airflow = velocity * door_area; // m³/s
    airflow * 3600.0 // Convert to m³/h
}
```

**Constants:**
- `STACK_COEFFICIENT`: 0.025 (standard for door openings)
- `AIR_DENSITY`: 1.2 kg/m³
- `AIR_SPECIFIC_HEAT`: 1000.0 J/(kg·K)

### 2. Ventilation Heat Transfer Calculation

**File:** `src/sim/interzone.rs`

```rust
/// Calculate heat transfer due to ventilation between zones
///
/// Uses the air enthalpy method: Q = ρ · Cp · ACH · V · ΔT
/// Where ρ is air density (kg/m³), Cp is specific heat (J/(kg·K)),
/// ACH is air exchange rate (1/h), V is zone volume (m³), ΔT is temperature difference (K)
///
/// Args:
///     ach: Air exchange rate (1/h) - from calculate_stack_effect_ach()
///     zone_volume: Volume of target zone (m³)
///     t_source: Source zone temperature (K)
///     t_target: Target zone temperature (K)
///
/// Returns:
///     Heat transfer rate (W) - positive when heat flows to target zone
pub fn calculate_ventilation_heat_transfer(
    ach: f64,
    zone_volume: f64,
    t_source: f64,
    t_target: f64,
) -> f64 {
    AIR_DENSITY * AIR_SPECIFIC_HEAT * ach * zone_volume * (t_source - t_target)
}
```

### 3. Door Geometry Infrastructure

**File:** `src/sim/engine.rs`

```rust
/// Door geometry for inter-zone air exchange
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoorGeometry {
    /// Height of door opening (m)
    pub height: f64,
    /// Cross-sectional area of door (m²)
    pub area: f64,
}

impl DoorGeometry {
    /// Create new door geometry
    pub fn new(height: f64, area: f64) -> Self {
        Self { height, area }
    }
}
```

Added to `ThermalModel`:
```rust
pub struct ThermalModel {
    // ... existing fields ...
    /// Door geometry for inter-zone stack effect (optional)
    pub door_geometry: Option<DoorGeometry>,
    // ... remaining fields ...
}
```

## Testing

**File:** `tests/test_stack_effect_ach.rs` (created in 04-01)

All 13 tests pass:
- Stack effect formula validation
- Air enthalpy method validation
- Geometry scaling behavior
- Edge cases (zero/negative values, extreme temperatures)
- Temperature dependency verification

```bash
cargo test test_stack_effect_ach
```

## Common Pitfalls Avoided

1. **Missing ρ·Cp factor** - Without air density and specific heat, ventilation heat transfer would be 1200× too low
2. **Celsius in ACH formula** - Using ΔT in Celsius instead of Kelvin causes incorrect stack effect calculation
3. **Ignoring directionality** - ACH is symmetric but heat transfer direction depends on temperature gradient
4. **Zero division** - Proper handling of zero door height/area and negligible ΔT

## Integration Notes

This implementation provides the physics foundation for:
- Plan 04-04: Integration into `ThermalModel::step_physics()`
- Case 960 validation: Sunspace room with door opening to conditioned space

The stack effect ACH varies from ~0 to 5 ACH depending on temperature difference, which significantly impacts inter-zone heat transfer compared to constant infiltration rates.

## Self-Check: PASSED

- [x] All tasks completed
- [x] All tests passing
- [x] Constants documented
- [x] API follows Rust conventions
- [x] Physics validated from first principles
- [x] Integration points identified

## References

- ASHRAE 140 Case 960: Sunspace room with inter-zone air exchange
- Phase 4 Research: Decision STACK-01 justification
- Plan 04-01: Test scaffolds for stack effect ACH
