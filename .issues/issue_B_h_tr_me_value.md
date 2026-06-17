## Problem
In the 6R2C model, h_tr_me (conductance between envelope mass and internal mass) was hardcoded to 100.0 W/K for 900 series cases. This was lower than appropriate, causing the envelope and internal mass nodes to behave too independently.

## Resolution (Issue 692)

**Changed**: h_tr_me is now calculated from physics in `from_spec()` rather than hardcoded in `configure_6r2c_model()`.

### Physics-Based h_tr_me Calculation

```rust
// thermal_model_core.rs:1000-1015
// h_tr_me (envelope-to-internal mass conductance) was previously hardcoded to 100.0 W/K
// but is now derived from construction like h_tr_ms and h_tr_em.
//
// The internal mass (furniture, partitions) couples to the envelope mass through
// the interior air and surfaces. The coupling is proportional to the interior
// surface area of the building envelope (A_int).
//
// Using h_ms = 4.5 W/(m²·K) as the coupling coefficient for furniture/partitions
// to interior air (similar to surface-to-air coupling in ISO 13790).
//
// A_int ≈ 2.0 × floor_area for typical buildings (walls + ceiling + floor surfaces)
let h_tr_me_vec: Vec<f64> = (0..num_zones)
    .map(|zone_idx| {
        let zone_floor_area = if zone_idx < spec.geometry.len() {
            spec.geometry[zone_idx].floor_area()
        } else {
            spec.geometry[0].floor_area()
        };
        let a_int = 2.0 * zone_floor_area; // Interior surface area
        let h_ms = 4.5; // Furniture/partitions coupling coefficient W/(m²·K)
        h_ms * a_int
    })
    .collect();
```

### New h_tr_me Values

For Case 900 (8m × 6m floor, 2.7m ceiling):
- `A_int = 2.0 × 48 m² = 96 m²`
- `h_tr_me = 4.5 × 96 = 432 W/K` (vs old hardcoded 100 W/K)

This is approximately **4× larger** than the old hardcoded value.

### configure_6r2c_model Change

The `configure_6r2c_model()` function no longer overwrites h_tr_me:

```rust
// thermal_model_solvers.rs:179-180 (before)
// Set conductance between envelope and internal mass
self.0.h_tr_me = self.0.zone_area.clone().map(|_| h_tr_me_value);

// thermal_model_solvers.rs:179-180 (after)
// h_tr_me is now set from physics in from_spec() - do not overwrite here
```

### Effect on Thermal Model

With the new higher h_tr_me (≈432 W/K vs old 100 W/K):

1. **Envelope and internal masses respond together**: The stronger coupling means they behave more as a single thermal buffer
2. **Effective time constant is based on h_tr_ms + h_tr_me**: With h_tr_me now dominant (432 vs h_tr_ms=117), the time constant is ≈8 hours for envelope mass
3. **Better thermal damping**: The combined mass effect better represents real high-mass building behavior

### Test Updates

Tests in `test_6r2c_comprehensive.rs` were updated to reflect the new physics:
- `test_configure_6r2c_model`: Verifies h_tr_me is NOT overwritten to 100.0
- `test_6r2c_thermal_mass_initialization`: Verifies physics-based h_tr_me ≈432 W/K
- `test_thermal_lag_envelope_vs_internal`: Updated to expect tight coupling (masses respond similarly)

## References
- thermal_model_core.rs:1000-1015 (h_tr_me physics calculation)
- thermal_model_solvers.rs:179-180 (removed h_tr_me overwrite)
- test_6r2c_comprehensive.rs (updated tests)