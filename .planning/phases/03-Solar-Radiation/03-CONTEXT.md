# Phase 3: Solar Radiation & External Boundaries - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

**Source:** User discussion - validation approach clarified

---

<domain>
## Phase Boundary

Validate solar gain calculations, beam/diffuse decomposition, and shading geometry to fix cooling load under-prediction and annual cooling energy discrepancies.

**Key insight from research:** Solar gain calculations are already implemented correctly in `src/sim/solar.rs` (NOAA solar positioning, Perez sky model, ASHRAE 140 window angular dependence). The bug is that **solar gains are calculated but never integrated into the 5R1C thermal network energy balance**. Phase 3 focuses on this integration point.

</domain>

---

<decisions>
## Implementation Decisions

### Solar Calculation Validation Approach

**Decision:** Create dedicated validation tests in `tests/solar_validation.rs` rather than relying on existing solar.rs unit tests.

**Rationale:**
- Existing solar.rs module has comprehensive unit tests (`test_solar_position`, `test_surface_irradiance`, `test_window_solar_gain`)
- These tests verify implementation correctness, not ASHRAE 140 reference value alignment
- Creating explicit validation tests gives us clear acceptance criteria with tolerances
- Provides better separation between "solar module is correct" vs "solar module is applied correctly in thermal network"

**Implementation:**
- Task 1-4 in Wave 1 will validate SOLAR-01 through SOLAR-04 using new tests/solar_validation.rs
- Tests will validate against ASHRAE 140 reference values with specific tolerances (±5-10%)
- Tests verify that ASHRAE 140 SHGC angular dependence lookup table is called correctly
- Tests verify that Perez sky model decomposes beam/diffuse correctly
- Tests verify that ground-reflected radiation is calculated with correct albedo

---

### Solar Gain Integration Approach

**Decision:** Integrate solar_gains VectorField into 5R1C thermal network energy balance in `step_physics()`.

**Rationale from research:**
- Solar gains are calculated correctly by `calculate_hourly_solar()` but discarded
- Need to add `phi_i_solar` term to internal heat source: `phi_i_total = phi_i_internal + phi_i_solar`
- Apply ASHRAE 140 beam-to-mass distribution: 70% to mass exterior, 30% to interior surface
- Maintain VectorField types for CTA compatibility

**Implementation:**
- Task 5-6 in Wave 1 will implement this integration
- Solar gains from `calculate_hourly_solar()` stored in `self.solar_gains` VectorField
- Convert to heat flux: `phi_i_solar = solar_gains / zone_area`
- Add to energy balance equation with proper beam-to-mass distribution

---

### Wave Structure

**Decision:** Single-wave execution with 6 tasks total.

**Rationale:**
- Solar validation tasks (Tasks 1-4) can run in parallel
- Solar gain integration tasks (Tasks 5-6) have dependencies on validation completing
- Single wave keeps plan simple and focused on the core bug
- Total scope: 6 tasks modifying 3 files (reasonable for Phase 3)

</decisions>

---

<specifics>
## Specific Ideas

### ASHRAE 140 Reference Tolerances

For solar calculation validation, use these tolerance bands:

**DNI (Direct Normal Irradiance):** ±10% for clear-sky conditions
- Example: Summer solstice noon, south orientation: 800-1000 W/m² expected

**DHI (Diffuse Horizontal Irradiance):** ±20% for clear-sky conditions
- Example: Summer solstice noon: 100-200 W/m² expected

**SHGC (Solar Heat Gain Coefficient):** ±0.005 tolerance
- Double clear glass: 0.789 expected (cases 600-650)
- Validate against case specifications, not generic range

**Beam/Diffuse Ratio:** ±15% tolerance
- Beam should dominate at solar noon, diffuse more constant
- Validate physical behavior, not single reference value

### Test File Structure

```
tests/solar_validation.rs
├── mod tests {
│   ├── test_solar_position_calculations()      // SOLAR-01: Incidence angles match ASHRAE 140
│   ├── test_surface_irradiance_perez_model()    // SOLAR-02: Perez decomposition correct
│   ├── test_beam_diffuse_behavior()           // SOLAR-02: Beam peaks at noon
│   ├── test_window_shgc_angular_dependence()     // SOLAR-03: SHGC lookup used
│   ├── test_window_transmittance_values()      // SOLAR-03: Normal transmittance matches specs
│   ├── test_ground_reflected_radiation()        // SOLAR-04: Albedo calculation correct
│   └── test_shading_validation()                // SOLAR-04: Shading effects verified
│
└── #[cfg(test)]
```

### Solar Gain Integration Code Pattern

**In `src/sim/engine.rs::step_physics()`:**
```rust
// After calculate_hourly_solar(), extract solar_gains_watts
let solar_gains_watts = self.solar_gains.clone();

// Convert to heat flux (W/m²)
let phi_i_solar = solar_gains_watts.clone() / self.zone_area.clone();

// Beam-to-mass distribution (ASHRAE 140 specification)
let phi_st_solar = solar_gains_watts * (1.0 - self.solar_beam_to_mass_fraction);
let phi_m_solar = solar_gains_watts * self.solar_beam_to_mass_fraction;

// Split mass gains to exterior/interior surfaces
let phi_m_env_solar = phi_m_solar * 0.7; // 70% to exterior mass
let phi_m_int_solar = phi_m_solar * 0.3; // 30% to interior mass

// Add solar to internal heat source
let phi_i_internal = self.internal_loads * self.zone_area.clone();
let phi_i_total = phi_i_internal.clone() + phi_i_solar_flux.clone();

// Energy balance includes solar contribution
let phi_i = (phi_i_total.clone() + phi_si.clone()
    + phi_mi.clone() + phi_m_int_solar.clone())
    * T_a.clone()
    + phi_m_env_solar.clone() * T_m.clone();
```

### Expected Validation Outcomes

After Phase 3 implementation:

**SOLAR-01 (Incidence angle effects):** ASHRAE 140 SHGC ratio lookup correctly applied
- Test validates that `ashrae_140_window_shgc_ratio()` called with correct incidence angle
- Passes: beam radiation varies correctly with sun position

**SOLAR-02 (Beam/diffuse decomposition):** Perez sky model correctly separates components
- Test validates beam peaks at solar noon (12-14:00)
- Test validates diffuse remains constant when beam zero
- Test validates ground-reflected component: GHI * albedo * (1-cos(tilt))/2

**SOLAR-03 (Window properties):** SHGC and normal transmittance values from case specs
- Test validates double clear glass SHGC = 0.789 (±0.005)
- Test validates normal transmittance = 0.86156 (±0.005)
- Test verifies properties are applied in `calculate_window_solar_gain()`

**SOLAR-04 (Ground-reflected radiation):** Albedo and tilt calculations correct
- Test validates ground reflectance factor = 0.2 (Denver default)
- Test validates (1-cos(tilt))/2 factor for tilted surfaces
- Test verifies ground-reflected adds to total irradiance correctly

</specifics>

---

<deferred>
## Deferred Ideas

None identified — Phase 3 scope focused on solar validation and integration as discussed.
</deferred>

---

*Phase: 03-solar-radiation*
*Context gathered: 2026-03-09*
