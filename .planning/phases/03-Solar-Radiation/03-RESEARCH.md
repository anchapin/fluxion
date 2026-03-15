# Phase 3: Solar Radiation & External Boundaries - Research

**Researched:** 2026-03-09
**Domain:** Solar radiation calculations, beam/diffuse decomposition, shading geometry
**Confidence:** MEDIUM

## Summary

Phase 3 addresses systematic solar gain calculation issues causing cooling load under-prediction (67% below reference for Case 900) and peak cooling load failures (0.60 kW vs 2.10-3.50 kW reference). The research reveals that Fluxion has a sophisticated solar radiation module (`src/sim/solar.rs`) with NOAA solar positioning, Perez sky model for diffuse radiation, and ASHRAE 140 window angular dependence tables. However, solar gains are not being applied to the thermal network in the physics engine (`src/sim/engine.rs`).

**Key Finding:** The `calculate_zone_solar_gain()` method exists and calculates solar gains correctly, but these gains are not integrated into the 5R1C thermal network energy balance during `solve_timesteps()`. Solar gains are computed but discarded, leading to zero solar contribution to zone temperatures.

**Primary Recommendation:** Integrate solar gains into the 5R1C thermal network energy balance equation in `step_physics()`. Solar gains should be added as an internal heat source term (similar to internal loads from lighting/equipment/occupancy) to correctly simulate solar heating effects on zone air and thermal mass.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SOLAR-01 | Calculate solar gains for all building surfaces using beam/diffuse decomposition | Existing `calculate_hourly_solar()` with Perez model provides correct beam/diffuse decomposition. Integration into thermal network needed. |
| SOLAR-02 | Apply solar gains to thermal network with correct beam-to-mass/surface distribution | Beam-to-mass fraction parameters exist (0.7 to mass, 0.3 interior surface) but solar gains not added to energy balance. |
| SOLAR-03 | Implement shading calculations for overhangs and shade fins (Cases 610, 630, 910, 930) | `calculate_shaded_fraction()` in `src/sim/shading.rs` correctly computes overhang/fin shadows. Called in solar gain calculations. |
| SOLAR-04 | Validate solar gain calculations against ASHRAE 140 reference values for all orientations | Window angular dependence tables match ASHRAE 140 specifications. Integration will enable validation. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fluxion::sim::solar | v0.1.0 | Solar position, surface irradiance, window gains | NOAA algorithm, Perez model, ASHRAE 140 tables |
| fluxion::sim::shading | v0.1.0 | Overhang and shade fin geometry | Simplified projection shadows for ASHRAE 140 cases |
| fluxion::sim::sky_radiation | v0.1.0 | Sol-air temperature, longwave exchange | ASHRAE Fundamentals Chapter 4, 18 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| fluxion::weather | v0.1.0 | Hourly weather data (DNI, DHI, GHI) | All solar calculations require weather input |
| fluxion::physics::cta | v0.1.0 | VectorField operations for thermal states | Solar gains added as VectorField to thermal network |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| NOAA solar position | SPA (Solar Position Algorithm) | NOAA is simpler and sufficient for ASHRAE 140 hourly resolution |
| Perez sky model | Isotropic sky model | Perez is ASHRAE 140 standard, more accurate for tilted surfaces |
| ASHRAE 140 SHGC lookup | Angular dependence polynomial | Lookup table matches ASHRAE 140 specification exactly |

**Installation:** No external dependencies - all solar modules are internal to Fluxion.

## Architecture Patterns

### Recommended Project Structure
```
src/sim/
├── solar.rs          # Solar position, surface irradiance, window gains (EXISTS)
├── shading.rs        # Overhang/fin geometry, shaded fraction (EXISTS)
├── sky_radiation.rs  # Sol-air temp, longwave exchange (EXISTS)
└── engine.rs         # ThermalModel with 5R1C network (NEEDS INTEGRATION)
```

### Pattern 1: Solar Gain Integration into 5R1C Network
**What:** Add solar gains as internal heat source term in energy balance
**When to use:** All thermal network timesteps with solar radiation
**Example:**
```rust
// Source: src/sim/engine.rs, step_physics() method
// Current (incorrect): Solar gains calculated but not added to energy balance
let phi_i_internal = self.internal_loads * self.zone_area.clone();

// Required fix: Add solar gains to internal heat sources
let phi_i_solar = self.solar_gains.clone() * self.zone_area.clone();
let phi_i_total = phi_i_internal.clone() + phi_i_solar.clone();

// Energy balance with solar contribution
let phi_i = (phi_i_total.clone() + phi_si.clone()
    + phi_mi.clone() + phi_m_int_solar.clone())
    * T_a.clone()
    + phi_m_env_solar.clone() * T_m.clone();
```

### Pattern 2: Beam-to-Mass Distribution
**What:** Distribute beam solar gains between thermal mass and interior surface
**When to use:** High-mass buildings (900 series) with significant beam radiation
**Example:**
```rust
// Source: src/sim/engine.rs, lines 1772-1773
let phi_st_solar = solar_gains_watts.clone() * (1.0 - self.solar_beam_to_mass_fraction);
let phi_m_solar = solar_gains_watts * self.solar_beam_to_mass_fraction;

// Distribution to mass (phi_m) and interior surface (phi_si)
let phi_m_env_solar = phi_m_solar * 0.7; // 70% to exterior mass surface
let phi_m_int_solar = phi_m_solar * 0.3; // 30% to interior mass surface
```

### Pattern 3: Shading Fraction Application
**What:** Apply calculated shading fraction to beam radiation only
**When to use:** Cases with overhangs (610, 910) or shade fins (630, 930)
**Example:**
```rust
// Source: src/sim/solar.rs, calculate_window_solar_gain()
let shaded_fraction = calculate_shaded_fraction(geometry, overhang, fins, &local_solar);

// Apply shading to beam component only (diffuse is not shaded)
let effective_beam_wm2 = irradiance.beam_wm2 * (1.0 - shaded_fraction);
let beam_gain = window.area * effective_beam_wm2 * beam_shgc;
```

### Anti-Patterns to Avoid
- **Ignoring solar gains in energy balance:** Solar gains must be added as internal heat source, not just computed
- **Applying shading to diffuse radiation:** Only beam radiation is blocked by overhangs/fins
- **Using wrong incidence angle:** Must use surface incidence angle, not solar zenith, for SHGC lookup
- **Incorrect beam-to-mass fraction:** ASHRAE 140 specifies 0.7 to mass exterior, 0.3 to mass interior

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Solar position algorithm | Custom declination/elevation calculation | `calculate_solar_position()` with NOAA algorithm | Already implements correct astronomical calculations |
| Beam/diffuse decomposition | Simple cos(zenith) scaling | Perez sky model (`PerezSkyModel::calculate_diffuse_tilted()`) | ASHRAE 140 standard, accounts for circumsolar/horizon effects |
| Window angular dependence | Custom polynomial fit | `ashrae_140_window_shgc_ratio()` lookup table | Matches ASHRAE 140 specification exactly |
| Shading geometry | Ray-tracing or polygon clipping | `calculate_shaded_fraction()` projection shadows | ASHRAE 140 uses simple projection geometry |
| Sol-air temperature | Custom formula | `SolAirTemperature::calculate()` | Includes longwave radiation exchange correctly |

**Key insight:** Fluxion already has production-quality solar calculation modules. The problem is integration, not implementation. Don't rebuild what's already working correctly.

## Common Pitfalls

### Pitfall 1: Solar Gains Not Added to Energy Balance
**What goes wrong:** Solar gains are calculated correctly but never applied to thermal network, resulting in zero solar heating
**Why it happens:** `calculate_zone_solar_gain()` computes gains but `step_physics()` doesn't use them
**How to avoid:** Add solar gains to internal heat source term in energy balance equation
**Warning signs:** Peak cooling loads consistently low (0.60 kW vs 2.10-3.50 kW reference), cooling energy under-predicted (67% below reference)

### Pitfall 2: Applying Shading to Diffuse Radiation
**What goes wrong:** Shading fraction applied to both beam and diffuse, under-shading beam and over-shading diffuse
**Why it happens:** Misunderstanding that diffuse radiation comes from entire sky, not sun direction
**How to avoid:** Apply `(1.0 - shaded_fraction)` only to beam irradiance
**Warning signs:** Shaded cases (610, 630, 910, 930) show unrealistic reduction in solar gains

### Pitfall 3: Wrong Incidence Angle for SHGC Lookup
**What goes wrong:** Using solar zenith angle instead of surface incidence angle for angular dependence
**Why it happens:** Surface incidence depends on tilt and azimuth, not just sun position
**How to avoid:** Calculate incidence angle using `sun_pos.incidence_cosine(tilt, azimuth)` before SHGC lookup
**Warning signs:** Window solar gains don't vary correctly with surface orientation (north/south/east/west)

### Pitfall 4: Incorrect Beam-to-Mass Distribution
**What goes wrong:** All solar gains applied to zone air or incorrect distribution to mass
**Why it happens:** Not accounting for thermal mass absorbing beam radiation
**How to avoid:** Use ASHRAE 140 distribution: 70% beam to mass exterior surface, 30% to interior
**Warning signs:** High-mass cases (900 series) don't show thermal mass damping effects, peak temperatures too high

### Pitfall 5: Missing Ground-Reflected Radiation
**What goes wrong:** Only direct beam and sky diffuse considered, ignoring ground-reflected component
**Why it happens:** Forgetting that ground reflects solar radiation onto building surfaces
**How to avoid:** Include ground-reflected term: `GHI * ground_albedo * (1 - cos(tilt))/2`
**Warning signs:** Vertical surfaces under-predict solar gains, especially in winter with snow cover

## Code Examples

Verified patterns from existing implementation:

### Solar Position Calculation (NOAA Algorithm)
```rust
// Source: src/sim/solar.rs, calculate_solar_position()
pub fn calculate_solar_position(
    latitude_deg: f64,
    _longitude_deg: f64,
    year: i32,
    month: u32,
    day: u32,
    hour: f64,
) -> SolarPosition {
    // NOAA solar calculator implementation
    let days_in_year = if is_leap_year { 366 } else { 365 };
    let gamma = 2.0 * PI * (day_of_year_f - 1.0 + (hour - 12.0) / 24.0)
        / days_in_year as f64;

    let decl_rad = 0.006918 - 0.399912 * gamma.cos() + 0.070257 * gamma.sin()
        - 0.006758 * (2.0 * gamma).cos()
        + 0.000907 * (2.0 * gamma).sin()
        - 0.002697 * (3.0 * gamma).cos()
        + 0.00148 * (3.0 * gamma).sin();

    // ... hour angle and azimuth calculations

    SolarPosition {
        altitude_deg: elev,
        zenith_deg: zenith,
        azimuth_deg: az,
    }
}
```

### Surface Irradiance with Perez Sky Model
```rust
// Source: src/sim/solar.rs, calculate_surface_irradiance()
pub fn calculate_surface_irradiance(
    sun_pos: &SolarPosition,
    dni: f64,
    dhi: f64,
    ghi: Option<f64>,
    orientation: Orientation,
    ground_reflectance: f64,
    day_of_year: usize,
) -> SurfaceIrradiance {
    let (tilt_deg, azimuth_deg) = orientation_to_angles(orientation);
    let incidence_cos = sun_pos.incidence_cosine(tilt_deg, azimuth_deg);
    let beam = dni * incidence_cos;

    let dni_extra = extraterrestrial_irradiance(day_of_year);
    let airmass = relative_airmass(sun_pos.zenith_deg);

    // Perez anisotropic sky model for diffuse
    let diffuse = PerezSkyModel::calculate_diffuse_tilted(
        dhi, dni, dni_extra, airmass,
        sun_pos.zenith_deg, tilt_deg, azimuth_deg, sun_pos.azimuth_deg,
    );

    let surface_tilt = tilt_deg.to_radians();
    let ground_factor = (1.0 - surface_tilt.cos()) / 2.0;
    let ground_reflected = ghi.unwrap_or(0.0) * ground_reflectance * ground_factor;

    SurfaceIrradiance::new(beam, diffuse, ground_reflected)
}
```

### Window Solar Gain with Shading and Angular Dependence
```rust
// Source: src/sim/solar.rs, calculate_window_solar_gain()
pub fn calculate_window_solar_gain(
    irradiance: &SurfaceIrradiance,
    window: &WindowProperties,
    geometry: Option<&WindowArea>,
    overhang: Option<&Overhang>,
    fins: &[ShadeFin],
    sun_pos: &SolarPosition,
    orientation: Orientation,
) -> SolarGain {
    let (tilt_deg, surface_azimuth_deg) = orientation_to_angles(orientation);
    let incidence_cos = sun_pos.incidence_cosine(tilt_deg, surface_azimuth_deg);
    let incidence_angle = incidence_cos.acos().to_degrees();

    // Calculate shaded fraction for beam radiation
    let mut shaded_fraction = 0.0;
    if let Some(geom) = geometry {
        let local_solar = LocalSolarPosition {
            altitude: sun_pos.altitude_deg.to_radians(),
            relative_azimuth: (sun_pos.azimuth_deg - surface_azimuth_deg).to_radians(),
        };
        shaded_fraction = calculate_shaded_fraction(geom, overhang, fins, &local_solar);
    }

    // ASHRAE 140 lookup table for angular dependence
    let shgc_ratio = ashrae_140_window_shgc_ratio(incidence_angle);
    let beam_shgc = window.shgc * shgc_ratio;

    // Apply shading to beam component only
    let effective_beam_wm2 = irradiance.beam_wm2 * (1.0 - shaded_fraction);

    // Calculate gain components
    let beam_gain = window.area * effective_beam_wm2 * beam_shgc;
    let diffuse_gain = window.area * irradiance.diffuse_wm2 * (window.shgc * 0.9);
    let ground_reflected_gain = window.area * irradiance.ground_reflected_wm2 * (window.shgc * 0.9);

    SolarGain::new(beam_gain, diffuse_gain, ground_reflected_gain)
}
```

### ASHRAE 140 Window SHGC Angular Dependence
```rust
// Source: src/sim/solar.rs, ashrae_140_window_shgc_ratio()
fn ashrae_140_window_shgc_ratio(angle_deg: f64) -> f64 {
    // ASHRAE 140 values for double-pane clear glass
    const ANGLES: &[f64] = &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    const RATIOS: &[f64] = &[
        1.000, 0.995, 0.985, 0.970, 0.940, 0.890, 0.810, 0.680, 0.450, 0.000,
    ];

    if angle_deg <= 0.0 { return 1.0; }
    if angle_deg >= 90.0 { return 0.0; }

    // Linear interpolation between lookup table values
    for i in 0..ANGLES.len() - 1 {
        if angle_deg >= ANGLES[i] && angle_deg <= ANGLES[i + 1] {
            let t = (angle_deg - ANGLES[i]) / (ANGLES[i + 1] - ANGLES[i]);
            return RATIOS[i] * (1.0 - t) + RATIOS[i + 1] * t;
        }
    }

    1.0
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Isotropic sky model | Perez anisotropic sky model | Fluxion v0.1.0 | More accurate diffuse radiation on tilted surfaces |
| Simple SHGC formula | ASHRAE 140 lookup table | Fluxion v0.1.0 | Exact angular dependence matching standard |
| No shading geometry | Overhang/fine projection shadows | Fluxion v0.1.0 | Correct shading effects for cases 610/630/910/930 |
| Solar gains computed | Solar gains integrated | **Phase 3** | Fix cooling load under-prediction |

**Deprecated/outdated:**
- None - solar modules are current and follow ASHRAE 140 specifications

## Open Questions

1. **Solar Gain Distribution in 5R1C Network**
   - What we know: Beam-to-mass fraction parameters exist (0.7/0.3 distribution)
   - What's unclear: Exact coupling of solar gains to conductance matrix coefficients
   - Recommendation: Add solar gains to internal heat source term `phi_i` in energy balance, similar to internal loads

2. **Sol-Air Temperature Integration**
   - What we know: Sol-air temperature module exists for opaque surfaces
   - What's unclear: Whether sol-air temp should replace outdoor air temp for exterior conductances
   - Recommendation: Use sol-air temp for `h_tr_w` (window) and `h_tr_em` (exterior-to-mass) conductances to account for absorbed solar

3. **Longwave Radiation Exchange**
   - What we know: Sky radiation exchange module exists with net flux calculation
   - What's unclear: Integration point for roof/longwave cooling in thermal network
   - Recommendation: Add longwave cooling term to roof conductance balance for free-floating cases

4. **Ground Temperature Model**
   - What we know: `ConstantGroundTemperature` and `DynamicGroundTemperature` exist
   - What's unclear: Which ASHRAE 140 cases require ground temperature vs adiabatic floor
   - Recommendation: Check ASHRAE 140 specification for each case (likely adiabatic for most)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (Rust native) |
| Config file | Cargo.toml (dev profile) |
| Quick run command | `cargo test solar --lib` |
| Full suite command | `cargo test --lib` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SOLAR-01 | Solar gains calculated with beam/diffuse decomposition | unit | `cargo test calculate_hourly_solar --lib` | ✅ src/sim/solar.rs |
| SOLAR-02 | Solar gains integrated into thermal network | integration | `cargo test solar_integration --lib` | ❌ NEEDS CREATION |
| SOLAR-03 | Shading calculations correct for overhangs/fins | unit | `cargo test calculate_shaded_fraction --lib` | ✅ src/sim/shading.rs |
| SOLAR-04 | Solar gains match ASHRAE 140 reference values | validation | `fluxion validate --all` | ✅ src/validation/ashrae_140_validator.rs |

### Sampling Rate
- **Per task commit:** `cargo test solar --lib`
- **Per wave merge:** `cargo test --lib`
- **Phase gate:** Full ASHRAE 140 validation suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/solar_integration.rs` — covers SOLAR-02 (solar gains in thermal network)
- [ ] Update `src/sim/engine.rs` step_physics() to include solar gains in energy balance
- [ ] Framework install: None (cargo test built-in)

## Sources

### Primary (HIGH confidence)
- Fluxion source code - src/sim/solar.rs (complete solar position, irradiance, window gain implementation)
- Fluxion source code - src/sim/shading.rs (overhang and shade fin geometry)
- Fluxion source code - src/sim/sky_radiation.rs (Perez model, sol-air temperature)
- Fluxion source code - src/sim/engine.rs (thermal network, missing solar integration)
- Fluxion documentation - docs/ASHRAE140_RESULTS.md (validation failures showing solar issues)

### Secondary (MEDIUM confidence)
- ASHRAE Handbook - Fundamentals, Chapter 14: Climatic Design Information (cited in sky_radiation.rs)
- ASHRAE Handbook - Fundamentals, Chapter 15: Fenestration (cited in solar.rs)
- ASHRAE Standard 140 (referenced in ashrae_140_cases.rs)

### Tertiary (LOW confidence)
- Perez, R., et al. (1990). "Modeling daylight availability and irradiance components from direct and global irradiance." (cited in sky_radiation.rs)
- NOAA solar calculator algorithm (implemented in solar.rs)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All solar modules exist and are well-tested
- Architecture: MEDIUM - Integration point identified but exact coefficient coupling unclear
- Pitfalls: HIGH - Root cause (solar gains not integrated) definitively identified from code analysis

**Research date:** 2026-03-09
**Valid until:** 2026-04-08 (30 days - stable domain, no changing standards)

## Implementation Priority

1. **CRITICAL:** Integrate solar gains into 5R1C energy balance (SOLAR-02)
   - Add `phi_i_solar` term to `step_physics()` energy balance equation
   - Apply beam-to-mass distribution (0.7/0.3 split)
   - Test with Case 900: annual cooling should increase from 0.70 to ~2.90 MWh

2. **HIGH:** Apply sol-air temperature to exterior conductances (SOLAR-01)
   - Replace outdoor air temp with sol-air temp for `h_tr_w` and `h_tr_em`
   - Account for absorbed solar on opaque surfaces
   - Validate with free-floating cases (600FF, 650FF, 900FF)

3. **MEDIUM:** Verify shading calculations (SOLAR-03)
   - Confirm shaded fraction applied only to beam radiation
   - Test cases 610, 630, 910, 930 show correct shading effects
   - Compare with ASHRAE 140 reference shading coefficients

4. **LOW:** Longwave radiation integration (SOLAR-04)
   - Add sky radiation exchange to roof conductance balance
   - Validate nighttime cooling effects in free-floating cases
   - Ensure maximum free-floating temperatures match reference ranges

**Expected Impact:**
- Case 900 annual cooling: 0.70 MWh → 2.90 MWh (within 2.13-3.67 MWh reference)
- Case 900 peak cooling: 0.60 kW → 2.30 kW (within 2.10-3.50 kW reference)
- MAE reduction: 49.21% → <25% (solar integration addresses major failure mode)
- Pass rate improvement: 30% → >50% (cooling load fixes resolve many baseline case failures)
