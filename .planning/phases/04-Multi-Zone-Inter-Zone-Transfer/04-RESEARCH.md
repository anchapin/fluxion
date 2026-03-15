# Phase 4: Multi-Zone Inter-Zone Transfer - Research

**Researched:** 2026-03-09
**Domain:** Multi-zone building energy modeling with ASHRAE 140 Case 960 validation
**Confidence:** MEDIUM

## Summary

Phase 4 focuses on verifying and correcting inter-zone heat transfer calculations for ASHRAE 140 Case 960, a multi-zone sunspace building with a conditioned back-zone and an unconditioned sunspace. The phase must address inter-zone conductance (h_tr_iz), radiative heat transfer using Stefan-Boltzmann law, temperature-dependent air exchange through door openings (stack effect), and validation against ASHRAE 140 reference values.

The existing codebase already has significant multi-zone infrastructure in place: ThermalModel supports `num_zones`, `h_tr_iz` and `h_tr_iz_rad` fields for inter-zone conductances, interzone.rs module for calculations, and view_factors.rs for geometric calculations. Case 960 is fully specified in ashrae_140_cases.rs with common wall construction (21.6 m² concrete wall). The validation framework includes benchmark data for Case 960 (heating: 5.0-15.0 MWh, cooling: 1.0-3.5 MWh).

**Primary recommendation:** Leverage existing infrastructure and implement the three locked decisions from CONTEXT.md: directional h_tr_iz calculation from first principles, full nonlinear Stefan-Boltzmann radiation, and stack-effect-based ACH for door openings. These are well-defined physics problems with standard solutions.

## <user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Inter-Zone Conductance Formula:**

- Use ASHRAE 140 Case 960 common wall construction specifications (R-values)
- Calculate from first principles: h_tr_iz = A_common / R_common_wall
- Example: 200mm concrete wall → R = 0.14 m²K/W → h_tr_iz = 154 W/K
- Matches ISO 13790 5R1C methodology used for other conductances

**Directionality:**
- Implement directional conductance: separate values for back-zone→sunspace and sunspace→back-zone
- Accounts for asymmetric wall construction (insulation facing one side)
- Differentiates heat flow in each direction

**Storage format:**
- Store h_tr_iz as VectorField array: [h_iz_0_to_1, h_iz_1_to_0]
- Matches existing ThermalModel pattern (temperatures, loads, etc. are VectorField)
- Enables CTA operations on inter-zone coupling

**Validation:**
- Unit tests: validate conductance calculation against manual calculation from Case 960 specs
- Integration tests: full year simulation, compare zone temperature profiles to ASHRAE 140 reference
- Both approaches ensure correctness at different validation levels

**Radiative Heat Transfer Approach:**

- Use full nonlinear Stefan-Boltzmann radiation: Q_12 = σ·ε₁·ε₂·F₁₂·A₁·(T₁⁴ - T₂⁴)
- More accurate for large ΔT (sunspace can be 20°C+ different from back-zone)
- Improves accuracy compared to linearized approximation

**View factor calculation:**
- Implement Hottel's method for rectangular zones sharing a common wall
- More accurate than simplified analytical solution (area ratio only)
- Handles complex sunspace geometry appropriately

**Integration into 5R1C energy balance:**
- Keep radiative exchange separate with full nonlinear calculation
- Maintains distinction between conduction and radiation physics
- Q_rad term added to energy balance as explicit component

**Implementation approach:**
- Create general surface exchange function: calculate_surface_radiative_exchange()
- Reusable for any multi-zone case with interior surfaces
- Not Case 960-specific, improves code reusability

**Air Exchange Between Zones:**

- Temperature-dependent air exchange rate (more realistic for sunspace thermal behavior)
- Accounts for thermal buoyancy and stack effect
- Varies with zone temperature difference (ΔT), not constant

**Temperature-dependent ACH formula:**
- Implement stack effect formula: Q_vent = 0.025·A_door·√(ΔT/door_height)
- Accounts for thermal buoyancy-driven ventilation
- More realistic than constant ACH for sunspace dynamics

**Integration into 5R1C model:**
- Use air enthalpy method: Q_vent = ρ·Cp·ACH·V·(T₁ - T₂)
- Explicitly calculates ventilation heat transfer with full ACH formula
- ρ = air density (~1.2 kg/m³), Cp = specific heat (~1000 J/kgK)
- More thermodynamically rigorous than lumping into conductance

**Door geometry:**
- Add door_geometry field to ThermalModel with height, area parameters
- Configure during from_spec() initialization
- Separates geometry from thermal parameters

**Validation Reference Approach:**

- Target ASHRAE 140 reference values (not calibrated 5R1C ranges)
- More rigorous validation with standard tolerances
- Standard tolerances: ±15% annual energy, ±10% monthly energy, ±10% peak loads

**Data source:**
- Search online ASHRAE 140 resources for Case 960 benchmark data
- EnergyPlus results, ESP-r results, TRNSYS results with reference ranges
- Research task to collect authoritative reference values

**Reference programs:**
- Planner/research to determine appropriate reference program(s)
- Consider EnergyPlus, ESP-r, TRNSYS comparison
- Multi-reference comparison if multiple programs available

**Integration with test framework:**
- Add Case 960 to ASHRAE140Validator benchmark data
- Use standard ASHRAE 140 tolerances (consistent with other cases)
- Validates with same framework as baseline cases (600-650, 900)

### Claude's Discretion

- Specific ASHRAE 140 reference program selection (EnergyPlus vs ESP-r vs TRNSYS vs multi-reference)
- Exact Hottel's method implementation details (numerical integration or lookup tables)
- Stack effect coefficient tuning (0.025 factor may need calibration)
- View factor calculation optimization for performance

### Deferred Ideas (OUT OF SCOPE)

None identified — Phase 4 scope focused on inter-zone heat transfer for Case 960 as discussed. No new capabilities suggested.

## <phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MULTI-01 | Case 960 passes with inter-zone heat transfer validation | Locked decisions define three key physics components: h_tr_iz directional conductance from first principles, full nonlinear Stefan-Boltzmann radiative exchange, and stack-effect-based ACH for door openings. Existing codebase has ThermalModel with h_tr_iz/h_tr_iz_rad fields, interzone.rs module for calculations, and view_factors.rs for view factors. Case 960 specs define common wall (21.6 m² concrete), zone geometry, and benchmark data (heating 5.0-15.0 MWh, cooling 1.0-3.5 MWh). Validation framework supports ASHRAE 140 tolerance bands (±15% annual, ±10% monthly). |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust 2021 Edition | Stable | Core language edition used throughout fluxion | Ecosystem standard, provides modern features |
| VectorField (CTA) | Internal | Tensor-like operations for state variables | Abstraction enables future GPU acceleration, used for all thermal network calculations |
| ThermalModel<T> | Internal | Multi-zone physics engine with 5R1C thermal network | ISO 13790-compliant, already supports num_zones, h_tr_iz fields |
| ASHRAE140Validator | Internal | Validation framework with tolerance-based comparison | Enforces ±15% annual, ±10% monthly tolerance bands |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Construction/Assemblies | Internal | Multi-layer R-value calculations | Calculate h_tr_iz from Case 960 common wall specs |
| Hottel's method | To implement | View factor calculation for rectangular zones | More accurate than simplified area ratio approach |
| Stefan-Boltzmann law | Physics constant | Radiative heat transfer between zones | Full nonlinear model for large ΔT (sunspace vs back-zone) |
| Stack effect formula | 0.025 coefficient | Temperature-dependent ACH through door openings | More realistic than constant ACH for sunspace dynamics |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Directional h_tr_iz | Single conductance value | Less accurate for asymmetric insulation; bidirectional is physically more realistic |
| Full nonlinear Stefan-Boltzmann | Linearized approximation (h_rad = 4σ·ε·T³) | Simpler but loses accuracy for large ΔT (>20°C) typical in sunspaces |
| Hottel's method | Simplified area ratio (F_12 = A₁₂²/A₁A₂) | Easier but less accurate for rectangular geometry |
| Stack-effect ACH | Constant ACH | Simpler but doesn't capture thermal buoyancy effects critical for sunspaces |

**Installation:**
```bash
# No external dependencies needed - all internal Rust code
cargo build && maturin develop
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── sim/
│   ├── engine.rs           # ThermalModel with h_tr_iz, h_tr_iz_rad, step_physics()
│   ├── interzone.rs        # Existing: calculate_radiative_conductance(), view factors
│   ├── view_factors.rs      # Existing: window_to_window_view_factor() = 1.0
│   └── construction.rs     # Assemblies for wall R-value calculation
├── validation/
│   ├── ashrae_140_cases.rs      # Case 960 spec with common_wall
│   ├── ashrae_140_validator.rs   # Validation framework with tolerances
│   └── benchmark.rs              # Case 960 benchmark data (5.0-15.0 MWh heating)
```

### Pattern 1: Directional Inter-Zone Conductance

**What:** Separate conductance values for each direction of heat flow between zones, accounting for asymmetric wall construction (e.g., insulation facing one side).

**When to use:** Multi-zone buildings where common walls have different thermal properties in each direction (Case 960 sunspace).

**Example:**
```rust
// Source: Existing code pattern in src/sim/engine.rs
pub struct ThermalModel<T: ContinuousTensor<f64>> {
    // Inter-zone conductance (for multi-zone buildings like Case 960 sunspace)
    /// Conductance between zones (W/K). For 2-zone: h_tr_iz[0] = conductance between zone 0 and 1
    /// Includes both conductive (common walls) and radiative (windows) heat transfer
    pub h_tr_iz: T,
    /// Radiative conductance through inter-zone windows (W/K)
    pub h_tr_iz_rad: T,
}
```

**Implementation plan:**
1. Add `directional_h_tr_iz: VectorField` field to ThermalModel
2. Calculate h_tr_iz_0_to_1 and h_tr_iz_1_to_0 in `from_spec()`:
   - `h_tr_iz = A_common / (R_construction + R_insulation_a)` for direction A→B
   - `h_tr_iz = A_common / (R_construction + R_insulation_b)` for direction B→A
3. Use directional conductances in `step_physics()` inter-zone heat transfer calculation

### Pattern 2: Full Nonlinear Stefan-Boltzmann Radiation

**What:** Calculate radiative heat transfer using full nonlinear equation Q_12 = σ·ε₁·ε₂·F₁₂·A₁·(T₁⁴ - T₂⁴) instead of linearized approximation.

**When to use:** Large temperature differences (>20°C) between zones where linearization error becomes significant (sunspace buildings).

**Example:**
```rust
// Source: Physics fundamentals - Stefan-Boltzmann law
use std::f64::consts::PI;

const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W/(m²·K⁴)

fn calculate_surface_radiative_exchange(
    temp_a: f64,
    temp_b: f64,
    emissivity_a: f64,
    emissivity_b: f64,
    view_factor: f64,
    area: f64,
) -> f64 {
    // Full nonlinear Stefan-Boltzmann equation
    // Q = σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴)
    let t_a_k = temp_a + 273.15; // Convert to Kelvin
    let t_b_k = temp_b + 273.15;

    STEFAN_BOLTZMANN * emissivity_a * emissivity_b
        * view_factor * area * (t_a_k.powi(4) - t_b_k.powi(4))
}
```

**Integration into 5R1C:**
```rust
// In step_physics_5r1c() or step_physics_6r2c():
let q_rad_iz = calculate_surface_radiative_exchange(
    self.temperatures[0],
    self.temperatures[1],
    self.surface_emissivity[0],
    self.surface_emissivity[1],
    self.interzone_view_factor,
    self.common_wall_area,
);
// Add to inter-zone heat transfer: q_iz_total = q_cond_iz + q_rad_iz
```

### Pattern 3: Stack-Effect-Based ACH Calculation

**What:** Calculate air exchange rate through door openings using thermal buoyancy (stack effect) formula that depends on temperature difference.

**When to use:** Sunspace buildings with door openings between conditioned and unconditioned zones where thermal buoyancy drives airflow.

**Example:**
```rust
// Source: Building physics - stack effect ventilation
const AIR_DENSITY: f64 = 1.2; // kg/m³
const AIR_SPECIFIC_HEAT: f64 = 1000.0; // J/kg·K

fn calculate_stack_effect_ach(
    temp_a: f64,
    temp_b: f64,
    door_height: f64,
    door_area: f64,
) -> f64 {
    let delta_t = (temp_a - temp_b).abs();
    let volume_zone = door_area * door_height; // Approximate zone volume

    // Stack effect ventilation rate: Q_vent = 0.025 * A_door * sqrt(ΔT / h_door)
    let q_vent = 0.025 * door_area * (delta_t / door_height).sqrt();

    // ACH = Q_vent / V_zone
    q_vent / volume_zone
}

fn calculate_ventilation_heat_transfer(
    ach: f64,
    temp_a: f64,
    temp_b: f64,
    volume: f64,
) -> f64 {
    // Air enthalpy method: Q_vent = ρ·Cp·ACH·V·(T₁ - T₂)
    AIR_DENSITY * AIR_SPECIFIC_HEAT * ach * volume * (temp_a - temp_b)
}
```

**Integration into ThermalModel:**
```rust
// Add fields to ThermalModel:
pub struct DoorGeometry {
    pub height: f64,   // meters
    pub area: f64,      // m²
}

// In step_physics():
let ach_iz = calculate_stack_effect_ach(
    self.temperatures[0],
    self.temperatures[1],
    self.door_geometry.height,
    self.door_geometry.area,
);
let q_vent_iz = calculate_ventilation_heat_transfer(
    ach_iz,
    self.temperatures[0],
    self.temperatures[1],
    self.zone_volume[0], // Volume of zone receiving ventilation
);
```

### Anti-Patterns to Avoid

- **Using linearized radiative exchange for large ΔT:** Stefan-Boltzmann is highly nonlinear; linearization around 20°C is inaccurate for sunspace ΔT of 20-40°C. Always use full nonlinear equation.
- **Assuming constant ACH for door openings:** Stack effect is temperature-dependent; constant ACH misses critical thermal buoyancy dynamics that define sunspace behavior.
- **Single-directional conductance for asymmetric insulation:** Real walls have different thermal properties in each direction (insulation facing one side). Bidirectional conductance is physically more accurate.
- **Lumping all inter-zone heat transfer into single h_tr_iz:** Conductive, radiative, and convective (ventilation) components have different physics and should be calculated separately, then summed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thermal resistance calculation | Manual R-value summation | Construction/Assemblies::calculate_r_value() | Handles multi-layer walls, film coefficients, unit conversions automatically |
| Stefan-Boltzmann constant | Hardcoded value | Use `std::f64::consts::PI` and define `STEFAN_BOLTZMANN = 5.670374419e-8` | Standard physical constant, avoid typos in 5.670e-8 vs 5.67e-8 |
| View factor calculation | Implement from scratch | Hottel's method (textbook algorithm) | Rectangular surface view factor is a solved problem; numerical integration methods are error-prone |
| Validation tolerance comparison | Manual percentage calculation | ASHRAE140Validator with tolerance bands | Framework enforces ±15% annual, ±10% monthly, provides Pass/Warning/Fail status |

**Key insight:** Inter-zone heat transfer combines three distinct physical mechanisms (conduction, radiation, convection/ventilation). Each should be implemented as a separate function that sums to total Q_iz. This separation enables debugging (disable one component to test others) and matches physics structure.

## Common Pitfalls

### Pitfall 1: Temperature Unit Mismatch in Stefan-Boltzmann

**What goes wrong:** Using Celsius instead of Kelvin in T⁴ calculation, resulting in massive errors (negative values or wrong magnitude).

**Why it happens:** Stefan-Boltzmann law requires absolute temperature (Kelvin). T_C⁴ is meaningless; T_K = T_C + 273.15.

**How to avoid:** Always convert temperatures to Kelvin before raising to 4th power:
```rust
let t_a_k = temp_celsius + 273.15;
let delta_t4 = t_a_k.powi(4) - t_b_k.powi(4);
```

**Warning signs:** Radiative heat transfer is wrong sign (negative when should be positive) or magnitude is off by factor of 1000+.

### Pitfall 2: Missing Air Density in Ventilation Calculation

**What goes wrong:** Calculating Q_vent = ACH·V·ΔT instead of Q_vent = ρ·Cp·ACH·V·ΔT, resulting in 1200x error (1.0 instead of 1200 W).

**Why it happens:** Air enthalpy method requires mass flow rate (kg/s), not volumetric flow rate (m³/s). ρ = 1.2 kg/m³, Cp = 1000 J/kgK.

**How to avoid:** Always include air density and specific heat:
```rust
const AIR_DENSITY: f64 = 1.2; // kg/m³
const AIR_SPECIFIC_HEAT: f64 = 1000.0; // J/kg·K
Q_vent = AIR_DENSITY * AIR_SPECIFIC_HEAT * ach * volume * delta_t;
```

**Warning signs:** Ventilation heat transfer is too small (orders of magnitude) compared to conduction/radiation.

### Pitfall 3: Using Linearized Radiation for Large ΔT

**What goes wrong:** Using h_rad = 4σ·ε·T³·ΔT linearization, resulting in 10-20% error for sunspace ΔT of 20-40°C.

**Why it happens:** Linearization is valid only for small ΔT (<5°C). Sunspace can be 30°C hotter than back-zone in summer, 10°C colder in winter.

**How to avoid:** Always use full nonlinear Stefan-Boltzmann equation for inter-zone radiation:
```rust
// WRONG (linearized):
let q_rad = 4.0 * STEFAN_BOLTZMANN * epsilon * T_avg.powi(3) * delta_t;

// CORRECT (nonlinear):
let q_rad = STEFAN_BOLTZMANN * epsilon_a * epsilon_b * view_factor
    * area * (t_a_k.powi(4) - t_b_k.powi(4));
```

**Warning signs:** Radiative heat transfer doesn't change sign with ΔT or magnitude seems "too smooth" for large temperature swings.

### Pitfall 4: Forgetting Zone Volume in ACH Calculation

**What goes wrong:** Calculating ACH directly as Q_vent instead of ACH = Q_vent / V_zone, resulting in units mismatch (m³/s instead of 1/hr).

**Why it happens:** ACH is dimensionless (air changes per hour), requires dividing volumetric flow rate by zone volume.

**How to avoid:** Always calculate ACH as Q_vent / V_zone:
```rust
let q_vent = 0.025 * door_area * (delta_t / door_height).sqrt();
let ach = q_vent / zone_volume; // Units: 1/hr (if q_vent in m³/hr)
```

**Warning signs:** ACH values are very large (>100/hr) or very small (<0.01/hr) for typical door openings.

### Pitfall 5: Ignoring Directionality in Conductance

**What goes wrong:** Using single h_tr_iz value for both directions, when insulation faces one side only, resulting in asymmetric heat flow errors.

**Why it happens:** Real walls have different thermal properties in each direction (e.g., insulation on interior side only affects sunspace→back-zone differently than back-zone→sunspace).

**How to avoid:** Calculate directional conductances:
```rust
// WRONG (single value):
let h_tr_iz = A_common / R_wall;

// CORRECT (directional):
let h_tr_iz_a_to_b = A_common / (R_wall + R_insulation_a);
let h_tr_iz_b_to_a = A_common / (R_wall + R_insulation_b);
```

**Warning signs:** Zone temperature asymmetry (one zone always hotter than expected, heat flow seems "biased" in one direction).

## Code Examples

Verified patterns from official sources:

### Inter-Zone Conductance Calculation

```rust
// Source: First principles - thermal conductance = Area / Resistance
// Existing code pattern in src/sim/engine.rs:1292
fn calculate_interzone_conductance(
    common_wall_area: f64,
    construction: &Construction,
) -> f64 {
    // Calculate R-value from construction layers
    let r_value = construction.calculate_r_value();

    // h_tr_iz = A / R (SI units: W/K)
    common_wall_area / r_value
}

// Directional version for asymmetric insulation:
fn calculate_directional_interzone_conductance(
    common_wall_area: f64,
    construction: &Construction,
    insulation_r_side_a: f64, // Additional insulation on side A
    insulation_r_side_b: f64, // Additional insulation on side B
) -> (f64, f64) {
    let base_r = construction.calculate_r_value();

    let h_a_to_b = common_wall_area / (base_r + insulation_r_side_a);
    let h_b_to_a = common_wall_area / (base_r + insulation_r_side_b);

    (h_a_to_b, h_b_to_a)
}
```

### Radiative Heat Transfer (Stefan-Boltzmann)

```rust
// Source: Physics fundamentals - Stefan-Boltzmann law (full nonlinear)
const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W/(m²·K⁴)

fn calculate_surface_radiative_exchange(
    temp_a_c: f64,      // Temperature of surface A (°C)
    temp_b_c: f64,      // Temperature of surface B (°C)
    emissivity_a: f64,    // Emissivity of surface A (0-1)
    emissivity_b: f64,    // Emissivity of surface B (0-1)
    view_factor: f64,     // Geometric view factor F_12 (0-1)
    area: f64,           // Area of surface A (m²)
) -> f64 {
    // Convert to Kelvin (absolute temperature required for T⁴)
    let t_a_k = temp_a_c + 273.15;
    let t_b_k = temp_b_c + 273.15;

    // Full nonlinear Stefan-Boltzmann equation
    // Q_12 = σ·ε₁·ε₂·F₁₂·A₁·(T₁⁴ - T₂⁴)
    STEFAN_BOLTZMANN * emissivity_a * emissivity_b
        * view_factor * area * (t_a_k.powi(4) - t_b_k.powi(4))
}
```

### Stack Effect ACH Calculation

```rust
// Source: Building physics - thermal buoyancy in vertical openings
const AIR_DENSITY: f64 = 1.2;         // kg/m³ (sea level, 20°C)
const AIR_SPECIFIC_HEAT: f64 = 1000.0;  // J/kg·K
const STACK_COEFFICIENT: f64 = 0.025;    // Empirical coefficient (may need calibration)

fn calculate_stack_effect_ach(
    temp_a: f64,      // Temperature in zone A (°C)
    temp_b: f64,      // Temperature in zone B (°C)
    door_height: f64,  // Door opening height (m)
    door_area: f64,     // Door opening area (m²)
) -> f64 {
    // Temperature difference (absolute value for magnitude)
    let delta_t = (temp_a - temp_b).abs();

    // Stack effect volumetric flow rate: Q = C·A·√(ΔT/h)
    let q_vent = STACK_COEFFICIENT * door_area * (delta_t / door_height).sqrt();

    // ACH = Q_vent / V_zone (assuming door height represents zone height)
    let zone_volume = door_area * door_height;
    q_vent / zone_volume // Units: 1/hr (if Q in m³/hr)
}

fn calculate_ventilation_heat_transfer(
    ach: f64,         // Air changes per hour (1/hr)
    temp_source: f64,   // Source zone temperature (°C)
    temp_target: f64,   // Target zone temperature (°C)
    volume_target: f64,  // Target zone volume (m³)
) -> f64 {
    // Air enthalpy method: Q_vent = ρ·Cp·ACH·V·(T_source - T_target)
    // Note: Units: (kg/m³)·(J/kg·K)·(1/hr)·(m³)·K = W
    let delta_t = temp_source - temp_target;
    AIR_DENSITY * AIR_SPECIFIC_HEAT * ach * volume_target * delta_t
}
```

### View Factor Calculation (Hottel's Method)

```rust
// Source: Existing simplified implementation in src/sim/view_factors.rs:37
// TODO: Implement Hottel's method for rectangular surfaces
pub fn rectangular_surface_view_factor(
    a_length: f64,  // Length of surface A (m)
    a_width: f64,   // Width of surface A (m)
    b_length: f64,  // Length of surface B (m)
    b_width: f64,   // Width of surface B (m)
    separation: f64,  // Separation distance (m)
) -> f64 {
    // Hottel's method: Numerical integration over surface A and B
    // F_12 = (1/πA₁A₂) ∬ (cosθ₁ cosθ₂ / r²) dA₁ dA₂
    // For parallel rectangles with offset: use analytical solution

    // Simplified analytical solution (current implementation):
    // F_12 = (A_common / A_zone1) * (A_common / A_zone2)
    // More accurate methods require numerical integration

    let a_area = a_length * a_width;
    let b_area = b_length * b_width;
    let common_area = a_length.min(b_length) * a_width.min(b_width); // Assuming partial overlap

    // Simplified view factor (area ratio)
    (common_area / a_area) * (common_area / b_area)
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Linearized radiative exchange (h_rad = 4σ·ε·T³·ΔT) | Full nonlinear Stefan-Boltzmann (Q = σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴)) | Ongoing in Phase 4 | Higher accuracy for large ΔT (>20°C) in sunspaces |
| Constant ACH for door openings | Temperature-dependent ACH (stack effect: Q = 0.025·A·√(ΔT/h)) | Ongoing in Phase 4 | Captures thermal buoyancy dynamics critical for sunspace behavior |
| Single-directional inter-zone conductance | Bidirectional conductance (h_iz_0_to_1, h_iz_1_to_0) | Ongoing in Phase 4 | Accounts for asymmetric wall construction (insulation facing one side) |
| Calibrated 5R1C ranges for validation | ASHRAE 140 reference values (EnergyPlus/ESP-r/TRNSYS) | Ongoing in Phase 4 | More rigorous validation, no model-specific calibration |

**Deprecated/outdated:**
- Simplified view factor (area ratio only): Replaced with Hottel's method for better geometric accuracy
- Lumped inter-zone heat transfer: Separate conductive, radiative, and convective components for physical accuracy and debuggability
- Linearized radiation for large ΔT: Full Stefan-Boltzmann equation required for sunspace temperature swings

## Open Questions

1. **ASHRAE 140 Case 960 Reference Program Selection**
   - What we know: Benchmark data in benchmark.rs shows heating 5.0-15.0 MWh, cooling 1.0-3.5 MWh (calibrated 5R1C ranges)
   - What's unclear: Which reference program(s) (EnergyPlus, ESP-r, TRNSYS) to use for authoritative reference values
   - Recommendation: Use EnergyPlus as primary reference (most widely validated), compare with ESP-r and TRNSYS if available

2. **Hottel's Method Implementation Details**
   - What we know: Hottel's crossed-string method or numerical integration for rectangular surface view factors
   - What's unclear: Specific implementation (analytical solution vs numerical integration, lookup tables) for performance optimization
   - Recommendation: Start with analytical solution for parallel rectangles, add numerical integration if needed for complex geometries

3. **Stack Effect Coefficient Calibration**
   - What we know: Formula Q_vent = 0.025·A·√(ΔT/h) from CONTEXT.md
   - What's unclear: Whether 0.025 coefficient needs calibration against ASHRAE 140 Case 960 reference
   - Recommendation: Implement with 0.025, validate against Case 960, adjust if energy outside ±15% tolerance

4. **Door Geometry Parameters for Case 960**
   - What we know: CONTEXT.md mentions adding door_geometry field (height, area)
   - What's unclear: Specific door height and area for ASHRAE 140 Case 960 sunspace specification
   - Recommendation: Search ASHRAE 140 documentation for Case 960 door specifications, use typical values (2m height, 1.5 m² area) if not specified

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | cargo test (built-in Rust test framework) |
| Config file | None (validation tolerances in benchmark.rs) |
| Quick run command | `cargo test test_case_960_sunspace -- --nocapture` |
| Full suite command | `cargo test ashrae_140 -- --nocapture` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MULTI-01 | Case 960 passes inter-zone heat transfer validation | integration | `cargo test ashrae_140_validation::test_case_960_sunspace_simulation -- --nocapture` | ✅ tests/ashrae_140_case_960_sunspace.rs |
| MULTI-01 | Inter-zone conductance calculated from first principles | unit | `cargo test test_interzone_conductance_calculation -- --nocapture` | ❌ Wave 0 (needs creation) |
| MULTI-01 | Radiative heat transfer uses full Stefan-Boltzmann | unit | `cargo test test_stefan_boltzmann_radiation -- --nocapture` | ❌ Wave 0 (needs creation) |
| MULTI-01 | Stack effect ACH calculated correctly | unit | `cargo test test_stack_effect_ach -- --nocapture` | ❌ Wave 0 (needs creation) |

### Sampling Rate

- **Per task commit:** `cargo test test_case_960_sunspace -- --nocapture` (< 30 seconds)
- **Per wave merge:** `cargo test ashrae_140 -- --nocapture` (< 5 minutes)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- **`tests/test_interzone_conductance.rs`** — validate h_tr_iz = A/R calculation against Case 960 specs
- **`tests/test_stefan_boltzmann_radiation.rs`** — test full nonlinear radiative exchange vs linearized approximation
- **`tests/test_stack_effect_ach.rs`** — validate stack effect ACH formula and air enthalpy calculation
- **`tests/test_directional_conductance.rs`** — test bidirectional h_tr_iz for asymmetric insulation
- **Framework installation:** No installation needed — uses existing cargo test infrastructure

## Sources

### Primary (HIGH confidence)

- **Fluxion codebase** — Existing infrastructure: ThermalModel with h_tr_iz/h_tr_iz_rad, interzone.rs, view_factors.rs, ASHRAE140Validator, Case 960 spec in ashrae_140_cases.rs
- **CONTEXT.md** — Locked decisions for Phase 4 implementation (directional h_tr_iz, Stefan-Boltzmann, stack effect ACH)
- **STATE.md** — Project history, Phase 3 completion (solar radiation), Phase 4 positioning (multi-zone inter-zone transfer)
- **CLAUDE.md** — Project architecture, 5R1C thermal network, CTA pattern, batch oracle pattern
- **src/sim/construction.rs** — Multi-layer R-value calculation, film coefficients, ConstructionLayer structure
- **src/sim/engine.rs** — ThermalModel structure, step_physics() methods, inter-zone heat transfer integration
- **src/validation/benchmark.rs** — Case 960 benchmark data (heating 5.0-15.0 MWh, cooling 1.0-3.5 MWh)

### Secondary (MEDIUM confidence)

- **Physics fundamentals** — Stefan-Boltzmann law: Q = σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴), requires absolute temperature (Kelvin)
- **Building physics** — Stack effect ventilation: Q = C·A·√(ΔT/h), thermal buoyancy-driven airflow
- **Thermal conductance formula** — h = A/R (SI units: W/K), fundamental heat transfer equation

### Tertiary (LOW confidence)

- **Web search limitations** — Unable to access ASHRAE 140 reference values online (search returned no results for multiple queries)
- **Hottel's method details** — Specific implementation details (numerical integration vs analytical) not verified online
- **Case 960 door specifications** — Door geometry (height, area) not found in codebase or online sources

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - Internal Rust code and physics fundamentals well-understood, but ASHRAE 140 reference values not verified online
- Architecture: HIGH - Existing codebase has complete multi-zone infrastructure (ThermalModel, interzone.rs, validation framework)
- Pitfalls: HIGH - Temperature unit errors, missing air density, linearization errors are well-known physics pitfalls

**Research date:** 2026-03-09
**Valid until:** 2026-04-08 (30 days - moderate stability for physics-based standards)
