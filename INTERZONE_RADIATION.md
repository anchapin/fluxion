# Inter-Zone Thermal Transfer Model

## Overview

This document describes the multi-zone inter-zone thermal transfer model implemented in the Fluxion engine. The model enables simulation of buildings with multiple thermal zones that exchange heat through common surfaces, including:

- **Conductive heat transfer** through opaque surfaces (walls, floors, ceilings)
- **Radiative heat transfer** between interior surfaces
- **Air mixing** through openings (doors, vents)
- **Solar gain distribution** across multiple zones

The model is designed for ASHRAE 140 Case 960 validation but is applicable to general multi-zone building simulations.

## Table of Contents

1. [Thermal Network Fundamentals](#thermal-network-fundamentals)
2. [Conductive Inter-Zone Transfer](#conductive-inter-zone-transfer)
3. [Radiative View Factor Model](#radiative-view-factor-model)
4. [Window-to-Window Radiation Exchange](#window-to-window-radiation-exchange)
5. [Multi-Zone Thermal Network Equations](#multi-zone-thermal-network-equations)
6. [Implementation Details](#implementation-details)
7. [Code Examples](#code-examples)
8. [Validation Test Expectations](#validation-test-expectations)
9. [Limitations and Assumptions](#limitations-and-assumptions)

---

## Thermal Network Fundamentals

### Heat Transfer Mechanisms

The inter-zone thermal transfer model implements three primary heat transfer mechanisms:

1. **Conductive Transfer** through opaque surfaces:
   ```
   Q_cond = U × A × (T_j - T_i)
   ```
   Where:
   - `U` = Overall heat transfer coefficient (W/m²K)
   - `A` = Surface area (m²)
   - `T_j, T_i` = Temperatures of zones j and i (K)

2. **Radiative Exchange** between interior surfaces:
   ```
   Q_rad = ε_i × σ × (T_j^4 - T_i^4) × F_ij × A_i
   ```
   Where:
   - `ε_i` = Surface emissivity
   - `σ` = Stefan-Boltzmann constant (5.67×10⁻⁸ W/m²K⁴)
   - `F_ij` = View factor from surface i to j
   - Linearized approximation for small temperature differences:
     ```
     Q_rad ≈ h_rad × A × (T_j - T_i)
     ```

3. **Air Mixing** through openings:
   ```
   Q_mix = ρ × cp × V_dot × (T_j - T_i)
   ```
   Where:
   - `ρ` = Air density (kg/m³)
   - `cp` = Specific heat capacity of air (J/kgK)
   - `V_dot` = Air flow rate through opening (m³/s)

---

## Conductive Inter-Zone Transfer

### Conductance Calculation

Conductive inter-zone conductance is computed from the U-value and area of common surfaces:

```rust
// Example: Conductive conductance through common wall
let u_value = construction.u_value;  // W/m²K
let wall_area = common_wall_area;    // m²
let h_tr_iz_conductive = u_value * wall_area;  // W/K
```

### Thermal Network Integration

Conductive heat transfer is integrated into the thermal network as a coupling term:

```
q_iz_i = Σ h_tr_iz × (T_j - T_i)
        for all j ≠ i
```

Where:
- `q_iz_i` = Net inter-zone heat flow into zone i (W)
- `h_tr_iz` = Total inter-zone conductance (W/K)
- `T_j` = Temperature of neighboring zone j (K)
- `T_i` = Temperature of zone i (K)

### Implementation

The conductive inter-zone heat flow is computed in `ThermalModel::step()`:

```rust
let inter_zone_heat: Vec<f64> = if num_zones > 1 {
    let temps = self.temperatures.as_ref();
    let h_iz_val = self.h_tr_iz.as_ref().first().copied().unwrap_or(0.0);

    (0..num_zones)
        .map(|i| {
            let mut q_iz = 0.0;
            for j in 0..num_zones {
                if i != j {
                    q_iz += h_iz_val * (temps[j] - temps[i]);
                }
            }
            q_iz
        })
        .collect()
} else {
    vec![0.0; num_zones]
};
```

---

## Radiative View Factor Model

### View Factor Fundamentals

The **view factor** `F_ij` represents the fraction of radiation leaving surface i that directly reaches surface j. Key properties:

- **Reciprocity**: `A_i × F_ij = A_j × F_ji`
- **Summation**: `Σ_j F_ij = 1` (for enclosures)
- **Symmetry**: `F_ij = F_ji` for identical, parallel surfaces

### Sky View Factor

For exterior surfaces, the sky view factor determines the fraction of radiation reaching the sky:

```
F_sky = (1 + cos(tilt)) / 2
```

Where:
- `tilt` = Surface tilt angle from horizontal (0° = horizontal roof, 90° = vertical wall)

Example values:
- Horizontal roof: `F_sky = 1.0` (full sky view)
- Vertical wall: `F_sky = 0.5` (half sky, half ground)
- Horizontal floor: `F_sky = 0.0` (full ground view)

### Interior Radiative Exchange

Interior surface-to-surface radiative exchange uses a simplified area-weighted approach:

```rust
// Linearized radiative conductance between surfaces
let h_rad = 4.0 × ε × σ × T_mean³ × F_ij × A_i
```

Where:
- `T_mean` = Mean temperature (K)
- `ε` = Effective emissivity: `1 / (1/ε_i + 1/ε_j - 1)`

### Implementation in Sky Radiation

The `SkyRadiation` module calculates radiative exchange with the sky:

```rust
pub struct SkyRadiation {
    pub surface_emissivity: f64,
    pub sky_view_factor: f64,
}

impl SkyRadiation {
    // Calculate radiative heat transfer to/from sky
    pub fn radiative_exchange(&self, t_surface: f64, t_sky: f64) -> f64 {
        self.surface_emissivity
            * STEFAN_BOLTZMANN
            * self.sky_view_factor
            * (t_sky.powi(4) - t_surface.powi(4))
    }

    // Linearized radiative conductance
    pub fn radiative_conductance(&self, t_mean: f64) -> f64 {
        4.0 * self.surface_emissivity * self.sky_view_factor * STEFAN_BOLTZMANN * t_mean.powi(3)
    }
}
```

---

## Window-to-Window Radiation Exchange

### Window Radiation Model

Windows exchange longwave radiation with other interior surfaces. The radiation exchange depends on:

1. **Window emissivity**: Typical values 0.84-0.90 for clear glass
2. **View factors**: Determined by relative position and area
3. **Temperature differences**: Driving force for radiation

### Combined Conductance

Window-to-window radiative conductance is combined with conductive conductance:

```rust
let total_h_iz = h_iz_conductive + h_iz_radiative;
```

### ASHRAE 140 Case 960 Application

In Case 960 (sunspace configuration):
- Sunspace glazing exchanges radiation with back-zone interior surfaces
- High view factor due to large glazing area (24 m²)
- Contributes significantly to inter-zone heat transfer

---

## Multi-Zone Thermal Network Equations

### Energy Balance for Zone i

The energy balance for each zone i includes:

```
C_i × dT_i/dt = Σ Q_external + Σ Q_interzone + Q_internal + Q_solar + Q_HVAC
```

Where:
- `C_i` = Thermal capacitance of zone i (J/K)
- `Q_external` = Heat transfer to/from external environment (W)
- `Q_interzone` = Heat transfer with other zones (W)
- `Q_internal` = Internal heat gains (W)
- `Q_solar` = Solar gains through windows (W)
- `Q_HVAC` = HVAC heating/cooling (W)

### Inter-Zone Heat Transfer Term

The inter-zone term for zone i is:

```
Q_interzone_i = Σ_j [h_tr_iz × (T_j - T_i)]
                for j ≠ i
```

### Thermal Network State Equations

Using superposition of contributions:

```
num = Q_internal + Q_solar + Q_HVAC + h_ext × T_ext + h_iz × T_neighbors
den = h_ext + h_iz + h_int
T_zone = num / den
```

Where:
- `h_ext` = External conductance (W/K)
- `h_iz` = Inter-zone conductance (W/K)
- `h_int` = Internal surface conductance (W/K)

---

## Implementation Details

### Data Structures

#### Inter-Zone Conductance Storage

```rust
// In ThermalModel
pub struct ThermalModel<T> {
    // ... other fields ...
    pub h_tr_iz: T,           // Conductive inter-zone conductance (W/K)
    pub h_tr_iz_rad: T,       // Radiative inter-zone conductance (W/K)
    // ...
}
```

Where `T` is a tensor field type (e.g., `VectorField`) enabling automatic differentiation.

### Computation Steps

1. **Initialization** (`ThermalModel::from_spec()`):
   - Calculate inter-zone conductances from common surfaces
   - Apply radiative corrections based on surface properties
   - Store in `h_tr_iz` and `h_tr_iz_rad`

2. **Simulation Step** (`ThermalModel::step()`):
   - Read current zone temperatures
   - Compute inter-zone heat flows: `q_iz = h_iz × ΔT`
   - Add to internal loads: `φ_ia_with_iz = φ_ia + q_iz`
   - Update zone temperatures using modified energy balance

3. **HVAC Integration**:
   - Each zone has independent HVAC setpoints
   - Unconditioned zones have `hvac_enabled = false`
   - HVAC energy is tracked per zone

### Sensitivity Computation

The inter-zone model supports automatic differentiation for sensitivity analysis:

```rust
let inter_zone_heat: Vec<f64> = ...;
let q_iz_tensor: T = VectorField::new(inter_zone_heat).into();
```

This allows computation of sensitivities to inter-zone conductances.

---

## Code Examples

### Example 1: Two-Zone Setup

```rust
use crate::sim::engine::ThermalModel;
use crate::sim::construction::SurfaceType;
use crate::validation::ashrae_140_cases::ASHRAE140Case;

// Create Case 960 (2-zone sunspace)
let spec = ASHRAE140Case::Case960.spec();
let mut model = ThermalModel::<VectorField>::from_spec(&spec);

// Check inter-zone conductance
let h_iz = model.h_tr_iz.as_ref();
println!("Inter-zone conductance: {:.2} W/K", h_iz[0]);

// Run simulation
let outdoor_temp = 10.0;  // °C
let solar_gain = 1000.0;  // W
model.set_outdoor_temperature(outdoor_temp);
model.set_loads(&[solar_gain / model.zone_area[0], 0.0]);
model.step();

// Check temperature difference
let temps = model.temperatures.as_ref();
println!("Zone 0 temp: {:.1}°C", temps[0]);
println!("Zone 1 temp: {:.1}°C", temps[1]);
```

### Example 2: Calculating View Factors

```rust
use crate::sim::sky_radiation::SkyRadiation;

// Vertical south wall
let wall_rad = SkyRadiation::from_tilt(90.0, 0.9);
println!("Wall sky view factor: {:.2}", wall_rad.sky_view_factor);
// Output: Wall sky view factor: 0.50

// Horizontal roof
let roof_rad = SkyRadiation::from_tilt(0.0, 0.9);
println!("Roof sky view factor: {:.2}", roof_rad.sky_view_factor);
// Output: Roof sky view factor: 1.00
```

### Example 3: Inter-Zone Heat Flow Calculation

```rust
// Calculate heat flow from zone 0 to zone 1
let h_iz = 50.0;  // W/K
let t0 = 25.0;    // °C
let t1 = 20.0;    // °C

let q_01 = h_iz * (t1 - t0);  // Heat flow into zone 0 from zone 1
println!("Heat flow: {:.2} W", q_01);
// Output: Heat flow: -250.00 W (flowing from zone 0 to zone 1)
```

### Example 4: Radiative Exchange Calculation

```rust
use crate::sim::sky_radiation::STEFAN_BOLTZMANN;

let surface_temp = 293.15;  // K (20°C)
let sky_temp = 263.15;      // K (-10°C)
let emissivity = 0.9;
let sky_view_factor = 0.5;

let q_rad = emissivity * STEFAN_BOLTZMANN * sky_view_factor
          * (sky_temp.powi(4) - surface_temp.powi(4));

println!("Radiative heat loss: {:.2} W/m²", q_rad);
// Output: Radiative heat loss: -28.47 W/m² (loss to sky)
```

---

## Validation Test Expectations

### ASHRAE 140 Case 960 Validation

#### Test Setup

```rust
#[test]
fn test_case_960_inter_zone_heat_transfer() {
    let spec = ASHRAE140Case::Case960.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Verify inter-zone conductance is set
    let h_iz = model.h_tr_iz.as_ref();
    assert!(h_iz[0] > 0.0, "Inter-zone conductance should be > 0");

    // Verify two zones
    assert_eq!(model.num_zones(), 2, "Case 960 should have 2 zones");

    // Verify sunspace is free-floating
    assert_eq!(model.hvac_enabled[1], false, "Zone 1 (sunspace) should be unconditioned");
}
```

#### Expected Behavior

1. **Temperature Coupling**:
   - Sunspace temperature fluctuates more than back-zone
   - Sunspace temperatures can exceed back-zone temperatures
   - Heat flows from warm to cool zones based on ΔT

2. **HVAC Energy**:
   - Back-zone (Zone 0) has both heating and cooling
   - Sunspace (Zone 1) has zero HVAC energy
   - Annual heating: 1.65-2.45 MWh (back-zone only)
   - Annual cooling: 1.55-2.78 MWh (back-zone only)

3. **Inter-Zone Heat Transfer**:
   - Significant heat transfer through common wall
   - Direction changes seasonally (winter: sunspace→back, summer: back→sunspace)
   - Magnitude depends on temperature difference and conductance

#### Validation Criteria

```rust
// After annual simulation
assert!(results.annual_heating_mwh >= 1.65 && results.annual_heating_mwh <= 2.45);
assert!(results.annual_cooling_mwh >= 1.55 && results.annual_cooling_mwh <= 2.78);

// Verify HVAC energy is only in back-zone
assert_eq!(results.hvac_energy_per_zone[0], results.total_hvac_energy);
assert_eq!(results.hvac_energy_per_zone[1], 0.0);

// Verify temperature coupling
let mean_t_back = results.zone_temperatures[0].iter().sum::<f64>() / 8760.0;
let mean_t_sun = results.zone_temperatures[1].iter().sum::<f64>() / 8760.0;
assert!((mean_t_sun - mean_t_back).abs() > 2.0, "Sunspace should be significantly different");
```

### Unit Tests

```rust
#[test]
fn test_inter_zone_conductance_calculation() {
    // Test that conductance is correctly calculated from U-value and area
    let u_value = 0.514;  // W/m²K (typical low-mass wall)
    let area = 21.6;      // m² (Case 960 common wall)
    let expected_h_iz = u_value * area;

    let spec = ASHRAE140Case::Case960.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    let h_iz = model.h_tr_iz.as_ref();
    assert!((h_iz[0] - expected_h_iz).abs() < 1.0, "Conductance mismatch");
}

#[test]
fn test_sky_view_factor_calculation() {
    // Test sky view factor for different orientations
    use crate::sim::sky_radiation::SkyRadiation;

    let roof = SkyRadiation::from_tilt(0.0, 0.9);
    assert!((roof.sky_view_factor - 1.0).abs() < 1e-6);

    let wall = SkyRadiation::from_tilt(90.0, 0.9);
    assert!((wall.sky_view_factor - 0.5).abs() < 1e-6);
}
```

---

## Limitations and Assumptions

### Current Limitations

1. **Linearized Radiation**:
   - Longwave radiation is linearized around mean temperature
   - Accurate for small ΔT (< 10K)
   - May introduce errors for large temperature differences

2. **Simplified View Factors**:
   - Interior view factors use area-weighted approximation
   - Exact radiosity/enclosure theory not implemented
   - May not capture complex geometry effects

3. **No Air Mixing**:
   - Air mixing through openings is not explicitly modeled
   - Conductive transfer through surfaces accounts for most effects
   - Future: Add explicit air mixing model

4. **Homogeneous Zone Temperature**:
   - Each zone is modeled as a single well-mixed node
   - No vertical or horizontal temperature gradients within zone
   - Valid for well-mixed spaces, not for large zones

5. **Constant Conductance**:
   - Inter-zone conductance is assumed constant
   - Does not vary with temperature or moisture
   - Future: Consider temperature-dependent properties

### Assumptions

1. **ASHRAE 140 Conformance**:
   - Models follow ASHRAE 140 specifications
   - Idealized constructions and control
   - May not match real building behavior exactly

2. **Steady-State Surface Temperatures**:
   - Surface temperatures computed from zone temperatures
   - No thermal mass coupling between zones
   - Reasonable for hourly timestep

3. **No Latent Heat Transfer**:
   - Only sensible heat transfer considered
   - No moisture effects
   - Future: Add humidity model

### Future Enhancements

1. **Radiosity Method**:
   - Implement full radiosity solver for interior radiation
   - Accurate for arbitrary geometry
   - Computationally more expensive

2. **Air Mixing Model**:
   - Explicit modeling of air flow through openings
   - Pressure-driven infiltration
   - Stack effect modeling

3. **Multi-Surface Radiation**:
   - Separate radiative exchange for each surface pair
   - Window-to-wall, wall-to-wall, etc.
   - Improved accuracy for complex geometries

4. **Temperature-Dependent Properties**:
   - Vary conductance with temperature
   - Moisture-dependent properties
   - Improved accuracy over wide ranges

---

## References

1. **ASHRAE 140** - Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs

2. **ISO 13790** - Energy performance of buildings — Calculation of energy use for space heating and cooling

3. **Incropera & DeWitt** - Fundamentals of Heat and Mass Transfer (Chapter 13: Radiation Exchange Between Surfaces)

4. **ASHRAE Handbook of Fundamentals** - Chapter 15: Fenestration, Chapter 25: Heat and Mass Transfer

5. **Modera, H.** - Review of Inter-Zone Air Flow in Buildings (ASHRAE Transactions, 1991)

---

## Appendix: Thermal Network Diagrams

### Two-Zone Network

```
          T_ext (Outdoor)
            │
            h_ext
            │
    ┌───────┴───────┐
    │               │
  T_0           T_1
  Zone 0        Zone 1
    │               │
    └───────┬───────┘
            │
            h_iz (inter-zone)
            │
           Q_iz
```

### Heat Flow Paths in Case 960

```
                    Solar Gain
                        ↓
                   ┌─────────┐
                   │ Sunspace│ Zone 1
                   │   (T1)  │ (free-floating)
                   └────┬────┘
                        │ h_iz (common wall)
                        ↓
                   ┌─────────┐
                   │ Back-Zone│ Zone 0
                   │   (T0)  │ (conditioned)
                   └────┬────┘
                        │
                    ┌───┴───┐
                    │       │
                  HVAC     External
                 (Q_HVAC)   (Q_ext)
```

### Radiative Exchange

```
            Sky
             ↑
             │ F_sky
             │
    ┌────────┴────────┐
    │                 │
  Surface 0        Surface 1
    │                 │
    └────────┬────────┘
             │
             ↓
           Ground
```

---

*Document Version: 1.0*
*Last Updated: 2025-02-27*
*Author: Fluxion Development Team*
