# Mathematical Model Documentation

## Overview

Fluxion implements the ISO 13790:2008 5R1C (five-resistance, one-capacitance) thermal network model as its primary heat conduction method for ASHRAE 140 compliance. For multi-zone buildings, a 6R2C (six-resistance, two-capacitance) extension provides inter-zone coupling.

---

## 1. 5R1C Thermal Network Model (ISO 13790)

### 1.1 Network Topology

The 5R1C model represents the building thermal zone as a lumped-parameter electrical analogy:

```
T_supply ───[R_is]──┬──[R_ms]──┬──[R_em]──┬──[R_em]── T_ext
                      │          │          │
                     [C_m]       │          │
                      │          │          │
T_air   ─────────────┘    T_m   │    T_surface
                                        │
                                   [R_opaque]
                                        │
                                    T_ground
```

The five resistances are:

| Resistance | Symbol | Description | Path |
|---|---|---|---|
| Internal surface | `R_is` (H_tr_is) | Convective + radiative from supply air to internal surface | T_supply → internal node |
| Thermal mass coupling | `R_ms` (H_tr_ms) | Conductance between internal surface and thermal mass node | Internal node → T_m |
| External surface, mass side | `R_em` (H_tr_em) | From thermal mass node to external surface | T_m → T_surface |
| External surface, ambient | Part of `R_em` | External surface convective resistance | T_surface → T_ext |
| Opaque/ground | `R_opaque` | Ground-coupled opaque surface resistance | T_surface → T_ground |

The single capacitance `C_m` represents the building thermal mass connected at the mass node `T_m`.

### 1.2 Heat Balance Equations

The zone heat balance at each timestep (default 1 hour) solves:

**Mass node (dynamic):**
```
C_m × dT_m/dt = Φ_m + H_tr_ms × (T_air - T_m) + H_tr_em × (T_ext - T_m)
```

**Air node (quasi-steady):**
```
0 = Φ_air + H_tr_is × (T_supply - T_air) + H_tr_ms × (T_air - T_m) + Q_hvac
```

Where:
- `Φ_m` = solar gains + internal gains allocated to mass node
- `Φ_air` = solar gains + internal gains allocated to air node
- `Q_hvac` = HVAC ideal loads (heating positive, cooling negative)
- `H_tr_*` = conductance (reciprocal of resistance)

### 1.3 Gain Allocation

Per ISO 13790 Section 7, solar and internal heat gains are split between the air and mass nodes:

- **Air node fraction**: Direct solar through glazing + ventilation heat gains
- **Mass node fraction**: Absorbed solar on opaque surfaces + internal radiant gains

### 1.4 Implementation

Source: `src/physics/five_r1c_solver.rs`

```rust
pub struct FiveR1CSolver { /* internal state */ }

impl HeatConductionSolver for FiveR1CSolver {
    fn name(&self) -> &str { "5R1C" }
    fn step(&mut self, ...) -> HeatConductionResult { ... }
    fn steady_state_flux(&self, T_int: f64, T_ext: f64) -> f64 { ... }
}
```

The solver is registered in the solver registry (`src/physics/solver_registry.rs`) and selected as the default for single-zone ASHRAE 140 cases.

---

## 2. 6R2C Multi-Zone Extension

### 2.1 Extension Rationale

The 5R1C model treats each zone as thermally independent. The 6R2C extension adds:

- A **sixth resistance** (`R_interzone`) connecting adjacent zone mass nodes through internal walls/partitions
- A **second capacitance** (`C_wall`) representing the internal partition thermal mass

### 2.2 Network Topology

```
Zone A                                Zone B
T_m,A ───[R_ms,A]──[R_is,A]── T_air,A    T_m,B ───[R_ms,B]──[R_is,B]── T_air,B
  │                                      │
  └────────────[R_interzone]──[C_wall]───┘
```

### 2.3 Coupled Equations

For N coupled zones, the system becomes a coupled ODE:

```
C_A × dT_m,A/dt = Φ_A + H_ms,A × (T_air,A - T_m,A) + H_inter × (T_m,B - T_m,A)
C_B × dT_m,B/dt = Φ_B + H_ms,B × (T_air,B - T_m,B) + H_inter × (T_m,A - T_m,B)
C_wall × dT_wall/dt = H_inter × (T_m,A - T_wall) + H_inter × (T_m,B - T_wall)
```

### 2.4 Implementation

Source: `src/thermal/coupled_solver.rs`, `src/thermal/zone_coupling.rs`, `src/thermal/inter_zone.rs`

The multi-zone solver uses implicit time-stepping for stability with the `multi-zone` feature flag.

---

## 3. Solar Gain Calculation

### 3.1 Solar Position

Solar geometry is computed from:
- **Latitude**: 39.83°N (Denver)
- **Longitude**: 104.65°W (Denver)
- **Hour angle**: Derived from solar time
- **Declination**: Cooper's equation

### 3.2 Incident Solar on Surfaces

For each opaque and transparent surface:

```
I_incident = I_beam × cos(θ_i) + I_diffuse × F_sky + I_ground_reflected
```

Where:
- `θ_i` = angle of incidence on the surface
- `F_sky` = sky view factor for the surface tilt
- `I_ground_reflected` = ground-reflected diffuse using ground albedo (default 0.2)

### 3.3 Window Solar Transmittance

Glazing transmittance varies with incident angle:

```
τ(θ_i) = τ_normal × f(angle_correction)
```

Per ASHRAE 140, the base transmittance for low-mass cases (600-series) is 0.789 (double-pane clear) and 0.568 for high-mass cases (900-series).

### 3.4 Implementation

Source: `src/sim/solar.rs`

---

## 4. Ground Temperature Model

### 4.1 Constant Ground Temperature (ASHRAE 140 Default)

Per ASHRAE 140 specification, the ground temperature is a constant 10°C for all baseline test cases:

```
T_ground = 10°C  (constant)
```

### 4.2 Dynamic Ground Temperature (Kusuda-Achenbach)

For non-ASHRAE simulations, the Kusuda-Achenbach model provides seasonal variation:

```
T_ground(t, d) = T_mean - A_surface × exp(-d × √(π / (365 × α)))
                 × cos(2π/365 × (t - t_shift) - d × √(π / (365 × α)))
```

Where:
- `T_mean` = mean annual ground surface temperature
- `A_surface` = amplitude of annual surface temperature variation
- `d` = depth below surface (m)
- `α` = soil thermal diffusivity (m²/day)
- `t_shift` = phase shift (day of minimum surface temperature)

### 4.3 Implementation

Source: `src/sim/boundary.rs`

```rust
pub trait GroundTemperature: Send + Sync {
    fn ground_temperature(&self, hour_of_year: usize) -> f64;
}

pub struct ConstantGroundTemperature { /* temperature: f64 */ }
pub struct DynamicGroundTemperature { /* Kusuda parameters */ }
```

---

## 5. Shading Model (Overhang + Fin)

### 5.1 Geometric Shading

Fluxion computes window shading from external devices using geometric projection:

**Horizontal Overhang** (`src/sim/shading.rs::Overhang`):
- Projects a horizontal shadow strip across the full window width
- Shadow depth depends on solar altitude angle and overhang projection depth

**Vertical Fin** (`src/sim/shading.rs::ShadeFin`):
- Projects a vertical shadow strip across the full window height
- Shadow width depends on solar azimuth angle relative to surface normal and fin projection depth

### 5.2 Combined Shading (Inclusion-Exclusion)

When both overhang and fin are present, their shadow regions may overlap at the window corner. The total shaded fraction uses the inclusion-exclusion principle:

```
shaded_fraction = overhang_shadow + fin_shadow - overlap

where:
  overlap = shaded_height × shaded_width  (rectangular intersection)
```

### 5.3 Solar Position Input

```rust
pub struct LocalSolarPosition {
    pub altitude: f64,   // Solar altitude angle (radians)
    pub azimuth: f64,    // Solar azimuth relative to surface normal (radians)
}
```

### 5.4 ASHRAE 140 Test Cases Using Shading

| Case | Overhang | Fins | Description |
|---|---|---|---|
| 610 | Yes (south) | No | South-facing shading |
| 630 | Yes (E/W) | Yes (E/W) | East/west overhang + fin shading |
| 910 | Yes (south) | No | High-mass south shading |
| 930 | Yes (E/W) | Yes (E/W) | High-mass E/W shading |

### 5.5 Implementation

Source: `src/sim/shading.rs`

```rust
pub fn calculate_shaded_fraction(
    overhang: Option<&Overhang>,
    fin: Option<&ShadeFin>,
    window_width: f64,
    window_height: f64,
    solar_pos: &LocalSolarPosition,
) -> f64  // Returns 0.0 (fully unshaded) to 1.0 (fully shaded)
```

---

## 6. HVAC Ideal Loads Model

### 6.1 Concept

The Ideal Loads model follows EnergyPlus "Ideal Loads Air System" terminology:

- **100% efficiency** — no equipment losses
- **Infinite capacity** — always meets setpoint if needed
- **Sensible + latent** — calculates both sensible and latent components
- **No duct/duct losses** — direct delivery to zone

### 6.2 Calculation

At each timestep, the ideal loads system computes the thermal energy required to maintain the zone at the heating or cooling setpoint:

```
Q_ideal = C_zone × (T_setpoint - T_zone_predicted)  [W]

If Q_ideal > 0:  Heating mode, Q_heating = Q_ideal
If Q_ideal < 0:  Cooling mode, Q_cooling = |Q_ideal|
If free-floating: Q_ideal = 0 (no HVAC)
```

### 6.3 Setpoints

| Parameter | Value | Notes |
|---|---|---|
| Heating setpoint | 20°C | Per ASHRAE 140 |
| Cooling setpoint | 27°C | Per ASHRAE 140 |
| Setback (Case 640) | 10°C (night) | Thermostat setback test |
| Deadband | 0°C (ideal) | No throttling range for ideal loads |

### 6.4 Electrical Conversion (Non-Ideal)

For non-ideal equipment, `SimpleHVACEquipment` converts thermal load to electrical power:

```
P_electrical = Q_thermal / COP    (cooling)
P_electrical = Q_thermal / η      (heating)
```

Default values (ASHRAE 140):
- Cooling COP: 3.0 (typical heat pump)
- Heating efficiency: 0.9 (electric resistance/furnace)

### 6.5 Implementation

Source: `src/sim/hvac/ideal_loads.rs`

```rust
pub struct ZoneIdealLoads { /* setpoints and state */ }
pub struct HVACEnergyResult { /* thermal_load_watts, electrical_kw, mode */ }

impl ZoneIdealLoads {
    pub fn calculate_sensible_cooling_load(...) -> f64;
    pub fn calculate_sensible_heating_load(...) -> f64;
}
```

---

## 7. Numerical Integration

### 7.1 Time Stepping

Default timestep: **1 hour** (3600 seconds) for ASHRAE 140 annual simulations.

An adaptive timestep module (`src/sim/adaptive_timestep.rs`) is available for simulations requiring higher temporal resolution near rapid transients.

### 7.2 Warmup Period

Per ISO 13790, a warmup period runs until the zone temperature converges between consecutive years. Default warmup tolerance and maximum iterations are configurable.

### 7.3 Annual Simulation

The annual simulation iterates over 8760 hours (non-leap year), accumulating:
- Hourly heating/cooling energy (J → MWh)
- Peak heating/cooling loads (W → kW) with timestamps
- Free-floating zone temperatures (min, max, mean)

---

## References

1. ISO 13790:2008 — *Energy performance of buildings — Calculation of energy use for space heating and cooling*, Sections 7.1–7.4
2. ASHRAE Standard 140-2023 — *Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs*, Section 7
3. Kusuda, T. & Achenbach, P.R. (1965) — *Earth Temperature and Thermal Diffusivity at Selected Stations in the United States*, ASHRAE Transactions, 71(1)
4. EnergyPlus Engineering Reference — *Ideal Loads Air System* documentation
5. Annex C, ISO 13790:2008 — *Tabulated values for building elements*
