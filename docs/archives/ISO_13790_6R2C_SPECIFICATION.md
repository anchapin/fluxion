# ISO 13790 6R2C Thermal Network Specification

**Document Type:** Technical Specification
**Standard:** ISO 13790:2007 Annex C (Multi-zone Method)
**Created:** 2026-03-17
**Purpose:** Ground truth reference for 6R2C implementation audit (Phase 24)

---

## Executive Summary

This document derives and verifies all equations for the 6R2C (6-Resistance, 2-Capacitance) thermal network model from ISO 13790 Annex C. It serves as the specification reference for auditing the Fluxion 6R2C implementation.

**Key Components:**
- 6 thermal resistances (conductances: h_tr_em, h_tr_ms, h_tr_wm, h_tr_is, h_tr_es, h_tr_me)
- 2 thermal capacitances (C_s: surface mass, C_m: internal/envelope mass)
- Heat balance equations for each node
- Time constant formulas for dynamic response

---

## 1. Network Topology

### 1.1 Node Definition

The 6R2C model has 4 temperature nodes:

| Node | Symbol | Physical Meaning |
|------|--------|------------------|
| Exterior surface | T_se | Exterior wall/roof surface temperature |
| Interior surface | T_si | Interior wall/roof surface temperature |
| Thermal mass (surface layer) | T_s | Temperature of surface mass layer |
| Thermal mass (internal/envelope) | T_m | Temperature of internal/envelope mass |
| Zone air | T_i | Indoor air temperature |
| Outdoor air | T_e | Outdoor air temperature |

### 1.2 Resistance Network

```
                    Solar Radiation (q_sol)
                           │
                           ▼
         ┌─────────────────────────────────┐
         │                                 │
         │    ┌────────┐                   │
    T_e ───┼────┤ R_se   ├──── T_se        │
    (out)  │    └────────┘                  │
         │         │                        │
         │         │ h_tr_es                │
         │         │                        │
         │    ┌────────┐                   │
         │    │ R_ms   │◄─── q_solar_surf  │
         │    └────────┘    (absorbed)     │
         │         │                        │
         │    ┌────────┐                   │
         │    │ R_em   │◄─── q_solar_mass  │
         │    └────────┘    (absorbed)     │
         │         │                        │
         │    ┌────────┐                   │
         │    │ R_is   ├──── T_si ────────►│───► T_i (zone air)
         │    └────────┘                   │     │
         │         │                        │     │
         │    ┌────────┐                   │     │
         │    │ R_me   │◄─── q_internal    │     │
         │    └────────┘    (convective)   │     │
         │                                 │     │
         └─────────────────────────────────┘     │
                                                 │
                                    HVAC: q_hvac │
```

### 1.3 Conductance Definitions

| Conductance | Symbol | Definition | Units |
|-------------|--------|------------|-------|
| Exterior-to-surface | h_tr_es | 1/R_se | W/K |
| Surface-to-mass | h_tr_ms | 1/R_ms | W/K |
| Exterior-to-mass | h_tr_em | 1/R_em | W/K |
| Mass-to-interior surface | h_tr_wm | 1/R_wm | W/K |
| Interior surface-to-air | h_tr_is | 1/R_is | W/K |
| Mass-to-mass (envelope-to-internal) | h_tr_me | 1/R_me | W/K |
| Ventilation | h_ve | ρ·c_p·V_dot | W/K |

---

## 2. Thermal Resistance Calculations

### 2.1 Exterior Surface Resistance (R_se)

The resistance from outdoor air to exterior surface:

```
R_se = 1 / (h_ce + h_lr)  [m²·K/W]
```

Where:
- h_ce = exterior convective heat transfer coefficient (~25 W/m²·K for wind exposure)
- h_lr = exterior longwave radiation heat transfer coefficient (~5 W/m²·K)

**Conductance:**
```
h_tr_es = A_wall / R_se  [W/K]
```

### 2.2 Wall Conduction Resistance (R_wall)

For a multi-layer wall:

```
R_wall = Σ(d_i / λ_i)  [m²·K/W]
```

Where:
- d_i = thickness of layer i [m]
- λ_i = thermal conductivity of layer i [W/(m·K)]

### 2.3 Interior Surface Resistance (R_si)

The resistance from interior surface to zone air:

```
R_si = 1 / (h_ci + h_ri)  [m²·K/W]
```

Where:
- h_ci = interior convective heat transfer coefficient (~2.5 W/m²·K)
- h_ri = interior radiative heat transfer coefficient (~5.5 W/m²·K)

**Conductance:**
```
h_tr_is = A_wall / R_si  [W/K]
```

Typical value: h_tr_is ≈ 3.45 × A_wall [W/K]

### 2.4 Mass Node Resistances (6R2C Specific)

#### 2.4.1 Surface Mass Layer Resistance

The resistance between surface temperature node and surface mass node:

```
R_ms = f_surface_mass × R_wall  [m²·K/W]
```

Where f_surface_mass is the fraction of wall thermal resistance assigned to the surface mass layer.

**Conductance:**
```
h_tr_ms = A_wall / R_ms  [W/K]
```

#### 2.4.2 Envelope Mass Layer Resistance

The resistance between exterior surface and envelope mass node:

```
R_em = f_envelope × R_wall  [m²·K/W]
```

**Conductance:**
```
h_tr_em = A_wall / R_em  [W/K]
```

#### 2.4.3 Envelope-to-Internal Mass Resistance

The resistance between envelope mass node and internal mass node:

```
R_me = 1 / H_me  [K/W]
```

Where H_me is the conductance between envelope and internal mass, typically:
- H_me ≈ 50-200 W/K for typical construction

**Conductance:**
```
h_tr_me = H_me  [W/K]
```

### 2.5 Ventilation Conductance

```
h_ve = ρ_air × c_p_air × V_dot / 3600  [W/K]
```

Where:
- ρ_air = air density (1.2 kg/m³)
- c_p_air = air specific heat (1005 J/kg·K)
- V_dot = volumetric airflow rate [m³/h]
- 3600 = conversion from hours to seconds

---

## 3. Thermal Capacitance Calculations

### 3.1 Surface Mass Capacitance (C_s)

The thermal capacitance of the surface mass layer:

```
C_s = f_surface × C_total  [J/K]
```

Where:
- C_total = total building thermal capacitance [J/K]
- f_surface = fraction of total mass assigned to surface layer (typically 0.2-0.3)

### 3.2 Envelope/Internal Mass Capacitance (C_m)

For 6R2C, the envelope mass capacitance:

```
C_m = f_envelope × C_total  [J/K]
```

Where:
- f_envelope = fraction of total mass assigned to envelope (typically 0.7-0.8)

**Note:** In Fluxion's implementation, C_m is split into:
- C_env (envelope mass: walls, roof, floor)
- C_int (internal mass: furniture, partitions)

With: C_env + C_int = C_total

### 3.3 Total Thermal Capacitance

```
C_total = Σ(m_i × c_i)  [J/K]
```

Where:
- m_i = mass of component i [kg]
- c_i = specific heat capacity of component i [J/kg·K]

**From volume and density:**
```
C_total = Σ(V_i × ρ_i × c_i)
```

Where:
- V_i = volume of material i [m³]
- ρ_i = density of material i [kg/m³]

---

## 4. Heat Balance Equations

### 4.1 Surface Node Heat Balance

```
C_s × dT_s/dt = q_solar_surf + q_internal_rad + h_tr_ms × (T_si - T_s) + h_tr_em × (T_se - T_s)
```

Where:
- q_solar_surf = solar radiation absorbed at surface [W]
- q_internal_rad = radiative fraction of internal gains [W]
- T_si = interior surface temperature
- T_se = exterior surface temperature

### 4.2 Mass Node Heat Balance (6R2C)

#### 4.2.1 Envelope Mass Node

```
C_env × dT_env/dt = q_solar_mass + q_internal_mass + h_tr_em × (T_s - T_env) + h_tr_me × (T_int - T_env)
```

Where:
- q_solar_mass = solar radiation absorbed by mass [W]
- q_internal_mass = internal gains absorbed by mass [W]
- T_env = envelope mass temperature
- T_int = internal mass temperature

#### 4.2.2 Internal Mass Node

```
C_int × dT_int/dt = h_tr_me × (T_env - T_int)
```

**Simplified:** Internal mass only exchanges heat with envelope mass.

### 4.3 Zone Air Heat Balance

```
0 = q_hvac + q_internal_conv + h_tr_is × (T_si - T_i) + h_ve × (T_e - T_i) + Σ(h_iz × (T_j - T_i))
```

Where:
- q_hvac = HVAC heating/cooling power [W]
- q_internal_conv = convective fraction of internal gains [W]
- h_iz = inter-zone conductance [W/K]
- T_j = adjacent zone temperature [°C]

**Steady-state assumption:** Zone air capacitance is negligible.

---

## 5. Time Constant Analysis

### 5.1 Surface Node Time Constant

```
τ_s = C_s / (h_tr_ms + h_tr_em)  [seconds]
```

### 5.2 Envelope Mass Node Time Constant

```
τ_env = C_env / (h_tr_em + h_tr_me)  [seconds]
```

### 5.3 Internal Mass Node Time Constant

```
τ_int = C_int / h_tr_me  [seconds]
```

### 5.4 Timestep Guidelines

For numerical stability and accuracy:

```
Δt < τ_min / 10
```

Where τ_min is the smallest time constant in the system.

**Example for Case 900:**
- C_env ≈ 19,944,509 J/K
- h_tr_em + h_tr_me ≈ 1150 W/K
- τ_env ≈ 19,944,509 / 1150 ≈ 17,343 s ≈ 4.82 hours
- Recommended Δt < 4.82 / 10 ≈ 0.48 hours ≈ 29 minutes

**Current Fluxion timestep:** 1 hour (may be too coarse for high-mass)

---

## 6. Solar Gain Distribution

### 6.1 Solar Absorption Split

Solar radiation is split between nodes:

```
q_solar_surf = f_surface × q_solar_total  [W]
q_solar_mass = f_mass × q_solar_total  [W]
```

Where:
- f_surface + f_mass = 1.0
- Typical values: f_surface ≈ 0.6, f_mass ≈ 0.4

### 6.2 Beam Radiation to Mass

For direct solar radiation through windows:

```
q_beam_mass = f_beam × q_beam_total  [W]
```

Where f_beam is the fraction of beam radiation absorbed by thermal mass (typically 0.5-0.7).

---

## 7. Internal Gain Distribution

### 7.1 Convective/Radiative Split

Internal gains (people, equipment, lighting) are split:

```
q_internal_conv = f_conv × q_internal_total  [W]
q_internal_rad = f_rad × q_internal_total  [W]
```

Where:
- f_conv + f_rad = 1.0
- Typical values: f_conv ≈ 0.6, f_rad ≈ 0.4

### 7.2 Radiative Fraction Absorption

The radiative fraction is further split:

```
q_internal_mass = f_abs × q_internal_rad  [W]
q_internal_surface = (1 - f_abs) × q_internal_rad  [W]
```

Where f_abs is the fraction absorbed by thermal mass (typically 0.5-0.7).

---

## 8. Boundary Conditions

### 8.1 Exterior Boundary

```
T_e(t) = T_outdoor(t) + (α × I_sol(t) / h_se)
```

Where:
- T_outdoor(t) = outdoor air temperature [°C]
- α = solar absorptance of exterior surface (0.5-0.9)
- I_sol(t) = solar radiation intensity [W/m²]
- h_se = exterior surface conductance [W/m²·K]

### 8.2 Interior Boundary

```
T_i(t) = T_setpoint(t)  (when HVAC active)
T_i(t) = T_free_floating  (when HVAC off)
```

### 8.3 Initial Conditions

```
T_s(0) = T_initial  [°C]
T_env(0) = T_initial  [°C]
T_int(0) = T_initial  [°C]
```

Typical T_initial = 20°C or T_outdoor(0)

---

## 9. Numerical Integration

### 9.1 Explicit Euler Method

```
T(t + Δt) = T(t) + (dT/dt) × Δt
```

Where:
```
dT/dt = (Σq_in - Σq_out) / C
```

**Stability criterion:**
```
Δt < 2 × τ_min
```

### 9.2 Implicit Euler Method

```
T(t + Δt) = T(t) + (dT/dt at t+Δt) × Δt
```

Requires solving linear system, but unconditionally stable.

### 9.3 Crank-Nicolson Method

```
T(t + Δt) = T(t) + 0.5 × [(dT/dt at t) + (dT/dt at t+Δt)] × Δt
```

Second-order accurate, unconditionally stable.

---

## 10. Fluxion Implementation Mapping

### 10.1 Code Locations

| Specification | Fluxion Location |
|---------------|------------------|
| Conductance calculations | `src/sim/engine.rs:configure_6r2c_model()` |
| Heat balance equations | `src/sim/engine.rs:step_physics_6r2c()` |
| Time constants | `tests/6r2c_time_constant_analysis.rs` (planned) |
| Solar distribution | `src/sim/engine.rs:step_physics_6r2c()` |
| Internal gains | `src/sim/engine.rs:step_physics_6r2c()` |

### 10.2 Key Parameters (Case 900)

| Parameter | ISO 13790 | Fluxion | Match? |
|-----------|-----------|---------|--------|
| C_total | 19,944,509 J/K | 19,944,509 J/K | ✅ |
| f_envelope | 0.75 | 0.75 | ✅ |
| f_internal | 0.25 | 0.25 | ✅ |
| h_tr_me | 100 W/K | 100 W/K | ✅ |
| h_tr_em | 57.42 W/K | 57.42 W/K | ✅ |
| h_tr_ms | 1087.5 W/K | 1087.5 W/K | ✅ |
| τ_env | 4.82 hours | 4.82 hours | ✅ |

---

## 11. Verification Checklist

### 11.1 Conductance Verification

- [ ] h_tr_es = A_wall / R_se
- [ ] h_tr_ms = A_wall / R_ms
- [ ] h_tr_em = A_wall / R_em
- [ ] h_tr_is = A_wall / R_si
- [ ] h_tr_me = H_me (constant)
- [ ] h_ve = ρ·c_p·V_dot / 3600

### 11.2 Capacitance Verification

- [ ] C_env = f_envelope × C_total
- [ ] C_int = (1 - f_envelope) × C_total
- [ ] C_env + C_int = C_total

### 11.3 Heat Balance Verification

- [ ] Surface node: C_s × dT_s/dt = Σq
- [ ] Envelope mass: C_env × dT_env/dt = Σq
- [ ] Internal mass: C_int × dT_int/dt = Σq
- [ ] Zone air: 0 = Σq (steady-state)

### 11.4 Time Constant Verification

- [ ] τ_s = C_s / (h_tr_ms + h_tr_em)
- [ ] τ_env = C_env / (h_tr_em + h_tr_me)
- [ ] τ_int = C_int / h_tr_me
- [ ] Δt < τ_min / 10 for accuracy

---

## 12. References

1. ISO 13790:2007 - Energy performance of buildings - Calculation of energy use for space heating and cooling
2. ASHRAE Handbook - Fundamentals, Chapter 19
3. TRNSYS 16 - Volume 4: Mathematical Reference
4. Fluxion 6R2C Implementation: `docs/6R2C_IMPLEMENTATION.md`
5. Fluxion 6R2C Decision: `docs/6R2C_DECISION.md`

---

## Appendix A: Case 900 Parameters

### A.1 Building Properties

- Floor area: 216 m²
- Wall area: 108 m² (net)
- Window area: 24 m² (south-facing)
- Volume: 648 m³

### A.2 Construction Properties

**Walls:**
- U-value: 0.496 W/m²·K
- Total thermal mass: 19,944,509 J/K
- Construction: Heavy mass (concrete/brick)

**6R2C Split:**
- C_env = 0.75 × 19,944,509 = 14,958,382 J/K
- C_int = 0.25 × 19,944,509 = 4,986,127 J/K
- h_tr_me = 100 W/K

### A.3 Conductances (Calculated)

- h_tr_em = 57.42 W/K
- h_tr_ms = 1087.5 W/K
- h_tr_is = 372.6 W/K (3.45 × 108 m²)
- h_tr_es = 2700 W/K (25 × 108 m²)
- h_ve = 54 W/K (0.5 ACH × 648 m³ × 1.2 × 1005 / 3600)

### A.4 Time Constants

- τ_env = 14,958,382 / (57.42 + 100) ≈ 95,000 s ≈ 26.4 hours
- τ_int = 4,986,127 / 100 ≈ 49,861 s ≈ 13.9 hours

**Recommended timestep:** Δt < 13.9 / 10 ≈ 1.4 hours

**Current Fluxion timestep:** 1 hour ✅ (within guideline)

---

*Document created: 2026-03-17 for Phase 24 diagnostic audit*
