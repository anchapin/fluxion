# Software Qualification Test (SQT) Report

**Product:** fluxion v0.8.0+
**Standard:** ASHRAE Standard 140-2023 (Building Thermal Envelope and Fabric Load Tests)
**Document Type:** Software Qualification Test Report
**Maintained by:** Building Standards Engineer
**Last Updated:** 2026-06-16
**Related Issue:** #750

---

## 1. Purpose and Scope

This Software Qualification Test (SQT) Report documents the numerical methods, timestep selection, convergence criteria, and known limitations of fluxion for the purpose of ASHRAE 140 compliance submission. This report accompanies the fluxion ASHRAE 140 validation suite and is required for formal compliance certification per ASHRAE 140-2023 Section 1.5 and Section 6.

fluxion is a building energy simulation engine written in Rust. It simulates heat transfer through building envelopes using multiple solver backends (CTF, FD, 5R1C RC-network) and is validated against ASHRAE 140-2023 test cases.

---

## 2. Numerical Methods

### 2.1 Heat Conduction Solvers

fluxion implements three heat conduction solver backends, selected per-case based on material time-constant criteria:

#### 2.1.1 Conduction Transfer Function (CTF) Solver
**File:** `src/physics/ctf_solver.rs`

The CTF solver uses pre-computed conduction transfer functions to solve the heat equation through multi-layer building fabric. CTF coefficients are derived from the material properties (thermal conductivity k, density ρ, specific heat Cp) and layer thicknesses.

- **Method:** Convolution of CTF coefficients with historical surface heat flux
- **Derivation:** Analytical solution for each material layer using the Laplace transform method
- **Order:** Variable per construction; determined by material thermal mass
- **Limitations:** Not suitable for materials with highly dissimilar layer time-constants (κ > 20,000 J/m²K criterion — see DEV-003, DEV-009)

#### 2.1.2 Finite Difference (FD) Solver
**File:** `src/physics/fd_solver.rs`

The FD solver discretizes the heat equation spatially using finite differences across material layers and temporally using an implicit scheme (backward Euler).

- **Method:** Implicit finite difference on multi-layer slab geometry
- **Spatial discretization:** Nodal network through layer depth (variable nodes per layer)
- **Temporal discretization:** Backward Euler with adaptive timestep capability
- **Stability:** Unconditionally stable for implicit scheme
- **Limitations:** Computational cost increases with node count; promoted to when CTF criteria are not met (see DEV-003, DEV-009)

#### 2.1.3 5R1C RC-Network Solver
**File:** `src/physics/solver_trait.rs`

The 5R1C solver models the building zone as an equivalent electrical circuit with five resistances and one capacitance. This is the simplest model and is used for calibration and rapid prototyping.

- **Method:** Explicit forward-Euler integration of RC-network ODEs
- **Order:** First-order accurate
- **Timestep constraint:** Stability limited by smallest RC time constant
- **Use case:** Preliminary analysis and ML surrogate baseline

### 2.2 Solar Radiation

**File:** `src/sim/solar.rs`

Solar radiation is computed using:
- **Solar position:** Based on astronomical algorithms (latitude, day of year, hour)
- **Direct normal irradiance (DNI):** From weather file (TMY format)
- **Diffuse horizontal irradiance (DHI):** From weather file
- **Beam tilt factor:** Geometric calculation based on surface orientation and tilt
- **Shading:** Overhang and fin geometric shadow calculations (see DEV-011 for overlap bug)
- **Distribution:** Per ASHRAE 140 §5.2.2 — distributed to interior surfaces by area × absorptance (see DEV-010)

### 2.3 Ventilation and Infiltration

**File:** `src/sim/ventilation.rs`

Ventilation is modeled as a schedule-driven mass flow rate into the zone:
- **Method:** Simple volumetric air change rate (ACH) or design airflow rate (L/s)
- **Schedule:** Constant, scheduled, or weather-dependent operation
- **Heat content:** Outdoor dry-bulb temperature from weather file

### 2.4 Zone Energy Balance

**File:** `src/sim/thermal_model.rs`

The zone energy balance solves for zone air temperature given:
- Conduction gains/losses through opaque surfaces (CTF/FD solvers)
- Solar gains transmitted through windows
- Internal gains (occupancy, equipment, lighting — per case definition)
- Ventilation heat addition/removal
- HVAC loads (heating/cooling setpoint control)

---

## 3. Timestep

### 3.1 Annual Simulation Timestep

The standard timestep for annual ASHRAE 140 simulations is **1 hour (3600 seconds)**.

This is consistent with:
- Weather data temporal resolution (hourly TMY data)
- ASHRAE 140 output requirements (annual and peak monthly/daily values)
- EnergyPlus and other reference program conventions

### 3.2 Sub-Hourly Timestep (Internal)

Internally, the FD solver uses a **variable sub-timestep** to maintain numerical stability and accuracy:

- **Minimum sub-timestep:** 60 seconds (1 minute)
- **Maximum sub-timestep:** 3600 seconds (1 hour)
- **Adaptation criterion:** Based on rate of change of node temperatures; triggers subdivision when `dT/dt` exceeds threshold

The sub-timestep is transparent to the user — the annual simulation proceeds in 1-hour intervals and the FD solver internally subdivides as needed.

### 3.3 Timestep for Free-Float Cases

Free-float cases (Cases 600FF, 650FF, 900FF, 950FF) also use 1-hour external timesteps. There is no special reduced timestep for these cases. See DEV-001 for the open issue regarding HVAC not being disabled in free-float mode.

---

## 4. Convergence Criteria

### 4.1 Energy Balance Convergence

The zone energy balance iteration converges when:

```
|T_zone_new - T_zone_old| < 0.01 K  (temperature tolerance)
|max(airflow_heat_balance residual)| < 1.0 W  (heat balance residual)
```

Maximum iterations per timestep: **50** (before stepping to next hour)

### 4.2 CTF Solver Convergence

CTF coefficients are pre-computed at simulation initialization. No per-timestep iteration is required — CTF uses convolution with previously computed flux history.

### 4.3 FD Solver Convergence

The FD solver uses a direct tridiagonal matrix solver (Thomas algorithm) per timestep, which converges in one pass. The implicit scheme is unconditionally stable.

### 4.4 Warm-Up Period

ASHRAE 140-2023 Section 1.4 requires steady-periodic conditions. Current fluxion implementation **does not** perform multi-year pre-conditioning — simulations start from cold initial conditions. This introduces bias in 900-series high-mass cases (see DEV-012). The warm-up period requirement is a known compliance gap.

---

## 5. Known Limitations

The following limitations are documented in the Deviations Register (`deviations-register.md`) and must be addressed before formal certification:

| Limitation | Reference | Impact | Status |
|------------|-----------|--------|--------|
| Free-float HVAC not disabled | DEV-001 | Unphysical zone temperatures (~125°C) | OPEN |
| Synthetic weather instead of Annex C TMY | DEV-004 | All 64 validation metrics affected | OPEN |
| No warm-up/pre-conditioning period | DEV-012 | 900-series biased by cold-start | OPEN |
| CTF surface area error (97.2 vs 63.6 m²) | DEV-009 | 900-series heat transfer overstated | SPEC ONLY |
| Surface film coefficients incorrect | DEV-006 | All U-value calculations affected | SPEC ONLY |
| 900-series concrete wrong thermal properties | DEV-002 | CTF coefficients wrong for 900-series | SPEC ONLY |
| Solar distribution uses wrong standard reference | DEV-010 | Cases 610/620/630/910-930 affected | IN PROGRESS |
| Ground floor BC not adiabatic | DEV-016 | Small systematic annual heating error | OPEN |
| Placeholder window conductance values | DEV-015 | Window cases not traceable to Annex B | OPEN |

---

## 6. Compliance Statement

This SQT Report is a living document. Formal ASHRAE 140 certification submission requires:
1. All P1 deviations resolved (DEV-001, DEV-004, DEV-008)
2. All SPEC-ONLY items implemented and verified
3. Passing results on all 64 ASHRAE 140-2023 validation metrics
4. This document signed by a qualified software engineer

Current status: **NOT SUBMISSION-READY** — multiple P1 compliance blockers remain open.

---

## 7. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-06-16 | Building Standards Engineer | Initial version for Issue #750 |

---

*End of SQT Report*
