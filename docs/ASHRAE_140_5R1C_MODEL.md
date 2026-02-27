# ASHRAE 140 5R1C Thermal Network Model

This document describes the 5R1C thermal network model implemented in Fluxion for ASHRAE 140 validation.

## Overview

The 5R1C model is a simplified thermal network with 5 resistances (R) and 1 capacitance (C):

- **R1 (h_tr_w)**: Window/conduction conductance
- **R2 (h_ve)**: Ventilation conductance  
- **R3 (h_tr_em)**: Exterior-to-mass conductance
- **R4 (h_tr_ms)**: Mass-to-surface conductance
- **R5 (h_tr_is)**: Surface-to-air conductance (interior surface conductance)
- **C1 (Cm)**: Combined thermal mass

## Thermal Resistance Network

```
                    ┌─────────────┐
     Solar ────────▶│             │
     (q_sol)        │   Surface   │◀──── h_tr_is ────▶ Indoor Air (T_i)
                    │  Temperature │
                    │    (T_s)    │
                    └──────┬──────┘
                           │ h_tr_ms
                    ┌──────▼──────┐
                    │             │
                    │    Mass     │
                    │Temperature  │
                    │   (T_m)     │
                    │             │
                    └──────┬──────┘
                           │ h_tr_em
                    ┌──────▼──────┐
                    │             │
         Outdoor ──▶│  Exterior  │◀──── h_tr_w, h_ve
        (T_o)       │   Surface  │
                    │  (T_ext)   │
                    │             │
                    └─────────────┘
```

## Key Conductance Definitions

### Derived Parameters (Cached)

```rust
derived_term_rest_1 = h_tr_ms + h_tr_is
derived_h_ms_is_prod = h_tr_ms * h_tr_is
derived_ground_coeff = (h_tr_ms + h_tr_is) * h_tr_floor
```

### Sensitivity Tensor Formula

The sensitivity tensor defines how much 1 Watt of HVAC power changes the indoor temperature:

```
sensitivity = (h_tr_ms + h_tr_is) / [h_ms_is + (h_tr_ms + h_tr_is) * (h_tr_w + h_ve + h_tr_floor)]
```

In code terms:
```rust
derived_den = derived_h_ms_is_prod 
            + derived_term_rest_1 * h_ext 
            + derived_ground_coeff

sensitivity = derived_term_rest_1 / derived_den
```

Where:
- `h_ext` = h_tr_w + h_ve (total external conductance)
- `h_ms_is` = h_tr_ms * h_tr_is (mass-surface to air product)

## Heat Balance Equations

### Surface Temperature
```
T_s = (h_tr_ms * T_m + h_tr_is * T_i + q_sol) / (h_tr_ms + h_tr_is)
```

### Mass Temperature
```
T_m = (h_tr_ms * T_s + q_m) / (h_tr_ms + h_tr_em + h_ve)
```

### Indoor Temperature (with HVAC)
```
T_i = T_free + sensitivity * Q_hvac
```

Where:
- `T_free` = Free-floating temperature (without HVAC)
- `Q_hvac` = HVAC heating/cooling power
- `sensitivity` = Temperature change per Watt

## Implementation Details

### Cached Parameters (update_derived_parameters)

Located in `src/sim/engine.rs` around line 1048:

```rust
self.derived_term_rest_1 = self.h_tr_ms.clone() + self.h_tr_is.clone();
self.derived_h_ms_is_prod = self.h_tr_ms.clone() * self.h_tr_is.clone();
self.derived_ground_coeff = self.derived_term_rest_1.clone() * self.h_tr_floor.clone();
self.derived_den = self.derived_h_ms_is_prod.clone()
    + self.derived_term_rest_1.clone() * self.derived_h_ext.clone()
    + self.derived_ground_coeff.clone();
self.derived_sensitivity = self.derived_term_rest_1.clone() / self.derived_den.clone();
```

### Dynamic Recalculation (step_physics)

For systems with variable infiltration/ventilation (night ventilation, etc.), sensitivity is recalculated at each timestep to maintain accuracy:

```rust
let den_val = self.derived_h_ms_is_prod.clone() 
    + term_rest_1.clone() * h_ext.clone() 
    + self.derived_ground_coeff.clone();
let sens_val = term_rest_1.clone() / den_val.clone();
```

This was implemented to fix Issue #366 where cached sensitivity became stale when ventilation rates changed.

## HVAC Power Demand Calculation

The HVAC power demand is calculated using:

```rust
let hvac_demand = self.hvac_power_demand(hour_of_day_idx, &t_i_free, &sensitivity_val);
```

The sensitivity tensor controls how HVAC power affects indoor temperature. If sensitivity is too small, HVAC demand becomes too large.

## Known Issues

### ASHRAE 140 Regression (Issue #340)

Historical issue where HVAC energy was inflated 300-500%. Fixed in PR #359 by:
- Correcting sensitivity denominator to include ground coupling term
- Implementing dynamic sensitivity recalculation at each timestep

### 5R1C Denominator Inconsistency (Issue #366)

Fixed by ensuring sensitivity calculation in `step_physics` uses the same denominator as cached values in `update_derived_parameters`.

## References

- ASHRAE Handbook - Fundamentals, Chapter 19: Residential Cooling and Heating Load Calculations
- ISO 13790:2008 - Energy performance of buildings - Calculation of energy use for space heating and cooling
- TRNSYS 16 - Volume 4: Mathematical Reference
