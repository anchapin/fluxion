# Issue #872: Fix HVAC Energy Formula for 9R4C Model

## Status: Planning

## Problem

Case 900 (high-mass building) produces grossly incorrect HVAC energy results:

| Metric | Current | Reference | Error |
|--------|---------|-----------|-------|
| Annual heating | 6.88 MWh | 1.17–2.04 MWh | 3.4× too high |
| Annual cooling | 0.01 MWh | 2.13–3.67 MWh | near zero |
| Peak cooling | 0.11 kW | 1.50–3.50 kW | near zero |
| Max zone temp | 28.01°C | 36–46°C | too low |

These are **pre-existing on `main`** — not caused by the sensitivity removal (#872 partial).

## Root Cause (BEM Analysis)

Two cascading bugs:

### Bug 1: Wrong HVAC Energy Formula

`hvac_demand_from_ideal_loads()` in `ideal_loads.rs` computes:
```
Q = mass_flow × cp × (T_zone - T_supply)
```

This is a **ventilation airflow capacity** formula, not an ideal loads formula.

For Case 900:
- `mass_flow` = ρ × cp × (ACH/3600) × V = **21.7 W/K**
- Building envelope heat loss = `derived_h_ext + h_tr_floor` ≈ **119 W/K**
- HVAC capacity is **5.5× undersized** relative to envelope losses

The formula literally cannot deliver enough heating/cooling to maintain setpoints.

### Bug 2: Temperature Feedback Death Spiral

The temperature update:
```
t_i_act = t_i_free + hvac_power / h_tr_is
```

Since `hvac_power` is too small (Bug 1), the zone temperature never reaches setpoint:
- Winter: t_i_free=15°C, hvac=543W, h_tr_is=500 W/K → t_i_act=16.1°C (4°C below 20°C setpoint)
- The building perpetually undershoots, accumulating excessive heating energy
- Cooling rarely triggers because the zone never gets hot enough

### Self-Consistency Principle

The energy formula and temperature update MUST use the same physics:
```
Q = H × (T_set - T_free)      ← energy formula
t_act = T_free + Q / H         ← temperature update
⇒ t_act = T_set                ← exact control
```

The current code breaks this: Q uses `mass_flow × cp`, temperature divides by `h_tr_is` — two different conductances.

## Solution: Replace ideal_loads with h_loss formula

### What changes

In `step_physics_9r4c` only, replace:
```rust
let hvac = hvac_demand_from_ideal_loads(t_i_free, heating_sp, cooling_sp);
```
With:
```rust
let hvac = h_loss_formula(t_i_free, heating_sp, cooling_sp, h_ext, h_floor, h_tr_is);
```

Where `h_loss_formula` computes:
```
H = h_tr_is  (NOT h_ext + h_floor)
Q = H × (T_set - T_free)
```

Using `h_tr_is` as the coefficient ensures self-consistency with the temperature update.

### Why h_tr_is (not h_ext)?

The temperature update is: `t_i_act = t_i_free + Q / h_tr_is`

For `t_i_act = T_setpoint` exactly: `Q = h_tr_is × (T_set - T_free)`

This is the ISO 13790 5R1C zone air energy balance. The HVAC injects heat into the air node; the air node's thermal conductance to the rest of the building is `h_tr_is`.

### Files affected

1. `src/sim/thermal_model_physics.rs` — step_physics_9r4c (~line 2530)
2. No changes to `ideal_loads.rs` — other paths (5R1C, 6R2C) continue using it

### Approach: Inline the formula

Rather than creating a new function, inline the h_loss formula directly in step_physics_9r4c:

```rust
let h_tr_is_ref = self.0.h_tr_is.as_ref();
let mut hvac_data = Vec::with_capacity(num_zones);
for i in 0..num_zones {
    let t_free = t_i_free.as_ref()[i];
    let h_is = h_tr_is_ref[i];
    let q = if t_free < heating_setpoint {
        h_is * (heating_setpoint - t_free)
    } else if t_free > cooling_setpoint {
        h_is * (cooling_setpoint - t_free)  // negative
    } else {
        0.0
    };
    hvac_data.push(q.clamp(-cool_cap, heat_cap));
}
let hvac = T::from(VectorField::new(hvac_data));
```

This replaces BOTH `hvac_for_temp_calc` and `hvac_for_energy` — using the same formula for both ensures self-consistency.

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| 5R1C path regression | Don't touch step_physics_5r1c — it already passes Cases 600/610/620/650 |
| Cooling overshoot | Clamp to hvac_cooling_capacity |
| Free-floating regression | Early return before HVAC section — no change to FF path |
| Energy vs temperature inconsistency | Same formula for both — self-consistent by construction |

## Pre-existing Issues (not fixed by this plan)

1. **Multi-node solver mass feedback** (#874) — conductance-weighted surface temps cause FF regressions
2. **Per-node gain injection** (#873) — step_with_gains causes FF regressions
3. **Annual heating still 2.4× reference** — even with h_loss formula, heating was 4.89 MWh in earlier tests (target 1.17–2.04). The `h_tr_is` coefficient may be too large for Case 900. May need `h_tr_is × h_tr_em / (h_tr_is + h_tr_em)` (series combination) instead.

## Success Criteria

- [ ] Case 900 annual cooling within reference range [2.13, 3.67] MWh
- [ ] Case 900 peak cooling within reference range [1.50, 3.50] kW
- [ ] Case 900 max zone temp within reasonable range (≥36°C)
- [ ] Free-floating tests still pass (13/13)
- [ ] Lib tests still pass (2457)
- [ ] Cases 600/610/620/650 unchanged
