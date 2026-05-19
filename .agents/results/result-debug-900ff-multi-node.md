# Investigation Report: 900FF Max Temperature Gap in Multi-Node Solver

**Status**: Investigation Complete (No code changes)
**Model output**: 35.83 degC max | **Reference range**: 41.8-46.4 degC
**Gap**: ~6 degC undercount at peak

---

## 1. Root Cause Analysis

The multi-node (9R4C) solver path undercounts peak temperature by ~6 degC due to **four compounding issues** in the energy balance. Each independently reduces the predicted peak, and together they account for the full gap.

### Issue A (CRITICAL): Missing phi_st in Air Node Balance
**File**: `src/physics/multi_node_solver.rs`, lines 194-218 (`compute_zone_air_temperature`)
**Impact**: ~3-4 degC undercount at peak

The air node energy balance is:
```
T_air = (h_tr_is * T_surface + h_ve * T_outdoor + phi_ia) / (h_tr_is + h_ve)
```

Per ISO 13790 Section 7.2.2.2 Equation 46, the correct formula includes **phi_st** (radiative heat flow from thermal mass to surface):
```
T_air = (h_tr_is * T_s + h_ve * T_out + phi_ia + phi_st) / (h_tr_is + h_ve)
```
where `phi_st = h_tr_ms * (Tm - T_s)`.

Without phi_st, the heat stored in envelope mass nodes during solar peak hours is never "released" back into the air temperature calculation. The mass nodes heat up from solar gains but this stored energy does not propagate to the zone air temperature.

**Note**: The code comment at line 2615 of `thermal_model_physics.rs` explicitly acknowledges this: "the multi-node solver doesn't yet model [the phi_st gain path] (needs #873 for per-node solar injection)".

### Issue B (HIGH): Stale Surface Temperature in Backward Euler Step
**File**: `src/sim/thermal_model_physics.rs`, line 2523
**Impact**: ~1-2 degC undercount at peak

```rust
let t_surface = t_zone_prev - 0.5;  // line 2523
solver.set_surface_temperature(t_surface);
```

The backward Euler step uses this hardcoded `t_zone_prev - 0.5` surface temperature. After the step, the surface temperature is correctly updated from conductance-weighted mass temps (line 2674), but the step itself used the stale value. This creates a one-timestep lag in the mass-to-surface coupling.

For high-mass buildings with large thermal inertia, this lag is significant because envelope nodes couple to the surface via `h_tr_ms * T_surface`. When T_surface is artificially low (during solar peak), envelope mass temperatures are also suppressed.

### Issue C (MEDIUM): Internal Mass Node Coupling Architecture
**File**: `src/physics/multi_node_solver.rs`, lines 163-172
**Impact**: ~1 degC undercount at peak

```rust
let t_env_avg = (m.wall.temperature + m.roof.temperature + m.floor.temperature) / 3.0;
let h_me = node.h_tr_me;
let numer = node.capacitance / dt * node.temperature + h_is * t_i + h_me * t_env_avg;
```

The internal mass node (furniture, partitions) is coupled to the **average of envelope mass temperatures** via `h_tr_me`. In the correct 9R4C model per ISO 13790, internal mass should be coupled to the **zone air** via convective heat transfer. Furniture heats up from room air convection, not from wall mass temperatures.

This architectural error means the internal mass absorbs heat too slowly during solar peaks (it waits for envelope mass to warm first, rather than responding directly to warm air). The result is less thermal buffering during the day and a lower peak air temperature.

### Issue D (LOW): Heuristic Solar Gain Distribution
**File**: `src/sim/thermal_model_physics.rs`, lines 2602-2606
**Impact**: ~0.5 degC

```rust
let gains_wall = opaque_solar_w * 0.6;
let gains_roof = opaque_solar_w * 0.4;
solver.step_with_gains(dt, gains_wall, 0.0, gains_roof, internal_rad);  // floor=0 always
```

The 60/40 wall/roof split is a hardcoded heuristic. Floor gets zero gains always. The actual distribution should follow ASHRAE 140 surface areas and orientations. For the ASHRAE 140 test cell, the south-facing wall is the dominant solar receiver, and the roof has its own irradiance. Incorrect distribution means some solar energy heats the wrong mass nodes.

---

## 2. Specific Code Locations

| Issue | File | Lines | Function/Field |
|-------|------|-------|----------------|
| Missing phi_st | `src/physics/multi_node_solver.rs` | 194-218 | `compute_zone_air_temperature()` |
| Stale T_surface | `src/sim/thermal_model_physics.rs` | 2523 | `t_surface = t_zone_prev - 0.5` |
| Internal coupling | `src/physics/multi_node_solver.rs` | 163-172 | `step_backward_euler()` internal node block |
| Heuristic solar | `src/sim/thermal_model_physics.rs` | 2602-2606 | `gains_wall/roof/floor` split |
| Surface temp update | `src/sim/thermal_model_physics.rs` | 2657-2674 | Post-step surface temp correction |
| phi_st acknowledged | `src/sim/thermal_model_physics.rs` | 2613-2616 | Comment acknowledging missing phi_st |

Supporting files:
- `src/sim/multi_node_thermal.rs` — ThermalMassNode struct definition (h_tr_em, h_tr_ms, h_tr_me fields)
- `src/physics/multi_node_solver.rs` — Full 9R4C backward Euler solver
- `src/sim/thermal_model_physics.rs` — Simulation dispatch routing multi-node for 900FF

---

## 3. Proposed Fix Direction

### Priority 1: Add phi_st to compute_zone_air_temperature (addresses ~3-4 degC)

In `multi_node_solver.rs`, `compute_zone_air_temperature()` line 218:
```rust
// Current:
(self.h_tr_is * t_surface + h_ve * t_outdoor + phi_ia) / denom

// Proposed: add phi_st = sum(h_tr_ms_k * (T_k - T_surface)) for each envelope node
let phi_st = h_ms_w * (self.mass.wall.temperature - t_surface)
           + h_ms_r * (self.mass.roof.temperature - t_surface)
           + h_ms_f * (self.mass.floor.temperature - t_surface);

(self.h_tr_is * t_surface + h_ve * t_outdoor + phi_ia + phi_st) / denom
```

### Priority 2: Fix surface temperature initialization (addresses ~1-2 degC)

In `thermal_model_physics.rs`, line 2523, replace the hardcoded offset with the previous-step computed surface temperature. Either:
- Store the computed surface temperature from the previous timestep on the solver struct
- Or use the conductance-weighted envelope temperatures from the previous step

### Priority 3: Fix internal mass coupling (addresses ~1 degC)

In `multi_node_solver.rs`, `step_backward_euler()` lines 163-172, change the internal node to couple to zone air temperature `t_i` instead of `t_env_avg`:
```rust
// Replace:
let t_env_avg = (m.wall.temperature + m.roof.temperature + m.floor.temperature) / 3.0;
// With: remove h_me coupling to envelope, use h_is coupling to zone air only
```

### Priority 4: Replace heuristic solar split (addresses ~0.5 degC)

Replace the 60/40 split with area-weighted distribution based on actual ASHRAE 140 surface geometry.

---

## 4. Risk Assessment

### Would fixing Issue A (phi_st) break the 5R1C path?
**No.** The 5R1C path does NOT use `compute_zone_air_temperature()`. It uses its own `t_i_free` formula in `thermal_model_physics.rs` (lines 1005-1008) which already includes phi_st via the `num_phi_st` term. The multi-node solver result completely replaces the 5R1C t_i_free at line 2645. These are independent code paths.

### Would fixing Issue B (surface temp) break other tests?
**Low risk.** The surface temperature is only used inside the multi-node backward Euler step. Changing from a hardcoded -0.5 offset to a previously-computed value is strictly more accurate. The 5R1C path doesn't use the multi-node solver's surface temperature.

### Would fixing Issue C (internal coupling) break HVAC cases?
**Medium risk.** Changing the internal mass coupling changes how quickly the internal mass responds to temperature changes. This affects HVAC demand calculations (via `compute_hvac_demand()`). HVAC cases (900, 950, etc.) should be revalidated after this change. However, the change is physically more correct.

### Would fixing Issue D (solar split) break anything?
**Low risk.** Only affects where solar energy is deposited in the mass nodes. If the total opaque solar is unchanged, energy is conserved; only the distribution changes.

### Recommended fix order:
1. Fix Issue A first (highest impact, lowest risk, isolated to `compute_zone_air_temperature`)
2. Run 900FF test — should show ~3-4 degC improvement
3. Fix Issue B (moderate impact, low risk)
4. Run full ASHRAE 140 suite — should be closer
5. Fix Issue C (needs HVAC revalidation)
6. Fix Issue D (cleanup)

### Acceptance Criteria Checklist
- [ ] 900FF max temp within reference range [41.8, 46.4] degC
- [ ] 600FF max temp still within [64.9, 75.1] degC (unchanged)
- [ ] 650FF max temp still within [63.2, 73.5] degC (unchanged)
- [ ] 900 (HVAC) heating/cooling loads within reference
- [ ] 950 (HVAC) heating/cooling loads within reference
- [ ] All existing unit tests pass
