# Architecture Boundary Analysis: 5R1C → ISO 13790 Reformulation for Case 900

**Status**: COMPLETE
**Date**: 2026-05-18
**Scope**: `step_physics_9r4c()` replacement with ISO 13790 §C.4–C.13

---

```
CHARTER_CHECK:
- Clarification level: LOW
- Task domain: architecture
- Must NOT do: modify code, change 5R1C/6R2C paths, alter test expectations
- Success criteria: identify exact isolation boundaries and shared-state risks
- Assumptions: model type is fixed per simulation run (NineRFourC for Case 900)
```

---

## 1. Module Dependency Graph

```
step_physics()  [dispatcher — thermal_model_physics.rs:508]
  ├── is_nine_r4c_model() → step_physics_9r4c()  [lines 2200-2760]
  ├── is_8r3c_model()     → step_physics_8r3c()  [calls 5r1c internally!]
  ├── is_6r2c_model()     → step_physics_6r2c()  [lines 1399-2142]
  └── else                → step_physics_5r1c()  [lines 536-1398]

step_physics_9r4c() depends on:
  ├── ThermalModelData (shared struct, all paths read/write same fields)
  ├── update_optimization_cache() (derived_den, derived_h_tr_1/2/3, etc.)
  ├── backward_euler_update() (thermal_integration.rs:110)
  ├── crank_nicolson_iso13790() (thermal_integration.rs:160) — EXISTS but UNUSED in 9R4C!
  ├── prepare_solvers_and_sol_air() (sol-air temperature)
  ├── multi_node_solvers[] (9R4C side-car, per-zone MultiNodeSolver)
  └── predictive_controller (HVAC modulation)

step_physics_5r1c() depends on:
  ├── Same ThermalModelData fields
  ├── Same update_optimization_cache() derived fields
  ├── backward_euler_update() (for mass update)
  └── NO multi_node_solvers

CRITICAL: step_physics_8r3c() CALLS step_physics_5r1c() internally (line 2151),
then post-processes 8R3C mass nodes. This means ANY change to 5r1c affects 8r3c.
```

## 2. Isolation Assessment

**YES — the 9R4C path can be changed in near-complete isolation, with 2 caveats.**

### What is isolated:
- **Routing**: `step_physics()` dispatches exclusively by `thermal_model_type`. NineRFourC never calls 5R1C. The 8R3C path calls 5R1C but is a separate branch.
- **Gain distribution**: Both paths compute their own `phi_ia`, `phi_st`, `phi_m` from scratch. No shared gain arrays.
- **Sol-air temperature**: Both paths call `prepare_solvers_and_sol_air()` independently.
- **Inter-zone heat transfer**: Both paths compute their own `h_tr_iz` contributions.
- **HVAC demand**: 9R4C uses `den/(2*term_rest_1)` coefficient; 5R1C uses `ideal_loads` formula. Completely separate logic.
- **Multi-node solver**: The side-car (lines 2409-2562) is exclusively owned by 9R4C. Comment at line 2532 explicitly says "Do NOT write multi-node mass temperatures back to self.0."

### Caveats:
1. **`derived_den`** is computed by `update_optimization_cache()` (shared), consumed by both 5r1c (t_i_free denominator) and 9r4c (HVAC coefficient). If ISO reformulation doesn't use `derived_den` for HVAC, the 5R1C path is unaffected — but the field must still exist.
2. **`mass_temperatures`** is a shared field on the struct. 9R4C writes it at line 2697 (`self.0.mass_temperatures = ...`). If model type is fixed per run (confirmed: `thermal_model_type` is set at build time), no cross-contamination.

## 3. Interface Contract

### `step_physics_9r4c()` promises to callers:

**Input expectations:**
- `timestep: usize` — hour index into weather data
- `outdoor_temp: f64` — current outdoor temperature (°C)
- `dt_seconds: f64` — timestep duration (seconds)
- `self.0` must have: `derived_den`, `derived_h_ext`, `derived_term_rest_1`, `derived_h_ms_is_prod`, `derived_ground_coeff`, `h_tr_em`, `h_tr_ms`, `h_tr_is`, `h_ve`, `mass_temperatures`, `thermal_capacitance`, `loads`, `solar_gains`, `zone_area`, `multi_node_solvers[]`
- HVAC fields: `free_float`, `heating_setpoint`, `cooling_setpoint`, `hvac_heating_capacity`, `hvac_cooling_capacity`

**Output guarantees:**
- Returns `f64` — total HVAC energy for this timestep (Joules). Returns 0.0 for free-floating.
- **Writes to `self.0.temperatures`** — zone air temperatures (the validator reads this)
- **Writes to `self.0.mass_temperatures`** — updated mass node temperatures
- **Writes to `self.0.previous_mass_temperatures`** — saved old mass temps
- **Writes to `self.0.previous_temperatures`** — saved old air temps (via dispatcher)
- **Updates `self.0.multi_node_solvers[]`** — side-car mass node state

**Side effects:**
- May call `self.calc_analytical_loads()` if weather is set (via dispatcher, line 515)
- Calls `prepare_solvers_and_sol_air()` for CTF/FD flux contributions
- Modifies `current_hvac_output` diagnostic field

## 4. Shared State Risks

### SAFE — No cross-path contamination if model type is fixed:

| Field | Written by 9R4C? | Read by 5R1C? | Risk |
|-------|------------------|---------------|------|
| `derived_den` | NO (read-only) | YES (t_i_free denom) | **None** — computed at build time |
| `derived_h_tr_1/2/3` | NO (read-only) | NOT USED in 5R1C | **None** — 5R1C doesn't use these |
| `mass_temperatures` | YES (line 2697) | YES (line 746) | **None if model fixed** — only one path runs |
| `temperatures` | YES (lines 2708-2718) | NO (validator reads) | **None** — post-hoc consumer |
| `multi_node_solvers[]` | YES | NO | **None** — exclusive to 9R4C |

### RISK — `derived_den` HVAC coefficient:

The 9R4C path uses `derived_den` for the HVAC coefficient:
```rust
let h_coeff = den_val / (2.0 * term_rest_1_zone);  // line 2599
```
If ISO 13790 §C.11 replaces this with `H_tr_1` or a different formula, the `derived_den` field becomes unused by 9R4C. **This is safe** — the 5R1C path still reads it independently. The field is computed once at build time and never mutated at runtime.

### CONFIRMED: `mass_temperatures` isolation

Since `thermal_model_type` is set at construction (`from_spec()` → `enable_9r4c_model()`) and never changes during a simulation run, the same path is called every timestep. The `mass_temperatures` written by 9R4C at step N is read by 9R4C at step N+1. The 5R1C path never executes.

### CONFIRMED: `temperatures` validator contract

The validator (engine.rs:319) reads `model.temperatures[0]` — it does not care about the computation method, only the value. Changing from 5R1C t_i_free to ISO §C.11 formula produces a different value, but the validator only compares against ASHRAE 140 tolerance bands.

## 5. Recommended Architecture

### Strategy: "Replace Internals, Preserve Interface"

```
step_physics_9r4c() {
    // PHASE 1: Gain distribution (KEEP — already ISO §C.4/C.5/C.6 compliant)
    phi_ia, phi_st, phi_m = compute_gains()

    // PHASE 2: Free-floating temperature — REPLACE with ISO §C.9–C.11
    // OLD: 5R1C t_i_free = (h_ms_is_prod * Tm + h_tr_is * phi_st + ...) / den
    // NEW: ISO §C.11 t_m = (Cm/dt * Tm_prev + phi_m_tot) / (Cm/dt + H_tr_3 + H_tr_em)
    //      ISO §C.9  t_supply = (phi_ia + ...) / H_tr_1
    //      ISO §C.10 t_air = ...
    t_i_free = iso13790_free_floating_temp(...)

    // PHASE 3: HVAC demand — REPLACE with ISO §C.12
    // OLD: h_coeff = den / (2 * term_rest_1)
    // NEW: phi_HVAC = H_tr_1 * (T_setpoint - T_free)  [or direct §C.12 formula]
    hvac_output, t_i_act = iso13790_hvac_demand(...)

    // PHASE 4: Mass update — REPLACE backward_euler with crank_nicolson_iso13790
    // OLD: backward_euler_update(tm_old, dt, cm, h_tr_em, h_tr_ms, t_ext, t_s, phi_m)
    // NEW: crank_nicolson_iso13790(tm_prev, dt, cm, h_tr_3, h_tr_em, phi_m_tot)
    //      using derived_h_tr_3 (already computed in update_optimization_cache!)
    mass_temperatures = crank_nicolson_iso13790(...)

    // PHASE 5: Multi-node solver side-car (KEEP or DISABLE for initial reformulation)
    // The multi-node solver currently runs as side-car. For ISO 13790 pure implementation,
    // it may conflict because CN already produces correct mass dynamics.
    // Recommendation: KEEP but add a feature flag to disable for validation.

    // PHASE 6: Write outputs (KEEP — same interface)
    self.0.temperatures = t_i_act
    self.0.mass_temperatures = new_mass
    return hvac_energy
}
```

### New derived fields needed:
- **None.** `derived_h_tr_1`, `derived_h_tr_2`, `derived_h_tr_3` already exist in `ThermalModelData` and are already computed by `update_optimization_cache()`. The `crank_nicolson_iso13790()` function already exists in `thermal_integration.rs` (line 160) with the correct signature. It's been implemented but never wired in.

### Key insight: The infrastructure is already built

The `crank_nicolson_iso13790()` function at `thermal_integration.rs:160` takes exactly the arguments needed:
```rust
pub fn crank_nicolson_iso13790(tm_prev, dt, cm, h_tr_3, h_tr_em, phi_m_tot) -> f64
```

The derived fields `derived_h_tr_1/2/3` are already computed. The reformulation is a wiring change, not a structural change.

## 6. Migration Strategy

### Phase 0: Baseline (no code changes)
- Run Case 900FF and 900HVAC validation with current 9R4C + backward_euler
- Record t_i, t_m, HVAC energy for every hour
- These become the regression baseline

### Phase 1: Replace mass update only (safest, most impactful)
- In `step_physics_9r4c()`, replace `backward_euler_update()` call (line ~2700) with `crank_nicolson_iso13790()`
- Use `derived_h_tr_3` (already computed) instead of `h_tr_ms`
- Compute `phi_m_tot` per ISO §C.7: `phi_m_tot = phi_m + H_tr_3 / H_tr_1 * (phi_ia + phi_st) + ...`
- **Risk**: Low. Only changes mass dynamics, not air temperature or HVAC.
- **Test**: Case 900FF peak temp should stay ~42.87°C; mass convergence should take ~500h instead of ~17h

### Phase 2: Replace t_i_free with ISO §C.11
- Replace the 5R1C numerator/denominator t_i_free formula with ISO §C.9–C.11
- `t_m` from Phase 1's CN output feeds into `t_s` via §C.9, then `t_air` via §C.10
- **Risk**: Medium. Changes free-floating temperature values.
- **Test**: Case 600FF must still pass (5R1C path unchanged). Case 900FF peak may change.

### Phase 3: Replace HVAC demand with ISO §C.12
- Replace `h_coeff = den / (2 * term_rest_1)` with `phi_HC = H_tr_1 * (T_set - T_air_without_HVAC)`
- **Risk**: Medium. Changes annual HVAC energy.
- **Test**: Case 900HVAC annual energy must be within ASHRAE 140 tolerance.

### Phase 4: Multi-node solver coexistence decision
- **Option A**: Disable multi-node solver for Case 900, rely purely on CN mass update
- **Option B**: Keep multi-node solver, use CN for `mass_temperatures`, multi-node for diagnostics
- **Recommendation**: Option A for ASHRAE 140 validation. The CN scheme with `H_tr_3 ≈ 40 W/K` already produces the correct 500h time constant without needing per-surface mass nodes.

### Phase 5: Cleanup
- Remove dead code: the 5R1C t_i_free computation within step_physics_9r4c
- Add documentation linking each formula to ISO 13790 §C.4–C.13 equation numbers
- Consider renaming `step_physics_9r4c` to `step_physics_iso13790` if the 9R4C network is fully replaced

---

## Specific Concerns — Resolutions

### 1. `derived_den` shared between paths
**Resolution**: No issue. `derived_den` is computed once at build time. 9R4C can stop reading it without affecting 5R1C. The 5R1C path reads `derived_den` independently for its own t_i_free formula.

### 2. `mass_temperatures` cross-contamination
**Resolution**: No issue. Model type is fixed per simulation. 9R4C's `mass_temperatures` are never seen by 5R1C code paths.

### 3. `temperatures` validator expectations
**Resolution**: The validator only reads `temperatures[0]` and compares against tolerance bands. It does not inspect the computation method. Different values from ISO §C.11 are acceptable as long as they fall within ASHRAE 140 bands.

### 4. Multi-node solver coexistence
**Resolution**: The multi-node solver runs as a side-car and its state is explicitly NOT written back to `self.0.mass_temperatures` (comment at line 2532). The ISO 13790 CN reformulation can coexist — the multi-node solver provides diagnostic per-surface temperatures while CN provides the authoritative `mass_temperatures`. For pure ISO validation, the multi-node solver can be disabled.
