# ISO 13790 Annex C Equation Mapping to Fluxion Codebase

Generated: 2026-05-18
Reference: ISO 13790:2008 §C.1–C.13

---

## §C.1–C.3: Heat Flow Splitting

| ISO Eq | ISO Formula | Our Code Location | Our Formula | Status | Notes |
|--------|-------------|-------------------|-------------|--------|-------|
| C.1 | `φ_ia = 0.5 × φ_int` | `src/sim/thermal_model_physics.rs:602` | `phi_ia = load_w × conv_frac + sol_w × solar_distribution_to_air` | ⚠️ | ISO splits internal gains 50/50 convective/radiative. Our code uses configurable `conv_frac` (defaults may differ from 0.5) and routes solar to air via `solar_distribution_to_air`. Not an exact 0.5 factor. |
| C.2 | `φ_st = (1 - A_m/A_t - H_tr_w/(9.1×A_t)) × (0.5×φ_int + φ_sol)` | `src/sim/thermal_model_physics.rs:603` | `phi_st = load_w × st_int_frac + remaining_sol × st_sol_frac` | ⚠️ | ISO uses area-based weighting fractions (A_m/A_t, H_tr_w/9.1A_t). Our code uses `st_int_frac = rad_frac × (1 - solar_distribution_to_air)` and `st_sol_frac = 1 - solar_beam_to_mass_fraction`. Structurally different parameterisation but same intent. |
| C.3 | `φ_m = (A_m/A_t) × (0.5×φ_int + φ_sol)` | `src/sim/thermal_model_physics.rs:604` | `phi_m = load_w × m_air_frac + remaining_sol × m_sol_frac + opaque_sol_w` | ⚠️ | ISO routes to mass node via area ratio A_m/A_t. Our code uses `m_air_frac = rad_frac × solar_distribution_to_air` and `m_sol_frac = solar_beam_to_mass_fraction`. Includes opaque solar separately. |

### Heat Flow Splitting — Key Differences

**ISO 13790 approach**: The three splitting fractions are derived purely from geometry:
- `A_m/A_t` = effective mass area / total internal surface area
- `H_tr_w/(9.1×A_t)` = window conductance fraction
- These guarantee `φ_ia + φ_st + φ_m = φ_int + φ_sol` by construction

**Our approach**: Uses configurable distribution fractions:
- `conv_frac` / `rad_frac` split internal gains
- `solar_distribution_to_air` routes solar
- `solar_beam_to_mass_fraction` routes beam solar to mass

The energy conservation `phi_ia + phi_st + phi_m == total_load` is verified in
`src/validation/thermal_mass_energy_accounting.rs:1158`.

**HVAC power routing**: ISO §C.4.2 adds `φ_H_cd` to `φ_ia` (air node only).
Our code adds HVAC to the air node at line 1164: `t_i_act = t_free + hvac[i] / h_is`.
This is **equivalent** — HVAC power heats the air, which then propagates to mass.

---

## §C.4: Mass Temperature (Crank-Nicolson)

| ISO Eq | ISO Formula | Our Code Location | Our Formula | Status | Notes |
|--------|-------------|-------------------|-------------|--------|-------|
| C.4 | `Tm_next = [Tm_prev×(Cm/Δt − 0.5×(H_tr_3+H_tr_em)) + φ_m_tot] / [Cm/Δt + 0.5×(H_tr_3+H_tr_em)]` | `src/sim/thermal_integration.rs:160-188` (`crank_nicolson_iso13790`) | `numer = tm_prev × (cm_dt − 0.5×(h_tr_3+h_tr_em)) + phi_m_tot; numer / (cm_dt + 0.5×(h_tr_3+h_tr_em))` | ✅ | **Exact match**. Function exists but is NOT called in the main `step_physics_5r1c` loop. |
| C.9 | `Tm = (Tm_next + Tm_prev) / 2` | ❌ MISSING | — | ❌ | No averaging step found. Our code uses `Tm_next` directly without averaging. |

### Mass Temperature — Key Differences

**The `crank_nicolson_iso13790` function exists** at `thermal_integration.rs:160` and implements
Eq C.4 exactly (takes `h_tr_3` + `h_tr_em` + `phi_m_tot`). However, the main simulation loop
at `thermal_model_physics.rs:1299-1355` calls **different** integration methods:

- **Backward Euler** (`backward_euler_update` at line 1305): Uses `h_tr_em` + `h_tr_ms` directly,
  NOT the combined `H_tr_3`. This is a **fundamental structural difference** from ISO 13790.
- **Crank-Nicolson** (`crank_nicolson_update` at line 1344): Uses `h_tr_em` + `h_tr_ms` directly,
  also NOT `H_tr_3`.

**Neither path computes `phi_m_tot`** (Eq C.5). They use `phi_m_zone` directly with the
un-combined conductances. This means the mass update does NOT propagate HVAC power through
the network (H_tr_1 → H_tr_2 → H_tr_3 chain).

**The Tm averaging (Eq C.9)** is missing entirely. Our code stores `Tm_next` directly.

---

## §C.5: Total Heat Flow to Mass (φ_m_tot)

| ISO Eq | ISO Formula | Our Code Location | Our Formula | Status | Notes |
|--------|-------------|-------------------|-------------|--------|-------|
| C.5 | `φ_m_tot = φ_m + H_tr_em×T_ext + H_tr_3×(φ_st + H_tr_w×T_ext + H_tr_1×(φ_ia_total/H_ve + T_supply)) / H_tr_2` | ❌ MISSING | — | ❌ | No equivalent computation found. Our code passes `phi_m_zone` directly to the mass integrator without the network propagation terms. |

### φ_m_tot — Impact of Missing This

This is the **most critical gap**. In ISO 13790, `φ_m_tot` propagates HVAC power (`φ_ia` includes
`φ_H_cd`) through the full thermal network via the `H_tr_1 → H_tr_2 → H_tr_3` chain. This means
HVAC power directly influences mass temperature evolution.

In our code, the mass update uses only `phi_m` (solar + internal gains to mass) and the direct
conductances `h_tr_em`/`h_tr_ms`. HVAC power enters ONLY through the air temperature
(`t_i_act = t_free + hvac/h_is`), which then influences the surface temperature `t_s`, which
then indirectly affects mass via `h_tr_ms × (t_s − Tm)`. This is a **weaker coupling** than
ISO 13790 intends.

---

## §C.6–C.8: Combined Conductances

| ISO Eq | ISO Formula | Our Code Location | Our Formula | Status | Notes |
|--------|-------------|-------------------|-------------|--------|-------|
| C.6 | `H_tr_1 = 1 / (1/H_ve + 1/H_tr_is)` | `src/sim/thermal_model_solvers.rs:141-143` | `derived_h_tr_1 = h_ve × h_tr_is / (h_ve + h_tr_is)` | ✅ | **Exact match** (parallel conductance formula). |
| C.7 | `H_tr_2 = H_tr_1 + H_tr_w` | `src/sim/thermal_model_solvers.rs:146` | `derived_h_tr_2 = derived_h_tr_1 + h_tr_w` | ✅ | **Exact match**. |
| C.8 | `H_tr_3 = 1 / (1/H_tr_2 + 1/H_tr_ms)` | `src/sim/thermal_model_solvers.rs:148-150` | `derived_h_tr_3 = derived_h_tr_2 × h_tr_ms / (derived_h_tr_2 + h_tr_ms)` | ✅ | **Exact match**. |

### Combined Conductances — Notes

All three are computed and cached in `update_optimization_cache()`. For Case 900 with
`h_ve ≈ 43`, `h_tr_is ≈ 1223`, `h_tr_ms ≈ 1300`, `h_tr_w ≈ 6`:
- `H_tr_1 ≈ 41.5 W/K` ✅
- `H_tr_2 ≈ 47.5 W/K` ✅
- `H_tr_3 ≈ 45.9 W/K` ✅

**However**: These values are computed but **NOT used in the mass temperature update**.
The `crank_nicolson_iso13790` function takes `h_tr_3` as a parameter, but it is never
called with `derived_h_tr_3` in the main simulation loop. The backward Euler and
standard Crank-Nicolson paths both use raw `h_tr_em`/`h_tr_ms` instead.

---

## §C.9–C.11: Temperature Recovery

| ISO Eq | ISO Formula | Our Code Location | Our Formula | Status | Notes |
|--------|-------------|-------------------|-------------|--------|-------|
| C.9 | `Tm = (Tm_next + Tm_prev) / 2` | ❌ MISSING | — | ❌ | No mass temperature averaging. `Tm_next` is stored directly. |
| C.10 | `T_s = (H_tr_ms×Tm + φ_st + H_tr_w×T_ext + H_tr_1×(T_supply + φ_ia/H_ve)) / (H_tr_ms + H_tr_w + H_tr_1)` | `src/sim/thermal_model_physics.rs:1230-1239` | `t_s_act = (h_tr_ms×Tm + h_tr_is×t_i_act + φ_st) / term_rest_1` | ⚠️ | **Structurally different**. Our formula uses `h_tr_is × t_i_act` (already-solved air temp) instead of propagating `H_tr_1×(T_supply + φ_ia/H_ve)` through the network. The denominator `term_rest_1 = h_tr_ms + h_tr_is + h_tr_me` ≠ ISO's `h_tr_ms + H_tr_w + H_tr_1`. |
| C.11 | `T_air = (H_tr_is×T_s + H_ve×T_supply + φ_ia) / (H_tr_is + H_ve)` | `src/sim/thermal_model_physics.rs:929-932` (t_i_free) + `1164` (t_i_act) | `t_i_free = (h_ms_is_prod×Tm + h_tr_is×φ_st + term_rest_1×(φ_ia + h_ext×T_ext)) / den` then `t_i_act = t_free + hvac/h_tr_is` | ⚠️ | **Different decomposition**. Our t_i_free is a combined equation solving T_air directly from all terms (not recovering it step-by-step as ISO does). The HVAC correction `t_i_act = t_free + hvac/h_tr_is` is a linear superposition, not the ISO linear interpolation. |

### Temperature Recovery — Structural Differences

**ISO 13790 recovers temperatures bottom-up**:
1. Average Tm (Eq C.9)
2. Compute T_s from Tm + network (Eq C.10)
3. Compute T_air from T_s + ventilation (Eq C.11)

**Our code solves T_air top-down** in a single combined equation:
- The `t_i_free` computation at line 929-932 is a single algebraic solution that combines
  all network terms into numerator/denominator form
- `t_s_act` at line 1230-1239 is computed AFTER `t_i_act`, not before
- This means surface temperature is a **derived result**, not an intermediate step

The combined equation should be algebraically equivalent IF all conductances match,
but the different decomposition means any discrepancy in conductances or gains
propagates differently.

---

## §C.12–C.13: HVAC Demand (Linear Interpolation / Thales)

| ISO Eq | ISO Formula | Our Code Location | Our Formula | Status | Notes |
|--------|-------------|-------------------|-------------|--------|-------|
| C.12 | `φ_ref = 10 × A_floor` | ❌ MISSING | — | ❌ | No reference power computation. |
| C.13 | `φ_H_cd = φ_ref × (T_set − T_air,0) / (T_air,10 − T_air,0)` | ❌ MISSING | — | ❌ | No linear interpolation (Thales theorem). |
| C.13 clamp | `φ_H_cd = clamp(φ_unrestricted, φ_max_cool, φ_max_heat)` | `src/sim/thermal_model_physics.rs:78` | `demands[0].clamp(-cool_cap, heat_cap)` | ✅ | Capacity clamping exists. |

### HVAC Demand — Our Approach vs ISO

**ISO 13790 (Thales Interpolation)**:
1. Compute `T_air` with `φ_H_cd = 0` → `T_air,0` (free-floating)
2. Compute `T_air` with `φ_H_cd = φ_ref = 10×A_floor` → `T_air,10`
3. Linear interpolate: `φ_H_cd = φ_ref × (T_set − T_air,0) / (T_air,10 − T_air,0)`
4. Clamp to capacity limits

**Our code (IdealLoadsSystem)**:
- `src/sim/thermal_model_physics.rs:45-83`: Uses `IdealLoadsSystem.calculate_power_demand_vector()`
- This computes HVAC demand as `mass_flow × cp × ΔT` (thermodynamic ideal loads formula)
- Then applies at line 1164: `t_i_act = t_free + hvac[i] / h_is`

**Critical difference**: ISO 13790's Thales interpolation **runs the full network twice**
(with φ=0 and φ=φ_ref), capturing how HVAC power propagates through H_tr_1→H_tr_2→H_tr_3
to the mass node. Our ideal loads formula only captures the **air-node** balance
(`Q = m_dot × cp × ΔT`), without network propagation.

The `compute_hvac_demand` in `multi_node_solver.rs:238-254` uses an even simpler formula:
`Q = h_tr_is × (T_set − T_air_free)` — just the interior film conductance, missing
ventilation and the full network coupling.

---

## Summary: Derived Parameter Equivalents

| ISO Symbol | Our Code Field | Location | Notes |
|------------|---------------|----------|-------|
| `H_ve_adj` | `self.0.h_ve` | `thermal_model_solvers.rs:103` | Ventilation conductance |
| `H_tr_is` | `self.0.h_tr_is` | `thermal_model_solvers.rs:107-108` | Interior surface to air |
| `H_tr_ms` | `self.0.h_tr_ms` | `thermal_model_solvers.rs:107-108` | Mass to surface |
| `H_tr_w` | `self.0.h_tr_w` | `thermal_model_solvers.rs:103` | Window conductance |
| `H_tr_em` | `self.0.h_tr_em` | `thermal_model_solvers.rs:101` | Mass to exterior |
| `H_tr_1` | `self.0.derived_h_tr_1` | `thermal_model_solvers.rs:142-143` | ✅ Computed but unused in mass update |
| `H_tr_2` | `self.0.derived_h_tr_2` | `thermal_model_solvers.rs:146` | ✅ Computed but unused in mass update |
| `H_tr_3` | `self.0.derived_h_tr_3` | `thermal_model_solvers.rs:148-150` | ✅ Computed but unused in mass update |
| `C_m` | `self.0.thermal_capacitance` | `thermal_model_physics.rs:1257` | Thermal capacitance |
| `A_floor` | `self.0.zone_area` | `thermal_model_physics.rs:587` | Floor area |

---

## Gap Analysis

### Completely Missing (❌)

1. **Eq C.5 — φ_m_tot computation**: The total heat flow to mass that propagates HVAC power
   through H_tr_1→H_tr_2→H_tr_3. This is the **single most important missing piece**.
   Without it, HVAC power cannot properly influence mass temperature evolution.

2. **Eq C.9 — Tm averaging**: `(Tm_next + Tm_prev) / 2`. Currently `Tm_next` is used directly.

3. **Eq C.12–C.13 — Thales linear interpolation**: The reference-power double-solve method
   for computing HVAC demand. Our IdealLoadsSystem uses `m_dot × cp × ΔT` instead.

### Different (⚠️)

4. **Eq C.1–C.3 — Heat flow splitting**: Uses configurable fractions instead of ISO's
   geometry-derived area ratios. Structurally different but can produce equivalent results
   with correct parameter values.

5. **Eq C.4 — Mass update integration**: The `crank_nicolson_iso13790` function exists and
   matches, but the main simulation loop uses backward Euler / standard Crank-Nicolson
   with raw conductances (h_tr_em, h_tr_ms) instead of the combined H_tr_3.

6. **Eq C.10 — Surface temperature recovery**: Uses `h_tr_is × t_i_act` (already-solved air
   temp) instead of ISO's network-propagated formula with H_tr_1.

7. **Eq C.11 — Air temperature recovery**: Solved as a single combined equation rather than
   the ISO step-by-step recovery from Tm → T_s → T_air.

### Reusable As-Is (✅)

8. **Eq C.6–C.8 — Combined conductances**: `derived_h_tr_1`, `derived_h_tr_2`, `derived_h_tr_3`
   are computed correctly and cached. Just need to be wired into the mass update.

9. **Capacity clamping**: The `.clamp(-cool_cap, heat_cap)` pattern matches ISO §C.13 step 3/4.

10. **The `crank_nicolson_iso13790` function**: Already exists with the correct signature
    and formula. Just needs to be called with the right inputs (h_tr_3, phi_m_tot).

### Recommended Implementation Order

1. **Wire H_tr_3 into mass update** (Eq C.4): Change the mass integration in
   `step_physics_5r1c` to call `crank_nicolson_iso13790` with `derived_h_tr_3` instead of
   the backward Euler with raw conductances. This is a one-line change per integration path.

2. **Implement φ_m_tot** (Eq C.5): Add computation before the mass update:
   ```rust
   let phi_m_tot = phi_m
       + h_tr_em * t_ext
       + h_tr_3 * (phi_st + h_tr_w * t_ext + h_tr_1 * (phi_ia_total / h_ve + t_supply)) / h_tr_2;
   ```

3. **Add Tm averaging** (Eq C.9): After computing `Tm_next`, set
   `Tm_used = (Tm_next + Tm_prev) / 2` for subsequent temperature recovery.

4. **Implement Thales interpolation** (Eq C.12–C.13): Replace IdealLoadsSystem with:
   - Solve network with φ_H_cd = 0 → T_air,0
   - Solve network with φ_H_cd = 10×A_floor → T_air,10
   - Interpolate: `φ_H_cd = φ_ref × (T_set − T_air,0) / (T_air,10 − T_air,0)`
   - Clamp to capacity limits

5. **Fix heat flow splitting** (Eq C.1–C.3): Optionally replace configurable fractions
   with ISO area-ratio formulas for strict compliance. Lower priority since current
   splitting conserves energy.

6. **Fix temperature recovery** (Eq C.10–C.11): Restructure to compute Tm → T_s → T_air
   sequentially instead of the combined equation. This ensures the network propagation
   is physically correct.

### Files Requiring Changes

| File | Changes |
|------|---------|
| `src/sim/thermal_model_physics.rs:1299-1355` | Wire `crank_nicolson_iso13790` + phi_m_tot |
| `src/sim/thermal_model_physics.rs:929-932` | Compute t_i_free after Tm averaging |
| `src/sim/thermal_model_physics.rs:1230-1239` | Fix T_s recovery (Eq C.10) |
| `src/sim/thermal_model_physics.rs:45-83` | Replace IdealLoadsSystem with Thales |
| `src/sim/thermal_integration.rs:160` | Already correct — just needs calling |
| `src/sim/thermal_model_solvers.rs:140-150` | Already correct — values available |
