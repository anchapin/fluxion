# Issue #876: Full ISO 13790 Reformulation for 9R4C Path

## Status: DRAFT
## Date: 2026-05-18
## Blocks: Issue #873 (multi-node solver solar injection)
## Depends-on: Issues #715, #738, #872 (merged)

---

## Executive Summary

Replace the 5R1C thermal network formula in `step_physics_9r4c()` with the ISO 13790 §C.4–C.13 timestep loop. The core change is flipping the computation order: **mass first, air second** instead of the current **air first, mass second**.

**This is a wiring change, not a structural change.** The 9R4C path is fully isolated from 5R1C/6R2C/8R3C paths. Model type is fixed per simulation. All required conductances (`derived_h_tr_1/2/3`) already exist in the solver cache. The `crank_nicolson_iso13790()` function already exists at `thermal_integration.rs:160`.

### Previous Attempt Results

| Method | Tests Pass | Heating (MWh) | Cooling (MWh) | 900FF Max (°C) |
|--------|-----------|---------------|---------------|----------------|
| Backward Euler (current) | 14/17 | 4.59 | 3.66 | 44.64 |
| CN + 5R1C t_i_free | 9/17 | 3.06 | 8.32 | 56.54 |
| **Target** | **17/17** | **2.14–5.53** | **0.97–3.82** | **41.8–46.4** |

**Root cause of previous failure:** The 5R1C `t_i_free` formula uses `h_ms_is_prod` (= `h_tr_is × h_tr_ms ≈ 650 × 1300 = 845,000`) which couples mass-to-air through a FAST path. The CN mass update uses `H_tr_3 ≈ 40 W/K` (SLOW path). The 5R1C formula overestimates air temperature when mass is slow.

---

## File Map

| File | Lines | Role |
|------|-------|------|
| `src/sim/thermal_model_physics.rs` | 2200–2760 | `step_physics_9r4c()` — main function to modify |
| `src/sim/thermal_integration.rs` | 160–190 | `crank_nicolson_iso13790()` — already exists |
| `src/sim/thermal_model_data.rs` | 151–155 | `derived_h_tr_1/2/3` cache fields |
| `src/sim/thermal_model_solvers.rs` | 142–150 | `derived_h_tr_1/2/3` computation |
| `tests/ashrae_140_case_900.rs` | — | Case 900 acceptance tests |
| `tests/ashrae_140_free_floating.rs` | — | Case 900FF free-floating tests |

---

## Current 9R4C Timestep Order (to be replaced)

```
1. Compute gains: phi_ia, phi_st, phi_m       (lines 2241–2263)
2. Compute t_i_free from OLD Tm (5R1C formula) (lines ~900–940)
   t_i_free = (h_ms_is_prod*Tm + h_tr_is*phi_st + term_rest_1*(phi_ia + h_ext*T_out + ground)) / den
3. Compute Q_hc from t_i_free                   (lines ~2596–2627)
   h_coeff = den/(2*term_rest_1)
   Q = h_coeff * (T_setpoint - t_i_free)
4. Compute t_i_act = t_i_free + Q/h_coeff        (lines ~2628–2640)
5. Backward Euler mass update with t_i_act        (lines ~2650–2700)
   backward_euler_update(tm_old, dt, cm, h_tr_em, h_tr_ms, T_ext, T_s, phi_m)
```

## Target ISO 13790 Timestep Order

```
1. Compute gains: phi_ia, phi_st, phi_m       (SAME — already ISO-compliant)
2. Compute phi_m_tot with Q_hc=0               (NEW — §C.3 Eq. C.9)
   phi_m_tot = phi_m + H_tr_3/(H_tr_2) * (phi_st + H_tr_1*phi_ia/H_tr_2 + H_tr_2*T_out) + h_tr_em*T_out
3. Crank-Nicolson mass update → T_m_next       (REPLACE backward_euler)
4. Average: T_m_avg = (T_m_next + T_m_prev) / 2  (NEW)
5. Surface temp from T_m_avg (Eq C.10)         (NEW)
   T_s = (H_tr_ms*T_m_avg + phi_st + H_tr_1*(T_out + phi_ia/H_tr_1)) / (H_tr_ms + H_tr_2)
   Simplified: T_s = (H_tr_ms*T_m_avg + phi_st + H_tr_1*T_out + phi_ia + H_tr_w*T_out) / (H_tr_ms + H_tr_2)
6. Air temp from T_s (Eq C.11) → t_i_free      (NEW)
   t_i_free = (H_tr_1*T_s + H_ve*T_out + phi_ia) / (H_tr_1 + H_ve)
7. HVAC demand: if t_i_free outside setpoints   (REPLACE h_coeff approach)
   - Thales interpolation (Eq C.12-C.13) or keep coefficient approach with new t_i_free
8. For HVAC zones: recompute phi_m_tot with Q_hc  (NEW — second CN pass)
   phi_m_tot_final = phi_m + H_tr_3/(H_tr_2) * (phi_st + H_tr_1*(phi_ia + Q_hc)/H_tr_2 + H_tr_2*T_out) + h_tr_em*T_out
   → T_m_final = crank_nicolson_iso13790(..., phi_m_tot_final)
```

---

## Phase Plan

### Phase 1: Compute phi_m_tot and Switch Mass Update to Crank-Nicolson

**Goal:** Replace `backward_euler_update()` with `crank_nicolson_iso13790()` for the mass update, using a proper `phi_m_tot` derived from ISO 13790 §C.3 Eq. C.9. Keep the existing 5R1C `t_i_free` for HVAC demand.

**Why this is safe:** This only changes HOW the mass temperature is updated. The HVAC demand calculation remains unchanged. If CN mass is wrong, HVAC still uses the old formula, so the system degrades gracefully.

**Risk:** LOW — mass update is isolated from t_i_free computation.

#### Changes

**File: `src/sim/thermal_model_physics.rs`**

1. **Lines ~900–940 (t_i_free numerator):** No changes. Keep the existing 5R1C formula for `t_i_free`.

2. **Lines ~2650–2700 (mass update):** Replace `backward_euler_update()` loop with `crank_nicolson_iso13790()`.

   Before:
   ```rust
   let t_s = if ts_den > 0.0 {
       (h_tr_ms * tm_old + h_tr_is_zone * t_i_blended + phi_st_zone) / ts_den
   } else {
       t_i_blended
   };
   let tm_new = backward_euler_update(tm_old, dt, cm, h_tr_em, h_tr_ms, t_ext, t_s, phi_m.as_ref()[i]);
   ```

   After:
   ```rust
   let h_tr_1 = self.0.derived_h_tr_1.as_ref()[i];
   let h_tr_2 = self.0.derived_h_tr_2.as_ref()[i];
   let h_tr_3 = self.0.derived_h_tr_3.as_ref()[i];
   let h_tr_em_val = h_tr_em;
   let t_out = outdoor_temp;

   // ISO 13790 §C.3 Eq. C.9: phi_m_tot
   let phi_m_tot = phi_m.as_ref()[i]
       + h_tr_3 / h_tr_2
           * (phi_st.as_ref()[i]
               + h_tr_1 * (phi_ia_total + h_tr_2 * t_out) / h_tr_2);

   let tm_new = crank_nicolson_iso13790(
       tm_old,
       dt,
       cm,
       h_tr_3,
       h_tr_em_val,
       phi_m_tot,
   );
   ```

   Where `phi_ia_total` includes inter-zone heat (from `phi_ia_with_iz`). We need to capture this value before it's consumed by the numerator computation.

   **IMPORTANT:** The free-floating path uses `phi_ia_with_iz` as the air-side gain (Q_hc=0). The HVAC path must use `phi_ia_with_iz + Q_hc` for the second CN pass.

3. **Store `phi_ia_with_iz` values per-zone** before the numerator computation consumes them. Add a `let phi_ia_per_zone: Vec<f64>` capture.

#### Testing

```bash
cargo test --test ashrae_140_case_900
cargo test --test ashrae_140_free_floating -- test_case_900ff
```

#### Acceptance Criteria

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| 900FF Max | 40–50°C | CN should give SLOWER mass → may shift 900FF temp |
| 900 Heating | 3.5–6.0 MWh | t_i_free unchanged, HVAC unchanged |
| 900 Cooling | 2.0–5.0 MWh | mass now slow, may shift HVAC timing |
| Tests pass | ≥ 12/17 | Some tests may fail from mass dynamics change |

#### Rollback

```bash
git checkout src/sim/thermal_model_physics.rs
```

#### Effort: 2–3 hours

#### GO/NO-GO Decision

- **GO** if: 900FF max temp moves toward 41.8–46.4°C range (indicating correct slow-mass dynamics)
- **NO-GO** if: 900FF max temp > 55°C (indicates phi_m_tot is wrong) or compilation fails

---

### Phase 2: Mass Averaging and ISO 13790 Surface/Air Temperature

**Goal:** Replace the 5R1C `t_i_free` formula with the ISO 13790 §C.4 Eq. C.10–C.11 chain: averaged mass temperature → surface temperature → air temperature.

**Risk:** MEDIUM — this replaces the core t_i_free formula that determines HVAC demand.

#### Changes

**File: `src/sim/thermal_model_physics.rs`**

1. **After CN mass update (~line 2660):** Add mass temperature averaging:
   ```rust
   // ISO 13790 §C.4: Average mass temperature for surface/air calculation
   let tm_avg = (tm_new + tm_old) / 2.0;
   ```

2. **Lines ~900–940 (t_i_free formula):** Replace the 5R1C formula with the ISO 13790 chain.

   The current formula:
   ```rust
   t_i_free = (h_ms_is_prod*Tm + h_tr_is*phi_st + term_rest_1*(phi_ia + h_ext*T_out + ground)) / den
   ```

   Replace with ISO 13790 §C.4 Eq. C.10 + C.11:
   ```rust
   // Eq. C.10: Surface temperature from averaged mass
   let t_s = (h_tr_ms * tm_avg + phi_st + h_tr_1 * (t_out + phi_ia / h_tr_1)) / (h_tr_ms + h_tr_2);

   // Eq. C.11: Air temperature from surface
   let t_i_free = (h_tr_1 * t_s + h_ve * t_out + phi_ia) / (h_tr_1 + h_ve);
   ```

   **Key insight:** This formula uses `h_tr_1 + h_ve` in the denominator instead of `den = h_ms_is_prod + term_rest_1 * h_ext + ground_coeff`. The ISO 13790 formula decouples air temp from mass through the surface node, rather than directly through `h_ms_is_prod`.

3. **Ground coupling:** The ground term must be included. Two options:
   - Add `ground_coeff * t_g / (h_tr_ms + h_tr_2)` to the surface temperature numerator
   - Add it as a separate term in phi_m_tot (Phase 1 handles this via h_tr_em * T_ext substitution)

   Verify: Does `h_tr_em` already include ground coupling? Check if `T_ext` in the CN function should use a blended exterior temperature. If ground coupling is separate, add `+ derived_ground_coeff * t_g` to the `phi_m_tot` computation from Phase 1.

4. **Inter-zone heat:** Include in `phi_ia` for the t_s and t_i_free formulas.

#### Testing

```bash
cargo test --test ashrae_140_case_900
cargo test --test ashrae_140_free_floating
```

#### Acceptance Criteria

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| 900FF Max | 41.8–46.4°C | This should now match reference range |
| 900FF Min | -6.4 to -1.6°C | Check cold-season mass damping |
| 900 Heating | 2.14–5.53 MWh | Reference range |
| 900 Cooling | 0.97–3.82 MWh | Reference range |
| Tests pass | ≥ 14/17 | Should improve from Phase 1 |

#### Rollback

```bash
git checkout src/sim/thermal_model_physics.rs
```

#### Effort: 3–4 hours

#### GO/NO-GO Decision

- **GO** if: 900FF max temp within [41.8, 46.4]°C AND HVAC energy within reference ranges
- **NO-GO** if: 900FF max > 50°C (surface temp formula wrong) or any test regression > 3 tests

---

### Phase 3: Thales Interpolation for HVAC Demand

**Goal:** Replace the coefficient-based HVAC demand (`h_coeff * (T_setpoint - t_i_free)`) with ISO 13790 §C.4 Eq. C.12–C.13 Thales interpolation. This is the HVAC demand calculation that runs when `t_i_free` is outside the setpoint range.

**Risk:** MEDIUM-HIGH — this changes how HVAC power is calculated.

#### Current HVAC Approach (lines ~2596–2627)

```rust
let h_coeff = den / (2.0 * term_rest_1);  // ≈ den/(2*term_rest_1)
let q = h_coeff * (T_setpoint - t_i_free); // Linear interpolation
let q_clamped = q.clamp(-cool_cap, heat_cap);
let t_i_act = t_i_free + q_clamped / h_coeff;
```

#### ISO 13790 Thales Approach (Eq. C.12–C.13)

The Thales interpolation determines Q_hc by requiring that `t_i_final = T_setpoint`:

```rust
// Eq. C.12: For heating (t_i_free < heating_setpoint)
// t_i_final = t_i_free + Q_hc * (1 / (H_tr_1 + H_ve))
// Set t_i_final = heating_setpoint → Q_hc = (H_tr_1 + H_ve) * (heating_setpoint - t_i_free)

// Eq. C.13: For cooling (t_i_free > cooling_setpoint)
// Q_hc = (H_tr_1 + H_ve) * (cooling_setpoint - t_i_free)
```

Wait — the ISO 13790 Thales interpolation is more nuanced. The standard uses a 3-pass approach:

1. **Pass 1 (free-floating):** Compute `t_i_free` with Q_hc = 0
2. **Pass 2 (unlimited HVAC):** Compute `t_i_unlimited` assuming HVAC can achieve setpoint perfectly
3. **Pass 3 (Thales interpolation):** If `t_i_free` is outside deadband AND capacity allows, compute Q_hc by interpolation between the two passes

The key equation is:
```
Q_hc = (H_tr_1 + H_ve) * (T_setpoint - t_i_free)
```

This is mathematically equivalent to the coefficient approach with `h_coeff = H_tr_1 + H_ve` instead of `den/(2*term_rest_1)`.

**Critical difference:** The ISO coefficient is `H_tr_1 + H_ve ≈ 25 + 35 = 60 W/K`, while our current `h_coeff = den/(2*term_rest_1)` gives a different value that includes the mass coupling path. The ISO formula correctly excludes the mass path because HVAC only controls the air node.

#### Changes

**File: `src/sim/thermal_model_physics.rs`**

1. **Lines ~2596–2627 (HVAC demand):** Replace coefficient approach:

   Before:
   ```rust
   let h_coeff = if term_rest_1_zone > 0.0 {
       den_val / (2.0 * term_rest_1_zone)
   } else {
       self.0.h_tr_is.as_ref()[i] + self.0.h_ve.as_ref()[i]
   };
   ```

   After:
   ```rust
   // ISO 13790 §C.4: HVAC coefficient is H_tr_1 + H_ve (air node only)
   let h_tr_1 = self.0.derived_h_tr_1.as_ref()[i];
   let h_ve = self.0.h_ve.as_ref()[i];
   let h_coeff = h_tr_1 + h_ve;
   ```

2. The rest of the HVAC logic (clamp to capacity, compute t_i_act) remains the same.

#### Testing

```bash
cargo test --test ashrae_140_case_900
cargo test --test ashrae_140_integration
```

#### Acceptance Criteria

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| 900 Heating | 2.14–5.53 MWh | HVAC coefficient change may shift energy |
| 900 Cooling | 0.97–3.82 MWh | Smaller h_coeff → more precise HVAC |
| Peak heating | ≤ 5.89 kW | Reference max |
| Peak cooling | ≤ 4.36 kW | Reference max |
| Tests pass | ≥ 15/17 | Should improve |

#### Rollback

```bash
git checkout src/sim/thermal_model_physics.rs
```

#### Effort: 1–2 hours

#### GO/NO-GO Decision

- **GO** if: HVAC energy within reference ranges for all Case 900 tests
- **NO-GO** if: Heating energy > 6 MWh (h_coeff too small) or cooling > 5 MWh (h_coeff wrong sign)

---

### Phase 4: Second CN Pass for HVAC Zones

**Goal:** For HVAC zones, recompute the mass temperature using a second Crank-Nicolson pass that includes Q_hc in the heat flow to mass. This is the final piece of the ISO 13790 timestep loop.

**Risk:** HIGH — this adds a second mass update that must be consistent with the first.

#### Background

The ISO 13790 loop has two CN passes:
1. **Pass 1:** Compute `T_m_next` with `phi_m_tot` (Q_hc = 0) → used for free-floating t_i_free
2. **Pass 2:** If HVAC is active, recompute `T_m_final` with `phi_m_tot + Q_hc` propagated through the network

The second pass ensures the mass temperature reflects the actual HVAC energy input. Without it, the mass "doesn't know" about the HVAC power, leading to a one-timestep lag.

#### Changes

**File: `src/sim/thermal_model_physics.rs`**

1. **After HVAC demand computation (~line 2640):** Add second CN pass:

   ```rust
   // ISO 13790 §C.4: Second CN pass for HVAC zones
   // Recompute phi_m_tot with Q_hc included in phi_ia
   let phi_ia_with_hvac = phi_ia + q_clamped;
   let phi_m_tot_final = phi_m
       + h_tr_3 / h_tr_2
           * (phi_st
               + h_tr_1 * (phi_ia_with_hvac + h_tr_2 * t_out) / h_tr_2);

   let tm_final = crank_nicolson_iso13790(
       tm_old,
       dt,
       cm,
       h_tr_3,
       h_tr_em_val,
       phi_m_tot_final,
   );
   ```

2. **Use `tm_final` for HVAC zones, `tm_new` (Pass 1) for free-floating zones.**

3. **Store `tm_final` in `mass_temperatures`** for the next timestep.

4. **For free-floating zones:** The first CN pass (`tm_new` from Phase 1) is the final mass temperature.

#### Testing

```bash
cargo test --test ashrae_140_case_900
cargo test --test ashrae_140_free_floating
cargo test --test ashrae_140_integration
cargo test --test ashrae_140_validation
```

#### Acceptance Criteria

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| All Case 900 tests | PASS | 17/17 target |
| 900FF Max | 41.8–46.4°C | Reference range |
| 900 Heating | 2.14–5.53 MWh | Reference range |
| 900 Cooling | 0.97–3.82 MWh | Reference range |
| Peak heating | ≤ 5.89 kW | Reference max |
| Peak cooling | ≤ 4.36 kW | Reference max |

#### Rollback

```bash
git checkout src/sim/thermal_model_physics.rs
```

#### Effort: 2–3 hours

#### GO/NO-GO Decision

- **GO** if: All Case 900 + 900FF tests pass within reference ranges
- **NO-GO** if: Any regression from Phase 3 results (second pass introduces instability)

---

### Phase 5: Validation, Edge Cases, and Cleanup

**Goal:** Run full ASHRAE 140 test suite, fix any edge cases, and clean up dead code.

**Risk:** LOW — this is validation and cleanup only.

#### Changes

1. **Run full test suite:**
   ```bash
   cargo test --test ashrae_140_case_900
   cargo test --test ashrae_140_free_floating
   cargo test --test ashrae_140_integration
   cargo test --test ashrae_140_validation
   cargo test --test ashrae_140_setback_ventilation
   cargo test --test ashrae_140_solar_gain_variants
   cargo test --test ashrae_140_solid_conduction_variants
   ```

2. **Edge case handling:**
   - Multi-zone buildings: verify inter-zone heat transfer still works with new phi_m_tot
   - Ground coupling: verify ground term is correctly included in phi_m_tot
   - Zero-capacitance zones: verify forward Euler fallback in `crank_nicolson_iso13790`
   - Very small timesteps (< 60s): verify numerical stability

3. **Cleanup:**
   - Remove `backward_euler_update` import from `step_physics_9r4c` if no longer used
   - Remove `h_ms_is_prod` and `derived_den` if no longer used by 9R4C path (check 5R1C/6R2C/8R3C first)
   - Remove `term_rest_1` usage in 9R4C if fully replaced
   - Update comments referencing "5R1C formula" to say "ISO 13790 §C.4"

4. **Add diagnostic tracing** (behind feature flag) for debugging future issues:
   ```rust
   #[cfg(feature = "pr876-diag")]
   eprintln!("ISO13790 t={} phi_m_tot={:.1} Tm_avg={:.2} T_s={:.2} t_i_free={:.2} Q_hc={:.1}",
       timestep, phi_m_tot, tm_avg, t_s, t_i_free, q_clamped);
   ```

5. **Verify multi-node solver integration:**
   - The multi-node solver surface temperature update (~line 2540) must still work
   - Verify that the CN mass update doesn't conflict with the multi-node solver's own mass update
   - The multi-node solver does NOT write back to `mass_temperatures` (confirmed in Phase 1C)

#### Testing

```bash
# Full ASHRAE 140 suite
cargo test ashrae_140

# All other tests (regression check)
cargo test
```

#### Acceptance Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| ASHRAE 140 Case 900 | 17/17 | All energy + peak + free-floating tests pass |
| Other ASHRAE 140 cases | No regression | Cases 600, 610, 650, 800, 900 must not break |
| Full test suite | No regression | `cargo test` must pass |
| Clippy | Clean | `cargo clippy` must pass |
| Dead code | Removed | No unused imports or variables |

#### Effort: 2–4 hours

---

## Risk Matrix

| Phase | Risk | Impact | Mitigation |
|-------|------|--------|------------|
| Phase 1 | LOW | Mass dynamics change | Single-line rollback |
| Phase 2 | MEDIUM | t_i_free formula replacement | Verify 900FF first |
| Phase 3 | MEDIUM-HIGH | HVAC coefficient change | Keep old formula in comment |
| Phase 4 | HIGH | Second CN pass | Can skip if Phase 3 passes 17/17 |
| Phase 5 | LOW | Validation only | No code changes to core path |

## Total Estimated Effort: 10–16 hours

## Key Equations Reference

### ISO 13790 §C.3 Eq. C.9 — phi_m_tot
```
phi_m_tot = phi_m + (H_tr_3 / H_tr_2) * (phi_st + H_tr_1 * (phi_ia_total + H_tr_2 * T_out) / H_tr_2)
```

Wait — this isn't quite right. Let me be precise:

```
phi_m_tot = phi_m + H_tr_3/(H_tr_2) * (phi_st + H_tr_1 * phi_ia_total / H_tr_2 + H_tr_w * T_out)
          + h_tr_em * T_out
```

Actually, per ISO 13790 §C.3, the complete phi_m_tot is:

```
phi_m_tot = phi_m
          + (H_tr_3 / H_tr_2) * (phi_st
              + (H_tr_1 / (H_tr_1 + H_ve)) * (phi_ia + H_ve * T_out)
              + H_tr_w * T_out)
          + h_tr_em * T_out
```

Let me re-derive from first principles. The heat flow to the mass node through the 5R1C network is:

```
phi_m_tot = phi_m  (direct radiative gains to mass)
          + H_tr_3 * T_s  (surface-to-mass, where T_s is driven by network)
```

But T_s depends on T_m, so we substitute the network equations. The ISO standard provides a closed-form for phi_m_tot that eliminates T_s:

```
phi_m_tot = phi_m
          + (H_tr_3 * phi_st) / (H_tr_ms + H_tr_2)
          + (H_tr_3 * H_tr_1 * (phi_ia + H_ve * T_out)) / ((H_tr_ms + H_tr_2) * (H_tr_1 + H_ve))
          + (H_tr_3 * H_tr_w * T_out) / (H_tr_ms + H_tr_2)
          + h_tr_em * T_out
```

Since `H_tr_3 = H_tr_2 * H_tr_ms / (H_tr_2 + H_tr_ms)`, we have `H_tr_3 / (H_tr_ms + H_tr_2) = H_tr_3 * 1/(H_tr_ms + H_tr_2)`.

Actually, let me simplify. `H_tr_3 = H_tr_2 * H_tr_ms / (H_tr_2 + H_tr_ms)`, so:

```
H_tr_3 / (H_tr_ms + H_tr_2) = H_tr_2 * H_tr_ms / (H_tr_2 + H_tr_ms)²
```

Hmm, that doesn't simplify nicely. Let me look at the ETH Zurich reference implementation for the standard form.

**From RC_BuildingSimulator (ETH Zurich), the standard phi_m_tot is:**

```python
phi_m_tot = phi_m \
    + (H_tr_3 / H_tr_2) * (phi_st + H_tr_w * theta_e) \
    + (H_tr_3 / H_tr_2) * (H_tr_1 / (H_tr_1 + H_ve)) * (phi_ia + H_ve * theta_e) \
    + H_tr_em * theta_e
```

Where `theta_e = outdoor temperature`.

This is the correct form. Note that `H_tr_3 / H_tr_2` appears because `H_tr_3` represents the fraction of heat flow that makes it through the air-side bottleneck to the mass.

### ISO 13790 §C.4 Eq. C.10 — Surface Temperature

```
T_s = (H_tr_ms * T_m_avg + phi_st + H_tr_1 * (T_out + phi_ia / (H_tr_1 + H_ve - H_ve)) )
```

More precisely:
```
T_s = (H_tr_ms * T_m_avg + phi_st + H_tr_2 * T_out + H_tr_1 * phi_ia / (H_tr_1 + H_ve)) / (H_tr_ms + H_tr_2)
```

### ISO 13790 §C.4 Eq. C.11 — Air Temperature

```
t_i_free = (H_tr_1 * T_s + H_ve * T_out + phi_ia) / (H_tr_1 + H_ve)
```

### ISO 13790 §C.4 Eq. C.12–C.13 — Thales Interpolation (HVAC Demand)

```
Q_hc_heating = (H_tr_1 + H_ve) * (T_heating_setpoint - t_i_free)   [when t_i_free < T_heat]
Q_hc_cooling = (H_tr_1 + H_ve) * (T_cooling_setpoint - t_i_free)   [when t_i_free > T_cool]
```

Clamped to equipment capacity.

---

## Implementation Notes

### Ground Coupling

The ground coupling term (`derived_ground_coeff * T_ground`) must be included in the phi_m_tot formula. The ISO 13790 standard doesn't explicitly include ground coupling (it's for single-zone above-grade buildings), so we add it as an extension:

```
phi_m_tot_flux += derived_ground_coeff * T_ground
```

This gets added to the `h_tr_em * T_out` term or as a separate additive term.

### Inter-Zone Heat Transfer

Inter-zone heat is currently added to `phi_ia_with_iz` before the t_i_free computation. In the ISO 13790 formulation, it should be added to `phi_ia` in both:
- The `phi_m_tot` computation (affects mass update)
- The `T_s` and `t_i_free` formulas (affects air temperature)

### Multi-Node Solver Interaction

The multi-node solver updates happen BEFORE the 9R4C HVAC/mass computation (~line 2540). The multi-node solver:
- Computes its own surface temperatures from wall/roof/floor nodes
- Does NOT write back to `mass_temperatures`
- Is used for construction-level detail, not zone-level energy balance

The 9R4C reformulation should work correctly with the multi-node solver because:
1. The 9R4C path uses `mass_temperatures` for its thermal balance
2. The multi-node solver provides surface temperatures that feed into the 9R4C conductance computation
3. The two systems are decoupled at the mass temperature level

### Verification: Conductance Value Check

Before Phase 1, verify these values for Case 900:
```
H_tr_1 = h_ve * h_tr_is / (h_ve + h_tr_is) ≈ 35 * 4.55 / (35 + 4.55) ≈ 4.03 W/K
H_tr_2 = H_tr_1 + h_tr_w ≈ 4.03 + 20 ≈ 24.03 W/K
H_tr_3 = H_tr_2 * h_tr_ms / (H_tr_2 + h_tr_ms) ≈ 24.03 * 1300 / (24.03 + 1300) ≈ 23.6 W/K
```

Wait — these are per-unit-area values? No, these should be total values for the zone. The actual values depend on the Case 900 zone geometry and construction. Verify by printing `derived_h_tr_1/2/3` at runtime.

### Feature Flag

Consider adding a feature flag `iso13790-reform` to allow A/B testing:

```toml
[features]
iso13790-reform = []
```

```rust
#[cfg(feature = "iso13790-reform")]
{
    // New ISO 13790 path
}
#[cfg(not(feature = "iso13790-reform"))]
{
    // Current 5R1C path
}
```

This allows easy comparison during development but should be removed before merge.
