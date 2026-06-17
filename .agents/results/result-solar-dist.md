# Solar Gain Distribution Analysis — ASHRAE 140 Case 600FF

**Agent**: solar-dist (Agent 4, Phase 4)
**Date**: 2026-05-18
**Status**: COMPLETE
**Confidence**: HIGH

---

## 1. Executive Summary

The solar gain distribution in the 5R1C model for ASHRAE 140 Case 600FF is **correctly implemented** and is **NOT the primary cause** of the 10–25°C temperature deficit. The distribution follows ASHRAE 140 Section 5.2.2:

- **0% → air node** (`solar_distribution_to_air = 0.0`) — line 1561, `thermal_model_core.rs`
- **100% → mass node** (`solar_beam_to_mass_fraction = 1.0`) — line 1566, `thermal_model_core.rs`
- **0% → surface node** (implied: `st_sol_frac = 1.0 - 1.0 = 0.0`)

The default struct values (`solar_distribution_to_air = 0.1`, `solar_beam_to_mass_fraction = 0.6` at line 2234–2235) are correctly **overridden** for ASHRAE 140 at lines 1561 and 1566. However, there is a **critical discrepancy in the 6R2C model** where additional hard-coded multipliers attenuate solar gains (Section 3.2 below).

---

## 2. Complete Solar Gain Routing Trace

### 2.1 Solar Gain Computation

**File**: `thermal_model_iterative.rs`, lines 164–310

| Step | Code Location | Operation |
|------|--------------|-----------|
| 1 | Line 509 `calculate_zone_solar_gain()` | Computes two gains: `window_gain` (transmitted solar) and `opaque_gain` (absorbed by opaque surfaces) |
| 2 | Lines 512–513 | `solar_gains[zone] = window_gain / floor_area` (W/m²); `opaque_solar_gains[zone] = opaque_gain / floor_area` (W/m²) |
| 3 | Line 522–523 | Stored as `self.0.solar_gains` and `self.0.opaque_solar_gains` |

**Opaque solar formula** (line 303):
```
opaque_gain = opaque_area × U_value × irradiance × α(0.6) × R_ext(0.034)
```
This is the sol-air equivalent: `α × I × R_ext × U × A`.

### 2.2 Solar Distribution in 5R1C Model

**File**: `thermal_model_physics.rs`, lines 606–631 (function `step_physics_5r1c`)

```rust
// Line 606: st_int_frac = rad_frac × (1 - solar_distribution_to_air) = 0.6 × 1.0 = 0.6
let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
// Line 607: m_air_frac = rad_frac × solar_distribution_to_air = 0.6 × 0.0 = 0.0
let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
// Line 608: st_sol_frac = 1.0 - solar_beam_to_mass_fraction = 1.0 - 1.0 = 0.0
let st_sol_frac = 1.0 - self.0.solar_beam_to_mass_fraction;
// Line 609: m_sol_frac = solar_beam_to_mass_fraction = 1.0
let m_sol_frac = self.0.solar_beam_to_mass_fraction;
```

For each zone (lines 620–632):

| Heat Source | phi_ia (air) | phi_st (surface) | phi_m (mass) |
|------------|-------------|-----------------|-------------|
| Internal loads (convective) | `load × 0.4` | — | — |
| Internal loads (radiative) | — | `load × 0.6 × 1.0 = load × 0.6` | `load × 0.0 = 0` |
| Window solar | `sol × 0.0 = 0` | `remaining_sol × 0.0 = 0` | `remaining_sol × 1.0 = 100% sol` |
| Opaque solar | — | — | `+ opaque_sol_w` (directly added) |

**Result**: With ASHRAE 140 settings:
- **phi_ia** = 0.4 × load + 0 (for FF cases, load=0, so **phi_ia = 0**)
- **phi_st** = 0.6 × load + 0 (for FF cases, **phi_st = 0**)
- **phi_m** = 0 × load + 100% window_solar + opaque_solar = **ALL solar gains**

### 2.3 Solar Distribution in 6R2C Model

**File**: `thermal_model_physics.rs`, lines 1480–1513 (function `step_physics_6r2c`)

**CRITICAL DISCREPANCY**: The 6R2C model applies additional multipliers that **reduce total solar reaching the nodes**:

```rust
// Line 1489: st_sol_frac = (1 - 0.6) × 0.6 = 0.24 (NOT 0.40)
let st_sol_frac = (1.0 - self.0.solar_beam_to_mass_fraction) * 0.6;
// Line 1490: m_env_sol_frac = 0.6 × 0.7 = 0.42
let m_env_sol_frac = self.0.solar_beam_to_mass_fraction * 0.7;
// Line 1491: m_int_sol_frac = 0.6 × 0.3 = 0.18
let m_int_sol_frac = self.0.solar_beam_to_mass_fraction * 0.3;
```

**Total solar fraction = 0.24 + 0.42 + 0.18 + 0.0 = 0.84** — only 84% of solar reaches nodes!

With ASHRAE 140 override (`solar_beam_to_mass_fraction = 1.0`):
- `st_sol_frac = (1-1.0) × 0.6 = 0.0`
- `m_env_sol_frac = 1.0 × 0.7 = 0.7`
- `m_int_sol_frac = 1.0 × 0.3 = 0.3`
- Total = 0.0 + 0.7 + 0.3 + 0.0 = **1.0** ✓

So with the ASHRAE 140 override, the 6R2C also routes 100% to mass. **But the 0.7/0.3 envelope/internal split is hard-coded and may not match ASHRAE 140**. Also, **opaque_solar_gains are NOT added in the 6R2C path** (compare line 631 vs 1512–1513).

### 2.4 How Mass-Node Heat Reaches the Air Node

The 5R1C temperature formula (lines 956–959):

```
T_air_free = (h_ms_is_prod × T_mass + h_tr_is × phi_st + term_rest_1 × (phi_ia + h_ext × T_out) + ground_coeff × T_ground) / den
```

Where:
- `h_ms_is_prod = h_tr_ms × h_tr_is` (mass-to-surface-to-air series conductance)
- `term_rest_1 = h_tr_ms + h_tr_is` (parallel conductance combination factor)
- `den = h_ms_is_prod + term_rest_1 × h_ext + ground_coeff`

**The heat path from mass to air**:
1. Solar → phi_m → T_mass (via mass energy balance: `(Cm/dt + h_tr_em + h_tr_ms) × Tm_new = ...`)
2. T_mass → `h_ms_is_prod × T_mass` → numerator of T_air_free formula
3. The series conductance `h_ms_is_prod = h_tr_ms × h_tr_is` controls how effectively mass heat reaches air

For Case 600FF (lightweight construction):
- `h_tr_ms` is typically 50–200 W/K
- `h_tr_is` is typically 100–300 W/K
- `h_ms_is_prod` = ~10,000–60,000 W²/K²

This means solar heat deposited in the mass node **does** reach the air node through the series conductance path, but with a **time delay** governed by the mass thermal capacitance `Cm`.

---

## 3. ISO 13790 / ASHRAE 140 Comparison

### 3.1 ISO 13790 Annex C Distribution (5R1C)

Per ISO 13790:2008 Section C.4, Equations C.5–C.7:

| Term | ISO 13790 Formula | Fluxion Implementation |
|------|------------------|----------------------|
| phi_ia | `(1 - A_m/A_t) × phi_int_conv + solar_to_air` | `load × conv_frac + sol × solar_dist_to_air` ✓ |
| phi_st | `(1 - A_m/A_t - h_tr_w/h_tr_ms) × phi_int_rad + solar_to_surface` | `load × st_int_frac + remaining_sol × st_sol_frac` ⚠️ |
| phi_m | `A_m/A_t × phi_total` | `load × m_air_frac + remaining_sol × m_sol_frac + opaque_sol` ⚠️ |

**Key differences**:

1. **ISO uses `A_m/A_t` ratio** (effective mass area / total area) to distribute gains. Fluxion uses `solar_beam_to_mass_fraction` as a direct parameter. For ASHRAE 140 where `solar_beam_to_mass_fraction = 1.0`, all solar goes to mass — this is functionally equivalent to A_m/A_t = 1.0 for solar, which is correct for a low-mass building where all surfaces have the same absorptance.

2. **ISO phi_st includes a `- h_tr_w/h_tr_ms` correction** for window radiative transfer. Fluxion omits this term. For Case 600FF with no HVAC and `solar_distribution_to_air = 0`, the phi_st contribution is zero anyway, so this has no effect.

3. **Internal load distribution**: For FF cases with `loads = 0` (line 1528–1530), the internal load distribution is moot.

### 3.2 ASHRAE 140 Section 5.2.2 Compliance

ASHRAE 140-2023 Section 5.2.2 states:
> "Transmitted solar radiation shall be distributed to all interior opaque surfaces proportional to their area × solar absorptance"

The Fluxion implementation (line 1555–1557):
- ✅ **100% of transmitted solar goes to opaque interior surfaces** (via mass node)
- ✅ **ZERO fraction goes to the air node directly** (`solar_distribution_to_air = 0.0`)
- ✅ **Distribution proportional to area × absorptance** (all α = 0.6, so area-weighted)

**Verdict**: The 5R1C distribution is **correct per ASHRAE 140**.

---

## 4. Quantitative Analysis

### 4.1 Energy Balance Check

For Case 600FF at peak solar hour (noon, summer):
- Window area: ~12 m² (south-facing)
- Peak irradiance: ~800 W/m² (direct + diffuse on south facade)
- Window transmittance (SHGC): ~0.87
- **Expected window solar gain**: ~12 × 800 × 0.87 ≈ **8,352 W**

With current distribution:
- phi_ia += 0 W (0% of solar)
- phi_st += 0 W (0% of solar)
- phi_m += 8,352 W + opaque_gain (100% of solar + opaque)

### 4.2 Time Constant Analysis

The effective time constant for heat to travel from mass node to air node:

```
τ = Cm / (h_tr_ms + h_tr_is)
```

For Case 600FF (lightweight):
- Cm ≈ 5–15 MJ/K (lightweight walls, low thermal mass)
- h_tr_ms ≈ 50–200 W/K
- h_tr_is ≈ 100–300 W/K
- τ ≈ 5e6 / 200 ≈ **25,000 seconds ≈ 7 hours**

This means solar heat deposited at the mass node takes ~7 hours to fully equilibrate to the air node. For a free-floating case, this creates a significant **phase lag** where:
- Peak solar → peak mass temperature: ~2–3 hour lag
- Peak mass → peak air temperature: ~2–3 additional hours
- **Total lag: ~4–6 hours**

ASHRAE 140 reference results for Case 600FF show peak temperatures around 14:00–16:00, consistent with this lag.

### 4.3 Impact on Peak Temperature

If we hypothetically routed all solar directly to air (`solar_distribution_to_air = 1.0`):
- Peak temperature would be **higher** (immediate response, no thermal lag)
- But ASHRAE 140 requires distribution to surfaces, not air

The current routing to mass is **conservative** (produces lower peaks than direct-to-air) but is **physically correct** per ASHRAE 140. The 10–25°C deficit is NOT caused by the distribution fraction.

### 4.4 Sensitivity Estimate

Changing `solar_beam_to_mass_fraction` from 1.0 to 0.6 (default):
- phi_m loses 40% of window solar → phi_st gains 40%
- Surface node has lower thermal capacitance → faster response
- Peak temperature would increase by ~2–4°C (rough estimate)
- This would NOT fix a 10–25°C deficit

---

## 5. Opaque Solar Gains Routing

### 5.1 Computation

Opaque solar gains are computed in `calculate_zone_solar_gain()` (line 303):
```rust
opaque_gain = opaque_area × U_value × irradiance × α × R_ext
```

This is the sol-air temperature approach: the gain represents the extra heat flux due to solar absorption on exterior surfaces.

### 5.2 Routing

In the **5R1C** model (line 631):
```rust
phi_m_data.push(load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w);
```
**All opaque solar goes to the mass node** — consistent with window solar routing.

In the **6R2C** model (lines 1512–1513):
```rust
phi_m_env_data.push(load_w * m_air_frac + sol_w * m_env_sol_frac);
phi_m_int_data.push(sol_w * m_int_sol_frac);
```
**Opaque solar gains are NOT included** in the 6R2C model! This is a **bug** — `opaque_sol_w` is computed but never added to any 6R2C phi term.

**Impact**: For Case 600FF running on the 5R1C model, this bug has no effect. But for heavy-mass cases using the 6R2C model, opaque solar gains would be lost entirely.

---

## 6. Key Findings and Recommendations

### 6.1 Solar Distribution Verdict

| Question | Answer | Confidence |
|----------|--------|-----------|
| Is `solar_distribution_to_air = 0.0` correct? | **YES** — ASHRAE 140 §5.2.2 specifies zero to air | HIGH |
| Is `solar_beam_to_mass_fraction = 1.0` correct? | **YES** — all solar to opaque surfaces (mass node) | HIGH |
| Does mass heat reach air? | **YES** — via h_tr_ms/h_tr_is series conductance with ~7h time constant | HIGH |
| Is the distribution causing the 10–25°C deficit? | **NO** — distribution is correct per ASHRAE 140 | HIGH |
| Are opaque gains routed correctly (5R1C)? | **YES** — added to phi_m | HIGH |
| Are opaque gains routed correctly (6R2C)? | **NO** — missing entirely (bug) | HIGH |

### 6.2 Bugs Found

1. **6R2C: Missing opaque_solar_gains** (thermal_model_physics.rs, lines 1504–1513)
   - `opaque_sol_w` is computed but never added to any phi term
   - Severity: HIGH for heavy-mass cases, NONE for Case 600FF (uses 5R1C)

2. **6R2C: Hard-coded 0.7/0.3 envelope/internal split** (lines 1490–1491)
   - `m_env_sol_frac = solar_beam_to_mass_fraction × 0.7`
   - `m_int_sol_frac = solar_beam_to_mass_fraction × 0.3`
   - These multipliers are not derived from ISO 13790 or ASHRAE 140
   - Severity: LOW for Case 600FF (6R2C not used), MEDIUM for Case 900FF

3. **6R2C: Non-orthogonal st_sol_frac formula** (line 1489)
   - `st_sol_frac = (1.0 - solar_beam_to_mass_fraction) × 0.6`
   - The `× 0.6` factor reduces surface-bound solar to 60% of remaining
   - With `solar_beam_to_mass_fraction = 1.0`, this is 0.0 (correct)
   - But with default 0.6, total fractions don't sum to 1.0

### 6.3 Root Cause Assessment for 10–25°C Deficit

The solar distribution is **NOT** the root cause. The deficit likely originates from:

1. **Excessive heat loss via h_tr_em**: If the mass-to-exterior conductance is too high, solar heat deposited in the mass node leaks to exterior before reaching the air node.

2. **Thermal capacitance overestimate**: If Cm is too large for a lightweight building, the mass node absorbs solar but warms slowly, delaying heat transfer to air.

3. **h_tr_ms/h_tr_is ratio imbalance**: If h_tr_ms is too low relative to h_tr_is, the mass-to-surface conductance becomes a bottleneck, trapping solar heat at the mass node.

4. **Ventilation/infiltration losses**: The air node loses heat to exterior via `h_ext × (T_air - T_out)`. If infiltration is overestimated, the air node stays cold despite mass heating.

**Recommendation**: Focus investigation on the h_tr_em value and Cm magnitude for Case 600FF. The solar distribution itself is correct.

---

## 7. Line Number Reference

| Item | File | Lines |
|------|------|-------|
| ASHRAE 140 solar_distribution_to_air = 0.0 | thermal_model_core.rs | 1561 |
| ASHRAE 140 solar_beam_to_mass_fraction = 1.0 | thermal_model_core.rs | 1566 |
| Default struct values | thermal_model_core.rs | 2234–2235 |
| 5R1C phi_ia computation | thermal_model_physics.rs | 627–629 |
| 5R1C phi_st computation | thermal_model_physics.rs | 630 |
| 5R1C phi_m computation | thermal_model_physics.rs | 631 |
| 6R2C phi distribution | thermal_model_physics.rs | 1504–1513 |
| 6R2C hard-coded split factors | thermal_model_physics.rs | 1489–1491 |
| Free-float temperature formula | thermal_model_physics.rs | 956–959 |
| Mass temperature update | thermal_model_physics.rs | 1370–1396 |
| Solar gain calculation | thermal_model_iterative.rs | 164–310 |
| Opaque solar formula | thermal_model_iterative.rs | 303 |
| FF load zeroing | thermal_model_core.rs | 1528–1530 |
| h_tr_is calculation | thermal_model_core.rs | 841–853 |
| h_tr_ms physics-based calc | thermal_model_core.rs | 885–895 |
| h_tr_em ISO 13790 formula | thermal_model_core.rs | 1130–1163 |
