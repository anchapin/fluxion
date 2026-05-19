# Issue #859: Per-Surface Boundary Conditions and Gain Distribution

**Status**: Research Complete
**Date**: 2025-05-17
**Author**: Backend Specialist (oma-backend)

---

## 1. Current Gain Distribution Mechanism

### 1.1 Data Structures

Solar and internal gains are stored as **per-zone scalar fields** (not per-surface):

| Field | Type | Location | Description |
|-------|------|----------|-------------|
| `solar_gains` | `T` (VectorField) | `thermal_model_data.rs:34` | Window/transparency solar gain per zone (W/m² of floor area) |
| `opaque_solar_gains` | `T` (VectorField) | `thermal_model_data.rs:35` | Opaque surface solar gain per zone (W/m² of floor area) |
| `loads` | `T` (VectorField) | `thermal_model_data.rs:33` | Internal loads per zone (W/m² of floor area) |
| `surfaces` | `Vec<Vec<WallSurface>>` | `thermal_model_data.rs:36` | Per-zone → per-surface geometry (area, window_area, u_value, orientation) |

### 1.2 Solar Gain Computation

**File**: `thermal_model_iterative.rs:164-310`

`calculate_zone_solar_gain()` computes gains per-zone by iterating over surfaces:

1. Groups surfaces by orientation to avoid double-counting (line 191)
2. Calls `calculate_hourly_solar()` per unique orientation (line 258)
3. Distributes window gain by `area_ratio = win_area / total_win_area` (line 290-291)
4. Computes opaque gain as `opaque_area × u_value × irradiance × α × R_ext` (line 302-303)
5. Returns **aggregated totals**: `(total_window_gain, total_opaque_gain)` for the zone (line 309)

The totals are then divided by floor area and stored in `solar_gains[i]` and `opaque_solar_gains[i]` (`thermal_model_iterative.rs:512-513`).

**Key finding**: Solar gains ARE computed per-surface internally (with correct per-orientation irradiance and per-surface area ratios), but the results are **aggregated into a single per-zone scalar** before being consumed by the thermal model.

### 1.3 Phi Term Computation (5R1C Model)

**File**: `thermal_model_physics.rs:555-608`

The 5R1C `step_physics` function computes three phi terms per zone:

```rust
// Line 556-581: Fraction definitions
let conv_frac = self.0.convective_fraction;         // e.g. 0.4
let rad_frac = 1.0 - conv_frac;                     // e.g. 0.6
let st_int_frac = rad_frac * (1.0 - solar_distribution_to_air);
let m_air_frac = rad_frac * solar_distribution_to_air;
let st_sol_frac = 1.0 - solar_beam_to_mass_fraction;
let m_sol_frac = solar_beam_to_mass_fraction;

// Line 592-603: Per-zone phi computation
for i in 0..num_zones {
    let load_w = loads_ref[i] * area_ref[i];         // W
    let sol_w = solar_ref[i] * area_ref[i];          // W
    let opaque_sol_w = opaque_solar_ref[i] * area_ref[i]; // W
    let sol_to_air = sol_w * solar_distribution_to_air;
    let remaining_sol = sol_w - sol_to_air;

    phi_ia = load_w * conv_frac + sol_to_air;
    phi_st = load_w * st_int_frac + remaining_sol * st_sol_frac;
    phi_m  = load_w * m_air_frac  + remaining_sol * m_sol_frac + opaque_sol_w;
}
```

**Distribution summary for 5R1C**:
- `phi_ia` (air node): convective internal loads + solar-to-air fraction
- `phi_st` (surface node): radiative internal loads × st_int_frac + remaining solar × st_sol_frac
- `phi_m` (mass node): radiative internal loads × m_air_frac + remaining solar × m_sol_frac + **all opaque solar**

### 1.4 Phi Term Computation (6R2C Model)

**File**: `thermal_model_physics.rs:1428-1485`

The 6R2C model has four phi terms with different solar splitting:

```rust
let st_sol_frac = (1.0 - solar_beam_to_mass_fraction) * 0.6;
let m_env_sol_frac = solar_beam_to_mass_fraction * 0.7;
let m_int_sol_frac = solar_beam_to_mass_fraction * 0.3;
let sol_to_air_frac = solar_distribution_to_air;

phi_ia     = load_w * conv_frac + sol_w * sol_to_air_frac;
phi_st     = load_w * st_int_frac + sol_w * st_sol_frac;
phi_m_env  = load_w * m_air_frac + sol_w * m_env_sol_frac;
phi_m_int  = sol_w * m_int_sol_frac;
```

### 1.5 Area-Weighted Distribution (Exists but Unused in Main Path)

**File**: `thermal_model_iterative.rs:326-375`

`calculate_area_weighted_radiative_distribution()` computes A_m/A_t ratio:
- `a_m = h_tr_ms / 9.1` (ISO 13790 Eq. 7)
- `f_m = (a_m / a_at).min(1.0)` — fraction to mass
- `f_st = (1 - f_m - h_tr_w / (9.1 × A_t)).max(0.0)` — fraction to surface

**This function exists but is NOT called** from the main `step_physics_5r1c` or `step_physics_6r2c` hot paths. The main paths use the fixed fraction approach (lines 578-581 / 1446-1459).

---

## 2. ISO 13790 Requirements vs Current Gaps

### 2.1 ISO 13790 Section C.4 Gain Distribution Rules

Per ISO 13790 (Section 7.2.2.2 and Section C.4), the correct distribution for the 5R1C (crmonthly) method:

| Term | ISO 13790 Formula | Description |
|------|-------------------|-------------|
| `φ_ia` | `0.5 × φ_int` | Half of internal gains → air node (convective) |
| `φ_st` | `A_m/A_t × (0.5 × φ_int + φ_sol)` | Internal radiative + solar → surface node |
| `φ_m` | `A_m/A_t × (0.5 × φ_int + φ_sol)` | Internal radiative + solar → mass node |

Where:
- `A_t` = total internal surface area of all surfaces facing the zone
- `A_m` = effective mass area = `h_tr_ms / 9.1` (from ISO 13790 Eq. 43)
- `φ_int` = total internal heat gains (W)
- `φ_sol` = total solar gains (W)

### 2.2 Gap Analysis

| Aspect | ISO 13790 | Current Code | Status |
|--------|-----------|--------------|--------|
| Internal gain convective split | `0.5 × φ_int` to air | `load_w × conv_frac` (conv_frac ≈ 0.4) | **PARTIAL** — uses configurable fraction, not fixed 0.5 |
| A_m/A_t ratio in phi_st | `A_m/A_t × (0.5 × φ_int + φ_sol)` | Fixed `st_int_frac` and `st_sol_frac` | **GAP** — A_m/A_t not used in main phi path |
| A_m/A_t ratio in phi_m | `A_m/A_t × (0.5 × φ_int + φ_sol)` | Fixed `m_air_frac` and `m_sol_frac` | **GAP** — A_m/A_t not used in main phi path |
| Per-surface solar distribution | Each surface gets its share | All solar lumped into single zone φ_sol | **GAP** — no per-surface phi terms |
| Opaque solar gain routing | Part of `φ_sol` → distributed via A_m/A_t | Routed entirely to `phi_m` (5R1C line 603) | **GAP** — opaque solar bypasses A_m/A_t split |

### 2.3 Critical Observations

1. **A_m/A_t ratio not used**: The code computes `h_tr_ms` (which encodes A_m via `h_tr_ms = 9.1 × A_m`) and has surfaces with individual areas, but the phi computation uses fixed global fractions (`solar_beam_to_mass_fraction`, `solar_distribution_to_air`) rather than the ISO 13790 A_m/A_t ratio.

2. **Existing but dead code**: `calculate_area_weighted_radiative_distribution()` at line 326 correctly implements A_m/A_t distribution, but is never called from the thermal physics hot path.

3. **Per-surface data available but aggregated**: `calculate_zone_solar_gain()` computes gains per-surface/orientation, then immediately sums them. The per-surface breakdown is lost.

---

## 3. Per-Surface Fields Currently Available

### 3.1 Data Structures (`thermal_model_data.rs`)

| Field | Type | Lines | Per-Surface? | Description |
|-------|------|-------|--------------|-------------|
| `surfaces` | `Vec<Vec<WallSurface>>` | 36 | **YES** | `surfaces[zone][surface_idx]` — area, window_area, u_value, orientation |
| `h_tr_ms` | `T` (per-zone) | 85 | No | Combined mass-surface conductance |
| `h_tr_ms_wall` | `Option<T>` | 154 | No (per-zone) | Wall mass conductance (9R4C) |
| `h_tr_ms_roof` | `Option<T>` | 155 | No (per-zone) | Roof mass conductance (9R4C) |
| `h_tr_ms_floor` | `Option<T>` | 156 | No (per-zone) | Floor mass conductance (9R4C) |
| `h_tr_em_wall` | `Option<T>` | 157 | No (per-zone) | Wall external-mass conductance |
| `h_tr_em_roof` | `Option<T>` | 158 | No (per-zone) | Roof external-mass conductance |
| `h_tr_em_floor` | `Option<T>` | 159 | No (per-zone) | Floor external-mass conductance |
| `cm_wall` | `Option<T>` | 161 | No (per-zone) | Wall thermal capacitance |
| `cm_roof` | `Option<T>` | 162 | No (per-zone) | Roof thermal capacitance |
| `cm_floor` | `Option<T>` | 163 | No (per-zone) | Floor thermal capacitance |
| `solar_gains` | `T` | 34 | No | Aggregated window solar per zone (W/m²) |
| `opaque_solar_gains` | `T` | 35 | No | Aggregated opaque solar per zone (W/m²) |

### 3.2 WallSurface Structure (`construction.rs:106-161`)

```rust
pub struct WallSurface {
    pub area: f64,           // Total surface area (m²)
    pub window_area: f64,    // Window area (m²)
    pub u_value: f64,        // U-value (W/m²K)
    pub orientation: Orientation,
    pub overhang: Option<Overhang>,
    pub fins: Vec<ShadeFin>,
}
```

### 3.3 What's Missing for Per-Surface Distribution

1. **Per-surface solar gain storage**: No `Vec<Vec<f64>>` for per-surface window solar gain or opaque solar gain
2. **Per-surface A_m values**: No `a_m_per_surface` array (A_m for wall vs. roof vs. floor)
3. **Per-surface phi terms**: No `phi_st_per_surface` or `phi_m_per_surface` vectors
4. **Surface type tagging**: `WallSurface` has no `surface_type` field (wall/roof/floor) — orientation is used as a proxy (`Orientation::Down` → floor)

---

## 4. Recommended Implementation Plan

### Phase 1: Add Per-Surface Solar Gain Storage

**Files to modify**: `thermal_model_data.rs`, `thermal_model_iterative.rs`

1. Add new fields to `ThermalModelData`:
   ```rust
   /// Per-surface window solar gains: [zone][surface_idx] in Watts
   pub surface_solar_gains: Vec<Vec<f64>>,
   /// Per-surface opaque solar gains: [zone][surface_idx] in Watts
   pub surface_opaque_gains: Vec<Vec<f64>>,
   ```

2. Modify `calculate_zone_solar_gain()` to return per-surface results:
   - Change return type from `(f64, f64)` to `(Vec<f64>, Vec<f64>)`
   - Store individual surface contributions instead of summing
   - Keep the existing aggregation for backward compatibility

3. Update clone/initialization in `ThermalModelData`.

**Estimated lines**: ~50 lines changed

### Phase 2: Compute Per-Surface A_m/A_t Ratio

**Files to modify**: `thermal_model_data.rs`, `thermal_model_core.rs`

1. Add surface-type enum or field to `WallSurface`:
   ```rust
   pub surface_type: SurfaceType, // Wall, Roof, Floor, Internal
   ```

2. Compute per-surface-type A_m:
   ```rust
   /// Per-surface effective mass area: [zone][surface_idx]
   pub a_m_per_surface: Vec<Vec<f64>>,
   /// Total internal surface area per zone: [zone]
   pub a_t_per_zone: Vec<f64>,
   ```

3. Derive from existing `h_tr_ms_wall/roof/floor`:
   - `a_m_wall = h_tr_ms_wall[zone] / 9.1`
   - `a_m_roof = h_tr_ms_roof[zone] / 9.1`
   - `a_m_floor = h_tr_ms_floor[zone] / 9.1`

**Estimated lines**: ~80 lines changed

### Phase 3: Implement Per-Surface Phi Distribution

**Files to modify**: `thermal_model_physics.rs`

1. Replace the current fixed-fraction phi computation (lines 592-603 for 5R1C, 1470-1480 for 6R2C) with ISO 13790 compliant per-surface distribution:

   ```rust
   // Per ISO 13790 Section C.4:
   // phi_st_surface[j] = (a_m_surface[j] / a_t) × (0.5 × phi_int + phi_sol_surface[j])
   // phi_m_surface[j]  = (a_m_surface[j] / a_t) × (0.5 × phi_int + phi_sol_surface[j])
   ```

2. Aggregate per-surface phi terms back to zone-level:
   ```rust
   phi_st = sum(phi_st_surface[j] for all surfaces j)
   phi_m  = sum(phi_m_surface[j] for all surfaces j)
   ```

3. Add a feature flag `per-surface-gains` for gradual rollout.

**Estimated lines**: ~120 lines changed

### Phase 4: Wire Into Multi-Node Models

**Files to modify**: `thermal_model_physics.rs`, `multi_node_thermal.rs`

1. For the 9R4C model, route per-surface phi terms directly to individual surface mass nodes:
   - `phi_st_wall`, `phi_st_roof`, `phi_st_floor`
   - `phi_m_wall`, `phi_m_roof`, `phi_m_floor`

2. Use existing `multi_node_solvers` and per-surface mass temperatures.

**Estimated lines**: ~100 lines changed

### Summary of Functions to Modify

| Function | File | Lines | Change |
|----------|------|-------|--------|
| `ThermalModelData` struct | `thermal_model_data.rs` | 30-182 | Add per-surface gain fields |
| `ThermalModelData::clone()` | `thermal_model_data.rs` | 184-328 | Clone new fields |
| `calculate_zone_solar_gain()` | `thermal_model_iterative.rs` | 164-310 | Return per-surface gains |
| `calc_analytical_loads()` | `thermal_model_iterative.rs` | 498-538 | Store per-surface results |
| `step_physics_5r1c()` phi block | `thermal_model_physics.rs` | 555-608 | Use A_m/A_t per-surface |
| `step_physics_6r2c()` phi block | `thermal_model_physics.rs` | 1428-1485 | Use A_m/A_t per-surface |
| `WallSurface` struct | `construction.rs` | 106-161 | Add surface_type field |

---

## 5. Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **ASHRAE 140 regression** | Changing phi distribution affects ALL validation cases (600, 650, 900, 950) | Run full ASHRAE 140 suite before/after; expect <0.5°C deviation tolerance |
| **Energy conservation violation** | Incorrect A_m/A_t normalization could add/remove energy | Add energy balance assertion: `phi_ia + phi_st + phi_m ≡ total_loads + total_solar` |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Feature flag complexity** | Dual code paths increase maintenance | Use a compile-time Cargo feature flag; plan to remove old path after validation |
| **9R4C model coupling** | Per-surface phi may interact with multi-node solver state | Test 9R4C model separately; the existing multi_node_solvers already handle per-surface temperatures |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Clone overhead** | Additional `Vec<Vec<f64>>` cloning per timestep | Gains are computed once per timestep; overhead is negligible vs. solar calculation |
| **API surface change** | `calculate_zone_solar_gain` return type changes | Internal function; no public API break |

### Dependency Map

```
WallSurface.surface_type ──► Phase 2 (A_m/A_t per surface)
                                   │
                                   ▼
calculate_zone_solar_gain() ──► Phase 1 (per-surface solar storage)
                                   │
                                   ▼
                           Phase 3 (per-surface phi in 5R1C/6R2C)
                                   │
                                   ▼
                           Phase 4 (9R4C multi-node wiring)
```

Phases 1 and 2 can proceed in parallel. Phase 3 depends on both. Phase 4 depends on Phase 3.

---

## Appendix A: Key File Locations

| File | Purpose | Size |
|------|---------|------|
| `src/sim/thermal_model_physics.rs` | Phi computation, physics stepping | 2622 lines |
| `src/sim/thermal_model_data.rs` | Data structures | 328 lines |
| `src/sim/thermal_model_iterative.rs` | Solar gain calculation | 860 lines |
| `src/sim/construction.rs` | WallSurface struct | 2303 lines |
| `src/sim/thermal_model_core.rs` | Model initialization | ~2000 lines |

## Appendix B: ISO 13790 Equations Referenced

- **Eq. 43**: `h_tr_ms = 9.1 × A_m` (mass-surface conductance)
- **Eq. C.5**: `φ_st = (1 - F_sup) × φ_int,rad` (radiative to surface)
- **Eq. C.6**: `φ_ia = F_sup × φ_int,rad` (radiative to air)
- **Section 7.2.2.2**: `φ_m = A_m/A_t × (0.5 × φ_int + φ_sol)` (gain to mass node)
- **Section C.3**: `A_t` = sum of all internal-facing surface areas
