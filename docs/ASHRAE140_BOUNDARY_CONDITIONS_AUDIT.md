# ASHRAE 140-2023 Boundary Conditions Audit

**Issue:** #727
**Standard:** ASHRAE Standard 140-2023 Annex B
**Status:** Complete

This document audits all hardcoded boundary condition values in fluxion against ASHRAE 140-2023 normative references.

---

## 1. Surface Heat Transfer Coefficients

### 1.1 Interior Combined Coefficients

ASHRAE 140 Section 5.2 specifies interior surface heat transfer coefficients. Fluxion uses three surface-type-specific values:

| Surface | Fluxion Constant | Value (W/m²K) | ASHRAE 140 Source | R_si (m²K/W) |
|---------|-----------------|---------------|-------------------|--------------|
| Wall (vertical) | `INTERIOR_FILM_COEFF_WALL` | 7.69 | Table B1-6 / Sec 5.2 | 0.13 |
| Ceiling (upward flow) | `INTERIOR_FILM_COEFF_CEILING` | 10.0 | Table B1-6 / Sec 5.2 | 0.10 |
| Floor (downward flow) | `INTERIOR_FILM_COEFF_FLOOR` | 5.88 | Table B1-6 / Sec 5.2 | 0.17 |
| Generic (combined) | `INTERIOR_FILM_COEFF` | 8.29 | Table B1-6 / Sec 5.2 | 0.1206 |

**ASHRAE 140-2023 Annex B requirement:** ASHRAE 140 uses **combined** coefficients (convective + radiative). The radiative component is implicit in the combined value.

**Code location:** `src/physics/constants/thermal/ashrae_140/v2023.rs:6-24`

```rust
pub const INTERIOR_FILM_COEFF: f64 = 8.29;           // Line 8
pub const INTERIOR_FILM_COEFF_WALL: f64 = 7.69;     // Line 16
pub const INTERIOR_FILM_COEFF_CEILING: f64 = 10.0;  // Line 20
pub const INTERIOR_FILM_COEFF_FLOOR: f64 = 5.88;   // Line 24
```

**Usage:** `src/sim/construction.rs:48-63` (`SurfaceType::interior_film_coeff()`)

---

### 1.2 Exterior Combined Coefficient

| Parameter | Fluxion Constant | Value | ASHRAE 140 Source |
|-----------|-----------------|-------|-------------------|
| Exterior at design wind (6.7 m/s) | `EXTERIOR_FILM_COEFF` | 29.3 W/m²K | Table B1-6 / Sec 5.2 |
| Exterior default | `EXTERIOR_FILM_COEFF_DEFAULT` | 29.3 W/m²K | Same |

**ASHRAE 140-2023 Annex B requirement:** ASHRAE 140 specifies h_ext = 29.3 W/m²K at the design wind speed of 6.7 m/s (15 mph).

**Wind speed correlation function:** `src/sim/construction.rs:86-91`
```rust
pub fn exterior_film_coeff(wind_speed: f64) -> f64 {
    10.0 + 4.0 * wind_speed.sqrt()  //ASHRAE correlation
}
```

**Code location:** `src/physics/constants/thermal/ashrae_140/v2023.rs:10-12, 26-28`

---

## 2. Material Properties

All material properties are defined in a single file:

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs`

This module is the **single source of truth** for ALL ASHRAE 140 material properties.

### 2.1 900-Series: Heavyweight Concrete Block Construction

ASHRAE 140 Table B1-3 (600-series uses similar materials with different assembly):

| Property | Fluxion Constant | Value | ASHRAE 140 Source |
|----------|-----------------|-------|-------------------|
| Thermal conductivity (k) | `HW_CONCRETE_K` | 0.51 W/mK | Table B1-3 |
| Density (ρ) | `HW_CONCRETE_RHO` | 1400 kg/m³ | Table B1-3 |
| Specific heat (Cp) | `HW_CONCRETE_CP` | 840 J/kgK | Table B1-3 |
| Thickness | `HW_CONCRETE_THICKNESS` | 0.200 m | Table B1-3 |
| Mass per unit area (κ) | `HW_CONCRETE_KAPPA` | 235,200 J/m²K | Computed |

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs:33-45`

### 2.2 Foam Board Insulation (900-series)

| Property | Constant | Value | ASHRAE 140 Source |
|----------|-----------|-------|-------------------|
| k | `FOAM_BOARD_K` | 0.040 W/mK | Table B1-3 |
| ρ | `FOAM_BOARD_RHO` | 10 kg/m³ | Table B1-3 |
| Cp | `FOAM_BOARD_CP` | 1400 J/kgK | Table B1-3 |
| Thickness | `FOAM_BOARD_THICKNESS` | 0.0615 m | Table B1-3 |

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs:47-58`

### 2.3 Wood Siding (600/900-series)

| Property | Constant | Value | ASHRAE 140 Source |
|----------|-----------|-------|-------------------|
| k | `WOOD_SIDING_K` | 0.14 W/mK | Table B1-3 |
| ρ | `WOOD_SIDING_RHO` | 530 kg/m³ | Table B1-3 |
| Cp | `WOOD_SIDING_CP` | 900 J/kgK | Table B1-3 |
| Thickness | `WOOD_SIDING_THICKNESS` | 0.009 m | Table B1-3 |

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs:60-71`

### 2.4 Fiberglass Batt Insulation (600-series)

| Property | Constant | Value | ASHRAE 140 Source |
|----------|-----------|-------|-------------------|
| k | `FIBREGLASS_BATT_K` | 0.040 W/mK | Table B1-3 |
| ρ | `FIBREGLASS_BATT_RHO` | 12 kg/m³ | Table B1-3 |
| Cp | `FIBREGLASS_BATT_CP` | 840 J/kgK | Table B1-3 |

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs:73-82`

### 2.5 Gypsum Board (600/900-series)

| Property | Constant | Value | ASHRAE 140 Source |
|----------|-----------|-------|-------------------|
| k | `GYPSUM_K` | 0.16 W/mK | Table B1-3 |
| ρ | `GYPSUM_RHO` | 784 kg/m³ | Table B1-3 |
| Cp | `GYPSUM_CP` | 840 J/kgK | Table B1-3 |
| Thickness | `GYPSUM_THICKNESS` | 0.012 m | Table B1-3 |

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs:84-95`

---

## 3. Internal Gains

### 3.1 Equipment Gains

ASHRAE 140 Table B1-7 specifies equipment schedule:

| Parameter | ASHRAE 140 Spec | Fluxion Implementation | Code Location |
|-----------|---------------|----------------------|---------------|
| Equipment power | 200 W continuous | `ComputerEquipment::new()` with `rated_power_w` | `src/sim/equipment.rs:50-59` |
| Day fraction (6:00-22:00) | 60% | Schedule fraction in `DailySchedule` | `src/sim/equipment.rs` |
| Night fraction (22:00-6:00) | 40% | Schedule fraction in `DailySchedule` | `src/sim/equipment.rs` |

**Validated by:** `tests/ashrae_140_input_validation/test_internal_gains.py`:
- `test_case_900_equipment_power` — validates 200W
- `test_case_900_equipment_schedule` — validates 60%/40% fractions
- `test_case_900_internal_loads_radiative_convective_split` — validates split

### 3.2 Sensible/Latent Split

ASHRAE 140 Table B1-7 does not specify explicit sensible/latent split for equipment (equipment is all sensible). Occupancy sensible/latent split is handled in `src/sim/occupancy.rs`:

**Code location:** `src/sim/occupancy.rs:79` (occupancy heat gains method)

---

## 4. Infiltration

### 4.1 Standard Conditioned Cases (600/900-series)

| Parameter | Value | ASHRAE 140 Source | Code Location |
|-----------|-------|-------------------|---------------|
| Infiltration rate | 0.5 ACH | Table B1-3 / Case spec | `src/validation/ashrae_140_cases.rs:1780` |

**Code:** `src/validation/ashrae_140_cases.rs` line 1780:
```rust
infiltration_ach: 0.5,
```

### 4.2 Free-Float Cases (600FF/900FF)

| Parameter | Value | ASHRAE 140 Source | Code Location |
|-----------|-------|-------------------|---------------|
| Infiltration rate | 0.0 ACH | Table B1-2 free-float definition | `src/validation/ashrae_140_cases.rs:2555` |

**Code:** `src/validation/ashrae_140_cases.rs` line 2555:
```rust
.with_infiltration(0.0) // No infiltration for free-float
```

### 4.3 Ventilation Model

Infiltration is applied as:
```
Q_vent = ρ × cp × (ACH / 3600) × V × ΔT
```

Where ρ × cp / 3600 ≈ 0.35 W/K·m³ (standard air properties).

**Code location:** `src/sim/construction.rs:1037-1055` (`VentilationConductance::new()`)

---

## 5. Ground Boundary Condition

### 5.1 Constant Ground Temperature Model

| Parameter | Value | ASHRAE 140 Source | Code Location |
|-----------|-------|-------------------|---------------|
| Ground temperature | 10°C | Table B1-2, Annex B | `src/sim/boundary.rs:113-147` |

**ASHRAE 140-2023 Annex B requirement:** Ground temperature is specified as a constant 10°C for baseline test cases.

**Code:**
```rust
// src/sim/boundary.rs
ConstantGroundTemperature::new(10.0)  // ASHRAE 140 default
```

**Related issue:** #680 (Ground coupling)

---

## 6. Building Geometry (ASHRAE 140 Table B1-1)

| Parameter | Constant | Value | ASHRAE 140 Source |
|-----------|-----------|-------|-------------------|
| E-W width | `BUILDING_WIDTH_M` | 8.0 m | Table B1-1 |
| N-S depth | `BUILDING_DEPTH_M` | 6.0 m | Table B1-1 |
| Height | `BUILDING_HEIGHT_M` | 2.7 m | Table B1-1 |
| Total wall area | `TOTAL_WALL_AREA_M2` | 75.6 m² | Computed |
| South window area | `SOUTH_WINDOW_AREA_M2` | 12.0 m² | Table B1-1 |
| Opaque wall area | `OPAQUE_WALL_AREA_M2` | 63.6 m² | Computed |
| Floor area | `FLOOR_AREA_M2` | 48.0 m² | Table B1-1 |

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs:98-114`

---

## 7. Window Properties

ASHRAE 140 Table B1-3 specifies double clear glass:

| Parameter | Constant | Value | ASHRAE 140 Source |
|-----------|-----------|-------|-------------------|
| Solar Heat Gain Coefficient | `WINDOW_SHGC` | 0.787 | Table B1-3 |
| Window U-value | `WINDOW_U_VALUE` | 3.0 W/m²K | Table B1-3 |

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs:124-127`

---

## 8. Surface Optical Properties

| Parameter | Constant | Value | ASHRAE 140 Source |
|-----------|-----------|-------|-------------------|
| Exterior solar absorptance | `EXTERIOR_SURFACE_ABSORPTANCE` | 0.6 | Table B1-3 (medium-color) |
| Surface long-wave emissivity | `SURFACE_EMISSIVITY` | 0.9 | Table B1-3 |

**Code location:** `src/physics/constants/thermal/ashrae_140/materials.rs:120-123`

---

## 9. Output Timestep Verification

| Output Type | ASHRAE 140 Requirement | Fluxion Implementation | Code Location |
|-------------|------------------------|------------------------|---------------|
| Annual energy | Jan 1 – Dec 31 (8760 hours) | Annual accumulation loop | `src/validation/ashrae_140_validator.rs` |
| Peak loads | Design-day peak (not annual peak) | Peak detection in `FreeFloatValidationResult` | `src/validation/diagnostic.rs` |

---

## Summary of Code Locations

| Category | Primary File | Key Constants |
|----------|--------------|--------------|
| Film coefficients | `src/physics/constants/thermal/ashrae_140/v2023.rs` | Lines 6-28 |
| Material properties | `src/physics/constants/thermal/ashrae_140/materials.rs` | Lines 1-156 |
| Construction assemblies | `src/sim/construction.rs` | `SurfaceType::interior_film_coeff()` at line 48 |
| Ground boundary | `src/sim/boundary.rs` | `ConstantGroundTemperature` at line 113 |
| Internal gains | `src/sim/equipment.rs`, `src/sim/profiles.rs` | Equipment trait at line 13 |
| Infiltration | `src/validation/ashrae_140_cases.rs` | Case definitions at lines 1437-1780 |
| Case specifications | `src/validation/ashrae_140_cases.rs` | Case builder at lines 1700+ |

---

## Related Issues

- #667 — [Phase C] Source true ASHRAE 140 reference data
- #680 — Ground coupling
- #672 — [Epic] v1.3 ASHRAE 140 Blind Validation
- #734 — Exterior film coefficient correction (6.7 m/s wind speed)
- #754 — Gypsum board density correction


---

## 9. Thermal Network Conductance Derivation (PR #821)

### 9.1 Surface-to-Mass Conductance `h_tr_ms`

**Standard:** ISO 13790:2008 §7.2.2.2 + Annex C.

The 5R1C lumped surface-to-mass coupling is computed as:

```
h_tr_ms = h_ms × A_m,    h_ms = 9.1 W/(m²·K)
A_m     = (Σ_j A_j κ_j)² / (Σ_j A_j κ_j²)
C_m     = Σ_j A_j κ_j
```

Where j indexes mass-bearing opaque elements (walls, roof, floor) and κ_j is
the construction's specific thermal capacitance per area (J/m²K) — taken from
`Construction::thermal_capacitance_per_area()` for consistency with how
`wall_cap`/`roof_cap`/`floor_cap` feed `C_m` in the same code block (Issue #585).

**Code:** `src/sim/thermal_model_core.rs:880-955` (lumped path used by 5R1C/6R2C).

The per-surface `h_tr_ms_wall` / `h_tr_ms_roof` / `h_tr_ms_floor` vectors
**continue to use the half-insulation-rule conduction values** because the
9R4C topology (Issue #715) dedicates one mass node per surface and would
double-count the ISO 13790 lumped correction.

### 9.2 Lumped `h_tr_em` — Floor Excluded

The floor's heat path to its boundary (ground) is captured by the dedicated
`h_tr_floor` node in the heat balance. The legacy code *also* summed
`h_tr_em_floor` (= `floor_U × A_floor`) into the lumped `h_tr_em`, which
`update_optimization_cache` then partially injects into `derived_h_ext`.

That double-counted ~9 W/K of phantom air-to-outdoor conductance for Case 600
and biased peak free-float air temperature low by ~3 °C. PR #821 removes the
floor from the lumped `h_tr_em`:

```rust
// src/sim/thermal_model_core.rs (Probe A+B)
let h_tr_em_total = h_tr_em_physics + h_tr_em_roof;  // floor lives in h_tr_floor
```

The 9R4C per-surface `h_tr_em_floor` value is preserved.

### 9.3 South-Wall Bypass (`h_south_series`)

The Issue #715 "south wall thermal bypass" added
`h_south_series = h_is_south × h_em_south / (h_is_south + h_em_south)` to
`derived_h_ext` as an extra parallel air↔outdoor path for the south wall.
With the legacy small `h_tr_ms` (~120 W/K) this was a partial compensation,
but the south wall was already in `h_tr_em` (mass side) and `h_tr_is` (air
side). Once `h_tr_ms` is restored to its ISO 13790 value (~1340 W/K), the
bypass becomes a 9 W/K double-count. PR #821 drops it from `derived_h_ext`.

The dedicated south-wall vectors (`h_tr_em_south`, `h_tr_is_no_south`) are
**kept** on the model so the 9R4C / CTF code paths owned by Issues #715 and
#730 continue to function unchanged.

### 9.4 Time Constant Derivation (Probe H)

`estimate_time_constant_hours` previously consulted a hard-coded τ table per
ASHRAE 140 case identifier (`TimeConstantAnalyzer::for_case(&case_id)`).
That breaks blind validation (the solver must not key off `case_id`) and can
disagree with the actual `Cm / h_tr_ms` after PR #821's conductance change.
PR #821 restores the physical derivation:

```
τ_hours = (Σ_zones C_m) / (3600 × Σ_zones h_tr_ms)
```

`TimeConstantAnalyzer::for_case` is still in the codebase for callers that
explicitly want the legacy table, but it is no longer consulted by the solver.

### 9.5 Orphaned ISO `h_tr_ms` Validation Files (Removed in #823)

PR #818 (issue #583) added two validation files to `tests/validation/`:

- `h_tr_ms_iso_regression_test.rs`
- `iso_h_tr_ms_comprehensive_validation.rs`

These files were **never compiled**: Cargo's integration-test discovery
only scans `tests/*.rs` (top-level), not subdirectories, and no parent
file declared them as modules. Wiring them in (via a top-level proxy)
revealed 16 syntactically invalid format strings (`{:.4f}` instead of
Rust's `{:.4}`), confirming they had never been built.

Their docstrings encoded a different formula than the one used by the
simulator:

```text
h_tr_ms = C_m / τ      with target τ = 15 h (light) / 150 h (heavy)
```

Those numbers correspond to ISO 13790 Table 9's *building* time
constant `τ = C_m / H_tr` (envelope total), not the surface-to-mass
coupling `h_ms` defined in §7.2.2.2 Eq. 65. Keeping the contradictory
text in-tree was a maintenance hazard for any future agent or
contributor reading the directory.

PR #823 removed both files. The canonical formula `h_ms = 9.1 × A_m`
(documented above in §9.1) is enforced by the live ASHRAE 140
free-float regression tests in `tests/ashrae_140_case_600_series.rs`,
which actually run in CI.
