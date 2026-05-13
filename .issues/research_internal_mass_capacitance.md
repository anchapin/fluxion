# Internal Mass (Furniture/Partitions) Thermal Capacitance Research

**Phase 6 Architecture**: T_me (internal mass node) for furniture and partitions
**Question 3**: How should internal mass capacitance be specified and computed?

---

## 1. EnergyPlus Internal Mass Approach

### EnergyPlus InternalMass Object
EnergyPlus models internal mass (furniture, partitions) using the `InternalMass` object:

```EnergyPlus
InternalMass,
    ZoneName,                  !- Zone Name
    Furniture,                 !- Name of internal mass surface group
    48.0,                      !- Design Surface Area {m²}
    GenericFurniture;          !- Construction Name (with thermal mass)
```

### Thermal Mass Calculation in EnergyPlus
EnergyPlus calculates internal thermal capacitance as:
```
C_int = Surface_Area × ThermMassPerUnitArea
```

Where `ThermMassPerUnitArea` defaults to **55,000 J/m²K** (per 2005 ASHRAE Handbook, Chapter 27):

| Component | Default Value |
|-----------|---------------|
| Furniture (generic) | 55,000 J/m²K |
| Partitions | 55,000 J/m²K |
| Equipment | 55,000 J/m²K |

For Case 900 (48 m² floor area):
- If furniture area = 48 m² (100% of floor - maximum case)
- C_int = 48 × 55,000 = **2.64e6 J/K**
- If furniture area = 24 m² (50% of floor - typical)
- C_int = 24 × 55,000 = **1.32e6 J/K**

### EnergyPlus Coupling to Air
EnergyPlus uses a fixed convective heat transfer coefficient for internal mass surfaces:
- h_int = 4.5 W/(m²·K) (similar to interior surface coefficient)
- Coupling to zone air node: Q = h_int × A_int × (T_surface - T_air)

---

## 2. Current Fluxion Implementation

### Internal Thermal Capacitance (C_me)
Location: `src/sim/thermal_model_solvers.rs:196`

```rust
self.0.internal_thermal_capacitance = total_cap * (1.0 - envelope_mass_fraction);
```

Where `envelope_mass_fraction` is typically **0.75** for high-mass buildings.

**Computed values for ASHRAE 140 cases:**

| Case | Total Cm (J/K) | Internal Cm (J/K) | Notes |
|------|---------------|-------------------|-------|
| Case 600 | ~2.4e6 | ~6.0e5 | Low-mass, 20% of total |
| Case 900 | ~1.2e7 | ~3.0e6 | High-mass, 20-25% of total |

### h_tr_me (Conductance: Envelope Mass → Internal Mass)
Location: `src/sim/thermal_model_core.rs:1119-1131`

**Current formula** (PHASE 36-04):
```rust
let a_int = 0.1 * zone_floor_area;  // Furniture surface area (~10% of floor area)
let h_ms = 4.5;                       // Furniture/partitions coupling coefficient W/(m²·K)
h_tr_me = h_ms * a_int
```

For Case 900 (48 m² floor):
- A_int = 0.1 × 48 = 4.8 m²
- h_tr_me = 4.5 × 4.8 = **21.6 W/K**

**Previous formula** (Issue 692, now superseded):
- A_int = 0.5 × floor_area → h_tr_me = 108 W/K
- Or A_int = 2.0 × floor_area → h_tr_me = 432 W/K

### Internal Mass Temperature Update
Location: `src/sim/thermal_model_physics.rs:1880-1888`

```rust
// Internal mass: receives heat from envelope mass (h_tr_me) and direct gains
// Physics: Cm * (Tm_int_new - Tm_int_old) / dt = h_tr_me * (Tm_env - Tm_int_new) + phi_m_int
// Rearranged: (Cm/dt + h_tr_me) * Tm_int_new = Cm/dt * Tm_int_old + h_tr_me * Tm_env + phi_m_int
let denom_int = cm_int / dt + h_tr_me;
tm_int_new = (cm_int / dt * tm_int_old + h_tr_me * tm_env_new + phi_m_int_zone) / denom_int;
```

---

## 3. ASHRAE 140 Requirements

### ASHRAE 140-2023 Internal Mass Specification

ASHRAE 140 does NOT explicitly specify internal mass furniture values for test cases. Instead:

1. **Case 600 series (lightweight)**: No additional internal mass specified - thermal mass comes from building envelope only
2. **Case 900 series (heavyweight)**: No additional internal mass specified - high mass comes from thick concrete walls, roof, floor

However, the standard implies internal thermal mass is included through:
- Table 4 (building description) - lists furniture for residential cases
- Implicit assumption: Typical office/housing furniture contributes ~55,000 J/m²K

### ASHRAE 140 Reference Values for Internal Mass

From ASHRAE 140-2023 Table 4 and Annex:
- **Residential cases**: Furniture load ~2.4 W/m² (small impact on heating/cooling)
- **Commercial cases**: Higher internal gains from equipment and furniture

The standard does NOT provide explicit C_int values for furniture - this is left to simulation programs to determine from building physics.

---

## 4. ISO 13790 Approach

### ISO 13790 Internal Heat Capacity (C_int)

ISO 13790 (EN ISO 13790:2008) Section 7.2 provides formulas for internal heat capacity:

```
C_int = Σ (ρ_i × c_i × V_i) for all internal surfaces
```

Where:
- ρ_i = density of material i (kg/m³)
- c_i = specific heat of material i (J/kg·K)
- V_i = volume of material i (m³)

### Default Values for Furniture (Informative Annex C)

ISO 13790 provides these indicative values for furniture/internal mass:

| Type | C_int (J/m²K) | Description |
|------|---------------|-------------|
| Light furniture | 20,000 - 30,000 | Wooden furniture, light partitions |
| Medium furniture | 40,000 - 60,000 | Heavy furniture, brick partitions |
| Heavy furniture | 80,000 - 120,000 | Stone, concrete, heavy equipment |

### Typical Assumption
For residential buildings with light furniture:
- A_furniture ≈ 0.3 × floor_area
- C_int_furniture ≈ 30,000 J/m²K
- Total C_int ≈ 0.3 × 30,000 = **9,000 J/m²** (of floor area)

For commercial buildings with heavy furniture:
- A_furniture ≈ 0.5 × floor_area
- C_int_furniture ≈ 60,000 J/m²K
- Total C_int ≈ 0.5 × 60,000 = **30,000 J/m²** (of floor area)

---

## 5. Recommended Formula for Phase 6

### Recommended Internal Thermal Capacitance

Based on EnergyPlus default (55,000 J/m²K) and ISO 13790 guidance:

```
C_me = A_floor × 55,000 × f_furniture
```

Where:
- A_floor = zone floor area (m²)
- f_furniture = furniture area factor (default 0.3 for residential, 0.5 for commercial)

**For Case 900:**
- A_floor = 48 m²
- f_furniture = 0.5 (commercial/institutional assumed)
- C_me = 48 × 55,000 × 0.5 = **1.32e6 J/K**

**For Case 600:**
- A_floor = 48 m²
- f_furniture = 0.3 (residential assumed)
- C_me = 48 × 55,000 × 0.3 = **7.92e5 J/K**

### Recommended h_tr_me (Envelope-to-Internal Mass Conductance)

Following EnergyPlus approach with interior surface convection coefficient:

```
h_tr_me = h_ms × A_int
h_ms = 4.5 W/(m²·K) (interior surface convection coefficient)
A_int = f_furniture × A_floor
```

Where:
- f_furniture = 0.3-0.5 (same as above)
- h_ms = 4.5 W/(m²·K)

**For Case 900 (f_furniture = 0.5):**
- A_int = 0.5 × 48 = 24 m²
- h_tr_me = 4.5 × 24 = **108 W/K**

**For Case 600 (f_furniture = 0.3):**
- A_int = 0.3 × 48 = 14.4 m²
- h_tr_me = 4.5 × 14.4 = **64.8 W/K**

### Time Constant Analysis

Internal mass time constant: τ_me = C_me / h_tr_me

**For Case 900:**
- C_me = 1.32e6 J/K
- h_tr_me = 108 W/K
- τ_me = 1.32e6 / 108 = **12,222 seconds ≈ 3.4 hours**

This gives internal mass a moderate time constant, consistent with furniture thermal mass behavior (faster response than heavy concrete walls).

### Sensitivity to Furniture Factor

| f_furniture | C_me (J/K) | h_tr_me (W/K) | τ_me (hours) |
|-------------|-----------|--------------|--------------|
| 0.2 | 5.28e5 | 43.2 | 3.4 |
| 0.3 | 7.92e5 | 64.8 | 3.4 |
| 0.5 | 1.32e6 | 108 | 3.4 |

**Note**: τ_me is independent of f_furniture because both C_me and h_tr_me scale with f_furniture.

---

## 6. Implementation Recommendations

### 1. Update Internal Thermal Capacitance Calculation

**Current** (in `configure_6r2c_model`):
```rust
self.0.internal_thermal_capacitance = total_cap * (1.0 - envelope_mass_fraction);
```

**Proposed** (in `from_spec`):
```rust
let furniture_factor = match spec.building_type {
    BuildingType::Residential => 0.3,
    BuildingType::Commercial => 0.5,
    BuildingType::Institutional => 0.5,
};
let c_me = zone_floor_area * 55_000.0 * furniture_factor;
```

### 2. Update h_tr_me Calculation

**Current** (line 1126):
```rust
let a_int = 0.1 * zone_floor_area;
```

**Proposed**:
```rust
let furniture_factor = 0.3;  // Could be made configurable per building type
let a_int = furniture_factor * zone_floor_area;
let h_ms = 4.5;  // Interior surface convection coefficient
let h_tr_me = h_ms * a_int;
```

### 3. Building Type Configuration

Add building type specification to `CaseSpec` or `ThermalModelSpec`:

```rust
pub enum BuildingType {
    Residential,   // f_furniture = 0.3
    Commercial,    // f_furniture = 0.5
    Institutional, // f_furniture = 0.5
}
```

### 4. Phase 6 Internal Mass Node Design

For Phase 6 multi-node thermal model, the internal mass node should:

1. **Receive**:
   - Heat from envelope mass via h_tr_me
   - Direct internal gains (people, equipment, lights) - fraction to internal mass
   - Solar gains (fraction via beam-to-mass distribution)

2. **Store**:
   - C_me thermal capacitance computed from furniture area
   - Typical: 1-2e6 J/K for residential zones

3. **Release**:
   - Heat to envelope mass (h_tr_me coupling)
   - Heat to interior air (via furniture surface convection)

---

## 7. Summary Table: Current vs Recommended

| Parameter | Current (Fluxion) | Recommended | EnergyPlus | ISO 13790 |
|-----------|------------------|------------|------------|------------|
| C_me formula | `total_cap * 0.25` | `A_floor × 55,000 × f_furn` | `A × 55,000` | `Σ ρcV` |
| C_me (Case 900) | ~3.0e6 J/K | ~1.32e6 J/K | ~2.64e6 | ~1-3e6 |
| h_tr_me formula | `h_ms × 0.1 × A_floor` | `h_ms × 0.3-0.5 × A_floor` | `h_ms × A_furn` | Not specified |
| h_tr_me (Case 900) | ~21.6 W/K | ~108 W/K | Variable | N/A |
| τ_me | ~38 hours (too slow) | ~3.4 hours | N/A | ~2-5 hours |

---

## 8. Next Steps for Phase 6

1. **Implement furniture factor** in `from_spec()` for C_me calculation
2. **Update h_tr_me** to use 0.3-0.5 × floor_area for furniture area
3. **Add building type** configuration to CaseSpec
4. **Validate** against ASHRAE 140 reference values for Case 600 and Case 900
5. **Test** thermal coupling behavior (internal mass should respond faster than envelope mass)

---

## References

1. ASHRAE 140-2023, "Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs"
2. EnergyPlus Engineering Reference, Section 4.2: Thermal Mass in Building Construction
3. ISO 13790:2008, "Energy performance of buildings - Calculation of energy use for space heating and cooling"
4. 2005 ASHRAE Handbook - Fundamentals, Chapter 27: Heat Transfer Theory
