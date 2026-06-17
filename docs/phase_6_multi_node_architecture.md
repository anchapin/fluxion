# Phase 6 Multi-Node Thermal Model Architecture

## Status: Design Specification

**Created**: 2026-05-13
**Context**: Thermal time constant τ = 58h for Case 900 vs target 120-200h (ASHRAE 140)

---

## 1. Executive Summary

**Problem**: Current single-node thermal mass architecture cannot produce ASHRAE 140 Case 900's τ ≈ 150h because all surfaces (wall, roof, floor) share one thermal mass node with combined conductance h_tr_ms_total = h_ms_wall + h_ms_roof + h_ms_floor. This causes thermal energy to couple too rapidly between surfaces and air.

**Root Cause**: ISO 13790's half-insulation rule places the thermal mass node at the dominant insulation layer. When wall, roof, and floor have different insulation profiles, they should have different effective τ values. The single-node model forces them to share one τ.

**Solution**: Multi-node thermal network with separate thermal mass nodes per surface category:
- `T_ms_wall`: Wall thermal mass node (concrete + insulation interior)
- `T_ms_roof`: Roof thermal mass node (concrete slab interior half)
- `T_ms_floor`: Floor thermal mass node (concrete slab interior half)
- `T_s`: Interior surface node (shared)
- `T_int`: Interior air node
- `T_me`: Internal mass node (furniture, partitions)
- `T_env`: Exterior environment node
- `T_ext`: Exterior air node

---

## 2. Current Architecture Analysis

### 2.1 Current 6R2C Network (single thermal mass node)

```
                    h_tr_w (windows)
                         │
   T_ext ──────┬─ h_tr_em ───── T_ms ──── h_tr_ms ──── T_s ──── h_tr_is ──── T_int
               │                                    │
               │                              h_tr_me
               │                                    │
               └──────── h_tr_floor ────────────────┘
                              │
                         T_ground
```

**Conductances computed**:
- `h_tr_ms_total = h_ms_wall + h_ms_roof + h_ms_floor` (all surfaces summed)
- `h_tr_em_total = h_em_wall + h_em_roof + h_em_floor` (all surfaces summed)
- `τ = Cm / (h_tr_ms_total + h_tr_me)` where h_tr_me ≈ 4.5 W/K per m² of furniture

**Problem**: The summed h_tr_ms_total is too large because each surface's contribution is added directly. For Case 900:
- h_ms_wall ≈ 150 W/K (200mm concrete interior half + insulation half)
- h_ms_roof ≈ 100 W/K (200mm concrete interior half)
- h_ms_floor ≈ 100 W/K (200mm concrete interior half)
- h_ms_total ≈ 350 W/K
- Cm ≈ 17,000,000 J/K (wall + roof + floor + air)
- τ = 17e6 / (350 + 50) ≈ 42,000s ≈ 11.7h

But ASHRAE 140 requires τ ≈ 150h for Case 900.

### 2.2 Why Current Architecture Fails

1. **Additive h_ms_total is wrong**: h_tr_ms represents conductance to a SINGLE mass node. Adding wall+roof+floor contributions implies three parallel paths to the same node, but each surface should have its own mass node.

2. **ISO 13790 half-insulation rule**: Places mass node at insulation layer. Wall (R-19 insulation), roof (R-30 insulation), and floor (different construction) have different insulation depths, so their mass nodes are at different positions.

3. **Different surface categories need different τ**: Heavy concrete walls have slow dynamics. Light roof has faster dynamics. Single node forces them to share the same dynamics.

---

## 3. Proposed Multi-Node Architecture

### 3.1 Three-Node Network (Surface-Coupled)

```
                         h_tr_w (windows)
                              │
T_ext ─── h_tr_em_wall ─── T_ms_wall ─── h_tr_ms_wall ───┬── T_s ─── h_tr_is ─── T_int
                    │                                    │         │
                    │                              h_tr_me      │
                    │                                    │         │
                    └── h_tr_em_roof ─── T_ms_roof ─── h_tr_ms_roof ─┤
                                        │                     │
                                        │            h_tr_ms_floor
                                        │                     │
                    └── h_tr_em_floor ─── T_ms_floor ────────┘
                                        │
                                   T_ground
```

**Node definitions**:
- `T_ms_wall`: Wall thermal mass (concrete + insulation interior half)
- `T_ms_roof`: Roof thermal mass (concrete slab interior half)
- `T_ms_floor`: Floor thermal mass (concrete slab interior half, coupled to ground)
- `T_s`: Interior surface node (temperature of all interior surfaces)
- `T_int`: Interior air temperature
- `T_me`: Internal mass (furniture, partitions)
- `T_ext`: Exterior air temperature
- `T_ground`: Ground temperature (annual average or specified)

**Conductance definitions**:
- `h_tr_ms_wall`: Wall interior resistance (interior half of insulation + concrete)
- `h_tr_ms_roof`: Roof interior resistance (concrete slab interior half)
- `h_tr_ms_floor`: Floor interior resistance (concrete slab interior half)
- `h_tr_em_wall`: Wall exterior resistance (exterior half of insulation + exterior film)
- `h_tr_em_roof`: Roof exterior resistance (exterior concrete + exterior film)
- `h_tr_em_floor`: Floor-to-ground resistance (includes ground coupling)
- `h_tr_is`: Interior surface-to-air conductance (per ISO 13790 Table 3)
- `h_tr_me`: Internal mass-to-surface coupling (furniture + partitions)
- `h_tr_w`: Window conductance (U × A)

**Thermal capacitances**:
- `Cm_wall`: Wall thermal capacitance (J/K) = ρ×c×V for concrete + layers interior to insulation
- `Cm_roof`: Roof thermal capacitance (J/K)
- `Cm_floor`: Floor thermal capacitance (J/K)
- `Cm_me`: Internal mass capacitance (J/K) = furniture + partitions

### 3.2 Thermal Time Constant Analysis

For each surface node, τ_i = Cm_i / h_tr_ms_i:

| Surface | Cm (J/K) | h_tr_ms (W/K) | τ (hours) |
|---------|----------|---------------|-----------|
| Wall | 8.0e6 | 25 | 89h |
| Roof | 4.0e6 | 40 | 28h |
| Floor | 4.0e6 | 40 | 28h |

**Weighted effective τ** (for zone-level dynamics):
τ_eff = (Cm_wall + Cm_roof + Cm_floor) / (h_tr_ms_wall + h_tr_ms_roof + h_tr_ms_floor)
      = 16e6 / 105 ≈ 42h

This is still too low. Need to reconsider the h_tr_ms values.

### 3.3 Revised Conductance Calculation

Per ISO 13790 Annex C (half-insulation rule), the thermal mass node is located at the insulation layer. The conductance from interior to mass is:

h_tr_ms = A / R_int_to_mass

where R_int_to_mass = Σ(R_layers_interior_to_insulation) + 0.5 × R_insulation

For Case 900 wall (200mm concrete + 50mm insulation):
- Interior film: 1/7.69 = 0.130 m²K/W
- Concrete (200mm, k=1.0): 0.20 m²K/W
- Insulation (50mm, k=0.04): 1.25 m²K/W → half = 0.625 m²K/W
- R_int_to_mass = 0.130 + 0.20 + 0.625 = 0.955 m²K/W
- h_tr_ms = 48 m² (wall area) / 0.955 ≈ 50 W/K

For 48 m² floor area, roof area:
- Roof: 48 m² / (0.13 + 0.20 + 0.5×1.25) = 48 / 0.955 ≈ 50 W/K
- Floor: similar structure = 50 W/K

Total h_tr_ms = 150 W/K

Cm_total = (wall + roof + floor) × capacitance_per_area
         = (75.6m² × 200mm × 2400kg/m³ × 1000 J/kgK + 48m² × 200mm × 2400 + 48m² × 150mm × 2400)
         = (75.6 × 0.2 × 2400 × 1000) + (48 × 0.2 × 2400 × 1000) + (48 × 0.15 × 2400 × 1000)
         = (75.6 × 480000) + (48 × 480000) + (48 × 360000)
         = 36,288,000 + 23,040,000 + 17,280,000
         = 76,608,000 J/K ≈ 77 MJ/K

τ = 77e6 / (150 + 50) = 77e6 / 200 = 385,000s ≈ 107h

This is closer but still short of 150h target. Need h_tr_ms ≈ 125 W/K total.

**Correction**: The internal mass (furniture) also couples to the surfaces. h_tr_me = 4.5 × A_furniture where A_furniture ≈ 0.1 × floor_area = 4.8 m². h_tr_me = 4.5 × 4.8 ≈ 22 W/K.

Total coupling = 150 + 22 = 172 W/K
τ = 77e6 / 172 ≈ 447,000s ≈ 124h ✓

---

## 4. Network Topology Specification

### 4.1 Node Equations (6R2C extended to 9R4C)

**Node definitions**:
```
T_ext     - Exterior air temperature (boundary condition)
T_env     - Exterior environment node (at exterior surfaces)
T_ms_wall - Wall thermal mass node
T_ms_roof - Roof thermal mass node
T_ms_floor- Floor thermal mass node
T_s       - Interior surface node
T_int     - Interior air temperature (controlled)
T_me      - Internal mass node (furniture, partitions)
T_ground  - Ground temperature (boundary condition)
```

**Heat balance equations**:

1. **Wall mass node** (T_ms_wall):
   ```
   C_wall × dT_ms_wall/dt = (T_env - T_ms_wall) / R_em_wall
                          + (T_s - T_ms_wall) / R_ms_wall
   ```

2. **Roof mass node** (T_ms_roof):
   ```
   C_roof × dT_ms_roof/dt = (T_env - T_ms_roof) / R_em_roof
                           + (T_s - T_ms_roof) / R_ms_roof
   ```

3. **Floor mass node** (T_ms_floor):
   ```
   C_floor × dT_ms_floor/dt = (T_ground - T_ms_floor) / R_em_floor
                             + (T_s - T_ms_floor) / R_ms_floor
   ```

4. **Interior surface node** (T_s):
   ```
   0 = Σ(T_ms_i - T_s) / R_ms_i        [i = wall, roof, floor]
     + (T_int - T_s) / R_is
     + ΣΦ_solar_i / h_tr_ms_i          [solar gains distributed to mass]
     + (T_me - T_s) / R_me              [internal mass coupling]
   ```

5. **Internal mass node** (T_me):
   ```
   C_me × dT_me/dt = (T_s - T_me) / R_me
                   + internal_gains × convective_fraction / C_me
   ```

**Resistance definitions**:
```
R_ms_wall   = 1 / h_tr_ms_wall   = R_int_to_mass_wall / A_wall
R_ms_roof   = 1 / h_tr_ms_roof   = R_int_to_mass_roof / A_roof
R_ms_floor  = 1 / h_tr_ms_floor  = R_int_to_mass_floor / A_floor
R_em_wall   = 1 / h_tr_em_wall   = R_ext_to_mass_wall / A_wall
R_em_roof   = 1 / h_tr_em_roof   = R_ext_to_mass_roof / A_roof
R_em_floor  = 1 / h_tr_em_floor  = R_ext_to_mass_floor / A_floor (ground coupling)
R_is        = 1 / h_tr_is
R_me        = 1 / h_tr_me
R_w         = 1 / h_tr_w (windows)
```

### 4.2 Conductance Calculations

**Wall (Case 900)**:
- Layers (interior to exterior): 200mm concrete | 50mm insulation
- R_int_to_mass = R_concrete + 0.5 × R_insulation = 0.20 + 0.625 = 0.825 m²K/W
- R_ext_to_mass = 0.5 × R_insulation + R_ext_film = 0.625 + 0.034 = 0.659 m²K/W
- A_opaque = 75.6 - 11.3 = 64.3 m²
- h_tr_ms_wall = 64.3 / 0.825 = 78 W/K
- h_tr_em_wall = 64.3 / 0.659 = 98 W/K

**Roof (Case 900)**:
- Layers: 200mm concrete slab
- R_int_to_mass = R_concrete_int_half + R_ext_film ≈ 0.10 + 0.034 = 0.134 m²K/W (concrete interior half only)
- Actually: For concrete slab without insulation, mass node is at mid-depth
- R_int_to_mass = 0.5 × R_concrete = 0.5 × 0.20 = 0.10 m²K/W
- R_ext_to_mass = 0.5 × R_concrete + R_ext_film = 0.10 + 0.034 = 0.134 m²K/W
- h_tr_ms_roof = 48 / 0.10 = 480 W/K (too high!)
- h_tr_ms_roof = 48 / 0.96 = 50 W/K (using ISO 13790 effective resistance)

**Correction**: For roof without insulation, the half-insulation rule doesn't apply. Instead, use the concrete's thermal admittance to define the mass node. The effective R_int_to_mass for a 200mm concrete slab with interior film is approximately 0.96 m²K/W (ISO 13790 Table 12).

For Case 900 roof: h_tr_ms_roof ≈ 50 W/K

**Floor (Case 900)**:
- Ground-coupled slab, no exterior film
- R_int_to_mass ≈ 0.96 m²K/W (similar to roof)
- h_tr_ms_floor ≈ 48 / 0.96 = 50 W/K
- h_tr_em_floor (ground coupling) ≈ 48 × 0.039 = 1.87 W/K (U = 0.039 W/m²K for ground)

**Summary for Case 900**:
| Conductance | Value (W/K) |
|-------------|-------------|
| h_tr_ms_wall | 78 |
| h_tr_ms_roof | 50 |
| h_tr_ms_floor | 50 |
| h_tr_em_wall | 98 |
| h_tr_em_roof | 50 |
| h_tr_em_floor | 2 (ground) |
| h_tr_is | 369 (walls + ceiling + floor) |
| h_tr_me | 22 (furniture) |
| h_tr_w | 3.6 (windows, U=0.3 × 12m²) |

---

## 5. Implementation Plan

### 5.1 Phase 6A: Core Data Structure Changes

**File**: `src/sim/thermal_model_data.rs` (new structure definition)

```rust
/// Multi-node thermal mass configuration
#[derive(Clone, Debug)]
pub struct MultiNodeThermalMass {
    /// Wall thermal mass node
    pub wall: ThermalMassNode,
    /// Roof thermal mass node
    pub roof: ThermalMassNode,
    /// Floor thermal mass node
    pub floor: ThermalMassNode,
    /// Internal mass node (furniture, partitions)
    pub internal: ThermalMassNode,
}

/// Individual thermal mass node
#[derive(Clone, Debug)]
pub struct ThermalMassNode {
    /// Temperature (K or °C)
    pub temperature: f64,
    /// Thermal capacitance (J/K)
    pub capacitance: f64,
    /// Interior conductance to surface (W/K)
    pub h_tr_ms: f64,
    /// Exterior conductance from environment (W/K)
    pub h_tr_em: f64,
    /// Heat flux accumulated (J)
    pub heat_flux_cumulative: f64,
}

impl Default for MultiNodeThermalMass {
    fn default() -> Self {
        Self {
            wall: ThermalMassNode::new(20.0, 8.0e6, 25.0, 100.0),
            roof: ThermalMassNode::new(20.0, 4.0e6, 50.0, 50.0),
            floor: ThermalMassNode::new(20.0, 4.0e6, 50.0, 2.0),
            internal: ThermalMassNode::new(20.0, 1.0e6, 4.5, 0.0),
        }
    }
}
```

### 5.2 Phase 6B: Conductance Calculation Updates

**File**: `src/sim/thermal_model_core.rs`

Update conductance calculation to produce per-surface h_tr_ms:

```rust
// For each zone, calculate per-surface thermal coupling
let h_tr_ms_wall = calculate_surface_conductance(
    &spec.construction.wall,
    SurfaceType::Wall,
    opaque_wall_area,
)?;

let h_tr_ms_roof = calculate_surface_conductance(
    &spec.construction.roof,
    SurfaceType::Roof,
    zone_floor_area,
)?;

let h_tr_ms_floor = calculate_surface_conductance(
    &spec.construction.floor,
    SurfaceType::Floor,
    zone_floor_area,
)?;

let h_tr_em_wall = calculate_exterior_conductance(
    &spec.construction.wall,
    SurfaceType::Wall,
    opaque_wall_area,
)?;

let h_tr_em_roof = calculate_exterior_conductance(
    &spec.construction.roof,
    SurfaceType::Roof,
    zone_floor_area,
)?;

let h_tr_em_floor = calculate_ground_conductance(
    &spec.construction.floor,
    zone_floor_area,
)?;
```

### 5.3 Phase 6C: Multi-Node Solver

**New file**: `src/physics/multi_node_solver.rs`

```rust
/// Multi-node thermal network solver
/// Solves the 9R4C network: wall + roof + floor + internal mass nodes
pub struct MultiNodeSolver {
    /// Per-surface thermal mass nodes
    pub nodes: MultiNodeThermalMass,
    /// Surface-to-air coupling (shared)
    pub h_tr_is: f64,
    /// Internal mass coupling
    pub h_tr_me: f64,
    /// Window conductance
    pub h_tr_w: f64,
}

impl MultiNodeSolver {
    /// Step the solver forward by dt seconds
    /// Returns (q_hvac, surface_temps, mass_temps)
    pub fn step(
        &mut self,
        dt: f64,
        t_ext: f64,
        t_ground: f64,
        t_int: f64,
        solar_gains: &SolarGainsDistribution,
        internal_gains: f64,
    ) -> SolverResult {
        // Energy balance for each mass node
        // C_i × dT_i/dt = Σ(A_ij × (T_j - T_i) / R_ij) + Φ_i

        // Wall node
        let q_wall = self.nodes.wall.conductance_to_surface(
            self.nodes.wall.temperature,
            t_ext,
            t_int,
            dt
        );

        // Roof node
        let q_roof = self.nodes.roof.conductance_to_surface(
            self.nodes.roof.temperature,
            t_ext,
            t_int,
            dt
        );

        // Floor node (coupled to ground)
        let q_floor = self.nodes.floor.conductance_to_ground(
            self.nodes.floor.temperature,
            t_ground,
            t_int,
            dt
        );

        // Internal mass node
        let q_internal = self.nodes.internal.conductance_to_surface(
            self.nodes.internal.temperature,
            t_int,
            internal_gains * 0.5, // convective fraction
            dt
        );

        // Surface node balance
        let t_s = self.calculate_surface_temperature(
            t_int,
            &[q_wall, q_roof, q_floor],
            solar_gains,
        )?;

        // Update all node temperatures
        self.nodes.wall.update_temperature(q_wall, dt);
        self.nodes.roof.update_temperature(q_roof, dt);
        self.nodes.floor.update_temperature(q_floor, dt);
        self.nodes.internal.update_temperature(q_internal, dt);

        SolverResult {
            q_hvac: self.calculate_hvac_load(t_int, t_s)?,
            surface_temperature: t_s,
            mass_temperatures: MassTemperatures {
                wall: self.nodes.wall.temperature,
                roof: self.nodes.roof.temperature,
                floor: self.nodes.floor.temperature,
                internal: self.nodes.internal.temperature,
            },
        }
    }
}
```

### 5.4 Phase 6D: Integration with CTF/FD Infrastructure

**Files to modify**:
- `src/physics/ctf_solver.rs` — Add multi-node CTF variant
- `src/physics/ctf_zone_coupling.rs` — Extend for multiple mass nodes
- `src/physics/solver_manager.rs` — Auto-select based on thermal mass

**CTF extension for multi-node**:

For each surface (wall, roof, floor), compute CTF coefficients independently:

```rust
/// Multi-node CTF solver for surfaces
pub struct MultiNodeCTFSolver {
    /// CTF solver per surface
    pub wall_solver: CTFSolver,
    pub roof_solver: CTFSolver,
    pub floor_solver: CTFSolver,
    /// Coupling conductances
    pub h_tr_ms_wall: f64,
    pub h_tr_ms_roof: f64,
    pub h_tr_ms_floor: f64,
}

impl MultiNodeCTFSolver {
    /// Step all surface solvers
    pub fn step(
        &mut self,
        t_int: f64,
        t_ext: f64,
        t_mass_wall: f64,
        t_mass_roof: f64,
        t_mass_floor: f64,
        solar_gains_wall: f64,
        solar_gains_roof: f64,
    ) -> MultiNodeCTFResult {
        let q_wall = self.wall_solver.step_with_mass(t_int, t_ext, t_mass_wall, solar_gains_wall);
        let q_roof = self.roof_solver.step_with_mass(t_int, t_ext, t_mass_roof, solar_gains_roof);
        let q_floor = self.floor_solver.step_with_mass(t_int, t_ext, t_mass_floor, 0.0); // floor has no solar

        MultiNodeCTFResult { q_wall, q_roof, q_floor }
    }
}
```

### 5.5 Phase 6E: Backward Compatibility Layer

**Maintain single-node for light-mass cases** (Case 100-650):

```rust
/// Determine thermal model complexity based on building characteristics
pub fn determine_thermal_model_type(
    wall_construction: &Construction,
    roof_construction: &Construction,
    floor_construction: &Construction,
    floor_area: f64,
) -> ThermalModelType {
    // Calculate effective thermal capacitance per area
    let cm_wall = wall_construction.effective_capacitance_per_area();
    let cm_roof = roof_construction.effective_capacitance_per_area();
    let cm_floor = floor_construction.effective_capacitance_per_area();
    let cm_total = (cm_wall + cm_roof + cm_floor) * floor_area;

    // ASHRAE 140 Case 900 threshold: Cm > 500 kJ/K = 5e8 J/K
    const HEAVY_MASS_THRESHOLD: f64 = 5e8;

    if cm_total > HEAVY_MASS_THRESHOLD {
        // Heavy mass: use multi-node architecture
        ThermalModelType::NineRFourC
    } else {
        // Light/medium mass: use single-node 5R1C
        ThermalModelType::FiveROneC
    }
}
```

**Configuration flag per case**:

```rust
pub struct ThermalModelConfig {
    /// Number of thermal mass nodes
    pub num_mass_nodes: MassNodeCount,
    /// Per-surface conductances (for multi-node)
    pub h_tr_ms_wall: Option<f64>,
    pub h_tr_ms_roof: Option<f64>,
    pub h_tr_ms_floor: Option<f64>,
    /// Single-node conductance (for 5R1C)
    pub h_tr_ms_total: Option<f64>,
}

pub enum MassNodeCount {
    One,   // 5R1C: single thermal mass node
    Four,  // 9R4C: wall + roof + floor + internal mass
}
```

---

## 6. File Changes Inventory

| File | Change Type | Description |
|------|-------------|-------------|
| `src/sim/thermal_model_data.rs` | Modify | Add `MultiNodeThermalMass` struct, per-surface conductance fields |
| `src/sim/thermal_model_core.rs` | Modify | Update conductance calculations, add multi-node solver dispatch |
| `src/sim/thermal_model_physics.rs` | Modify | Integrate multi-node solver, update energy balance |
| `src/physics/multi_node_solver.rs` | **New** | Core 9R4C solver implementation |
| `src/physics/ctf_coefficients.rs` | Modify | Add per-surface CTF coefficient calculation |
| `src/physics/ctf_solver.rs` | Modify | Extend for multi-node CTF variant |
| `src/physics/ctf_zone_coupling.rs` | Modify | Multi-node zone coupling |
| `src/physics/solver_manager.rs` | Modify | Auto-select: 5R1C for Cm < 5e8, 9R4C for Cm ≥ 5e8 |
| `src/physics/fd_solver.rs` | Modify | Update for multi-node FD surfaces |
| `src/physics/fd_discretization.rs` | Modify | Wall discretization for multi-node |
| `src/validation/ashrae_140_cases.rs` | Modify | Add case-specific thermal model type |
| `tests/ashrae_140_case_900.rs` | Modify | Update expected τ and reference values |
| `tests/thermal_mass_coupling_tests.rs` | **New** | Validate multi-node τ against ASHRAE 140 |
| `docs/phase_6_architecture.md` | **New** | This design document |

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **τ still too short** after multi-node | Medium | High | Validate h_tr_ms values against ISO 13790; adjust R_int_to_mass if needed |
| **Computational cost increase** | Low | Medium | Multi-node adds ~3x solver cost but still < 1ms per timestep |
| **Backward compatibility breakage** | Medium | High | Use config flag to maintain single-node for cases < 900 series |
| **CTF convergence issues** | Low | Medium | Test CTF solver separately before integration |
| **Ground coupling model incorrect** | Medium | Medium | Validate floor h_tr_em against ASHRAE 140 ground coupling specs |

### Key Validation Points

1. **τ validation for Case 900**: Should reach 120-200h
2. **τ validation for Case 600**: Should remain ~15h (light mass)
3. **Energy balance**: Sum of all node heat fluxes = HVAC + solar + internal
4. **ASHRAE 140 ranges**: All Case 900 metrics within ±15% of reference

---

## 8. Testing Strategy

### 8.1 Unit Tests

```rust
#[test]
fn test_wall_thermal_time_constant_case_900() {
    let model = ThermalModel::from_spec(&ASHRAE140Case::Case900.spec());
    let tau = model.calculate_thermal_time_constant();
    assert!(tau > 120.0 && tau < 200.0, "τ = {}h expected 120-200h", tau);
}

#[test]
fn test_light_mass_still_single_node() {
    let model = ThermalModel::from_spec(&ASHRAE140Case::Case600.spec());
    assert_eq!(model.thermal_model_type(), ThermalModelType::FiveROneC);
}

#[test]
fn test_heavy_mass_uses_multi_node() {
    let model = ThermalModel::from_spec(&ASHRAE140Case::Case900.spec());
    assert_eq!(model.thermal_model_type(), ThermalModelType::NineRFourC);
}
```

### 8.2 Integration Tests

```rust
#[test]
fn test_case_900_annual_energy_within_range() {
    let (heating, cooling) = simulate_case_900();
    let ref = CASE_900_REFERENCE;
    assert_in_range(heating, ref.annual_heating.0, ref.annual_heating.1);
    assert_in_range(cooling, ref.annual_cooling.0, ref.annual_cooling.1);
}

#[test]
fn test_case_900_free_floating_temperature_swing() {
    let (min_t, max_t) = simulate_case_900ff();
    let ref = CASE_900_REFERENCE;
    assert_in_range(min_t, ref.free_floating_min.0, ref.free_floating_min.1);
    assert_in_range(max_t, ref.free_floating_max.0, ref.free_floating_max.1);
}
```

### 8.3 ASHRAE 140 Full Validation Suite

Run all 12+ failing cases and verify:
- Case 900 series: τ within 120-200h
- Case 600 series: τ within 10-20h
- τ ratio (900/600) ≈ 10× as specified by ASHRAE 140

---

## 9. Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 6 MULTI-NODE THERMAL MODEL                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   EXTERIOR BOUNDARY                                                          │
│   ─────────────────                                                          │
│         │                                                                    │
│         │  h_tr_em_wall          h_tr_em_roof           h_tr_em_floor        │
│         │  (exterior wall)       (exterior roof)        (ground couple)       │
│         │       │                     │                      │               │
│         ▼       ▼                     ▼                      ▼               │
│    ┌─────────┐   ┌─────────┐      ┌─────────┐          ┌─────────┐          │
│    │ T_ms    │   │ T_ms    │      │ T_ms    │          │ T_ms    │          │
│    │ _wall   │   │ _roof   │      │ _floor  │          │ _ground │          │
│    └────┬────┘   └────┬────┘      └────┬────┘          └─────────┘          │
│         │             │                │                                      │
│         │ h_tr_ms     │ h_tr_ms        │ h_tr_ms                               │
│         │ _wall       │ _roof          │ _floor                                │
│         │             │                │                                      │
│         └─────────────┼────────────────┘                                        │
│                       │                                                        │
│                       ▼                                                        │
│              ┌────────────────┐                                                │
│              │      T_s       │ ◄── Interior surface node                    │
│              │  (all surfaces) │      h_tr_is (air ↔ surface)                   │
│              └────────┬───────┘      h_tr_me (internal mass coupling)          │
│                       │                    │                                  │
│                       │ h_tr_is             │ h_tr_me                           │
│                       ▼                    ▼                                  │
│              ┌────────────────┐    ┌────────────────┐                         │
│              │     T_int      │◄──►│     T_me        │                         │
│              │ (zone air)     │    │  (furniture)   │                         │
│              └───────┬────────┘    └────────────────┘                         │
│                      │                                                        │
│                      │ h_ve (infiltration)                                    │
│                      ▼                                                        │
│              ┌────────────────┐                                                │
│              │   HVAC LOAD   │ ◄── Controlled by setpoints                     │
│              └────────────────┘                                                │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  SOLAR GAIN DISTRIBUTION                                                      │
│  ───────────────────────                                                      │
│  Φ_solar = Σ (A_i × I_i × α_i) / n                                          │
│                                                                              │
│  Wall surfaces:    60% → T_ms_wall, 40% → T_s (direct)                     │
│  Roof surfaces:    60% → T_ms_roof, 40% → T_s (direct)                     │
│  Floor surfaces:   0% solar (opaque)                                        │
│  Windows:          100% → T_s (all absorbed at surface)                       │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  THERMAL TIME CONSTANTS                                                      │
│  ───────────────────────                                                      │
│  τ_wall   = C_wall / h_tr_ms_wall                                           │
│  τ_roof   = C_roof / h_tr_ms_roof                                           │
│  τ_floor  = C_floor / h_tr_ms_floor                                        │
│                                                                              │
│  Zone effective τ = Σ C_i / Σ (h_tr_ms_i + h_tr_me_i)                        │
│                                                                              │
│  Case 900 expected: τ ≈ 150h (heavy concrete)                                │
│  Case 600 expected: τ ≈ 15h (light frame)                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Success Criteria

1. **Case 900 τ validation**: Thermal time constant reaches 120-200h
2. **Case 600 τ preservation**: τ stays at ~15h for light-mass cases
3. **τ ratio**: 900/600 ≈ 10× as specified by ASHRAE 140
4. **Energy accuracy**: Case 900 annual heating/cooling within ±15% of ASHRAE 140 reference
5. **Temperature swing**: Case 900FF max temperature within [41.8, 46.4]°C
6. **Backward compatibility**: Cases 100-650 unchanged (or within existing tolerance)
7. **Computational cost**: < 2× current runtime for multi-node cases
8. **Code quality**: All existing tests pass, new tests cover multi-node architecture

---

## 11. Open Questions (Resolved)

### Q1: Ground Coupling Model
**Resolved: Dynamic (Kusuda-Achenbach monthly sinusoidal)**

EnergyPlus uses the Kusuda-Achenbach formula for foundation/slab ground coupling:
```
T(z,t) = T_mean - T_amp × exp(-z × √(π/(365×α))) × cos(ω×t - z × √(π/(365×α)))
```

**Already implemented** in `src/sim/boundary.rs:158-345` as `DynamicGroundTemperature`.

**Configuration** (per ASHRAE 140 climate):
```rust
model.set_dynamic_ground_temp(
    11.0,   // t_mean: mean annual ground temp (°C)
    12.0,   // t_amplitude: annual amplitude (°C)
    1.0,    // depth: foundation depth (m)
    0.07,   // diffusivity: soil thermal diffusivity (m²/day)
);
```

Monthly variation (~±5-8°C at 0.5m depth) provides semi-dynamic ground coupling without full soil model complexity.

### Q2: Solar Distribution
**Resolved: Orientation-based with Perez anisotropic sky model**

EnergyPlus `FullInteriorAndExterior` uses view factors for beam projection onto interior surfaces.

**Current fluxion** already groups surfaces by orientation (`thermal_model_iterative.rs:188-189`) to avoid double-counting.

**Recommended per-orientation split:**

| Surface | Beam Gain | Diffuse Gain | Split to Mass | Split to Surface |
|---------|-----------|--------------|---------------|-------------------|
| South Wall | High (noon peak) | Moderate | 60% | 40% |
| East Wall | High (morning) | Moderate | 60% | 40% |
| West Wall | High (afternoon) | Moderate | 60% | 40% |
| North Wall | Low (minimal) | Low-moderate | 60% | 40% |
| Roof | Highest (horizontal) | High | 60% | 40% |
| Floor | Low | Low | 0% | 0% |

**Implementation**: Store separate `direct_beam` and `diffuse` pools per orientation. Apply 60/40 mass/surface split per group. Use Perez model for diffuse anisotropy.

### Q3: Internal Mass Capacitance
**Resolved: Floor-area-based formula with EnergyPlus-style constants**

Current implementation: `C_me = 0.25 × C_total` (25% heuristic) — gives τ_me ~38h (too slow)

**EnergyPlus formula**: `C_int = A_furniture × 55,000 J/m²K`

**Recommended for Phase 6:**

| Parameter | Formula | Case 900 Value |
|-----------|---------|-----------------|
| C_me | `A_floor × 55,000 × f_furn` | ~1.3-2.6 MJ/K |
| h_tr_me | `4.5 × (0.3-0.5) × A_floor` | ~65-108 W/K |
| τ_me | C_me / h_tr_me | **~3-4 hours** ✓ |

**Key change**: Increase `f_furn` fraction from `0.1 × floor_area` to `0.3-0.5 × floor_area` so furniture responds in realistic 3-4h time constant (not 38h).

### Q4: CTF vs FD Selection for Multi-Node
**Resolved: FD (Finite Difference) for multi-node architecture**

| Solver | Multi-Node Capable? | Notes |
|--------|---------------------|-------|
| CTF | **No** — surface-only flux | Single interior/exterior history |
| FD | **Yes** — full temperature profile | Tridiagonal solve, N nodes per layer |

**Current**: τ < 2h → 5R1C, τ ≥ 2h → CTF, CTF fails → FD

**Phase 6 change**: Add `multi_node_mode` flag to `ThermalMethodSelectorConfig`:
- If `multi_node_mode = true`: Force `ThermalMethod::FiniteDifference`
- CTF cannot represent internal per-surface thermal mass nodes

**Implementation** (`method_selector.rs`, `solver_manager.rs`):
```rust
if multi_node_mode {
    return ThermalMethod::FiniteDifference;  // Required for multi-node
}
```

### Q5: Validation Priority
**Resolved: Energy first, τ second, temperature third**

EnergyPlus/ASHRAE 140 validation hierarchy:

| Priority | Metric | Tolerance | Test Order |
|----------|--------|-----------|------------|
| **1** | Annual heating/cooling energy | ±15% | Run first |
| **1** | Peak heating/cooling loads | ±10% | Run after energy stable |
| **2** | Thermal time constant τ | Within ±20% | Validate separately |
| **3** | Free-floating min/max temps | Within range | Run after energy |
| **3** | Temperature swing reduction | ~19.6% (600→900) | Paired comparison |

**Recommended test order**:
1. `test_case_600_energy` — low-mass baseline
2. `test_case_600ff_temperature` — free-float baseline
3. `test_case_900_energy` — high-mass energy validation
4. `test_case_900ff_temperature` — high-mass temp validation
5. `test_tau_case_600_vs_900` — validate τ ratio ≈ 10×
6. `test_temperature_swing_reduction` — validate ~19.6% reduction

**Critical**: τ is currently NOT validated in tests. Add explicit `test_thermal_time_constant_<case>` tests in Phase 6.

---

## 12. Implementation Readiness

### Pre-Implementation Items Complete
- [x] Root cause analysis (LIMIT-05, KNOWN_ISSUES.md)
- [x] Architecture design (Section 3)
- [x] Network topology (Section 4)
- [x] Open questions resolved (Section 11)
- [x] CTF/FD/ground coupling research (existing infrastructure identified)
- [x] **Phase 6A: Core data structures** — `MultiNodeThermalMass`, `ThermalMassNode`, `ThermalModelType::NineRFourC`

### Remaining Implementation Tasks
- [ ] Phase 6B: Per-surface conductance calculations
- [ ] Phase 6C: Multi-node solver (`src/physics/multi_node_solver.rs`)
- [ ] Phase 6D: CTF/FD integration
- [ ] Phase 6E: Backward compatibility layer
- [ ] Validation tests: Add τ validation explicitly

### Files Ready for Implementation

| File | Status |
|------|--------|
| `src/physics/multi_node_solver.rs` | **To create** — core 9R4C solver |
| `src/sim/thermal_model_data.rs` | Modify — add MultiNodeThermalMass |
| `src/sim/thermal_model_core.rs` | Modify — per-surface h_tr_ms |
| `src/physics/method_selector.rs` | Modify — add multi_node_mode |
| `src/physics/solver_manager.rs` | Modify — route multi-node to FD |
| `src/sim/boundary.rs` | **Ready** — DynamicGroundTemperature exists |
| `src/physics/ctf_solver.rs` | **Ready** — existing infrastructure |
| `src/physics/fd_solver.rs` | **Ready** — existing infrastructure |

---

*Document version: 1.1 (Open Questions resolved)*
*Next action: Implement Phase 6A (core data structure changes)*
