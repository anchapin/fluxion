# Option A Analysis: Adding Separate Concrete Thermal Mass Node

## Issue #715 Context

**Problem**: Case 900 thermal bypass causes ~200% heating error, τ=58h vs target 120-200h

**Root Cause**: ISO 13790 half-insulation rule places thermal mass node at foam insulation layer, excluding concrete block (on exterior side of insulation) from h_tr_ms thermal coupling.

## Code Architecture Analysis

### 1. Thermal Model Structure (`thermal_model_core.rs`)

**Thermal Network Types** (lines 64-87):
```rust
// 5R1C: h_tr_w, h_ve, h_tr_em, h_tr_ms, h_tr_is
// 6R2C: + h_tr_me (envelope mass ↔ internal mass coupling)
// 8R3C: + h_tr_ceiling, h_tr_floor, h_tr_partition (experimental)
```

**Half-Insulation Rule Implementation** (lines 776-797):
```rust
// Layers interior to insulation → full R contribution
// Insulation layer → half R contribution
// Layers exterior to insulation → excluded
for (idx, layer) in layers.iter().enumerate() {
    let layer_r = layer.r_value();
    if idx < ins_idx {
        r_interior_to_mass += layer_r;  // Full
    } else if idx == ins_idx {
        r_interior_to_mass += layer_r / 2.0;  // Half
        break;  // Stop at insulation
    }
}
```

**Case 900 High-Mass Wall Layers** (`construction.rs:988-993`):
```
Index 0: wood_siding (interior)     R = 0.064 m²K/W, density=500
Index 1: foam (insulation)        R = 1.54 m²K/W, density=10  ← DOMINANT INSULATION
Index 2: concrete_block (exterior) R = 0.196 m²K/W, density=1400  ← EXCLUDED!
```

**Result**: Only wood_siding (interior to insulation) contributes to thermal mass. Concrete block's ~140,000 J/m²K capacitance is excluded, causing τ ≈ 26h instead of target 120-200h.

### 2. Thermal Conductance Calculations

**h_tr_ms (Mass-to-Surface)** (lines 796-839):
```rust
// Physics-based: τ = Cm / h_tr_ms
// Uses half-insulation rule to find R_interior_to_mass
let h_ms_physics = opaque_area / r_interior_to_mass.max(0.001);
// Adds roof and floor contributions
let h_ms_total = h_ms_physics + h_ms_roof + h_ms_floor;
```

**h_tr_em (Exterior-to-Mass)** (lines 866-972):
```rust
// Iterates from exterior toward insulation
// Layers exterior to insulation → full R
// Insulation layer → half R
let h_tr_em_base = opaque_area / r_exterior_to_mass;
```

**τ Diagnostic Output** (lines 1057-1071):
```rust
let tau_seconds = cm / h_total.max(0.1);  // τ = Cm / (h_tr_ms + h_tr_me)
let tau_hours = tau_seconds / 3600.0;
```

### 3. Backward Euler Solver (`thermal_model_physics.rs`)

**5R1C Mass Update** (lines 1200-1222):
```rust
// Single thermal mass node update
backward_euler_update(
    tm_old, dt, cm,
    h_tr_em,  // exterior-to-mass
    h_tr_ms,  // mass-to-surface
    t_ext,     // sol-air temperature
    t_s,       // surface temperature
    phi_m_zone, // internal gains to mass
)
```

**6R2C Mass Update** (lines 1785-1853):
```rust
// Two thermal mass nodes: envelope + internal
backward_euler_update_2cond(
    tm_env_old, dt, cm_env,
    h_tr_ms,    // surface-to-envelope-mass
    h_tr_me,    // envelope-mass-to-internal-mass
    t_s,        // surface temperature
    tm_int,     // internal mass temperature
    phi_m_env_zone,
)
```

### 4. 8R3C Model (Experimental, Lines 2018-2065)

The 8R3C model already exists with ceiling/floor/partition mass nodes, but uses simplified relaxation-based updates rather than a proper coupled system. This is not a viable template.

## Concrete Node Integration Analysis

### Option A: Add Concrete Block as Separate Mass Node

**Proposed Thermal Network:**
```
T_ext → h_tr_ec → T_concrete → h_tr_cm → T_env_mass → h_tr_ms → T_surface → h_tr_is → T_zone
                                    ↑
                               (h_tr_me)
                                    ↑
                              T_int_mass
```

**Equations Governing Concrete Node Temperature:**

The concrete block would need its own heat balance:
```
C_concrete * dT_concrete/dt = h_tr_ec * (T_ext - T_concrete) + h_tr_cm * (T_env_mass - T_concrete)
```

**Changes Required:**

1. **ThermalModelData** (`thermal_model_core.rs:1815-2007`):
   - Add `concrete_mass_temperatures: VectorField`
   - Add `concrete_thermal_capacitance: f64`
   - Add `h_tr_ec: VectorField` (exterior-to-concrete)
   - Add `h_tr_cm: VectorField` (concrete-to-envelope-mass)

2. **Construction Properties** (`construction.rs:507-530`):
   - Modify `find_dominant_insulation_layer_index()` to handle high-mass externally-insulated constructions
   - OR add new method `find_concrete_layer_index()` for Case 900-type constructions

3. **h_tr Calculations** (`thermal_model_core.rs:772-972`):
   - Calculate h_tr_ec: R_exterior_to_concrete (exterior film + concrete_block + half insulation)
   - Calculate h_tr_cm: R_concrete_to_mass (half insulation + interior layers to mass node)
   - Modify h_tr_ms: now from interior surface to envelope mass (through interior layers only)

4. **Backward Euler Solver** (`thermal_model_physics.rs:1200-1260`):
   - Add third condition to `backward_euler_update_2cond` → `backward_euler_update_3cond`
   - Solve coupled system: T_ext → T_concrete → T_env_mass → T_int_mass

5. **Zone Air Heat Balance** (lines 1557-1562, 1666-1729):
   - Surface temperature T_s must account for concrete node influence
   - T_s = (h_tr_ms*Tm_env + h_tr_is*T_i + h_tr_cm*T_concrete + phi_st) / (h_tr_ms + h_tr_is + h_tr_cm)

6. **CTF/FD Integration** (lines 144-209, 1470-1512):
   - When CTF is enabled, concrete node should use CTF flux, not lumped h_tr_ec

### Is It Parallel or Series?

**Analysis**: The concrete block is physically **in series** with the thermal mass path:
- Heat must flow: exterior → concrete → insulation → interior surface → zone air
- BUT the concrete block's large thermal capacitance acts as a **thermal buffer**

**Proposed Model**: Series coupling with large capacitance
```
T_ext → [h_tr_ec, C_concrete] → T_concrete → [h_tr_cm, C_env] → T_env_mass
```

This is fundamentally different from the 6R2C internal mass (furniture/partitions), which couples to the envelope mass **in parallel** through air.

## Risks and Complexity Estimate

### Architectural Risk: HIGH

1. **Thermal Network Restructuring**: Adding a concrete node changes the fundamental thermal network topology. The 5R1C/6R2C equations are derived from a specific node structure.

2. **CTF Integration**: The CTF solver computes heat flux through the entire wall. If we add a separate concrete node, CTF flux must be properly coupled.

3. **Time Constant Implications**: τ = Cm/h_total will change significantly. Need to validate that new τ matches ASHRAE 140 references.

4. **Energy Conservation**: Adding new nodes requires re-verifying energy conservation through the thermal network.

### Implementation Complexity: 5-7 days

| Task | Estimate |
|------|---------|
| Add new fields to ThermalModelData | 0.5 day |
| Calculate h_tr_ec, h_tr_cm conductances | 1 day |
| Implement 3-condition backward Euler | 1.5 days |
| Update surface temperature calculation | 0.5 day |
| Update zone air heat balance | 0.5 day |
| CTF/FD coupling modifications | 1 day |
| Debug output and validation | 1 day |

### Alternative: Fix Half-Insulation Rule

Given the complexity of Option A, consider fixing the **root cause** instead:

**Modify h_tr_ms calculation** to include concrete block for externally-insulated constructions:

```rust
// For high-mass externally-insulated walls:
// Include concrete layer resistance in r_interior_to_mass
if construction.is_externally_insulated() {
    // Concrete block is interior to insulation for h_tr_ms purposes
    r_interior_to_mass += concrete_block_r;
}
```

**Benefits:**
- Changes only h_tr_ms calculation (1 location)
- Maintains existing thermal network structure
- No changes to backward Euler solver
- Preserves CTF/FD integration

**Complexity**: 1-2 days

## Recommendation

**Option A (Add Concrete Node)** is architecturally risky and complex (5-7 days) with potential energy conservation issues.

**Recommended Alternative**: Fix the half-insulation rule for externally-insulated constructions. This addresses the root cause (concrete block excluded from thermal mass) without restructuring the thermal network.

**If Option A is required**:
1. Implement as new thermal model type (e.g., `SevenRTwoC`) to avoid breaking existing 5R1C/6R2C
2. Add comprehensive energy conservation checks
3. Validate τ matches ASHRAE 140 reference values
4. Test CTF integration thoroughly
