# Architecture Patterns: Multi-Zone Thermal Simulation

**Domain:** Building Energy Modeling - Multi-Zone Thermal Networks
**Researched:** 2026-04-06
**Confidence:** MEDIUM

## Recommended Architecture

### Multi-Zone Thermal Network Overview

The multi-zone thermal model extends the single-zone 5R1C network to N zones, where each zone has its own 5R1C thermal network, and zones are coupled through inter-zone conductance.

```
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Zone Thermal Model                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Zone 1              Zone 2              Zone N            │
│  ┌─────────┐         ┌─────────┐         ┌─────────┐     │
│  │ Ti, Tm  │◄───────►│ Ti, Tm  │◄───────►│ Ti, Tm  │     │
│  │ 5R1C    │  h_tr_iz │ 5R1C    │  h_tr_iz │ 5R1C    │     │
│  └────┬────┘         └────┬────┘         └────┬────┘     │
│       │                   │                   │           │
│       └───────────────────┴───────────────────┘           │
│                     HVAC (per zone)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **ThermalModel** | N-zone thermal simulation | Accepts num_zones > 1, maintains VectorField of zone temps |
| **InterZoneConductance** | Heat transfer between zones | Calculated from common wall area, U-value |
| **ZoneHVACController** | Per-zone HVAC control | Uses zone-specific setpoints from VectorField |
| **ASHRAE140Validator** | Multi-zone case validation | Case 960, 970, 980 specifications |

### Data Flow

```
User specifies N zones with geometry, materials, HVAC per zone
    ↓
ThermalModel::new(num_zones) creates N-zone model
    ↓
For each timestep:
    1. Calculate solar gains per zone
    2. Calculate internal gains per zone
    3. Calculate inter-zone heat flow (h_tr_iz * (Ti_j - Ti_k))
    4. Solve coupled ODE system: Cm * dT/dt = A*T + Q
    5. Apply HVAC control per zone
    ↓
Extract annual energy, peak loads per zone
    ↓
Validate against ASHRAE 140 multi-zone cases
```

## Patterns to Follow

### Pattern 1: N×5R1C Thermal Network

**What:** Each zone has its own 5R1C thermal network, extended from single-zone

**When:** Basic multi-zone modeling with thermal coupling between zones

**Example:**
```rust
// In ThermalModel, extend from single-zone to N-zone
pub struct ThermalModel<T: ContinuousTensor<f64>> {
    pub num_zones: usize,
    pub temperatures: VectorField,      // [Ti_1, Ti_2, ..., Ti_N]
    pub mass_temperatures: VectorField, // [Tm_1, Tm_2, ..., Tm_N]
    pub heating_setpoints: VectorField, // [T_set_1, T_set_2, ..., T_set_N]
    pub cooling_setpoints: VectorField,
    pub h_tr_iz: VectorField,          // Inter-zone conductance
    // ... existing 5R1C fields extended to VectorField
}
```

### Pattern 2: Inter-Zone Conductance Calculation

**What:** Calculate heat transfer between adjacent zones based on common wall properties

**When:** Modeling heat flow between zones through internal walls, floors, ceilings

**Example:**
```rust
// Inter-zone conductance from common wall area
fn calculate_inter_zone_conductance(
    common_wall_area: f64,
    wall_u_value: f64,
) -> f64 {
    common_wall_area * wall_u_value // W/K
}

// Heat flow between zones i and j
fn inter_zone_heat_flow(h_tr_ij: f64, ti: f64, tj: f64) -> f64 {
    h_tr_ij * (ti - tj) // Watts
}
```

### Pattern 3: Zone-Level HVAC Control

**What:** Each zone has independent HVAC control based on its own setpoint

**When:** Buildings with different temperature requirements per zone

**Example:**
```rust
// Per-zone HVAC control
for zone_idx in 0..num_zones {
    let ti = temperatures.get(zone_idx);
    let t_set_heat = heating_setpoints.get(zone_idx);
    let t_set_cool = cooling_setpoints.get(zone_idx);

    let load = if ti < t_set_heat {
        // Heating needed
        (t_set_heat - ti) * h_total
    } else if ti > t_set_cool {
        // Cooling needed
        (t_set_cool - ti) * h_total
    } else {
        0.0
    };
    loads.set(zone_idx, load);
}
```

### Pattern 4: Coupled ODE Solver for Multi-Zone

**What:** Solve system of N coupled differential equations for zone temperatures

**When:** Buildings with significant thermal coupling between zones

**Example:**
```rust
// Coupled system: C * dT/dt = A * T + Q
// Where C is diagonal matrix of zone thermal capacitances
// A is coupling matrix (including inter-zone conductance)
// Q is vector of heat gains (solar, internal, HVAC)

fn solve_coupled_system(
    c: &[f64],      // Zone thermal capacitances
    h_tr_iz: &[f64], // Inter-zone conductances
    q: &[f64],      // Heat gains
    dt: f64,        // Timestep
) -> Vec<f64> {
    // Use implicit (backward Euler) or semi-implicit method
    // Build and solve: (C/dt - A) * T_new = C/dt * T_old + Q
}
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Independent Zone Simulation

**What:** Simulating each zone separately without inter-zone coupling

**Why bad:** Misses heat transfer between zones, significant errors for tightly-coupled zones

**Instead:** Implement inter-zone conductance and solve coupled system

### Anti-Pattern 2: Single HVAC Setpoint for All Zones

**What:** Using one heating/cooling setpoint for entire building

**Why bad:** Doesn't model real buildings with zone-level control

**Instead:** Use VectorField for zone-specific setpoints

### Anti-Pattern 3: No Energy Balance Check

**What:** Not verifying energy conservation in multi-zone system

**Why bad:** Errors can accumulate, particularly with inter-zone flows

**Instead:** At each timestep, verify sum of zone energies + inter-zone transfer = total

## Scalability Considerations

| Concern | At 2 Zones | At 10 Zones | At 50 Zones |
|---------|------------|-------------|-------------|
| **Matrix size** | 2×2 trivial | 10×10 small | 50×50 moderate |
| **Solver time** | <1ms | <5ms | <50ms |
| **Memory** | negligible | negligible | <1MB |
| **Inter-zone pairs** | 1 | 9 | 49 |

### Performance Notes

- **2-10 zones:** Direct solve is fine, no special optimization needed
- **10-50 zones:** Consider sparse matrix representation for h_tr_iz
- **50+ zones:** May need iterative solver or zone grouping

## Sources

- ISO 13790: Energy performance of buildings - Calculation of energy use for space heating and cooling
- EnergyPlus Engineering Reference: Zone coupling methodology
- ASHRAE Standard 140-2017: Case 960, 970, 980 specifications
- Fluxion existing 5R1C implementation

---

*Architecture research for: Multi-Zone Thermal Simulation*
*Researched: 2026-04-06*
