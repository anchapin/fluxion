# CTF Implementation Requirements

## Overview

This document specifies the requirements for Conduction Transfer Function (CTF) implementation in Fluxion. The CTF method is the industry-standard approach for calculating time-dependent heat transfer through building envelope elements, used by EnergyPlus and other major building energy simulation tools.

---

## 1. EnergyPlus CTF Methodology

### 1.1 CTF Concept

CTF coefficients relate current heat flux to past temperatures through a convolution:

```
Q(t) = Σ(CTF_coefficients[i] * T(t-i)) for i = 0 to τ
```

Where:
- τ is the "time constant" or number of terms needed for convergence (typically 10-50)
- Q(t) is heat flux at time t [W/m²]
- T(t-i) is temperature at time t-i [°C]

### 1.2 CTF Coefficient Formula (ASHRAE Standard)

The standard CTF formulation calculates interior surface heat flux:

```
q''_interior,t = Σ(X_j · T_outside,t-j) - Σ(Y_j · T_inside,t-j) - Σ(Φ_j · q''_t-j)
```

Where:
- **X coefficients**: Exterior temperature response terms
- **Y coefficients**: Interior temperature response terms  
- **Φ (phi) coefficients**: Heat flux history terms (recursive contribution)
- j ranges from 0 to num_coeffs-1

### 1.3 CTF Calculation Algorithm

1. **Build transmission matrices** for each material layer using Laplace domain
2. **Multiply layer matrices** to get overall wall transfer function
3. **Decompose via partial fractions** into poles and residues
4. **Sample coefficients** at discrete time intervals (typically 1 hour)
5. **Truncate** to convergence criteria (typically 10-50 terms)

### 1.4 Key Parameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Timestep | 3600 seconds | 1-hour for ASHRAE 140 |
| Number of terms | 10-50 | Based on thermal mass |
| Convergence | 0.99 | Sum of Φ terms < 0.01 |
| Warmup period | 7 days | Initialize history buffers |

---

## 2. Current Implementation Analysis

### 2.1 Existing CTF Modules

Fluxion has the following CTF implementations:

| Module | Status | Description |
|--------|--------|-------------|
| `ctf_coefficients.rs` | ✅ Active | Coefficient calculation from material properties |
| `ctf_solver.rs` | ✅ Active | Runtime solver using precomputed coefficients |
| `multi_node_ctf.rs` | ✅ Active | State-space method (EnergyPlus-style) |
| `per_surface_ctf.rs` | ✅ Active | Per-surface heat balance integration |
| `ctf_solver_wrapper.rs` | Legacy | Wrapper for solver access |

### 2.2 Integration with Thermal Model

From `engine.rs` grep results:
- CTF coefficients are precomputed during model initialization
- CTF solvers are created per zone for parallel evaluation
- Method `enable_ctf()` is available for CTF activation
- 5R1C fallback exists when CTF disabled

### 2.3 Material Properties

The system already handles:
- **CTFMaterial**: thickness, conductivity, density, specific_heat
- **MaterialLayer**: Used in multi-node CTF and validation
- **ConstructionLayer**: Mapped to CTF materials in validator

---

## 3. Gaps and Issues

### 3.1 Identified Gaps

1. **Coefficient calculation not integrated**: CTF coefficients calculated but may not be properly connected to all thermal paths
2. **Multi-layer support**: Need to verify full multi-layer wall handling
3. **Validation against EnergyPlus**: Need benchmark comparisons
4. **Surface-specific CTF**: Per-surface CTF not fully connected to all surfaces

### 3.2 Technical Issues to Address

1. **Flux direction**: Ensure correct sign convention (positive = into zone)
2. **History initialization**: Proper warmup for temperature/flux history
3. **Boundary conditions**: Interior/exterior convection correctly applied
4. **Solar absorption**: Surface solar absorptance integrated with CTF

---

## 4. Implementation Requirements

### 4.1 Material Properties Needed

For each construction layer:

```rust
pub struct CTFMaterial {
    pub name: String,           // Layer name for diagnostics
    pub thickness: f64,         // [m]
    pub conductivity: f64,      // [W/m·K]
    pub density: f64,           // [kg/m³]
    pub specific_heat: f64,     // [J/kg·K]
}
```

Computed properties:
- Thermal diffusivity: α = k/(ρ·c_p) [m²/s]
- Thermal resistance: R = L/k [m²·K/W]
- Time constant: τ = R·C where C = ρ·c_p·L [s]

### 4.2 CTF Coefficient Structure

```rust
pub struct CTFCoefficients {
    pub x: Vec<f64>,     // Exterior temperature response
    pub y: Vec<f64>,     // Interior temperature response  
    pub z: Vec<f64>,     // (not used in standard CTF)
    pub phi: Vec<f64>,   // Flux history (recursive terms)
    pub timestep: f64,   // [s]
    pub num_coeffs: usize,
}
```

### 4.3 Algorithm Steps

1. **Initialization Phase** (once per construction):
   - Build material layer list from construction
   - Calculate CTF coefficients using transfer function method
   - Initialize history buffers (temperature, flux)

2. **Runtime Phase** (each timestep):
   - Update temperature history (shift and push new value)
   - Calculate heat flux using CTF formula
   - Update flux history
   - Return flux to thermal model

### 4.4 Integration Points

| Location | Action |
|----------|--------|
| `ThermalModel::new()` | Create CTF solver instances |
| `ThermalModel::init_ctf()` | Compute coefficients from construction |
| `step_physics_5r1c()` | Use CTF instead of RC network for conduction |
| `step_physics_6r2c()` | Use CTF for mass wall conduction |
| `enable_advanced_solver()` | Select CTF vs 5R1C based on case |

---

## 5. Validation Strategy

### 5.1 Unit Tests

- **Coefficient calculation**: Verify against known analytical solutions
- **Flux calculation**: Check sign convention and magnitude
- **Multi-layer**: Compare with single-layer equivalent
- **History update**: Verify buffer shift behavior

### 5.2 Integration Tests

- **ASHRAE 140 Cases**: Compare with reference values
- **600-series (low-mass)**: Should use 5R1C or fast CTF
- **900-series (high-mass)**: Should use CTF for accurate thermal mass
- **Case 960 (sunspace)**: Multi-zone CTF validation

### 5.3 EnergyPlus Comparison

- Identify EnergyPlus CTF output for test cases
- Compare heat flux time series
- Validate temperature predictions

---

## 6. Architecture After Full CTF

```
Thermal Model
├── CTF Solver (replaces RC network conduction)
│   ├── Coefficient Calculator (initialization)
│   │   ├── Parse construction layers
│   │   ├── Build transfer functions
│   │   └── Compute X, Y, Φ coefficients
│   └── Runtime Solver (each timestep)
│       ├── Update temperature history
│       ├── Calculate heat flux from CTF
│       └── Update flux history
├── Zone Heat Balance (unchanged)
└── HVAC (IdealLoads)
```

### 6.1 Method Selection

| Case Type | Method | Rationale |
|-----------|--------|-----------|
| 600-series (low-mass) | 5R1C or fast CTF | Thermal mass not significant |
| 900-series (high-mass) | Multi-node CTF | Proper thermal lag modeling |
| Free-floating (FF) | Same as parent case | No HVAC, thermal mass critical |
| Case 960 (sunspace) | Multi-zone CTF | Inter-zone heat transfer |

---

## 7. Recommendations for Session 6+ Implementation

### 7.1 Priority Tasks

1. **Verify current CTF integration**: Ensure 900-series cases use CTF
2. **Fix flux direction bugs**: Confirm sign convention correct
3. **Add warmup initialization**: Proper history buffer setup
4. **Validate against EnergyPlus**: Benchmark comparison

### 7.2 Testing Strategy

1. Start with simple single-layer wall (analytical solution exists)
2. Progress to multi-layer walls (ASHRAE 140 test cases)
3. Validate thermal mass behavior (900-series)
4. Test multi-zone (Case 960)

### 7.3 Performance Considerations

- CTF is O(n) per surface where n = num_coeffs (typically 10-50)
- Multi-node CTF adds O(m) where m = nodes_per_layer (typically 10)
- Precompute coefficients once, reuse for all timesteps
- Parallelize across zones using rayon

---

## 8. Summary

The CTF implementation in Fluxion is substantially complete with:
- Coefficient calculation from material properties
- Runtime solver with history management  
- Multi-node state-space option
- Per-surface heat balance integration

The primary task for Session 6+ is **validation and integration verification**:
1. Confirm CTF is activated for 900-series cases
2. Validate against ASHRAE 140 reference values
3. Compare with EnergyPlus CTF outputs
4. Fix any integration issues found

The architecture supports both 5R1C (for low-mass) and CTF (for high-mass) approaches with automatic selection based on case type.

---

## References

- ASHRAE Standard 140-2020, Appendix C: Test Cases and Verification
- DOE EnergyPlus Engineering Reference, Section on Conduction Transfer Functions
- Ceylan & Myers (1980): Long time-step heat conduction models using state-space
- Seem (1987): Modeling of heat transfer in buildings