# Finite Difference Method Design Document

**Date:** 2026-03-17
**Phase:** 25-03 (Alternative Physics Implementation)
**Author:** Fluxion Development Team

---

## 1. Overview

This document specifies the finite difference (FD) method for 1D heat conduction through multi-layer building envelope constructions. The FD method replaces the lumped capacitance (5R1C) approach for high-mass buildings to achieve ±3-5% accuracy.

**Target Accuracy:** ±3-5% annual energy for ASHRAE 140 Case 900 (currently ±229-322% with 5R1C)

**Target Performance:** ≥500 configs/sec throughput

---

## 2. Mathematical Formulation

### 2.1 Governing Equation

The 1D heat conduction equation:

$$\rho c_p \frac{\partial T}{\partial t} = k \frac{\partial^2 T}{\partial x^2}$$

Where:
- ρ = density (kg/m³)
- c_p = specific heat (J/kg·K)
- k = thermal conductivity (W/m·K)
- T = temperature (°C)
- t = time (s)
- x = spatial coordinate (m)

### 2.2 Discretization Scheme

**Selected:** Implicit (Backward Time, Central Space - BTCS)

**Rationale:**
- Unconditionally stable (no CFL restriction)
- First-order accurate in time, second-order in space
- Suitable for building simulation (slow thermal dynamics)
- Crank-Nicolson (2nd order) offers marginal accuracy improvement at higher complexity

**Implicit Scheme:**

$$\frac{T_i^{n+1} - T_i^n}{\Delta t} = \alpha \frac{T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1}}{\Delta x^2}$$

Where α = k/(ρ·c_p) = thermal diffusivity (m²/s)

### 2.3 Tridiagonal System

Rearranging the implicit scheme:

$$-Fo \cdot T_{i-1}^{n+1} + (1 + 2Fo) \cdot T_i^{n+1} - Fo \cdot T_{i+1}^{n+1} = T_i^n$$

Where Fo = α·Δt/Δx² = Fourier number

**Matrix Form:**

```
┌                    ┐ ┌       ┐   ┌       ┐
│ B₀  C₀   0   ...   │ │ T₀    │   │ D₀    │
│ A₁  B₁  C₁   ...   │ │ T₁    │   │ D₁    │
│  0  A₂  B₂  C₂ ... │ │ T₂    │ = │ D₂    │
│ ...               │ │ ...   │   │ ...   │
│            Aₙ  Bₙ  │ │ Tₙ    │   │ Dₙ    │
└                    ┘ └       ┘   └       ┘
      A·T^(n+1)     =   D
```

Where:
- A_i = -Fo (lower diagonal)
- B_i = 1 + 2Fo (main diagonal)
- C_i = -Fo (upper diagonal)
- D_i = T_i^n (known from previous timestep)

### 2.4 Boundary Conditions

**Interior Surface (x = 0):** Robin BC (convection + radiation)

$$-k \frac{\partial T}{\partial x}\bigg|_{x=0} = h_{c,i} (T_{zone} - T_{surf,i}) + q_{rad,i}$$

Discretized using ghost node:

$$T_{-1} = T_1 - \frac{2\Delta x}{k}(h_{c,i}(T_{zone} - T_0) + q_{rad,i})$$

Substituting into BTCS equation:

$$B_0 \cdot T_0^{n+1} + C_0 \cdot T_1^{n+1} = D_0 + \frac{2Fo \cdot \Delta x}{k}(h_{c,i}(T_{zone} - T_0^{n+1}) + q_{rad,i})$$

**Exterior Surface (x = L):** Robin BC with sol-air temperature

$$-k \frac{\partial T}{\partial x}\bigg|_{x=L} = h_{c,e} (T_{sol-air} - T_{surf,e})$$

Where:
- T_sol-air = T_outdoor + (α_solar · G_solar) / h_c,e - ΔT_LR (longwave radiation)

Discretized:

$$T_{N+1} = T_{N-1} + \frac{2\Delta x}{k}(h_{c,e}(T_{sol-air} - T_N))$$

### 2.5 Layer Interfaces

At interface between layer j and layer j+1:

**Continuity Conditions:**
1. Temperature: T_j = T_{j+1}
2. Heat flux: -k_j · ∂T/∂x|_j = -k_{j+1} · ∂T/∂x|_{j+1}

**Harmonic Mean for Conductivity:**

$$k_{interface} = \frac{2 k_j k_{j+1}}{k_j + k_{j+1}}$$

This ensures flux continuity at interfaces.

### 2.6 Stability Analysis

**Implicit scheme:** Unconditionally stable (no Δt restriction)

**Accuracy constraint:** For building walls, recommend:
- Δt ≤ 600 s (10 min) for dynamic accuracy
- Δx ≤ 0.01 m (1 cm) for spatial resolution
- Fo = α·Δt/Δx² ≤ 5 (accuracy, not stability)

**Typical values for concrete:**
- α = 6.9×10⁻⁷ m²/s
- Δt = 3600 s (1 hour)
- Δx = 0.015 m (1.5 cm for 150mm wall with 10 nodes)
- Fo = 0.011 (well within accuracy limit)

---

## 3. Implementation Architecture

### 3.1 Module Structure

```
src/physics/
├── fd_discretization.rs    # Wall discretization (Task 2)
├── fd_solver.rs            # Implicit solver (Task 3)
├── fd_surface_balance.rs   # Surface coupling (Task 4)
└── fd_thermal_model.rs     # Thermal model interface (Task 5)
```

### 3.2 Data Structures

```rust
/// Finite difference wall discretization
pub struct WallDiscretization {
    pub num_layers: usize,
    pub nodes_per_layer: usize,
    pub total_nodes: usize,
    pub layer_thickness: Vec<f64>,      // [m]
    pub node_positions: Vec<f64>,       // [m] from interior
    pub node_volumes: Vec<f64>,         // [m³] per node (control volume)
    pub density: Vec<f64>,              // [kg/m³] per node
    pub specific_heat: Vec<f64>,        // [J/kg·K] per node
    pub conductivity: Vec<f64>,         // [W/m·K] per node
    pub interface_conductivity: Vec<f64>, // [W/m·K] at interfaces
}

/// Implicit FD solver state
pub struct ImplicitFDSolver {
    pub discretization: WallDiscretization,
    pub temperatures: Vec<f64>,         // Current T [°C]
    pub coefficients: TridiagonalCoeffs, // A, B, C coefficients
}

/// Tridiagonal matrix coefficients
pub struct TridiagonalCoeffs {
    pub lower: Vec<f64>,    // A (sub-diagonal)
    pub main: Vec<f64>,     // B (main diagonal)
    pub upper: Vec<f64>,    // C (super-diagonal)
    pub rhs: Vec<f64>,      // D (right-hand side)
}

/// Surface boundary conditions
pub struct SurfaceBC {
    pub h_conv: f64,        // Convective coefficient [W/m²·K]
    pub h_rad: f64,         // Radiative coefficient [W/m²·K]
    pub q_solar: f64,       // Solar flux [W/m²]
    pub t_zone: f64,        // Zone air temperature [°C]
    pub t_sky: f64,         // Sky temperature [°C]
}
```

### 3.3 Algorithm Flow

```
For each timestep:
1. Update weather (T_outdoor, solar, wind)
2. Calculate sol-air temperature (exterior BC)
3. Assemble tridiagonal system (A, B, C, D)
4. Apply boundary conditions (modify A[0], C[0], A[N], B[N], D)
5. Solve using Thomas algorithm (TDMA)
6. Update temperatures
7. Calculate surface heat flux
8. Couple to zone air heat balance
```

---

## 4. Thomas Algorithm (TDMA)

The tridiagonal system is solved in O(n) operations:

```rust
pub fn thomas_algorithm(lower: &[f64], main: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Forward sweep
    c_prime[0] = upper[0] / main[0];
    d_prime[0] = rhs[0] / main[0];

    for i in 1..n {
        let denom = main[i] - lower[i] * c_prime[i-1];
        if i < n-1 {
            c_prime[i] = upper[i] / denom;
        }
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i-1]) / denom;
    }

    // Back substitution
    x[n-1] = d_prime[n-1];
    for i in (0..n-1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i+1];
    }

    x
}
```

---

## 5. Verification Criteria

### 5.1 Unit Tests

1. **Steady-state conduction:** Compare to analytical solution (linear T profile)
2. **Transient step response:** Compare to exact solution for semi-infinite solid
3. **Multi-layer wall:** Verify flux continuity at interfaces
4. **Energy balance:** ΣE_in - ΣE_out = ΔE_stored (error < 0.01%)

### 5.2 Integration Tests

1. **Case 900 annual simulation:** Compare to EnergyPlus (target ±5%)
2. **Timestep sensitivity:** Δt = 1hr, 15min, 1min → convergence
3. **Node sensitivity:** 5, 10, 20 nodes/layer → convergence

### 5.3 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Single-config latency | <2 ms | 100-config batch |
| Throughput | ≥500 configs/sec | Population run |
| Memory per simulation | <10 MB | Peak allocation |

---

## 6. References

[1] Chen, Y., & Athienitis, A.K. (2008). "A Method for Calculating Conduction Transfer Functions of Multi-Layer Walls." *Journal of Building Physics*, 32(1), 57-75.

[2] Hittle, D.C., & Anderson, R.K. (2003). "Comparison of Conduction Transfer Function Coefficient Calculation Methods." *ASHRAE Transactions*, 109(1), 174-183.

[3] Wang, S., & Chen, Y. (2003). "Transient Heat Transfer through Multi-Layer Walls with CTF Method." *Energy and Buildings*, 35(7), 675-684.

---

*Document created: 2026-03-17*
*Phase 25-03 Task 1 Deliverable*
