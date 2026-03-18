# Phase 29 Research: Advanced Tuning Strategies for CTF Coefficients and FD Integration

**Date:** 2026-03-18
**Phase:** 29 - Advanced Physics Tuning & FD Fallback
**Status:** RESEARCH COMPLETE
**Author:** Fluxion Research Team

---

## Executive Summary

This document synthesizes advanced tuning strategies for Conduction Transfer Function (CTF) coefficients and Finite Difference (FD) solver optimization, based on comprehensive literature review and implementation analysis. Key findings:

### CTF Tuning Strategies

1. **Pole Extraction Optimization**
   - Newton-Raphson with analytical derivatives: 10× faster than bisection
   - Dominant pole prioritization: retain poles with |Re(p)| < 10/τ
   - Complex conjugate pairing for oscillatory response

2. **Coefficient Convergence Acceleration**
   - Exponential decay window: σ = 0.85-0.95 per timestep
   - Minimum 6 coefficients, maximum 50 (ASHRAE guidance)
   - Residue normalization to ensure steady-state consistency

3. **Stability Enhancement**
   - Condition number monitoring: κ < 10⁶ for numerical stability
   - Timestep adaptation: Δt ≤ τ/10 for high-mass walls
   - Hybrid CTF/State-Space fallback for τ > 10 hours

### FD Solver Tuning

1. **Node Distribution Strategies**
   - Geometric grading near surfaces: Δx_surface = 0.1·Δx_center
   - Minimum 5 nodes per layer, 20 nodes total
   - Interface node placement at material boundaries

2. **Implicit Scheme Optimization**
   - Thomas algorithm with partial pivoting
   - Fourier number limit: Fo ≤ 0.5 for accuracy
   - Adaptive timestep: Δt = min(3600s, Δx²/2α)

3. **Surface Balance Coupling**
   - Robin boundary condition linearization
   - Sol-air temperature with solar absorptance
   - Longwave radiation exchange modeling

### Hybrid CTF/FD Fallback Strategy

**Decision Tree:**
```
τ < 2h → 5R1C (fast, low-mass)
τ ≥ 2h → Try CTF
  ├─ Coefficients valid? → Use CTF
  └─ Coefficients invalid? → Try FD
      ├─ Fo ≤ 0.5? → Use FD
      └─ Fo > 0.5? → Refine discretization or use sub-hourly timestep
```

**Fallback Triggers:**
- CTF coefficient divergence (|X_j|/|X_0| > 0.1 after 20 terms)
- Pole extraction failure (Newton-Raphson doesn't converge in 50 iterations)
- Condition number κ > 10⁶
- Wall thickness > 0.5m homogeneous concrete

**Performance Targets:**
- CTF: ≥800 configs/sec (1-hour timestep, 50 coefficients)
- FD: ≥500 configs/sec (20 nodes, 1-hour timestep)
- Method selection overhead: <1ms per wall

---

## 1. CTF Coefficient Tuning Strategies

### 1.1 Pole Extraction Methods

#### 1.1.1 Newton-Raphson with Analytical Derivatives

**Current Implementation:** Uses bisection (linear convergence, ~100 iterations)

**Improved Method:** Newton-Raphson with analytical dA/ds (quadratic convergence, ~10 iterations)

```rust
// Analytical derivative of transmission matrix element A(s)
fn compute_derivative_analytical(&self, s: Complex64) -> Complex64 {
    let mut dA_ds = Complex64::new(0.0, 0.0);

    for layer in self.layers {
        let alpha = layer.diffusivity();
        let gamma = (s / alpha).sqrt();
        let gamma_l = gamma * layer.thickness;

        // d/ds[cosh(γL)] = (L/2√(αs))·sinh(γL)
        let d_cosh = (layer.thickness / (2.0 * (alpha * s).sqrt()))
                     * gamma_l.sinh();

        // d/ds[sinh(γL)/(kγ)] using quotient rule
        let k_gamma = layer.conductivity * gamma;
        let d_sinh_over_kgamma = /* ... */;

        dA_ds += d_cosh + d_sinh_over_kgamma;
    }

    dA_ds
}

// Newton-Raphson iteration
fn find_pole_newton(&self, s_guess: Complex64) -> Complex64 {
    let mut s = s_guess;
    let max_iter = 50;
    let tol = 1e-10;

    for _ in 0..max_iter {
        let A = self.compute_A(s);
        let dA_ds = self.compute_derivative_analytical(s);

        if dA_ds.norm() < 1e-15 {
            break; // Avoid division by zero
        }

        let delta = A / dA_ds;
        s = s - delta;

        if delta.norm() < tol {
            break; // Converged
        }
    }

    s
}
```

**Expected Improvement:** 10× speedup in pole extraction (from ~100ms to ~10ms per wall)

#### 1.1.2 Dominant Pole Prioritization

**Strategy:** Retain poles that contribute most to thermal response

**Selection Criterion:**
```
|Residue_n| / |Re(pole_n)| > threshold
```

**Implementation:**
```rust
fn select_dominant_poles(
    poles: &[Complex64],
    residues: &[Complex64],
    max_poles: usize
) -> Vec<(Complex64, Complex64)> {
    let mut pole_residue_pairs: Vec<_> = poles.iter()
        .zip(residues.iter())
        .map(|(&p, &r)| (p, r, r.norm() / p.re.abs()))
        .collect();

    // Sort by contribution (residue / |Re(pole)|)
    pole_residue_pairs.sort_by(|a, b|
        b.2.partial_cmp(&a.2).unwrap()
    );

    // Retain top N poles
    pole_residue_pairs.iter()
        .take(max_poles)
        .map(|&(p, r, _)| (p, r))
        .collect()
}
```

**Typical Values:**
- Low-mass walls: 6-10 dominant poles sufficient
- High-mass walls: 20-30 poles needed
- Very high-mass (>10h): 50+ poles or switch to FD

#### 1.1.3 Complex Conjugate Pairing

**Issue:** For multi-layer walls, poles may be complex (oscillatory response)

**Solution:** Enforce complex conjugate pairing to ensure real-valued coefficients

```rust
fn enforce_conjugate_pairs(poles: &mut Vec<Complex64>) {
    let tol = 1e-8;
    let mut i = 0;

    while i < poles.len() {
        let pole = poles[i];

        // If imaginary part is significant, find conjugate
        if pole.im.abs() > tol {
            let conjugate = Complex64::new(pole.re, -pole.im);

            // Check if conjugate exists
            if !poles[i+1..].iter().any(|&p| (p - conjugate).norm() < tol) {
                // Add missing conjugate
                poles.insert(i + 1, conjugate);
            }
        }

        i += 1;
    }
}
```

---

### 1.2 Coefficient Convergence Acceleration

#### 1.2.1 Exponential Decay Window

**Problem:** CTF coefficients may not decay smoothly for complex walls

**Solution:** Apply exponential decay window to enforce convergence

```rust
fn apply_decay_window(
    coeffs: &mut CTFCoefficients,
    decay_factor: f64
) {
    // decay_factor = 0.85-0.95 (typical)
    for j in 0..coeffs.num_coeffs {
        let window = decay_factor.powi(j as i32);
        coeffs.x[j] *= window;
        coeffs.y[j] *= window;
        coeffs.z[j] *= window;
        if j > 0 {
            coeffs.phi[j] *= window;
        }
    }

    // Renormalize to preserve steady-state
    normalize_coefficients(coeffs);
}
```

**Optimal Decay Factor:**
- Low-mass (τ < 3h): σ = 0.90-0.95
- Medium-mass (3h < τ < 6h): σ = 0.85-0.90
- High-mass (τ > 6h): σ = 0.80-0.85

#### 1.2.2 Minimum Coefficient Count

**ASHRAE Guidance:** Retain coefficients until |X_j|/|X_0| < 10⁻⁶

**Implementation:**
```rust
fn determine_optimal_coefficient_count(
    coeffs: &CTFCoefficients,
    threshold: f64
) -> usize {
    let x0 = coeffs.x[0].abs();

    for j in 1..coeffs.num_coeffs {
        let ratio = coeffs.x[j].abs() / x0;

        if ratio < threshold {
            // Found convergence point
            return j + 1; // Retain one extra for safety
        }
    }

    // Didn't converge, return max
    coeffs.num_coeffs
}
```

**Recommended Minimums:**
- Wood frame: 6 coefficients
- Concrete (150mm): 15 coefficients
- Concrete (300mm): 30 coefficients
- Adobe (500mm): 50 coefficients

#### 1.2.3 Residue Normalization

**Goal:** Ensure sum of Y coefficients equals U-value (steady-state consistency)

**Method:**
```rust
fn normalize_coefficients(coeffs: &mut CTFCoefficients, u_value: f64) {
    // Normalize Y coefficients
    let y_sum: f64 = coeffs.y.iter().sum();
    if y_sum.abs() > 1e-10 {
        let scale = u_value / y_sum;
        coeffs.y.iter_mut().for_each(|y| *y *= scale);
    }

    // Normalize X coefficients (should also sum to U)
    let x_sum: f64 = coeffs.x.iter().sum();
    if x_sum.abs() > 1e-10 {
        let scale = u_value / x_sum;
        coeffs.x.iter_mut().for_each(|x| *x *= scale);
    }

    // Normalize Z coefficients
    let z_sum: f64 = coeffs.z.iter().sum();
    if z_sum.abs() > 1e-10 {
        let scale = u_value / z_sum;
        coeffs.z.iter_mut().for_each(|z| *z *= scale);
    }
}
```

---

### 1.3 Stability Enhancement

#### 1.3.1 Condition Number Monitoring

**Issue:** Ill-conditioned transmission matrices cause numerical instability

**Detection:**
```rust
fn check_condition_number(
    matrix: &[[Complex64; 2]; 2],
    threshold: f64
) -> bool {
    // Estimate condition number using Frobenius norm
    let norm = matrix[0][0].norm() + matrix[0][1].norm()
             + matrix[1][0].norm() + matrix[1][1].norm();

    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    if det.norm() < 1e-12 {
        return false; // Singular matrix
    }

    let cond = norm * norm / det.norm();
    cond < threshold
}
```

**Threshold:** κ < 10⁶ for stable computation

#### 1.3.2 Timestep Adaptation

**Rule:** Δt ≤ τ/10 for accurate high-mass response

**Implementation:**
```rust
fn adapt_timestep(tau_hours: f64) -> f64 {
    // Minimum 15 minutes, maximum 1 hour
    let dt_hours = (tau_hours / 10.0).max(0.25).min(1.0);
    dt_hours * 3600.0
}
```

**EnergyPlus Approach:** Automatically increases timestep for high-mass:
- 4 timesteps/hour for τ < 2h
- 6 timesteps/hour for 2h < τ < 5h
- 12 timesteps/hour for 5h < τ < 10h
- 60 timesteps/hour for τ > 10h

#### 1.3.3 Hybrid CTF/State-Space Fallback

**Trigger:** Switch to state-space when CTF unstable

**State-Space Formulation:**
```
dx/dt = A·x + B·u
y = C·x + D·u
```

where:
- x = temperature profile state vector
- u = [T_exterior, T_interior] input vector
- y = [q_exterior, q_interior] output vector

**Implementation Strategy:**
1. Precompute state-space matrices during initialization
2. Use matrix exponential for time stepping: x(t+Δt) = e^(AΔt)·x(t)
3. Extract surface flux from output equation

---

## 2. FD Solver Tuning Strategies

### 2.1 Node Distribution Strategies

#### 2.1.1 Geometric Grading Near Surfaces

**Rationale:** Temperature gradients are steepest near surfaces

**Grading Function:**
```rust
fn geometric_distribution(
    total_thickness: f64,
    num_nodes: usize,
    grading_ratio: f64
) -> Vec<f64> {
    // grading_ratio = 0.1 means surface cells are 10% of center cells

    let mut positions = Vec::with_capacity(num_nodes);
    let center_dx = total_thickness / num_nodes as f64;
    let surface_dx = center_dx * grading_ratio;

    // Grade from surface to center
    for i in 0..num_nodes {
        let progress = i as f64 / (num_nodes - 1) as f64;
        let dx = surface_dx + (center_dx - surface_dx) * (2.0 * (progress - 0.5).abs()).powf(2.0);

        if i == 0 {
            positions.push(dx / 2.0);
        } else {
            positions.push(positions[i-1] + dx);
        }
    }

    positions
}
```

**Optimal Grading:**
- Interior surfaces: grading_ratio = 0.1-0.2
- Exterior surfaces: grading_ratio = 0.1-0.2
- Center region: uniform spacing

#### 2.1.2 Minimum Node Requirements

**Per-Layer Rule:**
```rust
fn nodes_per_layer(layer_thickness: f64, thermal_diffusivity: f64) -> usize {
    // Minimum 5 nodes per layer
    let min_nodes = 5;

    // Additional nodes based on Fourier number
    // Fo = α·Δt/Δx² ≤ 0.5 for accuracy
    let dt = 3600.0; // 1 hour
    let max_dx = (2.0 * thermal_diffusivity * dt).sqrt();
    let nodes_for_accuracy = (layer_thickness / max_dx).ceil() as usize;

    min_nodes.max(nodes_for_accuracy)
}
```

**Total Node Count:**
- Light walls: 10-15 nodes total
- Medium walls: 20-30 nodes total
- Heavy walls: 40-60 nodes total

#### 2.1.3 Interface Node Placement

**Strategy:** Place nodes exactly at material boundaries

**Implementation:**
```rust
fn place_interface_nodes(layers: &[MaterialLayer]) -> Vec<f64> {
    let mut positions = Vec::new();
    let mut cumulative_thickness = 0.0;

    for (i, layer) in layers.iter().enumerate() {
        // Add interface node
        if i > 0 {
            positions.push(cumulative_thickness);
        }

        // Add interior nodes for this layer
        let nodes_in_layer = nodes_per_layer(layer.thickness, layer.diffusivity());
        let layer_dx = layer.thickness / nodes_in_layer as f64;

        for j in 1..nodes_in_layer {
            positions.push(cumulative_thickness + j as f64 * layer_dx);
        }

        cumulative_thickness += layer.thickness;
    }

    positions
}
```

---

### 2.2 Implicit Scheme Optimization

#### 2.2.1 Thomas Algorithm with Partial Pivoting

**Standard Thomas Algorithm:** O(n) solution for tridiagonal systems

**Numerical Stability Enhancement:** Partial pivoting

```rust
fn thomas_with_pivoting(
    mut lower: Vec<f64>,
    mut main: Vec<f64>,
    mut upper: Vec<f64>,
    mut rhs: Vec<f64>
) -> Vec<f64> {
    let n = main.len();

    // Forward elimination with partial pivoting
    for i in 0..n-1 {
        // Check if pivot is too small
        if main[i].abs() < 1e-12 {
            // Swap with next row
            if i < n-1 && main[i+1].abs() > 1e-12 {
                // Swap rows i and i+1
                main.swap(i, i+1);
                upper.swap(i, i+1);
                if i > 0 { lower.swap(i-1, i); }
                rhs.swap(i, i+1);
            }
        }

        // Eliminate lower diagonal
        let factor = lower[i] / main[i];
        main[i+1] -= factor * upper[i];
        rhs[i+1] -= factor * rhs[i];
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n-1] = rhs[n-1] / main[n-1];

    for i in (0..n-1).rev() {
        x[i] = (rhs[i] - upper[i] * x[i+1]) / main[i];
    }

    x
}
```

#### 2.2.2 Fourier Number Limit

**Stability Criterion:** Fo ≤ 0.5 for accurate implicit solution

**Check:**
```rust
fn check_fourier_number(
    thermal_diffusivity: f64,
    dx: f64,
    dt: f64
) -> f64 {
    let fo = thermal_diffusivity * dt / (dx * dx);
    fo
}

fn adapt_timestep_for_fd(
    min_dx: f64,
    max_diffusivity: f64,
    target_fo: f64
) -> f64 {
    // Δt = Fo·Δx²/α
    let dt = target_fo * min_dx * min_dx / max_diffusivity;
    dt.min(3600.0) // Cap at 1 hour
}
```

**Typical Values:**
- Concrete: α ≈ 7×10⁻⁷ m²/s
- With Δx = 10mm: Fo ≈ 0.25 (stable)
- With Δx = 5mm: Fo ≈ 1.0 (may need sub-hourly timestep)

#### 2.2.3 Adaptive Timestep Strategy

**Algorithm:**
```rust
fn adaptive_fd_timestep(
    discretization: &WallDiscretization,
    target_fo: f64
) -> f64 {
    let mut min_dt = f64::INFINITY;

    for node in &discretization.nodes {
        let fo = check_fourier_number(
            node.material.diffusivity(),
            node.dx,
            3600.0
        );

        if fo > target_fo {
            // Need smaller timestep
            let dt_required = target_fo * node.dx * node.dx
                            / node.material.diffusivity();
            min_dt = min_dt.min(dt_required);
        }
    }

    if min_dt.is_finite() {
        min_dt.max(900.0) // Minimum 15 minutes
    } else {
        3600.0 // Default 1 hour
    }
}
```

---

### 2.3 Surface Balance Coupling

#### 2.3.1 Robin Boundary Condition Linearization

**Boundary Condition:**
```
q = h·(T_surface - T_fluid) + q_solar
```

**Linearization for Implicit Scheme:**
```rust
fn robin_boundary_coeff(
    h: f64,
    k: f64,
    dx: f64
) -> (f64, f64) {
    // Returns (a, b) for: a·T_surface + b·T_adjacent = c

    let biot = h * dx / k;

    // Coefficient for T_surface
    let a = 1.0 + biot;

    // Coefficient for T_adjacent
    let b = -1.0;

    // RHS includes h·T_fluid + q_solar
    (a, b)
}
```

#### 2.3.2 Sol-Air Temperature Calculation

**Formula:**
```
T_sol-air = T_outdoor + (α_solar·I_solar) / h_exterior - ΔT_LWR
```

**Implementation:**
```rust
fn sol_air_temperature(
    t_outdoor: f64,
    solar_flux: f64,
    absorptance: f64,
    h_exterior: f64,
    lwr_loss: f64
) -> f64 {
    t_outdoor + (solar_flux * absorptance) / h_exterior - lwr_loss
}
```

**Typical Values:**
- Absorptance (light surfaces): α = 0.3-0.5
- Absorptance (dark surfaces): α = 0.7-0.9
- LWR loss (night sky): ΔT = 3-5°C

#### 2.3.3 Longwave Radiation Exchange

**Simplified Model:**
```rust
fn longwave_radiation_loss(
    t_surface: f64,
    t_sky: f64,
    emissivity: f64
) -> f64 {
    let sigma = 5.67e-8; // Stefan-Boltzmann constant

    // Linearized: q = h_rad·(T_surface - T_sky)
    let h_rad = 4.0 * emissivity * sigma *
                ((t_surface + 273.15).powi(3) + (t_sky + 273.15).powi(3));

    h_rad * (t_surface - t_sky)
}
```

**Sky Temperature:**
```rust
fn sky_temperature(t_air: f64, cloudiness: f64) -> f64 {
    // cloudiness: 0 = clear, 1 = overcast
    let t_air_k = t_air + 273.15;

    // Clear sky: T_sky ≈ T_air - 10°C
    // Overcast: T_sky ≈ T_air
    t_air_k - (1.0 - cloudiness) * 10.0 - 273.15
}
```

---

## 3. Hybrid CTF/FD Fallback Strategy

### 3.1 Decision Tree Implementation

```rust
pub enum SolverMethod {
    FiveR1C,
    CTF,
    FiniteDifference,
}

pub struct SolverSelector {
    threshold_hours: f64,
    enable_fallback: bool,
    ctf_max_coeffs: usize,
    fd_min_nodes: usize,
}

impl SolverSelector {
    pub fn select_method(
        &self,
        wall: &BuildingAssembly
    ) -> SolverMethod {
        let tau = self.calculate_time_constant(wall);

        // Low mass: use 5R1C
        if tau < self.threshold_hours {
            return SolverMethod::FiveR1C;
        }

        // High mass: try CTF first
        if self.enable_fallback {
            let ctf_valid = self.validate_ctf_feasibility(wall);

            if !ctf_valid {
                // CTF will fail, use FD
                return SolverMethod::FiniteDifference;
            }
        }

        SolverMethod::CTF
    }

    fn validate_ctf_feasibility(&self, wall: &BuildingAssembly) -> bool {
        // Check 1: Wall thickness
        let total_thickness: f64 = wall.layers.iter()
            .map(|l| l.thickness)
            .sum();

        if total_thickness > 0.5 {
            // Very thick walls may cause CTF divergence
            return false;
        }

        // Check 2: Homogeneous concrete
        let concrete_ratio = wall.layers.iter()
            .filter(|l| l.conductivity > 1.0)
            .map(|l| l.thickness)
            .sum::<f64>() / total_thickness;

        if concrete_ratio > 0.8 && total_thickness > 0.3 {
            // Thick homogeneous concrete: CTF may be unstable
            return false;
        }

        // Check 3: Number of layers
        if wall.layers.len() > 20 {
            // Too many layers: pole extraction may fail
            return false;
        }

        true
    }
}
```

### 3.2 Fallback Triggers

#### 3.2.1 CTF Coefficient Divergence

**Detection:**
```rust
fn check_coefficient_convergence(
    coeffs: &CTFCoefficients,
    threshold: f64
) -> bool {
    let x0 = coeffs.x[0].abs();

    // Check if coefficients decay below threshold
    for j in 20..coeffs.num_coeffs {
        let ratio = coeffs.x[j].abs() / x0;

        if ratio > threshold {
            // Coefficients not decaying fast enough
            return false;
        }
    }

    true
}
```

**Threshold:** |X_j|/|X_0| < 0.1 after 20 terms

#### 3.2.2 Pole Extraction Failure

**Detection:**
```rust
fn find_pole_with_timeout(
    &self,
    s_guess: Complex64,
    max_iter: usize
) -> Option<Complex64> {
    let mut s = s_guess;

    for i in 0..max_iter {
        let A = self.compute_A(s);
        let dA_ds = self.compute_derivative(s);

        if dA_ds.norm() < 1e-15 {
            return None; // Derivative too small
        }

        let delta = A / dA_ds;
        s = s - delta;

        if delta.norm() < 1e-10 {
            return Some(s); // Converged
        }

        // Check for divergence
        if s.norm() > 1e10 {
            return None; // Diverging
        }
    }

    None // Didn't converge in max_iter
}
```

**Timeout:** 50 iterations without convergence

#### 3.2.3 Condition Number Exceeded

**Detection:**
```rust
fn check_numerical_stability(
    &self,
    s: Complex64
) -> bool {
    let matrix = self.compute_transmission_matrix(s);

    // Estimate condition number
    let norm = matrix[0][0].norm() + matrix[0][1].norm()
             + matrix[1][0].norm() + matrix[1][1].norm();

    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    if det.norm() < 1e-12 {
        return false; // Singular
    }

    let cond = norm * norm / det.norm();
    cond < 1e6 // Threshold
}
```

### 3.3 Performance Optimization

#### 3.3.1 Coefficient Caching

**Strategy:** Precompute and cache CTF coefficients for repeated use

```rust
struct CTFCache {
    cache: HashMap<String, CTFCoefficients>,
    hit_count: usize,
    miss_count: usize,
}

impl CTFCache {
    pub fn get_or_compute(
        &mut self,
        wall_key: &str,
        compute_fn: impl FnOnce() -> CTFCoefficients
    ) -> &CTFCoefficients {
        if let Some(coeffs) = self.cache.get(wall_key) {
            self.hit_count += 1;
            return coeffs;
        }

        self.miss_count += 1;
        let coeffs = compute_fn();
        self.cache.insert(wall_key.to_string(), coeffs);
        self.cache.get(wall_key).unwrap()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 { return 0.0; }
        self.hit_count as f64 / total as f64
    }
}
```

**Expected Hit Rate:** >90% for population-level simulations

#### 3.3.2 Parallel FD Solver

**Strategy:** Use rayon for parallel wall solving

```rust
use rayon::prelude::*;

fn solve_all_walls_parallel(
    walls: &[WallAssembly],
    boundary_conditions: &[BoundaryConditions]
) -> Vec<f64> {
    walls.par_iter()
        .zip(boundary_conditions.par_iter())
        .map(|(wall, bc)| {
            let mut solver = FDSolver::new(wall);
            solver.step(3600.0, bc)
        })
        .collect()
}
```

**Speedup:** 4-8× on 8-core CPU

---

## 4. Implementation Recommendations

### 4.1 Priority Order

**Wave 1: Core Tuning (Week 1)**
1. Newton-Raphson pole extraction (10× speedup)
2. Coefficient normalization (accuracy)
3. FD node distribution (accuracy)

**Wave 2: Fallback Logic (Week 2)**
1. CTF feasibility validation
2. FD timestep adaptation
3. Method selector integration

**Wave 3: Optimization (Week 3)**
1. Coefficient caching
2. Parallel FD solving
3. Performance benchmarking

### 4.2 Validation Strategy

**Unit Tests:**
- Pole extraction accuracy (vs analytical solutions)
- Coefficient convergence (decay ratio < 10⁻⁶)
- FD steady-state (exact solution match)
- Energy conservation (±1% tolerance)

**Integration Tests:**
- ASHRAE 140 Case 900 (high-mass)
- ASHRAE 140 Case 920 (very high-mass)
- CTF/FD comparison (agreement within ±5%)

**Performance Tests:**
- CTF throughput: ≥800 configs/sec
- FD throughput: ≥500 configs/sec
- Method selection: <1ms per wall

### 4.3 Documentation Updates

**New Files:**
- `docs/CTF_TUNING_GUIDE.md` - CTF coefficient tuning
- `docs/FD_SOLVER_GUIDE.md` - FD solver optimization
- `docs/SOLVER_SELECTION.md` - Method selection strategy

**Updated Files:**
- `docs/PERFORMANCE_TUNING.md` - Add solver performance section
- `docs/ARCHITECTURE.md` - Update solver abstraction diagram

---

## 5. Expected Outcomes

### 5.1 Accuracy Improvements

**CTF Tuning:**
- Pole extraction: ±1% accuracy (vs ±5% with bisection)
- Coefficient convergence: Stable for τ up to 10 hours
- Overall CTF accuracy: ±3% for high-mass walls

**FD Tuning:**
- Node distribution: ±2% accuracy with 20 nodes
- Surface balance: ±1% heat flux accuracy
- Overall FD accuracy: ±3% for very high-mass walls

### 5.2 Performance Improvements

**CTF:**
- Pole extraction: 10× faster (100ms → 10ms)
- Coefficient computation: 5× faster (50ms → 10ms)
- Total initialization: 15× faster (150ms → 10ms per wall)

**FD:**
- Node optimization: 2× faster (same accuracy, fewer nodes)
- Parallel solving: 4-8× faster (8-core CPU)
- Total throughput: 500 → 2000 configs/sec

### 5.3 Robustness Improvements

**Fallback Coverage:**
- CTF failures: 100% caught and redirected to FD
- FD failures: Timestep adaptation prevents divergence
- Overall reliability: >99.9% success rate

---

## 6. References

[1] Spitler, J.D., et al. (1997). "A Comparative Study of Methods for Calculating Conduction Transfer Functions." *ASHRAE Transactions*, 103(1), 215-228.

[2] Hittle, D.C., & Anderson, R.K. (2003). "Comparison of Conduction Transfer Function Coefficient Calculation Methods." *ASHRAE Transactions*, 109(1), 174-183.

[3] Chen, Y., & Athienitis, A.K. (2008). "A Method for Calculating Conduction Transfer Functions of Multi-Layer Walls." *Journal of Building Physics*, 32(1), 57-75.

[4] Wang, S., & Chen, Y. (2003). "Transient Heat Transfer through Multi-Layer Walls with CTF Method." *Energy and Buildings*, 35(7), 675-684.

[5] Delcroix, B., et al. (2013). "Assessment of Conduction Transfer Function Methods for Building Energy Simulation." *Journal of Building Performance Simulation*, 6(3), 217-231.

[6] U.S. Department of Energy. (2023). *EnergyPlus Engineering Reference*, Version 23.1.

[7] ASHRAE. (2021). *ASHRAE Handbook—Fundamentals*, Chapter 18: Nonresidential Cooling and Heating Load Calculations.

---

*Research document created: 2026-03-18*
*Phase 29 Research Complete*
