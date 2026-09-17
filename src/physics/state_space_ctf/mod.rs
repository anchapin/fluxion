#![allow(clippy::needless_range_loop)]
//! State-space method for CTF coefficient calculation (Seem 1987).
//!
//! This implements the same algorithm used by EnergyPlus internally:
//! discretize the wall into finite-difference nodes, build state-space
//! matrices (A, B, C, D), compute the matrix exponential, and extract
//! CTF coefficients (X, Y, Z, Φ) using Seem's method.
//!
//! References:
//! - Seem, J.E. "Modeling of Heat Transfer in Buildings", PhD Dissertation,
//!   University of Wisconsin-Madison, 1987. Equations 2.1.12-2.1.26.
//! - EnergyPlus source code: Construction.cc (calculateExponentialMatrix,
//!   calculateInverseMatrix, calculateGammas, calculateFinalCoefficients)

use super::ctf_coefficients::{CTFCoefficients, CTFMaterial};

// Note: the previous version of `matrix_exponential_faer` used `faer`'s public
// `evd_real` for the eigendecomposition, but the eigendecomposition is
// numerically unstable for the state-space matrices with clustered eigenvalues
// that we encounter in multi-layer walls. The current implementation uses
// the in-tree Schur decomposition (Householder + Francis QR) followed by
// Higham's Pade [13/13] scaling-and-squaring on the small Schur form, which
// is robust for clustered eigenvalues.

/// Surface film resistances [m²K/W] (ASHRAE 140 standard values).
const R_SI: f64 = 0.125; // Interior film
const R_SE: f64 = 0.044; // Exterior film

/// Minimum and maximum nodes per material layer.
///
/// E+ uses 1-18 nodes per layer, based on the Fourier number criterion
/// N = max(1, ceil(thickness / sqrt(2*alpha*timestep))).
///
/// Previous MIN_NODES=6 caused artificially high surface conductances for thin
/// layers (e.g. Wood Siding at 0.009m → dx=0.0015m → h_surf=108.9 W/m²K,
/// 195x larger than U=0.556). This made Y₀ explode and caused Newton-Raphson
/// divergence in the CTF coupling solver.
///
/// With MIN_NODES=1, thin low-mass layers get 1 node, matching EnergyPlus.
const MIN_NODES: usize = 1;
const MAX_NODES: usize = 18;

/// Convergence limit for CTF coefficient iteration (ratio).
///
/// NOTE: This threshold is applied to the RESIDUAL of the partial sum ΣX → U_bare,
/// NOT to individual `e[j]` matrix-exponential terms. The `e[j]` series can converge
/// (via cancellation) long before the `s[j][k][j]` series has built up to balance
/// the steady-state sum. For multi-layer walls with slow modes, the `e[j]` series
/// hits ~1e-13 by inum=9, but `Σs` is still short of U_bare by ~99%.
///
/// EnergyPlus evaluates convergence on the residual of the ΣX sum, not on e[j].
const CONVRG_LIM: f64 = 1.0e-3;

/// Minimum number of CTF terms to compute before checking convergence.
///
/// Must be high enough to capture the slowest eigenmode of A_exp for multi-layer
/// walls. After the boundary lumping fix (cap = 1.5*rho*cp*dx), the eigenvalues
/// of A_exp are realistic, and high-mass walls have modes with time constants
/// of 10-50 hours. 50 terms at 1-hour timestep = 50 hours of history.
const MIN_CTF_TERMS: usize = 20;

/// Maximum number of CTF terms before giving up.
const MAX_CTF_TERMS: usize = 200;

// ==================== FlatMatrix ====================
/// Flat matrix representation: row-major flat storage with explicit stride.
///
/// This replaces `Vec<Vec<f64>>` which suffers from:
///
/// 1. **Cache locality**: Vec<Vec> has N separate heap allocations (one per row),
///    causing poor cache utilization for matrix operations. FlatMatrix keeps
///    all data in a single allocation.
///
/// 2. **Aliasing safety**: With Vec<Vec>, two matrices can share inner Vec
///    references, causing subtle read-during-write bugs. FlatMatrix's
///    data is fully owned and distinct between instances.
///
/// 3. **Memory aliasing in Leverrier**: The r_prev/r_new update loop
///    `r_prev[i][j] = r_new[i][j]; r_new[i][j] = phi_r0[i][j];` corrupts
///    diagonal elements when both matrices reference the same buffer via
///    different row views. Using a snapshot clone of r_new fixes this.
///
/// The indexing formula is `data[i * stride + j]` for row i, column j.
#[derive(Debug, Clone)]
pub struct FlatMatrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    stride: usize,
}

impl FlatMatrix {
    pub fn new(rows: usize, cols: usize, stride: usize) -> Self {
        Self {
            data: vec![0.0; rows * stride],
            rows,
            cols,
            stride,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
            stride: cols,
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn from_vec_vec(m: &[Vec<f64>]) -> Self {
        if m.is_empty() {
            return Self::zeros(0, 0);
        }
        let rows = m.len();
        let cols = m[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for row in m {
            data.extend_from_slice(row);
        }
        Self {
            data,
            rows,
            cols,
            stride: cols,
        }
    }

    pub fn to_vec_vec(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let start = i * self.stride;
            result.push(self.data[start..start + self.cols].to_vec());
        }
        result
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[i * self.stride + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[i * self.stride + j] = v;
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn fill(&mut self, v: f64) {
        self.data.fill(v);
    }
}

impl FlatMatrix {
    pub fn as_ref_vec_vec(&self) -> Vec<Vec<f64>> {
        self.to_vec_vec()
    }
}

/// Compute CTF coefficients using the state-space method (Seem 1987).
///
/// This is the algorithm EnergyPlus actually uses internally — NOT pole/residue.
/// It avoids all the problems with degenerate pole structure that plague the
/// Laplace-domain approach for film-dominated walls.
///
/// **Two-phase approach matching E+:**
/// 1. Compute BARE-WALL CTFs (no films in state-space) → exact DC gain
/// 2. Scale coefficients analytically to include film resistances
///
/// The film scaling factor:
///   denom = 1 + U_bare · (R_ext_film + R_int_film)
///   X_filmed[i] = X_bare[i] / denom
///   Y_filmed[i] = Y_bare[i] / denom
///   Φ_filmed[i] = Φ_bare[i] / denom
///
/// At steady state: ΣX_filmed = U_bare / denom = U_filmed ✓
pub fn compute_state_space_ctf(layers: &[CTFMaterial], timestep: f64) -> CTFCoefficients {
    // Step 1: Determine number of nodes per layer (E+ method)
    let nodes_per_layer = compute_nodes_per_layer(layers, timestep);
    let total_nodes: usize = nodes_per_layer.iter().sum();

    if total_nodes == 0 {
        let mut coeffs = CTFCoefficients::new(timestep, 1);
        coeffs.num_coeffs = 1;
        coeffs.total_state_nodes = 0;
        let total_r_wall: f64 = layers.iter().map(|l| l.resistance()).sum();
        let u_filmed = 1.0 / (R_SI + total_r_wall + R_SE);
        coeffs.x[0] = u_filmed;
        coeffs.y[0] = u_filmed;
        coeffs.z[0] = u_filmed;
        coeffs.phi[0] = 0.0;
        return coeffs;
    }

    // Step 2: Build BARE-WALL state-space matrices (no films)
    let n = total_nodes;
    let (a_mat, b_mat, c_mat, d_mat) = build_state_space_matrices(layers, &nodes_per_layer, n);

    // Step 3: Compute matrix exponential e^(A·Δt)
    let a_exp = matrix_exponential(&a_mat, timestep);

    // Debug: inspect A matrix and Phi eigenvalues
    #[cfg(feature = "debug-physics")]
    {
        eprintln!("\n=== DIAGNOSTIC: A matrix (n={}) ===", n);
        for i in 0..n {
            eprintln!("  A[{}] = {:?}", i, a_mat[i]);
        }
        // Compute trace and determinant of Phi = exp(A*dt)
        let trace_phi: f64 = (0..n).map(|i| a_exp[i][i]).sum();
        eprintln!("  trace(Phi) = {:.6}", trace_phi);
        // Check diagonal dominance of Phi
        let mut max_offdiag = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    max_offdiag = max_offdiag.max(a_exp[i][j].abs());
                }
            }
        }
        eprintln!("  max |Phi_ij| (i≠j) = {:.6e}", max_offdiag);
        // Frobenius norm of Phi
        let frob_phi: f64 = a_exp.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!("  ||Phi||_F = {:.6}", frob_phi);
    }

    // Step 4: Compute matrix inverse A^(-1)
    let a_inv = matrix_inverse(&a_mat).expect("A matrix should be invertible for stable wall");

    // Step 5: Compute Gamma1 and Gamma2 (Seem eq 2.1.12 and 2.1.13)
    // Gamma1 = A_inv · (A_exp - I) · B   [n×2 result]
    // Gamma2 = A_inv · (Gamma1/Δt - B)   [n×2 result]
    let a_exp_minus_i = matrix_sub_identity(&a_exp);
    let temp = mat_mat_mul_col(&a_exp_minus_i, &b_mat); // n×2
    let gamma1 = mat_mat_mul_col(&a_inv, &temp); // n×2
    let gamma1_scaled = scale_columns(&gamma1, 1.0 / timestep);
    let gamma2_diff = matrix_sub_col(&gamma1_scaled, &b_mat);
    let gamma2 = mat_mat_mul_col(&a_inv, &gamma2_diff); // n×2

    // Debug: verify DC gain using full matrix formula: -C·A⁻¹·B + D
    #[cfg(feature = "debug-physics")]
    {
        let dc_gain_ct = {
            let ca_inv = mat_mul_gen(&c_mat, &a_inv); // 2×n × n×n = 2×n
            let ca_inv_b = mat_mul_gen(&ca_inv, &b_mat); // 2×n × n×2 = 2×2
            vec![
                vec![-ca_inv_b[0][0] + d_mat[0][0], -ca_inv_b[0][1] + d_mat[0][1]],
                vec![-ca_inv_b[1][0] + d_mat[1][0], -ca_inv_b[1][1] + d_mat[1][1]],
            ]
        };

        // Discrete-time DC gain: D + C*(I-Phi)^(-1)*(Gamma1+Gamma2)
        let dc_gain_dt = {
            let i_minus_phi = {
                let mut m = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        m[i][j] = if i == j { 1.0 } else { 0.0 } - a_exp[i][j];
                    }
                }
                m
            };
            let i_minus_phi_inv =
                matrix_inverse(&i_minus_phi).unwrap_or_else(|| vec![vec![0.0; n]; n]);
            let g12 = {
                let mut m = vec![vec![0.0; 2]; n];
                for i in 0..n {
                    m[i][0] = gamma1[i][0] + gamma2[i][0];
                    m[i][1] = gamma1[i][1] + gamma2[i][1];
                }
                m
            };
            let ci = mat_mul_gen(&c_mat, &i_minus_phi_inv); // 2×n
            let ci_g12 = mat_mul_gen(&ci, &g12); // 2×2
            vec![
                vec![ci_g12[0][0] + d_mat[0][0], ci_g12[0][1] + d_mat[0][1]],
                vec![ci_g12[1][0] + d_mat[1][0], ci_g12[1][1] + d_mat[1][1]],
            ]
        };

        let total_r_wall: f64 = layers.iter().map(|l| l.resistance()).sum();
        let u_bare_check = 1.0 / total_r_wall;
        eprintln!(
            "  DC gain CT: [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
            dc_gain_ct[0][0], dc_gain_ct[0][1], dc_gain_ct[1][0], dc_gain_ct[1][1]
        );
        eprintln!(
            "  DC gain DT (G1+G2): [[{:.6}, {:.6}], [{:.6}, {:.6}]], U_bare = {:.6}",
            dc_gain_dt[0][0], dc_gain_dt[0][1], dc_gain_dt[1][0], dc_gain_dt[1][1], u_bare_check
        );
        eprintln!("  Gamma1[0..2] = {:?}", &gamma1[..2.min(n)]);
        eprintln!("  Gamma2[0..2] = {:?}", &gamma2[..2.min(n)]);
        eprintln!(
            "  C = [{:.6}, {:.6}], D = [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
            c_mat[0][0],
            c_mat[1][n - 1],
            d_mat[0][0],
            d_mat[0][1],
            d_mat[1][0],
            d_mat[1][1]
        );
    }

    // Step 6: Compute bare-wall s0, s, and e coefficients (Seem step 5)
    let mut coeffs = compute_ctf_from_state_space(
        layers, &a_exp, &a_inv, &b_mat, &c_mat, &d_mat, &gamma1, &gamma2, n, timestep,
    );

    // Step 7: Apply film resistance scaling
    // Convert bare-wall CTFs to filmed CTFs analytically.
    // The bare-wall CTFs relate surface temperatures to conduction flux.
    // With films: T_surf = T_air - q·R_film.
    //
    // After uniform scaling by 1/denom, the DC gain becomes:
    //   DC_f = (ΣX/denom) / (1 + ΣΦ/denom) = ΣX / (denom + ΣΦ)
    //
    // We want DC_f = U_filmed = 1/(R_wall + R_SE + R_SI), so:
    //   denom = ΣX / U_filmed - ΣΦ
    let x_sum_bare: f64 = coeffs.x.iter().sum();
    let phi_sum_bare: f64 = coeffs.phi.iter().sum();
    let r_wall: f64 = layers.iter().map(|l| l.resistance()).sum();
    let u_filmed = 1.0 / (R_SI + r_wall + R_SE);
    let denom = x_sum_bare / u_filmed - phi_sum_bare;

    #[cfg(feature = "debug-physics")]
    {
        let u_bare = 1.0 / r_wall;
        eprintln!(
            "  Bare-wall: ΣX = {:.6}, U_bare = {:.6}",
            x_sum_bare, u_bare
        );
        eprintln!(
            "  Film scaling: denom = {:.6}, U_filmed = {:.6}",
            denom,
            u_bare / (1.0 + u_bare * (R_SE + R_SI))
        );
    }

    // Scale all CTF coefficients by the film factor
    for x in &mut coeffs.x {
        *x /= denom;
    }
    for y in &mut coeffs.y {
        *y /= denom;
    }
    for z in &mut coeffs.z {
        *z /= denom;
    }
    for phi in &mut coeffs.phi {
        *phi /= denom;
    }

    // Final verification
    #[cfg(feature = "debug-physics")]
    {
        let x_sum: f64 = coeffs.x.iter().sum();
        let _y_sum: f64 = coeffs.y.iter().sum();
        let phi_sum: f64 = coeffs.phi.iter().sum();
        let u_filmed = 1.0 / (R_SI + r_wall + R_SE);
        let dc_gain = x_sum / (1.0 + phi_sum);
        eprintln!("  Filmed: ΣX = {:.6}, ΣΦ = {:.6}", x_sum, phi_sum);
        eprintln!(
            "  DC gain ΣX/(1+ΣΦ) = {:.6} (target U_filmed = {:.6}, err = {:.4}%)",
            dc_gain,
            u_filmed,
            (dc_gain / u_filmed - 1.0) * 100.0
        );
        eprintln!("  Φ[0:5] = {:?}", &coeffs.phi[..5.min(coeffs.num_coeffs)]);
    }

    coeffs
}

/// Determine number of finite-difference nodes per layer.
///
/// Uses the E+ criterion: dxn = sqrt(2·α·Δt) for stability,
/// then N = thickness/dxn, clamped to [MIN_NODES, MAX_NODES].
fn compute_nodes_per_layer(layers: &[CTFMaterial], timestep: f64) -> Vec<usize> {
    let all_lightweight = layers.iter().all(|layer| {
        let alpha = layer.diffusivity();
        let fo = alpha * timestep / (layer.thickness * layer.thickness);
        fo > 2.5
    });

    if all_lightweight {
        return vec![0; layers.len()];
    }

    layers
        .iter()
        .map(|layer| {
            let alpha = layer.diffusivity();
            let dxn = (2.0 * alpha * timestep).sqrt();
            if dxn < 1e-15 {
                return MIN_NODES;
            }
            let n = (layer.thickness / dxn).ceil() as usize;
            n.clamp(MIN_NODES, MAX_NODES)
        })
        .collect()
}

/// Build 1-D state-space matrices for bare-wall conduction (no films).
///
/// The state vector x contains nodal temperatures at cell centers.
/// Node 0 is nearest the exterior surface, node N-1 nearest the interior.
///
/// Inputs u = [T_ext_surf, T_int_surf] (wall surface temperatures, NOT air temps).
/// Outputs y = [q_ext, q_int] (conduction fluxes at surfaces).
///
/// Boundary nodes use half-cell discretization:
///   Node 0 at x=dx/2: flux from surface through half-cell distance
///   A[0][0] = -3α, A[0][1] = α, B[0][0] = 2α
///   Node N-1 at x=L-dx/2: symmetric on interior side
///   A[N-1][N-1] = -3α, A[N-1][N-2] = α, B[N-1][1] = 2α
/// Interior nodes: A[i][i] = -2α, A[i][i±1] = α
///
/// C/D matrices use k/dx_half = 2k/dx for surface-to-node-center flux.
#[allow(clippy::type_complexity)]
pub fn build_state_space_matrices(
    layers: &[CTFMaterial],
    nodes_per_layer: &[usize],
    n: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // A: n×n, B: n×2, C: 2×n, D: 2×2
    let mut a_mat = vec![vec![0.0; n]; n];
    let mut b_mat = vec![vec![0.0; 2]; n];
    let mut c_mat = vec![vec![0.0; n]; 2];
    let mut d_mat = vec![vec![0.0; 2]; 2];

    // Compute dx (node spacing) for each layer.
    //
    // FIXED: Phase 2 of Issue #951 — switch from half-cell scheme to E+'s
    // lumped-mass boundary scheme. This matches EnergyPlus Construction.cc
    // v25.2.0 exactly:
    //
    //   - dx = L/N (E+ uses N cells; with N nodes spaced at x=dx/2, 3dx/2, ...,
    //     the surface-to-first-node distance is dx/2)
    //   - Boundary nodes use `cap = 1.5 * rho * cp * dx` (lumped mass including
    //     the half-cell beyond the surface)
    //   - A[0,0] = -2*k*dxtmp = -(4/3)*alpha_node (vs old -3*alpha_node)
    //   - B[0,0] = +k*dxtmp   = +(2/3)*alpha_node (vs old 2*alpha_node)
    //   - C[0,0] = -k/dx/(N-1), D[0,0] = +k/dx/(N-1) (with (N-1) divisor
    //     matching E+'s surface-flux scaling)
    //
    // Reference: EnergyPlus Construction.cc v25.2.0
    //   calculateExponentialMatrix() — sets up A, B matrices
    //   calculateFinalCoefficients() — sets up C, D, s0, s coefficients
    let dx: Vec<f64> = layers
        .iter()
        .zip(nodes_per_layer.iter())
        .map(|(l, &nn)| {
            if nn > 1 {
                l.thickness / nn as f64
            } else {
                // Single node: full thickness (half-cell each side)
                l.thickness
            }
        })
        .collect();

    // Build A and B matrices
    let mut global_node = 0;

    for (layer_idx, layer) in layers.iter().enumerate() {
        let nn = nodes_per_layer[layer_idx];
        let dx_l = dx[layer_idx];
        let k = layer.conductivity;
        let rho = layer.density;
        let cp = layer.specific_heat;

        // Interior node dxtmp = 1 / (rho * cp * dx^2) (no lumping)
        let cap_interior = rho * cp * dx_l;
        let dxtmp_interior = 1.0 / dx_l / cap_interior;
        // Boundary node dxtmp with lumped mass cap = 1.5 * rho * cp * dx
        let cap_boundary = 1.5 * cap_interior;
        let dxtmp_boundary = 1.0 / dx_l / cap_boundary;

        for local_node in 0..nn {
            let i = global_node + local_node;

            let is_exterior_boundary = layer_idx == 0 && local_node == 0;
            let is_interior_boundary = layer_idx == layers.len() - 1 && local_node == nn - 1;

            if is_exterior_boundary {
                // E+ lumped-mass boundary scheme (Construction.cc):
                //   cap = 1.5 * rho * cp * dx; dxtmp = 1/(dx*cap)
                //   dT0/dt = -2*k*dxtmp*T0 + k*dxtmp*T1 + k*dxtmp*T_ext_surf
                a_mat[i][i] = -2.0 * k * dxtmp_boundary;
                if i + 1 < n {
                    a_mat[i][i + 1] = k * dxtmp_boundary;
                }
                b_mat[i][0] = k * dxtmp_boundary;
                b_mat[i][1] = 0.0;
            } else if is_interior_boundary {
                // E+ lumped-mass boundary scheme (interior side):
                //   dT_{N-1}/dt = k*dxtmp*T_{N-2} - 2*k*dxtmp*T_{N-1}
                //                  + k*dxtmp*T_int_surf
                a_mat[i][i] = -2.0 * k * dxtmp_boundary;
                if i > 0 {
                    a_mat[i][i - 1] = k * dxtmp_boundary;
                }
                b_mat[i][0] = 0.0;
                b_mat[i][1] = k * dxtmp_boundary;
            } else {
                // Interior node — check for layer interface
                let (is_interface, next_layer_idx) =
                    if local_node == nn - 1 && layer_idx < layers.len() - 1 {
                        (true, layer_idx + 1)
                    } else {
                        (false, 0)
                    };

                if is_interface {
                    // Interface node: average properties from adjacent layers.
                    // E+ uses `amatx = rk/dx/capavg` with `capavg = (cap_left + cap_right) / 2`.
                    let next_layer = &layers[next_layer_idx];
                    let dx_next = dx[next_layer_idx];
                    let capavg = 0.5
                        * (cap_interior + next_layer.density * next_layer.specific_heat * dx_next);
                    let alpha_left = k / (capavg * dx_l);
                    let alpha_right = next_layer.conductivity / (capavg * dx_next);

                    a_mat[i][i] = -alpha_left - alpha_right;
                    if i > 0 {
                        a_mat[i][i - 1] = alpha_left;
                    }
                    if i + 1 < n {
                        a_mat[i][i + 1] = alpha_right;
                    }
                } else {
                    // Standard interior node (E+ scheme):
                    //   dxtmp = 1/(rho*cp*dx)  (no 1.5x mass lumping)
                    //   A[i][i] = -2*k*dxtmp = -2*alpha_node
                    a_mat[i][i] = -2.0 * k * dxtmp_interior;
                    if i > 0 {
                        a_mat[i][i - 1] = k * dxtmp_interior;
                    }
                    if i + 1 < n {
                        a_mat[i][i + 1] = k * dxtmp_interior;
                    }
                }
                b_mat[i][0] = 0.0;
                b_mat[i][1] = 0.0;
            }
        }
        global_node += nn;
    }

    // C matrix (2×n): conduction fluxes at surfaces
    // D matrix (2×2): direct throughput from input temps
    //
    // E+ Construction.cc uses CMat = k*(N+1)/(N*dx) for the surface-to-node
    // conductance. This scaling makes the continuous-time DC gain exactly U_bare.
    //
    // Issue #951 FIX: The previous s0 formula used per-surface C-value selection
    // instead of the full C-matrix multiply. This caused sign inversion on s0[1][0].
    // The fix is in compute_ctf_from_state_space (s0 = C·Γ₂ + D), not in C/D scaling.
    let k_ext = layers.first().map(|l| l.conductivity).unwrap_or(1.0);
    let dx_ext = dx.first().unwrap_or(&1.0);
    let n_ext = nodes_per_layer.first().unwrap_or(&1);
    let h_surf_ext = k_ext * (*n_ext as f64 + 1.0) / (*n_ext as f64 * dx_ext);

    let k_int = layers.last().map(|l| l.conductivity).unwrap_or(1.0);
    let dx_int = dx.last().unwrap_or(&1.0);
    let n_int = nodes_per_layer.last().unwrap_or(&1);
    let h_surf_int = k_int * (*n_int as f64 + 1.0) / (*n_int as f64 * dx_int);

    c_mat[0][0] = -h_surf_ext;
    d_mat[0][0] = h_surf_ext;
    d_mat[0][1] = 0.0;

    c_mat[1][n - 1] = h_surf_int;
    d_mat[1][0] = 0.0;
    d_mat[1][1] = -h_surf_int;

    (a_mat, b_mat, c_mat, d_mat)
}

/// Compute CTF coefficients from state-space matrices using Seem's method.
///
/// Follows Seem (1987) Step 5 (pages 26-27) and Appendix C, matching
/// EnergyPlus Construction.cc SolutionDimensions=1 exactly.
///
/// Key algorithm flow (matching E+):
///   1. Compute s0: s0(j,k) = CMat(k)*Γ₂(j,k_node) + DMat(k)*δ(j,k)
///   2. For each iteration inum:
///      a. PhiR0 = A_exp * R(j-1)
///      b. e(j) = -trace(PhiR0) / j
///      c. R(j) = PhiR0 + e(j)*I  [BEFORE s computation]
///      d. s(j,k) = CMat(k)*Σ_m[R(j-1)[m,k_node]*Γ₁(j,m) + R(j)[m,k_node]*Γ₂(j,m)]
///       + e(j)*DMat(k)*δ(j,k)
#[allow(clippy::too_many_arguments)]
fn compute_ctf_from_state_space(
    layers: &[CTFMaterial],
    a_exp: &[Vec<f64>],
    _a_inv: &[Vec<f64>],
    _b_mat: &[Vec<f64>],
    c_mat: &[Vec<f64>],
    d_mat: &[Vec<f64>],
    gamma1: &[Vec<f64>],
    gamma2: &[Vec<f64>],
    n: usize,
    timestep: f64,
) -> CTFCoefficients {
    // CTF DC gain ΣX should equal U_bare when h = k*(N+1)/(N*dx).
    // Note: with the lumped boundary scheme, the discrete-time DC gain
    // ΣX/(1+ΣΦ) may differ slightly from U_bare due to the mass correction,
    // but the raw ΣX should converge to U_bare * (1 + ΣΦ).
    let total_r_wall: f64 = layers.iter().map(|l| l.resistance()).sum();
    let u_bare = 1.0 / total_r_wall;

    // ========================================================================
    // COORDINATE TRANSFORM: Eliminate Γ₂ from the Seem extraction phase.
    //
    // The FOH discretization (Seem 1987) uses u̇(n) = (u(n+1)-u(n))/Δt:
    //   x(n+1) = Φ·x(n) + Γ₁·u(n) + Γ₂·u̇(n)
    // Rewriting with u̇(n) = (u(n+1)-u(n))/Δt:
    //   x(n+1) = Φ·x(n) + (Γ₁ - Γ₂/Δt)·u(n) + (Γ₂/Δt)·u(n+1)
    // Let a = Γ₁ - Γ₂/Δt, b = Γ₂/Δt, and define z(n) = x(n) - b·u(n):
    //   z(n+1) = Φ·z(n) + Γ̃·u(n)   where Γ̃ = Φ·b + a = (Φ-I)·Γ₂/Δt + Γ₁
    //   y(n)   = C·z(n) + D̃·u(n)    where D̃ = C·b + D = C·Γ₂/Δt + D
    //
    // DC gain is preserved: D̃ + C·(I-Φ)⁻¹·Γ̃ = D + C·(I-Φ)⁻¹·Γ₁ = D - C·A⁻¹·B
    // ========================================================================

    // Compute combined input matrix: Γ̃ = (Φ-I)·Γ₂/Δt + Γ₁
    let phi_gamma2 = mat_mat_mul_col(a_exp, gamma2); // Φ·Γ₂  (n×2)
    let gamma_tilde = {
        let mut g = vec![vec![0.0; 2]; n];
        for i in 0..n {
            for j in 0..2 {
                // (Φ·Γ₂ - Γ₂)/Δt + Γ₁ = (Φ-I)·Γ₂/Δt + Γ₁
                g[i][j] = (phi_gamma2[i][j] - gamma2[i][j]) / timestep + gamma1[i][j];
            }
        }
        FlatMatrix::from_vec_vec(&g)
    };

    // Compute combined direct-transmission matrix: D̃ = C·Γ₂/Δt + D
    let c_gamma2 = mat_mul_gen(c_mat, gamma2); // C·Γ₂ (2×2)
    let d_tilde = {
        let mut d = vec![vec![0.0; 2]; 2];
        for j in 0..2 {
            for k in 0..2 {
                d[j][k] = c_gamma2[j][k] / timestep + d_mat[j][k];
            }
        }
        d
    };

    // Convert c_mat to FlatMatrix for use in Leverrier iteration
    let c_mat_fm = FlatMatrix::from_vec_vec(c_mat);

    // s0(2,2): initial CTF coefficients (j=0 term) = D̃
    // s(2,2,max_terms): history CTF coefficients (j>=1 terms)
    // e(max_terms): flux history coefficients (Φ terms)
    let mut s0 = vec![vec![0.0f64; 2]; 2];
    let mut s: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; MAX_CTF_TERMS]; 2]; 2];
    let mut e = vec![0.0f64; MAX_CTF_TERMS];

    // Standard Seem: s0 = D̃ (direct transmission only)
    for j in 0..2 {
        for k in 0..2 {
            s0[j][k] = d_tilde[j][k];
        }
    }
    #[cfg(feature = "debug-physics")]
    eprintln!(
        "  s0 = D̃ = [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
        s0[0][0], s0[0][1], s0[1][0], s0[1][1]
    );

    // R matrix iteration (Seem Appendix C) — standard form after coordinate transform.
    //
    // For each iteration j (1-indexed):
    //   1. PhiR0 = Φ · R(j-1)
    //   2. e(j) = -trace(PhiR0) / j
    //   3. R(j) = PhiR0 + e(j) · I
    //   4. s(j,k) = C · Σ_m [R(j-1)(m,k_node) · Γ̃(m,input)] + e(j) · D̃(k) · δ(j,k)
    //
    // FIX: Using FlatMatrix with snapshot clone to avoid read-during-write aliasing
    // that corrupted diagonal elements in the original Vec<Vec<f64>> implementation.
    // r_new_snapshot = r_new.clone() captures R(j-1) before r_new is overwritten with R(j).
    let mut r_new = FlatMatrix::identity(n);
    let mut r_prev = FlatMatrix::zeros(n, n); // R(j-1), starts as R(0) = 0

    let mut num_ctf_terms = 0;
    let mut converged = false;

    for inum in 1..=MAX_CTF_TERMS {
        // Step 1: Compute PhiR0 = A_exp · R(j-1)  [r_new currently holds R(j-1)]
        let phi_r0 = mat_mat_mul_flat(a_exp, &r_new);

        // Step 2: e(j) = -trace(A_exp · R(j-1)) / j
        let trace: f64 = (0..n).map(|i| phi_r0.get(i, i)).sum();
        e[inum - 1] = -trace / inum as f64;

        // Step 3: Snapshot r_new before overwriting (fixes read-during-write aliasing)
        // R(j) = PhiR0 + e(j) * I
        // After: r_prev = R(j-1), r_new = R(j)
        let r_new_snapshot = r_new.clone();
        for i in 0..n {
            for j in 0..n {
                r_prev.set(i, j, r_new_snapshot.get(i, j));
                r_new.set(i, j, phi_r0.get(i, j));
            }
            r_new.set(i, i, r_new.get(i, i) + e[inum - 1]);
        }

        // Step 4: Standard Leverrier s coefficients for the transformed system.
        //
        // From the Leverrier algorithm applied to (Φ, Γ̃, C, D̃):
        //   s(k) = C · R_{k-1} · Γ̃ + e(k) · D̃     for ALL (output, input) pairs
        //
        // NOTE: This uses R (not R^T). For single-layer walls, R is symmetric
        // so R^T = R. For multi-layer walls, Φ is NOT symmetric at layer
        // interfaces, so R^T ≠ R — using R^T here was the root-cause bug.
        let rg = mat_mat_mul_col_flat(&r_prev, &gamma_tilde); // R(j-1) · Γ̃ → n×2
        let s_partial = mat_mul_gen_flat(&c_mat_fm, &rg); // C · (R · Γ̃) → 2×2
        for j in 0..2 {
            for k in 0..2 {
                s[j][k][inum - 1] = s_partial.get(j, k) + e[inum - 1] * d_tilde[j][k];
            }
        }

        // Debug: trace s coefficient evolution
        #[cfg(feature = "debug-physics")]
        if inum <= 10 || inum % 20 == 0 {
            let total_r_wall: f64 = layers.iter().map(|l| l.resistance()).sum();
            let u_bare_p = 1.0 / total_r_wall;
            let x_partial: f64 = s0[1][0] + (0..inum).map(|j| s[1][0][j]).sum::<f64>();
            let y_partial: f64 = -s0[1][1] - (0..inum).map(|j| s[1][1][j]).sum::<f64>();
            eprintln!(
                "  [inum={:3}] e={:.6e} s[1][0]={:.6} s[1][1]={:.6} ΣX={:.6} ΣY={:.6} ΣX/U={:.4}",
                inum,
                e[inum - 1],
                s[1][0][inum - 1],
                s[1][1][inum - 1],
                x_partial,
                y_partial,
                x_partial / u_bare_p
            );
        }

        // Check convergence: two criteria (hybrid approach)
        //
        // 1. PARTIAL-SUM CHECK (primary): For thin walls, the s series oscillates
        //    in sign so individual terms never become small, but the partial sum
        //    ΣX converges to U_bare. Stop when |ΣX - U_bare| / U_bare < CONVRG_LIM.
        //
        // 2. S-TAIL CHECK (fallback): For thick walls with monotonic decay, the
        //    individual s terms become tiny. Stop when max|s_tail| / U_bare < CONVRG_LIM.
        //
        // Either criterion is sufficient to stop. Cayley-Hamilton guarantees
        // R(n)=0 so extraction is exact after n iterations minimum.
        if inum >= MIN_CTF_TERMS.max(n) {
            // Criterion 1: partial sum convergence
            let x_partial: f64 = s0[1][0] + (0..inum).map(|j| s[1][0][j]).sum::<f64>();
            let x_residual_rel = (x_partial - u_bare).abs() / u_bare.max(1e-10);

            // Criterion 2: s-tail magnitude (fallback for monotonic-decay walls)
            let max_s_tail = s[0]
                .iter()
                .chain(s[1].iter())
                .map(|v| v[inum - 1].abs())
                .fold(0.0f64, f64::max);
            let s_tail_rel = max_s_tail / u_bare.max(1e-10);

            if x_residual_rel < CONVRG_LIM || s_tail_rel < CONVRG_LIM {
                num_ctf_terms = inum;
                converged = true;
                break;
            }
        }

        num_ctf_terms = inum;
    }

    if !converged {
        num_ctf_terms = MAX_CTF_TERMS;
    }

    // Map s/s0/e to standard CTF coefficients
    // The CTF equation for interior heat flux (E+ convention):
    //   q_int(t) = -Z0·T_int(t) + Y0·T_ext(t)
    //              + sum_j(Y_j·T_ext(t-j) - Z_j·T_int(t-j))
    //              + sum_j(Φ_j·q_int(t-j))
    //
    // In our state-space:
    //   output 1 (interior flux) is computed from:
    //   y_1(t) = s0[1][0]·T_ext(t) + s0[1][1]·T_int(t)
    //            + sum(s[1][0][j]·T_ext(t-j) + s[1][1][j]·T_int(t-j))
    //            + sum(e[j]·y_1(t-j))
    //
    // Mapping: X_j = s[1][0][j] (or s0[1][0] for j=0)
    //          Y_j = s[1][1][j] (note: in E+ notation Y is cross, so s[1][0] is Y)
    //          Z_j = s[1][1][j]
    //          Φ_j = e[j]
    //
    // Wait — need to be careful with E+ sign conventions.
    // E+ CTF equation (inside heat flux):
    //   q''_ki(t) = -Z0·Ti,t + Y0·To,t + sum(Z_j·Ti,t-jδ) + sum(Y_j·To,t-jδ) + sum(Φ_j·q''_ki,t-jδ)
    //
    // The sign convention in our code (calculate_interior_flux):
    //   q = X·T_ext - Y·T_int - Φ·q_prev
    //   where X = Y0, Y = Z0 in E+ notation
    //
    // So: X[j] = s0[1][0] for j=0, s[1][0][j] for j>=1  (cross term, exterior temp)
    //     Y[j] = -s0[1][1] for j=0, -s[1][1][j] for j>=1  (interior temp, subtracted)
    //     Phi[j] = -e[j-1] for j>=1  (flux history, subtracted)
    //
    // But wait — let's verify with steady state. At steady state with constant T_ext, T_int:
    //   q = sum(X)·T_ext - sum(Y)·T_int = U·(T_ext - T_int)
    // So sum(X) = sum(Y) = U.
    //
    // Let me check: sum of all X = s0[1][0] + sum(s[1][0][j]) should equal U.
    // sum of all Y_coeff = -s0[1][1] - sum(s[1][1][j]) should equal U.
    //   => s0[1][1] + sum(s[1][1]) = -U (should be negative)
    //
    // The interior flux output: q_int = C_int · x + D_int · u
    // C_int = -k/dx (negative), D_int = k/dx (positive for T_int)
    // So s0[1][1] (interior temp coefficient) starts negative, and
    // s0[1][0] (exterior temp coefficient) depends on Gamma2.
    //
    // Let's just compute and see what we get, then verify with steady-state check.

    let num = num_ctf_terms + 1; // +1 for the j=0 term
    let mut coeffs = CTFCoefficients::new(timestep, num);
    coeffs.num_coeffs = num;
    coeffs.total_state_nodes = n;

    // j=0 terms
    coeffs.x[0] = s0[1][0]; // Exterior temp → interior flux
    coeffs.y[0] = -s0[1][1]; // Interior temp → interior flux (negated for our sign convention)
    coeffs.z[0] = s0[1][1]; // Keep Z as-is for reference
    coeffs.phi[0] = 0.0; // No self-feedback at j=0

    // j>=1 terms
    for j in 0..num_ctf_terms {
        let idx = j + 1;
        if idx < num {
            coeffs.x[idx] = s[1][0][j];
            coeffs.y[idx] = -s[1][1][j]; // Negated
            coeffs.z[idx] = s[1][1][j];
            coeffs.phi[idx] = e[j]; // Seem e[j] are negative for stable walls; positive phi gives correct DC gain ΣX/(1+ΣΦ)=U
        }
    }

    // Diagnostic output (no normalization — the math should be exact now)
    #[cfg(feature = "debug-physics")]
    {
        let total_r_wall: f64 = layers.iter().map(|l| l.resistance()).sum();
        let u_bare = 1.0 / total_r_wall;
        let x_sum: f64 = coeffs.x.iter().sum();
        let y_sum: f64 = coeffs.y.iter().sum();
        let phi_sum: f64 = coeffs.phi.iter().sum();

        eprintln!("Bare-wall CTF ({} layers, {} nodes):", layers.len(), n);
        eprintln!("  U_bare = {:.6} W/m²K", u_bare);
        eprintln!(
            "  s0 = [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
            s0[0][0], s0[0][1], s0[1][0], s0[1][1]
        );
        eprintln!("  e[0:5] = {:?}", &e[..5.min(MAX_CTF_TERMS)]);
        eprintln!("  Sum(X) = {:.6} (ratio: {:.4})", x_sum, x_sum / u_bare);
        eprintln!("  Sum(Y) = {:.6} (ratio: {:.4})", y_sum, y_sum / u_bare);
        eprintln!("  Num CTF terms: {}", num);
        eprintln!("  X[0:5] = {:?}", &coeffs.x[..5.min(num)]);
        eprintln!("  Phi[0:5] = {:?}", &coeffs.phi[..5.min(num)]);
        eprintln!(
            "  Steady-state check: ΣX={:.6}, ΣΦ={:.6}, ΣX/(1+ΣΦ)={:.6}, U_bare={:.6}",
            x_sum,
            phi_sum,
            x_sum / (1.0 + phi_sum),
            u_bare
        );
    }

    coeffs
}
mod linalg;

// Pipeline imports from the linear-algebra child (Issue #3787, decomposition
// stage 2). The child is the dense math kernel (`state_space_ctf::linalg`,
// pure functions, no CTF-domain coupling). All names are `pub(super)` so they
// are visible here but stay inside the `state_space_ctf` module from the
// outside world's perspective.
use linalg::{
    mat_mat_mul_col, mat_mat_mul_col_flat, mat_mat_mul_flat, mat_mul_gen,
    mat_mul_gen_flat, matrix_exponential, matrix_inverse, matrix_sub_col,
    matrix_sub_identity, scale_columns,
};

// Public-API re-export (used by `tests/all_tests/determinism_matrix_exponential.rs`
// and the per-case determinism fixtures). The path
// `crate::physics::state_space_ctf::matrix_exponential_faer` is preserved
// because of this re-export.
pub use linalg::matrix_exponential_faer;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod expm_debug_tests;
#[cfg(test)]
mod debug_new_expm_tests;
#[cfg(test)]
mod coverage_tests;
