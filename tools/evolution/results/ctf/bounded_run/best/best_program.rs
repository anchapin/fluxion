// Seed kernel for the state-space CTF evolution campaign (#3337).
//
// This file is **not** a runnable binary — it is the seed that the
// `fluxion-evaluator` harness recompiles and dispatches. It contains:
//
// 1. A self-contained copy of the state-space CTF discretization +
//    extraction pipeline from `src/physics/state_space_ctf.rs`,
//    verbatim so the baseline settings produce **exactly** the same
//    coefficients as the in-tree production code (verified by the
//    golden-coefficient test, `tests/evolution_ctf_golden.rs`).
//
// 2. Three `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` markers
//    isolating exactly the tunable heuristic functions:
//
//    - **`node_grading_heuristic`**: per-layer FD node placement
//      (uniform today; evolver may grade spacing near
//      high-effusivity interfaces for better accuracy per state).
//
//    - **`fom_matrix_exp_thresholds`**: FOH / matrix-exponential
//      scale-and-squaring threshold + minimax switching (Higham's
//      θ₁₃ bound today).
//
//    - **`extraction_truncation_policy`**: R-matrix coefficient
//      extraction — which modes are kept, absorbed, or dropped
//      when the s/Leverrier series is truncated.
//
// Everything outside the markers is **frozen context**: the
// mathematical skeleton (Seem state-space + FOH + Higham Padé [13/13]
// + Leverrier s-coefficient extraction + DC-gain film scaling) is
// fixed by the issue's "what actually varies" constraint.
//
// ## Baseline determinism
//
// The functions inside the EVOLVE-BLOCKs, with their **baseline**
// values, reproduce the current CTF implementation bit-for-bit on the
// wall library (`tests/reference_data/evolution/ctf/`). The
// golden-coefficient test enforces this. The evolver's job is to
// find improved heuristics — but the harness explicitly requires
// the seed to converge to the production coefficients at the
// starting point so we can distinguish "improvement" from "drift".

use fluxion_evaluator::kernel::{Kernel, KernelError, KernelInput, KernelOutput};

// =====================================================================
// SHARED STATE-SPACE TYPES (mirrors src/physics/state_space_ctf.rs)
// =====================================================================

/// Material layer with thermal properties (mirrors `CTFMaterial` in
/// `src/physics/ctf_coefficients.rs`).
#[derive(Debug, Clone)]
pub struct CTFMaterial {
    pub name: String,
    pub thickness: f64,
    pub conductivity: f64,
    pub density: f64,
    pub specific_heat: f64,
}

impl CTFMaterial {
    pub fn from_params(
        name: &str,
        thickness: f64,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            thickness,
            conductivity,
            density,
            specific_heat,
        }
    }
    #[inline]
    pub fn diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.specific_heat)
    }
    #[inline]
    pub fn resistance(&self) -> f64 {
        self.thickness / self.conductivity
    }
}

/// CTF coefficients (X, Y, Z, Φ).
#[derive(Debug, Clone)]
pub struct CTFCoefficients {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub phi: Vec<f64>,
    pub timestep: f64,
    pub num_coeffs: usize,
    pub total_state_nodes: usize,
}

impl CTFCoefficients {
    pub fn new(timestep: f64, num_coeffs: usize) -> Self {
        Self {
            x: vec![0.0; num_coeffs],
            y: vec![0.0; num_coeffs],
            z: vec![0.0; num_coeffs],
            phi: vec![0.0; num_coeffs],
            timestep,
            num_coeffs,
            total_state_nodes: 1,
        }
    }
    pub fn x_sum(&self) -> f64 {
        self.x.iter().sum()
    }
    pub fn phi_sum(&self) -> f64 {
        self.phi.iter().sum()
    }
    pub fn u_value(&self) -> f64 {
        self.x_sum() / (1.0 + self.phi_sum())
    }
}

// =====================================================================
// Frozen skeleton constants (matches state_space_ctf.rs exactly)
// =====================================================================

const R_SI: f64 = 0.125; // Interior film [m²K/W]
const R_SE: f64 = 0.044; // Exterior film [m²K/W]
const MIN_NODES: usize = 1;
const MAX_NODES: usize = 18;
const MIN_CTF_TERMS: usize = 20;
const MAX_CTF_TERMS: usize = 200;
const CONVRG_LIM: f64 = 1.0e-3;
// Higham's θ₁₃ bound for Padé [13/13] scaling-and-squaring.
const THETA_13: f64 = 1.495585217958292e-2;

// =====================================================================
// EVOLVE-BLOCK-1: Per-layer FD node grading heuristic
// =====================================================================
//
// The baseline returns the E+ nodes-per-layer rule: uniform grading
// with `dxn = sqrt(2·α·Δt)`, clamped to [MIN_NODES, MAX_NODES], and a
// short-circuit for "all-lightweight" walls. The evolver may refine
// this to graded spacing near high-effusivity interfaces (the classic
// accuracy-per-state win), or to alternative stability criteria.
//
// Contract:
//   Input: layers + timestep
//   Output: Vec<usize> with one nodes-per-layer value per input layer
//
// Fitness signal: lower error on the frequency-response reference for
// the same total state count, or fewer states for the same accuracy.

fn node_grading_heuristic(layers: &[CTFMaterial], timestep: f64) -> Vec<usize> {
    // EVOLVE-BLOCK-START
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
    // EVOLVE-BLOCK-END
}

// =====================================================================
// EVOLVE-BLOCK-2: FOH / matrix-exponential scale thresholds
// =====================================================================
//
// The baseline returns the Higham (2005) θ₁₃ = 1.495585217958292e-2
// bound for Padé [13/13] scaling-and-squaring. The evolver may tune
// the squaring factor (`scale_exponent` and the early-return
// condition for stiff matrices) to trade accuracy for fewer
// matrix-matrix multiplications.
//
// Contract:
//   Input: 1-norm of `A·Δt` (precomputed)
//   Output: (scale_factor, scaling_exponent s such that ||A/2^s|| < threshold)
//
// Fitness signal: lower eval latency at equal or better frequency-response
// accuracy, OR an improved Padé order selection for stiff walls.

fn fom_matrix_exp_thresholds(norm_1: f64) -> (f64, usize) {
    // EVOLVE-BLOCK-START
    if norm_1 <= THETA_13 {
        return (1.0, 0);
    }
    let s_f = (norm_1 / THETA_13).log2().ceil();
    let s = (s_f.max(0.0)) as usize;
    let scale = 1.0_f64 / (1u64 << s.min(63)) as f64;
    (scale, s)
    // EVOLVE-BLOCK-END
}

// =====================================================================
// Frozen skeleton: state-space matrices (verbatim from
// src/physics/state_space_ctf.rs::build_state_space_matrices)
// =====================================================================

#[allow(clippy::type_complexity)]
fn build_state_space_matrices(
    layers: &[CTFMaterial],
    nodes_per_layer: &[usize],
    n: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut a_mat = vec![vec![0.0; n]; n];
    let mut b_mat = vec![vec![0.0; 2]; n];
    let mut c_mat = vec![vec![0.0; n]; 2];
    let mut d_mat = vec![vec![0.0; 2]; 2];

    let dx: Vec<f64> = layers
        .iter()
        .zip(nodes_per_layer.iter())
        .map(|(l, &nn)| {
            if nn > 1 {
                l.thickness / nn as f64
            } else {
                l.thickness
            }
        })
        .collect();

    let mut global_node = 0;
    for (layer_idx, layer) in layers.iter().enumerate() {
        let nn = nodes_per_layer[layer_idx];
        let dx_l = dx[layer_idx];
        let k = layer.conductivity;
        let cap_interior = layer.density * layer.specific_heat * dx_l;
        let dxtmp_interior = 1.0 / dx_l / cap_interior;
        let cap_boundary = 1.5 * cap_interior;
        let dxtmp_boundary = 1.0 / dx_l / cap_boundary;

        for local_node in 0..nn {
            let i = global_node + local_node;
            let is_exterior_boundary = layer_idx == 0 && local_node == 0;
            let is_interior_boundary = layer_idx == layers.len() - 1 && local_node == nn - 1;

            if is_exterior_boundary {
                a_mat[i][i] = -2.0 * k * dxtmp_boundary;
                if i + 1 < n {
                    a_mat[i][i + 1] = k * dxtmp_boundary;
                }
                b_mat[i][0] = k * dxtmp_boundary;
            } else if is_interior_boundary {
                a_mat[i][i] = -2.0 * k * dxtmp_boundary;
                if i > 0 {
                    a_mat[i][i - 1] = k * dxtmp_boundary;
                }
                b_mat[i][1] = k * dxtmp_boundary;
            } else {
                let is_interface =
                    local_node == nn - 1 && layer_idx < layers.len() - 1;
                if is_interface {
                    let next_layer = &layers[layer_idx + 1];
                    let dx_next = dx[layer_idx + 1];
                    let capavg = 0.5
                        * (cap_interior
                            + next_layer.density * next_layer.specific_heat * dx_next);
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
                    a_mat[i][i] = -2.0 * k * dxtmp_interior;
                    if i > 0 {
                        a_mat[i][i - 1] = k * dxtmp_interior;
                    }
                    if i + 1 < n {
                        a_mat[i][i + 1] = k * dxtmp_interior;
                    }
                }
            }
        }
        global_node += nn;
    }

    let k_ext = layers.first().map(|l| l.conductivity).unwrap_or(1.0);
    let dx_ext = dx.first().copied().unwrap_or(1.0);
    let n_ext = nodes_per_layer.first().copied().unwrap_or(1);
    let h_surf_ext = k_ext * (n_ext as f64 + 1.0) / (n_ext as f64 * dx_ext);
    let k_int = layers.last().map(|l| l.conductivity).unwrap_or(1.0);
    let dx_int = dx.last().copied().unwrap_or(1.0);
    let n_int = nodes_per_layer.last().copied().unwrap_or(1);
    let h_surf_int = k_int * (n_int as f64 + 1.0) / (n_int as f64 * dx_int);

    c_mat[0][0] = -h_surf_ext;
    d_mat[0][0] = h_surf_ext;
    c_mat[1][n - 1] = h_surf_int;
    d_mat[1][1] = -h_surf_int;

    (a_mat, b_mat, c_mat, d_mat)
}

// =====================================================================
// Frozen skeleton: matrix operations (verbatim from state_space_ctf.rs)
// =====================================================================

fn mat_mul_gen(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r1 = a.len();
    let c1 = a[0].len();
    let c2 = b[0].len();
    let mut c = vec![vec![0.0; c2]; r1];
    for i in 0..r1 {
        for j in 0..c2 {
            let mut sum = 0.0;
            for k in 0..c1 {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

fn matrix_sub_identity(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = a.to_vec();
    for i in 0..n {
        result[i][i] -= 1.0;
    }
    result
}

fn mat_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

fn mat_mat_mul_col(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

fn scale_columns(mat: &[Vec<f64>], factor: f64) -> Vec<Vec<f64>> {
    mat.iter().map(|row| row.iter().map(|&v| v * factor).collect()).collect()
}

fn matrix_sub_col(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

fn matrix_inverse(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in col + 1..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }
        if max_row != col {
            aug.swap(col, max_row);
        }
        let pivot = aug[col][col];
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Some(inv)
}

fn matrix_norm_1(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let m = a[0].len();
    let mut max_col_sum = 0.0_f64;
    for j in 0..m {
        let col_sum: f64 = (0..n).map(|i| a[i][j].abs()).sum();
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}

fn expm_2x2(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let m11 = a[0][0] * t;
    let m12 = a[0][1] * t;
    let m21 = a[1][0] * t;
    let m22 = a[1][1] * t;
    let trace = m11 + m22;
    let half_trace = 0.5 * trace;
    let det = m11 * m22 - m12 * m21;
    let disc = half_trace * half_trace - det;
    let exp_half = half_trace.exp();
    let b11 = m11 - half_trace;
    let b12 = m12;
    let b21 = m21;
    let b22 = m22 - half_trace;
    let (c, s) = if disc.abs() < 1e-14 {
        (1.0, 1.0)
    } else if disc > 0.0 {
        let d = disc.sqrt();
        (d.cosh(), d.sinh() / d)
    } else {
        let w = (-disc).sqrt();
        (w.cos(), w.sin() / w)
    };
    let r11 = exp_half * (c + s * b11);
    let r12 = exp_half * (s * b12);
    let r21 = exp_half * (s * b21);
    let r22 = exp_half * (c + s * b22);
    vec![vec![r11, r12], vec![r21, r22]]
}

// =====================================================================
// Frozen skeleton: Padé [13/13] Higham scaling-and-squaring
// =====================================================================

fn expm_higham_pade13(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vec![a[0][0].exp()]];
    }
    let pade_b: [f64; 14] = [
        1.0, -5.0e-1, 1.2e-1, -1.833_333_333_333_333_3e-2,
        1.992_753_623_188_405_7e-3, -1.630_434_782_608_695_8e-4, 1.035_196_687_370_600_5e-5,
        -5.175_983_436_853_002e-7, 2.043_151_356_652_501e-8, -6.306_022_705_717_595e-10,
        1.483_770_048_404_14e-11, -2.529_153_491_597_966e-13, 2.810_170_546_219_962e-15,
        -1.544_049_750_670_309e-17,
    ];
    let norm_1 = matrix_norm_1(a);
    let (scale, s) = fom_matrix_exp_thresholds(norm_1);
    let b_mat: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| a[i][j] * scale).collect())
        .collect();
    let b_powers = compute_powers(&b_mat, 13);
    let mut d_mat: Vec<Vec<f64>> = identity(n);
    for k in 1..=13 {
        for i in 0..n {
            for j in 0..n {
                d_mat[i][j] += pade_b[k] * b_powers[k][i][j];
            }
        }
    }
    let mut numer = identity(n);
    for k in 1..=13 {
        let abs_bk = pade_b[k].abs();
        for i in 0..n {
            for j in 0..n {
                numer[i][j] += abs_bk * b_powers[k][i][j];
            }
        }
    }
    let exp_b = solve_linear_system_lu(&d_mat, &numer);
    let mut result = exp_b;
    for _ in 0..s {
        result = mat_mat_mul(&result, &result);
    }
    result
}

fn solve_linear_system_lu(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_val = lu[k][k].abs();
        for i in (k + 1)..n {
            if lu[i][k].abs() > pivot_val {
                pivot_val = lu[i][k].abs();
                pivot_row = i;
            }
        }
        if pivot_val < 1e-15 {
            return identity(n);
        }
        if pivot_row != k {
            lu.swap(k, pivot_row);
            perm.swap(k, pivot_row);
        }
        let pivot = lu[k][k];
        for i in (k + 1)..n {
            lu[i][k] /= pivot;
            for j in (k + 1)..n {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }
    let mut pb: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| b[perm[i]][j]).collect())
        .collect();
    for j in 0..n {
        for i in 1..n {
            let mut s = pb[i][j];
            for kk in 0..i {
                s -= lu[i][kk] * pb[kk][j];
            }
            pb[i][j] = s;
        }
    }
    for j in 0..n {
        for i in (0..n).rev() {
            let mut s = pb[i][j];
            for kk in (i + 1)..n {
                s -= lu[i][kk] * pb[kk][j];
            }
            pb[i][j] = s / lu[i][i];
        }
    }
    pb
}

fn compute_powers(b: &[Vec<f64>], max_power: usize) -> Vec<Vec<Vec<f64>>> {
    let n = b.len();
    let mut powers = Vec::with_capacity(max_power + 1);
    powers.push(identity(n));
    if max_power >= 1 {
        powers.push(b.to_vec());
    }
    for k in 2..=max_power {
        powers.push(mat_mat_mul(&powers[k - 1], b));
    }
    powers
}

fn matrix_exponential_faer(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vec![(a[0][0] * t).exp()]];
    }
    if n == 2 {
        return expm_2x2(a, t);
    }
    let mut a_t = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a_t[i][j] = a[i][j] * t;
        }
    }
    expm_higham_pade13(&a_t)
}

fn matrix_exponential(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    matrix_exponential_faer(a, t)
}

// =====================================================================
// EVOLVE-BLOCK-3: Leverrier R-matrix coefficient extraction truncation
// =====================================================================
//
// The baseline returns the partial-sum convergence criterion from
// state_space_ctf.rs::compute_ctf_from_state_space: stop at MAX_CTF_TERMS
// or when |ΣX - U_bare|/U_bare < CONVRG_LIM, whichever comes first.
//
// The evolver may tighten the threshold, switch to absolute-error
// convergence, or implement a per-mode truncation policy (keep modes
// with significant contribution; drop modes below a floor).
//
// Contract:
//   Input: per-iteration inum, partial ΣX sum, max s tail magnitude,
//          bare-wall U-value, max_terms
//   Output: (converged: bool, num_ctf_terms: usize)
//
// Fitness signal: lower eval latency (fewer iterations) at equal or
// better frequency-response accuracy, and never worse than 2× the
// baseline error.

fn extraction_truncation_policy(
    inum: usize,
    x_partial: f64,
    s_tail_max: f64,
    u_bare: f64,
    min_terms: usize,
    n: usize,
    _max_terms: usize,
) -> (bool, usize) {
    // EVOLVE-BLOCK-START
    if inum < min_terms.max(n) {
        return (false, 0);
    }
    let u_bare_safe = u_bare.max(1e-10);
    let x_residual_rel = (x_partial - u_bare).abs() / u_bare_safe;
    let s_tail_rel = s_tail_max / u_bare_safe;
    if x_residual_rel < CONVRG_LIM || s_tail_rel < CONVRG_LIM {
        (true, inum)
    } else {
        (false, 0)
    }
    // EVOLVE-BLOCK-END
}

// =====================================================================
// Frozen skeleton: compute_ctf_from_state_space (verbatim)
// =====================================================================

fn mat_mul_gen_flat(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let r1 = a.len();
    if r1 == 0 {
        return vec![];
    }
    let c1 = a[0].len();
    let c2 = b[0].len();
    let mut c = vec![vec![0.0; c2]; r1];
    for i in 0..r1 {
        for j in 0..c2 {
            let mut sum = 0.0;
            for k in 0..c1 {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

fn mat_mat_mul_col_flat(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    let m = b[0].len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

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
    let total_r_wall: f64 = layers.iter().map(|l| l.resistance()).sum();
    let u_bare = 1.0 / total_r_wall;

    let phi_gamma2 = mat_mat_mul_col(a_exp, gamma2);
    let gamma_tilde = {
        let mut g = vec![vec![0.0; 2]; n];
        for i in 0..n {
            for j in 0..2 {
                g[i][j] = (phi_gamma2[i][j] - gamma2[i][j]) / timestep + gamma1[i][j];
            }
        }
        g
    };
    let c_gamma2 = mat_mul_gen_flat(c_mat, gamma2);
    let d_tilde = {
        let mut d = vec![vec![0.0; 2]; 2];
        for j in 0..2 {
            for k in 0..2 {
                d[j][k] = c_gamma2[j][k] / timestep + d_mat[j][k];
            }
        }
        d
    };

    let mut s0 = vec![vec![0.0f64; 2]; 2];
    let mut s: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; MAX_CTF_TERMS]; 2]; 2];
    let mut e = vec![0.0f64; MAX_CTF_TERMS];

    for j in 0..2 {
        for k in 0..2 {
            s0[j][k] = d_tilde[j][k];
        }
    }

    let mut r_new = identity(n);
    let mut r_prev = vec![vec![0.0; n]; n];

    let mut num_ctf_terms = 0;
    let mut converged = false;

    for inum in 1..=MAX_CTF_TERMS {
        let phi_r0 = mat_mul_gen(a_exp, &r_new);
        let trace: f64 = (0..n).map(|i| phi_r0[i][i]).sum();
        e[inum - 1] = -trace / inum as f64;
        let r_new_snapshot = r_new.clone();
        for i in 0..n {
            for j in 0..n {
                r_prev[i][j] = r_new_snapshot[i][j];
                r_new[i][j] = phi_r0[i][j];
            }
            r_new[i][i] += e[inum - 1];
        }
        let rg = mat_mat_mul_col_flat(&r_prev, &gamma_tilde);
        let s_partial = mat_mul_gen_flat(c_mat, &rg);
        for j in 0..2 {
            for k in 0..2 {
                s[j][k][inum - 1] = s_partial[j][k] + e[inum - 1] * d_tilde[j][k];
            }
        }
        let x_partial: f64 = s0[1][0] + (0..inum).map(|j| s[1][0][j]).sum::<f64>();
        let max_s_tail = s[0]
            .iter()
            .chain(s[1].iter())
            .map(|v| v[inum - 1].abs())
            .fold(0.0f64, f64::max);
        let (done, n_terms) =
            extraction_truncation_policy(inum, x_partial, max_s_tail, u_bare, MIN_CTF_TERMS, n, MAX_CTF_TERMS);
        if done {
            num_ctf_terms = n_terms;
            converged = true;
            break;
        }
        num_ctf_terms = inum;
    }
    if !converged {
        num_ctf_terms = MAX_CTF_TERMS;
    }

    let num = num_ctf_terms + 1;
    let mut coeffs = CTFCoefficients::new(timestep, num);
    coeffs.num_coeffs = num;
    coeffs.total_state_nodes = n;

    coeffs.x[0] = s0[1][0];
    coeffs.y[0] = -s0[1][1];
    coeffs.z[0] = s0[1][1];
    coeffs.phi[0] = 0.0;

    for j in 0..num_ctf_terms {
        let idx = j + 1;
        if idx < num {
            coeffs.x[idx] = s[1][0][j];
            coeffs.y[idx] = -s[1][1][j];
            coeffs.z[idx] = s[1][1][j];
            coeffs.phi[idx] = e[j];
        }
    }
    coeffs
}

// =====================================================================
// Top-level: compute_state_space_ctf (matches production verbatim)
// =====================================================================

pub fn compute_state_space_ctf(
    layers: &[CTFMaterial],
    timestep: f64,
) -> CTFCoefficients {
    let nodes_per_layer = node_grading_heuristic(layers, timestep);
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

    let n = total_nodes;
    let (a_mat, b_mat, c_mat, d_mat) =
        build_state_space_matrices(layers, &nodes_per_layer, n);
    let a_exp = matrix_exponential(&a_mat, timestep);
    let a_inv = matrix_inverse(&a_mat).expect("A matrix should be invertible for stable wall");
    let a_exp_minus_i = matrix_sub_identity(&a_exp);
    let temp = mat_mat_mul_col(&a_exp_minus_i, &b_mat);
    let gamma1 = mat_mat_mul_col(&a_inv, &temp);
    let gamma1_scaled = scale_columns(&gamma1, 1.0 / timestep);
    let gamma2_diff = matrix_sub_col(&gamma1_scaled, &b_mat);
    let gamma2 = mat_mat_mul_col(&a_inv, &gamma2_diff);

    let mut coeffs = compute_ctf_from_state_space(
        layers, &a_exp, &a_inv, &b_mat, &c_mat, &d_mat,
        &gamma1, &gamma2, n, timestep,
    );

    let x_sum_bare: f64 = coeffs.x.iter().sum();
    let phi_sum_bare: f64 = coeffs.phi.iter().sum();
    let r_wall: f64 = layers.iter().map(|l| l.resistance()).sum();
    let u_filmed = 1.0 / (R_SI + r_wall + R_SE);
    let denom = x_sum_bare / u_filmed - phi_sum_bare;

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
    coeffs
}

// =====================================================================
// Candidate struct (Kernel trait impl)
// =====================================================================

#[derive(Default)]
pub struct Candidate;

impl Kernel for Candidate {
    fn evaluate(
        &self,
        input: &KernelInput,
    ) -> Result<KernelOutput, KernelError> {
        // Parse input params: { "layers": [...], "timestep_s": f64 }
        let layers_v = input
            .params
            .get("layers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                KernelError::BadInput("missing or non-array `layers`".to_string())
            })?;
        let timestep = input
            .params
            .get("timestep_s")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                KernelError::BadInput("missing or non-numeric `timestep_s`".to_string())
            })?;

        let mut layers = Vec::with_capacity(layers_v.len());
        for l_v in layers_v {
            let obj = l_v.as_object().ok_or_else(|| {
                KernelError::BadInput("layer must be an object".to_string())
            })?;
            let name = obj
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unnamed")
                .to_string();
            let thickness = obj.get("thickness_m").and_then(|v| v.as_f64()).ok_or_else(|| {
                KernelError::BadInput("layer missing `thickness_m`".to_string())
            })?;
            let conductivity = obj.get("k_w_mk").and_then(|v| v.as_f64()).ok_or_else(|| {
                KernelError::BadInput("layer missing `k_w_mk`".to_string())
            })?;
            let density = obj.get("rho_kg_m3").and_then(|v| v.as_f64()).ok_or_else(|| {
                KernelError::BadInput("layer missing `rho_kg_m3`".to_string())
            })?;
            let specific_heat =
                obj.get("cp_j_kgk").and_then(|v| v.as_f64()).ok_or_else(|| {
                    KernelError::BadInput("layer missing `cp_j_kgk`".to_string())
                })?;
            layers.push(CTFMaterial::from_params(
                &name, thickness, conductivity, density, specific_heat,
            ));
        }
        if layers.is_empty() {
            return Err(KernelError::BadInput("`layers` is empty".to_string()));
        }

        let coeffs = compute_state_space_ctf(&layers, timestep);
        Ok(KernelOutput {
            payload: serde_json::json!({
                "x": coeffs.x,
                "y": coeffs.y,
                "z": coeffs.z,
                "phi": coeffs.phi,
                "num_coeffs": coeffs.num_coeffs,
                "total_state_nodes": coeffs.total_state_nodes,
                "x_sum": coeffs.x_sum(),
                "phi_sum": coeffs.phi_sum(),
                "u_value": coeffs.u_value(),
            }),
        })
    }
}
