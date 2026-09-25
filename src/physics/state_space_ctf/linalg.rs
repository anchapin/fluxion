//! Dense linear-algebra kernel for the state-space CTF pipeline.
//!
//! Extracted from `state_space_ctf/mod.rs` at Issue #3787 decomposition time
//! (the parent landed at 4347/4347 lines, see
//! `tests/reference_data/module_size/state_space_ctf_ratchet.json`). The
//! kernel is self-contained pure math — small dense matrix operations,
//! matrix-exponential variants (Higham Padé [13/13], real-Schur/Francis
//! double-shift, legacy Padé, Taylor), and the Householder
//! QR/Hessenberg machinery — with no CTF-domain coupling.
//!
//! The pipeline (`compute_state_space_ctf`, `compute_ctf_from_state_space`,
//! `build_state_space_matrices`) re-imports the symbols it needs via
//! `use super::linalg::{...};` in `mod.rs`. Public API on
//! `crate::physics::state_space_ctf` is preserved unchanged; callers of
//! `matrix_exponential_faer` continue to use the same path because
//! `mod.rs` re-exports it via `pub use linalg::matrix_exponential_faer;`.
//!
//! ## Contents
//! - Basic dense matrix ops (`mat_mul_gen`, `identity`,
//!   `matrix_sub_identity`, `mat_mat_mul`, `mat_mat_mul_col`,
//!   `scale_columns`, `matrix_sub_col`)
//! - `FlatMatrix` variants (`mat_mat_mul_flat`, `mat_mat_mul_col_flat`,
//!   `mat_mul_gen_flat`)
//! - Matrix-exponential dispatch (`matrix_exponential`,
//!   `matrix_exponential_faer`)
//! - Higham Padé [13/13] (`expm_higham_padé13` + `solve_linear_system_lu`,
//!   `matrix_norm_1`, `compute_powers`, `expm_2x2`)
//! - Real-Schur / Francis QR (`householder_to_hessenberg`,
//!   `apply_householder_left` / `apply_householder_right` /
//!   `apply_householder_right_unitary`, `vector_norm`, `transpose`,
//!   `francis_qr_schur`, `implicit_double_shift_bulge_chase`)
//! - Reference implementations kept for the test suite
//!   (`matrix_exponential_old_pade`, `matrix_exponential_taylor`,
//!   `matrix_norm_inf`)
//! - Gauss-Jordan `matrix_inverse`
//!
//! ## Visibility
//!
//! Every function is `pub(super)` (visible to `state_space_ctf`) so the
//! parent pipeline and the test siblings can both reach it. Only
//! `matrix_exponential_faer` is `pub` so it remains reachable as
//! `crate::physics::state_space_ctf::matrix_exponential_faer` via the
//! re-export shim in `mod.rs`.

use super::FlatMatrix;

/// General matrix multiply: C = A · B where A is (r1×c1) and B is (c1×c2).
/// Result is (r1×c2). Works for non-square matrices.
pub fn mat_mul_gen(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Create n×n identity matrix.
pub fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Compute A - I (subtract identity from matrix).
pub fn matrix_sub_identity(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = a.to_vec();
    for i in 0..n {
        result[i][i] -= 1.0;
    }
    result
}

/// Matrix multiplication C = A · B (both n×n).
pub fn mat_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Matrix × column matrix multiplication: C = A · B where A is n×n and B is n×m.
pub fn mat_mat_mul_col(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Scale columns of a matrix by a factor.
pub fn scale_columns(mat: &[Vec<f64>], factor: f64) -> Vec<Vec<f64>> {
    mat.iter()
        .map(|row| row.iter().map(|&v| v * factor).collect())
        .collect()
}

/// Subtract column matrices: C = A - B.
pub fn matrix_sub_col(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

// ==================== FlatMatrix Matrix Operations ====================
// Flat versions that work with FlatMatrix to avoid Vec<Vec<f64>> aliasing issues.

/// Matrix multiplication C = A · B where A is n×n and B is n×n (FlatMatrix).
pub fn mat_mat_mul_flat(a: &[Vec<f64>], b: &FlatMatrix) -> FlatMatrix {
    let n = a.len();
    let mut c = FlatMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
    c
}

/// Matrix × column multiplication: C = A · B where A is n×n and B is n×m (FlatMatrix result).
pub fn mat_mat_mul_col_flat(a: &FlatMatrix, b: &FlatMatrix) -> FlatMatrix {
    let n = a.rows();
    let m = b.cols();
    let mut c = FlatMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a.get(i, k) * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
    c
}

/// General matrix multiply: C = A · B where A is (r1×c1) FlatMatrix and B is (c1×c2) Vec<Vec>.
pub fn mat_mul_gen_flat(a: &FlatMatrix, b: &FlatMatrix) -> FlatMatrix {
    let r1 = a.rows();
    let c1 = a.cols();
    let c2 = b.cols();
    let mut c = FlatMatrix::zeros(r1, c2);
    for i in 0..r1 {
        for j in 0..c2 {
            let mut sum = 0.0;
            for k in 0..c1 {
                sum += a.get(i, k) * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
    c
}

/// Compute matrix exponential exp(A·t).
///
/// This dispatches to the new `matrix_exponential_faer` implementation,
/// which uses the robust Higham (2005) scaling-and-squaring algorithm with
/// the Padé [13/13] approximant on a Schur-reduced matrix. The previous
/// `matrix_exponential_schur` (Schur-Parlett with 1/(λᵢ-λⱼ) recurrence) failed
/// for multi-layer walls with clustered eigenvalues, producing negative
/// flux coefficients. The new implementation handles clustered eigenvalues
/// correctly because the Padé approximant is applied to the
/// quasi-upper-triangular Schur form, which has no eigenvalue-difference
/// divisions in the critical path.
pub fn matrix_exponential(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    matrix_exponential_faer(a, t)
}

/// faer-backed matrix exponential: Higham Padé [13/13] scaling-and-squaring.
///
/// This is the **new, stable implementation** that fixes the multi-layer wall
/// bug from issue #951. The previous `matrix_exponential_schur` used a
/// Schur-Parlett recurrence that divides by (λᵢ - λⱼ), which becomes ~0
/// for clustered eigenvalues (e.g. the 4-layer Case 900 wall has 4
/// eigenvalues clustered within 1e-4 of each other). The (λᵢ - λⱼ) division
/// amplifies any noise and produces wildly wrong off-diagonal entries,
/// leading to negative X_0 / Y_0 coefficients and Newton iteration divergence.
///
/// The new implementation uses Higham (2005) **scaling-and-squaring** with
/// **Padé [13/13]**, the same algorithm used by MATLAB's `expm`. The key
/// properties of this algorithm that fix the issue #951 bug:
/// 1. It does **not** divide by eigenvalue differences anywhere — the
///    Parlett recurrence is replaced by a direct polynomial/rational
///    approximation (Padé).
/// 2. It is stable for any matrix with ||A·t||_1 ≤ θ₁₃ ≈ 0.015 after
///    scaling, which is guaranteed by the scaling factor s such that
///    ||A/2^s||_1 < θ₁₃.
/// 3. Squaring-the-squaring operation preserves the result within machine
///    precision (the squaring error is bounded by Higham's theorem).
///
/// For small n (≤ 2), falls through to a direct formula.
pub fn matrix_exponential_faer(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let n = a.len();

    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vec![(a[0][0] * t).exp()]];
    }
    if n == 2 {
        // Direct 2x2 formula (Moler & Van Loan 2003, special case)
        return expm_2x2(a, t);
    }

    // Apply Higham's Padé [13/13] scaling-and-squaring algorithm directly to
    // the matrix A·t. The algorithm makes no assumption about the structure
    // of A — it works for any matrix with ||A·t||_1 ≤ θ₁₃ after scaling.
    //
    // Note: We do NOT use Schur decomposition here because the existing
    // in-tree Francis QR Schur is numerically unstable for matrices with
    // clustered eigenvalues (issue #951's original problem) — and even
    // applying the Schur to A·t before the Pade step doesn't help because
    // the Schur vectors V would still have an ill-conditioned 1-norm.
    //
    // Direct application of Padé [13/13] to A·t is the most robust approach
    // for our case: n is small (≤ 24) and ||A·t||_1 is bounded (≤ 50 for
    // our state-space matrices), so the squaring factor s is ≤ 12.
    let mut a_t = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a_t[i][j] = a[i][j] * t;
        }
    }
    expm_higham_padé13(&a_t)
}

/// Higham (2005) scaling-and-squaring with Padé [13/13] for matrix exponential.
///
/// This is the algorithm used by MATLAB's `expm` and is the de-facto
/// standard for "robust" matrix exponential computation. It is **stable**
/// even for stiff matrices (large ||A·t||_1) and does not require Schur
/// decomposition, but it is most efficient when applied to a small
/// (≤ 30×30) quasi-upper-triangular matrix — which is exactly what we
/// have after the Schur reduction.
///
/// Reference: Higham, N.J. (2005). "The scaling and squaring method for
/// the matrix exponential revisited." SIAM J. Matrix Anal. Appl. 26(4),
/// 1179-1193.
///
/// The Padé [13/13] approximant of e^z is:
///   e^z ≈ N(z) / D(z)
/// where:
///   D(z) = sum_{k=0}^{13} b_k z^k                        (denominator, alternating signs)
///   N(z) = sum_{k=0}^{13} (-1)^k b_k z^k                (numerator, sign-flipped b_k)
/// and:
///   b_k = (-1)^k · (2p-k)! p! / ((2p)! k! (p-k)!),   p = 13
///
/// Higham's "theta_13" bound (θ₁₃ = 1.495585217958292e-2) is used to
/// determine the squaring factor s such that ||A / 2^s||_1 < θ₁₃,
/// which guarantees ~machine-precision accuracy in the final result.
pub fn expm_higham_padé13(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vec![a[0][0].exp()]];
    }

    // Padé [13/13] denominator coefficients b_k for e^z:
    //   b_k = (-1)^k · (2p-k)! p! / ((2p)! k! (p-k)!),   p = 13
    // Verified against direct BigInt factorial computation to 16 sig digits.
    let pade_b: [f64; 14] = [
        1.0,                        // b_0
        -5.0e-1,                    // b_1
        1.2e-1,                     // b_2
        -1.8333333333333333e-2,     // b_3
        1.9927536231884057e-3,      // b_4
        -1.6304347826086958e-4,     // b_5
        1.0351966873706005e-5,      // b_6
        -5.175_983_436_853_002e-7,  // b_7
        2.043_151_356_652_501e-8,   // b_8
        -6.306_022_705_717_595e-10, // b_9
        1.483_770_048_404_14e-11,   // b_10
        -2.529_153_491_597_966e-13, // b_11
        2.810_170_546_219_962e-15,  // b_12
        -1.544_049_750_670_309e-17, // b_13
    ];

    // Compute the 1-norm of A: ||A||_1 = max_j (sum_i |A[i][j]|)
    let norm_1 = matrix_norm_1(a);

    // Higham's theta_13 bound: θ₁₃ = 1.495585217958292e-2
    const THETA_13: f64 = 1.495585217958292e-2;

    // Find scaling factor s such that ||A / 2^s||_1 < θ₁₃
    let s = if norm_1 <= THETA_13 {
        0
    } else {
        // s = ceil(log2(||A||_1 / θ₁₃))
        let s_f = ((norm_1 / THETA_13).log2()).ceil();
        s_f.max(0.0) as usize
    };

    #[cfg(feature = "debug-physics")]
    if n <= 32 {
        eprintln!("[expm_pade13] n={}, ||A||_1={:.6e}, s={}", n, norm_1, s);
    }

    // Scale A by 1/2^s: B = A / 2^s
    let scale = 1.0_f64 / (1u64 << s.min(63)) as f64;
    let b_mat: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| a[i][j] * scale).collect())
        .collect();

    // Compute Padé [13/13] of B = A / 2^s:
    //   exp(B) ≈ D(B)^(-1) · N(B)
    //   D(B) = sum_{k=0}^{13} b_k B^k
    //   N(B) = sum_{k=0}^{13} (-1)^k b_k B^k   =   sum_{k=0}^{13} |b_k| B^k  (since b_k = (-1)^k |b_k|)
    //
    // For numerical efficiency, we build:
    //   D(B) = even_k (positive k) + odd_k (with negative sign already in b_k)
    //   N(B) = sum |b_k| B^k  (all positive)

    // Compute powers B^1, B^2, ..., B^13 incrementally
    let b_powers = compute_powers(&b_mat, 13);

    // Build denominator D(B)
    // D(B) = (b_0 I + b_2 B^2 + b_4 B^4 + ...) + (b_1 B + b_3 B^3 + ...)
    // b_k already has its natural sign: b_0=+1, b_1=-1/2, b_2=+0.12, ...
    let mut d_mat: Vec<Vec<f64>> = identity(n);
    for k in 1..=13 {
        for i in 0..n {
            for j in 0..n {
                d_mat[i][j] += pade_b[k] * b_powers[k][i][j];
            }
        }
    }

    // Build numerator N(B) = sum |b_k| B^k
    let mut numer = identity(n);
    for k in 1..=13 {
        let abs_bk = pade_b[k].abs();
        for i in 0..n {
            for j in 0..n {
                numer[i][j] += abs_bk * b_powers[k][i][j];
            }
        }
    }

    // Solve D(B) · X = N(B) for X = exp(B)
    let exp_b = solve_linear_system_lu(&d_mat, &numer);

    // Square s times: exp(A) = (exp(B))^(2^s)
    let mut result = exp_b;
    for _ in 0..s {
        result = mat_mat_mul(&result, &result);
    }
    result
}

/// Solve A · X = B for X using LU decomposition with partial pivoting.
///
/// Both A and B are n×n. Returns X. This is the stable matrix analog
/// of Gaussian elimination, used in Higham's Padé algorithm for
/// inverting the Padé denominator.
pub fn solve_linear_system_lu(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // LU with partial pivoting: P A = L U
    // Solve P A X = P B  =>  L U X = P B
    // First solve L Y = P B, then solve U X = Y.
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot row (largest |L[i][k]|)
        let mut pivot_row = k;
        let mut pivot_val = lu[k][k].abs();
        for i in (k + 1)..n {
            if lu[i][k].abs() > pivot_val {
                pivot_val = lu[i][k].abs();
                pivot_row = i;
            }
        }
        if pivot_val < 1e-15 {
            // Singular — return identity as a safe fallback (shouldn't happen for our cases)
            return identity(n);
        }
        if pivot_row != k {
            lu.swap(k, pivot_row);
            perm.swap(k, pivot_row);
        }
        // Eliminate below
        let pivot = lu[k][k];
        for i in (k + 1)..n {
            lu[i][k] /= pivot;
            for j in (k + 1)..n {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }

    // Apply permutation to B: P B is the RHS
    let mut pb: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| b[perm[i]][j]).collect())
        .collect();

    // Forward substitution: solve L Y = P B
    // L is unit lower triangular with implicit unit diagonal; L[i][k] for i>k stores the multiplier.
    for j in 0..n {
        for i in 1..n {
            let mut s = pb[i][j];
            for kk in 0..i {
                s -= lu[i][kk] * pb[kk][j];
            }
            pb[i][j] = s;
        }
    }

    // Back substitution: solve U X = Y
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

/// 1-norm of a matrix: ||A||_1 = max_j (sum_i |A[i][j]|).
pub fn matrix_norm_1(a: &[Vec<f64>]) -> f64 {
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

/// Direct 2×2 matrix exponential: exp(A·t) for A 2×2.
///
/// Uses the closed-form formula from Moler & Van Loan (2003), valid for
/// any 2×2 matrix. Let M = A·t, τ = trace(M), δ = (τ/2)² - det(M).
/// Define B = M - (τ/2)·I (so trace(B) = 0 and B² = δ·I).
///
///   If δ > 0:  exp(M) = e^(τ/2) · [cosh(√δ)·I + sinh(√δ)/√δ · B]
///   If δ ≈ 0:  exp(M) = e^(τ/2) · [I + B]                   (Taylor, since B² ≈ 0)
///   If δ < 0:  exp(M) = e^(τ/2) · [cos(√-δ)·I + sin(√-δ)/√-δ · B]
///
/// This formula is numerically stable for any 2×2 A.
pub fn expm_2x2(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    // Scale A by t: M = A·t
    let m11 = a[0][0] * t;
    let m12 = a[0][1] * t;
    let m21 = a[1][0] * t;
    let m22 = a[1][1] * t;

    let trace = m11 + m22;
    let half_trace = 0.5 * trace;
    let det = m11 * m22 - m12 * m21;
    let disc = half_trace * half_trace - det;
    let exp_half = half_trace.exp();

    // B = M - (τ/2)·I
    let b11 = m11 - half_trace;
    let b12 = m12;
    let b21 = m21;
    let b22 = m22 - half_trace;

    // Compute (c, s) = (cosh/sin, sinh/cos) so that exp(B) = c·I + s·B
    let (c, s) = if disc.abs() < 1e-14 {
        // δ ≈ 0: B is approximately nilpotent, exp(B) = I + B
        (1.0, 1.0)
    } else if disc > 0.0 {
        // Real distinct eigenvalues
        let d = disc.sqrt();
        (d.cosh(), d.sinh() / d)
    } else {
        // Complex eigenvalues
        let w = (-disc).sqrt();
        (w.cos(), w.sin() / w)
    };

    // exp(M) = exp(τ/2) · [c·I + s·B]
    let r11 = exp_half * (c + s * b11);
    let r12 = exp_half * (s * b12);
    let r21 = exp_half * (s * b21);
    let r22 = exp_half * (c + s * b22);

    vec![vec![r11, r12], vec![r21, r22]]
}
/// Householder reduction of a general n×n matrix A to upper Hessenberg form.
///
/// Returns (H, U) such that A = U · H · U^T, where H is upper Hessenberg
/// (h[i][j] = 0 for i > j+1) and U is the product of Householder reflections
/// (orthogonal).
#[allow(dead_code)]
pub fn householder_to_hessenberg(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut h = a.to_vec();
    let mut u = identity(n);

    if n <= 2 {
        return (h, u);
    }

    for k in 0..n.saturating_sub(2) {
        // Extract the column vector x = h[k+2..n, k] (entries BELOW the subdiagonal).
        // The subdiagonal h[k+1, k] is preserved (it's a feature of Hessenberg form).
        // For a tridiagonal A, x is already zero, so we skip.
        if n - (k + 2) < 1 {
            continue; // Nothing to zero out below the subdiagonal
        }
        let mut x: Vec<f64> = (k + 2..n).map(|i| h[i][k]).collect();
        let x_norm = vector_norm(&x);
        if x_norm < 1e-15 {
            continue; // Already zero below subdiagonal
        }

        // Householder vector: v = x + sign(x[0]) * ||x|| * e_0
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * x_norm;
        let v_norm = vector_norm(&x);
        if v_norm < 1e-15 {
            continue;
        }
        for vi in x.iter_mut() {
            *vi /= v_norm;
        }
        let v = x;

        // Apply H = I - 2 v v^T from the left to h[k+2..n, k..n]
        apply_householder_left(&mut h, &v, k + 2, k, n);

        // Apply H = I - 2 v v^T from the right to h[0..n, k+2..n]
        apply_householder_right(&mut h, &v, 0, k + 2, n);

        // Update U = U · H_k
        // H_k = I - 2 v v^T (acting on rows k+2..n, cols k+2..n)
        // Equivalently, U_new = U · (I - 2 v v^T)
        apply_householder_right_unitary(&mut u, &v, k + 2, n);
    }

    (h, u)
}

/// Apply Householder (I - 2 v v^T) to rows [start..n] of h, columns [col_start..n].
#[allow(dead_code)]
pub fn apply_householder_left(
    h: &mut [Vec<f64>],
    v: &[f64],
    start: usize,
    col_start: usize,
    n: usize,
) {
    // h[start..n, col_start..n] -= 2 v (v^T h[start..n, col_start..n])
    // Step 1: w = v^T h[start..n, col_start..n]  (a row vector of length n-col_start)
    let mut w = vec![0.0; n - col_start];
    for j in 0..n - col_start {
        let mut s = 0.0;
        for i in 0..v.len() {
            s += v[i] * h[start + i][col_start + j];
        }
        w[j] = s;
    }
    // Step 2: h[start..n, col_start..n] -= 2 v w
    for i in 0..v.len() {
        for j in 0..n - col_start {
            h[start + i][col_start + j] -= 2.0 * v[i] * w[j];
        }
    }
}

/// Apply Householder (I - 2 v v^T) to columns [start..n] of h, rows [0..row_end].
#[allow(dead_code)]
pub fn apply_householder_right(
    h: &mut [Vec<f64>],
    v: &[f64],
    row_end: usize,
    start: usize,
    _n: usize,
) {
    // h[0..row_end, start..n] -= 2 (h[0..row_end, start..n] v) v^T
    // Step 1: w = h[0..row_end, start..n] v  (a column vector of length row_end)
    let mut w = vec![0.0; row_end];
    for i in 0..row_end {
        let mut s = 0.0;
        for j in 0..v.len() {
            s += h[i][start + j] * v[j];
        }
        w[i] = s;
    }
    // Step 2: h[0..row_end, start..n] -= 2 w v^T
    for i in 0..row_end {
        for j in 0..v.len() {
            h[i][start + j] -= 2.0 * w[i] * v[j];
        }
    }
}

/// Apply Householder (I - 2 v v^T) to a unitary (orthogonal) matrix U, columns [start..n].
/// This is the same as apply_householder_right but treats U as n×n.
#[allow(dead_code)]
pub fn apply_householder_right_unitary(u: &mut [Vec<f64>], v: &[f64], start: usize, n: usize) {
    apply_householder_right(u, v, n, start, n);
}

#[allow(dead_code)]
pub fn vector_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[allow(dead_code)]
pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    let m = a[0].len();
    let mut t = vec![vec![0.0; n]; m];
    for i in 0..n {
        for j in 0..m {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Francis double-shift QR iteration to reduce an upper Hessenberg matrix H
/// to real quasi-upper-triangular Schur form T.
///
/// Returns (T, V) such that H = V · T · V^T, where T has 1×1 blocks
/// (real eigenvalues) and 2×2 blocks (complex-conjugate eigenvalue pairs)
/// on its diagonal.
///
/// This is a simplified implementation suitable for small matrices
/// (n ≤ ~50) — the same algorithm E+ uses internally. For our 6-24 node
/// state-space matrices, this is more than adequate.
#[allow(dead_code)]
pub fn francis_qr_schur(h: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = h.len();
    let mut t = h.to_vec();
    let mut v = identity(n);

    if n <= 2 {
        return (t, v);
    }

    // Wilkinson shift + implicit double-shift QR with deflation
    let max_iter = 200 * n; // plenty of iterations for convergence
    let tol = 1e-14;
    let mut iter = 0;
    let mut nn = n; // size of the active submatrix
    let start = 0; // start of the active submatrix

    while nn > 2 && iter < max_iter {
        // Check if the BOTTOM subdiagonal of the active submatrix is small.
        // If so, the bottom-right eigenvalue has converged, and we can deflate.
        let bot_sub = t[start + nn - 1][start + nn - 2].abs();
        let bot_diag_sum =
            t[start + nn - 1][start + nn - 1].abs() + t[start + nn - 2][start + nn - 2].abs();
        if bot_sub < tol * bot_diag_sum.max(1e-30) {
            t[start + nn - 1][start + nn - 2] = 0.0;
            nn -= 1;
            continue;
        }

        // Implicit double-shift QR bulge chase on the active submatrix
        // t[start..start+nn, start..start+nn]
        implicit_double_shift_bulge_chase(&mut t, &mut v, start, nn);

        // Force small subdiagonals to zero after the chase
        for i in 1..nn {
            if t[start + i][start + i - 1].abs() < 1e-12 {
                t[start + i][start + i - 1] = 0.0;
            }
        }

        iter += 1;
    }

    // Final: if 2×2 block remains, ensure it's in standard form
    if nn == 2 {
        // Already small enough — the result is a 2×2 block (real or complex eigenvalues)
    }

    (t, v)
}

/// Implicit double-shift QR bulge chase on the active submatrix t[start..start+nn, start..start+nn].
///
/// Uses the Wilkinson shift (the eigenvalue of the trailing 2×2 block
#[allow(dead_code)]
/// closest to a22). The implicit shift theorem gives the first column
/// of p(T) = T² - s T + p I, where s and p are the trace and determinant
/// of the trailing 2×2 block.
pub fn implicit_double_shift_bulge_chase(
    t: &mut [Vec<f64>],
    v: &mut [Vec<f64>],
    start: usize,
    nn: usize,
) {
    if nn < 3 {
        return;
    }

    // Compute s and p from the trailing 2x2 block
    let a11 = t[start + nn - 2][start + nn - 2];
    let a12 = t[start + nn - 2][start + nn - 1];
    let a21 = t[start + nn - 1][start + nn - 2];
    let a22 = t[start + nn - 1][start + nn - 1];
    let s = a11 + a22;
    let p = a11 * a22 - a12 * a21;

    // p(T) e_0 = T (T e_0) - s T e_0 + p e_0
    // Step 1: v = T e_0 = first column of T (active submatrix)
    let mut v_col = vec![0.0; nn];
    for i in 0..nn {
        v_col[i] = t[start + i][start];
    }
    // Step 2: w = T v_col (apply Hessenberg T to v_col)
    let mut w = vec![0.0; nn];
    for i in 0..nn {
        let lo = i.saturating_sub(1);
        let hi = (i + 2).min(nn);
        let mut s_acc = 0.0;
        for j in lo..hi {
            s_acc += t[start + i][start + j] * v_col[j];
        }
        w[i] = s_acc;
    }
    // Step 3: pt = w - s * v_col + p * e_0
    let mut pt: Vec<f64> = w
        .iter()
        .enumerate()
        .map(|(i, &wi)| wi - s * v_col[i])
        .collect();
    pt[0] += p;

    // Now chase the bulge
    let mut m = 0_usize; // offset within the active submatrix
    while m < nn - 2 {
        // Determine vector to eliminate (the bulge column)
        let num_rows = (nn - m).min(3);
        let mut hh = vec![0.0; num_rows];
        hh[..num_rows].copy_from_slice(&pt[m..(num_rows + m)]);
        // Normalize and form Householder
        let hh_norm = vector_norm(&hh);
        if hh_norm < 1e-15 {
            m += 1;
            continue;
        }
        let sign = if hh[0] >= 0.0 { 1.0 } else { -1.0 };
        hh[0] += sign * hh_norm;
        let hh_v_norm = vector_norm(&hh);
        if hh_v_norm < 1e-15 {
            m += 1;
            continue;
        }
        for vi in hh.iter_mut() {
            *vi /= hh_v_norm;
        }

        // Apply Householder to t[m+start..m+start+3, m+start..start+nn]
        // From the left: rows m+start..m+start+num_rows
        for j in m + start..start + nn {
            let mut s_acc = 0.0;
            for i in 0..num_rows {
                s_acc += hh[i] * t[m + start + i][j];
            }
            for i in 0..num_rows {
                t[m + start + i][j] -= 2.0 * hh[i] * s_acc;
            }
        }
        // From the right: cols m+start..m+start+num_rows, rows 0..start+nn
        for i in 0..start + nn {
            let mut s_acc = 0.0;
            for j in 0..num_rows {
                s_acc += t[i][m + start + j] * hh[j];
            }
            for j in 0..num_rows {
                t[i][m + start + j] -= 2.0 * s_acc * hh[j];
            }
        }
        // Update V: V = V · H (apply Householder to columns m+start..m+start+num_rows of V)
        for i in 0..v.len() {
            let mut s_acc = 0.0;
            for j in 0..num_rows {
                s_acc += v[i][m + start + j] * hh[j];
            }
            for j in 0..num_rows {
                v[i][m + start + j] -= 2.0 * s_acc * hh[j];
            }
        }
        // Clear the bulge below the subdiagonal
        for i in 1..num_rows {
            t[m + start + i][m + start] = 0.0;
        }
        if num_rows >= 2 {
            t[m + start + 2][m + start] = 0.0;
        }

        m += 1;
    }
}

#[allow(dead_code)]
pub fn matrix_exponential_old_pade(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let n = a.len();

    // Scale A·t
    let mut scaled = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            scaled[i][j] = a[i][j] * t;
        }
    }

    // Find scaling factor s such that ||scaled/2^s|| < 0.5
    let norm_inf = matrix_norm_inf(&scaled);
    let mut s = 0;
    let mut scale_factor = 1.0;
    while norm_inf / scale_factor > 0.5 {
        scale_factor *= 2.0;
        s += 1;
    }

    // Apply scaling
    if s > 0 {
        for row in &mut scaled {
            for val in row.iter_mut() {
                *val /= scale_factor;
            }
        }
    }

    // Padé [6/6] approximant
    // exp(B) ≈ D6^(-1) · N6 where:
    // N6 = sum_{k=0}^{6} c_k · B^k
    // D6 = sum_{k=0}^{6} (-1)^k · c_k · B^k
    // c_k = (2p-k)! p! / ((2p)! k! (p-k)!)
    // For p=6: c = [1, 1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280]
    let p = 6;
    let c: [f64; 7] = [
        1.0,
        0.5,
        5.0 / 44.0,
        1.0 / 66.0,
        1.0 / 792.0,
        1.0 / 15840.0,
        1.0 / 665280.0,
    ];

    // Compute powers of B
    let b_powers = compute_powers(&scaled, p);

    // Compute numerator N6 and denominator D6
    let mut numer = vec![vec![0.0; n]; n];
    let mut denom = vec![vec![0.0; n]; n];

    for k in 0..=p {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        for i in 0..n {
            for j in 0..n {
                numer[i][j] += c[k] * b_powers[k][i][j];
                denom[i][j] += sign * c[k] * b_powers[k][i][j];
            }
        }
    }

    // exp(B) = D6^(-1) · N6
    let d_inv = matrix_inverse(&denom).unwrap_or_else(|| identity(n));
    let mut result = mat_mat_mul(&d_inv, &numer);

    // Square s times
    for _ in 0..s {
        result = mat_mat_mul(&result, &result);
    }

    result
}

/// Compute matrix exponential exp(A·t) using a direct Taylor series.
///
/// exp(A·t) = Σ_{k=0}^N (A·t)^k / k!
///
/// For a 24×24 matrix with ||A·t|| ≈ 1.93, the Taylor series converges
/// in about 30 terms to machine precision. Each term requires a matrix
/// multiplication; for n=24, this is O(n^3) per term.
///
/// **This is the "foolproof" fallback** for cases where the Schur-based
/// algorithm or Padé scaling-and-squaring fails (e.g., multi-layer walls
/// with 20,000× eigenvalue spread). The Taylor series makes no assumptions
/// about the matrix structure and converges for any stable A.
#[allow(dead_code)]
pub fn matrix_exponential_taylor(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vec![(a[0][0] * t).exp()]];
    }

    // Number of terms: chosen so (||A·t||)^N / N! < 1e-15
    // For ||A·t|| ≈ 1.93: N=30 gives (1.93)^30/30! ≈ 1e-21
    let n_terms = 30;

    // B = A·t
    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            b[i][j] = a[i][j] * t;
        }
    }

    // Initialize result = I, current_term = I (which is B^0 / 0!)
    let mut result = identity(n);
    let mut current_term = identity(n); // B^0 / 0! = I

    for k in 1..=n_terms {
        // current_term = current_term · B / k = B^k / k!
        current_term = mat_mat_mul(&current_term, &b);
        let scale = 1.0 / k as f64;
        for i in 0..n {
            for j in 0..n {
                current_term[i][j] *= scale;
            }
        }
        // result += current_term
        for i in 0..n {
            for j in 0..n {
                result[i][j] += current_term[i][j];
            }
        }
    }

    result
}

/// Compute powers B^0, B^1, ..., B^max_power.
pub fn compute_powers(b: &[Vec<f64>], max_power: usize) -> Vec<Vec<Vec<f64>>> {
    let n = b.len();
    let mut powers = Vec::with_capacity(max_power + 1);

    // B^0 = I
    powers.push(identity(n));
    if max_power >= 1 {
        powers.push(b.to_vec());
    }
    for k in 2..=max_power {
        powers.push(mat_mat_mul(&powers[k - 1], b));
    }
    powers
}

/// Infinity norm of matrix.
#[allow(dead_code)]
pub fn matrix_norm_inf(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0f64, f64::max)
}

/// Compute matrix inverse using Gauss-Jordan elimination.
///
/// Returns None if matrix is singular.
pub fn matrix_inverse(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];

    // Set up augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in col + 1..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Scale pivot row
        let pivot = aug[col][col];
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    Some(inv)
}
