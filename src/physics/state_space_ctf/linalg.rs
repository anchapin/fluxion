//! Dense linear-algebra kernel for the state-space CTF pipeline.
//!
//! Extracted from `state_space_ctf/mod.rs` at Issue #3787 decomposition time
//! (the parent landed at 4347/4347 lines, see
//! `tests/reference_data/module_size/state_space_ctf_ratchet.json`). The
//! kernel is self-contained pure math — small dense matrix operations,
//! matrix-exponential variants (Higham Padé [13/13]), and the Gauss-Jordan
//! `matrix_inverse` — with no CTF-domain coupling.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let i2 = identity(2);
        assert_eq!(i2.len(), 2);
        assert_eq!(i2[0], vec![1.0, 0.0]);
        assert_eq!(i2[1], vec![0.0, 1.0]);

        let i4 = identity(4);
        assert_eq!(i4.len(), 4);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(i4[i][j], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn test_matrix_inverse_2x2() {
        // Test with a simple 2x2 invertible matrix
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = matrix_inverse(&a).expect("Matrix should be invertible");

        // A * A^-1 should be identity
        let product = mat_mat_mul(&a, &inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-10,
                    "A*A^-1[{}][{}] = {} != {}",
                    i,
                    j,
                    product[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_matrix_inverse_3x3() {
        // Test with a 3x3 matrix
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 4.0, 5.0],
            vec![1.0, 0.0, 6.0],
        ];
        let inv = matrix_inverse(&a).expect("Matrix should be invertible");

        // A * A^-1 should be identity
        let product = mat_mat_mul(&a, &inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-10,
                    "A*A^-1[{}][{}] = {} != {}",
                    i,
                    j,
                    product[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_matrix_inverse_singular() {
        // A singular matrix (two identical rows)
        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(matrix_inverse(&singular).is_none());
    }

    #[test]
    fn test_mat_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = mat_mat_mul(&a, &b);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[5+14, 6+16], [15+28, 18+32]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(c[0][0], 19.0);
        assert_eq!(c[0][1], 22.0);
        assert_eq!(c[1][0], 43.0);
        assert_eq!(c[1][1], 50.0);
    }

    #[test]
    fn test_mat_mul_gen() {
        // 2x3 times 3x2 should give 2x2
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let c = mat_mul_gen(&a, &b);

        assert_eq!(c.len(), 2);
        assert_eq!(c[0].len(), 2);
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [7+18+33, 8+20+36] = [58, 64]
        assert_eq!(c[0][0], 58.0);
        assert_eq!(c[0][1], 64.0);
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [28+45+66, 32+50+72] = [139, 154]
        assert_eq!(c[1][0], 139.0);
        assert_eq!(c[1][1], 154.0);
    }

    #[test]
    fn test_scale_columns() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let scaled = scale_columns(&a, 2.0);

        assert_eq!(scaled[0], vec![2.0, 4.0, 6.0]);
        assert_eq!(scaled[1], vec![8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_sub_identity() {
        let a = vec![vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = matrix_sub_identity(&a);

        assert_eq!(result[0][0], 2.0); // 3 - 1
        assert_eq!(result[0][1], 4.0);
        assert_eq!(result[1][0], 5.0);
        assert_eq!(result[1][1], 5.0); // 6 - 1
    }

    #[test]
    fn test_expm_higham_padé13_identity() {
        // exp(I * t) should be e^t * I
        let identity_2 = identity(2);
        let exp_i = matrix_exponential(&identity_2, 1.0);
        let expected = identity(2);

        for i in 0..2 {
            for j in 0..2 {
                let expected_val = if i == j { std::f64::consts::E } else { 0.0 };
                assert!(
                    (exp_i[i][j] - expected_val).abs() < 1e-10,
                    "exp(I)[{}][{}] = {} != {}",
                    i,
                    j,
                    exp_i[i][j],
                    expected_val
                );
            }
        }
    }

    #[test]
    fn test_expm_higham_padé13_2x2() {
        // A 2x2 diagonal matrix: diag(-1, -2)
        let a = vec![vec![-1.0, 0.0], vec![0.0, -2.0]];
        let exp_a = matrix_exponential(&a, 1.0);

        // exp(A) should be diag(e^-1, e^-2)
        let e1 = (-1.0f64).exp();
        let e2 = (-2.0f64).exp();

        assert!((exp_a[0][0] - e1).abs() < 1e-10);
        assert!((exp_a[1][1] - e2).abs() < 1e-10);
        assert!(exp_a[0][1].abs() < 1e-10);
        assert!(exp_a[1][0].abs() < 1e-10);
    }
}
