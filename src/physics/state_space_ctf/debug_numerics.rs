//! Debug/reference numerical linear-algebra kernels for the state-space CTF
//! test suite. These are hand-rolled implementations of:
//! - Francis QR iteration for real-Schur decomposition
//! - Householder reflections for Hessenberg reduction  
//! - Taylor-series and Padé [6/6] matrix exponentials (reference implementations)
//!
//! These functions are NOT used in production. The production code uses
//! `matrix_exponential_faer` (Higham Padé [13/13] on a Schur-reduced matrix).
//! These reference implementations are kept here for test comparisons only.
//!
//! The module is `#[cfg(test)]`-gated so these never appear in production builds.

use super::linalg::{self, compute_powers, identity, mat_mat_mul, matrix_inverse};

/// Householder reduction of a general n×n matrix A to upper Hessenberg form.
///
/// Returns (H, U) such that A = U · H · U^T, where H is upper Hessenberg
/// (h[i][j] = 0 for i > j+1) and U is the product of Householder reflections
/// (orthogonal).
pub fn householder_to_hessenberg(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut h = a.to_vec();
    let mut u = identity(n);

    if n <= 2 {
        return (h, u);
    }

    for k in 0..n.saturating_sub(2) {
        if n - (k + 2) < 1 {
            continue;
        }
        let mut x: Vec<f64> = (k + 2..n).map(|i| h[i][k]).collect();
        let x_norm = vector_norm(&x);
        if x_norm < 1e-15 {
            continue;
        }

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

        apply_householder_left(&mut h, &v, k + 2, k, n);
        apply_householder_right(&mut h, &v, 0, k + 2, n);
        apply_householder_right_unitary(&mut u, &v, k + 2, n);
    }

    (h, u)
}

/// Apply Householder (I - 2 v v^T) to rows [start..n] of h, columns [col_start..n].
pub fn apply_householder_left(
    h: &mut [Vec<f64>],
    v: &[f64],
    start: usize,
    col_start: usize,
    n: usize,
) {
    let mut w = vec![0.0; n - col_start];
    for j in 0..n - col_start {
        let mut s = 0.0;
        for i in 0..v.len() {
            s += v[i] * h[start + i][col_start + j];
        }
        w[j] = s;
    }
    for i in 0..v.len() {
        for j in 0..n - col_start {
            h[start + i][col_start + j] -= 2.0 * v[i] * w[j];
        }
    }
}

/// Apply Householder (I - 2 v v^T) to columns [start..n] of h, rows [0..row_end].
pub fn apply_householder_right(
    h: &mut [Vec<f64>],
    v: &[f64],
    row_end: usize,
    start: usize,
    _n: usize,
) {
    let mut w = vec![0.0; row_end];
    for i in 0..row_end {
        let mut s = 0.0;
        for j in 0..v.len() {
            s += h[i][start + j] * v[j];
        }
        w[i] = s;
    }
    for i in 0..row_end {
        for j in 0..v.len() {
            h[i][start + j] -= 2.0 * w[i] * v[j];
        }
    }
}

/// Apply Householder (I - 2 v v^T) to a unitary (orthogonal) matrix U, columns [start..n].
pub fn apply_householder_right_unitary(u: &mut [Vec<f64>], v: &[f64], start: usize, n: usize) {
    apply_householder_right(u, v, n, start, n);
}

/// Euclidean (L2) norm of a vector.
pub fn vector_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Transpose of a matrix.
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
pub fn francis_qr_schur(h: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = h.len();
    let mut t = h.to_vec();
    let mut v = identity(n);

    if n <= 2 {
        return (t, v);
    }

    let max_iter = 200 * n;
    let tol = 1e-14;
    let mut iter = 0;
    let mut nn = n;
    let start = 0;

    while nn > 2 && iter < max_iter {
        let bot_sub = t[start + nn - 1][start + nn - 2].abs();
        let bot_diag_sum =
            t[start + nn - 1][start + nn - 1].abs() + t[start + nn - 2][start + nn - 2].abs();
        if bot_sub < tol * bot_diag_sum.max(1e-30) {
            t[start + nn - 1][start + nn - 2] = 0.0;
            nn -= 1;
            continue;
        }

        implicit_double_shift_bulge_chase(&mut t, &mut v, start, nn);

        for i in 1..nn {
            if t[start + i][start + i - 1].abs() < 1e-12 {
                t[start + i][start + i - 1] = 0.0;
            }
        }

        iter += 1;
    }

    (t, v)
}

/// Implicit double-shift QR bulge chase on the active submatrix
/// t[start..start+nn, start..start+nn].
pub fn implicit_double_shift_bulge_chase(
    t: &mut [Vec<f64>],
    v: &mut [Vec<f64>],
    start: usize,
    nn: usize,
) {
    if nn < 3 {
        return;
    }

    let a11 = t[start + nn - 2][start + nn - 2];
    let a12 = t[start + nn - 2][start + nn - 1];
    let a21 = t[start + nn - 1][start + nn - 2];
    let a22 = t[start + nn - 1][start + nn - 1];
    let s = a11 + a22;
    let p = a11 * a22 - a12 * a21;

    let mut v_col = vec![0.0; nn];
    for i in 0..nn {
        v_col[i] = t[start + i][start];
    }
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
    let mut pt: Vec<f64> = w
        .iter()
        .enumerate()
        .map(|(i, &wi)| wi - s * v_col[i])
        .collect();
    pt[0] += p;

    let mut m = 0_usize;
    while m < nn - 2 {
        let num_rows = (nn - m).min(3);
        let mut hh = vec![0.0; num_rows];
        hh[..num_rows].copy_from_slice(&pt[m..(num_rows + m)]);
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

        for j in m + start..start + nn {
            let mut s_acc = 0.0;
            for i in 0..num_rows {
                s_acc += hh[i] * t[m + start + i][j];
            }
            for i in 0..num_rows {
                t[m + start + i][j] -= 2.0 * hh[i] * s_acc;
            }
        }
        for i in 0..start + nn {
            let mut s_acc = 0.0;
            for j in 0..num_rows {
                s_acc += t[i][m + start + j] * hh[j];
            }
            for j in 0..num_rows {
                t[i][m + start + j] -= 2.0 * s_acc * hh[j];
            }
        }
        for i in 0..v.len() {
            let mut s_acc = 0.0;
            for j in 0..num_rows {
                s_acc += v[i][m + start + j] * hh[j];
            }
            for j in 0..num_rows {
                v[i][m + start + j] -= 2.0 * s_acc * hh[j];
            }
        }
        for i in 1..num_rows {
            t[m + start + i][m + start] = 0.0;
        }
        if num_rows >= 2 {
            t[m + start + 2][m + start] = 0.0;
        }

        m += 1;
    }
}

/// Infinity norm of matrix.
pub fn matrix_norm_inf(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0f64, f64::max)
}

/// Legacy Padé [6/6] matrix exponential (reference implementation).
/// Production code uses `matrix_exponential_faer` (Higham Padé [13/13]).
pub fn matrix_exponential_old_pade(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let n = a.len();

    let mut scaled = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            scaled[i][j] = a[i][j] * t;
        }
    }

    let norm_inf = matrix_norm_inf(&scaled);
    let mut s = 0;
    let mut scale_factor = 1.0;
    while norm_inf / scale_factor > 0.5 {
        scale_factor *= 2.0;
        s += 1;
    }

    if s > 0 {
        for row in &mut scaled {
            for val in row.iter_mut() {
                *val /= scale_factor;
            }
        }
    }

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

    let b_powers = compute_powers(&scaled, p);

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

    let d_inv = matrix_inverse(&denom).unwrap_or_else(|| identity(n));
    let mut result = mat_mat_mul(&d_inv, &numer);

    for _ in 0..s {
        result = mat_mat_mul(&result, &result);
    }

    result
}

/// Compute matrix exponential exp(A·t) using a direct Taylor series.
/// This is a reference implementation for testing; production uses
/// `matrix_exponential_faer`.
pub fn matrix_exponential_taylor(a: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![vec![(a[0][0] * t).exp()]];
    }

    let n_terms = 30;

    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            b[i][j] = a[i][j] * t;
        }
    }

    let mut result = identity(n);
    let mut current_term = identity(n);

    for k in 1..=n_terms {
        current_term = mat_mat_mul(&current_term, &b);
        let scale = 1.0 / k as f64;
        for i in 0..n {
            for j in 0..n {
                current_term[i][j] *= scale;
            }
        }
        for i in 0..n {
            for j in 0..n {
                result[i][j] += current_term[i][j];
            }
        }
    }

    result
}
