//! expm / matrix-exponential convergence debug tests (extracted from
//! `state_space_ctf/mod.rs` at Issue #3787 decomposition time to keep
//! the parent file under the Issue #3457 module-size ratchet ceiling).
//!
//! Children see the parent module's items through `use super::*;` —
//! the PR #3688 `coverage_tests` precedent.

use super::linalg::*;
use super::*;

/// Verify that expm_higham_padé13 matches the Taylor series for a 6×6 wall matrix.
/// This test isolates the matrix exponential from the CTF extraction pipeline.
#[test]
fn debug_pade13_vs_taylor_6x6() {
    // Build 1-layer concrete wall A matrix (same as test_ctf_wrapper_diurnal_simulation)
    let layers = vec![CTFMaterial::new("Concrete", 0.2, 1.4, 2300.0, 840.0)];
    let nodes = compute_nodes_per_layer(&layers, 3600.0);
    let n: usize = nodes.iter().sum();
    let (a, _b, _c, _d) = build_state_space_matrices(&layers, &nodes, n);

    let t = 3600.0;

    // Compute A*t
    let mut at = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            at[i][j] = a[i][j] * t;
        }
    }
    eprintln!("A*t matrix ({}x{}):", n, n);
    for i in 0..n {
        eprintln!("  row {}: [{:.6e}, {:.6e}, ...]", i, at[i][0], at[i][n - 1]);
    }

    // Compute via Padé [13/13]
    let exp_pade = matrix_exponential(&a, t);

    // Compute via scaled Taylor (reference)
    let exp_taylor = {
        // Scale down so ||A*t/2^s|| < 0.5
        let norm1 = matrix_norm_1(&at);
        let s = if norm1 > 0.5 {
            norm1.log2().ceil() as usize
        } else {
            0
        };
        eprintln!("||A*t||_1 = {:.6}, s = {}", norm1, s);
        let scale = 1.0 / (1u64 << s.min(63)) as f64;
        let mut b = at.clone();
        for row in &mut b {
            for val in row.iter_mut() {
                *val *= scale;
            }
        }
        // Taylor series with 60 terms
        let mut result = identity(n);
        let mut term = identity(n);
        for k in 1..=60 {
            term = mat_mat_mul(&term, &b);
            let sc = 1.0 / k as f64;
            for i in 0..n {
                for j in 0..n {
                    term[i][j] *= sc;
                }
            }
            for i in 0..n {
                for j in 0..n {
                    result[i][j] += term[i][j];
                }
            }
        }
        // Square s times
        for _ in 0..s {
            result = mat_mat_mul(&result, &result);
        }
        result
    };

    // Compare entry-by-entry
    let mut max_err = 0.0f64;
    let mut max_err_idx = (0, 0);
    for i in 0..n {
        for j in 0..n {
            let err = (exp_pade[i][j] - exp_taylor[i][j]).abs();
            if err > max_err {
                max_err = err;
                max_err_idx = (i, j);
            }
        }
    }

    eprintln!(
        "\nMax entry error: {:.6e} at ({},{})",
        max_err, max_err_idx.0, max_err_idx.1
    );
    eprintln!(
        "Padé exp[0][0] = {:.10}, Taylor = {:.10}",
        exp_pade[0][0], exp_taylor[0][0]
    );
    eprintln!(
        "Padé exp[0][n-1] = {:.10}, Taylor = {:.10}",
        exp_pade[0][n - 1],
        exp_taylor[0][n - 1]
    );
    eprintln!(
        "Padé exp[n-1][0] = {:.10}, Taylor = {:.10}",
        exp_pade[n - 1][0],
        exp_taylor[n - 1][0]
    );
    eprintln!(
        "Padé exp[n-1][n-1] = {:.10}, Taylor = {:.10}",
        exp_pade[n - 1][n - 1],
        exp_taylor[n - 1][n - 1]
    );

    // Column sums of exp(A*t) should give (exp(A*t)) * ones
    // For a symmetric tridiagonal matrix, this has a nice closed form
    eprintln!("\nColumn sums:");
    for j in 0..n {
        let cs_pade: f64 = (0..n).map(|i| exp_pade[i][j]).sum();
        let cs_taylor: f64 = (0..n).map(|i| exp_taylor[i][j]).sum();
        eprintln!(
            "  col {}: Padé={:.10e}, Taylor={:.10e}, diff={:.2e}",
            j,
            cs_pade,
            cs_taylor,
            (cs_pade - cs_taylor).abs()
        );
    }

    // Also write results to file for inspection
    let mut f = std::fs::File::create("/tmp/pade_vs_taylor.txt").unwrap();
    use std::io::Write;
    writeln!(
        f,
        "Max entry error: {:.6e} at ({},{})",
        max_err, max_err_idx.0, max_err_idx.1
    )
    .unwrap();
    writeln!(f, "Padé exp[0][0] = {:.15}", exp_pade[0][0]).unwrap();
    writeln!(f, "Taylor exp[0][0] = {:.15}", exp_taylor[0][0]).unwrap();
    writeln!(f, "Padé exp[n-1][n-1] = {:.15}", exp_pade[n - 1][n - 1]).unwrap();
    writeln!(f, "Taylor exp[n-1][n-1] = {:.15}", exp_taylor[n - 1][n - 1]).unwrap();
    for j in 0..n {
        let cs_pade: f64 = (0..n).map(|i| exp_pade[i][j]).sum();
        let cs_taylor: f64 = (0..n).map(|i| exp_taylor[i][j]).sum();
        writeln!(
            f,
            "col {}: Padé={:.15e}, Taylor={:.15e}, diff={:.2e}",
            j,
            cs_pade,
            cs_taylor,
            (cs_pade - cs_taylor).abs()
        )
        .unwrap();
    }

    // The two should agree to ~13 digits
    assert!(
        max_err < 1e-10,
        "Padé and Taylor differ by {} at ({},{})",
        max_err,
        max_err_idx.0,
        max_err_idx.1
    );
}

/// Verify the CTF extraction pipeline: gamma1, gamma2, s0, and the steady-state check.
/// The matrix exponential is verified correct above, so any CTF error is in the pipeline.
#[test]
fn debug_ctf_pipeline_1layer() {
    let layers = vec![CTFMaterial::new("Concrete", 0.2, 1.4, 2300.0, 840.0)];
    let timestep = 3600.0;
    let nodes = compute_nodes_per_layer(&layers, timestep);
    let n: usize = nodes.iter().sum();
    let (a_mat, b_mat, c_mat, d_mat) = build_state_space_matrices(&layers, &nodes, n);

    let mut f = std::fs::File::create("/tmp/ctf_pipeline.txt").unwrap();
    use std::io::Write;

    writeln!(
        f,
        "=== 1-layer concrete wall (L=0.2, k=1.4, rho=2300, cp=840) ==="
    )
    .unwrap();
    writeln!(f, "n = {}, nodes_per_layer = {:?}", n, nodes).unwrap();

    let u_bare = 1.0 / layers.iter().map(|l| l.resistance()).sum::<f64>();
    writeln!(f, "U_bare = {:.6}", u_bare).unwrap();

    // Step 1: matrix exponential
    let a_exp = matrix_exponential(&a_mat, timestep);
    let a_inv = matrix_inverse(&a_mat).expect("A should be invertible");

    // Check A_inv * A ≈ I
    let a_inv_a = mat_mat_mul(&a_inv, &a_mat);
    let mut err_a_inv = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            err_a_inv = err_a_inv.max((a_inv_a[i][j] - expected).abs());
        }
    }
    writeln!(f, "\n||A_inv * A - I||_max = {:.2e}", err_a_inv).unwrap();

    // Step 2: Gamma1 = A_inv · (A_exp - I) · B
    let a_exp_minus_i = matrix_sub_identity(&a_exp);
    let temp = mat_mat_mul_col(&a_exp_minus_i, &b_mat);
    let gamma1 = mat_mat_mul_col(&a_inv, &temp);

    writeln!(f, "\nB matrix ({}x2):", n).unwrap();
    for i in 0..n {
        writeln!(f, "  B[{}] = [{:.6e}, {:.6e}]", i, b_mat[i][0], b_mat[i][1]).unwrap();
    }

    writeln!(f, "\nGamma1 = A_inv · (exp(A·t) - I) · B ({}x2):", n).unwrap();
    for i in 0..n {
        writeln!(
            f,
            "  G1[{}] = [{:.6e}, {:.6e}]",
            i, gamma1[i][0], gamma1[i][1]
        )
        .unwrap();
    }

    // Step 3: Gamma2 = A_inv · (Gamma1/dt - B)
    let gamma1_scaled = scale_columns(&gamma1, 1.0 / timestep);
    let gamma2_diff = matrix_sub_col(&gamma1_scaled, &b_mat);
    let gamma2 = mat_mat_mul_col(&a_inv, &gamma2_diff);

    writeln!(f, "\nGamma2 = A_inv · (Gamma1/dt - B) ({}x2):", n).unwrap();
    for i in 0..n {
        writeln!(
            f,
            "  G2[{}] = [{:.6e}, {:.6e}]",
            i, gamma2[i][0], gamma2[i][1]
        )
        .unwrap();
    }

    // Step 4: s0 = D + C · Gamma2
    let mut s0 = vec![vec![0.0f64; 2]; 2];
    for j in 0..2 {
        for k in 0..2 {
            s0[j][k] = d_mat[j][k];
            for i in 0..n {
                s0[j][k] += c_mat[j][i] * gamma2[i][k];
            }
        }
    }

    writeln!(f, "\nC matrix (2x{}):", n).unwrap();
    writeln!(
        f,
        "  C[0] = [{:.6e}, ..., {:.6e}]",
        c_mat[0][0],
        c_mat[0][n - 1]
    )
    .unwrap();
    writeln!(
        f,
        "  C[1] = [{:.6e}, ..., {:.6e}]",
        c_mat[1][0],
        c_mat[1][n - 1]
    )
    .unwrap();

    writeln!(f, "\nD matrix (2x2):").unwrap();
    writeln!(
        f,
        "  D = [[{:.6e}, {:.6e}], [{:.6e}, {:.6e}]]",
        d_mat[0][0], d_mat[0][1], d_mat[1][0], d_mat[1][1]
    )
    .unwrap();

    writeln!(f, "\ns0 = D + C·Gamma2:").unwrap();
    writeln!(
        f,
        "  s0 = [[{:.10}, {:.10}], [{:.10}, {:.10}]]",
        s0[0][0], s0[0][1], s0[1][0], s0[1][1]
    )
    .unwrap();

    // Steady-state check: sum(s0[j]) + sum(all s[j]) should equal U_bare
    // At steady state: y = C·x_ss + D·u where x_ss = -A_inv·B·u
    // x_ss = -A_inv · B · [T_ext, T_int]^T
    // For T_ext = 1, T_int = 0: x_ss = -A_inv · B · [1, 0]^T
    let x_ss_ext: Vec<f64> = (0..n)
        .map(|i| -(0..n).map(|j| a_inv[i][j] * b_mat[j][0]).sum::<f64>())
        .collect();
    let x_ss_int: Vec<f64> = (0..n)
        .map(|i| -(0..n).map(|j| a_inv[i][j] * b_mat[j][1]).sum::<f64>())
        .collect();

    // DC gain: y = C · x_ss + D · u
    let dc_gain_10 = (0..n).map(|i| c_mat[1][i] * x_ss_ext[i]).sum::<f64>() + d_mat[1][0];
    let dc_gain_11 = (0..n).map(|i| c_mat[1][i] * x_ss_int[i]).sum::<f64>() + d_mat[1][1];

    writeln!(f, "\nDC gain check (steady-state transfer function):").unwrap();
    writeln!(
        f,
        "  DC gain: T_ext → q_int = {:.10} (should be +{:.10}=U_bare)",
        dc_gain_10, u_bare
    )
    .unwrap();
    writeln!(
        f,
        "  DC gain: T_int → q_int = {:.10} (should be -{:.10}=-U_bare)",
        dc_gain_11, u_bare
    )
    .unwrap();
    writeln!(f, "  U_bare = {:.10}", u_bare).unwrap();

    // Now check: does s0 match the DC gain?
    // The total CTF sum should match the DC gain
    // sum(X) = s0[1][0] + sum(s[1][0]) = DC gain for T_ext → q_int
    writeln!(
        f,
        "\ns0[1][0] = {:.10} (DC gain component from Gamma2)",
        s0[1][0]
    )
    .unwrap();
    writeln!(
        f,
        "s0[1][1] = {:.10} (DC gain component from Gamma2)",
        s0[1][1]
    )
    .unwrap();

    // Check gamma2 steady-state:
    // Gamma2 should satisfy: C · Gamma2 + D = DC gain matrix
    let cg2_10 = (0..n).map(|i| c_mat[1][i] * gamma2[i][0]).sum::<f64>() + d_mat[1][0];
    let cg2_11 = (0..n).map(|i| c_mat[1][i] * gamma2[i][1]).sum::<f64>() + d_mat[1][1];

    writeln!(f, "\nC·Gamma2 + D:").unwrap();
    writeln!(
        f,
        "  [1][0] = {:.10} (should match DC gain {:.10})",
        cg2_10, dc_gain_10
    )
    .unwrap();
    writeln!(
        f,
        "  [1][1] = {:.10} (should match DC gain {:.10})",
        cg2_11, dc_gain_11
    )
    .unwrap();

    // Seem eq 2.1.24: s0 = D + C · Gamma2 (check that the series terms s[j] sum to
    // the difference between the total DC gain and s0)
    // Total: s0 + sum(s[j]) should give the same DC gain when combined with Phi terms.
    // At steady state with constant u: q = s0·u + sum(s·u) + sum(Phi·q)
    // => q * (1 - sum(Phi)) = (s0 + sum(s)) * u
    // And q = DC_gain * u
    // So: (s0 + sum(s)) / (1 - sum(Phi)) = DC_gain

    writeln!(f, "\n=== Sanity check ===").unwrap();
    if (dc_gain_10 - u_bare).abs() / u_bare > 0.01 {
        writeln!(
            f,
            "WARNING: DC gain T_ext→q_int = {:.6} != U_bare = {:.6}",
            dc_gain_10, u_bare
        )
        .unwrap();
    } else {
        writeln!(f, "DC gain T_ext→q_int matches U_bare: OK").unwrap();
    }
}

#[test]
fn debug_schur_expm_2x2() {
    // A 2x2 symmetric tridiagonal with eigenvalues -1, -2
    let a = vec![vec![-1.5, 0.5], vec![0.5, -1.5]];
    let r = matrix_exponential(&a, 1.0);
    eprintln!("exp(2x2 A) = {:?}", r);
    let e1 = (-1.0f64).exp();
    let e2 = (-2.0f64).exp();
    eprintln!(
        "Expected: [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
        (e1 + e2) / 2.0,
        (e1 - e2) / 2.0,
        (e1 - e2) / 2.0,
        (e1 + e2) / 2.0
    );
}

// Compare Schur vs old Pade on the actual 4-layer Case 900
#[test]
fn debug_compare_schur_vs_pade_4layer() {
    // Build the 4-layer Case 900 A matrix
    let layers = vec![
        CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
        CTFMaterial::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
        CTFMaterial::new("Insulation", 0.050, 0.04, 50.0, 840.0),
        CTFMaterial::new("Brick", 0.100, 0.81, 1920.0, 790.0),
    ];
    let nodes = compute_nodes_per_layer(&layers, 3600.0);
    let n: usize = nodes.iter().sum();
    eprintln!("Case 900: nodes per layer = {:?}, total n = {}", nodes, n);
    let (a, _b, _c, _d) = build_state_space_matrices(&layers, &nodes, n);

    // Schur
    let schur = matrix_exponential(&a, 3600.0);
    // Old Pade
    let pade = matrix_exponential_old_pade(&a, 3600.0);

    // Check eigenvalues
    let _a_inv = matrix_inverse(&a).unwrap();
    eprintln!("DC gain = D - C * A_inv * B (exterior→int)");
    let u_bare = 1.0
        / layers
            .iter()
            .map(|l| l.thickness / l.conductivity)
            .sum::<f64>();
    eprintln!("U_bare = {:.6}", u_bare);

    // Sum rows and columns of exp(A*t)
    eprintln!("\nSchur exp(A*3600) col sums (should be 1 - U_bare/eigenvalue... ish):");
    for j in 0..n {
        let col_sum: f64 = (0..n).map(|i| schur[i][j]).sum();
        eprintln!("  col {}: sum = {:.6e}", j, col_sum);
    }
    eprintln!("\nPade exp(A*3600) col sums:");
    for j in 0..n {
        let col_sum: f64 = (0..n).map(|i| pade[i][j]).sum();
        eprintln!("  col {}: sum = {:.6e}", j, col_sum);
    }

    // Compare sample entries
    eprintln!("\nSample entries (i, j, schur, pade, diff):");
    for &(i, j) in &[
        (0, 0),
        (0, 5),
        (0, 12),
        (5, 0),
        (5, 5),
        (12, 0),
        (23, 23),
        (23, 0),
    ] {
        if i < n && j < n {
            eprintln!(
                "  ({},{}): schur={:.6e}, pade={:.6e}, diff={:.6e}",
                i,
                j,
                schur[i][j],
                pade[i][j],
                schur[i][j] - pade[i][j]
            );
        }
    }
}

#[test]
fn debug_schur_expm_6x6() {
    // 6x6 tridiagonal A similar to 200mm concrete
    let a = vec![
        vec![-1.0e-3, 5.0e-4, 0.0, 0.0, 0.0, 0.0],
        vec![5.0e-4, -1.0e-3, 5.0e-4, 0.0, 0.0, 0.0],
        vec![0.0, 5.0e-4, -1.0e-3, 5.0e-4, 0.0, 0.0],
        vec![0.0, 0.0, 5.0e-4, -1.0e-3, 5.0e-4, 0.0],
        vec![0.0, 0.0, 0.0, 5.0e-4, -1.0e-3, 5.0e-4],
        vec![0.0, 0.0, 0.0, 0.0, 5.0e-4, -1.0e-3],
    ];
    let r = matrix_exponential(&a, 3600.0);
    eprintln!("exp(6x6 A * 3600) col 0:");
    for i in 0..6 {
        eprintln!("  r[{}][0] = {:.6e}", i, r[i][0]);
    }
    eprintln!("exp(6x6 A * 3600) col 5:");
    for i in 0..6 {
        eprintln!("  r[{}][5] = {:.6e}", i, r[i][5]);
    }
    // All entries should be small (eigenvalues are -1e-3, exp(-3.6) ~ 0.027)
}
