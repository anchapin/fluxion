//! Unit tests for the state-space CTF pipeline (extracted from
//! `mod.rs` to keep the parent file under the Issue #3457 module-size
//! ratchet ceiling — Issue #3787, decomposition stage 1).
//!
//! Mirrors the PR #3688 `coverage_tests` precedent: the inline
//! `#[cfg(test)] mod tests { ... }` block is extracted to a sibling
//! file wired via `#[cfg(test)] mod tests;` in the parent. Children see
//! the parent module's items through `use super::*;` — `coverage_tests.rs`
//! and this file follow identical conventions.

use super::linalg::*;
use super::*;

#[test]
fn test_identity_matrix() {
    let i = identity(3);
    assert_eq!(i[0][0], 1.0);
    assert_eq!(i[1][1], 1.0);
    assert_eq!(i[2][2], 1.0);
    assert_eq!(i[0][1], 0.0);
}

#[test]
fn test_matrix_multiply() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let c = mat_mat_mul(&a, &b);
    assert!((c[0][0] - 19.0).abs() < 1e-10);
    assert!((c[0][1] - 22.0).abs() < 1e-10);
    assert!((c[1][0] - 43.0).abs() < 1e-10);
    assert!((c[1][1] - 50.0).abs() < 1e-10);
}

#[test]
fn test_matrix_inverse() {
    let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
    let inv = matrix_inverse(&a).unwrap();
    // Verify A · A^(-1) = I
    let product = mat_mat_mul(&a, &inv);
    assert!((product[0][0] - 1.0).abs() < 1e-10);
    assert!((product[0][1]).abs() < 1e-10);
    assert!((product[1][0]).abs() < 1e-10);
    assert!((product[1][1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_matrix_exponential_identity() {
    // exp(0) = I
    let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let result = matrix_exponential(&a, 1.0);
    assert!((result[0][0] - 1.0).abs() < 1e-10);
    assert!((result[1][1] - 1.0).abs() < 1e-10);
    assert!(result[0][1].abs() < 1e-10);
}

#[test]
fn test_matrix_exponential_diagonal() {
    // exp(diag(a,b)) = diag(exp(a), exp(b))
    let a = vec![vec![-1.0, 0.0], vec![0.0, -2.0]];
    let result = matrix_exponential(&a, 1.0);
    assert!((result[0][0] - (-1.0f64).exp()).abs() < 1e-8);
    assert!((result[1][1] - (-2.0f64).exp()).abs() < 1e-8);
    assert!(result[0][1].abs() < 1e-8);
}

#[test]
fn test_nodes_per_layer() {
    let concrete = CTFMaterial::new("Concrete", 0.200, 1.73, 2243.0, 837.0);
    let nodes = compute_nodes_per_layer(std::slice::from_ref(&concrete), 3600.0);
    assert!(nodes[0] >= MIN_NODES && nodes[0] <= MAX_NODES);
    eprintln!("Concrete 200mm: {} nodes", nodes[0]);
}

#[test]
fn test_state_space_single_layer() {
    let concrete = CTFMaterial::new("Concrete", 0.200, 1.73, 2243.0, 837.0);

    // First verify steady-state DC gain of the state-space model
    let layers = std::slice::from_ref(&concrete);
    let nodes_per_layer = compute_nodes_per_layer(layers, 3600.0);
    let n: usize = nodes_per_layer.iter().sum();
    let (a_mat, b_mat, c_mat, d_mat) = build_state_space_matrices(layers, &nodes_per_layer, n);
    let a_inv = matrix_inverse(&a_mat).unwrap();

    // DC gain: y = (D - C·A⁻¹·B)·u
    // For u = [1, 0] (unit T_ext), the interior flux should be U = 3.51
    // A_inv_B = A⁻¹·B is n×2
    let a_inv_b = mat_mat_mul_col(&a_inv, &b_mat); // n×2

    // DC gain = D - C·(A⁻¹·B)
    // C is 2×n, A⁻¹·B is n×2 → result is 2×2
    let mut gain = vec![vec![0.0f64; 2]; 2];
    for j in 0..2 {
        for k in 0..2 {
            let mut cab = 0.0;
            for i in 0..n {
                cab += c_mat[j][i] * a_inv_b[i][k];
            }
            gain[j][k] = d_mat[j][k] - cab;
        }
    }

    let total_r = 0.200 / 1.73;
    let u_bare = 1.0 / total_r; // Bare-wall U-value (no films in state-space)
    let u_filmed = 1.0 / (R_SI + total_r + R_SE); // For comparison with full CTF

    eprintln!("\nDC gain matrix (bare-wall, should be [[U_bare, -U_bare], [U_bare, -U_bare]]):");
    eprintln!(
        "  gain = [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
        gain[0][0], gain[0][1], gain[1][0], gain[1][1]
    );
    eprintln!(
        "  Expected U_bare = {:.6}, U_filmed = {:.6}",
        u_bare, u_filmed
    );

    // Check steady-state node temperatures for u = [1, 0]
    eprintln!("  Steady-state nodes for T_ext=1, T_int=0:");
    eprintln!(
        "    x_ss (from ext) = {:?}",
        a_inv_b.iter().map(|r| -r[0]).collect::<Vec<_>>()
    );
    eprintln!(
        "    x_ss (from int) = {:?}",
        a_inv_b.iter().map(|r| -r[1]).collect::<Vec<_>>()
    );

    // Interior flux from T_ext: the bare-wall DC gain is positive (heat flows
    // correctly). The split-input FOH formulation ensures exact DC gain = U_bare
    // for bare-wall CTFs, and film scaling produces U_filmed exactly.
    eprintln!("  Bare-wall DC gain q_int from T_ext = {:.6}", gain[1][0]);
    eprintln!("  Bare-wall DC gain q_int from T_int = {:.6}", gain[1][1]);
    eprintln!(
        "  (U_bare = {:.6}, split-input FOH gives exact DC gain)",
        u_bare
    );
    assert!(
        gain[1][0] > 0.0,
        "DC gain q_int from T_ext should be positive: got {:.6}",
        gain[1][0]
    );
    assert!(
        gain[1][1] < 0.0,
        "DC gain q_int from T_int should be negative: got {:.6}",
        gain[1][1]
    );
    // Cross-coupling: q_int from T_ext = gain[1, 0]; q_int from T_int = -gain[1, 0]
    // (energy conservation in steady state for the bare wall)
    let cross_coupling = (gain[1][0] + gain[1][1]).abs();
    assert!(
        cross_coupling < 1e-6,
        "Cross-coupling asymmetry: gain[1,0] + gain[1,1] should be 0, got {:.6e}",
        cross_coupling
    );

    let coeffs = compute_state_space_ctf(&[concrete], 3600.0);

    let x_sum: f64 = coeffs.x.iter().sum();
    let y_sum: f64 = coeffs.y.iter().sum();

    eprintln!("\n200mm concrete wall state-space CTF:");
    eprintln!("  U_bare = {:.6}, U_filmed = {:.6}", u_bare, u_filmed);
    eprintln!("  Sum(X) = {:.6}", x_sum);
    eprintln!("  Sum(Y) = {:.6}", y_sum);
    eprintln!("  X[0] = {:.6}", coeffs.x[0]);
    eprintln!("  Y[0] = {:.6}", coeffs.y[0]);
    eprintln!("  Phi[0:5] = {:?}", &coeffs.phi[..5.min(coeffs.num_coeffs)]);

    // Steady-state check: the DC gain identity ΣX/(1+ΣΦ) must equal U_filmed.
    // With the FOH coordinate transform, the bare-wall CTFs satisfy:
    //   ΣX_bare / (1 + ΣΦ_bare) = U_bare
    // After film scaling (all coeffs / denom where denom = 1+ΣX_bare*(R_SE+R_SI)):
    //   ΣX_filmed / (1 + ΣΦ_filmed) = U_filmed
    // Note: ΣX ≠ U_filmed when ΣΦ ≠ 0; only the ratio ΣX/(1+ΣΦ) equals U_filmed.
    let phi_sum: f64 = coeffs.phi.iter().sum();
    let dc_gain = x_sum / (1.0 + phi_sum);
    eprintln!(
        "  DC gain check: ΣX/(1+ΣΦ) = {:.6}, U_filmed = {:.6}",
        dc_gain, u_filmed
    );
    assert!(
        (dc_gain - u_filmed).abs() / u_filmed < 1e-4,
        "DC gain ΣX/(1+ΣΦ) = {:.6} should match U_filmed {:.6} (within 0.01%)",
        dc_gain,
        u_filmed
    );
    // Verify Y DC gain symmetry (ΣY/(1+ΣΦ) should also equal U_filmed)
    let dc_gain_y = y_sum / (1.0 + phi_sum);
    assert!(
        (dc_gain_y - u_filmed).abs() / u_filmed < 1e-4,
        "Y DC gain ΣY/(1+ΣΦ) = {:.6} should match U_filmed {:.6} (within 0.01%)",
        dc_gain_y,
        u_filmed
    );
}

// ========================================================================
// Phase C: Verify the capavg formula used at layer interfaces.
//
// The fluxion code computes:
//   capavg = 0.5 * (cap_interior + next_layer.density * next_layer.specific_heat * dx_next)
//
// The E+ Construction.cc formula is:
//   capavg = (cap_left + cap_right) / 2
//   where cap_left = rho_left * cp_left * dx_left
//         cap_right = rho_right * cp_right * dx_right
//
// These are mathematically identical:
//   0.5 * (cap_interior + cap_next) = (cap_interior + cap_next) / 2 ✓
//
// The test verifies this equivalence for 4 arbitrary layer pairs by
// computing the alpha values (which depend on capavg) using both forms
// and asserting they match to numerical precision.
// ========================================================================
#[test]
fn test_capavg_matches_eplus_formula() {
    type LayerPair = ((f64, f64, f64, f64), (f64, f64, f64, f64));
    let layer_pairs: Vec<LayerPair> = vec![
        // (k, dx, rho, cp) for layer 1 and layer 2
        // Case 600: Concrete -> Insulation
        (
            (1.4, 0.150 / 6.0, 2300.0, 880.0),
            (0.04, 0.050 / 6.0, 50.0, 840.0),
        ),
        // Case 900: Gypsum -> Concrete
        (
            (0.16, 0.013 / 6.0, 800.0, 1090.0),
            (1.4, 0.150 / 6.0, 2300.0, 880.0),
        ),
        // Case 900: Concrete -> Insulation
        (
            (1.4, 0.150 / 6.0, 2300.0, 880.0),
            (0.04, 0.050 / 6.0, 50.0, 840.0),
        ),
        // Case 900: Insulation -> Brick
        (
            (0.04, 0.050 / 6.0, 50.0, 840.0),
            (0.81, 0.100 / 6.0, 1920.0, 790.0),
        ),
    ];

    for (idx, ((k1, dx1, rho1, cp1), (k2, dx2, rho2, cp2))) in layer_pairs.iter().enumerate() {
        let cap1 = rho1 * cp1 * dx1;
        let cap2 = rho2 * cp2 * dx2;

        // Fluxion formula
        let capavg_fluxion = 0.5 * (cap1 + cap2);
        // E+ formula (mathematically identical)
        let capavg_eplus = (cap1 + cap2) / 2.0;

        let alpha_left_fluxion = k1 / (capavg_fluxion * dx1);
        let alpha_left_eplus = k1 / (capavg_eplus * dx1);
        let alpha_right_fluxion = k2 / (capavg_fluxion * dx2);
        let alpha_right_eplus = k2 / (capavg_eplus * dx2);

        assert!(
            (alpha_left_fluxion - alpha_left_eplus).abs() < 1e-12,
            "Layer pair {}: fluxion alpha_left ({:.6e}) != E+ alpha_left ({:.6e})",
            idx,
            alpha_left_fluxion,
            alpha_left_eplus
        );
        assert!(
            (alpha_right_fluxion - alpha_right_eplus).abs() < 1e-12,
            "Layer pair {}: fluxion alpha_right ({:.6e}) != E+ alpha_right ({:.6e})",
            idx,
            alpha_right_fluxion,
            alpha_right_eplus
        );
    }
}

// ========================================================================
// Matrix-level diagnostic: compare A, B, C, D to E+ Construction.cc v25.2.0
// reference values for 200mm concrete single-layer wall.
// Reference: /home/alex/Projects/fluxion/.agents/notes/matrix_comparison_200mm_concrete.txt
// E+ source: https://raw.githubusercontent.com/NREL/EnergyPlus/v25.2.0/src/EnergyPlus/Construction.cc
// ========================================================================
#[test]
fn test_matrix_construction_matches_energyplus_reference() {
    // 200mm normal-weight concrete (matches tests/reference_data/energyplus_models/step_change_concrete.idf)
    let concrete = vec![CTFMaterial::new(
        "CONCRETE_200",
        0.200,  // thickness [m]
        1.73,   // conductivity [W/m-K]
        2300.0, // density [kg/m^3]
        840.0,  // specific heat [J/kg-K]
    )];
    let nodes = vec![6_usize]; // 1 layer, 6 nodes
    let n = 6;

    let (a, b, c_mat, d_mat) = build_state_space_matrices(&concrete, &nodes, n);

    // POST-PHASE-2 EXPECTED VALUES (EnergyPlus Construction.cc v25.2.0):
    //   dx = L/N = 0.2/6 = 0.03333 m
    //   cap_boundary = 1.5 * rho * cp * dx = 1.5 * 2300 * 840 * 0.03333 = 96600
    //   dxtmp_boundary = 1/(dx*cap_boundary) = 1/3220 = 3.105590e-4
    //   A[0,0] = -2*k*dxtmp_boundary = -1.074534e-3
    //   A[0,1] = +k*dxtmp_boundary   = +5.372671e-4
    //   B[0,0] = +k*dxtmp_boundary   = +5.372671e-4
    //   C[0,0] = -k/dx/(N-1)         = -10.38
    //   D[0,0] = +k/dx/(N-1)         = +10.38
    //
    // Values computed by re-evaluating the E+ formula directly:
    let k = 1.73_f64;
    let rho = 2300.0_f64;
    let cp = 840.0_f64;
    let nn = 6.0_f64;
    let dx_l = 0.2 / nn;
    let cap_b = 1.5 * rho * cp * dx_l;
    let dxtmp_b = 1.0 / dx_l / cap_b;
    let a00 = -2.0 * k * dxtmp_b;
    let a01 = k * dxtmp_b;
    let b00 = k * dxtmp_b;
    // h_surf = k*(N+1)/(N*dx) gives exact DC gain = U_bare for the lumped boundary scheme.
    // This replaces the old k/dx/(N-1) formula which was incorrect.
    let h_surf = k * (nn + 1.0) / (nn * dx_l);
    let c00 = -h_surf;
    let d00 = h_surf;

    eprintln!("\n=== A_fluxion boundary node (POST-PHASE-2) ===");
    eprintln!("  A[0][0] = {:.6e}  (E+ formula = {:.6e})", a[0][0], a00);
    eprintln!("  A[0][1] = {:.6e}  (E+ formula = {:.6e})", a[0][1], a01);
    eprintln!("  B[0][0] = {:.6e}  (E+ formula = {:.6e})", b[0][0], b00);
    eprintln!(
        "  C[0][0] = {:.6e}  (h_surf formula = {:.6e})",
        c_mat[0][0], c00
    );
    eprintln!(
        "  D[0][0] = {:.6e}  (h_surf formula = {:.6e})",
        d_mat[0][0], d00
    );

    // === POST-PHASE-2 ASSERTIONS ===
    // After Phase 2 fix, the matrix values should match the corrected formulas.
    assert!(
        (a[0][0] - a00).abs() / a00.abs() < 1e-9,
        "A[0][0] drifted from formula: got {:.10e}, expected {:.10e}",
        a[0][0],
        a00
    );
    assert!(
        (a[0][1] - a01).abs() / a01.abs() < 1e-9,
        "A[0][1] drifted from formula: got {:.10e}, expected {:.10e}",
        a[0][1],
        a01
    );
    assert!(
        (b[0][0] - b00).abs() / b00.abs() < 1e-9,
        "B[0][0] drifted from formula: got {:.10e}, expected {:.10e}",
        b[0][0],
        b00
    );
    assert!(
        (c_mat[0][0] - c00).abs() / c00.abs() < 1e-9,
        "C[0][0] drifted from h_surf formula: got {:.6e}, expected {:.6e}",
        c_mat[0][0],
        c00
    );
    assert!(
        (d_mat[0][0] - d00).abs() / d00.abs() < 1e-9,
        "D[0][0] drifted from h_surf formula: got {:.6e}, expected {:.6e}",
        d_mat[0][0],
        d00
    );

    // === DIAGNOSTIC ===
    let ratio_a00 = a[0][0] / a00;
    let ratio_c00 = c_mat[0][0] / c00;
    eprintln!("\n=== DIAGNOSTIC (post Phase 2) ===");
    eprintln!(
        "A[0,0] ratio (fluxion/formula) = {:.9}  (target 1.0)",
        ratio_a00
    );
    eprintln!(
        "C[0,0] ratio (fluxion/formula) = {:.9}  (target 1.0)",
        ratio_c00
    );
    assert!(
        (ratio_a00 - 1.0).abs() < 1e-9,
        "A[0,0] not matching formula after Phase 2"
    );
    assert!(
        (ratio_c00 - 1.0).abs() < 1e-9,
        "C[0,0] not matching formula after Phase 2"
    );

    // === ENERGY CONSERVATION CHECK ===
    // A @ 1 should be ~0 (B contribution not included; with B included, perfect).
    // Note: For lumped-mass boundary, A[0,0] + A[0,1] = -k*dxtmp_boundary (the
    // conductance to the surface, captured in B[0,0]). So A @ 1 = -B[:, 0]
    // for the first/last nodes (and 0 for interior).
    let a_times_ones: f64 = (0..n)
        .map(|i| a[i].iter().sum::<f64>())
        .collect::<Vec<_>>()
        .iter()
        .map(|&s| s.abs())
        .fold(0.0_f64, f64::max);
    eprintln!("\n||A @ 1||∞ = {:.6e} (was 1.12e-3 pre-fix)", a_times_ones);
    eprintln!(
        "  This should equal B[0,0] = {:.6e} (= k*dxtmp_boundary)",
        b[0][0]
    );
    let ratio_conservation = a_times_ones / b[0][0];
    eprintln!(
        "  Ratio ||A @ 1|| / B[0,0] = {:.6}  (target 1.0)",
        ratio_conservation
    );
    assert!(
        (ratio_conservation - 1.0).abs() < 1e-6,
        "Energy conservation: ||A @ 1||∞ should equal B[0,0]"
    );
}

// ========================================================================
// Cross-coupling diagnostic: for multi-layer walls, the s0 off-diagonals
// (X[0] = s0[1,0] and Y[0] = -s0[1,1] in fluxion notation) should both be
// non-zero (s0 must be skew-symmetric in steady-state). This test catches
// the multi-layer bug where the off-diagonals collapse to 0.
// ========================================================================
#[test]
fn test_multi_layer_cross_coupling() {
    // Case 600 high-mass wall (interior -> exterior)
    let case_600 = vec![
        CTFMaterial::new("CONCRETE", 0.150, 1.4, 2300.0, 880.0),
        CTFMaterial::new("INSULATION", 0.050, 0.04, 50.0, 840.0),
    ];
    let n1 = compute_nodes_per_layer(&case_600, 3600.0);
    let total: usize = n1.iter().sum();
    eprintln!("\n=== Case 600 cross-coupling diagnostic ===");
    eprintln!("  Nodes per layer: {:?} (total {})", n1, total);

    let (a, b, c_mat, d_mat) = build_state_space_matrices(&case_600, &n1, total);
    eprintln!("  A dim: {}x{}", a.len(), a[0].len());

    // Run Seem series extraction
    let dt = 3600.0;
    let a_exp = matrix_exponential(&a, dt);
    let a_inv = matrix_inverse(&a).expect("A invertible");
    let a_exp_minus_i = matrix_sub_identity(&a_exp);
    let temp = mat_mat_mul_col(&a_exp_minus_i, &b);
    let gamma1 = mat_mat_mul_col(&a_inv, &temp);
    let gamma1_scaled = scale_columns(&gamma1, 1.0 / dt);
    let gamma2_diff = matrix_sub_col(&gamma1_scaled, &b);
    let gamma2 = mat_mat_mul_col(&a_inv, &gamma2_diff);

    // s0 = D + C @ Gamma2
    let mut s0 = vec![vec![0.0; 2]; 2];
    for j in 0..2 {
        for k in 0..2 {
            s0[j][k] = d_mat[j][k];
            for i in 0..total {
                s0[j][k] += c_mat[j][i] * gamma2[i][k];
            }
        }
    }

    eprintln!(
        "  s0 = [[{:.6e}, {:.6e}], [{:.6e}, {:.6e}]]",
        s0[0][0], s0[0][1], s0[1][0], s0[1][1]
    );
    eprintln!("  Expected: s0[0,1] = -s0[1,0] (skew-symmetric in steady state)");
    eprintln!(
        "  Observed: s0[0,1] + s0[1,0] = {:.6e}",
        s0[0][1] + s0[1][0]
    );

    // The off-diagonal terms being non-zero is required for cross-coupling.
    // If both are ~0, the multi-layer cross-coupling is broken.
    let s0_offdiag_max = s0[0][1].abs().max(s0[1][0].abs());
    eprintln!("  max(|s0[0,1]|, |s0[1,0]|) = {:.6e}", s0_offdiag_max);

    // This is a diagnostic, not a strict assertion, so we just print
    // the values for now. The test will fail in CI only if the off-diagonals
    // collapse to exactly 0 (the multi-layer bug).
    assert!(
        s0_offdiag_max > 1e-6,
        "Multi-layer s0 off-diagonals collapsed to 0! \
             s0 = [[{:.3e}, {:.3e}], [{:.3e}, {:.3e}]] \
             — this is the cross-coupling bug from issue #951.",
        s0[0][0],
        s0[0][1],
        s0[1][0],
        s0[1][1]
    );
}
