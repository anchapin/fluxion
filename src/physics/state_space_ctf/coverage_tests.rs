//! Coverage tests for the parent module (extracted to keep the
//! ratcheted parent file under its Issue #2878/#3574 module-size
//! ceiling; child modules can see parent-private items).
//!
//! Coverage-expansion tests (PR #5): `FlatMatrix` accessors, the public
//! `matrix_exponential_faer` entry point, the lightweight-wall fallback
//! branch of `compute_state_space_ctf`, and filmed-CTF known-answer /
//! reciprocity properties that the existing debug modules do not pin down.

use super::*;
use crate::physics::ctf_coefficients::CTFMaterial;

#[test]
fn flat_matrix_new_is_zeroed_and_reports_shape() {
    let m = FlatMatrix::new(2, 3, 4);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    // The backing store is rows × stride, including padding slots.
    assert_eq!(m.as_slice().len(), 8);
    assert!(m.as_slice().iter().all(|&v| v == 0.0));
}

#[test]
fn flat_matrix_get_set_honours_stride_padding() {
    let mut m = FlatMatrix::new(2, 2, 3);
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);
    m.set(1, 1, 4.0);
    assert!((m.get(0, 0) - 1.0).abs() < 1e-12);
    assert!((m.get(0, 1) - 2.0).abs() < 1e-12);
    assert!((m.get(1, 0) - 3.0).abs() < 1e-12);
    assert!((m.get(1, 1) - 4.0).abs() < 1e-12);
    // Padding slots (stride 3 > cols 2) stay untouched by set().
    assert_eq!(m.as_slice(), &[1.0, 2.0, 0.0, 3.0, 4.0, 0.0]);
}

#[test]
fn flat_matrix_identity_is_diagonal() {
    let m = FlatMatrix::identity(3);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((m.get(i, j) - want).abs() < 1e-12);
        }
    }
}

#[test]
fn flat_matrix_zeros_is_all_zero() {
    let m = FlatMatrix::zeros(3, 2);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 2);
    assert!(m.as_slice().iter().all(|&v| v == 0.0));
}

#[test]
fn flat_matrix_from_vec_vec_round_trip() {
    let original = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let m = FlatMatrix::from_vec_vec(&original);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    assert_eq!(m.to_vec_vec(), original);
    assert_eq!(m.as_ref_vec_vec(), original);
}

#[test]
fn flat_matrix_fill_writes_entire_buffer() {
    let mut m = FlatMatrix::new(2, 2, 3);
    m.fill(7.5);
    // fill() covers the whole backing buffer, padding included.
    assert_eq!(m.as_slice(), &[7.5; 6]);
    assert!((m.get(1, 1) - 7.5).abs() < 1e-12);
}

#[test]
fn flat_matrix_mut_slice_allows_direct_edit() {
    let mut m = FlatMatrix::identity(2);
    m.as_mut_slice()[1] = 9.0;
    assert!((m.get(0, 1) - 9.0).abs() < 1e-12);
}

// ---------- matrix_exponential_faer ----------

#[test]
fn faer_expm_empty_and_scalar() {
    let empty: Vec<Vec<f64>> = matrix_exponential_faer(&[], 1.0);
    assert!(empty.is_empty());
    let scalar = matrix_exponential_faer(&[vec![-0.5]], 2.0);
    assert!((scalar[0][0] - (-1.0f64).exp()).abs() < 1e-14);
}

#[test]
fn faer_expm_rotation_generator_known_answer() {
    // exp([[0,-w],[w,0]]) = [[cos w, -sin w],[sin w, cos w]]
    // (Moler & Van Loan 2×2 complex-eigenvalue branch).
    let w = 0.7f64;
    let a = vec![vec![0.0, -w], vec![w, 0.0]];
    let r = matrix_exponential_faer(&a, 1.0);
    let (c, s) = (w.cos(), w.sin());
    assert!((r[0][0] - c).abs() < 1e-12, "r[0][0]={}", r[0][0]);
    assert!((r[0][1] + s).abs() < 1e-12, "r[0][1]={}", r[0][1]);
    assert!((r[1][0] - s).abs() < 1e-12, "r[1][0]={}", r[1][0]);
    assert!((r[1][1] - c).abs() < 1e-12, "r[1][1]={}", r[1][1]);
}

#[test]
fn faer_expm_diagonal_known_answer() {
    let a = vec![vec![-1.5, 0.0], vec![0.0, 0.25]];
    let r = matrix_exponential_faer(&a, 2.0);
    assert!((r[0][0] - (-3.0f64).exp()).abs() < 1e-12);
    assert!((r[1][1] - 0.5f64.exp()).abs() < 1e-12);
    assert!(r[0][1].abs() < 1e-14);
    assert!(r[1][0].abs() < 1e-14);
}

#[test]
fn faer_expm_block_diagonal_known_answer() {
    // Exercises the n>=3 Padé path against analytic truth: a block
    // diagonal matrix exponentiates block-wise.
    //   A = diag(R(w), -λ), R(w) = [[0,-w],[w,0]]
    //   exp(A·t) = diag([[cos wt, -sin wt],[sin wt, cos wt]], e^{-λt})
    let w = 0.5f64;
    let lambda = 1.25f64;
    let t = 2.0f64;
    let a = vec![
        vec![0.0, -w, 0.0],
        vec![w, 0.0, 0.0],
        vec![0.0, 0.0, -lambda],
    ];
    let r = matrix_exponential_faer(&a, t);
    let (c, s_) = ((w * t).cos(), (w * t).sin());
    let e = (-lambda * t).exp();
    let expected = vec![vec![c, -s_, 0.0], vec![s_, c, 0.0], vec![0.0, 0.0, e]];
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (r[i][j] - expected[i][j]).abs() < 1e-10,
                "expm mismatch at [{i}][{j}]: got {} want {}",
                r[i][j],
                expected[i][j]
            );
        }
    }
}

// ---------- compute_state_space_ctf known answers ----------

/// A single thin, low-density layer: Fo = α·Δt/L² ≫ 2.5, so the
/// all-lightweight branch fires and the coefficients collapse to the
/// filmed U-value with no dynamics.
fn lightweight_layer() -> CTFMaterial {
    CTFMaterial::new("LightMembrane", 0.010, 0.04, 20.0, 1000.0)
}

#[test]
fn ctf_lightweight_fallback_returns_filmed_u() {
    let layer = lightweight_layer();
    let alpha = layer.diffusivity();
    let fo = alpha * 3600.0 / (layer.thickness * layer.thickness);
    assert!(fo > 2.5, "fixture must be lightweight, Fo={fo}");

    let layers = std::slice::from_ref(&layer);
    let nodes = compute_nodes_per_layer(layers, 3600.0);
    assert_eq!(nodes, vec![0], "lightweight layer must need zero nodes");

    let coeffs = compute_state_space_ctf(layers, 3600.0);
    let r_wall = layer.resistance();
    let u_filmed = 1.0 / (R_SI + r_wall + R_SE);

    assert_eq!(coeffs.num_coeffs, 1);
    assert_eq!(coeffs.total_state_nodes, 0);
    assert!(
        (coeffs.x[0] - u_filmed).abs() < 1e-12,
        "x[0]={}",
        coeffs.x[0]
    );
    assert!(
        (coeffs.y[0] - u_filmed).abs() < 1e-12,
        "y[0]={}",
        coeffs.y[0]
    );
    assert!(
        (coeffs.z[0] - u_filmed).abs() < 1e-12,
        "z[0]={}",
        coeffs.z[0]
    );
    assert!(coeffs.phi[0].abs() < 1e-15);
    assert!((coeffs.u_value() - u_filmed).abs() < 1e-12);
}

#[test]
fn ctf_xy_sums_agree_and_z_is_negated_y() {
    // Documented steady-state invariant (see the coefficient-mapping
    // comment in `compute_ctf_from_state_space`): at steady state
    //   q = ΣX·T_ext − ΣY·T_int = U·(T_ext − T_int),
    // so ΣX and ΣY must agree. Z is stored "as-is for reference" as the
    // negation of Y (Z[j] = −Y[j] term-by-term).
    let layers = vec![CTFMaterial::new("Concrete", 0.200, 1.73, 2243.0, 837.0)];
    let coeffs = compute_state_space_ctf(&layers, 3600.0);
    assert!(coeffs.num_coeffs > 1);

    let sum_x: f64 = coeffs.x.iter().sum();
    let sum_y: f64 = coeffs.y.iter().sum();
    let scale = sum_x.abs().max(sum_y.abs()).max(1e-12);
    assert!(
        (sum_x - sum_y).abs() < 1e-9 * scale,
        "ΣX={sum_x} vs ΣY={sum_y}"
    );

    let y_scale = coeffs
        .y
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1e-12);
    for j in 0..coeffs.num_coeffs {
        assert!(
            (coeffs.z[j] + coeffs.y[j]).abs() < 1e-12 * y_scale,
            "Z[j] must be −Y[j] at j={j}: z={} y={}",
            coeffs.z[j],
            coeffs.y[j]
        );
    }

    // The agreed sum is exactly the DC-gain numerator: ΣX/(1+ΣΦ).
    let sum_phi: f64 = coeffs.phi.iter().sum();
    assert!((sum_x / (1.0 + sum_phi) - coeffs.u_value()).abs() < 1e-12);
}

#[test]
fn ctf_filmed_u_value_matches_closed_form() {
    // Film scaling is constructed so ΣX/(1+ΣΦ) == U_filmed exactly;
    // pin the closed form U = 1/(R_SI + R_wall + R_SE).
    let layers = vec![CTFMaterial::new("Concrete", 0.200, 1.73, 2243.0, 837.0)];
    let coeffs = compute_state_space_ctf(&layers, 3600.0);
    let r_wall = 0.200 / 1.73;
    let u_expected = 1.0 / (R_SI + r_wall + R_SE);
    let u = coeffs.u_value();
    assert!(
        (u - u_expected).abs() / u_expected < 1e-4,
        "u_value={u} vs closed-form U={u_expected}"
    );
}

#[test]
fn ctf_interior_flux_converges_to_steady_state_answer() {
    // Dynamic known-answer: hold T_ext/T_int constant and iterate the
    // ASHRAE recurrence. The flux must settle to U·(T_ext − T_int)
    // (positive = into the zone; here negative = heat loss).
    let layers = vec![CTFMaterial::new("Concrete", 0.200, 1.73, 2243.0, 837.0)];
    let coeffs = compute_state_space_ctf(&layers, 3600.0);
    let n = coeffs.num_coeffs;

    let t_ext = 10.0;
    let t_int = 20.0;
    let t_ext_hist = vec![t_ext; n];
    let t_int_hist = vec![t_int; n.saturating_sub(1).max(1)];
    let mut q_hist = vec![0.0; n.saturating_sub(1).max(1)];

    let mut q = 0.0;
    for _ in 0..400 {
        q = coeffs.calculate_interior_flux(t_int, &t_ext_hist, &t_int_hist, &q_hist);
        assert!(q.is_finite(), "non-finite flux during transient");
        q_hist.insert(0, q);
        q_hist.pop();
    }

    let u = coeffs.u_value();
    let q_expected = u * (t_ext - t_int); // = −10·U
    assert!(
        (q - q_expected).abs() < 0.02 * q_expected.abs().max(1e-9),
        "steady flux q={q} vs U·ΔT={q_expected}"
    );
}

#[test]
fn ctf_two_layer_canonical_assembly_is_finite_convergent_and_accurate() {
    // Deterministic counterpart to the proptest sweeps: a canonical
    // concrete + insulation wall must have finite coefficients,
    // a stable Φ series, and a DC gain close to the filmed U-value.
    let layers = vec![
        CTFMaterial::new("Concrete", 0.200, 1.73, 2243.0, 837.0),
        CTFMaterial::new("Insulation", 0.100, 0.035, 30.0, 1400.0),
    ];
    let coeffs = compute_state_space_ctf(&layers, 3600.0);

    assert!(coeffs.x.iter().all(|v| v.is_finite()));
    assert!(coeffs.y.iter().all(|v| v.is_finite()));
    assert!(coeffs.z.iter().all(|v| v.is_finite()));
    assert!(coeffs.phi.iter().all(|v| v.is_finite()));
    assert!(
        coeffs.check_convergence(0.1),
        "coefficients must decay for the canonical assembly"
    );
    let sum_phi: f64 = coeffs.phi.iter().sum();
    assert!(
        sum_phi.abs() < 1.0,
        "stability requires |ΣΦ| < 1, got {sum_phi}"
    );

    let r_wall = 0.200 / 1.73 + 0.100 / 0.035;
    let u_expected = 1.0 / (R_SI + r_wall + R_SE);
    let u = coeffs.u_value();
    assert!(
        (u - u_expected).abs() / u_expected < 0.05,
        "DC gain {u} vs filmed U {u_expected}"
    );
}
