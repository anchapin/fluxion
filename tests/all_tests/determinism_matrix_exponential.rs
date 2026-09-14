//! Cross-Platform Determinism Test for `matrix_exponential_faer` (Issue #2549).
//!
//! # Background
//! The determinism CI matrix runs on x86_64 (Linux/Windows) and ARM64
//! (Apple-Silicon macOS). Even with LLVM's unsafe fp-math disabled
//! (see `.github/workflows/determinism_check.yml`), strict byte-for-byte
//! equality of floating-point matrix output across x86_64 and ARM64 is
//! *not physically achievable*:
//!
//!   - FMA (fused-multiply-add) instructions differ: x86_64 has vfmadd231pd
//!     (3-input, single-rounded), Apple Silicon has fmadd (ARMv8.0-A),
//!     Windows/MSVC may contract differently under `#[cfg(target_feature)]`.
//!   - Vector widths differ (AVX2 256-bit vs NEON 128-bit), changing which
//!     intermediates LLVM keeps in extended precision during horizontal sums.
//!   - Transcendental approximations (used by Padé squaring) round
//!     differently per backend.
//!
//! ## Contract under test
//! This test fixes a deterministic input, computes `matrix_exponential_faer`,
//! and asserts the result is (a) finite and (b) equal to a Python
//! `scipy.linalg.expm`-computed reference within a tight tolerance:
//!
//!   - **absolute tolerance: 1e-12** (well above ULP-of-f64 at this magnitude
//!     which is ~1e-16, but tight enough to catch any real algorithmic or
//!     codegen regression — e.g. wrong Padé coefficient, FMA contraction
//!     that loses a bit, or a reordered reduction).
//!   - **relative tolerance: 1e-10** (allows for the ~1e-13 level
//!     discrepancy expected from FMA/SIMD differences across ISAs while
//!     still rejecting any meaningful divergence).
//!
//! This per-arch test runs on every matrix row of the determinism CI job,
//! so each platform independently verifies its output against the same
//! reference. That catches per-arch divergence without requiring a fragile
//! cross-OS artifact-comparison step.
//!
//! # Reference computation
//! Reference values were computed once via Python (RULES.md mandate:
//! "always write and execute Python code for calculations") using
//! `scipy.linalg.expm(A @ t)` with the same Higham scaling-and-squaring
//! Padé[13/13] algorithm that `matrix_exponential_faer` implements.

use fluxion::physics::state_space_ctf::matrix_exponential_faer;

/// Absolute tolerance: catches any algorithmic or codegen regression while
/// accommodating ~1e-13-level ULP noise from cross-ISA FMA contraction.
const ABS_TOL: f64 = 1e-12;
/// Relative tolerance: per-element relative slack for cross-ISA agreement.
const REL_TOL: f64 = 1e-10;

/// Deterministic 4×4 input matrix in per-timestep units (eigenvalues all
/// in (-1, 0), well-separated — no clustered-eigenvalue pathologies that
/// would stress the Padé squaring factor).
const INPUT_4X4: [[f64; 4]; 4] = [
    [-0.5, 0.1, 0.0, 0.0],
    [0.1, -0.3, 0.05, 0.0],
    [0.0, 0.05, -0.2, 0.02],
    [0.0, 0.0, 0.02, -0.15],
];

const T: f64 = 1.0;

/// Reference output computed via `scipy.linalg.expm(INPUT_4X4 * T)` on
/// 2026-08-10 (Python 3.11, scipy 1.17.1, NumPy 2.2). Padé[13/13]
/// scaling-and-squaring — same algorithm as `matrix_exponential_faer`.
/// Literals are Python `repr()` (shortest round-tripping form for f64) to
/// satisfy clippy::excessive_precision.
const REFERENCE_4X4: [[f64; 4]; 4] = [
    [
        0.609779485952006,
        0.06728584285657727,
        0.0017966979368028635,
        1.253241480417585e-05,
    ],
    [
        0.06728584285657727,
        0.745249520633562,
        0.039035521721658056,
        0.00040320303917518825,
    ],
    [
        0.0017966979368028637,
        0.03903552172165805,
        0.8198884494189423,
        0.016798752976580435,
    ],
    [
        1.2532414804175852e-05,
        0.00040320303917518825,
        0.016798752976580438,
        0.8608773242624554,
    ],
];

#[test]
fn matrix_exponential_faer_matches_reference() {
    let a: Vec<Vec<f64>> = INPUT_4X4.iter().map(|r| r.to_vec()).collect();
    let out = matrix_exponential_faer(&a, T);

    assert_eq!(out.len(), 4, "expected 4 rows");
    let mut worst_abs = 0.0f64;
    let mut worst_rel = 0.0f64;
    let mut worst_pos = String::new();
    for (i, row) in out.iter().enumerate() {
        assert_eq!(row.len(), 4, "expected 4 cols in row {i}");
        for (j, &got) in row.iter().enumerate() {
            assert!(got.is_finite(), "non-finite output at [{i}][{j}]: {got}");
            let want = REFERENCE_4X4[i][j];
            let abs = (got - want).abs();
            let rel = if want.abs() > f64::MIN_POSITIVE {
                abs / want.abs()
            } else {
                0.0
            };
            if abs > worst_abs || rel > worst_rel {
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                worst_pos = format!("[{i}][{j}]");
            }
            assert!(
                abs <= ABS_TOL || rel <= REL_TOL,
                "[{i}][{j}] diverged: got={got:.18e} want={want:.18e} abs={abs:.3e} rel={rel:.3e} \
                 (tolerances abs={ABS_TOL:.0e} rel={REL_TOL:.0e})"
            );
        }
    }

    println!("matrix_exponential_faer 4x4 OK — worst abs {worst_abs:.3e} rel {worst_rel:.3e} at {worst_pos}");
}

#[test]
fn matrix_exponential_faer_output_is_finite() {
    // Stress test: a non-symmetric matrix with off-diagonal coupling that
    // exercises FMA opportunities. We only assert finiteness + a coarse
    // sanity bound (det > 0) — this guards against NaN/Inf regressions
    // introduced by codegen changes without coupling to a fixed reference.
    let a: Vec<Vec<f64>> = vec![
        vec![-0.2, 0.3, 0.0, 0.0],
        vec![0.05, -0.4, 0.1, 0.0],
        vec![0.0, 0.1, -0.3, 0.05],
        vec![0.0, 0.0, 0.2, -0.25],
    ];
    let out = matrix_exponential_faer(&a, 1.0);
    assert_eq!(out.len(), 4);
    let mut all_finite = true;
    let mut det = 1.0f64;
    for (i, row) in out.iter().enumerate() {
        assert_eq!(row.len(), 4);
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                all_finite = false;
            }
            if i == j {
                det *= v;
            }
        }
    }
    assert!(all_finite, "expm output contained non-finite values");
    assert!(
        det > 0.0 && det < 1.0,
        "expm diagonal-product sanity check failed: det={det:.6e}"
    );
}
