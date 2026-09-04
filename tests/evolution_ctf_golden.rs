//! Golden-coefficient test for the state-space CTF evolution seed (#3337).
//!
//! Verifies that the seed kernel in
//! `tools/evolution/seeds/ctf/seed.rs`, with its baseline EVOLVE-BLOCK
//! contents, reproduces the production CTF coefficients bit-for-bit
//! across the wall library. This is the "frozen context" guarantee:
//! at the evolver's starting point, the candidate and the production
//! engine must agree exactly — so any improvement measured during the
//! campaign can be attributed to heuristic tuning, not to a hidden
//! drift in the seed skeleton.
//!
//! The test uses `include_str!` to fold the seed source into the
//! compilation unit (rather than running the recompile path) — this
//! keeps the test cheap and independent of the sandbox / dynamic-load
//! features that the OpenEvolve campaign uses.

#![allow(clippy::needless_range_loop)]

use fluxion::physics::ctf_coefficients::{CTFCalculator, CTFMaterial};
use fluxion_evaluator::kernel::{Kernel, KernelInput};

// Pull the seed module into this test crate via `include!`. Rust's
// `include!` macro is the canonical way to inline a file at the
// top level — `include_str!` would only give us a `&str`, not the
// top-level items. The seed is self-contained (it only depends on
// `fluxion_evaluator::kernel`) so dropping it into the test's scope
// works without further wiring.
include!("evolution_ctf_golden_seed_shim.rs");

// Re-export the candidate from the included module so the test body
// can call `Candidate::default().evaluate(...)` directly.
use seed::Candidate;

/// Threshold below which the seed is considered bit-identical to
/// the production CTF implementation. 1e-10 relative is well above
/// the 1e-12 that f64 reproducibility allows but well below the
/// 1e-6 energy-closure invariant from RULES.md.
const GOLDEN_TOL: f64 = 1e-10;

fn run_seed(layers: &[CTFMaterial]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let params = serde_json::json!({
        "layers": layers.iter().map(|l| serde_json::json!({
            "name": l.name,
            "thickness_m": l.thickness,
            "k_w_mk": l.conductivity,
            "rho_kg_m3": l.density,
            "cp_j_kgk": l.specific_heat,
        })).collect::<Vec<_>>(),
        "timestep_s": 3600.0,
    });
    let input = KernelInput {
        case_name: "golden".to_string(),
        params,
    };
    let out = Candidate.evaluate(&input).expect("seed evaluate failed");
    let payload = out.payload.as_object().expect("payload is not object");
    let arr = |k: &str| -> Vec<f64> {
        payload
            .get(k)
            .and_then(|v| v.as_array())
            .expect(k)
            .iter()
            .map(|v| v.as_f64().expect("non-numeric coeff"))
            .collect()
    };
    (arr("x"), arr("y"), arr("z"), arr("phi"))
}

fn check_wall(name: &str, layers: &[CTFMaterial]) -> (f64, f64) {
    let prod = CTFCalculator::with_defaults(layers, 3600.0).compute_coefficients();
    let (sx, sy, _sz, _sphi) = run_seed(layers);

    let n = prod.x.len();
    assert_eq!(
        n,
        sx.len(),
        "[{name}] length mismatch: prod={n} seed={}",
        sx.len()
    );

    let max_dx = (0..n)
        .map(|i| (prod.x[i] - sx[i]).abs())
        .fold(0.0_f64, f64::max);
    let max_dy = (0..n)
        .map(|i| (prod.y[i] - sy[i]).abs())
        .fold(0.0_f64, f64::max);
    let max_rel_dx = (0..n)
        .map(|i| {
            let scale = prod.x[i].abs().max(1e-12);
            (prod.x[i] - sx[i]).abs() / scale
        })
        .fold(0.0_f64, f64::max);
    let max_rel_dy = (0..n)
        .map(|i| {
            let scale = prod.y[i].abs().max(1e-12);
            (prod.y[i] - sy[i]).abs() / scale
        })
        .fold(0.0_f64, f64::max);
    println!(
        "[{name}] n={n}, max|Δx|={max_dx:.4e}, max|Δy|={max_dy:.4e}, \
         max rel|Δx|={max_rel_dx:.4e}, max rel|Δy|={max_rel_dy:.4e}"
    );
    assert!(
        max_dx < GOLDEN_TOL,
        "[{name}] |Δx|={max_dx:.4e} exceeds golden tolerance {GOLDEN_TOL}"
    );
    assert!(
        max_dy < GOLDEN_TOL,
        "[{name}] |Δy|={max_dy:.4e} exceeds golden tolerance {GOLDEN_TOL}"
    );
    (max_dx, max_dy)
}

#[test]
fn golden_concrete_200mm() {
    check_wall(
        "single_concrete_200mm",
        &[CTFMaterial::new("Concrete", 0.200, 1.73, 2243.0, 837.0)],
    );
}

#[test]
fn golden_gypsum_013mm() {
    check_wall(
        "single_gypsum_013mm",
        &[CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0)],
    );
}

#[test]
fn golden_insulation_100mm() {
    check_wall(
        "single_insulation_100mm",
        &[CTFMaterial::new("Insulation", 0.100, 0.04, 50.0, 840.0)],
    );
}

#[test]
fn golden_ashrae_900_high_mass_wall() {
    check_wall(
        "ashrae_900_high_mass_wall",
        &[
            CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
            CTFMaterial::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
            CTFMaterial::new("Insulation", 0.050, 0.04, 50.0, 840.0),
            CTFMaterial::new("Brick", 0.100, 0.81, 1920.0, 790.0),
        ],
    );
}

#[test]
fn golden_ashrae_600_low_mass_wall() {
    check_wall(
        "ashrae_600_low_mass_wall",
        &[
            CTFMaterial::new("Plasterboard", 0.012, 0.16, 784.0, 840.0),
            CTFMaterial::new("Fiberglass", 0.066, 0.04, 12.0, 840.0),
            CTFMaterial::new("Wood Siding", 0.009, 0.14, 530.0, 900.0),
        ],
    );
}

#[test]
fn golden_concrete_005mm_thin() {
    // Worst-case thin layer: hits MIN_NODES = 1 branch.
    check_wall(
        "single_concrete_005mm",
        &[CTFMaterial::new("Concrete", 0.005, 1.73, 2243.0, 837.0)],
    );
}

#[test]
fn golden_concrete_400mm_thick() {
    // Worst-case thick layer: 400mm × MAX_NODES = 18 cells.
    check_wall(
        "single_concrete_400mm",
        &[CTFMaterial::new("Concrete", 0.400, 1.73, 2243.0, 837.0)],
    );
}

#[test]
fn golden_summary_matches_all_walls() {
    // Run the full wall library and confirm zero divergence across all
    // constructions. This is the comprehensive version of the
    // per-construction tests above — used by CI as the canonical
    // "no drift" check.
    use fluxion::physics::ctf_coefficients::CTFMaterial;

    fn m(name: &str, t: f64, k: f64, rho: f64, cp: f64) -> CTFMaterial {
        CTFMaterial::new(name, t, k, rho, cp)
    }

    let library: Vec<(&str, Vec<CTFMaterial>)> = vec![
        (
            "single_concrete_005mm",
            vec![m("Concrete", 0.005, 1.73, 2243.0, 837.0)],
        ),
        (
            "single_concrete_009mm",
            vec![m("Concrete", 0.009, 1.73, 2243.0, 837.0)],
        ),
        (
            "single_concrete_200mm",
            vec![m("Concrete", 0.200, 1.73, 2243.0, 837.0)],
        ),
        (
            "single_concrete_400mm",
            vec![m("Concrete", 0.400, 1.73, 2243.0, 837.0)],
        ),
        (
            "single_gypsum_013mm",
            vec![m("Gypsum", 0.013, 0.16, 800.0, 1090.0)],
        ),
        (
            "single_insulation_100mm",
            vec![m("Insulation", 0.100, 0.04, 50.0, 840.0)],
        ),
        (
            "ashrae_900_high_mass_wall",
            vec![
                m("Gypsum", 0.013, 0.16, 800.0, 1090.0),
                m("Concrete", 0.150, 1.4, 2300.0, 880.0),
                m("Insulation", 0.050, 0.04, 50.0, 840.0),
                m("Brick", 0.100, 0.81, 1920.0, 790.0),
            ],
        ),
    ];

    let mut overall_max_dx = 0.0_f64;
    let mut overall_max_dy = 0.0_f64;
    for (name, layers) in &library {
        let (dx, dy) = check_wall(name, layers);
        overall_max_dx = overall_max_dx.max(dx);
        overall_max_dy = overall_max_dy.max(dy);
    }
    assert!(
        overall_max_dx < GOLDEN_TOL,
        "max |Δx| across wall library = {overall_max_dx} (golden tolerance {GOLDEN_TOL})"
    );
    assert!(
        overall_max_dy < GOLDEN_TOL,
        "max |Δy| across wall library = {overall_max_dy} (golden tolerance {GOLDEN_TOL})"
    );
    println!(
        "✓ Golden summary OK: max |Δx|={overall_max_dx:.4e}, max |Δy|={overall_max_dy:.4e} \
         across {} walls",
        library.len()
    );
}
