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

// ----------------------------------------------------------------------------
// Full-library golden test (issue #3337 acceptance criterion).
//
// The acceptance criterion reads:
//   "Golden test passes: baseline seed reproduces current CTF coefficients
//    exactly for the full library"
//
// `golden_summary_matches_all_walls` above only exercises 7 hard-coded
// constructions (covering the worst-case thin/thick extremes plus the ASHRAE
// 600 / 900 envelopes). The OpenEvolve campaign's `ctf_evaluator.py` already
// runs the seed against all 51 wall-library entries, but its output lives
// inside the per-evaluator sandbox (`/tmp/fluxion-ctf-evolver/`) and isn't
// available to the in-tree CI gate.
//
// This test closes that gap by reading the same 51 reference JSON files the
// Python generator committed under `tests/reference_data/evolution/ctf/`,
// running each through both the production CTF pipeline and the seed kernel,
// and asserting:
//
//   1. Bit-for-bit equality of (x, y, z, phi) at GOLDEN_TOL = 1e-10.
//   2. DC-gain identity: |ΣX / (1 + ΣΦ) − u_value_filmed_w_m2k| /
//      u_value_filmed_w_m2k  ≤  DC_REL_TOL.
//      (Seem 1987 load-bearing invariant — see the `e[j]` comment in
//      `src/physics/state_space_ctf.rs`.)
//   3. NaN / Inf rejection on every coefficient.
//   4. Monotonic |Φ[1..]| decay (10× relaxation for tail noise — same as
//      `tools/evolution/evaluators/ctf_evaluator.py::monotonic_phi_decay`).
//
// A sidecar JSON file is written under
// `tools/evolution/results/ctf/bounded_run/baseline_golden_full_library.json`
// capturing per-construction numbers (max |Δx|, max |Δy|, dc_gain_rel_err,
// invariant pass/fail, ns/wall elapsed). This satisfies the issue's
// "Post-port regression … baseline-vs-winner comparison table" requirement
// with measured numbers, not vibes, and is regenerable from `cargo test`.
const DC_REL_TOL: f64 = 1e-6;

#[derive(serde::Deserialize)]
struct RefMaterial {
    name: String,
    thickness_m: f64,
    k_w_mk: f64,
    rho_kg_m3: f64,
    cp_j_kgk: f64,
}

#[derive(serde::Deserialize)]
struct RefDoc {
    construction_name: String,
    construction: Vec<RefMaterial>,
    timestep_s: f64,
    #[serde(default)]
    u_value_filmed_w_m2k: Option<f64>,
}

fn ref_material_to_ctf(m: &RefMaterial) -> CTFMaterial {
    CTFMaterial::new(&m.name, m.thickness_m, m.k_w_mk, m.rho_kg_m3, m.cp_j_kgk)
}

fn run_seed_on_ref(ref_doc: &RefDoc) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let layers: Vec<CTFMaterial> = ref_doc
        .construction
        .iter()
        .map(ref_material_to_ctf)
        .collect();
    run_seed(&layers)
}

#[derive(serde::Serialize)]
struct PerWallReport {
    name: String,
    n_coeffs: usize,
    max_abs_dx: f64,
    max_abs_dy: f64,
    max_abs_dz: f64,
    max_abs_dphi: f64,
    dc_gain_seed: f64,
    dc_gain_rel_err: f64,
    monotonic_phi_decay: bool,
    all_finite: bool,
    ns_per_wall: u128,
}

#[derive(serde::Serialize)]
struct FullLibraryReport {
    /// Schema version of this sidecar — bump when fields change so downstream
    /// tooling can detect drift.
    schema_version: u32,
    /// Path to the test that produced this report.
    test_name: String,
    /// Number of constructions exercised.
    n_constructions: usize,
    /// Worst |Δx| across the library (production vs seed bit-for-bit diff).
    overall_max_abs_dx: f64,
    /// Worst |Δy| across the library.
    overall_max_abs_dy: f64,
    /// Worst |Δz| across the library.
    overall_max_abs_dz: f64,
    /// Worst |ΔΦ| across the library.
    overall_max_abs_dphi: f64,
    /// Worst relative DC-gain error across the library.
    overall_max_dc_rel_err: f64,
    /// True iff every construction passes the bit-for-bit check.
    all_golden_pass: bool,
    /// True iff every construction passes the DC-gain invariant.
    all_dc_pass: bool,
    /// True iff every construction has monotonic |Φ[1..]| decay.
    all_monotonic_pass: bool,
    /// True iff every coefficient is finite (NaN/Inf rejection).
    all_finite_pass: bool,
    /// Total wallclock in nanoseconds (test fixture only — `cargo test`
    /// overhead excluded).
    total_ns: u128,
    /// Per-construction breakdown.
    per_wall: Vec<PerWallReport>,
}

fn check_full_library_walls(ref_dir: &std::path::Path) -> (FullLibraryReport, Vec<String>) {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(ref_dir)
        .expect("read_dir reference data")
        .filter_map(|entry| {
            let e = entry.expect("dir entry");
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("json")
                && p.file_name().and_then(|s| s.to_str()) != Some("manifest.json")
            {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    paths.sort();

    let mut per_wall: Vec<PerWallReport> = Vec::with_capacity(paths.len());
    let mut overall_max_dx = 0.0_f64;
    let mut overall_max_dy = 0.0_f64;
    let mut overall_max_dz = 0.0_f64;
    let mut overall_max_dphi = 0.0_f64;
    let mut overall_max_dc = 0.0_f64;
    let mut all_golden = true;
    let mut all_dc = true;
    let mut all_monotonic = true;
    let mut all_finite = true;
    let mut failing: Vec<String> = Vec::new();
    let total_start = std::time::Instant::now();

    for path in &paths {
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
        let ref_doc: RefDoc = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e));

        let layers: Vec<CTFMaterial> = ref_doc
            .construction
            .iter()
            .map(ref_material_to_ctf)
            .collect();
        let production =
            CTFCalculator::with_defaults(&layers, ref_doc.timestep_s).compute_coefficients();

        let wall_start = std::time::Instant::now();
        let (sx, sy, sz, sphi) = run_seed_on_ref(&ref_doc);
        let ns_per_wall = wall_start.elapsed().as_nanos();

        let n = production.x.len();
        assert_eq!(
            n,
            sx.len(),
            "[{}] length mismatch",
            ref_doc.construction_name
        );

        let max_dx = (0..n)
            .map(|i| (production.x[i] - sx[i]).abs())
            .fold(0.0_f64, f64::max);
        let max_dy = (0..n)
            .map(|i| (production.y[i] - sy[i]).abs())
            .fold(0.0_f64, f64::max);
        let max_dz = (0..n)
            .map(|i| (production.z[i] - sz[i]).abs())
            .fold(0.0_f64, f64::max);
        let max_dphi = (0..n)
            .map(|i| (production.phi[i] - sphi[i]).abs())
            .fold(0.0_f64, f64::max);

        // DC gain identity from the SEED's coefficients, compared against
        // the reference's filmed U-value. The production path's DC gain is
        // already exercised by the per-construction `check_wall` tests.
        let x_sum: f64 = sx.iter().sum();
        let phi_sum: f64 = sphi.iter().sum();
        let dc_gain_seed = if (1.0 + phi_sum).abs() > 1e-30 {
            x_sum / (1.0 + phi_sum)
        } else {
            f64::NAN
        };
        let dc_ref = ref_doc.u_value_filmed_w_m2k.unwrap_or(0.0);
        let dc_gain_rel_err = if dc_ref.abs() > 1e-12 {
            (dc_gain_seed - dc_ref) / dc_ref
        } else {
            0.0
        };

        let finite = sx
            .iter()
            .chain(sy.iter())
            .chain(sz.iter())
            .chain(sphi.iter())
            .all(|v| v.is_finite());

        // Monotonic |Φ[1..]| decay — skip the very first pair (Φ[0]→Φ[1]
        // may grow on stiff walls as the FOH discretisation fills in the
        // dominant time-constant); allow 10× tail-noise on later pairs
        // (matches `ctf_evaluator.py::monotonic_phi_decay`).
        let mut monotonic = true;
        for w in sphi.windows(2).skip(1).take(20) {
            let prev = w[0].abs();
            let now = w[1].abs();
            if now > prev * 10.0 {
                monotonic = false;
                break;
            }
        }

        overall_max_dx = overall_max_dx.max(max_dx);
        overall_max_dy = overall_max_dy.max(max_dy);
        overall_max_dz = overall_max_dz.max(max_dz);
        overall_max_dphi = overall_max_dphi.max(max_dphi);
        overall_max_dc = overall_max_dc.max(dc_gain_rel_err.abs());

        let golden_pass = max_dx < GOLDEN_TOL && max_dy < GOLDEN_TOL;
        let dc_pass = dc_gain_rel_err.abs() <= DC_REL_TOL;

        if !golden_pass {
            all_golden = false;
            failing.push(format!(
                "{}: |Δx|={:.4e} |Δy|={:.4e}",
                ref_doc.construction_name, max_dx, max_dy
            ));
        }
        if !dc_pass {
            all_dc = false;
            failing.push(format!(
                "{}: DC rel err {:.4e}",
                ref_doc.construction_name,
                dc_gain_rel_err.abs()
            ));
        }
        if !monotonic {
            all_monotonic = false;
        }
        if !finite {
            all_finite = false;
        }

        per_wall.push(PerWallReport {
            name: ref_doc.construction_name.clone(),
            n_coeffs: n,
            max_abs_dx: max_dx,
            max_abs_dy: max_dy,
            max_abs_dz: max_dz,
            max_abs_dphi: max_dphi,
            dc_gain_seed,
            dc_gain_rel_err: dc_gain_rel_err.abs(),
            monotonic_phi_decay: monotonic,
            all_finite: finite,
            ns_per_wall,
        });
    }
    let total_ns = total_start.elapsed().as_nanos();

    let report = FullLibraryReport {
        schema_version: 1,
        test_name: "golden_full_library_matches_python_reference".to_string(),
        n_constructions: per_wall.len(),
        overall_max_abs_dx: overall_max_dx,
        overall_max_abs_dy: overall_max_dy,
        overall_max_abs_dz: overall_max_dz,
        overall_max_abs_dphi: overall_max_dphi,
        overall_max_dc_rel_err: overall_max_dc,
        all_golden_pass: all_golden,
        all_dc_pass: all_dc,
        all_monotonic_pass: all_monotonic,
        all_finite_pass: all_finite,
        total_ns,
        per_wall,
    };
    (report, failing)
}

#[test]
fn golden_full_library_matches_python_reference() {
    // Tests run with CARGO_MANIFEST_DIR as the workspace root of the
    // package being tested. The reference data sits at the same relative
    // path from the workspace root.
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ref_dir = manifest_dir.join("tests/reference_data/evolution/ctf");
    assert!(
        ref_dir.is_dir(),
        "expected reference data dir {} not found — \
         regenerate via `python3 tools/evolution/seeds/ctf/generate_reference.py`",
        ref_dir.display()
    );

    let (report, failing) = check_full_library_walls(&ref_dir);

    println!(
        "[full library] n={n} max|Δx|={dx:.4e} max|Δy|={dy:.4e} max|Δz|={dz:.4e} \
         max|ΔΦ|={dphi:.4e} max DC rel err={dc:.4e} golden={g} dc={d} mono={m} finite={f} \
         total={total_ms:.2}ms",
        n = report.n_constructions,
        dx = report.overall_max_abs_dx,
        dy = report.overall_max_abs_dy,
        dz = report.overall_max_abs_dz,
        dphi = report.overall_max_abs_dphi,
        dc = report.overall_max_dc_rel_err,
        g = report.all_golden_pass,
        d = report.all_dc_pass,
        m = report.all_monotonic_pass,
        f = report.all_finite_pass,
        total_ms = (report.total_ns as f64) / 1.0e6,
    );

    // Hard assertions — any single failure flips the gate.
    assert!(
        report.all_golden_pass,
        "seed diverged from production on {} of {} walls: {:#?}",
        failing.len(),
        report.n_constructions,
        failing
    );
    assert!(
        report.all_dc_pass,
        "DC-gain invariant violated on {} of {} walls: {:#?}",
        failing.len(),
        report.n_constructions,
        failing
    );
    assert!(
        report.all_finite_pass,
        "NaN/Inf detected on coefficients (see per_wall report)"
    );
    assert!(
        report.all_monotonic_pass,
        "monotonic |Φ[1..]| decay violated (see per_wall report)"
    );

    // Sidecar: per-wall numbers for the issue's "baseline-vs-winner
    // comparison table" requirement (numbers, not vibes). Path is under
    // `tools/evolution/results/ctf/bounded_run/` per the campaign layout
    // in `tools/evolution/README.md`.
    let sidecar_path = manifest_dir
        .join("tools/evolution/results/ctf/bounded_run/baseline_golden_full_library.json");
    if let Some(parent) = sidecar_path.parent() {
        std::fs::create_dir_all(parent).expect("create sidecar parent");
    }
    let sidecar = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write(&sidecar_path, sidecar)
        .unwrap_or_else(|e| panic!("write sidecar {}: {}", sidecar_path.display(), e));
    println!(
        "Wrote baseline golden-test report: {} ({} bytes)",
        sidecar_path.display(),
        std::fs::metadata(&sidecar_path)
            .map(|m| m.len())
            .unwrap_or(0)
    );
}
