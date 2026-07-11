//! CTF (Conduction Transfer Function) step-response vs EnergyPlus reference data.
//!
//! Issue #1417: Close Module 3 (Conduction) 1% validation gap for CTF solver.
//!
//! `CTFSolverWrapper` (`src/physics/ctf_solver_wrapper.rs:115-228`) has the
//! same `HeatConductionSolver` trait as the FD solver, but until this test
//! there was no step-response-vs-E+ validation for it — only the synthetic
//! 3-term step coverage in `tests/ctf_coefficient_validation.rs`.
//!
//! This test mirrors the structure of `tests/conduction_step_response_vs_energyplus.rs`
//! (the FD suite), but drives the CTF wrapper end-to-end against the same E+
//! 25.2.0 reference CSVs and asserts the same 1% / 2% tolerance bands.
//!
//! # Reference Data Sources
//!
//! | Construction | CSV | Source |
//! |--------------|-----|--------|
//! | 200mm Concrete | `step_response_200mm_concrete.csv` | **EnergyPlus 25.2.0**, free-floating south wall, Golden-NREL TMY3 Jan 1-3 |
//! | Composite | `step_response_composite.csv` | Synthetic analytical fixture (per `tests/generate_conduction_reference_data.py`); used here as a regression-locked multi-layer check |
//!
//! # CTF Solver Specifics
//!
//! The CTF coefficients are precomputed for a 1-hour timestep and include
//! film-resistance scaling (R_SI = 0.125 m²K/W, R_SE = 0.044 m²K/W). The CSV
//! data is 15-min (288 rows over 72 hours), so we aggregate to hourly
//! resolution before stepping the solver — the CTF transfer function is
//! auto-regressive in flux history, so re-driving it with 4 × repeated
//! surface temps per hour would inflate the result. The wrapper's
//! `initialize()` already runs 7 days of diurnal warmup internally, so a
//! short additional warmup (12 hourly rows ≈ 12 h) covers the rest.
//!
//! # Acceptance Criteria (Issue #1417)
//!
//! - [x] `test_ctf_solver_concrete_200mm`: ≤1% relative error on hours
//!       where `|q_ref| > 1.0 W/m²` (the 200mm concrete free-float CSV has
//!       only 19/288 hours above the 5 W/m² threshold prescribed in the
//!       issue body, so we drop the threshold to 1.0 W/m² to keep ≥40% of
//!       the post-warmup hours in the statistical sample — the original
//!       threshold applies when the CSV has larger flux swings, e.g.
//!       `step_response_fixed_zone_20c.csv`).
//! - [x] `test_ctf_solver_composite_wall`: ≤2% relative error against the
//!       composite synthetic CSV (100mm concrete + 61.5mm foam + 100mm
//!       concrete block).
//! - [x] `cargo test --test conduction_ctf_step_response_vs_energyplus`
//!       passes in CI as a required PR-gate test (no `#\[ignore\]`).

use std::fs;
use std::path::Path;

use fluxion::physics::ctf_solver_wrapper::CTFSolverWrapper;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{
    FromF64, HeatTransferCoefficient, Temperature, Time, ToF64,
};
use fluxion::physics::wall_spec::WallSpec;

/// Reference data row from the EnergyPlus CSV output (supports both
/// `step_response_200mm_concrete.csv` and `step_response_composite.csv`).
///
/// 200mm concrete CSV columns:
///   `hour(0-72), T_ext(C), T_surface_inside(C), T_surface_outside(C),
///    heat_flux_inside(W/m2), heat_flux_outside(W/m2)`
///
/// Composite CSV columns:
///   `hour, T_outdoor, T_zone, T_surface_inside, T_surface_outside,
///    q_inside_Wm2, q_outside_Wm2`
#[derive(Debug, Clone)]
struct ReferenceRow {
    #[allow(dead_code)]
    hour: f64,
    #[allow(dead_code)]
    t_outdoor: f64,
    #[allow(dead_code)]
    t_zone: Option<f64>,
    t_surface_inside: f64,
    t_surface_outside: f64,
    q_inside_wm2: f64,
    #[allow(dead_code)]
    q_outside_wm2: f64,
}

/// Standard interior combined film coefficient (convection + LW radiation).
/// E+ default for vertical surfaces: ~8.29 W/(m²·K).
const H_INTERIOR: f64 = 8.29;

/// Standard exterior combined film coefficient.
/// E+ default for wind-exposed surfaces: ~29.3 W/(m²·K).
const H_EXTERIOR: f64 = 29.3;

/// CTF solver is configured for 1-hour coefficients, so we aggregate the
/// 15-min CSV (4 rows/hour) to hourly resolution.
const ROWS_PER_HOUR: usize = 4;

/// Number of hourly rows to skip after `initialize()` for additional
/// thermal-history stabilization (the wrapper already runs a 7-day diurnal
/// warmup internally — this only covers the CSV ramp-in).
const WARMUP_HOURS: usize = 12;

/// CTF tolerance bands (Issue #1417 acceptance criteria).
const CTF_TOLERANCE_REL_CONCRETE: f64 = 0.01;
const CTF_TOLERANCE_REL_COMPOSITE: f64 = 0.02;

/// Minimum |q_ref| for a row to enter the relative-error sample. Below this
/// the relative error is meaningless — both E+ and CTF report ~0 W/m² and
/// any tiny absolute drift blows up to infinity.
const MIN_Q_REF_FOR_REL: f64 = 1.0;

/// Load reference data from a `step_response_*.csv` file.
///
/// Handles both the 200mm concrete CSV (no T_zone column) and the
/// composite CSV (with T_zone). Returns rows in file order.
fn load_reference_data(construction_name: &str) -> Vec<ReferenceRow> {
    let path = Path::new("tests/reference_data/conduction").join(format!(
        "step_response_{}.csv",
        construction_name.to_lowercase().replace(' ', "_")
    ));
    let content = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read reference data {:?}: {}. \
             Generate E+ reference data for {} first.",
            path, e, construction_name
        )
    });

    let mut rows = Vec::new();
    let mut header: Option<String> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if header.is_none() {
            header = Some(line.to_string());
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();

        let row = if parts.len() == 6 {
            // 200mm concrete CSV: hour, T_ext, T_surface_inside,
            // T_surface_outside, heat_flux_inside, heat_flux_outside
            ReferenceRow {
                hour: parts[0].trim().parse().expect("hour"),
                t_outdoor: parts[1].trim().parse().expect("T_ext"),
                t_zone: None,
                t_surface_inside: parts[2].trim().parse().expect("T_surface_inside"),
                t_surface_outside: parts[3].trim().parse().expect("T_surface_outside"),
                q_inside_wm2: parts[4].trim().parse().expect("heat_flux_inside"),
                q_outside_wm2: parts[5].trim().parse().expect("heat_flux_outside"),
            }
        } else if parts.len() == 7 {
            // Composite CSV: hour, T_outdoor, T_zone, T_surface_inside,
            // T_surface_outside, q_inside_Wm2, q_outside_Wm2
            ReferenceRow {
                hour: parts[0].trim().parse().expect("hour"),
                t_outdoor: parts[1].trim().parse().expect("T_outdoor"),
                t_zone: Some(parts[2].trim().parse().expect("T_zone")),
                t_surface_inside: parts[3].trim().parse().expect("T_surface_inside"),
                t_surface_outside: parts[4].trim().parse().expect("T_surface_outside"),
                q_inside_wm2: parts[5].trim().parse().expect("q_inside_Wm2"),
                q_outside_wm2: parts[6].trim().parse().expect("q_outside_Wm2"),
            }
        } else {
            panic!(
                "Unexpected column count {} in {:?}: {:?}",
                parts.len(),
                path,
                parts
            );
        };
        rows.push(row);
    }

    assert!(
        !rows.is_empty(),
        "Reference data should contain rows for {}",
        construction_name
    );
    rows
}

/// Aggregate 15-min rows to hourly by taking every Nth row (i.e., the
/// snapshot at the top of each hour group).
fn aggregate_hourly(rows: &[ReferenceRow]) -> Vec<ReferenceRow> {
    rows.iter()
        .step_by(ROWS_PER_HOUR)
        .cloned()
        .collect()
}

/// Test result for a single CTF-vs-E+ comparison run.
struct CtfTestResult {
    name: String,
    tolerance: f64,
    passed: bool,
    max_absolute_error: f64,
    max_relative_error: f64,
    p95_relative_error: f64,
    sample_size: usize,
    total_post_warmup: usize,
}

/// Drive the CTF wrapper hourly through the reference data and collect
/// flux-error statistics against `q_inside_wm2`.
///
/// Steps the wrapper with `(T_surface_inside, T_surface_outside)` as the
/// interior/exterior boundary — the CTF coefficient formulation already
/// encodes R_SI/R_SE film resistance scaling, so this is the path that
/// produces an inside-surface heat flux directly comparable to E+'s
/// `Surface Inside Face Convection + Radiation Heat Flux` (`heat_flux_inside`
/// in the CSV).
fn run_ctf_step_response(
    spec: &WallSpec,
    ref_data: &[ReferenceRow],
    construction_name: &str,
    tolerance: f64,
) -> CtfTestResult {
    let mut wrapper = CTFSolverWrapper::with_convection(H_INTERIOR, H_EXTERIOR);
    wrapper.initialize(spec).expect("CTF initialize");

    let hourly = aggregate_hourly(ref_data);

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut post_warmup_count = 0_usize;
    let mut rel_errors: Vec<f64> = Vec::new();

    for (i, row) in hourly.iter().enumerate() {
        let q_computed = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(row.t_surface_inside),
                Temperature::from_value(row.t_surface_outside),
                HeatTransferCoefficient::from_value(H_INTERIOR),
                HeatTransferCoefficient::from_value(H_EXTERIOR),
            )
            .expect("CTF step")
            .to_value();

        // Skip warmup so any residual coefficient-history ramp-in does not
        // contaminate the statistical sample.
        if i < WARMUP_HOURS {
            continue;
        }
        post_warmup_count += 1;

        let q_ref = row.q_inside_wm2;
        let abs_err = (q_computed - q_ref).abs();
        max_abs = max_abs.max(abs_err);

        if q_ref.abs() > MIN_Q_REF_FOR_REL {
            let rel_err = abs_err / q_ref.abs();
            max_rel = max_rel.max(rel_err);
            rel_errors.push(rel_err);
        }
    }

    // 95th-percentile of the relative-error sample. The issue body asks for
    // "1% on the inner 95th-percentile of |q|>5 W/m² hours"; here we keep
    // the same percentile statistic but widen the flux filter to |q|>1
    // W/m² because the free-floating 200mm concrete CSV barely crosses
    // 5 W/m² at all (19/288 hours).
    rel_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_idx = ((rel_errors.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(rel_errors.len().saturating_sub(1));
    let p95_rel = rel_errors.get(p95_idx).copied().unwrap_or(0.0);

    // Pass = p95 relative error is within tolerance AND max relative error
    // does not catastrophically exceed 3× the tolerance (one bad row).
    let passed = p95_rel <= tolerance && max_rel <= 3.0 * tolerance;

    CtfTestResult {
        name: construction_name.to_string(),
        tolerance,
        passed,
        max_absolute_error: max_abs,
        max_relative_error: max_rel,
        p95_relative_error: p95_rel,
        sample_size: rel_errors.len(),
        total_post_warmup: post_warmup_count,
    }
}

fn report(result: &CtfTestResult) {
    eprintln!("\n=== CTF Solver: {} ===", result.name);
    eprintln!(
        "Result: {} ({}/{} rows in sample, |q_ref|>{:.1} W/m²)",
        if result.passed { "PASS" } else { "FAIL" },
        result.sample_size,
        result.total_post_warmup,
        MIN_Q_REF_FOR_REL,
    );
    eprintln!("Tolerance (p95 rel error): {:.2}%", result.tolerance * 100.0);
    eprintln!("Max absolute error: {:.4} W/m²", result.max_absolute_error);
    eprintln!(
        "Max relative error:  {:.2}%",
        result.max_relative_error * 100.0
    );
    eprintln!(
        "p95 relative error:  {:.2}%",
        result.p95_relative_error * 100.0
    );
}

// ===========================================================================
// CTF vs E+ Tests
// ===========================================================================

#[test]
fn test_ctf_solver_concrete_200mm() {
    // 200mm concrete, k=1.73, ρ=2300, cp=880 (per issue body spec)
    let spec = WallSpec::single_layer(
        "200mm Concrete",
        0.2,    // thickness [m]
        1.73,   // conductivity [W/(m·K)]
        2300.0, // density [kg/m³]
        880.0,  // specific heat [J/(kg·K)]
    );
    let ref_data = load_reference_data("200mm_concrete");

    let result = run_ctf_step_response(&spec, &ref_data, "200mm Concrete", CTF_TOLERANCE_REL_CONCRETE);
    report(&result);

    assert!(
        result.passed,
        "CTF solver vs E+ failed for {}: p95 rel error {:.2}% > {:.2}%, \
         max rel error {:.2}%, max abs error {:.4} W/m² ({}/{} sampled rows)",
        result.name,
        result.p95_relative_error * 100.0,
        result.tolerance * 100.0,
        result.max_relative_error * 100.0,
        result.max_absolute_error,
        result.sample_size,
        result.total_post_warmup,
    );
}

#[test]
fn test_ctf_solver_composite_wall() {
    // 100mm concrete + 61.5mm foam + 100mm concrete block (multi-layer)
    // This CSV is the synthetic analytical fixture, not E+ — used here as
    // a regression-locked multi-layer check (≤2% tolerance).
    let spec = WallSpec::multi_layer(
        "Composite Concrete",
        vec![
            fluxion::physics::wall_spec::LayerSpec::new(
                "Concrete Inner", 0.100, 1.13, 1400.0, 1000.0,
            ),
            fluxion::physics::wall_spec::LayerSpec::new(
                "Foam Insulation", 0.0615, 0.04, 14.0, 1400.0,
            ),
            fluxion::physics::wall_spec::LayerSpec::new(
                "Concrete Block", 0.100, 0.51, 1400.0, 840.0,
            ),
        ],
    );
    let ref_data = load_reference_data("composite");

    let result = run_ctf_step_response(&spec, &ref_data, "Composite Wall", CTF_TOLERANCE_REL_COMPOSITE);
    report(&result);

    assert!(
        result.passed,
        "CTF solver vs synthetic failed for {}: p95 rel error {:.2}% > {:.2}%, \
         max rel error {:.2}%, max abs error {:.4} W/m² ({}/{} sampled rows)",
        result.name,
        result.p95_relative_error * 100.0,
        result.tolerance * 100.0,
        result.max_relative_error * 100.0,
        result.max_absolute_error,
        result.sample_size,
        result.total_post_warmup,
    );
}

#[test]
fn test_ctf_solver_summary() {
    let constructions: Vec<(&str, WallSpec, &str, f64)> = vec![
        (
            "200mm Concrete",
            WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2300.0, 880.0),
            "200mm_concrete",
            CTF_TOLERANCE_REL_CONCRETE,
        ),
        (
            "Composite Wall",
            WallSpec::multi_layer(
                "Composite Concrete",
                vec![
                    fluxion::physics::wall_spec::LayerSpec::new(
                        "Concrete Inner", 0.100, 1.13, 1400.0, 1000.0,
                    ),
                    fluxion::physics::wall_spec::LayerSpec::new(
                        "Foam Insulation", 0.0615, 0.04, 14.0, 1400.0,
                    ),
                    fluxion::physics::wall_spec::LayerSpec::new(
                        "Concrete Block", 0.100, 0.51, 1400.0, 840.0,
                    ),
                ],
            ),
            "composite",
            CTF_TOLERANCE_REL_COMPOSITE,
        ),
    ];

    let mut results = Vec::new();
    for (name, spec, ref_name, tol) in &constructions {
        let ref_data = load_reference_data(ref_name);
        results.push(run_ctf_step_response(spec, &ref_data, name, *tol));
    }

    eprintln!("\n========================================");
    eprintln!("CTF Solver Per-Construction Summary");
    eprintln!("========================================");
    eprintln!(
        "{:<20} {:>12} {:>12} {:>12} {:>8}",
        "Construction", "Max Abs Err", "p95 Rel Err", "Max Rel Err", "Status"
    );
    eprintln!("----------------------------------------");
    for r in &results {
        eprintln!(
            "{:<20} {:>10.4} W/m² {:>10.2}% {:>10.2}% {:>8}",
            r.name,
            r.max_absolute_error,
            r.p95_relative_error * 100.0,
            r.max_relative_error * 100.0,
            if r.passed { "PASS" } else { "FAIL" }
        );
    }
    eprintln!("========================================\n");

    assert!(
        results.iter().all(|r| r.passed),
        "Some CTF validation runs failed"
    );
}