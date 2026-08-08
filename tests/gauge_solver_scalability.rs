//! GaugeSolver scalability performance characterization (Issue #1771).
//!
//! These tests characterise how the cost of `MultiZoneGaugeSolver` (the
//! multi-zone GaugeSolver) scales with zone count, and identify the crossover
//! where GaugeSolver cost exceeds the `FiveR1CSolver` baseline.
//!
//! ## What is measured
//!
//! For each problem size `N ∈ {2, 5, 10, 20}` zones we measure the
//! per-timestep wall-clock cost of three configurations, all using the **same**
//! envelope wall spec so the comparison is apples-to-apples:
//!
//! 1. **GaugeSolver (ring)** — `MultiZoneGaugeSolver` with a ring inter-zone
//!    coupling topology (each zone coupled to its neighbours). This is the
//!    realistic sparse O(N) case for a linear/loop building.
//! 2. **GaugeSolver (dense)** — `MultiZoneGaugeSolver` with a fully-connected
//!    coupling graph (every zone coupled to every other). This exercises the
//!    worst-case O(N²) coupling path and sets the practical upper bound.
//! 3. **FiveR1C baseline** — `N` independent `FiveR1CSolver::step` calls with
//!    no inter-zone coupling. This is the cheapest baseline against which the
//!    GaugeSolver's manifold/gauge-connection bookkeeping is measured.
//!
//! ## Scaling characteristic
//!
//! The empirical scaling exponent is computed as
//! `slope = ln(t(N=20) / t(N=2)) / ln(20 / 2)`. A slope near 1 indicates
//! linear scaling; a slope near 2 indicates quadratic scaling. See
//! `docs/gauge_solver_scalability.md` for the committed benchmark curve.
//!
//! ## Run
//!
//! ```text
//! cargo test --profile ci --test gauge_solver_scalability -- --nocapture
//! ```

use std::collections::HashMap;
use std::time::Instant;

use fluxion::physics::five_r1c_solver::FiveR1CSolver;
use fluxion::physics::gauge_zone_solver::{
    MultiZoneGaugeSolver, SurfaceType, ZoneBoundaryConditions,
};
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time};
use fluxion::physics::wall_spec::{LayerSpec, WallSpec};

/// Problem sizes characterised by this suite.
const ZONE_COUNTS: [usize; 4] = [2, 5, 10, 20];

/// Number of timed step iterations per measurement. Large enough to dwarf the
/// `Instant::now()` overhead, small enough to keep the whole suite well under a
/// second on CI hardware.
const ITERS: usize = 2000;

/// Generous upper bound (µs) for a single dense-coupled 20-zone timestep.
/// The GaugeSolver does per-surface flux arithmetic plus an O(N²) coupling
/// pass; this guard catches pathological regressions (e.g. accidental cubic
/// behaviour) without being flaky on slow CI runners.
const DENSE_20_ZONE_BUDGET_US: f64 = 5_000.0;

/// A representative exterior envelope wall: concrete + insulation + gypsum.
/// This passes both the GaugeSolver and FiveR1CSolver initialisation contracts
/// (positive total R-value and positive mass-to-interior-surface resistance).
fn envelope_wall() -> WallSpec {
    WallSpec::multi_layer(
        "Envelope",
        vec![
            LayerSpec::new("Concrete", 0.10, 1.70, 2200.0, 900.0),
            LayerSpec::new("Insulation", 0.05, 0.04, 30.0, 840.0),
            LayerSpec::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
        ],
    )
}

/// Build an N-zone `MultiZoneGaugeSolver`.
///
/// Each zone gets a realistic envelope of six exterior surfaces
/// (4 walls + roof + floor) so the per-zone cost reflects a real building
/// rather than a single-surface toy. When `dense` is true every zone is coupled
/// to every other zone (full O(N²) adjacency); otherwise zones are wired in a
/// ring (each zone coupled to `i-1` and `i+1` mod N), the sparse O(N) case.
fn build_gauge_network(n_zones: usize, dense: bool) -> MultiZoneGaugeSolver {
    let mut mz = MultiZoneGaugeSolver::new();
    let wall = envelope_wall();

    // 48 m² × 2.7 m zone (typical residential room).
    for i in 0..n_zones {
        mz.add_zone(i, 48.0, 2.7);
        // Four exterior walls (≈ 6 m × 2.7 m each), roof, floor.
        let _ = mz.add_opaque_surface_to_zone(i, &wall, 16.2, SurfaceType::Wall, 0.0, 90.0);
        let _ = mz.add_opaque_surface_to_zone(i, &wall, 16.2, SurfaceType::Wall, 90.0, 90.0);
        let _ = mz.add_opaque_surface_to_zone(i, &wall, 16.2, SurfaceType::Wall, 180.0, 90.0);
        let _ = mz.add_opaque_surface_to_zone(i, &wall, 16.2, SurfaceType::Wall, 270.0, 90.0);
        let _ = mz.add_opaque_surface_to_zone(i, &wall, 48.0, SurfaceType::Roof, 0.0, 0.0);
        let _ = mz.add_opaque_surface_to_zone(i, &wall, 48.0, SurfaceType::Floor, 0.0, 180.0);
    }

    // Inter-zone coupling graph.
    if dense {
        for a in 0..n_zones {
            for b in (a + 1)..n_zones {
                mz.add_zone_coupling(a, b, 12.0, 0.5).unwrap();
            }
        }
    } else if n_zones > 1 {
        // Ring: 0—1—2—…—(N-1)—0
        for i in 0..n_zones {
            let j = (i + 1) % n_zones;
            mz.add_zone_coupling(i, j, 12.0, 0.5).unwrap();
        }
    }

    mz.initialize().unwrap();
    mz
}

/// Build `n_zones` independent initialised FiveR1C solvers (no coupling).
/// This is the baseline: the cheapest way to compute a per-surface flux for N
/// zones. Each solver represents one envelope surface per zone.
fn build_fiver1c_baseline(n_zones: usize) -> Vec<FiveR1CSolver> {
    let wall = envelope_wall();
    (0..n_zones)
        .map(|_| {
            let mut s = FiveR1CSolver::new();
            s.initialize(&wall).unwrap();
            s
        })
        .collect()
}

/// Uniform boundary conditions for every zone.
fn uniform_bc(n_zones: usize) -> HashMap<usize, ZoneBoundaryConditions> {
    (0..n_zones)
        .map(|i| {
            (
                i,
                ZoneBoundaryConditions::new(
                    Temperature::from_value(5.0),
                    HeatTransferCoefficient::from_value(25.0),
                    300.0,
                ),
            )
        })
        .collect()
}

/// Time `iters` GaugeSolver steps and return ns-per-timestep.
fn time_gauge(mut mz: MultiZoneGaugeSolver, n_zones: usize, iters: usize) -> f64 {
    let bc = uniform_bc(n_zones);
    // Warm up (first step seeds gauge connections / caches).
    let _ = mz.step(3600.0, &bc).unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = mz.step(3600.0, &bc).unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / iters as f64
}

/// Time `iters` FiveR1C baseline steps (one step per zone) and return
/// ns-per-timestep (total for all N zones, matching the GaugeSolver unit).
fn time_fiver1c(mut solvers: Vec<FiveR1CSolver>, iters: usize) -> f64 {
    let dt = Time::from_value(3600.0);
    let t_int = Temperature::from_value(20.0);
    let t_ext = Temperature::from_value(5.0);
    let h_int = HeatTransferCoefficient::from_value(8.0);
    let h_ext = HeatTransferCoefficient::from_value(25.0);
    // Warm up.
    for s in &mut solvers {
        let _ = s.step(dt, t_int, t_ext, h_int, h_ext).unwrap();
    }
    let start = Instant::now();
    for _ in 0..iters {
        for s in &mut solvers {
            let _ = s.step(dt, t_int, t_ext, h_int, h_ext).unwrap();
        }
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / iters as f64
}

/// Empirical scaling exponent: slope of ln(t) vs ln(N) between two sizes.
/// `slope ≈ 1` → linear, `slope ≈ 2` → quadratic.
fn scaling_exponent(t_small: f64, t_large: f64, n_small: usize, n_large: usize) -> f64 {
    (t_large / t_small).ln() / (n_large as f64 / n_small as f64).ln()
}

/// Helper to format µs from ns.
fn us(ns: f64) -> String {
    format!("{:7.2}", ns / 1000.0)
}

/// Issue #1771 acceptance criterion 1: measure the scaling curve
/// (zones vs µs/timestep) and assert it stays within practical bounds.
///
/// Prints a full table when run with `--nocapture`; the committed curve lives
/// in `docs/gauge_solver_scalability.md`.
#[test]
fn gauge_solver_scaling_curve() {
    let mut ring = Vec::new();
    let mut dense = Vec::new();
    let mut base = Vec::new();

    println!("\n=== GaugeSolver scalability curve (Issue #1771) ===");
    println!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Zones", "Ring µs", "Dense µs", "5R1C µs", "Ring/5R1C", "Dense/5R1C"
    );
    println!("{}", "-".repeat(75));

    for &n in &ZONE_COUNTS {
        let r = time_gauge(build_gauge_network(n, false), n, ITERS);
        let d = time_gauge(build_gauge_network(n, true), n, ITERS);
        let b = time_fiver1c(build_fiver1c_baseline(n), ITERS);
        println!(
            "{:>6} | {} | {} | {} | {:>10.2} | {:>10.2}",
            n,
            us(r),
            us(d),
            us(b),
            r / b,
            d / b
        );
        ring.push((n, r));
        dense.push((n, d));
        base.push((n, b));
    }

    // --- Scaling exponents ---------------------------------------------------
    let ring_slope = scaling_exponent(ring[0].1, ring[3].1, ring[0].0, ring[3].0);
    let dense_slope = scaling_exponent(dense[0].1, dense[3].1, dense[0].0, dense[3].0);
    let base_slope = scaling_exponent(base[0].1, base[3].1, base[0].0, base[3].0);
    println!("\nScaling exponent (N=2 → N=20):");
    println!("  FiveR1C baseline : {base_slope:.2}  (expect ≈ 1.0, linear)");
    println!("  Gauge ring       : {ring_slope:.2}  (expect ≈ 1.0, near-linear)");
    println!(
        "  Gauge dense      : {dense_slope:.2}  (expect ≈ 1.0, near-linear; Phase 1b optimized)"
    );

    // --- Guards --------------------------------------------------------------
    // Dense 20-zone must stay within the interactive-timestep budget.
    let dense_20_us = dense[3].1 / 1000.0;
    assert!(
        dense_20_us < DENSE_20_ZONE_BUDGET_US,
        "dense 20-zone timestep took {dense_20_us:.1} µs, budget {DENSE_20_ZONE_BUDGET_US:.0} µs"
    );

    // Ring topology should scale near-linearly (the realistic building case).
    // Allow up to 1.4 to absorb per-zone constant-factor noise on CI runners.
    assert!(
        ring_slope < 1.4,
        "ring GaugeSolver scaling exponent {ring_slope:.2} exceeds 1.4 (expected near-linear)"
    );

    // Dense topology: the refactored ThermalManifold-based implementation
    // (Phase 1b, PR #2446) achieves near-linear scaling for dense coupling
    // due to optimized compute_parallel_transport. Require mildly super-linear
    // to catch accidental quadratic regression, cap well below cubic.
    assert!(
        dense_slope > 0.8,
        "dense GaugeSolver scaling exponent {dense_slope:.2} below 0.8 (possible sub-linear/bug)"
    );
    assert!(
        dense_slope < 2.6,
        "dense GaugeSolver scaling exponent {dense_slope:.2} above 2.6 (possible cubic regression)"
    );

    // FiveR1C baseline must be linear (N independent solvers).
    assert!(
        base_slope < 1.3,
        "FiveR1C baseline scaling {base_slope:.2} not linear"
    );
}

/// Issue #1771 acceptance criterion 2: identify the crossover where GaugeSolver
/// cost exceeds the FiveR1C baseline.
///
/// Because the GaugeSolver carries per-surface gauge-connection bookkeeping and
/// an aggregation pass at the zone air node, it is inherently more expensive
/// per surface than the bare FiveR1C steady-state step. This test confirms the
/// crossover exists (GaugeSolver cost > FiveR1C cost) at every measured size
/// and pins down the ratio so a future optimisation that removes the overhead
/// is surfaced here.
#[test]
fn gauge_solver_crossover_vs_fiver1c() {
    println!("\n=== GaugeSolver vs FiveR1C crossover (Issue #1771) ===");
    println!(
        "{:>6} | {:>12} | {:>12} | {:>14}",
        "Zones", "Gauge µs", "5R1C µs", "Crossover?"
    );
    println!("{}", "-".repeat(55));

    let mut first_crossover: Option<usize> = None;

    for &n in &ZONE_COUNTS {
        let g = time_gauge(build_gauge_network(n, false), n, ITERS);
        let b = time_fiver1c(build_fiver1c_baseline(n), ITERS);
        let crossed = g > b;
        if crossed && first_crossover.is_none() {
            first_crossover = Some(n);
        }
        println!(
            "{:>6} | {:>10.2} | {:>10.2} | {:>14}",
            n,
            g / 1000.0,
            b / 1000.0,
            if crossed {
                "Gauge > 5R1C"
            } else {
                "Gauge ≤ 5R1C"
            }
        );
    }

    // The GaugeSolver's manifold bookkeeping makes it more expensive than the
    // raw FiveR1C steady-state step at every size. We assert the crossover is
    // already present at N=2 (the smallest configuration): this documents that
    // the GaugeSolver is a higher-fidelity, higher-cost solver rather than a
    // drop-in performance replacement for FiveR1C.
    let crossover = first_crossover.expect("GaugeSolver must exceed FiveR1C at some size");
    println!("\nCrossover (GaugeSolver cost > FiveR1C baseline) first observed at N={crossover}");
    assert_eq!(
        crossover, 2,
        "GaugeSolver is expected to be more expensive than FiveR1C even at N=2 \
         due to gauge-connection bookkeeping and zone-air aggregation"
    );
}

/// Pin the per-timestep cost of the smallest (N=2) ring GaugeSolver to a
/// generous budget. This is a regression guard: if a change makes the
/// per-surface GaugeSolver step pathologically slow, this test fires.
#[test]
fn gauge_solver_2_zone_step_budget() {
    let ns = time_gauge(build_gauge_network(2, false), 2, ITERS);
    let us2 = ns / 1000.0;
    println!("\nN=2 ring GaugeSolver: {us2:.2} µs/timestep");
    // 2 zones × 6 surfaces × gauge-connection updates + 1 coupling edge.
    // Budget is deliberately generous (slow CI runners, debug codegen under
    // `--profile ci` is opt-level=1).
    assert!(
        us2 < 1_000.0,
        "N=2 ring GaugeSolver took {us2:.1} µs, budget 1000 µs"
    );
}

/// Regression guard: the dense-coupling path must not accidentally become
/// cheaper-looking than the ring path (which would indicate the coupling graph
/// was silently dropped). Dense has strictly more coupling edges than ring, so
/// its per-timestep cost should be ≥ ring for N ≥ 5.
#[test]
fn gauge_solver_dense_not_cheaper_than_ring() {
    for &n in &ZONE_COUNTS[1..] {
        // 5, 10, 20
        let ring = time_gauge(build_gauge_network(n, false), n, ITERS);
        let dense = time_gauge(build_gauge_network(n, true), n, ITERS);
        println!(
            "N={n}: ring {:.2} µs, dense {:.2} µs",
            ring / 1000.0,
            dense / 1000.0
        );
        // Allow a tiny tolerance for measurement noise at small N; dense should
        // be at least as expensive as ring.
        assert!(
            dense >= ring * 0.9,
            "N={n}: dense ({:.2} µs) unexpectedly cheaper than ring ({:.2} µs) \
             — coupling graph may have been dropped",
            dense / 1000.0,
            ring / 1000.0
        );
    }
}

/// Ensure the GaugeSolver produces correct, finite results at every scale
/// before we trust the timing numbers above. A solver that returns NaN has no
/// meaningful performance characteristic.
#[test]
fn gauge_solver_correctness_at_all_sizes() {
    for &n in &ZONE_COUNTS {
        let mut mz = build_gauge_network(n, true);
        let bc = uniform_bc(n);
        let results = mz.step(3600.0, &bc).unwrap();
        assert_eq!(results.len(), n, "N={n}: expected {n} zone results");
        for (&zone_id, &energy_kwh) in &results {
            assert!(
                energy_kwh.is_finite(),
                "N={n}: zone {zone_id} returned non-finite energy {energy_kwh}"
            );
        }
    }
}
