//! Coverage tests for the parent module (extracted to keep the
//! ratcheted parent file under its Issue #2878/#3574 module-size
//! ceiling; child modules can see parent-private items).
//!
//! Coverage-expansion tests (PR #5): builder entry points
//! (`with_timestep`, `from_wall_spec*`, `boxed_from_wall_spec`),
//! `snapshot_temperatures` ordering, `build_per_surface_solver` /
//! `step_per_surface` integration, and the sky-aware dispatcher — none of
//! which had dedicated tests.

use super::*;
use fluxion_core::multi_node::ThermalMassNode;

fn make_solver() -> MultiNodeSolver {
    let wall = ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
    let roof = ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
    let floor = ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
    let internal = ThermalMassNode::new(20.0, 1e6, 10.0, 5.0);
    MultiNodeSolver::new(10.0, wall, roof, floor, internal)
}

fn concrete_wall_spec() -> WallSpec {
    WallSpec::single_layer("Concrete", 0.200, 1.73, 2300.0, 840.0)
}

#[test]
fn with_timestep_sets_timestep_seconds() {
    let solver = make_solver().with_timestep(600.0);
    assert!((solver.timestep_seconds - 600.0).abs() < 1e-12);
}

#[test]
fn from_wall_spec_partitions_mass_and_film_conductances() {
    // Closed-form expectations straight from the documented partition
    // table in `from_wall_spec`'s doc comment.
    let wall = concrete_wall_spec();
    let floor_area = 50.0;
    let solver = MultiNodeSolver::from_wall_spec(&wall, floor_area);

    let r_total = 0.200 / 1.73;
    let c_total = 2300.0 * 0.200 * 840.0;
    let r_si = 1.0 / 8.0;
    let r_se = 1.0 / 25.0;
    let h_tr_ms = 1.0 / (r_total / 2.0 + r_si);
    let h_tr_em = 1.0 / (r_total / 2.0 + r_se);

    assert!((solver.r_total - r_total).abs() < 1e-12);
    assert!(solver.initialized);
    assert!((solver.h_tr_is - 8.0).abs() < 1e-12);
    assert!((solver.r_se - r_se).abs() < 1e-12);

    // Capacitance fractions: wall 45%, roof 30%, floor 18%, internal 10%.
    assert!((solver.mass.wall.capacitance - 0.45 * c_total).abs() < 1e-6);
    assert!((solver.mass.roof.capacitance - 0.30 * c_total).abs() < 1e-6);
    assert!((solver.mass.floor.capacitance - 0.18 * c_total).abs() < 1e-6);
    assert!((solver.mass.internal.capacitance - 0.10 * c_total).abs() < 1e-6);

    // Symmetric centroid partition of the layer resistance.
    assert!((solver.mass.wall.h_tr_ms - h_tr_ms).abs() < 1e-12);
    assert!((solver.mass.wall.h_tr_em - h_tr_em).abs() < 1e-12);

    // Issue #1593: physics-based internal coupling h_tr_me = 9.1·0.5·A.
    assert!((solver.mass.internal.h_tr_me - 9.1 * 0.5 * floor_area).abs() < 1e-9);

    // Temperatures are seeded at 20 °C.
    assert!((solver.wall_temperature() - 20.0).abs() < 1e-12);
    assert!((solver.zone_temperature - 20.0).abs() < 1e-12);
    assert_eq!(solver.coupling_mode, MassAirCouplingMode::AdditiveSum);
}

#[test]
fn from_wall_spec_with_mode_sets_coupling_mode() {
    let solver = MultiNodeSolver::from_wall_spec_with_mode(
        &concrete_wall_spec(),
        50.0,
        MassAirCouplingMode::ParallelResistance,
    );
    assert_eq!(
        solver.coupling_mode,
        MassAirCouplingMode::ParallelResistance
    );
    assert!(solver.initialized);
}

#[test]
fn boxed_from_wall_spec_exposes_trait_contract() {
    let solver = MultiNodeSolver::boxed_from_wall_spec(&concrete_wall_spec(), 50.0);
    assert_eq!(solver.name(), "MultiNode9R4C");
    assert!(
        solver.is_valid(),
        "from_wall_spec must produce a valid HeatConductionSolver"
    );
}

#[test]
fn snapshot_temperatures_uses_canonical_node_order() {
    let mut solver = make_solver();
    solver.initialize_temperatures(15.5);
    let snap = solver.snapshot_temperatures();
    assert!(
        snap.iter().all(|t| (t - 15.5).abs() < 1e-12),
        "snapshot_temperatures should be all 15.5, got {snap:?}"
    );
    assert_eq!(MultiNodeSolver::NUM_NODES, 4);
    assert_eq!(
        MultiNodeSolver::NODE_NAMES,
        ["wall", "roof", "floor", "internal"]
    );
    // The snapshot order matches the individual accessors.
    assert!((snap[0] - solver.wall_temperature()).abs() < 1e-12);
    assert!((snap[1] - solver.roof_temperature()).abs() < 1e-12);
    assert!((snap[2] - solver.floor_temperature()).abs() < 1e-12);
    assert!((snap[3] - solver.internal_temperature()).abs() < 1e-12);
}

#[test]
fn step_per_surface_writes_conductance_weighted_average() {
    // With mass nodes at 20 °C, uniform exterior at 10 °C and no solar
    // gains, each per-surface temperature must lie strictly between the
    // exterior and mass temperatures (backward Euler, no overshoot), and
    // `self.surface_temperature` must be the documented h_tr_ms-weighted
    // average of the three per-surface temperatures.
    let mut solver = make_solver();
    solver.initialize_temperatures(20.0);

    let (t_w, t_r, t_f) = solver.step_per_surface(3600.0, (20.0, 20.0, 20.0), (0.0, 0.0, 0.0));

    for (name, t) in [("wall", t_w), ("roof", t_r), ("floor", t_f)] {
        assert!(
            (10.0..=20.0).contains(&t),
            "{name} surface temperature {t} outside [10, 20]"
        );
    }

    let h_w = solver.mass.wall.h_tr_ms;
    let h_r = solver.mass.roof.h_tr_ms;
    let h_f = solver.mass.floor.h_tr_ms;
    let expected = (h_w * t_w + h_r * t_r + h_f * t_f) / (h_w + h_r + h_f);
    assert!(
        (solver.surface_temperature - expected).abs() < 1e-9,
        "surface_temperature={} vs weighted average {expected}",
        solver.surface_temperature
    );
}

#[test]
fn step_per_surface_with_gains_raises_surface_temperatures() {
    // Positive solar gains to the surface nodes must raise the
    // per-surface temperatures relative to the no-gain case.
    let mut base = make_solver();
    base.initialize_temperatures(20.0);
    let (bw, br, bf) = base.step_per_surface(3600.0, (20.0, 20.0, 20.0), (0.0, 0.0, 0.0));

    let mut gained = make_solver();
    gained.initialize_temperatures(20.0);
    let (gw, gr, gf) = gained.step_per_surface(3600.0, (20.0, 20.0, 20.0), (500.0, 500.0, 500.0));

    assert!(gw > bw, "wall: {gw} should exceed no-gain {bw}");
    assert!(gr > br, "roof: {gr} should exceed no-gain {br}");
    assert!(gf > bf, "floor: {gf} should exceed no-gain {bf}");
}

#[test]
fn sky_dispatcher_with_zero_sky_conductance_matches_base() {
    // The `compute_zone_air_temperature_with_sky` dispatcher must reduce
    // exactly to `compute_zone_air_temperature` when h_rad_sky == 0 —
    // for both coupling modes.
    for mode in [
        MassAirCouplingMode::AdditiveSum,
        MassAirCouplingMode::ParallelResistance,
    ] {
        let solver = make_solver().with_coupling_mode(mode);
        let base = solver.compute_zone_air_temperature(5.0, 50.0, 0.0, 100.0);
        let with_sky =
            solver.compute_zone_air_temperature_with_sky(5.0, 50.0, 0.0, 100.0, -10.0, 0.0);
        assert!(
            (with_sky - base).abs() < 1e-12,
            "mode {mode:?}: dispatcher {with_sky} != base {base}"
        );
        assert!(with_sky.is_finite());
    }
}

#[test]
fn sky_dispatcher_cold_sky_lowers_air_temperature() {
    // Genuine physics through the dispatcher: a cold sky with a
    // positive sky conductance must cool the free-floating air node.
    let solver = make_solver().with_coupling_mode(MassAirCouplingMode::AdditiveSum);
    let no_sky = solver.compute_zone_air_temperature_with_sky(5.0, 50.0, 0.0, 0.0, 5.0, 0.0);
    let cold_sky = solver.compute_zone_air_temperature_with_sky(5.0, 50.0, 0.0, 0.0, -20.0, 4.0);
    assert!(
        cold_sky < no_sky,
        "cold sky should lower T_air: {cold_sky} vs {no_sky}"
    );
}
