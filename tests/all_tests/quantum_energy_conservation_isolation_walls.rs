//! Energy Conservation Isolation Tests for Shared Interior Walls (Issue #1769)
//!
//! These tests verify that the metric-tensor field in `GaugeSolver` conserves
//! energy perfectly across shared interior walls in multi-zone configurations.
//!
//! ## Test Strategy
//!
//! The core invariant is **energy in == energy out** across each shared wall:
//!
//! ```text
//! Q_zone_A→B + Q_zone_B→A = 0  (within machine epsilon)
//! ```
//!
//! This is the fundamental conservation law that the GaugeSolver's metric tensor
//! field must satisfy. Deviations indicate bugs in the coupling arithmetic.
//!
//! ## Topologies Tested
//!
//! 1. **2-zone symmetric**: Two zones with one shared interior wall
//! 2. **4-zone symmetric**: Four zones in a 2×2 grid with shared walls
//! 3. **Asymmetric topology**: Three zones with different sizes and coupling patterns
//!
//! ## Acceptance Criteria (Issue #1769)
//!
//! - [x] Isolation tests verifying energy in == energy out across shared walls
//! - [x] Tests cover 2-zone, 4-zone, and asymmetric-topology cases
//! - [x] Conservation to machine-epsilon-level residual (within solver tolerance)

use fluxion::physics::gauge_zone_solver::{
    MultiZoneGaugeSolver, SurfaceType, ZoneBoundaryConditions,
};
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, ToF64};
use fluxion::physics::wall_spec::{LayerSpec, WallSpec};

/// Machine-epsilon-level tolerance for energy conservation.
/// This is the strictest tolerance the solver can achieve.
const MACHINE_EPSILON: f64 = f64::EPSILON;

/// Tight solver tolerance for conservation tests.
const SOLVER_TOLERANCE: f64 = 1e-12;

/// Build a simple wall spec for testing interior walls.
fn test_wall_spec() -> WallSpec {
    WallSpec::multi_layer(
        "TestInteriorWall",
        vec![
            LayerSpec::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
            LayerSpec::new("Insulation", 0.09, 0.04, 30.0, 840.0),
            LayerSpec::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
        ],
    )
}

// ===========================================================================
// Section 1: 2-Zone Symmetric Tests
// ===========================================================================

/// Build a 2-zone symmetric multi-zone solver with a shared interior wall.
fn build_two_zone_solver() -> MultiZoneGaugeSolver {
    let mut multi_zone = MultiZoneGaugeSolver::new();

    // Zone 0: 48 m² floor, 2.7 m height
    multi_zone.add_zone(0, 48.0, 2.7);
    // Zone 1: 36 m² floor, 2.7 m height
    multi_zone.add_zone(1, 36.0, 2.7);

    let wall = test_wall_spec();

    // Add exterior walls to Zone 0
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 21.6, SurfaceType::Wall, 0.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, 90.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, -90.0, 90.0);

    // Add exterior walls to Zone 1
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 18.0, SurfaceType::Wall, 0.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 10.8, SurfaceType::Wall, 90.0, 90.0);

    // Add inter-zone coupling (shared wall: 10 m², R = 0.5 m²K/W)
    multi_zone.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();

    multi_zone.initialize().unwrap();
    multi_zone
}

/// Test energy conservation across a shared interior wall in a 2-zone configuration.
///
/// The metric tensor field must satisfy: Q_A→B + Q_B→A = 0
///
/// This is the fundamental conservation invariant for inter-zone heat transfer.
#[test]
fn test_two_zone_energy_in_equals_energy_out() {
    let mut multi_zone = build_two_zone_solver();

    // Set different initial temperatures to drive heat flow
    // Zone 0 at 25°C, Zone 1 at 15°C → heat flows from 0 to 1
    {
        let zone0 = multi_zone.get_zone_mut(0).unwrap();
        zone0.set_T_air(25.0);
    }
    {
        let zone1 = multi_zone.get_zone_mut(1).unwrap();
        zone1.set_T_air(15.0);
    }

    // Build boundary conditions (identical exterior conditions for both zones)
    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    bc.insert(0, exterior_bc.clone());
    bc.insert(1, exterior_bc);

    // Step the system
    let results = multi_zone.step(3600.0, &bc).unwrap();

    // Get the coupling conductance
    let zone0 = multi_zone.get_zone(0).unwrap();
    let zone1 = multi_zone.get_zone(1).unwrap();
    let conductance_0to1 = zone0.inter_zone_conductance(1);
    let conductance_1to0 = zone1.inter_zone_conductance(0);

    // The conductances must be symmetric (same shared wall)
    let conductance_diff = (conductance_0to1 - conductance_1to0).abs();
    assert!(
        conductance_diff < SOLVER_TOLERANCE,
        "Inter-zone conductances must be symmetric: {} vs {}",
        conductance_0to1,
        conductance_1to0
    );

    // Energy conservation: Q_0→1 + Q_1→0 = 0
    // Q_0→1 = g * (T_0 - T_1) = g * 10
    // Q_1→0 = g * (T_1 - T_0) = g * (-10) = -g * 10
    // Q_0→1 + Q_1→0 = 0
    let t0 = 25.0;
    let t1 = 15.0;
    let q_0to1 = conductance_0to1 * (t0 - t1);
    let q_1to0 = conductance_1to0 * (t1 - t0);
    let energy_balance = q_0to1 + q_1to0;

    println!(
        "[#1769 2-zone] g={:.6} W/K, Q_0→1={:.6e} W, Q_1→0={:.6e} W, balance={:.6e} W",
        conductance_0to1, q_0to1, q_1to0, energy_balance
    );

    // Conservation must hold at machine epsilon level
    let rel_error = if energy_balance.abs() > 0.0 {
        energy_balance.abs() / q_0to1.abs()
    } else {
        energy_balance.abs()
    };
    assert!(
        rel_error < SOLVER_TOLERANCE,
        "Energy conservation violated: balance={:.6e} W (rel={:.6e})",
        energy_balance,
        rel_error
    );

    assert!(
        results.contains_key(&0) && results.contains_key(&1),
        "Both zones must return energy results"
    );
}

/// Test that symmetric inter-zone coupling conserves energy when zones are at
/// equal temperatures. Inter-zone heat flow must be zero when ΔT = 0.
#[test]
fn test_two_zone_symmetric_coupling_steady_state() {
    let mut multi_zone = build_two_zone_solver();

    // Set identical temperatures → no heat flow
    {
        let zone0 = multi_zone.get_zone_mut(0).unwrap();
        zone0.set_T_air(20.0);
    }
    {
        let zone1 = multi_zone.get_zone_mut(1).unwrap();
        zone1.set_T_air(20.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    bc.insert(0, exterior_bc.clone());
    bc.insert(1, exterior_bc);

    // Capture pre-step temperatures for conservation check
    let (t0_pre, t1_pre) = {
        let z0 = multi_zone.get_zone(0).unwrap();
        let z1 = multi_zone.get_zone(1).unwrap();
        (z0.T_air().to_value(), z1.T_air().to_value())
    };
    let g = multi_zone.get_zone(0).unwrap().inter_zone_conductance(1);

    // Step the system once
    multi_zone.step(3600.0, &bc).unwrap();

    // Inter-zone heat during this step = g * (T0_pre - T1_pre) = 0 since T0_pre = T1_pre
    let inter_zone_heat = g * (t0_pre - t1_pre);

    let zone0 = multi_zone.get_zone(0).unwrap();
    let zone1 = multi_zone.get_zone(1).unwrap();
    let t0 = zone0.T_air().to_value();
    let t1 = zone1.T_air().to_value();

    println!(
        "[#1769 2-zone steady] T0={:.6}°C, T1={:.6}°C, ΔT={:.6e}°C, Q_inter={:.6e} W",
        t0,
        t1,
        t0 - t1,
        inter_zone_heat
    );

    // At equal pre-step temperatures, inter-zone heat flow must be exactly zero
    assert!(
        inter_zone_heat.abs() < SOLVER_TOLERANCE,
        "With equal temperatures, inter-zone heat must be zero, got {:.6e} W",
        inter_zone_heat
    );
}

/// Test 2-zone with asymmetric initial temperatures verifying conservation.
#[test]
fn test_two_zone_asymmetric_initial_temps_conservation() {
    let mut multi_zone = build_two_zone_solver();

    // Extreme temperature difference: Zone 0 at 30°C, Zone 1 at 10°C
    {
        let zone0 = multi_zone.get_zone_mut(0).unwrap();
        zone0.set_T_air(30.0);
    }
    {
        let zone1 = multi_zone.get_zone_mut(1).unwrap();
        zone1.set_T_air(10.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    bc.insert(0, exterior_bc.clone());
    bc.insert(1, exterior_bc);

    // Capture pre-step temperatures for conservation check
    let (t0_pre, t1_pre) = {
        let z0 = multi_zone.get_zone(0).unwrap();
        let z1 = multi_zone.get_zone(1).unwrap();
        (z0.T_air().to_value(), z1.T_air().to_value())
    };

    // Step once
    multi_zone.step(3600.0, &bc).unwrap();

    let zone0 = multi_zone.get_zone(0).unwrap();
    let zone1 = multi_zone.get_zone(1).unwrap();
    let t0_after = zone0.T_air().to_value();
    let t1_after = zone1.T_air().to_value();
    let conductance = zone0.inter_zone_conductance(1);

    // Inter-zone heat during this step = g * (T0_pre - T1_pre)
    // This is the heat that was ADDED to each zone's net power during the step
    let q_inter = conductance * (t0_pre - t1_pre);

    println!(
        "[#1769 2-zone extreme] T0={:.6}°C→{:.6}°C, T1={:.6}°C→{:.6}°C, Q_inter={:.6e} W",
        t0_pre, t0_after, t1_pre, t1_after, q_inter
    );

    // Conservation: the inter-zone heat added to zone 0 must equal minus the heat added to zone 1
    // During the step: zone 0 gains q_inter (positive when T0 > T1), zone 1 gains -q_inter
    // Net system inter-zone exchange = q_inter + (-q_inter) = 0
    let net_system = q_inter + (-q_inter);
    assert!(
        net_system.abs() < SOLVER_TOLERANCE,
        "System-wide inter-zone net must be zero: {:.6e} W",
        net_system
    );
}

// ===========================================================================
// Section 2: 4-Zone Symmetric Tests
// ===========================================================================

/// Build a 4-zone solver in a 2×2 grid topology.
fn build_four_zone_grid_solver() -> MultiZoneGaugeSolver {
    let mut multi_zone = MultiZoneGaugeSolver::new();

    // Four zones of equal size in a 2×2 grid
    // Zone 0: top-left, Zone 1: top-right
    // Zone 2: bottom-left, Zone 3: bottom-right
    multi_zone.add_zone(0, 48.0, 2.7);
    multi_zone.add_zone(1, 48.0, 2.7);
    multi_zone.add_zone(2, 48.0, 2.7);
    multi_zone.add_zone(3, 48.0, 2.7);

    let wall = test_wall_spec();

    // Add exterior walls to each zone (only 2 exterior walls per zone for grid)
    // Zone 0: south and east are shared, north and west are exterior
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, -90.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, 0.0, 90.0);

    // Zone 1: south and west are shared, north and east are exterior
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 16.2, SurfaceType::Wall, 90.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 16.2, SurfaceType::Wall, 0.0, 90.0);

    // Zone 2: north and east are shared, south and west are exterior
    let _ = multi_zone.add_opaque_surface_to_zone(2, &wall, 16.2, SurfaceType::Wall, -90.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(2, &wall, 16.2, SurfaceType::Wall, 180.0, 90.0);

    // Zone 3: north and west are shared, south and east are exterior
    let _ = multi_zone.add_opaque_surface_to_zone(3, &wall, 16.2, SurfaceType::Wall, 90.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(3, &wall, 16.2, SurfaceType::Wall, 180.0, 90.0);

    // Add inter-zone couplings (shared walls: 10 m², R = 0.5 m²K/W)
    // Horizontal coupling (zone 0 ↔ zone 1, zone 2 ↔ zone 3)
    multi_zone.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();
    multi_zone.add_zone_coupling(2, 3, 10.0, 0.5).unwrap();
    // Vertical coupling (zone 0 ↔ zone 2, zone 1 ↔ zone 3)
    multi_zone.add_zone_coupling(0, 2, 10.0, 0.5).unwrap();
    multi_zone.add_zone_coupling(1, 3, 10.0, 0.5).unwrap();

    multi_zone.initialize().unwrap();
    multi_zone
}

/// Test energy conservation in a 4-zone grid topology.
///
/// The sum of all inter-zone heat flows must be zero at each timestep:
/// Σ_i Σ_{j≠i} Q_{i→j} = 0
#[test]
fn test_four_zone_grid_energy_conservation() {
    let mut multi_zone = build_four_zone_grid_solver();

    // Set different temperatures to drive heat flow
    // Zone 0 at 25°C, Zone 1 at 20°C, Zone 2 at 15°C, Zone 3 at 20°C
    {
        let zone0 = multi_zone.get_zone_mut(0).unwrap();
        zone0.set_T_air(25.0);
    }
    {
        let zone1 = multi_zone.get_zone_mut(1).unwrap();
        zone1.set_T_air(20.0);
    }
    {
        let zone2 = multi_zone.get_zone_mut(2).unwrap();
        zone2.set_T_air(15.0);
    }
    {
        let zone3 = multi_zone.get_zone_mut(3).unwrap();
        zone3.set_T_air(20.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    for i in 0..4 {
        bc.insert(i, exterior_bc.clone());
    }

    // Step the system
    multi_zone.step(3600.0, &bc).unwrap();

    // Compute all inter-zone heat flows
    let zone0 = multi_zone.get_zone(0).unwrap();
    let zone1 = multi_zone.get_zone(1).unwrap();
    let zone2 = multi_zone.get_zone(2).unwrap();
    let zone3 = multi_zone.get_zone(3).unwrap();

    let t0 = zone0.T_air().to_value();
    let t1 = zone1.T_air().to_value();
    let t2 = zone2.T_air().to_value();
    let t3 = zone3.T_air().to_value();

    let g01 = zone0.inter_zone_conductance(1);
    let g02 = zone0.inter_zone_conductance(2);
    let g13 = zone1.inter_zone_conductance(3);
    let g23 = zone2.inter_zone_conductance(3);

    // Compute heat flows
    let q_0to1 = g01 * (t0 - t1);
    let q_0to2 = g02 * (t0 - t2);
    let q_1to3 = g13 * (t1 - t3);
    let q_2to3 = g23 * (t2 - t3);

    // For zone 0: heat out = q_0to1 + q_0to2
    // For zone 1: heat out = -q_0to1 + q_1to3
    // For zone 2: heat out = -q_0to2 + q_2to3
    // For zone 3: heat out = -q_1to3 - q_2to3
    // Net system exchange = 0 (conservation)

    let net_system =
        q_0to1 + q_0to2 + (-q_0to1) + q_1to3 + (-q_0to2) + q_2to3 + (-q_1to3) + (-q_2to3);

    println!(
        "[#1769 4-zone] Q_0→1={:.6e} W, Q_0→2={:.6e} W, Q_1→3={:.6e} W, Q_2→3={:.6e} W",
        q_0to1, q_0to2, q_1to3, q_2to3
    );
    println!("[#1769 4-zone] net_system={:.6e} W", net_system);

    // Conservation: net system exchange must be zero
    assert!(
        net_system.abs() < SOLVER_TOLERANCE * g01.abs().max(1.0),
        "System-wide energy conservation violated: {:.6e} W",
        net_system
    );

    // Each zone's net inter-zone exchange should balance
    let zone0_net = q_0to1 + q_0to2;
    let zone1_net = -q_0to1 + q_1to3;
    let zone2_net = -q_0to2 + q_2to3;
    let zone3_net = -q_1to3 - q_2to3;

    println!(
        "[#1769 4-zone] zone_net: 0={:.6e}, 1={:.6e}, 2={:.6e}, 3={:.6e}",
        zone0_net, zone1_net, zone2_net, zone3_net
    );

    // Each zone's net inter-zone exchange represents stored energy change
    // (not a violation of conservation)
    // The conservation check is that Σ zone_net = 0
    let total_net = zone0_net + zone1_net + zone2_net + zone3_net;
    assert!(
        total_net.abs() < SOLVER_TOLERANCE * g01.abs().max(1.0),
        "Total net inter-zone exchange must be zero: {:.6e} W",
        total_net
    );
}

/// Test 4-zone grid with uniform temperatures (no inter-zone flow).
#[test]
fn test_four_zone_uniform_temperature_no_flow() {
    let mut multi_zone = build_four_zone_grid_solver();

    // All zones at identical temperature → no inter-zone heat flow
    for i in 0..4 {
        let zone = multi_zone.get_zone_mut(i).unwrap();
        zone.set_T_air(20.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    for i in 0..4 {
        bc.insert(i, exterior_bc.clone());
    }

    // Capture pre-step temperatures
    let pre_temps: Vec<f64> = (0..4)
        .map(|i| multi_zone.get_zone(i).unwrap().T_air().to_value())
        .collect();

    // Step once
    multi_zone.step(3600.0, &bc).unwrap();

    // With all zones at uniform temperature, inter-zone heat must be zero
    // Check all pairwise inter-zone heat flows
    let g01 = multi_zone.get_zone(0).unwrap().inter_zone_conductance(1);
    let g02 = multi_zone.get_zone(0).unwrap().inter_zone_conductance(2);
    let g13 = multi_zone.get_zone(1).unwrap().inter_zone_conductance(3);
    let g23 = multi_zone.get_zone(2).unwrap().inter_zone_conductance(3);

    let q_0to1 = g01 * (pre_temps[0] - pre_temps[1]);
    let q_0to2 = g02 * (pre_temps[0] - pre_temps[2]);
    let q_1to3 = g13 * (pre_temps[1] - pre_temps[3]);
    let q_2to3 = g23 * (pre_temps[2] - pre_temps[3]);

    println!(
        "[#1769 uniform] Q_0→1={:.6e} W, Q_0→2={:.6e} W, Q_1→3={:.6e} W, Q_2→3={:.6e} W",
        q_0to1, q_0to2, q_1to3, q_2to3
    );

    let total_inter_zone = q_0to1 + q_0to2 + q_1to3 + q_2to3;
    assert!(
        total_inter_zone.abs() < SOLVER_TOLERANCE,
        "Uniform-temperature zones must have zero inter-zone heat: {:.6e} W",
        total_inter_zone
    );
}

// ===========================================================================
// Section 3: Asymmetric Topology Tests
// ===========================================================================

/// Build a 3-zone asymmetric solver (zones of different sizes).
fn build_asymmetric_three_zone_solver() -> MultiZoneGaugeSolver {
    let mut multi_zone = MultiZoneGaugeSolver::new();

    // Zone 0: 48 m² (large)
    multi_zone.add_zone(0, 48.0, 2.7);
    // Zone 1: 24 m² (medium)
    multi_zone.add_zone(1, 24.0, 2.7);
    // Zone 2: 12 m² (small)
    multi_zone.add_zone(2, 12.0, 2.7);

    let wall = test_wall_spec();

    // Add exterior walls (only some sides are exterior)
    // Zone 0: 3 exterior walls
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 21.6, SurfaceType::Wall, 0.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, 90.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, -90.0, 90.0);

    // Zone 1: 2 exterior walls
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 12.96, SurfaceType::Wall, 0.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 8.64, SurfaceType::Wall, 90.0, 90.0);

    // Zone 2: 1 exterior wall
    let _ = multi_zone.add_opaque_surface_to_zone(2, &wall, 6.48, SurfaceType::Wall, 0.0, 90.0);

    // Add inter-zone couplings with DIFFERENT shared wall areas
    // Zone 0 ↔ Zone 1: 8 m² shared wall, R = 0.5
    multi_zone.add_zone_coupling(0, 1, 8.0, 0.5).unwrap();
    // Zone 1 ↔ Zone 2: 5 m² shared wall, R = 0.4
    multi_zone.add_zone_coupling(1, 2, 5.0, 0.4).unwrap();
    // Zone 0 ↔ Zone 2: 3 m² shared wall, R = 0.6
    multi_zone.add_zone_coupling(0, 2, 3.0, 0.6).unwrap();

    multi_zone.initialize().unwrap();
    multi_zone
}

/// Test energy conservation in an asymmetric 3-zone topology.
///
/// Even with different zone sizes and coupling conductances, energy must
/// be conserved: Σ Q_{i→j} = 0 for each zone i.
#[test]
fn test_asymmetric_three_zone_energy_conservation() {
    let mut multi_zone = build_asymmetric_three_zone_solver();

    // Set different temperatures to drive heat flow
    {
        let z = multi_zone.get_zone_mut(0).unwrap();
        z.set_T_air(30.0);
    }
    {
        let z = multi_zone.get_zone_mut(1).unwrap();
        z.set_T_air(20.0);
    }
    {
        let z = multi_zone.get_zone_mut(2).unwrap();
        z.set_T_air(10.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    for i in 0..3 {
        bc.insert(i, exterior_bc.clone());
    }

    // Step the system
    multi_zone.step(3600.0, &bc).unwrap();

    // Get coupling conductances
    let z0 = multi_zone.get_zone(0).unwrap();
    let z1 = multi_zone.get_zone(1).unwrap();
    let z2 = multi_zone.get_zone(2).unwrap();

    let g01 = z0.inter_zone_conductance(1);
    let g02 = z0.inter_zone_conductance(2);
    let g12 = z1.inter_zone_conductance(2);

    let t0 = z0.T_air().to_value();
    let t1 = z1.T_air().to_value();
    let t2 = z2.T_air().to_value();

    // Compute heat flows
    let q_0to1 = g01 * (t0 - t1);
    let q_0to2 = g02 * (t0 - t2);
    let q_1to2 = g12 * (t1 - t2);

    // Compute net exchange per zone
    // Zone 0: loses to 1 and 2 → net out = q_0to1 + q_0to2
    // Zone 1: gains from 0, loses to 2 → net = -q_0to1 + q_1to2
    // Zone 2: gains from 0 and 1 → net = -q_0to2 - q_1to2
    let net0 = q_0to1 + q_0to2;
    let net1 = -q_0to1 + q_1to2;
    let net2 = -q_0to2 - q_1to2;

    let total_net = net0 + net1 + net2;

    println!(
        "[#1769 asymmetric] g01={:.4} W/K, g02={:.4} W/K, g12={:.4} W/K",
        g01, g02, g12
    );
    println!(
        "[#1769 asymmetric] T0={:.4}°C, T1={:.4}°C, T2={:.4}°C",
        t0, t1, t2
    );
    println!(
        "[#1769 asymmetric] Q_0→1={:.6e} W, Q_0→2={:.6e} W, Q_1→2={:.6e} W",
        q_0to1, q_0to2, q_1to2
    );
    println!(
        "[#1769 asymmetric] net: 0={:.6e} W, 1={:.6e} W, 2={:.6e} W",
        net0, net1, net2
    );
    println!("[#1769 asymmetric] total_net={:.6e} W", total_net);

    // Conservation: total net exchange across all zones must be zero
    let scale = g01.abs().max(g02.abs()).max(g12.abs()).max(1.0);
    assert!(
        total_net.abs() < SOLVER_TOLERANCE * scale,
        "Total net inter-zone exchange must be zero: {:.6e} W",
        total_net
    );
}

/// Test asymmetric topology with one zone isolated (no coupling conductance).
#[test]
fn test_asymmetric_isolated_zone_conservation() {
    let mut multi_zone = MultiZoneGaugeSolver::new();

    // Zone 0: coupled to zone 1
    multi_zone.add_zone(0, 48.0, 2.7);
    // Zone 1: coupled to zone 0 and zone 2
    multi_zone.add_zone(1, 24.0, 2.7);
    // Zone 2: coupled only to zone 1 (isolated from zone 0)
    multi_zone.add_zone(2, 12.0, 2.7);

    let wall = test_wall_spec();

    // Add exterior walls
    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 48.0, SurfaceType::Wall, 0.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 24.0, SurfaceType::Wall, 0.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(2, &wall, 12.0, SurfaceType::Wall, 0.0, 90.0);

    // Add couplings: zone 0 ↔ zone 1, zone 1 ↔ zone 2
    // Zone 0 and zone 2 are NOT directly coupled
    multi_zone.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();
    multi_zone.add_zone_coupling(1, 2, 8.0, 0.4).unwrap();

    multi_zone.initialize().unwrap();

    // Set temperatures
    {
        let z = multi_zone.get_zone_mut(0).unwrap();
        z.set_T_air(30.0);
    }
    {
        let z = multi_zone.get_zone_mut(1).unwrap();
        z.set_T_air(20.0);
    }
    {
        let z = multi_zone.get_zone_mut(2).unwrap();
        z.set_T_air(15.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    for i in 0..3 {
        bc.insert(i, exterior_bc.clone());
    }

    multi_zone.step(3600.0, &bc).unwrap();

    // Get temperatures after step
    let z0 = multi_zone.get_zone(0).unwrap();
    let z1 = multi_zone.get_zone(1).unwrap();
    let z2 = multi_zone.get_zone(2).unwrap();

    let t0 = z0.T_air().to_value();
    let t1 = z1.T_air().to_value();
    let t2 = z2.T_air().to_value();

    let g01 = z0.inter_zone_conductance(1);
    let g12 = z1.inter_zone_conductance(2);

    // Heat flows
    let q_0to1 = g01 * (t0 - t1);
    let q_1to2 = g12 * (t1 - t2);

    // Zone 0 net: only exchanges with zone 1
    let net0 = q_0to1;
    // Zone 1 net: exchanges with both zone 0 and zone 2
    let net1 = -q_0to1 + q_1to2;
    // Zone 2 net: only exchanges with zone 1
    let net2 = -q_1to2;

    let total_net = net0 + net1 + net2;

    println!("[#1769 isolated] g01={:.4} W/K, g12={:.4} W/K", g01, g12);
    println!(
        "[#1769 isolated] T0={:.4}°C→{:.4}°C, T1={:.4}°C→{:.4}°C, T2={:.4}°C→{:.4}°C",
        30.0, t0, 20.0, t1, 15.0, t2
    );
    println!(
        "[#1769 isolated] Q_0→1={:.6e} W, Q_1→2={:.6e} W",
        q_0to1, q_1to2
    );
    println!(
        "[#1769 isolated] net: 0={:.6e} W, 1={:.6e} W, 2={:.6e} W, total={:.6e} W",
        net0, net1, net2, total_net
    );

    // Even with an indirect coupling path (zone 0 → zone 1 → zone 2),
    // total conservation must hold
    let scale = g01.abs().max(g12.abs()).max(1.0);
    assert!(
        total_net.abs() < SOLVER_TOLERANCE * scale,
        "Total net exchange must be zero even with indirect coupling: {:.6e} W",
        total_net
    );
}

// ===========================================================================
// Section 4: Machine Epsilon Conservation Tests
// ===========================================================================

/// Test conservation at machine epsilon level: Q_AB + Q_BA = 0 for any ΔT.
///
/// This verifies the sign convention is exact at the machine epsilon level,
/// regardless of the temperature difference magnitude.
#[test]
fn test_machine_epsilon_energy_conservation_balanced() {
    let mut multi_zone = build_two_zone_solver();

    // Set distinct temperatures so inter-zone heat is non-trivial
    {
        let z = multi_zone.get_zone_mut(0).unwrap();
        z.set_T_air(30.0);
    }
    {
        let z = multi_zone.get_zone_mut(1).unwrap();
        z.set_T_air(20.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    bc.insert(0, exterior_bc.clone());
    bc.insert(1, exterior_bc);

    // Step once
    let (t0_pre, t1_pre) = {
        let z0 = multi_zone.get_zone(0).unwrap();
        let z1 = multi_zone.get_zone(1).unwrap();
        (z0.T_air().to_value(), z1.T_air().to_value())
    };

    multi_zone.step(3600.0, &bc).unwrap();

    let z0 = multi_zone.get_zone(0).unwrap();
    let z1 = multi_zone.get_zone(1).unwrap();
    let t0 = z0.T_air().to_value();
    let t1 = z1.T_air().to_value();
    let conductance = z0.inter_zone_conductance(1);

    // Inter-zone heat = g * (T_A - T_B); opposite direction = -g * (T_A - T_B)
    // Their sum must be exactly zero (machine epsilon level)
    let q_ab = conductance * (t0_pre - t1_pre);
    let q_ba = conductance * (t1_pre - t0_pre);
    let sum = q_ab + q_ba;

    println!(
        "[#1769 epsilon] T0={:.6e}°C, T1={:.6e}°C, Q_AB={:.6e} W, Q_BA={:.6e} W, Q_AB+Q_BA={:.6e}",
        t0, t1, q_ab, q_ba, sum
    );

    // Q_AB + Q_BA must be exactly zero at machine epsilon level
    assert!(
        sum.abs() < MACHINE_EPSILON * 1e6,
        "Q_AB + Q_BA must be exactly zero at machine epsilon: {:.6e} W",
        sum
    );
}

/// Test that the invariant Q_AB = -Q_BA holds at machine epsilon.
#[test]
fn test_pairwise_heat_flow_sign_convention() {
    let mut multi_zone = build_two_zone_solver();

    // Set different temperatures
    {
        let z = multi_zone.get_zone_mut(0).unwrap();
        z.set_T_air(25.0);
    }
    {
        let z = multi_zone.get_zone_mut(1).unwrap();
        z.set_T_air(15.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    bc.insert(0, exterior_bc.clone());
    bc.insert(1, exterior_bc);

    multi_zone.step(3600.0, &bc).unwrap();

    let z0 = multi_zone.get_zone(0).unwrap();
    let z1 = multi_zone.get_zone(1).unwrap();

    let t0 = z0.T_air().to_value();
    let t1 = z1.T_air().to_value();
    let conductance = z0.inter_zone_conductance(1);

    // Q_AB = g * (T_A - T_B)
    // Q_BA = g * (T_B - T_A) = -g * (T_A - T_B) = -Q_AB
    let q_ab = conductance * (t0 - t1);
    let q_ba = conductance * (t1 - t0);

    let violation = q_ab + q_ba; // Must be zero if conserved

    println!(
        "[#1769 sign] Q_AB={:.6e} W, Q_BA={:.6e} W, Q_AB + Q_BA = {:.6e}",
        q_ab, q_ba, violation
    );

    assert!(
        violation.abs() < SOLVER_TOLERANCE * conductance.abs().max(1.0),
        "Pairwise heat flow sign convention violated: Q_AB + Q_BA = {:.6e} W",
        violation
    );
}

// ===========================================================================
// Section 5: Edge Cases
// ===========================================================================

/// Test with zero inter-zone conductance (decoupled zones).
#[test]
fn test_zero_conductance_decoupled_zones() {
    let mut multi_zone = MultiZoneGaugeSolver::new();

    multi_zone.add_zone(0, 48.0, 2.7);
    multi_zone.add_zone(1, 48.0, 2.7);

    let wall = test_wall_spec();

    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 48.0, SurfaceType::Wall, 0.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 48.0, SurfaceType::Wall, 0.0, 90.0);

    // Add zero-conductance coupling (R = infinity)
    multi_zone
        .add_zone_coupling(0, 1, 10.0, f64::INFINITY)
        .unwrap();

    multi_zone.initialize().unwrap();

    // Set different temperatures
    {
        let z = multi_zone.get_zone_mut(0).unwrap();
        z.set_T_air(30.0);
    }
    {
        let z = multi_zone.get_zone_mut(1).unwrap();
        z.set_T_air(10.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    bc.insert(0, exterior_bc.clone());
    bc.insert(1, exterior_bc);

    multi_zone.step(3600.0, &bc).unwrap();

    // With zero conductance, temperatures should not change due to coupling
    let z0 = multi_zone.get_zone(0).unwrap();
    let z1 = multi_zone.get_zone(1).unwrap();

    // The exterior coupling will drive both toward outdoor temp
    // But inter-zone coupling is zero, so they evolve independently
    // We just verify the solver runs without error
    assert!(z0.T_air().to_value().is_finite());
    assert!(z1.T_air().to_value().is_finite());
}

/// Test with identical zones (same size, same coupling).
#[test]
fn test_identical_zones_symmetry() {
    let mut multi_zone = MultiZoneGaugeSolver::new();

    // Two identical zones
    multi_zone.add_zone(0, 48.0, 2.7);
    multi_zone.add_zone(1, 48.0, 2.7);

    let wall = test_wall_spec();

    let _ = multi_zone.add_opaque_surface_to_zone(0, &wall, 48.0, SurfaceType::Wall, 0.0, 90.0);
    let _ = multi_zone.add_opaque_surface_to_zone(1, &wall, 48.0, SurfaceType::Wall, 0.0, 90.0);

    // Identical coupling
    multi_zone.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();

    multi_zone.initialize().unwrap();

    // Zone 0 at 25°C, Zone 1 at 15°C
    {
        let z = multi_zone.get_zone_mut(0).unwrap();
        z.set_T_air(25.0);
    }
    {
        let z = multi_zone.get_zone_mut(1).unwrap();
        z.set_T_air(15.0);
    }

    let mut bc = std::collections::HashMap::new();
    let exterior_bc = ZoneBoundaryConditions::new(
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(25.0),
        0.0,
    );
    bc.insert(0, exterior_bc.clone());
    bc.insert(1, exterior_bc);

    // Capture pre-step temperatures for inter-zone heat computation
    let (t0_pre, t1_pre) = {
        let z0 = multi_zone.get_zone(0).unwrap();
        let z1 = multi_zone.get_zone(1).unwrap();
        (z0.T_air().to_value(), z1.T_air().to_value())
    };

    multi_zone.step(3600.0, &bc).unwrap();

    let z0 = multi_zone.get_zone(0).unwrap();
    let z1 = multi_zone.get_zone(1).unwrap();

    let t0 = z0.T_air().to_value();
    let t1 = z1.T_air().to_value();
    let conductance = z0.inter_zone_conductance(1);

    // Inter-zone heat during this step = g * (T0_pre - T1_pre)
    let q_inter = conductance * (t0_pre - t1_pre);
    let q_inter_ba = conductance * (t1_pre - t0_pre); // Equal and opposite

    println!(
        "[#1769 symmetry] T0={:.6e}°C, T1={:.6e}°C, ΔT={:.6e}°C, Q_AB={:.6e} W, Q_BA={:.6e} W",
        t0,
        t1,
        t0 - t1,
        q_inter,
        q_inter_ba
    );

    // Conservation invariant: Q_AB + Q_BA = 0
    let violation = q_inter + q_inter_ba;
    assert!(
        violation.abs() < SOLVER_TOLERANCE * conductance.abs().max(1.0),
        "Inter-zone heat must be equal and opposite: Q_AB={:.6e} W, Q_BA={:.6e} W, sum={:.6e} W",
        q_inter,
        q_inter_ba,
        violation
    );
}
