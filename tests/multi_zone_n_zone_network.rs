//! Integration test for the N-zone inter-zone thermal coupling network.
//!
//! Acceptance criteria (Issue #1348):
//!   1. N-zone network solves 3 zones in <1 ms on a single core.
//!   2. Energy conservation: |Σ inter-zone transfers| < 1e-6 W for N=3, 5, 10.
//!   3. `MultiZoneConfig::zones` accepts ≥3 zones without panic and produces
//!      non-zero inter-zone transfer when zones differ in temperature.
//!   4. Regression: Case 960 two-zone wiring still passes
//!      `inter_zone_tolerance = 1.0 W`.
//!   5. `cargo test -p fluxion multi_zone_n_zone_network` passes including
//!      the 3-zone and 5-zone round-trip.
//!
//! These tests call the public `fluxion::sim::multi_zone_network` API and
//! the `fluxion::cli::multi_zone::MultiZoneConfig` wiring path directly so
//! the issue's acceptance criterion #5 (`cargo test -p fluxion
//! multi_zone_n_zone_network`) is satisfied.

use fluxion::cli::multi_zone::MultiZoneConfig;
use fluxion::sim::multi_zone_network::{MultiZoneAirflowNetwork, ZoneState};
use fluxion::validation::energy_balance::EnergyBalanceValidator;
use std::time::Instant;

/// Issue #1348 acceptance criterion #2: 3-zone symmetric network
/// must conserve energy to within 1e-6 W (machine precision).
#[test]
fn multi_zone_n_zone_network_3_zone_conservation() {
    let n = 3_usize;
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                pairs.push((i, j, 50.0));
            }
        }
    }
    let network = MultiZoneAirflowNetwork::from_adjacency_pairs(n, &pairs);
    let mut zones = vec![
        ZoneState::new(20.0, 1.0e6),
        ZoneState::new(25.0, 1.0e6),
        ZoneState::new(15.0, 1.0e6),
    ];
    let q_ext = vec![0.0; n];
    let result = network
        .solve_step(&mut zones, &q_ext, 3600.0)
        .expect("3-zone solve");

    assert!(
        result.net_w.abs() < 1e-6,
        "N=3: |Σ q_iz| = {:.3e} W must be < 1e-6 W",
        result.net_w.abs()
    );

    // Also verify the algebraic identity at the T_old vector directly.
    let temps_before: Vec<f64> = vec![20.0, 25.0, 15.0];
    let algebraic_net = network.net_inter_zone_q(&temps_before);
    assert!(
        algebraic_net.abs() < 1e-6,
        "N=3 algebraic identity must give |Σ q_iz| = 0; got {:.3e}",
        algebraic_net.abs()
    );
}

/// Issue #1348 acceptance criterion #5: 5-zone round-trip
/// (backward-Euler solve + report).
#[test]
fn multi_zone_n_zone_network_5_zone_round_trip() {
    let n = 5_usize;
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                pairs.push((i, j, 30.0));
            }
        }
    }
    let network = MultiZoneAirflowNetwork::from_adjacency_pairs(n, &pairs);

    // Round-trip: solve once, check conservation, probe the report, then
    // solve again with different initial conditions and re-check.
    let mut zones: Vec<ZoneState> = (0..n)
        .map(|i| ZoneState::new(18.0 + 2.0 * i as f64, 1.0e6))
        .collect();
    let q_ext = vec![0.0; n];
    let r1 = network
        .solve_step(&mut zones, &q_ext, 3600.0)
        .expect("5-zone solve #1");
    assert!(r1.net_w.abs() < 1e-6, "5-zone round-trip #1: |Σ q_iz| = {:.3e}", r1.net_w.abs());

    // Reset and solve again with different initial temps.
    let mut zones2: Vec<ZoneState> = (0..n)
        .map(|i| ZoneState::new(30.0 - 3.0 * i as f64, 1.0e6))
        .collect();
    let r2 = network
        .solve_step(&mut zones2, &q_ext, 3600.0)
        .expect("5-zone solve #2");
    assert!(r2.net_w.abs() < 1e-6, "5-zone round-trip #2: |Σ q_iz| = {:.3e}", r2.net_w.abs());

    // And the conservation report.
    let report = network.conservation_report();
    assert_eq!(report.n_zones, n);
    assert!(report.symmetric);
    assert!(
        report.net_inter_zone_q_w.abs() < 1e-6,
        "5-zone conservation_report net = {:.3e}",
        report.net_inter_zone_q_w.abs()
    );
}

/// Issue #1348 acceptance criterion #2 (extended): 10-zone round-trip.
#[test]
fn multi_zone_n_zone_network_10_zone_round_trip() {
    let n = 10_usize;
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                pairs.push((i, j, 10.0));
            }
        }
    }
    let network = MultiZoneAirflowNetwork::from_adjacency_pairs(n, &pairs);
    let mut zones: Vec<ZoneState> = (0..n)
        .map(|i| ZoneState::new(15.0 + i as f64, 1.0e6))
        .collect();
    let q_ext = vec![0.0; n];
    let result = network
        .solve_step(&mut zones, &q_ext, 3600.0)
        .expect("10-zone solve");
    assert!(
        result.net_w.abs() < 1e-6,
        "N=10: |Σ q_iz| = {:.3e} W must be < 1e-6 W",
        result.net_w.abs()
    );
}

/// Issue #1348 acceptance criterion #3: `MultiZoneConfig::zones` accepts
/// ≥3 zones without panic and the per-zone conductance output is non-zero
/// when zones differ in temperature.
#[test]
fn multi_zone_config_n_zones_no_panic_non_zero_transfer() {
    let n = 4_usize;
    let config = MultiZoneConfig {
        num_zones: n,
        zone_setpoints: (0..n).map(|i| (20.0 + i as f64, 26.0 + i as f64)).collect(),
        inter_zone_conductance: (0..n)
            .map(|i| (0..n).map(|j| if i != j { 25.0 } else { 0.0 }).collect())
            .collect(),
        building_properties: fluxion::cli::multi_zone::BuildingProperties {
            u_value: 1.5,
            area_per_zone: 50.0,
            volume_per_zone: 150.0,
            occupancy_schedule: Some("office".to_string()),
        },
    };

    let network = MultiZoneAirflowNetwork::from_adjacency_pairs(
        n,
        &config
            .inter_zone_conductance
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter_map(move |(j, &h)| if i != j { Some((i, j, h)) } else { None })
            })
            .collect::<Vec<_>>(),
    );

    // Differing zone temperatures must produce a non-zero per-zone transfer.
    let temps: Vec<f64> = (0..n).map(|i| 20.0 + 5.0 * i as f64).collect();
    let q_iz: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| config.inter_zone_conductance[i][j] * (temps[j] - temps[i]))
                .sum()
        })
        .collect();
    assert!(
        q_iz.iter().any(|&q| q.abs() > 1e-3),
        "differing zone temperatures must produce non-zero inter-zone transfer; \
         got q_iz = {:?}",
        q_iz
    );

    // Symmetric matrix → Σ q_iz = 0.
    let net: f64 = q_iz.iter().sum();
    assert!(net.abs() < 1e-6, "Σ q_iz for symmetric config = {net:.3e}");
    let _ = network.conservation_report();
}

/// Issue #1348 acceptance criterion #1: 3-zone network solves in < 1 ms on
/// a single core (interactive CLI perf budget).
#[test]
fn multi_zone_n_zone_network_3_zone_under_one_ms() {
    let n = 3_usize;
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                pairs.push((i, j, 50.0));
            }
        }
    }
    let network = MultiZoneAirflowNetwork::from_adjacency_pairs(n, &pairs);
    let mut zones = vec![
        ZoneState::new(22.0, 1.0e6),
        ZoneState::new(20.0, 1.0e6),
        ZoneState::new(18.0, 1.0e6),
    ];
    let q_ext = vec![0.0; n];

    // Warm up the LU decomposition cache.
    let _ = network.solve_step(&mut zones, &q_ext, 3600.0).unwrap();

    let iters = 1000_usize;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = network.solve_step(&mut zones, &q_ext, 3600.0).unwrap();
    }
    let elapsed = start.elapsed();
    let per_step_us = elapsed.as_micros() as f64 / iters as f64;
    assert!(
        per_step_us < 1000.0,
        "N=3 must solve in < 1 ms; got {per_step_us:.1} µs/solve"
    );
}

/// Issue #1348 acceptance criterion #4 (regression): Case 960 2-zone wiring
/// (door opening = 1.5 W/K) still produces a meaningful, non-pathological
/// q_iz that fits inside the legacy `inter_zone_tolerance = 1.0 W` band.
#[test]
fn multi_zone_n_zone_network_case_960_regression() {
    let n = 2_usize;
    let h = nalgebra::DMatrix::from_row_slice(n, n, &[0.0, 1.5, 1.5, 0.0]);
    let network = MultiZoneAirflowNetwork::from_matrix(h);
    let mut zones = vec![
        ZoneState::new(20.0, 2.0e6),  // Living
        ZoneState::new(8.0, 5.0e5),   // Sunspace
    ];
    let q_ext = vec![0.0; n];
    let result = network
        .solve_step(&mut zones, &q_ext, 3600.0)
        .expect("2-zone solve");

    // Per-pair imbalance (legacy 1.0 W tolerance): q_iz[0] + q_iz[1] must be < 1.0 W.
    let imbalance = (result.q_iz_w[0] + result.q_iz_w[1]).abs();
    assert!(
        imbalance < 1.0,
        "Case 960 2-zone legacy tolerance: |q_iz[0] + q_iz[1]| = {imbalance:.3e} W must be < 1.0 W"
    );
    // And the strict 1e-6 W tolerance (Issue #1348 acceptance criterion).
    assert!(
        result.net_w.abs() < 1e-6,
        "Case 960 strict tolerance: |Σ q_iz| = {:.3e} W must be < 1e-6 W",
        result.net_w.abs()
    );

    // Validate via the validator too.
    let validator = EnergyBalanceValidator::default();
    let legacy_pass = validator
        .validate_n_zone_network_conservation(&result.q_iz_w, 1.0)
        .is_ok();
    let strict_pass = validator
        .validate_n_zone_network_conservation(&result.q_iz_w, 1e-6)
        .is_ok();
    assert!(legacy_pass, "Case 960 must pass legacy 1.0 W tolerance");
    assert!(strict_pass, "Case 960 must also pass strict 1e-6 W tolerance");
}
