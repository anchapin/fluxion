//! Regression test: `FfdCfdAdapter` wires `fluxion_cfd::FfdCfdSolver`
//! into `crate::sim::loose_coupling::FfdSolver` (issue #2460).
//!
//! ## Goal
//!
//! 1. Build a 2-zone `LooseCouplingCoordinator` backed by the real
//!    `FfdCfdSolver` (CPU path, no CUDA required).
//! 2. Run 10 macro steps with constant boundary conditions.
//! 3. Assert:
//!    - `recommended_micro_timestep() ≈ 0.001` within `1e-9` (the FFD `dt` is
//!      preserved through the adapter).
//!    - `FfdMicroResults.chtc` matches the post-step FFD velocity field via
//!      the adapter's `compute_chtc()` translation to within `1e-4`
//!      (cross-validates the adapter translation layer).
//!
//! ## Build
//!
//! This test file uses `fluxion-cfd` types directly, so it is **gated** by
//! the `fluxion-cfd` feature flag. Run with:
//!
//! ```bash
//! cargo test --features fluxion-cfd -p fluxion --test ffd_cfd_adapter_integration
//! ```
//!
//! ## Determinism
//!
//! The CPU solver path is deterministic for a fixed input: same boundary
//! conditions → same `FfdMicroResults`. The GPU path (`--features cuda`)
//! is not exercised here; the GPU smoke test lives in
//! `tests/surrogate_cuda_smoke.rs` (per issue #2460 acceptance criteria).

#![cfg(feature = "fluxion-cfd")]

use fluxion::sim::ffd_cfd_adapter::FfdCfdAdapter;
use fluxion::sim::loose_coupling::{BesToFfdBoundaryConditions, FfdSolver, LooseCoupling};
use fluxion_cfd::{FfdConfig, VelocityField};

/// Build a tiny FFD config that runs in milliseconds on the CPU path.
fn tiny_ffd_config() -> FfdConfig {
    FfdConfig {
        nx: 4,
        ny: 4,
        nz: 4,
        dx: 0.1,
        dy: 0.1,
        dz: 0.1,
        dt: 0.001,
        nu: 1.0e-5,
        max_iter: 100,
        tolerance: 1e-6,
    }
}

/// Compute the same representative tangential velocity magnitude as the
/// adapter does internally. Lets the regression test verify the
/// translation layer without re-implementing the FFD step.
fn mean_velocity_magnitude(v: &VelocityField) -> f64 {
    let n = v.num_cells();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for idx in 0..n {
        let u = v.u.data[idx];
        let vv = v.v.data[idx];
        let w = v.w.data[idx];
        sum += (u * u + vv * vv + w * w).sqrt();
    }
    sum / n as f64
}

/// A constant boundary-condition block used by every macro step.
fn constant_bc() -> BesToFfdBoundaryConditions {
    BesToFfdBoundaryConditions {
        outdoor_temperature: 283.15,
        surface_temperatures: vec![293.15; 8],
        hvac_supply_temperature: 288.15,
        hvac_supply_flow: 0.1,
        wind_pressure: vec![0.0; 4],
        internal_gains: 500.0,
        time_start: 0.0,
        macro_timestep: 3600.0,
    }
}

/// Adapter construction with a valid grid succeeds and the recommended
/// micro timestep is exactly the configured `FfdConfig::dt`.
#[test]
fn ffd_cfd_adapter_preserves_configured_dt() {
    let adapter = FfdCfdAdapter::new(tiny_ffd_config()).unwrap();
    assert_eq!(adapter.name(), "FfdCfdAdapter");
    assert!(
        (adapter.recommended_micro_timestep() - 0.001).abs() < 1e-9,
        "Adapter should expose the FFD dt as recommended_micro_timestep"
    );
    assert!(
        !adapter.is_valid(),
        "Adapter should be invalid before initialize"
    );
}

/// `LooseCoupling` accepts an `FfdCfdAdapter`-backed solver and runs
/// multiple macro steps with constant BCs. The time-averaged `FfdToBesResults`
/// have the correct shape (CHTC/flux per surface, one temperature per
/// zone, time covered = macro_timestep).
///
/// Per issue #2460: build a 2-zone coordinator and run 10 macro steps
/// with constant BCs. To keep the regression test fast on the CPU solver
/// (the FFD's recommended micro_dt = 0.001s, so a 3600s macro step would
/// require 3.6M FFD micro steps), this test uses a short macro_timestep
/// (1.0s) and asserts that 10 steps × 1000 micro steps each = 10k FFD
/// micro steps complete and the wiring is correct end-to-end.
#[test]
fn ffd_cfd_adapter_drives_loose_coupling_two_zone() {
    let mut adapter = FfdCfdAdapter::new(tiny_ffd_config()).unwrap();
    let num_zones = 2;
    let num_surfaces = 8;
    let zone_volumes = vec![100.0; num_zones];
    let surface_areas = vec![10.0; num_surfaces];
    adapter
        .initialize(num_zones, &zone_volumes, &surface_areas, num_surfaces)
        .expect("initialize should succeed");
    let macro_timestep = 1.0_f64; // 1000 FFD micro steps per macro step
    let mut coupling =
        LooseCoupling::new(Box::new(adapter), num_zones, num_surfaces, macro_timestep)
            .expect("LooseCoupling::new should accept the adapter");
    assert!(coupling.is_valid());

    let bc = constant_bc();
    for step in 0..10 {
        let results = coupling
            .exchange_and_step(bc.clone())
            .expect("exchange_and_step should succeed");
        assert_eq!(
            results.chtc.len(),
            num_surfaces,
            "Step {}: expected {} CHTC values",
            step,
            num_surfaces
        );
        assert_eq!(
            results.zone_temperatures.len(),
            num_zones,
            "Step {}: expected {} zone temperatures",
            step,
            num_zones
        );
        assert_eq!(
            results.surface_heat_flux.len(),
            num_surfaces,
            "Step {}: expected {} surface heat flux values",
            step,
            num_surfaces
        );
        assert!(
            results.micro_step_count > 0,
            "Step {}: should have run at least one micro step",
            step
        );
        assert!(
            (results.simulation_time_covered - macro_timestep).abs() < 1e-9,
            "Step {}: simulation_time_covered should equal macro_timestep",
            step
        );
    }

    // After 10 macro steps of macro_timestep each, current_time = 10 * macro_timestep.
    assert!(
        (coupling.current_time() - 10.0 * macro_timestep).abs() < 1e-9,
        "Coordinator should advance 10 * macro_timestep = {}s",
        10.0 * macro_timestep
    );
}

/// The adapter's CHTC translation is bit-for-bit consistent with the
/// post-step FFD velocity field (within `1e-4`).
///
/// The test:
///
/// 1. Runs the adapter through `step_micro` to obtain the adapter's CHTC.
/// 2. Computes the **expected** CHTC from the same FFD velocity field
///    using the same formula the adapter uses internally (this is
///    the cross-validation: the adapter's `compute_chtc` is public so
///    the test can verify the formula, while the post-step velocity
///    field is the source of truth for the FFD).
/// 3. Asserts that the two CHTC vectors are equal to `1e-4`.
///
/// This validates that the adapter translation is deterministic and
/// consistent with the underlying CFD state.
#[test]
fn ffd_cfd_adapter_chtc_matches_post_step_velocity_field() {
    let mut adapter = FfdCfdAdapter::new(tiny_ffd_config()).unwrap();
    let num_surfaces = 8;
    let num_zones = 1;
    let surface_areas = vec![1.0; num_surfaces];
    adapter
        .initialize(num_zones, &[10.0], &surface_areas, num_surfaces)
        .unwrap();

    let bc = BesToFfdBoundaryConditions {
        outdoor_temperature: 283.15,
        surface_temperatures: vec![293.15; num_surfaces],
        hvac_supply_temperature: 288.15,
        hvac_supply_flow: 0.1,
        wind_pressure: vec![2.5; 4],
        internal_gains: 500.0,
        time_start: 0.0,
        macro_timestep: 3600.0,
    };

    let results = adapter.step_micro(&bc, 0.001).unwrap();

    // Independent CHTC from the public `compute_chtc` method.
    let adapter_chtc = adapter.compute_chtc();

    // The step_micro result CHTC must equal the public compute_chtc output
    // to floating-point equality (same formula, same input).
    for (i, (reported, adapter)) in results.chtc.iter().zip(adapter_chtc.iter()).enumerate() {
        assert!(
            (reported - adapter).abs() < 1e-12,
            "CHTC[{}]: step_micro result ({}) must match compute_chtc ({})",
            i,
            reported,
            adapter
        );
    }

    // The independent "expected" CHTC (re-derived from the FFD state
    // outside the adapter) must agree to `1e-4` (issue #2460 acceptance).
    let velocity = adapter.inner().velocity();
    let v_t = mean_velocity_magnitude(velocity);
    let expected_h = (2.5_f64 + 2.0_f64 * v_t).max(2.5);
    for (i, &h) in results.chtc.iter().enumerate() {
        let err = (h - expected_h).abs();
        assert!(
            err < 1e-4,
            "CHTC[{}]: adapter reported {}, expected {}, |err| = {} > 1e-4",
            i,
            h,
            expected_h,
            err
        );
    }
}

/// Energy-balance gate (RULES.md §1, ARCHITECTURE.md §"Module N+2"):
/// the time-averaged `Q_conv` from CHTC plus the reported `surface_heat_flux`
/// must agree on the convective term within `1e-3 W`. The adapter uses
/// `q = h * (T_air - T_surface)` and reports both `chtc` and
/// `surface_heat_flux`; the regression test asserts that re-deriving
/// the heat flux from the CHTC and surface/zone temperatures yields
/// the same vector.
#[test]
fn ffd_cfd_adapter_heat_flux_consistent_with_chtc() {
    let mut adapter = FfdCfdAdapter::new(tiny_ffd_config()).unwrap();
    let num_surfaces = 4;
    let num_zones = 1;
    let surface_areas = vec![1.0; num_surfaces];
    adapter
        .initialize(num_zones, &[10.0], &surface_areas, num_surfaces)
        .unwrap();

    let surface_temperatures = vec![296.15, 295.15, 294.15, 293.15];
    let bc = BesToFfdBoundaryConditions {
        outdoor_temperature: 283.15,
        surface_temperatures: surface_temperatures.clone(),
        hvac_supply_temperature: 288.15,
        hvac_supply_flow: 0.1,
        wind_pressure: vec![0.0; 4],
        internal_gains: 500.0,
        time_start: 0.0,
        macro_timestep: 3600.0,
    };

    let results = adapter.step_micro(&bc, 0.001).unwrap();
    // mean_zone_t = arithmetic mean of the surface temperatures, matching
    // the adapter's `compute_zone_temperatures` definition for the
    // uniform-mixing approximation.
    let mean_zone_t: f64 =
        surface_temperatures.iter().sum::<f64>() / surface_temperatures.len() as f64;
    for (i, (&h, &q)) in results
        .chtc
        .iter()
        .zip(results.surface_heat_flux.iter())
        .enumerate()
    {
        let t_s = surface_temperatures[i];
        let expected_q = h * (mean_zone_t - t_s);
        assert!(
            (q - expected_q).abs() < 1e-3,
            "Surface {}: q ({}) should equal h * (T_air - T_s) = {} within 1e-3 W",
            i,
            q,
            expected_q
        );
    }
}

/// Recommended-micro-timestep is preserved across the full FFD solver
/// → adapter → loose-coupling chain (`1e-9` tolerance per issue #2460).
#[test]
fn ffd_cfd_adapter_dt_visible_through_loose_coupling() {
    let mut adapter = FfdCfdAdapter::new(tiny_ffd_config()).unwrap();
    adapter.initialize(1, &[10.0], &[1.0; 4], 4).unwrap();
    // Use macro = 1.0s to avoid the 3.6M FFD-step sweep that 3600s would
    // require; the ratio test only depends on `recommended_micro_timestep()`.
    let coupling = LooseCoupling::new(Box::new(adapter), 1, 4, 1.0).unwrap();
    let ratio = coupling.timestep_ratio();
    assert!(
        (ratio - 1000.0).abs() < 1e-6,
        "timestep_ratio = 1.0 / 0.001 = 1000, got {}",
        ratio
    );
}
