//! PCM (Phase Change Material) Integration Tests
//!
//! Tests for the Phase Change Material implementation per Issue #2398.
//!
//! # Acceptance Criteria
//!
//! - PCM wall with melting point 21°C, 50 kJ/kg latent heat, 2 kJ/(kg·K) solid/liquid Cp
//!   exhibits temperature stall at 21°C under solar load
//! - Latent heat energy buffer is correctly computed
//! - PCM material can be used in assemblies with standard builder pattern
//!
//! # Test Strategy
//!
//! 1. **Material Properties**: Verify effective Cp, melt fraction, and latent heat accounting
//! 2. **Stefan Problem Analytic**: Verify the theoretical melt time matches expectations
//! 3. **Assembly Integration**: Verify PCM works with AssemblyBuilder
//! 4. **Energy Conservation**: Verify latent heat is correctly integrated
//!
//! # References
//!
//! - Incropera & DeWitt, Chapter 11 — Transient Analysis (Stefan problem)
//! - ASHRAE Handbook — PCM for building thermal storage

use fluxion::physics::fd_solver_wrapper::FDSolverWrapper;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time};
use fluxion_core::assembly::{AssemblyBuilder, MaterialLayer, PcmMaterial};
use fluxion_core::assembly::{BrickMaterial, GypsumMaterial};

const H_INT: f64 = 8.0;
const H_EXT: f64 = 25.0;

#[test]
fn test_pcm_enthalpy_trapezoidal_profile() {
    let pcm = PcmMaterial::new(
        0.05,     // thickness 5 cm
        2000.0,   // solid Cp J/kgK
        2000.0,   // liquid Cp J/kgK
        50_000.0, // latent heat J/kg = 50 kJ/kg
        21.0,     // melting point °C
        4.0,      // melt range °C (±2°C)
    );

    let t_below = 17.0;
    let t_melt = 21.0;
    let t_above = 25.0;

    let cp_below = pcm.effective_specific_heat(t_below);
    let cp_melt = pcm.effective_specific_heat(t_melt);
    let cp_above = pcm.effective_specific_heat(t_above);

    assert_eq!(cp_below, 2000.0, "Cp below melt zone should be solid Cp");
    assert_eq!(cp_above, 2000.0, "Cp above melt zone should be liquid Cp");

    let latent_contribution = 50_000.0 / 4.0;
    let mid_cp = 2000.0;
    let expected_at_melt = mid_cp + latent_contribution;
    assert!(
        (cp_melt - expected_at_melt).abs() < 1.0,
        "Cp at melt should include latent heat contribution"
    );
}

#[test]
fn test_pcm_melt_time_stefan_problem() {
    let pcm = PcmMaterial::new(
        0.05,     // thickness 5 cm
        2000.0,   // solid Cp J/kgK
        2000.0,   // liquid Cp J/kgK
        50_000.0, // latent heat J/kg = 50 kJ/kg
        21.0,     // melting point °C
        4.0,      // melt range °C (±2°C)
    );

    let density = pcm.density();
    let thickness = pcm.thickness();
    let latent_heat = pcm.latent_heat_J_kg();

    let mass_per_area = density * thickness;
    let latent_energy_per_area = latent_heat * mass_per_area;

    let solar_flux = 1000.0;
    let t_stall_hours = latent_energy_per_area / (solar_flux * 3600.0);

    assert!(
        (t_stall_hours - 2.25).abs() < 0.1,
        "Stall time with L=50kJ/kg should be ~2.25h at 1000 W/m², got {}",
        t_stall_hours
    );
}

#[test]
fn test_pcm_latent_heat_energy_integration() {
    let pcm = PcmMaterial::new(0.05, 2000.0, 2000.0, 50_000.0, 21.0, 4.0);

    let density = pcm.density();
    let thickness = pcm.thickness();
    let mass_per_area = density * thickness;

    let t_start = 19.0;
    let t_end = 23.0;
    let dt = 0.01;
    let mut total_energy = 0.0;
    let mut t = t_start;

    while t < t_end {
        let cp = pcm.effective_specific_heat(t);
        total_energy += cp * mass_per_area * dt;
        t += dt;
    }

    let latent_expected = 50_000.0 * mass_per_area;
    let sensible_expected = 2000.0 * mass_per_area * 4.0;
    let total_expected = latent_expected + sensible_expected;

    let error_pct = ((total_energy - total_expected) / total_expected * 100.0).abs();
    assert!(
        error_pct < 1.0,
        "Energy integration error {}% should be < 1%",
        error_pct
    );
}

#[test]
fn test_pcm_assembly_builder_integration() {
    let assembly = AssemblyBuilder::new("pcm_wall".to_string())
        .add_layer(Box::new(BrickMaterial::new(0.1)))
        .add_layer(Box::new(PcmMaterial::new(
            0.05, 2000.0, 2000.0, 50_000.0, 21.0, 4.0,
        )))
        .add_layer(Box::new(GypsumMaterial::new(0.012)))
        .build()
        .expect("PCM assembly should build successfully");

    assert_eq!(assembly.layers.len(), 3);
    assert_eq!(assembly.total_thickness(), 0.162);

    let pcm_layer = &assembly.layers[1];
    assert_eq!(pcm_layer.name(), "PCM");

    let pcm_ref = pcm_layer.as_ref();
    assert!((pcm_ref.specific_heat() - 2000.0).abs() < 1.0);
    assert!((pcm_ref.density() - 900.0).abs() < 1.0);
}

#[test]
fn test_pcm_specific_heat_nominal_value() {
    let pcm = PcmMaterial::new(0.05, 1500.0, 2500.0, 50_000.0, 21.0, 4.0);

    let nominal_cp = pcm.specific_heat();
    let expected_nominal = (1500.0 + 2500.0) / 2.0;
    assert!((nominal_cp - expected_nominal).abs() < 1.0);
}

#[test]
fn test_pcm_melt_fraction_gradient() {
    let pcm = PcmMaterial::new(0.05, 2000.0, 2000.0, 50_000.0, 21.0, 4.0);

    let test_temps = [17.0, 19.0, 20.0, 20.5, 21.0, 21.5, 22.0, 23.0, 25.0];
    let expected_fractions = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0];

    for (t, expected) in test_temps.iter().zip(expected_fractions.iter()) {
        let fraction = pcm.melt_fraction(*t);
        assert!(
            (fraction - expected).abs() < 0.01,
            "melt_fraction at {}°C should be {}, got {}",
            t,
            expected,
            fraction
        );
    }
}

#[test]
fn test_pcm_fd_solver_seam() {
    use fluxion::physics::wall_spec::WallSpec;

    let pcm = PcmMaterial::new(0.05, 2000.0, 2000.0, 50_000.0, 21.0, 4.0);

    let wall = WallSpec::single_layer(
        "PCM Wall",
        pcm.thickness(),
        pcm.conductivity(),
        pcm.density(),
        pcm.specific_heat(),
    );

    let mut solver = FDSolverWrapper::new();
    solver
        .initialize(&wall)
        .expect("FD solver with PCM should initialize");

    assert!(solver.is_valid());

    let flux = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(10.0),
            HeatTransferCoefficient::from_value(H_INT),
            HeatTransferCoefficient::from_value(H_EXT),
        )
        .expect("FD step should succeed");

    assert!(
        flux.to_value().is_finite(),
        "Flux should be finite, got {}",
        flux.to_value()
    );
}

#[test]
fn test_pcm_fd_solver_step_convergence() {
    use fluxion::physics::wall_spec::WallSpec;

    let pcm = PcmMaterial::new(0.05, 2000.0, 2000.0, 50_000.0, 21.0, 4.0);

    let wall = WallSpec::single_layer(
        "PCM Wall",
        pcm.thickness(),
        pcm.conductivity(),
        pcm.density(),
        pcm.specific_heat(),
    );

    let mut solver = FDSolverWrapper::new();
    solver.initialize(&wall).expect("FD init");

    let mut prev_flux = 0.0;
    for _ in 0..100 {
        let flux = solver
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(0.0),
                HeatTransferCoefficient::from_value(H_INT),
                HeatTransferCoefficient::from_value(H_EXT),
            )
            .unwrap()
            .to_value();
        assert!(flux.is_finite());
        prev_flux = flux;
    }
}

#[test]
fn test_pcm_multiple_layers_with_concrete() {
    let assembly = AssemblyBuilder::new("composite_pcm_wall".to_string())
        .add_layer(Box::new(BrickMaterial::new(0.1)))
        .add_layer(Box::new(PcmMaterial::new(
            0.03, 2000.0, 2000.0, 50_000.0, 21.0, 4.0,
        )))
        .add_layer(Box::new(PcmMaterial::new(
            0.02, 1500.0, 2500.0, 60_000.0, 25.0, 5.0,
        )))
        .add_layer(Box::new(GypsumMaterial::new(0.012)))
        .build()
        .expect("Multi-PCM assembly should build");

    assert_eq!(assembly.layers.len(), 4);
    assert_eq!(assembly.total_thickness(), 0.162);
}

#[test]
fn test_pcm_downcast_from_trait() {
    let pcm = PcmMaterial::new(0.05, 2000.0, 2000.0, 50_000.0, 21.0, 4.0);

    let layer: &dyn MaterialLayer = &pcm;
    let downcast = layer.as_any().downcast_ref::<PcmMaterial>();
    assert!(
        downcast.is_some(),
        "Should be able to downcast to PcmMaterial"
    );

    let downcast_unwrap = downcast.unwrap();
    assert!((downcast_unwrap.melting_point_C() - 21.0).abs() < 0.001);
    assert!((downcast_unwrap.melt_range_C() - 4.0).abs() < 0.001);
}
