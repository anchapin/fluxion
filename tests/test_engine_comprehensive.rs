use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::physics::ctf_coefficients::{CTFCalculator, CTFMaterial};
use fluxion::physics::ctf_solver::{CTFSolver, CTFSolverConfig};
use fluxion::sim::engine::{IdealHVACController, ThermalModel};
use fluxion::sim::hvac::{AnyEquipment, HeatPump};
use fluxion::validation::ashrae_140_cases::NightVentilation;

#[test]
fn test_night_ventilation_activation() {
    let mut model = ThermalModel::<VectorField>::new(1);
    let surrogates = SurrogateManager::default();

    // Set up night ventilation (active from 22:00 to 07:00)
    let night_vent = NightVentilation::new(1700.0, 22, 7);
    model.night_ventilation = Some(night_vent);

    // Test at 12:00 (inactive)
    let energy_inactive = model.solve_single_step(
        12,
        10.0,
        false,
        &surrogates,
        false,
        None,
        None,
        None,
        3600.0,
    );

    // Test at 23:00 (active)
    let energy_active = model.solve_single_step(
        23,
        10.0,
        false,
        &surrogates,
        false,
        None,
        None,
        None,
        3600.0,
    );

    // Both should run without panic
    assert!(energy_inactive >= 0.0);
    assert!(energy_active >= 0.0);
}

#[test]
fn test_multi_zone_initialization() {
    let mut model = ThermalModel::<VectorField>::new(2);
    let surrogates = SurrogateManager::default();
    assert_eq!(model.num_zones, 2);

    // Set inter-zone conductance
    model.h_tr_iz = VectorField::new(vec![10.0, 10.0]);

    // Run a step
    let energy =
        model.solve_single_step(0, 10.0, false, &surrogates, false, None, None, None, 3600.0);
    assert!(energy.is_finite());
}

#[test]
fn test_8r3c_initialization() {
    let mut model = ThermalModel::<VectorField>::new_8r3c(1);
    let surrogates = SurrogateManager::default();
    assert!(model.is_8r3c_model());

    // Run a step (should call step_physics_8r3c)
    let energy =
        model.solve_single_step(0, 10.0, false, &surrogates, false, None, None, None, 3600.0);
    assert!(energy.is_finite());
}

#[test]
fn test_ctf_integration() {
    let mut model = ThermalModel::<VectorField>::new(1);
    let surrogates = SurrogateManager::default();

    // Create dummy CTF solver
    let layers = vec![CTFMaterial::new("Concrete", 0.1, 1.0, 2000.0, 800.0)];
    let coeffs = CTFCalculator::new(&layers, 3600.0, 10).compute_coefficients();
    let config = CTFSolverConfig::new(3600.0, 10);
    let solver = CTFSolver::new(coeffs, config);

    model.ctf_solvers = vec![solver];
    model.ctf_enabled = true;

    // Run a step
    let energy =
        model.solve_single_step(0, 10.0, false, &surrogates, false, None, None, None, 3600.0);
    assert!(energy.is_finite());
}

#[test]
fn test_hvac_equipment_integration() {
    let mut model = ThermalModel::<VectorField>::new(1);
    let surrogates = SurrogateManager::default();

    // Create heat pump equipment
    let hp = HeatPump::new("HP-1".to_string(), 5000.0, 5000.0, 3.0, 3.0);
    model.hvac_equipment = Some(AnyEquipment::HeatPump(hp));

    // Run a step
    let energy =
        model.solve_single_step(0, 10.0, false, &surrogates, false, None, None, None, 3600.0);
    assert!(energy.is_finite());
}

#[test]
fn test_ideal_hvac_controller_staging() {
    let mut controller = IdealHVACController::new(20.0, 27.0);
    controller.heating_stages = 2;
    controller.heating_capacity_per_stage = 2500.0;

    // Zone at 19.4 (0.6 deficit, threshold is 20.0 - 0.5 = 19.5)
    let power = controller.calculate_power(19.4, 19.4, 0.001);
    assert!(power > 0.0);
}
