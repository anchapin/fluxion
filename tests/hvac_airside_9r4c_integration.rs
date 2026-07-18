//! Coupled 9R4C envelope + airside integration tests (issue #1767).

use fluxion::multi_node::{MassAirCouplingMode, ThermalMassNode};
use fluxion::physics::cta::VectorField;
use fluxion::physics::multi_node_solver::{MultiNodeSolver, SurfaceExteriorTemperatures};
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::hvac::{
    AirsideCouplingError, AirsideEnvelopeCoupler, AirsideFlow, CoupledStepForcing, MoistAirState,
    DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

const PRESSURE_PA: f64 = 101_325.0;
const TIMESTEP_SECONDS: f64 = 360.0;
const TIMESTEPS_PER_YEAR: usize = 87_600;

fn high_mass_coupler() -> AirsideEnvelopeCoupler {
    high_mass_coupler_with_mode(MassAirCouplingMode::ParallelResistance)
}

fn high_mass_coupler_with_mode(mode: MassAirCouplingMode) -> AirsideEnvelopeCoupler {
    let wall = ThermalMassNode::new(20.0, 5.0e6, 76.4, 25.0);
    let roof = ThermalMassNode::new(20.0, 3.0e6, 32.9, 20.0);
    let floor = ThermalMassNode::new(20.0, 2.0e6, 18.0, 10.0);
    let internal = ThermalMassNode::new(20.0, 1.0e6, 0.0, 0.0).with_h_tr_me(100.0);
    let solver = MultiNodeSolver::new_with_mode(165.6, wall, roof, floor, internal, mode);
    let zone_air = MoistAirState::try_new(22.0, 50.0, PRESSURE_PA)
        .expect("initial zone air state must be physical");

    AirsideEnvelopeCoupler::new(solver, zone_air, 300.0)
        .expect("representative high-mass zone must construct")
}

#[test]
fn test_coupled_annual_run_no_nan() {
    let mut coupled = high_mass_coupler();
    let mut completed_steps = 0_usize;
    let mut max_balance_residual_w = 0.0_f64;

    for step in 0..TIMESTEPS_PER_YEAR {
        let day = step as f64 / 240.0;
        let hour = (step as f64 / 10.0) % 24.0;
        let annual_phase = 2.0 * std::f64::consts::PI * (day - 30.0) / 365.0;
        let daily_phase = 2.0 * std::f64::consts::PI * (hour - 8.0) / 24.0;
        let outdoor_temperature_c = 10.0 + 18.0 * annual_phase.sin() + 7.0 * daily_phase.sin();
        let outdoor_rh_percent = 50.0 + 20.0 * daily_phase.cos();
        let outdoor_air =
            MoistAirState::try_new(outdoor_temperature_c, outdoor_rh_percent, PRESSURE_PA)
                .expect("annual outdoor state must remain physical");

        let (supply_temperature_c, supply_rh_percent) = if outdoor_temperature_c > 18.0 {
            (14.0, 55.0)
        } else {
            (32.0, 20.0)
        };
        let supply_air =
            MoistAirState::try_new(supply_temperature_c, supply_rh_percent, PRESSURE_PA)
                .expect("VAV/DOAS-equivalent supply state must remain physical");
        let airside = AirsideFlow::new(supply_air, 0.55)
            .expect("supply volume flow must produce a physical dry-air flow");

        let occupied = (7.0..19.0).contains(&hour);
        let convective_gain_w = if occupied { 500.0 } else { 50.0 };
        let solar_gain_w = (std::f64::consts::PI * (hour - 6.0) / 12.0).sin().max(0.0) * 1_500.0;
        let forcing = CoupledStepForcing {
            exterior_temperatures: SurfaceExteriorTemperatures::uniform(outdoor_temperature_c),
            outdoor_air,
            ventilation_conductance_w_per_k: 20.0,
            convective_gain_w,
            envelope_gains_w: [
                0.45 * solar_gain_w,
                0.30 * solar_gain_w,
                0.0,
                0.25 * solar_gain_w,
            ],
        };

        let result = coupled
            .step(TIMESTEP_SECONDS, &forcing, &airside)
            .unwrap_or_else(|error| panic!("coupled step {step} failed: {error}"));

        assert!(
            result.is_finite(),
            "non-finite result at step {step}: {result:?}"
        );
        assert!(
            coupled.zone_air().is_finite(),
            "non-finite zone psychrometric state at step {step}: {:?}",
            coupled.zone_air()
        );
        assert!(
            coupled
                .envelope()
                .snapshot_temperatures()
                .into_iter()
                .all(f64::is_finite),
            "non-finite 9R4C node at step {step}: {:?}",
            coupled.envelope().snapshot_temperatures()
        );

        max_balance_residual_w = max_balance_residual_w.max(result.energy_balance_residual_w.abs());
        completed_steps += 1;
    }

    assert_eq!(completed_steps, TIMESTEPS_PER_YEAR);
    assert!(
        max_balance_residual_w <= DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
        "maximum per-step residual {max_balance_residual_w:.3e} W exceeded tolerance {:.3e} W",
        DEFAULT_ENERGY_BALANCE_TOLERANCE_W
    );
    println!(
        "COUPLED_ANNUAL_RESULT|steps={completed_steps}|max_residual_w={max_balance_residual_w:.12e}"
    );
}

#[test]
fn test_energy_balance_closes() {
    let outdoor_air =
        MoistAirState::try_new(5.0, 80.0, PRESSURE_PA).expect("outdoor state must be physical");
    let supply_air = MoistAirState::try_new(35.0, 20.0, PRESSURE_PA)
        .expect("heating supply state must be physical");
    let airside = AirsideFlow::new(supply_air, 0.55)
        .expect("supply flow must produce a physical dry-air flow");
    let forcing = CoupledStepForcing {
        exterior_temperatures: SurfaceExteriorTemperatures::uniform(5.0),
        outdoor_air,
        ventilation_conductance_w_per_k: 20.0,
        convective_gain_w: 350.0,
        envelope_gains_w: [120.0, 80.0, 0.0, 100.0],
    };

    for mode in [
        MassAirCouplingMode::AdditiveSum,
        MassAirCouplingMode::ParallelResistance,
    ] {
        let mut coupled = high_mass_coupler_with_mode(mode);
        let result = coupled
            .step(TIMESTEP_SECONDS, &forcing, &airside)
            .unwrap_or_else(|error| panic!("{mode:?} coupled step failed: {error}"));

        assert!(
            result.sensible_balance_residual_w.abs() <= DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
            "{mode:?} sensible residual {:.3e} W exceeds tolerance",
            result.sensible_balance_residual_w
        );
        assert!(
            result.latent_balance_residual_w.abs() <= DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
            "{mode:?} latent residual {:.3e} W exceeds tolerance",
            result.latent_balance_residual_w
        );
        assert!(
            result.energy_balance_residual_w.abs() <= DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
            "{mode:?} total residual {:.3e} W exceeds tolerance",
            result.energy_balance_residual_w
        );
        assert!(
            result.moisture_balance_residual_kg_per_s.abs() <= 1.0e-12,
            "{mode:?} dry-air moisture residual {:.3e} kg/s exceeds tolerance",
            result.moisture_balance_residual_kg_per_s
        );
        assert!(
            (result.supply_total_heat_w
                - result.supply_sensible_heat_w
                - result.supply_latent_heat_w)
                .abs()
                <= DEFAULT_ENERGY_BALANCE_TOLERANCE_W,
            "{mode:?} supply sensible + latent heat must reconstruct total enthalpy flow"
        );
    }
}

#[test]
fn test_ashrae_140_envelope_unchanged() {
    // Baseline captured from unmodified origin/develop before the issue #1767
    // integration module was added. Case 900FF is envelope-only, so these
    // deterministic annual temperatures detect any accidental production-path
    // change independently of airside equipment behavior.
    const BASELINE_MIN_C: f64 = 2.159_094;
    const BASELINE_MAX_C: f64 = 48.030_329;
    const BASELINE_AVERAGE_C: f64 = 26.532_328;
    const BASELINE_TOLERANCE_C: f64 = 1.0e-6;

    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();
    let mut minimum_c = f64::MAX;
    let mut maximum_c = f64::MIN;
    let mut sum_c = 0.0_f64;

    for step in 0..8_760 {
        let weather_data = weather
            .get_hourly_data(step)
            .expect("Denver TMY must contain every annual timestep");
        model.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3_600.0);
        let zone_temperature_c = model.temperatures.as_slice()[0];
        assert!(
            zone_temperature_c.is_finite(),
            "Case 900FF produced non-finite zone temperature at hour {step}"
        );
        minimum_c = minimum_c.min(zone_temperature_c);
        maximum_c = maximum_c.max(zone_temperature_c);
        sum_c += zone_temperature_c;
    }

    let average_c = sum_c / 8_760.0;
    assert!(
        (minimum_c - BASELINE_MIN_C).abs() <= BASELINE_TOLERANCE_C,
        "Case 900FF minimum changed: baseline {BASELINE_MIN_C:.6}°C, actual {minimum_c:.6}°C"
    );
    assert!(
        (maximum_c - BASELINE_MAX_C).abs() <= BASELINE_TOLERANCE_C,
        "Case 900FF maximum changed: baseline {BASELINE_MAX_C:.6}°C, actual {maximum_c:.6}°C"
    );
    assert!(
        (average_c - BASELINE_AVERAGE_C).abs() <= BASELINE_TOLERANCE_C,
        "Case 900FF average changed: baseline {BASELINE_AVERAGE_C:.6}°C, actual {average_c:.6}°C"
    );
    println!(
        "ASHRAE140_900FF_REGRESSION|min_c={minimum_c:.6}|max_c={maximum_c:.6}|avg_c={average_c:.6}"
    );
}

#[test]
fn test_non_finite_guard_is_transactional() {
    let mut coupled = high_mass_coupler();
    let initial_nodes = coupled.envelope().snapshot_temperatures();
    let initial_zone_air = *coupled.zone_air();
    let outdoor_air =
        MoistAirState::try_new(5.0, 80.0, PRESSURE_PA).expect("outdoor state must be physical");
    let supply_air =
        MoistAirState::try_new(35.0, 20.0, PRESSURE_PA).expect("supply state must be physical");
    let airside = AirsideFlow::new(supply_air, 0.55).expect("supply flow must be physical");
    let forcing = CoupledStepForcing {
        exterior_temperatures: SurfaceExteriorTemperatures::uniform(5.0),
        outdoor_air,
        ventilation_conductance_w_per_k: 20.0,
        convective_gain_w: f64::NAN,
        envelope_gains_w: [0.0; 4],
    };

    let error = coupled
        .step(TIMESTEP_SECONDS, &forcing, &airside)
        .expect_err("NaN forcing must be rejected");
    assert!(matches!(
        error,
        AirsideCouplingError::InvalidInput {
            field: "convective_gain_w",
            ..
        }
    ));
    assert_eq!(coupled.envelope().snapshot_temperatures(), initial_nodes);
    assert_eq!(*coupled.zone_air(), initial_zone_air);
    assert!(
        AirsideFlow::new(supply_air, f64::MAX).is_err(),
        "overflowing volume flow must not produce an infinite mass flow"
    );
}
