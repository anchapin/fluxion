//! Issue #4101 — hourly end-use metering tests.
//!
//! Covers the timestep-indexed kWh series (heating / cooling / lighting /
//! equipment) recorded by the solver loops:
//!
//! - series lengths match the step count on the physics path
//! - hand-computable lighting and equipment values
//! - explicit equipment is threaded into `StepParameters` (thermal effect)
//!   instead of silently dropped
//! - determinism across identical runs
//! - series sums agree with the annual accumulators
//! - `DiagnosticsState::clone` drops the new series
//! - auto-loaded profile equipment stays unapplied (strict-gate guard)
//! - schema `EquipmentSpec` validation / building and wire-shape stability

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::api::schema::{EquipmentSpec, SimulationOutput};
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::equipment::{ComputerEquipment, Equipment};
use fluxion::sim::lighting::LightingSchedule;
use fluxion::sim::schedule::DailySchedule;

fn test_surrogates() -> SurrogateManager {
    SurrogateManager::new().expect("Failed to create SurrogateManager")
}

/// One-zone model with sane setpoints, mirroring `lib_tests.rs`.
fn test_model() -> ThermalModel<VectorField> {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.apply_parameters(&[1.5, 20.0, 24.0]);
    model
}

/// Lighting schedule with a hand-computable shape: full power 08:00-17:00,
/// off otherwise.
fn test_lighting() -> LightingSchedule {
    let mut sched = LightingSchedule::new(10.0, 100.0);
    for h in 0..24 {
        sched.hourly_schedule[h] = if (8..18).contains(&h) { 1.0 } else { 0.0 };
    }
    sched
}

/// 2 × 500 W computers, always on.
fn test_equipment() -> Vec<Box<dyn Equipment>> {
    let schedule = DailySchedule::constant(1.0).expect("constant schedule cannot fail");
    vec![Box::new(
        ComputerEquipment::new("test-computers".to_string(), 500.0, 2).with_schedule(schedule),
    )]
}

#[test]
fn metering_series_lengths_match_step_count() {
    let mut model = test_model();
    let surrogates = test_surrogates();
    let lighting = test_lighting();
    let equipment = test_equipment();
    let steps = 48;

    let eui = model.solve_timesteps(
        steps,
        &surrogates,
        false,
        Some(&lighting),
        Some(equipment.as_slice()),
        None,
    );
    assert!(eui.is_finite(), "EUI must be finite");

    for (name, series) in [
        ("heating", model.get_hourly_heating_kwh()),
        ("cooling", model.get_hourly_cooling_kwh()),
        ("lighting", model.get_hourly_lighting_kwh()),
        ("equipment", model.get_hourly_equipment_kwh()),
    ] {
        let series = series.unwrap_or_else(|| panic!("{name} series must be recorded"));
        assert_eq!(series.len(), steps, "{name} series length must equal steps");
        assert!(
            series.iter().all(|v| v.is_finite() && *v >= 0.0),
            "{name} series must be finite and non-negative"
        );
    }
}

#[test]
fn metering_lighting_values_are_hand_computable() {
    let mut model = test_model();
    let surrogates = test_surrogates();
    let lighting = test_lighting();
    let steps = 72;

    let dt = model.calculate_timestep_seconds();
    model.solve_timesteps(steps, &surrogates, false, Some(&lighting), None, None);

    let series = model
        .get_hourly_lighting_kwh()
        .expect("lighting series must be recorded");
    assert_eq!(series.len(), steps);
    for (t, &metered) in series.iter().enumerate() {
        // lighting_power(t) = density * area * schedule[t % 24], in Watts.
        let expected = lighting.lighting_power(t) * dt / 3.6e6;
        assert!(
            (metered - expected).abs() < 1e-12,
            "t={t}: metered {metered} != expected {expected}"
        );
    }
    // Shape check: on during 08-17, off at night.
    assert!(series[10] > 0.0, "lighting must be on at 10:00");
    assert_eq!(series[2], 0.0, "lighting must be off at 02:00");
    assert!(
        (series[10] - 10.0 * 100.0 * dt / 3.6e6).abs() < 1e-12,
        "10 W/m² × 100 m² for one timestep"
    );
}

#[test]
fn metering_equipment_values_are_hand_computable() {
    let mut model = test_model();
    let surrogates = test_surrogates();
    let equipment = test_equipment();
    let steps = 48;

    let dt = model.calculate_timestep_seconds();
    // Expected electric draw: 2 × 500 W always on.
    let expected_per_step = 1000.0 * dt / 3.6e6;

    model.solve_timesteps(
        steps,
        &surrogates,
        false,
        None,
        Some(equipment.as_slice()),
        None,
    );

    let series = model
        .get_hourly_equipment_kwh()
        .expect("equipment series must be recorded");
    assert_eq!(series.len(), steps);
    for (t, &metered) in series.iter().enumerate() {
        assert!(
            (metered - expected_per_step).abs() < 1e-12,
            "t={t}: metered {metered} != expected {expected_per_step}"
        );
    }
    // Lighting was not supplied: its series exists but is all zeros.
    let lighting = model
        .get_hourly_lighting_kwh()
        .expect("lighting series must be recorded");
    assert!(lighting.iter().all(|&v| v == 0.0));
}

#[test]
fn metering_explicit_equipment_reaches_thermal_path() {
    // Issue #4101: explicit equipment must be threaded into StepParameters
    // (previously hardcoded to None). 1 kW of always-on internal gains must
    // move the thermal result.
    let surrogates = test_surrogates();
    let lighting = test_lighting();

    let mut without = test_model();
    without.solve_timesteps(72, &surrogates, false, Some(&lighting), None, None);
    let heating_without = without.get_heating_energy_kwh();
    let temps_without = without.get_temperatures();

    let mut with = test_model();
    let equipment = test_equipment();
    with.solve_timesteps(
        72,
        &surrogates,
        false,
        Some(&lighting),
        Some(equipment.as_slice()),
        None,
    );
    let heating_with = with.get_heating_energy_kwh();
    let temps_with = with.get_temperatures();

    assert!(
        (heating_without - heating_with).abs() > 1e-9,
        "1 kW internal gains must change heating energy ({heating_without} vs {heating_with})"
    );
    assert!(
        temps_without != temps_with,
        "1 kW internal gains must change zone temperatures"
    );
    // Internal gains displace heating, never increase it.
    assert!(
        heating_with < heating_without,
        "equipment gains must reduce heating demand"
    );
}

#[test]
fn metering_is_deterministic() {
    let run = || {
        let mut model = test_model();
        let surrogates = test_surrogates();
        let lighting = test_lighting();
        let equipment = test_equipment();
        model.solve_timesteps(
            48,
            &surrogates,
            false,
            Some(&lighting),
            Some(equipment.as_slice()),
            None,
        );
        (
            model.get_hourly_heating_kwh().unwrap(),
            model.get_hourly_cooling_kwh().unwrap(),
            model.get_hourly_lighting_kwh().unwrap(),
            model.get_hourly_equipment_kwh().unwrap(),
        )
    };
    let (h1, c1, l1, e1) = run();
    let (h2, c2, l2, e2) = run();
    assert_eq!(h1, h2, "heating series must be deterministic");
    assert_eq!(c1, c2, "cooling series must be deterministic");
    assert_eq!(l1, l2, "lighting series must be deterministic");
    assert_eq!(e1, e2, "equipment series must be deterministic");
}

#[test]
fn metering_series_agree_with_annual_accumulators() {
    let mut model = test_model();
    let surrogates = test_surrogates();
    let lighting = test_lighting();
    let equipment = test_equipment();

    model.solve_timesteps(
        96,
        &surrogates,
        false,
        Some(&lighting),
        Some(equipment.as_slice()),
        None,
    );

    let heating: f64 = model.get_hourly_heating_kwh().unwrap().iter().sum();
    let cooling: f64 = model.get_hourly_cooling_kwh().unwrap().iter().sum();
    let annual_heating = model.get_heating_energy_kwh();
    let annual_cooling = model.get_cooling_energy_kwh();

    let heating_tol = 1e-9 * annual_heating.abs().max(1.0);
    let cooling_tol = 1e-9 * annual_cooling.abs().max(1.0);
    assert!(
        (heating - annual_heating).abs() <= heating_tol,
        "heating series sum {heating} must match accumulator {annual_heating}"
    );
    assert!(
        (cooling - annual_cooling).abs() <= cooling_tol,
        "cooling series sum {cooling} must match accumulator {annual_cooling}"
    );

    // Lighting/equipment sums match their closed-form totals.
    let dt = model.calculate_timestep_seconds();
    let lighting_sum: f64 = model.get_hourly_lighting_kwh().unwrap().iter().sum();
    let expected_lighting: f64 = (0..96)
        .map(|t| lighting.lighting_power(t) * dt / 3.6e6)
        .sum();
    assert!((lighting_sum - expected_lighting).abs() < 1e-9);

    let equipment_sum: f64 = model.get_hourly_equipment_kwh().unwrap().iter().sum();
    let expected_equipment = 96.0 * 1000.0 * dt / 3.6e6;
    assert!((equipment_sum - expected_equipment).abs() < 1e-9);
}

#[test]
fn metering_clone_drops_series() {
    let mut model = test_model();
    let surrogates = test_surrogates();
    let lighting = test_lighting();
    model.solve_timesteps(24, &surrogates, false, Some(&lighting), None, None);
    assert!(model.get_hourly_heating_kwh().is_some());

    let cloned = model.clone();
    assert!(
        cloned.get_hourly_heating_kwh().is_none(),
        "clone must drop the heating series"
    );
    assert!(cloned.get_hourly_cooling_kwh().is_none());
    assert!(cloned.get_hourly_lighting_kwh().is_none());
    assert!(cloned.get_hourly_equipment_kwh().is_none());
    // Hourly temperatures are dropped by the same Clone impl; the series
    // follow that established behaviour.
    assert!(cloned.get_hourly_temperatures().is_none());
}

#[test]
fn metering_auto_loaded_profile_equipment_stays_unapplied() {
    // Strict-gate guard: the Office factory profile carries ~8.5 kW of
    // plug loads. Auto-loaded profile equipment is intentionally NOT
    // applied to the thermal path (it would move ASHRAE baselines), so the
    // metered equipment series on the all-None path must be all zeros.
    // (If the profile JSON is missing, auto-load degrades to no loads and
    // the assertion still holds.)
    let mut model = test_model();
    let surrogates = test_surrogates();
    model.solve_timesteps(48, &surrogates, false, None, None, None);

    let equipment = model
        .get_hourly_equipment_kwh()
        .expect("equipment series must be recorded");
    assert_eq!(equipment.len(), 48);
    assert!(
        equipment.iter().all(|&v| v == 0.0),
        "auto-loaded profile equipment must not be metered as applied load"
    );
}

#[test]
fn metering_series_none_before_simulation() {
    let model = test_model();
    assert!(model.get_hourly_heating_kwh().is_none());
    assert!(model.get_hourly_cooling_kwh().is_none());
    assert!(model.get_hourly_lighting_kwh().is_none());
    assert!(model.get_hourly_equipment_kwh().is_none());
}

#[test]
fn equipment_spec_validation_rejects_bad_input() {
    let valid = EquipmentSpec {
        equipment_type: "computer".to_string(),
        rated_power_w: 150.0,
        count: 50,
        hourly_fractions: [1.0; 24],
        radiative_fraction: 0.3,
        convective_fraction: 0.7,
        mass_coupling_factor: 0.2,
    };
    assert!(valid.validate(0).is_empty());

    let bad_type = EquipmentSpec {
        equipment_type: "toaster".to_string(),
        ..valid.clone()
    };
    assert!(!bad_type.validate(0).is_empty());

    let bad_power = EquipmentSpec {
        rated_power_w: -5.0,
        ..valid.clone()
    };
    assert!(!bad_power.validate(0).is_empty());

    let mut bad_fractions = valid.clone();
    bad_fractions.hourly_fractions[3] = 1.5;
    assert!(!bad_fractions.validate(0).is_empty());

    let bad_split = EquipmentSpec {
        radiative_fraction: 0.5,
        convective_fraction: 0.7,
        ..valid.clone()
    };
    assert!(!bad_split.validate(0).is_empty());
}

#[test]
fn equipment_spec_build_produces_working_equipment() {
    let spec = EquipmentSpec {
        equipment_type: "server".to_string(),
        rated_power_w: 500.0,
        count: 2,
        hourly_fractions: [1.0; 24],
        radiative_fraction: 0.5,
        convective_fraction: 0.5,
        mass_coupling_factor: 0.8,
    };
    let item = spec
        .build("spec-test".to_string())
        .expect("valid spec must build");
    // 2 × 500 W always on.
    assert_eq!(item.power_at_hour(0), 1000.0);
    assert_eq!(item.power_at_hour(5133), 1000.0);
    assert_eq!(item.convective_gains(7), 500.0);
    assert_eq!(item.radiative_gains(7), 500.0);

    let bad = EquipmentSpec {
        equipment_type: "toaster".to_string(),
        ..spec
    };
    assert!(bad.build("bad".to_string()).is_err());
}

#[test]
fn simulation_output_omits_unrecorded_series() {
    // Wire-shape stability: without metering, the new fields are omitted
    // from JSON exactly like effective_solver.
    let output = SimulationOutput::default();
    let json = serde_json::to_string(&output).expect("must serialize");
    for field in [
        "hourly_heating_kwh",
        "hourly_cooling_kwh",
        "hourly_lighting_kwh",
        "hourly_equipment_kwh",
    ] {
        assert!(
            !json.contains(field),
            "{field} must be omitted when None (wire shape unchanged)"
        );
    }

    // And old JSON without the fields still deserializes.
    let legacy = r#"{"eui":1.0,"total_energy":2.0,"peak_heating_load":3.0,
        "peak_cooling_load":4.0,"heating_energy":5.0,"cooling_energy":6.0,
        "unmet_heating_hours":0.0,"unmet_cooling_hours":0.0}"#;
    let parsed: SimulationOutput = serde_json::from_str(legacy).expect("legacy JSON must parse");
    assert!(parsed.hourly_heating_kwh.is_none());
}

/// Issue #4101 — REST end-to-end: schema-supplied lighting density and
/// equipment specs flow through `run_simulation` into the returned
/// `SimulationOutput` metering series with hand-computable values.
#[test]
fn rest_simulation_exposes_schema_lighting_and_equipment_metering() {
    use fluxion::api::schema::{
        ConstructionSet, ControlSet, EquipmentSpec, Geometry, ScheduleSet, SchemaMetadata,
        SchemaVersion, SimulationOutput, SimulationSchemaV1, WeatherData,
    };
    use fluxion::api::server::run_simulation;
    use fluxion::sim::thermal_selector::ThermalSelector;

    let mut schema = SimulationSchemaV1 {
        version: SchemaVersion::V1,
        metadata: SchemaMetadata::default(),
        geometry: Geometry::default(),
        constructions: ConstructionSet::default(),
        schedules: ScheduleSet::default(),
        weather: WeatherData::default(),
        controls: ControlSet::default(),
        output: SimulationOutput::default(),
    };
    // Explicit 10 W/m² lighting, on 08:00–18:00 (Monday row drives the
    // schema lighting schedule's `value(h)`).
    schema.schedules.lighting_power_density_w_m2 = Some(10.0);
    for h in 0..24 {
        let frac = if (8..18).contains(&h) { 1.0 } else { 0.0 };
        schema.schedules.lighting.set_hour_for_day(0, h, frac);
    }
    // Explicit 500 W computer, always on.
    schema.schedules.equipment = vec![EquipmentSpec {
        equipment_type: "computer".to_string(),
        rated_power_w: 500.0,
        count: 1,
        hourly_fractions: [1.0; 24],
        radiative_fraction: 0.2,
        convective_fraction: 0.8,
        mass_coupling_factor: 0.0,
    }];

    let out = run_simulation(
        &schema,
        1,
        false,
        ThermalSelector::default(),
        "metering-test",
    )
    .expect("REST run_simulation succeeds");

    let area = schema.geometry.total_floor_area;
    let lighting = out.hourly_lighting_kwh.expect("lighting series recorded");
    let equipment = out.hourly_equipment_kwh.expect("equipment series recorded");
    assert_eq!(lighting.len(), 8760, "one entry per hourly timestep");
    assert_eq!(equipment.len(), 8760, "one entry per hourly timestep");
    for t in 0..8760 {
        let frac = if (8..18).contains(&(t % 24)) {
            1.0
        } else {
            0.0
        };
        let expected_lighting = 10.0 * area * frac / 1000.0;
        assert!(
            (lighting[t] - expected_lighting).abs() < 1e-9,
            "t={t}: lighting {} != expected {expected_lighting}",
            lighting[t]
        );
        let expected_equipment = 500.0 / 1000.0;
        assert!(
            (equipment[t] - expected_equipment).abs() < 1e-9,
            "t={t}: equipment {} != expected {expected_equipment}",
            equipment[t]
        );
    }
    // Heating/cooling series are recorded alongside.
    assert_eq!(out.hourly_heating_kwh.map(|s| s.len()), Some(8760));
    assert_eq!(out.hourly_cooling_kwh.map(|s| s.len()), Some(8760));
}
