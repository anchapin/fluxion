//! Grid-adapter integration tests (Issue #4005).
//!
//! End-to-end: a real ASHRAE 140 Case 600 `ThermalModel::step_physics` timestep loop
//! (Denver TMY weather, 48-hour January window) feeds each step's HVAC thermal load
//! into `GridAdapter::step` — the phase-1 post-processing pattern and the exact call
//! sequence a phase-2 in-loop demand-response controller will use.
//!
//! Lives in `tests/` (not `src/sim/`) to avoid sim→validation cycle edges — the same
//! precedent as `gauge_dispatcher_cases` (see `scripts/check_ashrae_cases_cycle.py`).
//!
//! Run with: `cargo test --features grid -p fluxion --test grid_adapter_integration`

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::grid_adapter::{
    ElectricalResults, GridAdapter, GridAdapterConfig, GridTimestepInput,
};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

fn test_config() -> GridAdapterConfig {
    GridAdapterConfig {
        cop: 3.0,
        pv_panel_area_m2: 10.0,
        pv_rated_dc_power_w: 2000.0,
        inverter_efficiency: 0.96,
        battery_capacity_wh: 10_000.0,
        battery_max_power_w: 5_000.0,
        battery_initial_soc_fraction: 0.5,
        grid_import_cost_per_wh: 0.00015, // $0.15/kWh
    }
}

/// Run 48 January hours of Case 600. When `drive_adapter` is true, each step's
/// thermal load is also fed through `GridAdapter::step` (the phase-1 pattern).
/// Returns the per-step thermal series (kWh) and, when driven, the adapter output.
fn run_case600_january(drive_adapter: bool) -> (Vec<f64>, Option<ElectricalResults>) {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();
    // Separate adapter for the in-loop step() driving: post_process() below gets
    // a fresh adapter so the battery is never stepped twice over the same series.
    let mut step_adapter = GridAdapter::new(&test_config());

    let mut thermal_kwh = Vec::with_capacity(48);
    let mut series = Vec::with_capacity(48);
    for step in 0..48 {
        let w = weather.get_hourly_data(step).unwrap();
        let hvac_kwh = model.step_physics(step, w.dry_bulb_temp, 3600.0);
        thermal_kwh.push(hvac_kwh);
        let input = GridTimestepInput {
            thermal_load_w: hvac_kwh * 1000.0,
            // `ghi` is a plane-of-array irradiance proxy for this test.
            plane_irradiance_wm2: w.ghi.max(0.0),
            ambient_temp_c: w.dry_bulb_temp,
            dt_seconds: 3600.0,
        };
        if drive_adapter {
            // Phase-1 pattern: thermal step in, electrical step out — per timestep.
            let _ = step_adapter.step(&input);
        }
        series.push(input);
    }
    let results = drive_adapter.then(|| {
        let mut batch_adapter = GridAdapter::new(&test_config());
        batch_adapter.post_process(&series)
    });
    (thermal_kwh, results)
}

#[test]
fn test_thermal_physics_bit_identical_with_and_without_adapter() {
    // The adapter is strictly post-processing: stepping it alongside the thermal
    // loop must not change the thermal results by even one ulp.
    let (plain, _) = run_case600_january(false);
    let (with_adapter, _) = run_case600_january(true);
    assert_eq!(plain.len(), with_adapter.len());
    for (a, b) in plain.iter().zip(with_adapter.iter()) {
        assert_eq!(a, b, "thermal series must be bit-identical");
        assert!(a.is_finite());
    }
    // Sanity: January in Denver means real heating load.
    let total: f64 = plain.iter().sum();
    assert!(total > 0.0, "expected heating load in January, got {total}");
}

#[test]
fn test_step_loop_matches_post_process_on_engine_series() {
    // post_process must be exactly the step() fold — the property phase 2 relies on.
    let (_, results) = run_case600_january(true);
    let results = results.unwrap();
    assert_eq!(results.num_timesteps, 48);

    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();
    let mut adapter = GridAdapter::new(&test_config());
    let mut import_kwh = 0.0;
    for step in 0..48 {
        let w = weather.get_hourly_data(step).unwrap();
        let hvac_kwh = model.step_physics(step, w.dry_bulb_temp, 3600.0);
        let out = adapter.step(&GridTimestepInput {
            thermal_load_w: hvac_kwh * 1000.0,
            plane_irradiance_wm2: w.ghi.max(0.0),
            ambient_temp_c: w.dry_bulb_temp,
            dt_seconds: 3600.0,
        });
        import_kwh += out.grid_import_wh / 1000.0;
    }
    assert!((results.total_grid_import_kwh - import_kwh).abs() < 1e-12);
    assert!((results.final_soc_fraction - adapter.soc_fraction()).abs() < 1e-12);
}

#[test]
fn test_electrical_results_match_recorded_baseline() {
    let (thermal_kwh, results) = run_case600_january(true);
    let results = results.unwrap();
    let expected_thermal_kwh: f64 = thermal_kwh.iter().map(|k| k.max(0.0)).sum();

    // Recorded baseline for: ASHRAE 140 Case 600, Denver TMY steps 0..48
    // (January), `test_config()`, `ghi`-as-POA proxy. All 48 steps are heating
    // load (no clamping needed); the battery rides the SoC floor at 10%.
    //
    // Re-recorded 2026-10-07 by the LIMIT-33 fix (issue #4314, Option A
    // state feedback): the #4241 residual load now drives the 5R1C zone air
    // to the active setpoint on unclamped conditioned hours (controlled
    // state persisted in `setpoints.temperatures`, free-float state kept in
    // `mass.air_temperatures`), so the storage term charges once per
    // recovery transient instead of re-charging every hour. Thermal
    // 178.31811814894678 -> 139.35135840182244 kWh. Genuine physics change,
    // not a tuned constant.
    //
    // Re-recorded 2026-09-29 by PR #4258 (issue #4241): the ideal-HVAC
    // conductance now includes h_ve (5R1C/9R4C unification) and the load
    // uses the discrete zone energy-balance residual. Genuine physics
    // change, not a tuned constant.
    //
    // Re-recorded 2026-09-28 by PR #4222 (issue #4166): the per-surface
    // exterior boundary corrected the sol-air longwave sign to the ASHRAE
    // direction (cold sky now depresses sol-air), raising January heating
    // load. Genuine physics change, not a tuned constant.
    //
    // Re-recorded 2026-10-09 by the WINDOW-01 fix (loop round 7): the ASHRAE
    // 140 spec window U-value 3.0 W/m2K now threads through the CaseSpec path
    // (previously ~2.27 W/m2K effective on the strict path), raising Case 600
    // January heating. Thermal 139.35135840182244 -> 150.25454453392467 kWh;
    // electrical load 46.45045280060749 -> 50.084848177974884 kWh (thermal/3);
    // grid import 33.70974375036340 -> 37.321170194804544 kWh; peak net demand
    // 1.22160526193861 -> 1.3152976644842214 kW. PV, battery and SoC unchanged.
    // Genuine physics change, not a tuned constant.
    //
    // Cross-checks the physics wiring, not just the adapter: thermal/3 ==
    // electrical_load, and the battery stock (0.5 → 0.1 of 10 kWh) equals
    // charge − discharge. Regenerate by running with
    // FLUXION_PRINT_GRID_BASELINE=1 -- --nocapture.
    const EXPECTED_THERMAL_KWH: f64 = 150.25454453392467;
    const EXPECTED_ELECTRICAL_LOAD_KWH: f64 = 50.084848177974884;
    const EXPECTED_PV_KWH: f64 = 8.92391297937677;
    const EXPECTED_BATTERY_CHARGE_KWH: f64 = 0.00000000000000;
    const EXPECTED_BATTERY_DISCHARGE_KWH: f64 = 4.00000000000000;
    const EXPECTED_GRID_IMPORT_KWH: f64 = 37.321170194804544;
    const EXPECTED_GRID_EXPORT_KWH: f64 = 0.00000000000000;
    const EXPECTED_PEAK_NET_DEMAND_KW: f64 = 1.3152976644842214;
    const EXPECTED_FINAL_SOC_FRACTION: f64 = 0.1;

    let tol = 1e-9;
    assert!((expected_thermal_kwh - EXPECTED_THERMAL_KWH).abs() < tol);
    assert!((results.total_electrical_load_kwh - EXPECTED_ELECTRICAL_LOAD_KWH).abs() < tol);
    assert!((results.total_pv_kwh - EXPECTED_PV_KWH).abs() < tol);
    assert!((results.total_battery_charge_kwh - EXPECTED_BATTERY_CHARGE_KWH).abs() < tol);
    assert!((results.total_battery_discharge_kwh - EXPECTED_BATTERY_DISCHARGE_KWH).abs() < tol);
    assert!((results.total_grid_import_kwh - EXPECTED_GRID_IMPORT_KWH).abs() < tol);
    assert!((results.total_grid_export_kwh - EXPECTED_GRID_EXPORT_KWH).abs() < tol);
    assert!((results.peak_net_demand_kw - EXPECTED_PEAK_NET_DEMAND_KW).abs() < tol);
    assert!((results.final_soc_fraction - EXPECTED_FINAL_SOC_FRACTION).abs() < 1e-12);
}
