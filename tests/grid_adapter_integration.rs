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

    // Baseline regeneration hook (the comment below has promised this since
    // round 39; restored with the round-73 re-record). Run with
    // FLUXION_PRINT_GRID_BASELINE=1 -- --nocapture to print the exact
    // full-precision baseline values.
    if std::env::var("FLUXION_PRINT_GRID_BASELINE").is_ok() {
        println!(
            "GRID_BASELINE thermal_kwh={:.17} electrical_load_kwh={:.17} pv_kwh={:.17} battery_charge_kwh={:.17} battery_discharge_kwh={:.17} grid_import_kwh={:.17} grid_export_kwh={:.17} peak_net_demand_kw={:.17} final_soc_fraction={:.17}",
            expected_thermal_kwh,
            results.total_electrical_load_kwh,
            results.total_pv_kwh,
            results.total_battery_charge_kwh,
            results.total_battery_discharge_kwh,
            results.total_grid_import_kwh,
            results.total_grid_export_kwh,
            results.peak_net_demand_kw,
            results.final_soc_fraction
        );
    }

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
    // Re-recorded 2026-10-15 by the ASHRAE 140 Table 5-42 infiltration
    // altitude adjustment (loop round 39, commit 1878300e): the engine uses a
    // fixed sea-level air density, so the spec-adjusted input is 0.414 ach,
    // not 0.5 (0.5 x 1.0156/1.2255); this lowers Case 600 January heating.
    // Thermal 150.25454453392467 -> 145.58652387252914 kWh; electrical load
    // 50.084848177974884 -> 48.52884129084303 kWh (thermal/3); grid import
    // 37.321170194804544 -> 35.803981845836475 kWh; peak net demand
    // 1.3152976644842214 -> 1.2751549581267079 kW. PV, battery and SoC
    // unchanged. Genuine physics change, not a tuned constant.
    //
    // Re-recorded 2026-10-19 (loop round 73) by the R71 sub-hour air-driver
    // default flip (#1522; env-gated `FLX_R71_AIR_CAP` harness in round 72,
    // PR #4505): the 5R1C air-node advance changed, so the Case 600 January
    // heating baseline moves. Thermal 138.81059158981444 ->
    // 138.16772740591088109 kWh; electrical load 46.270197196604805 ->
    // 46.05590913530362229 kWh (thermal/3); grid import 33.534319651216784 ->
    // 33.31587775390797646 kWh; peak net demand 1.2180336040388786 ->
    // 1.21467317388113916 kW. PV, battery and SoC unchanged. Genuine physics
    // change, not a tuned constant. (Prior re-record 2026-10-19 round 71, D5
    // flip: thermal 145.58652387252914 -> 138.81059158981444.)
    //
    // Cross-checks the physics wiring, not just the adapter: thermal/3 ==
    // electrical_load, and the battery stock (0.5 → 0.1 of 10 kWh) equals
    // charge − discharge. Regenerate by running with
    // FLUXION_PRINT_GRID_BASELINE=1 -- --nocapture.
    // Re-recorded 2026-10-20 (loop round 78): the coupled single-N default
    // flip (`FLX_R77_COUPLED_N` default 12, Alex-directed, #3983 §63/§64)
    // changed the 5R1C advance again: thermal 138.16772740591088109 ->
    // 89.10660390761354677 kWh; electrical load 46.05590913530362229 ->
    // 29.70220130253784419 kWh (thermal/3); grid import 33.31587775390797646
    // -> 17.11561664071873068 kWh; peak net demand 1.21467317388113916 ->
    // 0.79829515924816252 kW. PV unchanged; battery now cycles (charge
    // 1.60696394365722406, discharge 5.60696394365722384, export
    // 0.08457704966616977) because the lighter January heating load no
    // longer pins the battery at its charge floor. Genuine physics change,
    // not a tuned constant. (Prior re-records: round 71 D5 flip, round 73
    // R71 air-driver flip, 2026-10-07 LIMIT-33.)
    //
    // Re-recorded 2026-10-21 (loop round 85): the per-class coupled
    // sub-step mass advance promoted to the engine default
    // (`FLX_R84_PERCLASS`, Alex-directed 2026-10-21 21:18 ET, #3983 §65)
    // changed the 5R1C advance again. Thermal 89.10660390761354677 ->
    // 90.89525579502979724 kWh; electrical load 29.70220130253784419 ->
    // 30.29841859834327167 kWh (thermal/3); grid import
    // 17.11561664071873068 -> 17.70933307992547512 kWh; peak net demand
    // 0.79829515924816252 -> 0.81848002062825786 kW; battery charge
    // 1.60696394365722406 -> 1.49900965903126360 and discharge
    // 5.60696394365722384 -> 5.49900965903126249; export
    // 0.08457704966616977 -> 0.07889524521217187. PV unchanged; battery
    // still cycles; SoC still 0.1. Genuine physics change, not a tuned
    // constant.
    //
    // Re-recorded 2026-10-22 (loop round 89): the ISO 13790 §C.3
    // surface-pool mass share promoted to the engine default
    // (`FLX_R88_PHIST_MASS_SHARE`, Alex-directed 2026-10-22 14:57 ET,
    // round-88 arm #4543) changed the 5R1C advance again. Thermal
    // 90.89525579502979724 -> 91.03243793720206156 kWh; electrical load
    // 30.29841859834327167 -> 30.34414597906735267 kWh (thermal/3);
    // grid import 17.70933307992547512 -> 17.75529617272509597 kWh;
    // peak net demand 0.81848002062825786 -> 0.81835574092125685 kW;
    // battery charge 1.49900965903126360 -> 1.48987563680459334 and
    // discharge 5.49900965903126249 -> 5.48987563680459356; export
    // 0.07889524521217187 -> 0.07841450720024182. PV unchanged; battery
    // still cycles; SoC still 0.1. Genuine physics change, not a tuned
    // constant.
    //
    // Re-recorded 2026-10-22 (loop round 89b, item 2a): the mass-node
    // full time-coverage fix promoted to the engine default
    // (`FLX_R89_MASS_DT_ALIGN`, flip commit ceb19b0b). Thermal
    // 91.03243793720206156 -> 148.75919979557599504 kWh; electrical load
    // 30.34414597906735267 -> 49.58639993185867212 kWh (thermal/3);
    // grid import 17.75529617272509597 -> 36.82096414386504080 kWh;
    // peak net demand 0.81835574092125685 -> 1.30003625310234994 kW.
    // PV unchanged; the heavier January heating load now pins the
    // battery at its charge floor (charge 0, discharge 4.0, export 0);
    // SoC still 0.1. Genuine physics change, not a tuned constant.
    // Re-recorded 2026-10-22 (loop round 90, R77 lag-filter removal
    // default flip, `FLX_R90_NO_LAG_FILTER`, §LIMIT-28 round-90 record).
    // Thermal 148.75919979557599504 -> 148.72623543486619724 kWh;
    // electrical load 49.58639993185867212 -> 49.57541181162207522 kWh
    // (thermal/3); grid import 36.82096414386504080 ->
    // 36.80961168993702870 kWh; peak net demand 1.30003625310234994 ->
    // 1.30001939788464482 kW. PV unchanged; battery still pinned at its
    // charge floor (charge 0, discharge 4.0, export 0); SoC still 0.1.
    // Genuine physics change, not a tuned constant.
    // Re-recorded 2026-10-23 (loop round 91, flip 2): the incumbent
    // density-driven solar split retired from the default (`FLX_R91_SOL2ST`,
    // §LIMIT-28 round-91 record) moves the Case 600 January battery
    // baseline: thermal 148.72623543486619724 -> 149.00055716995336752 kWh; electrical load 49.57541181162207522 ->
    // 49.66685238998446295 kWh (thermal/3); grid import 36.80961168993702870 -> 36.90089020695563704 kWh; peak net demand
    // 1.30001939788464482 -> 1.30254369998193242 kW; battery
    // charge/discharge, grid export and PV unchanged (0.0/4.0/0.0).
    // Genuine physics change,
    // not a tuned constant.
    // Re-recorded 2026-10-23 (loop round 91, flip 3): the Table 5-138
    // per-class solar routing default (`FLX_R91_T5138`, §LIMIT-28
    // round-91 record) moves the Case 600 January battery baseline:
    // thermal 149.00055716995336752 -> 146.99272085576484415 kWh;
    // electrical load 49.66685238998446295 -> 48.99757361858829086 kWh
    // (thermal/3); grid import 36.90089020695563704 ->
    // 36.23090323412515090 kWh; peak net demand 1.30254369998193242 ->
    // 1.28615852919839990 kW. PV unchanged; battery still pinned at its
    // charge floor (charge 0, discharge 4.0, export 0); SoC still 0.1.
    // Genuine physics change, not a tuned constant.
    // Re-recorded 2026-10-24 (loop round 93, physics/r93-t5138-airshare): the
    // ISO 13790 .C3 lag-site air-share term retires under the T5138 class
    // route (FLX_R93_T5138_AIRSHARE falsy-token restore; premise verified
    // against ASHRAE 140-2017 Table 5-138 + ISO 13790 Annex C) moves
    // thermal 146.99272085576484415 -> 148.08443806665744091 kWh (+0.74 %),
    // electrical load 48.99757361858829086 -> 49.36147935555248267 kWh,
    // grid import 36.23090323412515090 -> 36.59605677268299928 kWh, peak net
    // demand 1.28615852919839990 -> 1.29411537792255760 kW (via
    // FLUXION_PRINT_GRID_BASELINE=1). PV unchanged; battery still pinned at
    // its charge floor (charge 0, discharge 4.0, export 0); SoC still 0.1.
    // Genuine physics change, not a tuned constant.
    const EXPECTED_THERMAL_KWH: f64 = 148.08443806665744091;
    const EXPECTED_ELECTRICAL_LOAD_KWH: f64 = 49.36147935555248267;
    const EXPECTED_PV_KWH: f64 = 8.92391297937676597;
    const EXPECTED_BATTERY_CHARGE_KWH: f64 = 0.0;
    const EXPECTED_BATTERY_DISCHARGE_KWH: f64 = 4.0;
    const EXPECTED_GRID_IMPORT_KWH: f64 = 36.59605677268299928;
    const EXPECTED_GRID_EXPORT_KWH: f64 = 0.0;
    const EXPECTED_PEAK_NET_DEMAND_KW: f64 = 1.29411537792255760;
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
