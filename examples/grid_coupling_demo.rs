//! Grid-coupling demo (Issue #4005).
//!
//! Demonstrates the phase-1 thermal→electrical post-processing path: a real
//! ASHRAE 140 Case 600 `ThermalModel::step_physics` timestep loop (Denver TMY
//! weather, 48-hour January window) feeds each step's HVAC thermal load into
//! `GridAdapter::step`, and `GridAdapter::post_process` folds the recorded
//! series into an additive `ElectricalResults` block.
//!
//! The per-timestep `step()` call inside the loop is also the exact call sequence
//! a phase-2 in-loop demand-response controller will use — the batch path here is
//! literally `step()` folded over the same recorded inputs.
//!
//! Run with: `cargo run --example grid_coupling_demo --features grid`

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::grid_adapter::{GridAdapter, GridAdapterConfig, GridTimestepInput};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

fn main() {
    let config = GridAdapterConfig {
        cop: 3.0,
        pv_panel_area_m2: 10.0,
        pv_rated_dc_power_w: 2000.0,
        inverter_efficiency: 0.96,
        battery_capacity_wh: 10_000.0,
        battery_max_power_w: 5_000.0,
        battery_initial_soc_fraction: 0.5,
        grid_import_cost_per_wh: 0.00015, // $0.15/kWh
    };
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();
    let mut adapter = GridAdapter::new(&config);

    // 48-hour January run. Thermal inputs are recorded once and reused for the
    // batch pass below — never reconstructed from electrical outputs.
    let mut series = Vec::with_capacity(48);
    for step in 0..48 {
        let w = weather.get_hourly_data(step).unwrap();
        let hvac_kwh = model.step_physics(step, w.dry_bulb_temp, 3600.0);
        // Phase-1 pattern: thermal step in, electrical step out — per timestep.
        // `ghi` is a plane-of-array irradiance proxy for this demo.
        let input = GridTimestepInput {
            thermal_load_w: hvac_kwh * 1000.0, // kWh per 1-h step → average W
            plane_irradiance_wm2: w.ghi.max(0.0),
            ambient_temp_c: w.dry_bulb_temp,
            dt_seconds: 3600.0,
        };
        let out = adapter.step(&input);
        if step % 12 == 0 {
            println!(
                "hour {step:>2}: thermal {:>7.1} W → electrical {:>6.1} W, PV {:>6.1} W, SoC {:>4.1}%",
                input.thermal_load_w,
                out.electrical_load_w,
                out.pv_ac_w,
                out.soc_fraction * 100.0,
            );
        }
        series.push(input);
    }

    // Batch post-processing over the recorded per-step inputs (identical to
    // folding step() — a fresh adapter proves the equivalence).
    let mut batch_adapter = GridAdapter::new(&config);
    let results = batch_adapter.post_process(&series);

    println!("\n48-h electrical summary (additive block):");
    println!(
        "  thermal energy:        {:>8.2} kWh",
        results.total_thermal_kwh
    );
    println!(
        "  electrical load:       {:>8.2} kWh",
        results.total_electrical_load_kwh
    );
    println!("  PV generation:         {:>8.2} kWh", results.total_pv_kwh);
    println!(
        "  battery charged:       {:>8.2} kWh",
        results.total_battery_charge_kwh
    );
    println!(
        "  battery discharged:    {:>8.2} kWh",
        results.total_battery_discharge_kwh
    );
    println!(
        "  grid import:           {:>8.2} kWh",
        results.total_grid_import_kwh
    );
    println!(
        "  grid export:           {:>8.2} kWh",
        results.total_grid_export_kwh
    );
    println!(
        "  peak net demand:       {:>8.2} kW",
        results.peak_net_demand_kw
    );
    println!(
        "  final battery SoC:     {:>8.1} %",
        results.final_soc_fraction * 100.0
    );
}
