//! Adapter: per-timestep thermal→electrical coupling via `fluxion-grid` (Issue #4005).
//!
//! [`GridAdapter`] wraps `fluxion_grid::{ThermalElectricalCoupler, PvSystem,
//! BatteryStorage}` behind a per-timestep [`GridAdapter::step`] API — thermal state in,
//! electrical state out, each timestep — with the batch [`GridAdapter::post_process`]
//! built directly on top of `step()`.
//!
//! Phase 1 is post-processing only: the caller runs the thermal simulation (or replays a
//! recorded thermal-load series), feeds each timestep's thermal load through `step()`, and
//! gets back an additive [`ElectricalResults`] block. The adapter never touches the thermal
//! solver, timesteps, tolerances, or constants, so physics results are identical with the
//! `grid` feature on or off.
//!
//! Phase 2 (in-loop co-simulation for demand-response / pre-cooling controllers) is a
//! wiring change, not a rewrite: a future controller calls `step()` inside the timestep
//! loop — where it can observe each step's electrical load and dispatch the battery
//! against tariffs or demand charges — instead of calling `post_process()` after the run.
//! Solar *thermal* panels are deliberately out of scope here: they exchange heat with the
//! thermal model directly and belong in `crate::solar`, not in this electrical adapter.
//!
//! Compile with `--features grid` to enable (default off).

use fluxion_grid::{BatteryStorage, PvSystem, ThermalElectricalCoupler};
use serde::{Deserialize, Serialize};

/// Configuration for [`GridAdapter`].
///
/// All fields are plain physical parameters; the adapter performs no tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridAdapterConfig {
    /// Heat-pump COP used for thermal→electrical conversion (electrical = thermal / COP).
    pub cop: f64,
    /// Total PV panel area (m²).
    pub pv_panel_area_m2: f64,
    /// PV rated DC power at STC (W).
    pub pv_rated_dc_power_w: f64,
    /// Inverter efficiency (0.0–1.0).
    pub inverter_efficiency: f64,
    /// Battery usable capacity (Wh).
    pub battery_capacity_wh: f64,
    /// Battery max charge/discharge power (W).
    pub battery_max_power_w: f64,
    /// Battery initial SoC as fraction of capacity (0.0–1.0).
    pub battery_initial_soc_fraction: f64,
    /// Grid import cost ($/Wh), passed through to the battery dispatch policy.
    /// The phase-1 self-consumption policy treats this as informational; it is reserved
    /// for phase-2 tariff-aware dispatch.
    pub grid_import_cost_per_wh: f64,
}

/// Per-timestep input to [`GridAdapter::step`].
///
/// This is the phase-2 contract surface: a future in-loop controller supplies one of
/// these per thermal timestep and reads the [`GridTimestepOutput`] back.
#[derive(Debug, Clone, Copy)]
pub struct GridTimestepInput {
    /// HVAC thermal load for this timestep (W, signed): positive = heating,
    /// negative = cooling — the engine's `step_physics` sign convention. The
    /// adapter couples the *magnitude* (both modes consume electricity); the
    /// signed mode is preserved in this input for future phase-2 control use.
    pub thermal_load_w: f64,
    /// Plane-of-array solar irradiance (W/m², >= 0).
    pub plane_irradiance_wm2: f64,
    /// Outdoor dry-bulb temperature (°C).
    pub ambient_temp_c: f64,
    /// Timestep duration (s, > 0).
    pub dt_seconds: f64,
}

/// Per-timestep output of [`GridAdapter::step`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridTimestepOutput {
    /// Building electrical load converted from thermal (W).
    pub electrical_load_w: f64,
    /// PV AC generation (W).
    pub pv_ac_w: f64,
    /// Battery charge energy this step (Wh).
    pub battery_charge_wh: f64,
    /// Battery discharge energy this step (Wh).
    pub battery_discharge_wh: f64,
    /// Grid import energy this step (Wh).
    pub grid_import_wh: f64,
    /// Grid export energy this step (Wh).
    pub grid_export_wh: f64,
    /// Battery SoC fraction after this step.
    pub soc_fraction: f64,
}

/// Additive electrical results block produced by [`GridAdapter::post_process`].
///
/// All energy fields are in kWh; the block is purely additive to whatever thermal
/// results the caller already has.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectricalResults {
    /// Number of timesteps folded.
    pub num_timesteps: usize,
    /// Total HVAC thermal energy (kWh).
    pub total_thermal_kwh: f64,
    /// Total building electrical load from thermal (kWh).
    pub total_electrical_load_kwh: f64,
    /// Total PV AC generation (kWh).
    pub total_pv_kwh: f64,
    /// Total battery charge energy (kWh).
    pub total_battery_charge_kwh: f64,
    /// Total battery discharge energy (kWh).
    pub total_battery_discharge_kwh: f64,
    /// Total grid import energy (kWh).
    pub total_grid_import_kwh: f64,
    /// Total grid export energy (kWh).
    pub total_grid_export_kwh: f64,
    /// Peak net grid import power (kW) — the demand-charge basis.
    pub peak_net_demand_kw: f64,
    /// Battery SoC fraction after the last step.
    pub final_soc_fraction: f64,
}

/// Per-timestep thermal→electrical adapter (Issue #4005).
///
/// Holds the `fluxion-grid` component models and their state (battery SoC). The
/// [`step`](Self::step) method is the primitive; [`post_process`](Self::post_process)
/// folds it over a series.
pub struct GridAdapter {
    coupler: ThermalElectricalCoupler,
    pv: PvSystem,
    battery: BatteryStorage,
    grid_import_cost_per_wh: f64,
}

impl GridAdapter {
    /// Build an adapter from a [`GridAdapterConfig`].
    pub fn new(config: &GridAdapterConfig) -> Self {
        Self {
            coupler: ThermalElectricalCoupler::new(config.cop),
            pv: PvSystem::new(
                config.pv_panel_area_m2,
                config.pv_rated_dc_power_w,
                config.inverter_efficiency,
            ),
            battery: BatteryStorage::new(
                config.battery_capacity_wh,
                config.battery_max_power_w,
                config.battery_initial_soc_fraction,
            ),
            grid_import_cost_per_wh: config.grid_import_cost_per_wh,
        }
    }

    /// Step the adapter one timestep: thermal state in, electrical state out.
    ///
    /// This is the phase-2 contract: an in-loop demand-response controller calls this
    /// inside the thermal timestep loop. Conversion order per step:
    /// 1. thermal load → electrical load via COP (`thermal / cop`),
    /// 2. PV AC generation from plane irradiance and ambient temperature,
    /// 3. battery self-consumption dispatch against the net load.
    pub fn step(&mut self, input: &GridTimestepInput) -> GridTimestepOutput {
        // Signed thermal semantics: positive = heating, negative = cooling (the
        // engine's step_physics convention). Both modes are thermal energy the
        // HVAC plant must serve, so the electrical conversion couples the
        // magnitude — never a negative electrical load.
        let thermal_load_w = input.thermal_load_w.abs();
        let electrical_load_w = self.coupler.thermal_to_electrical_simple(thermal_load_w);
        let pv_ac_w = self
            .pv
            .ac_power(input.plane_irradiance_wm2, input.ambient_temp_c);
        let (battery_charge_wh, battery_discharge_wh, grid_import_wh, grid_export_wh) =
            self.battery.step(
                pv_ac_w,
                electrical_load_w,
                self.grid_import_cost_per_wh,
                input.dt_seconds,
            );
        GridTimestepOutput {
            electrical_load_w,
            pv_ac_w,
            battery_charge_wh,
            battery_discharge_wh,
            grid_import_wh,
            grid_export_wh,
            soc_fraction: self.battery.soc_fraction(),
        }
    }

    /// Batch post-processing: fold [`step`](Self::step) over a thermal-load series.
    ///
    /// Phase-1 entry point. Because this is literally `step()` folded, the
    /// `step()`-loop and `post_process()` always agree exactly (covered by test).
    pub fn post_process(&mut self, series: &[GridTimestepInput]) -> ElectricalResults {
        let mut results = ElectricalResults {
            num_timesteps: series.len(),
            total_thermal_kwh: 0.0,
            total_electrical_load_kwh: 0.0,
            total_pv_kwh: 0.0,
            total_battery_charge_kwh: 0.0,
            total_battery_discharge_kwh: 0.0,
            total_grid_import_kwh: 0.0,
            total_grid_export_kwh: 0.0,
            peak_net_demand_kw: 0.0,
            final_soc_fraction: self.battery.soc_fraction(),
        };
        for input in series {
            let dt_hours = input.dt_seconds / 3600.0;
            let out = self.step(input);
            results.total_thermal_kwh += input.thermal_load_w.abs() * dt_hours / 1000.0;
            results.total_electrical_load_kwh += out.electrical_load_w * dt_hours / 1000.0;
            results.total_pv_kwh += out.pv_ac_w * dt_hours / 1000.0;
            results.total_battery_charge_kwh += out.battery_charge_wh / 1000.0;
            results.total_battery_discharge_kwh += out.battery_discharge_wh / 1000.0;
            results.total_grid_import_kwh += out.grid_import_wh / 1000.0;
            results.total_grid_export_kwh += out.grid_export_wh / 1000.0;
            // Net grid import power for this step (kW); the peak is the demand-charge basis.
            let net_demand_kw = out.grid_import_wh / dt_hours / 1000.0;
            if net_demand_kw > results.peak_net_demand_kw {
                results.peak_net_demand_kw = net_demand_kw;
            }
            results.final_soc_fraction = out.soc_fraction;
        }
        results
    }

    /// Current battery SoC fraction (0.0–1.0).
    pub fn soc_fraction(&self) -> f64 {
        self.battery.soc_fraction()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn step_input(thermal_load_w: f64, irradiance: f64, ambient_c: f64) -> GridTimestepInput {
        GridTimestepInput {
            thermal_load_w,
            plane_irradiance_wm2: irradiance,
            ambient_temp_c: ambient_c,
            dt_seconds: 3600.0,
        }
    }

    #[test]
    fn test_signed_thermal_semantics_cooling_uses_magnitude() {
        // Sign convention: positive = heating, negative = cooling (the engine's
        // step_physics convention). Both modes consume electricity — the adapter
        // couples the magnitude, never producing a negative electrical load.
        let mut heat_adapter = GridAdapter::new(&test_config());
        let heat = heat_adapter.step(&step_input(3000.0, 0.0, 20.0));
        let mut cool_adapter = GridAdapter::new(&test_config());
        let cool = cool_adapter.step(&step_input(-3000.0, 0.0, 20.0));
        // Hand-computed: |±3000| W thermal / COP 3.0 = 1000 W electrical.
        assert!((heat.electrical_load_w - 1000.0).abs() < 1e-9);
        assert!((cool.electrical_load_w - 1000.0).abs() < 1e-9);
        assert!(cool.electrical_load_w > 0.0);
        // The additive block sums magnitudes: 3 kWh heating + 3 kWh cooling.
        let mut batch_adapter = GridAdapter::new(&test_config());
        let batch = batch_adapter.post_process(&[
            step_input(3000.0, 0.0, 20.0),
            step_input(-3000.0, 0.0, 20.0),
        ]);
        assert!((batch.total_thermal_kwh - 6.0).abs() < 1e-9);
        assert!((batch.total_electrical_load_kwh - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_step_thermal_to_electrical_via_cop() {
        // Hand-computed: 3000 W thermal / COP 3.0 = 1000 W electrical.
        let mut adapter = GridAdapter::new(&test_config());
        let out = adapter.step(&step_input(3000.0, 0.0, 20.0));
        assert!((out.electrical_load_w - 1000.0).abs() < 1e-9);
        // No sun: no PV, no export.
        assert_eq!(out.pv_ac_w, 0.0);
        assert_eq!(out.grid_export_wh, 0.0);
    }

    #[test]
    fn test_step_pv_ac_power_hand_computed() {
        // Hand-computed from fluxion-grid's NOCT + derating model:
        // cell_temp = 25 + (45-20)*1000/800 = 56.25 °C
        // derating = 1 - 0.004*31.25 = 0.875
        // dc = 1000*10*0.2*0.875*0.94 = 1645 W; ac = 1645*0.96 = 1579.2 W.
        let mut adapter = GridAdapter::new(&test_config());
        let out = adapter.step(&step_input(0.0, 1000.0, 25.0));
        assert!((out.pv_ac_w - 1579.2).abs() < 1e-6);
    }

    #[test]
    fn test_step_battery_self_consumption_charges_from_excess_pv() {
        // Hand-computed: load 1000 W, PV 1579.2 W → 579.2 Wh excess.
        // charge = min(579.2*0.95, available) = 550.24 Wh; export = 579.2-550.24 = 28.96 Wh.
        // SoC: 5000 + 550.24 = 5550.24 Wh → 0.555024.
        let mut adapter = GridAdapter::new(&test_config());
        let out = adapter.step(&step_input(3000.0, 1000.0, 25.0));
        assert!((out.battery_charge_wh - 550.24).abs() < 1e-6);
        assert_eq!(out.battery_discharge_wh, 0.0);
        assert!((out.grid_export_wh - 28.96).abs() < 1e-6);
        assert_eq!(out.grid_import_wh, 0.0);
        assert!((out.soc_fraction - 0.555024).abs() < 1e-9);
    }

    #[test]
    fn test_step_battery_discharges_to_meet_deficit() {
        // Night: load 1000 W, no PV → 1000 Wh deficit.
        // discharge_from_battery = min(1000/0.95, available) = 1052.6315789 Wh;
        // SoC: 5000 - 1052.6315789 = 3947.3684211 → 0.39473684211.
        let mut adapter = GridAdapter::new(&test_config());
        let out = adapter.step(&step_input(3000.0, 0.0, 20.0));
        assert!((out.battery_discharge_wh - 1052.6315789473683).abs() < 1e-6);
        assert_eq!(out.battery_charge_wh, 0.0);
        // deficit (1000) < discharge_from_battery (1052.63) → no import.
        assert_eq!(out.grid_import_wh, 0.0);
        assert!((out.soc_fraction - 0.39473684210526316).abs() < 1e-9);
    }

    #[test]
    fn test_post_process_matches_step_loop_exactly() {
        // The batch path must be exactly the step() fold — this is the property
        // phase 2 relies on when moving the step() call inside the timestep loop.
        let series = vec![
            step_input(3000.0, 1000.0, 25.0),
            step_input(3000.0, 0.0, 20.0),
            step_input(1500.0, 500.0, 22.0),
        ];
        let mut a = GridAdapter::new(&test_config());
        let batch = a.post_process(&series);

        let mut b = GridAdapter::new(&test_config());
        let mut import_kwh = 0.0;
        let mut peak_kw = 0.0f64;
        for input in &series {
            let out = b.step(input);
            import_kwh += out.grid_import_wh / 1000.0;
            peak_kw = peak_kw.max(out.grid_import_wh / 1000.0);
        }
        assert!((batch.total_grid_import_kwh - import_kwh).abs() < 1e-12);
        assert!((batch.peak_net_demand_kw - peak_kw).abs() < 1e-12);
        assert_eq!(batch.num_timesteps, 3);
        assert!((batch.total_thermal_kwh - 7.5).abs() < 1e-9);
        assert!((batch.final_soc_fraction - b.soc_fraction()).abs() < 1e-12);
    }

    #[test]
    fn test_post_process_is_deterministic() {
        let series = vec![step_input(3000.0, 800.0, 24.0); 24];
        let mut a = GridAdapter::new(&test_config());
        let mut b = GridAdapter::new(&test_config());
        let ra = a.post_process(&series);
        let rb = b.post_process(&series);
        for (x, y) in [
            (ra.total_thermal_kwh, rb.total_thermal_kwh),
            (ra.total_electrical_load_kwh, rb.total_electrical_load_kwh),
            (ra.total_pv_kwh, rb.total_pv_kwh),
            (ra.total_grid_import_kwh, rb.total_grid_import_kwh),
            (ra.total_grid_export_kwh, rb.total_grid_export_kwh),
            (ra.peak_net_demand_kw, rb.peak_net_demand_kw),
            (ra.final_soc_fraction, rb.final_soc_fraction),
        ] {
            assert_eq!(x, y, "post_process must be bit-deterministic");
        }
    }

    #[test]
    fn test_post_process_empty_series() {
        let mut adapter = GridAdapter::new(&test_config());
        let results = adapter.post_process(&[]);
        assert_eq!(results.num_timesteps, 0);
        assert_eq!(results.total_grid_import_kwh, 0.0);
        assert_eq!(results.peak_net_demand_kw, 0.0);
        assert!((results.final_soc_fraction - 0.5).abs() < 1e-12);
    }
}
