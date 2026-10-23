//! Thermostat sensor node — grey-box model for the TRV sensor temperature.
//!
//! # Motivation (Issue #4555)
//!
//! Van Hove et al., *Energy and Buildings* 2026 (DOI
//! 10.1016/j.enbuild.2026.118337) show that the temperature reported by a
//! thermostatic radiator valve (TRV) sensor, `Ts`, is **not** the indoor air
//! temperature `Ti` plus a fixed offset: the offset is *dynamic*, driven by
//! radiator activity and by weather. Their grey-box model therefore keeps the
//! two temperatures as separate states.
//!
//! Field numbers from that paper (the calibration context for this module):
//! using a dynamic `Ts` cut the sensor RMSE from **3.35 °C** (constant-offset
//! baseline) to **1.08 °C**.
//!
//! # Model
//!
//! The sensor node is a first-order (lumped-capacitance) state `Ts` with its
//! own thermal mass. Its *target* temperature is the zone air temperature
//! `Ti`, shifted by two couplings:
//!
//! ```text
//! target = Ti + β · Q_rad + γ · (T_out − Ti)
//!
//! τ_s · dTs/dt = target − Ts
//! ```
//!
//! * `τ_s` — sensor time constant [s] (its own thermal mass divided by the
//!   conductance to the room air). This is the first-order lag that makes the
//!   reported temperature trail the air temperature.
//! * `β` — radiator coupling [K/kW]: the extra sensor temperature the local
//!   radiator raises the sensor above room air while it is delivering heat.
//! * `γ` — ambient/weather coupling [–]: how much of the indoor–outdoor
//!   temperature difference bleeds into the sensor reading (e.g. a sensor
//!   mounted on an exterior wall). `γ = 0` disables the term.
//! * `Q_rad` — radiator activity [kW]. This is the *simplest existing signal*
//!   in the zone/HVAC model: the per-zone heating power delivered by the
//!   HVAC system (positive = heating). The step function also accepts it as
//!   an explicit parameter, so the model can be driven from any caller.
//!
//! Integration is exact for a piecewise-constant target over the step:
//!
//! ```text
//! Ts ← Ts + (target − Ts) · (1 − exp(−dt/τ_s))
//! ```
//!
//! # Integration contract
//!
//! The sensor node is **additive and observational**: it *reads* zone state
//! (air temperature, radiator activity, outdoor temperature) and writes only
//! its own state. It never feeds back into the zone energy balance, so
//! enabling it cannot change any simulation output. The wiring point is the
//! zone physics step (see `crate::sim::thermal_model_physics`), right after
//! the zone air temperature and the HVAC/radiator output are known.
//!
//! See [`SensorNode`], [`SensorNodeParams`] and [`SensorNodeBank`].

use serde::{Deserialize, Serialize};

/// Parameters of a single thermostat sensor node.
///
/// Defaults correspond to a typical wall-mounted TRV head: a ~30 min thermal
/// lag (`τ_s = 1800 s`), a modest local radiator coupling (`β = 0.25 K/kW`)
/// and no ambient coupling (`γ = 0`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorNodeParams {
    /// Sensor time constant [s] — the sensor's own thermal mass / coupling
    /// to the room air. Must be > 0.
    pub time_constant_s: f64,
    /// Radiator coupling [K/kW] — sensor temperature rise per kW of local
    /// radiator output. Must be >= 0.
    pub radiator_coupling_k_per_kw: f64,
    /// Ambient/weather coupling [–] — fraction of the indoor–outdoor
    /// temperature difference that bleeds into the sensor reading.
    /// Must be >= 0.
    pub ambient_coupling: f64,
    /// Initial sensor temperature [°C] used when the node is created.
    pub initial_temperature_c: f64,
}

impl Default for SensorNodeParams {
    fn default() -> Self {
        Self {
            time_constant_s: 1800.0,
            radiator_coupling_k_per_kw: 0.25,
            ambient_coupling: 0.0,
            initial_temperature_c: 20.0,
        }
    }
}

impl SensorNodeParams {
    /// Create parameters with an explicit time constant, keeping the other
    /// couplings at their defaults.
    pub fn with_time_constant_s(time_constant_s: f64) -> Self {
        Self {
            time_constant_s,
            ..Self::default()
        }
    }

    /// Validate the parameter set; returns `Err` with a human-readable
    /// reason when a value is not physically admissible.
    pub fn validate(&self) -> Result<(), String> {
        if !self.time_constant_s.is_finite() || self.time_constant_s <= 0.0 {
            return Err(format!(
                "sensor time constant must be finite and > 0 s, got {}",
                self.time_constant_s
            ));
        }
        if !self.radiator_coupling_k_per_kw.is_finite() || self.radiator_coupling_k_per_kw < 0.0 {
            return Err(format!(
                "radiator coupling must be finite and >= 0 K/kW, got {}",
                self.radiator_coupling_k_per_kw
            ));
        }
        if !self.ambient_coupling.is_finite() || self.ambient_coupling < 0.0 {
            return Err(format!(
                "ambient coupling must be finite and >= 0, got {}",
                self.ambient_coupling
            ));
        }
        if !self.initial_temperature_c.is_finite() {
            return Err(format!(
                "initial sensor temperature must be finite, got {}",
                self.initial_temperature_c
            ));
        }
        Ok(())
    }
}

/// A single thermostat sensor node with its own thermal mass.
///
/// The node owns one state: the sensor temperature `Ts` [°C].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorNode {
    params: SensorNodeParams,
    /// Sensor temperature [°C].
    ts: f64,
}

impl SensorNode {
    /// Create a sensor node from `params` (the node starts at
    /// `params.initial_temperature_c`).
    ///
    /// # Panics
    /// Panics if the parameters are invalid — use [`SensorNode::try_new`] for
    /// a fallible constructor.
    pub fn new(params: SensorNodeParams) -> Self {
        Self::try_new(params).expect("invalid SensorNodeParams")
    }

    /// Fallible constructor: returns `Err` (with reason) on invalid params.
    pub fn try_new(params: SensorNodeParams) -> Result<Self, String> {
        params.validate()?;
        Ok(Self {
            ts: params.initial_temperature_c,
            params,
        })
    }

    /// The node parameters.
    pub fn params(&self) -> &SensorNodeParams {
        &self.params
    }

    /// Current sensor temperature [°C].
    pub fn temperature(&self) -> f64 {
        self.ts
    }

    /// Current sensor temperature minus the zone air temperature of the last
    /// step [K] — the "offset" the paper shows to be dynamic.
    pub fn offset_from_air(&self, t_air_c: f64) -> f64 {
        self.ts - t_air_c
    }

    /// Reset the node to its initial temperature.
    pub fn reset(&mut self) {
        self.ts = self.params.initial_temperature_c;
    }

    /// Advance the node by `dt_seconds` and return the new sensor
    /// temperature [°C].
    ///
    /// * `t_air_c` — zone air temperature `Ti` [°C].
    /// * `radiator_power_kw` — radiator activity `Q_rad` [kW] (positive =
    ///   heating). Use the per-zone HVAC heating output for this.
    /// * `t_outdoor_c` — outdoor air temperature [°C] (only used when the
    ///   ambient coupling `γ` is non-zero).
    ///
    /// A negative `dt_seconds` or a non-finite input leaves the state
    /// untouched (the step is rejected) and returns the current temperature.
    pub fn step(
        &mut self,
        dt_seconds: f64,
        t_air_c: f64,
        radiator_power_kw: f64,
        t_outdoor_c: f64,
    ) -> f64 {
        if !dt_seconds.is_finite() || dt_seconds < 0.0 {
            return self.ts;
        }
        if !t_air_c.is_finite() || !radiator_power_kw.is_finite() || !t_outdoor_c.is_finite() {
            return self.ts;
        }

        let p = self.params;
        // Radiator coupling only counts heating output (a cooling TRV does
        // not warm its sensor).
        let q_rad_kw = radiator_power_kw.max(0.0);
        let target = t_air_c
            + p.radiator_coupling_k_per_kw * q_rad_kw
            + p.ambient_coupling * (t_outdoor_c - t_air_c);

        // Exact solution of τ_s dTs/dt = target − Ts over a step with a
        // piecewise-constant target.
        let decay = (-dt_seconds / p.time_constant_s).exp();
        self.ts = target + (self.ts - target) * decay;
        self.ts
    }
}

/// A per-zone bank of sensor nodes (one node per zone).
///
/// The bank is the object the simulation layer owns: it is stepped once per
/// zone-timestep with the zone air temperature and the zone radiator
/// activity, and it never writes back into the zone energy balance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorNodeBank {
    nodes: Vec<SensorNode>,
}

impl SensorNodeBank {
    /// Create a bank of `num_zones` identical nodes from `params`.
    ///
    /// # Panics
    /// Panics if `params` are invalid.
    pub fn new(num_zones: usize, params: SensorNodeParams) -> Self {
        Self::try_new(num_zones, params).expect("invalid SensorNodeParams")
    }

    /// Fallible constructor.
    pub fn try_new(num_zones: usize, params: SensorNodeParams) -> Result<Self, String> {
        params.validate()?;
        Ok(Self {
            nodes: vec![SensorNode::new(params); num_zones],
        })
    }

    /// Number of zones in the bank.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the bank holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Sensor temperature of zone `zone` [°C] (`None` if out of range).
    pub fn temperature(&self, zone: usize) -> Option<f64> {
        self.nodes.get(zone).map(|n| n.temperature())
    }

    /// All sensor temperatures [°C], one per zone.
    pub fn temperatures(&self) -> Vec<f64> {
        self.nodes.iter().map(|n| n.temperature()).collect()
    }

    /// Mutable access to a single node (used for per-zone overrides).
    pub fn node_mut(&mut self, zone: usize) -> Option<&mut SensorNode> {
        self.nodes.get_mut(zone)
    }

    /// Step every zone of the bank.
    ///
    /// * `dt_seconds` — timestep length [s].
    /// * `t_air_c` — per-zone air temperature [°C].
    /// * `radiator_power_kw` — per-zone radiator activity [kW] (positive =
    ///   heating).
    /// * `t_outdoor_c` — outdoor air temperature [°C].
    ///
    /// Slices shorter than the number of zones are treated as missing zones:
    /// those nodes keep their state.
    pub fn step(
        &mut self,
        dt_seconds: f64,
        t_air_c: &[f64],
        radiator_power_kw: &[f64],
        t_outdoor_c: f64,
    ) {
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let Some(&t_air) = t_air_c.get(i) else {
                continue;
            };
            let q_rad = radiator_power_kw.get(i).copied().unwrap_or(0.0);
            node.step(dt_seconds, t_air, q_rad, t_outdoor_c);
        }
    }

    /// Step every zone of the bank, taking the radiator activity directly in
    /// watts (the unit used by the zone/HVAC model) and converting to kW
    /// internally.
    ///
    /// This is the allocation-free variant used by the physics step: it
    /// avoids the intermediate `Vec` the kW-based [`Self::step`] needs.
    pub fn step_watts(
        &mut self,
        dt_seconds: f64,
        t_air_c: &[f64],
        radiator_output_w: &[f64],
        t_outdoor_c: f64,
    ) {
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let Some(&t_air) = t_air_c.get(i) else {
                continue;
            };
            let q_kw = radiator_output_w.get(i).copied().unwrap_or(0.0).max(0.0) / 1000.0;
            node.step(dt_seconds, t_air, q_kw, t_outdoor_c);
        }
    }

    /// Reset every node to its initial temperature.
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            node.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation-layer wiring
// ---------------------------------------------------------------------------

use crate::sim::thermal_model_core::ThermalModel;
use crate::sim::thermal_model_data::ContinuousTensor;

impl<T: ContinuousTensor<f64>> ThermalModel<T> {
    /// Attach a bank of thermostat sensor nodes to the model.
    ///
    /// The nodes are *observational*: they observe the zone air temperature
    /// and the radiator activity on every physics step and write only their
    /// own state. Attaching (or removing) them cannot change the zone energy
    /// balance or any simulation output.
    ///
    /// Returns an error string if the parameters are invalid.
    pub fn attach_sensor_nodes(&mut self, params: SensorNodeParams) -> Result<(), String> {
        let bank = SensorNodeBank::try_new(self.0.hvac.num_zones, params)?;
        self.0.mass.sensor_nodes = Some(bank);
        Ok(())
    }

    /// Detach any attached sensor nodes (restores the default, inert state).
    pub fn detach_sensor_nodes(&mut self) {
        self.0.mass.sensor_nodes = None;
    }

    /// Current sensor temperature per zone [°C], or `None` when no sensor
    /// nodes are attached.
    pub fn sensor_node_temperatures(&self) -> Option<Vec<f64>> {
        self.0
            .mass
            .sensor_nodes
            .as_ref()
            .map(|bank| bank.temperatures())
    }

    /// Advance the attached sensor nodes by one timestep.
    ///
    /// * `dt_seconds` — timestep length [s].
    /// * `t_air_c` — per-zone air temperature [°C].
    /// * `radiator_output_w` — per-zone radiator/HVAC output [W] (positive =
    ///   heating). This is the simplest radiator-activity signal already
    ///   present in the zone/HVAC model.
    /// * `t_outdoor_c` — outdoor air temperature [°C].
    ///
    /// This method is a no-op when no sensor nodes are attached.
    pub fn step_sensor_nodes(
        &mut self,
        dt_seconds: f64,
        t_air_c: &[f64],
        radiator_output_w: &[f64],
        t_outdoor_c: f64,
    ) {
        let Some(bank) = self.0.mass.sensor_nodes.as_mut() else {
            return;
        };
        bank.step_watts(dt_seconds, t_air_c, radiator_output_w, t_outdoor_c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: RMSE between two series.
    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        assert!(n > 0, "empty series");
        let s: f64 = a[..n]
            .iter()
            .zip(b[..n].iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        (s / n as f64).sqrt()
    }

    /// Test 1 — step response with the radiator off: the sensor temperature
    /// converges to the zone air temperature.
    #[test]
    fn sensor_converges_to_air_temperature_with_radiator_off() {
        let mut node = SensorNode::new(SensorNodeParams {
            time_constant_s: 1800.0,
            radiator_coupling_k_per_kw: 0.25,
            ambient_coupling: 0.0,
            initial_temperature_c: 15.0,
        });
        let t_air = 21.0;
        let mut last = node.temperature();
        // tau = 1800 s, dt = 60 s -> decay factor exp(-1/30) per step; 1000
        // steps (~17 h) is exp(-33) ~ 4e-15 of the initial error.
        for _ in 0..1000 {
            last = node.step(60.0, t_air, 0.0, 5.0);
        }
        assert!(
            (last - t_air).abs() < 1e-6,
            "sensor must converge to air temperature, got {last} vs {t_air}"
        );
        // Monotone approach, never overshooting.
        let mut prev = 15.0;
        for _ in 0..50 {
            let t = node.step(60.0, t_air, 0.0, 5.0);
            assert!(
                t >= prev && t <= t_air,
                "non-monotone/overshoot: {prev} -> {t}"
            );
            prev = t;
        }
    }

    /// Test 2 — radiator step-on: the offset above air temperature grows
    /// dynamically and then settles. It is *not* a constant offset: a
    /// constant-offset model would jump straight to the final value.
    #[test]
    fn radiator_step_on_produces_dynamic_offset_not_constant() {
        let params = SensorNodeParams {
            time_constant_s: 1800.0,
            radiator_coupling_k_per_kw: 0.25,
            ambient_coupling: 0.0,
            initial_temperature_c: 20.0,
        };
        let mut node = SensorNode::new(params);
        let t_air = 20.0;

        // Settle with the radiator off.
        for _ in 0..100 {
            node.step(60.0, t_air, 0.0, 5.0);
        }
        assert!((node.temperature() - t_air).abs() < 1e-6);

        // Step the radiator on: 4 kW -> +1.0 K steady-state offset.
        let first = node.step(60.0, t_air, 4.0, 5.0);
        let offset_first = first - t_air;
        assert!(
            offset_first > 0.0 && offset_first < 0.05,
            "offset must start from zero and build up, got {offset_first}"
        );

        // Keep stepping: the offset grows monotonically toward the
        // asymptote beta * Q_rad = 0.25 K/kW * 4 kW = 1.0 K.
        let mut prev = offset_first;
        for _ in 0..1000 {
            let t = node.step(60.0, t_air, 4.0, 5.0);
            let offset = t - t_air;
            // Monotone non-decreasing (it saturates at machine precision once
            // the asymptote is reached, so an exact `>` would trip on the
            // last few steps).
            assert!(
                offset >= prev - 1e-12,
                "offset must not shrink: {prev} -> {offset}"
            );
            assert!(
                offset > prev || offset > 0.999,
                "offset must keep growing before it saturates: {prev} -> {offset}"
            );
            assert!(offset <= 1.0 + 1e-9, "offset must not exceed the asymptote");
            prev = offset;
        }
        let offset_final = node.temperature() - t_air;
        assert!(
            (offset_final - 1.0).abs() < 1e-3,
            "offset must settle at beta*Q = 1.0 K, got {offset_final}"
        );

        // A constant-offset model (offset = final value applied instantly)
        // is wrong during the transient.
        let constant_model = t_air + offset_final;
        assert!(
            (constant_model - first).abs() > 0.5,
            "constant-offset model must not reproduce the transient"
        );
    }

    /// Test 3 — validation-style check on a synthetic series.
    ///
    /// Truth is generated with the sensor model using *known* parameters
    /// (a different parameter set from the estimator's, so this is a real
    /// estimation problem). The dynamic sensor model is compared against a
    /// constant-offset baseline (`Ts = Ti + c`, with `c` chosen as the
    /// RMSE-optimal constant, i.e. the best case for the baseline).
    ///
    /// Field context (Van Hove et al. 2026): dynamic `Ts` cut the sensor
    /// RMSE from 3.35 °C to 1.08 °C. This synthetic reproduction shows the
    /// same ordering: the dynamic model beats the constant-offset baseline.
    #[test]
    fn dynamic_model_beats_constant_offset_baseline_on_synthetic_series() {
        // Truth parameters.
        let truth_params = SensorNodeParams {
            time_constant_s: 2400.0,
            radiator_coupling_k_per_kw: 0.30,
            ambient_coupling: 0.05,
            initial_temperature_c: 18.0,
        };
        // Estimator parameters (slightly mismatched, as in a real fit).
        let est_params = SensorNodeParams {
            time_constant_s: 2000.0,
            radiator_coupling_k_per_kw: 0.26,
            ambient_coupling: 0.04,
            initial_temperature_c: 18.0,
        };

        let dt = 300.0;
        let steps = 480; // 40 h of 5-minute samples.
        let mut truth_node = SensorNode::new(truth_params);
        let mut est_node = SensorNode::new(est_params);

        let mut air = Vec::with_capacity(steps);
        let mut truth = Vec::with_capacity(steps);
        let mut est = Vec::with_capacity(steps);

        for k in 0..steps {
            let hours = k as f64 * dt / 3600.0;
            // Diurnal zone air temperature.
            let t_air = 20.0 + 2.0 * (2.0 * std::f64::consts::PI * hours / 24.0).sin();
            // Radiator cycles: on during the cold part of the day.
            let q_rad = if (hours % 24.0) < 8.0 { 5.0 } else { 0.0 };
            let t_out = 6.0 + 4.0 * (2.0 * std::f64::consts::PI * (hours + 8.0) / 24.0).sin();

            air.push(t_air);
            truth.push(truth_node.step(dt, t_air, q_rad, t_out));
            est.push(est_node.step(dt, t_air, q_rad, t_out));
        }

        // Constant-offset baseline with the RMSE-optimal constant offset.
        let mean_offset: f64 = truth
            .iter()
            .zip(air.iter())
            .map(|(ts, ti)| ts - ti)
            .sum::<f64>()
            / steps as f64;
        let baseline: Vec<f64> = air.iter().map(|ti| ti + mean_offset).collect();

        let rmse_dynamic = rmse(&est, &truth);
        let rmse_constant = rmse(&baseline, &truth);

        assert!(
            rmse_dynamic < rmse_constant,
            "dynamic model ({rmse_dynamic:.4} °C) must beat the constant-offset \
             baseline ({rmse_constant:.4} °C)"
        );
        // Sanity: the dynamic error is small in absolute terms, the baseline
        // error is of the order of the radiator-induced offset swing.
        assert!(
            rmse_dynamic < 0.1,
            "dynamic RMSE should be small, got {rmse_dynamic:.4} °C"
        );
        assert!(
            rmse_constant > 0.3,
            "constant-offset baseline RMSE should be clearly worse, got {rmse_constant:.4} °C"
        );
    }

    /// Ambient coupling pulls the sensor reading toward the outdoor
    /// temperature difference (weather coupling term).
    #[test]
    fn ambient_coupling_shifts_sensor_toward_outdoor() {
        let mut node = SensorNode::new(SensorNodeParams {
            time_constant_s: 3600.0,
            radiator_coupling_k_per_kw: 0.0,
            ambient_coupling: 0.2,
            initial_temperature_c: 20.0,
        });
        for _ in 0..2000 {
            node.step(60.0, 20.0, 0.0, 0.0);
        }
        // target = 20 + 0.2 * (0 - 20) = 16 °C
        assert!(
            (node.temperature() - 16.0).abs() < 1e-6,
            "got {}",
            node.temperature()
        );
    }

    /// Invalid parameters are rejected.
    #[test]
    fn invalid_params_rejected() {
        assert!(SensorNode::try_new(SensorNodeParams {
            time_constant_s: 0.0,
            ..SensorNodeParams::default()
        })
        .is_err());
        assert!(SensorNode::try_new(SensorNodeParams {
            radiator_coupling_k_per_kw: -1.0,
            ..SensorNodeParams::default()
        })
        .is_err());
    }

    /// The bank steps every zone independently.
    #[test]
    fn bank_steps_each_zone() {
        let mut bank = SensorNodeBank::new(2, SensorNodeParams::default());
        bank.step(600.0, &[20.0, 25.0], &[0.0, 2.0], 10.0);
        let temps = bank.temperatures();
        assert_eq!(temps.len(), 2);
        assert!(
            temps[1] > temps[0],
            "heated zone must read warmer: {temps:?}"
        );
    }
}
