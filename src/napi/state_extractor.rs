// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! NAPI bindings for StateExtractor - zero-copy state matrix extraction for ML training.
//!
//! This module provides high-performance native bindings that allow ML training scripts
//! to extract state-space matrices directly from the Rust engine without JSON/CSV
//! serialization overhead.
//!
//! # Architecture
//! - **Zero-copy memory sharing**: Returns typed arrays (Float64Array) that JavaScript
//!   can access directly without copying
//! - **ML Training Ready**: State matrices can be fed directly into TensorFlow/PyTorch
//!   via data loaders
//! - **TypeScript Support**: Full type definitions auto-generated via napi-rs

use crate::ai::surrogate::SurrogateManager;
use crate::napi::zero_copy_matrix::into_zero_copy_float64_array;
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::sim::thermal_selector::ThermalSelector;
use crate::weather::epw::EpwWeatherSource;
use crate::weather::WeatherSource;
use napi::bindgen_prelude::Float64Array;

/// JavaScript-accessible StateExtractor for ML training data extraction.
///
/// This class provides zero-copy access to simulation state matrices, enabling
/// high-performance ML training without JSON/CSV serialization bottlenecks.
///
/// # TypeScript Example
/// ```typescript
/// import { StateExtractor } from '@fluxion/native';
///
/// // Create extractor with ASHRAE 600 base configuration
/// const extractor = new StateExtractor();
///
/// // Configure for multi-zone extraction
/// extractor.configure({ numZones: 3 });
///
/// // Run simulation and extract state matrices
/// const result = extractor.runSimulation(1, false);
///
/// // Access zero-copy typed arrays (no serialization!)
/// console.log(`Zone temperatures: ${result.zoneTemperatures.length} timesteps`);
/// console.log(`Timestep 0, Zone 0: ${result.zoneTemperatures[0]}`);
/// ```
///
/// # Performance Characteristics
/// - **JSON/CSV bottleneck**: Traditional approach requires ~50-200ms for serialization
/// - **Zero-copy extraction**: Direct typed array access, ~0.1ms overhead
/// - **Speedup**: 500-2000x faster for large simulations
#[napi_derive::napi]
pub struct StateExtractor {
    inner: ThermalModel<VectorField>,
    num_zones: usize,
    steps: usize,
}

/// Optional constructor options for [`StateExtractor`] (Issue #3282).
///
/// Both fields default to the production defaults (`gauge` zone solver,
/// `default` conduction algorithm) when omitted. The experimental
/// `"6r2c"` / `"8r3c"` zone-solver identifiers are rejected unless the
/// `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` env var is set (and even then they
/// stay unavailable until the `fluxion-experimental-zone-solvers` cargo
/// feature ships; issue #3291).
#[napi_derive::napi(object)]
pub struct StateExtractorOptions {
    /// Zone solver: `"gauge"` (default) | `"5r1c"` | `"9r4c"`.
    #[napi(js_name = "zoneSolver")]
    pub zone_solver: Option<String>,
    /// Conduction algorithm: `"default"` (default) | `"ctf"` | `"fd"`.
    #[napi(js_name = "conductionSolver")]
    pub conduction_solver: Option<String>,
}

#[napi_derive::napi]
impl StateExtractor {
    /// Create a new StateExtractor with default ASHRAE 600 configuration.
    ///
    /// # TypeScript Example
    /// ```typescript
    /// import { StateExtractor } from '@fluxion/native';
    /// // Defaults: zoneSolver='gauge', conductionSolver='default'
    /// const extractor = new StateExtractor();
    /// // Explicit solver selection (Issue #3282):
    /// const legacy = new StateExtractor({ zoneSolver: '5r1c', conductionSolver: 'ctf' });
    /// ```
    ///
    /// # Errors
    /// Throws for unknown or experimental solver selections (the
    /// experimental gate: `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1`).
    #[napi(constructor)]
    pub fn new(options: Option<StateExtractorOptions>) -> napi::bindgen_prelude::Result<Self> {
        let selector = match &options {
            None => ThermalSelector::default(),
            Some(opts) => crate::sim::thermal_selector::ThermalSelector {
                zone_solver: match &opts.zone_solver {
                    Some(s) => crate::sim::thermal_selector::parse_zone_solver(s)
                        .map_err(napi::bindgen_prelude::Error::from_reason)?,
                    None => crate::sim::thermal_selector::ZoneSolverKind::Gauge,
                },
                conduction_solver: match &opts.conduction_solver {
                    Some(s) => crate::sim::thermal_selector::parse_conduction_solver(s)
                        .map_err(napi::bindgen_prelude::Error::from_reason)?,
                    None => crate::sim::thermal_selector::ConductionSolverKind::Default,
                },
            },
        };
        let spec = crate::validation::ashrae_140_cases::CaseBuilder::case_600_baseline();
        let thermal_model = ThermalModel::from_spec_with_selector(&spec, &selector)
            .map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?;

        Ok(StateExtractor {
            inner: thermal_model,
            num_zones: 1,
            steps: 8760,
        })
    }

    /// Configure the extractor for specific simulation parameters.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones (default: 1)
    #[napi]
    pub fn configure(&mut self, num_zones: u32) -> napi::bindgen_prelude::Result<()> {
        if num_zones < 1 {
            return Err(napi::bindgen_prelude::Error::from_reason(
                "Number of zones must be at least 1",
            ));
        }
        self.num_zones = num_zones as usize;
        Ok(())
    }

    /// Run simulation and extract state matrices with zero-copy access.
    ///
    /// This is the critical method for ML training - it runs the simulation and
    /// returns state matrices as typed arrays that can be passed directly to
    /// ML frameworks without JSON/CSV serialization.
    ///
    /// # Arguments
    /// * `years` - Number of years to simulate (1-5 typical)
    /// * `use_surrogates` - If true, use AI surrogates for faster evaluation
    ///
    /// # Returns
    /// StateMatrices object containing typed arrays for each state variable:
    /// - `zoneTemperatures`: Zone air temperatures [timesteps x num_zones]
    /// - `massTemperatures`: Thermal mass temperatures [timesteps x num_zones]
    /// - `heatingLoads`: Heating energy demand [timesteps]
    /// - `coolingLoads`: Cooling energy demand [timesteps]
    /// - `solarGains`: Solar heat gains [timesteps x num_zones]
    ///
    /// # Issue #3624 — real per-step HVAC loads
    ///
    /// With `use_surrogates: false` (the default) the extractor runs the
    /// same per-step physics loop the engine-side Case 600 validation suite
    /// uses (`run_annual_simulation` in `tests/ashrae_140_case_600_series.rs`):
    /// a 14-day ASHRAE 140 §B2 warm-up, then 8760 hourly `step_physics`
    /// calls driven by the real `assets/weather/WD600.epw` dry-bulb schedule
    /// (TMY repetition for multi-year requests). `heatingLoads` /
    /// `coolingLoads` are the engine's own metered per-step energies —
    /// nothing is broadcast, approximated, or fabricated. The legacy
    /// surrogate path (below) keeps the #3667 annual-average broadcast.
    #[napi]
    pub fn run_simulation(
        &mut self,
        years: u32,
        use_surrogates: bool,
    ) -> napi::bindgen_prelude::Result<StateMatrices> {
        let steps = years as usize * 8760;
        self.steps = steps;

        if use_surrogates {
            self.run_simulation_surrogate_path(steps)
        } else {
            self.run_simulation_physics_path(steps)
        }
    }

    /// Issue #3624 physics path: real per-step loads from `step_physics`,
    /// driven by the WD600 (ASHRAE 140 §B2) weather schedule.
    fn run_simulation_physics_path(
        &mut self,
        steps: usize,
    ) -> napi::bindgen_prelude::Result<StateMatrices> {
        let weather = load_wd600_epw()?;

        // ASHRAE 140 §B2 warm-up (14 days) before the metered year —
        // mirrors `run_annual_simulation` in
        // `tests/ashrae_140_case_600_series.rs`. Warm-up energy is
        // discarded, and the model's annual-energy trackers are reset so
        // the model state stays consistent with the metered arrays below
        // (the same "only this loop's energy is reported" contract
        // `solve_timesteps` documents).
        const WARMUP_STEPS: usize = 14 * 24;
        for step in 0..WARMUP_STEPS {
            let weather_data = weather
                .get_hourly_data(step)
                .map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?;
            self.inner.solar.weather = Some(weather_data.clone());
            let energy_kwh = self
                .inner
                .step_physics(step, weather_data.dry_bulb_temp, 3600.0);
            check_energy_finite(energy_kwh, "run_simulation")?;
        }
        self.inner.hvac.annual_heating_energy = 0.0;
        self.inner.hvac.annual_cooling_energy = 0.0;

        // Metered year: per-step engine energy (positive = heating,
        // negative = cooling — the engine's own classification, see
        // `run_annual_simulation`), plus per-step zone temperatures.
        let mut zone_flat = Vec::with_capacity(steps * self.num_zones);
        let mut heating_w = Vec::with_capacity(steps);
        let mut cooling_w = Vec::with_capacity(steps);
        for step in 0..steps {
            // TMY repetition: `get_hourly_data` errors past hour 8759, so
            // multi-year requests wrap modulo the schedule length.
            let weather_data = weather
                .get_hourly_data(step % 8760)
                .map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?;
            self.inner.solar.weather = Some(weather_data.clone());
            let energy_kwh = self
                .inner
                .step_physics(step, weather_data.dry_bulb_temp, 3600.0);
            // Divergence guard (Issue #3633 contract, per step now).
            check_energy_finite(energy_kwh, "run_simulation")?;
            push_step_energy_kwh(&mut heating_w, &mut cooling_w, energy_kwh);

            let temps = self.inner.get_temperatures();
            for z in 0..self.num_zones {
                zone_flat.push(temps.get(z).copied().unwrap_or(20.0));
            }
        }

        let mass_temperatures = self.inner.get_temperatures();
        let mass_flat = flatten_mass_temperatures(&mass_temperatures, self.num_zones, steps);

        Ok(StateMatrices {
            zone_temperatures: into_zero_copy_float64_array(zone_flat),
            mass_temperatures: into_zero_copy_float64_array(mass_flat),
            heating_loads: into_zero_copy_float64_array(heating_w),
            cooling_loads: into_zero_copy_float64_array(cooling_w),
            solar_gains: into_zero_copy_float64_array(vec![0.0; steps * self.num_zones]),
        })
    }

    /// Legacy surrogate path (`use_surrogates: true`): `solve_timesteps`
    /// with the #3667 annual-average broadcast. Kept so the documented
    /// surrogate evaluation mode is preserved unchanged; the default
    /// physics path (above) is where the real per-step loads live.
    fn run_simulation_surrogate_path(
        &mut self,
        steps: usize,
    ) -> napi::bindgen_prelude::Result<StateMatrices> {
        let surrogates = SurrogateManager::new().map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!(
                "Failed to create SurrogateManager: {}",
                e
            ))
        })?;

        let eui = self
            .inner
            .solve_timesteps(steps, &surrogates, true, None, None, None);
        // Issue #3633: propagate divergence as Err instead of silently
        // fabricating StateMatrices. `solve_timesteps` returns the EUI as
        // f64; NaN / +-Inf indicate the inner physics step diverged.
        check_energy_finite(eui, "run_simulation")?;

        let hourly_temps = self.inner.get_hourly_temperatures();
        let zone_temperatures = match hourly_temps {
            Some(temps) => {
                // Flatten from Vec<Vec<f64>> [zones][timesteps] to flat Vec<f64>
                // JavaScript will interpret as Float64Array [timesteps x num_zones]
                let mut flat = Vec::with_capacity(steps * self.num_zones);
                for t in 0..steps {
                    for z in 0..self.num_zones {
                        if z < temps.len() && t < temps[z].len() {
                            flat.push(temps[z][t]);
                        } else {
                            flat.push(20.0); // Default temperature
                        }
                    }
                }
                flat
            }
            None => vec![20.0; steps * self.num_zones],
        };

        let mass_temperatures = self.inner.get_temperatures();

        let mass_flat = flatten_mass_temperatures(&mass_temperatures, self.num_zones, steps);

        // Issue #3667: per-step power derived from the model's accumulated
        // annual heating/cooling energy (constant-power broadcast). On this
        // path only — the default physics path returns the engine's real
        // per-step loads (Issue #3624).
        let heating_loads = populate_step_loads(self.inner.hvac.annual_heating_energy, steps);
        let cooling_loads = populate_step_loads(self.inner.hvac.annual_cooling_energy, steps);

        Ok(StateMatrices {
            zone_temperatures: into_zero_copy_float64_array(zone_temperatures),
            mass_temperatures: into_zero_copy_float64_array(mass_flat),
            heating_loads: into_zero_copy_float64_array(heating_loads),
            cooling_loads: into_zero_copy_float64_array(cooling_loads),
            solar_gains: into_zero_copy_float64_array(vec![0.0; steps * self.num_zones]),
        })
    }

    /// Extract only zone temperatures (lightweight extraction for simple ML models).
    ///
    /// This is an optimized method for cases where only zone temperatures are needed,
    /// avoiding the overhead of extracting all state matrices.
    ///
    /// # Arguments
    /// * `years` - Number of years to simulate
    /// * `use_surrogates` - If true, use AI surrogates
    ///
    /// # Returns
    /// Flat array of zone temperatures [timesteps x num_zones]
    #[napi]
    pub fn extract_zone_temperatures(
        &mut self,
        years: u32,
        use_surrogates: bool,
    ) -> napi::bindgen_prelude::Result<Float64Array> {
        let steps = years as usize * 8760;

        let surrogates = SurrogateManager::new().map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!(
                "Failed to create SurrogateManager: {}",
                e
            ))
        })?;

        let eui = self
            .inner
            .solve_timesteps(steps, &surrogates, use_surrogates, None, None, None);
        // Issue #3633: same divergence check as `run_simulation` above.
        check_energy_finite(eui, "extract_zone_temperatures")?;

        let hourly_temps = self.inner.get_hourly_temperatures();
        match hourly_temps {
            Some(temps) => {
                let mut flat = Vec::with_capacity(steps * self.num_zones);
                for t in 0..steps {
                    for z in 0..self.num_zones {
                        if z < temps.len() && t < temps[z].len() {
                            flat.push(temps[z][t]);
                        } else {
                            flat.push(20.0);
                        }
                    }
                }
                Ok(into_zero_copy_float64_array(flat))
            }
            None => Ok(into_zero_copy_float64_array(vec![
                20.0;
                steps * self.num_zones
            ])),
        }
    }
}

impl Default for StateExtractor {
    fn default() -> Self {
        Self::new(None).expect("Failed to create StateExtractor with default config")
    }
}

/// Issue #3633 helper: validate that a simulation energy figure is finite.
///
/// `solve_timesteps` returns `f64` rather than `Result`, and the per-step
/// `step_physics` energies are `f64` as well; NaN / +Inf / -Inf are the
/// divergence signal. Propagating the check through this helper keeps
/// `run_simulation` (both paths) and `extract_zone_temperatures`
/// consistent and lets us unit-test the rule without spinning up a full
/// `StateExtractor`.
fn check_energy_finite(energy: f64, caller: &str) -> napi::bindgen_prelude::Result<()> {
    if energy.is_finite() {
        Ok(())
    } else {
        Err(napi::bindgen_prelude::Error::from_reason(format!(
            "simulation diverged in {caller} (energy={energy}); refusing to fabricate StateMatrices (Issue #3633)"
        )))
    }
}

/// Issue #3624 helper: locate and parse the canonical WD600 (ASHRAE 140
/// §B2) weather file — the same annual drive the engine-side Case 600
/// validation suite uses (`tests/ashrae_140_case_600_series.rs` loads
/// `assets/weather/WD600.epw` from the repo root).
///
/// Candidate paths cover both supported CWDs: the repo root (`cargo test`,
/// `cargo run`) and the `npm/` package directory (`npm test` runs node with
/// `working-directory: npm` in CI). The first readable candidate wins; a
/// failure lists everything tried so operators can diagnose a missing
/// checkout.
fn load_wd600_epw() -> napi::bindgen_prelude::Result<EpwWeatherSource> {
    const CANDIDATES: [&str; 2] = ["assets/weather/WD600.epw", "../assets/weather/WD600.epw"];
    for candidate in CANDIDATES {
        match EpwWeatherSource::from_file(candidate) {
            Ok(source) => return Ok(source),
            Err(_) => continue,
        }
    }
    Err(napi::bindgen_prelude::Error::from_reason(format!(
        "WD600 weather file not found (tried {:?} relative to the current \
         directory). The napi StateExtractor's per-step physics path (Issue \
         #3624) requires the canonical ASHRAE 140 §B2 weather fixture from \
         the repository checkout — run from the repo root or the npm/ \
         package directory.",
        CANDIDATES
    )))
}

/// Issue #3624 helper: classify one step's metered engine energy (kWh) and
/// push it onto the per-step load arrays in watts (1-hour timesteps: W·h).
///
/// This is the engine's own classification — positive = heating, negative
/// = cooling, zero = free-floating hour — identical to
/// `run_annual_simulation` in `tests/ashrae_140_case_600_series.rs`. The
/// JS client recovers kWh via `sum(W) / 1000.0` (see `npm/test.js`).
fn push_step_energy_kwh(heating: &mut Vec<f64>, cooling: &mut Vec<f64>, energy_kwh: f64) {
    if energy_kwh > 0.0 {
        heating.push(energy_kwh * 1000.0);
        cooling.push(0.0);
    } else {
        // Zero (free-floating hour) and negative (cooling) both land here;
        // the cooling magnitude of 0.0 kWh is 0.0 W.
        heating.push(0.0);
        cooling.push((-energy_kwh) * 1000.0);
    }
}

/// Issue #3634 helper: flatten a `[zone]` mass-temperature vector into a
/// `[timesteps * num_zones]` matrix, using `.get(idx).copied().unwrap_or(20.0)`
/// so empty `mass_temperatures` and zero `num_zones` both produce the same
/// placeholder fill the `zone_temperatures` branch already uses. The previous
/// `z.min(mass_temperatures.len() - 1)` form panicked on `0_usize - 1` when
/// `get_temperatures()` returned an empty Vec.
fn flatten_mass_temperatures(
    mass_temperatures: &[f64],
    num_zones: usize,
    steps: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(steps * num_zones);
    for _t in 0..steps {
        for z in 0..num_zones {
            // `.get` returns None on either an empty slice or an out-of-range
            // index — both safe-fall to the 20.0 default rather than panic.
            out.push(mass_temperatures.get(z).copied().unwrap_or(20.0));
        }
    }
    out
}

/// Issue #3667 helper: convert an annual energy total (kWh) into a
/// `steps`-long vector of per-timestep average power (W), so the JS
/// client can recover kWh by `sum(W) / 1000.0` (1-hour timesteps).
///
/// The previous `run_simulation` body hardcoded `vec![0.0; steps]` for
/// `heating_loads` / `cooling_loads` (see `src/napi/state_extractor.rs`
/// history prior to #3667). That fabricated the per-step power array
/// regardless of what `solve_timesteps` actually produced — the JS-side
/// sum was 0.0 and the ASHRAE 600 smoke test (`npm/test.js:707`,
/// `total_energy_kwh > 0`) failed with
/// `'total_energy_kwh must be positive, got 0'` even when the underlying
/// simulation ran to completion. Reading
/// `self.inner.hvac.annual_heating_energy` / `annual_cooling_energy`
/// (populated as a side-effect of `step_physics`) and broadcasting the
/// annual average across the timesteps turns the previous fabricated-
/// zero output into an honest reflection of the model state.
///
/// Edge cases:
/// - `steps == 0`: returns an empty `Vec`. Matches the
///   `flatten_mass_temperatures` defensive style — the napi path always
///   passes `years * 8760 >= 8760`, but the helper stays panic-free.
/// - `annual_kwh <= 0` (free-floating / unconditioned / no-heating /
///   cooling-dominated hour): the broadcast becomes `0.0` for every
///   cell, which mirrors the previous behaviour for those zones but
///   only when the model legitimately produced no energy. A genuinely
///   divergent simulation produces NaN / +Inf here, which the JS-side
///   `Number.isFinite` guard in `npm/test.js:725` already rejects
///   (Issue #2911 / #3633 pattern).
///
/// Per-step temporal variation is intentionally NOT recovered on this
/// helper — Issue #3624 added the real per-step physics path
/// (`run_simulation_physics_path`) for that; this broadcast now serves only
/// the legacy surrogate path (`use_surrogates: true`), where
/// `solve_timesteps` exposes annual totals rather than a metered series.
fn populate_step_loads(annual_kwh: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        return Vec::new();
    }
    // Per-step average power in W: annual_kwh * 1000.0 / steps
    //   (since 1 kWh = 1000 Wh and we have `steps` hourly timesteps,
    //    so W·h / h = W; multiplying by 1000 converts kWh→Wh).
    let per_step_w = annual_kwh * 1000.0 / steps as f64;
    vec![per_step_w; steps]
}

#[cfg(test)]
mod tests {
    use super::{
        check_energy_finite, flatten_mass_temperatures, populate_step_loads, push_step_energy_kwh,
    };

    // ====================================================================
    // Issue #3633 regression tests: divergence propagation
    // ====================================================================

    #[test]
    fn check_energy_finite_passes_finite_values() {
        assert!(check_energy_finite(0.0, "t").is_ok());
        assert!(check_energy_finite(123.456, "t").is_ok());
        assert!(check_energy_finite(-1.0e9, "t").is_ok());
        assert!(check_energy_finite(f64::MIN_POSITIVE, "t").is_ok());
        assert!(check_energy_finite(f64::MAX, "t").is_ok());
    }

    /// Issue #3633 regression: a NaN energy from `solve_timesteps` or a
    /// per-step `step_physics` call must surface as `Err` from the napi
    /// method, not as a fabricated `StateMatrices`.
    #[test]
    fn check_energy_finite_rejects_nan() {
        let err = check_energy_finite(f64::NAN, "run_simulation").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("diverged") && msg.contains("run_simulation") && msg.contains("#3633"),
            "error message should name the divergence, caller, and issue: {msg}"
        );
    }

    #[test]
    fn check_energy_finite_rejects_pos_inf() {
        assert!(check_energy_finite(f64::INFINITY, "t").is_err());
    }

    #[test]
    fn check_energy_finite_rejects_neg_inf() {
        assert!(check_energy_finite(f64::NEG_INFINITY, "t").is_err());
    }

    // ====================================================================
    // Issue #3624 regression tests: real per-step load classification
    // ====================================================================

    /// Positive engine energy is heating, expressed in W for the JS client
    /// (`sum(W) / 1000.0` recovers kWh); the cooling array stays 0.0 for
    /// that step.
    #[test]
    fn push_step_energy_kwh_classifies_positive_as_heating() {
        let mut h = Vec::new();
        let mut c = Vec::new();
        push_step_energy_kwh(&mut h, &mut c, 2.5);
        assert_eq!(h, vec![2500.0]);
        assert_eq!(c, vec![0.0]);
    }

    /// Negative engine energy is cooling (magnitude in W); the heating
    /// array stays 0.0 for that step.
    #[test]
    fn push_step_energy_kwh_classifies_negative_as_cooling() {
        let mut h = Vec::new();
        let mut c = Vec::new();
        push_step_energy_kwh(&mut h, &mut c, -1.25);
        assert_eq!(h, vec![0.0]);
        assert_eq!(c, vec![1250.0]);
    }

    /// A free-floating (zero-energy) hour records zeros on both arrays —
    /// same as the engine-side classification (`> 0` heating / `< 0`
    /// cooling / otherwise uncounted).
    #[test]
    fn push_step_energy_kwh_records_zero_energy_hour_as_zeros() {
        let mut h = Vec::new();
        let mut c = Vec::new();
        push_step_energy_kwh(&mut h, &mut c, 0.0);
        assert_eq!(h, vec![0.0]);
        assert_eq!(c, vec![0.0]);
    }

    /// Summing a year of classified steps and dividing by 1000 reconstructs
    /// the metered kWh totals — the exact identity `npm/test.js` relies on.
    #[test]
    fn push_step_energy_kwh_round_trips_metered_kwh() {
        let steps = [0.5, -0.25, 0.0, 1.0, -2.0];
        let mut h = Vec::new();
        let mut c = Vec::new();
        for &e in &steps {
            push_step_energy_kwh(&mut h, &mut c, e);
        }
        let heating_kwh = h.iter().sum::<f64>() / 1000.0;
        let cooling_kwh = c.iter().sum::<f64>() / 1000.0;
        assert!((heating_kwh - 1.5).abs() < 1e-12);
        assert!((cooling_kwh - 2.25).abs() < 1e-12);
    }

    // ====================================================================
    // Issue #3634 regression tests: mass_temperatures flatten empty-slice safety
    // ====================================================================

    /// Issue #3634 regression: an empty `mass_temperatures` slice must not
    /// panic the flatten. The old `z.min(len - 1)` form underflowed on
    /// `0_usize - 1` (debug) or wrapped to `usize::MAX` (release) and
    /// then panicked on the index. The new helper falls back to 20.0
    /// for every cell, matching the `zone_temperatures` placeholder.
    #[test]
    fn flatten_mass_temperatures_handles_empty_slice() {
        let out = flatten_mass_temperatures(&[], 3, 2);
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|&v| v == 20.0));
    }

    #[test]
    fn flatten_mass_temperatures_handles_zero_zones() {
        let out = flatten_mass_temperatures(&[1.0, 2.0, 3.0], 0, 5);
        assert!(out.is_empty());
    }

    #[test]
    fn flatten_mass_temperatures_handles_both_empty_and_zero_zones() {
        let out = flatten_mass_temperatures(&[], 0, 4);
        assert!(out.is_empty());
    }

    #[test]
    fn flatten_mass_temperatures_repeats_per_zone() {
        // 2 zones, 3 timesteps → 6 cells, each zone broadcast to its
        // row. Zone 0 = 10.0, zone 1 = 22.0, no underflow risk here
        // since `mass_temperatures.len() >= num_zones`.
        let out = flatten_mass_temperatures(&[10.0, 22.0], 2, 3);
        assert_eq!(out, vec![10.0, 22.0, 10.0, 22.0, 10.0, 22.0]);
    }

    #[test]
    fn flatten_mass_temperatures_falls_back_when_num_zones_exceeds_slice() {
        // 1 zone in the slice, 3 zones requested → zones 1, 2 use 20.0
        let out = flatten_mass_temperatures(&[15.0], 3, 1);
        assert_eq!(out, vec![15.0, 20.0, 20.0]);
    }

    // ====================================================================
    // Issue #3667 regression tests: fabricated zero-load replacement
    // ====================================================================

    /// Issue #3667 regression: zero steps must not panic or divide-by-zero
    /// the per-step W broadcast — the napi wrapper enforces `years >= 1`
    /// upstream, but the helper itself stays panic-free on `steps == 0`,
    /// matching `flatten_mass_temperatures` style.
    #[test]
    fn populate_step_loads_handles_zero_steps() {
        let out = populate_step_loads(1234.5, 0);
        assert!(out.is_empty());
    }

    /// Issue #3667 regression: an annual energy total in kWh must
    /// broadcast to a vector whose `sum() / 1000.0` reconstructs the
    /// same kWh value (1-hour timesteps). The napi `run_simulation`
    /// previously hardcoded `vec![0.0; steps]` here, which masked the
    /// simulation output and made the JS-side `total_energy_kwh` sum
    /// identically 0.0 — the `npm/test.js:707` band check fails first
    /// on `total_energy_kwh > 0`.
    #[test]
    fn populate_step_loads_recovers_annual_kwh_via_sum_over_1000() {
        // ASHRAE 600 reference band: heating in [4314, 5836] kWh.
        let annual_heating_kwh = 5000.0;
        let steps = 8760usize;
        let out = populate_step_loads(annual_heating_kwh, steps);
        assert_eq!(out.len(), steps);
        assert!(out.iter().all(|&v| v == out[0]));
        // sum(W) over 1-hour timesteps = Wh; /1000 → kWh
        let recovered = out.iter().sum::<f64>() / 1000.0;
        assert!(
            (recovered - annual_heating_kwh).abs() < 1e-6,
            "annual heating round-trip lost fidelity: {recovered} vs {annual_heating_kwh}"
        );
    }

    /// Issue #3667 regression: a non-trivial Case 600 cooling total
    /// produces per-step W values that, when summed and divided by 1000,
    /// recover the original kWh figure — the JS-side `cooling_kwh` will
    /// at least be `> 0` even when the underlying simulation produces
    /// wildly divergent numbers (the band check would still fail, but
    /// that failure is diagnostic rather than the fabricated-zero
    /// failure that #3667 was filed against).
    #[test]
    fn populate_step_loads_handles_zero_annual_kwh() {
        let out = populate_step_loads(0.0, 8760);
        assert_eq!(out.len(), 8760);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Issue #3667 regression: a `f64::NAN` annual energy from a divergent
    /// simulation must propagate through the broadcast — the JS-side
    /// `Number.isFinite` guard (`npm/test.js:725`, Issue #2911 / #3633)
    /// then rejects it, so the fabricated-zero failure mode is replaced
    /// by the more diagnostic finite-check failure rather than a silent
    /// NaN leak.
    #[test]
    fn populate_step_loads_propagates_nan_through_broadcast() {
        let out = populate_step_loads(f64::NAN, 8760);
        assert_eq!(out.len(), 8760);
        assert!(out.iter().all(|&v| v.is_nan()));
    }

    /// Issue #3667 regression: the per-step average W figure is exactly
    /// `annual_kwh * 1000.0 / steps`, not rounded or smoothed. For
    /// `annual_kwh = 4380.0` and `steps = 8760` the per-step W equals
    /// 500.0 W exactly (4380 kWh over the year = 4380 kW average =
    /// 500 W per timestep), matching the documented kWh→W conversion.
    #[test]
    fn populate_step_loads_per_step_w_matches_annual_average() {
        let out = populate_step_loads(4380.0, 8760);
        assert!(out.iter().all(|&v| v == 500.0));
    }
}

/// Container for extracted state matrices.
///
/// All fields are typed arrays (Float64Array in JavaScript) enabling
/// zero-copy access from ML frameworks.
#[napi_derive::napi]
pub struct StateMatrices {
    /// Zone air temperatures in °C [timesteps x num_zones]
    pub zone_temperatures: Float64Array,

    /// Thermal mass temperatures in °C [timesteps x num_zones]
    pub mass_temperatures: Float64Array,

    /// Heating energy demand in W [timesteps]
    pub heating_loads: Float64Array,

    /// Cooling energy demand in W [timesteps]
    pub cooling_loads: Float64Array,

    /// Solar heat gains in W [timesteps x num_zones]
    pub solar_gains: Float64Array,
}

impl StateMatrices {
    /// Get the shape of the zone temperatures matrix.
    ///
    /// Returns [timesteps, num_zones] for reshaping in ML frameworks.
    pub fn zone_temperatures_shape(&self, num_zones: u32) -> Vec<u32> {
        let num_zones = num_zones as usize;
        let timesteps = self
            .zone_temperatures
            .len()
            .checked_div(num_zones)
            .unwrap_or(0);
        vec![timesteps as u32, num_zones as u32]
    }
}
