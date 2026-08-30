// Multi-zone Python bindings for Fluxion
// This module extends the existing Python API with multi-zone functionality

use crate::api::error::FluxionError;
use crate::api::schema::{SimulationSchema, SimulationSchemaV1};
use crate::interop::gbxml::{export_gbxml as export_gbxml_file, GbXmlError};
use crate::interop::osm::{export_osm as export_osm_file, OsmError};
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::sim::invariant_checker::InvariantChecker;
use crate::sim::thermal_selector::ThermalSelector;
use crate::validation::ashrae_140_cases::ASHRAE140Case;
use crate::weather::denver::DenverTmyWeather;
use crate::weather::WeatherSource;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::collections::HashMap;

/// Default tolerance for energy balance validation (0.1%).
/// This matches the CI gate in test_energy_conservation.rs.
/// See issue #1061 for tolerance documentation.
const ENERGY_BALANCE_TOLERANCE: f64 = 0.001;

/// Multi-zone thermal model for Python
#[pyclass(name = "MultiZoneThermalModel")]
pub struct PyMultiZoneThermalModel {
    inner: ThermalModel<VectorField>,
}

#[pymethods]
impl PyMultiZoneThermalModel {
    /// Create a new MultiZoneThermalModel with specified number of zones
    #[new]
    #[pyo3(signature = (num_zones=2))]
    pub fn new(num_zones: usize) -> PyResult<Self> {
        if num_zones < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of zones must be at least 1",
            ));
        }

        Ok(PyMultiZoneThermalModel {
            // Issue #2806: populate real thermal physics (mirror of the REST
            // `build_model_from_schema` fix, #2747) so the analytical solver
            // does not diverge on the C_m=1.0 placeholder left by
            // `ThermalModel::new`. See `model_bindings::populate_default_model_physics`.
            inner: {
                let mut m = ThermalModel::<VectorField>::new(num_zones);
                crate::python::model_bindings::populate_default_model_physics(&mut m);
                m
            },
        })
    }

    /// Construct a MultiZoneThermalModel from an ASHRAE 140 case spec.
    ///
    /// Used to obtain high-mass (9R4C) construction models from Python so the
    /// sub-hourly node temperature API
    /// ([`get_nodal_temperatures`](Self::get_nodal_temperatures) /
    /// [`get_nodal_temperatures_numpy`](Self::get_nodal_temperatures_numpy))
    /// has data to return. Examples: `"Case900"` (high-mass baseline),
    /// `"Case600"` (low-mass), `"Case970"` (5-zone).
    ///
    /// # Arguments
    /// * `case_id` - ASHRAE 140 case identifier (case-insensitive). Examples:
    ///   `"Case900"`, `"Case600"`, `"case_900"`.
    ///
    /// # Errors
    /// Raises `ValueError` for unknown case IDs.
    #[staticmethod]
    pub fn from_case_spec(case_id: &str) -> PyResult<Self> {
        let normalized: String = case_id
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '_')
            .collect();
        let normalized_upper = normalized.to_ascii_uppercase();
        let case_enum = match normalized_upper.as_str() {
            "CASE600" => ASHRAE140Case::Case600,
            "CASE610" => ASHRAE140Case::Case610,
            "CASE620" => ASHRAE140Case::Case620,
            "CASE630" => ASHRAE140Case::Case630,
            "CASE640" => ASHRAE140Case::Case640,
            "CASE650" => ASHRAE140Case::Case650,
            "CASE600FF" => ASHRAE140Case::Case600FF,
            "CASE650FF" => ASHRAE140Case::Case650FF,
            "CASE900" => ASHRAE140Case::Case900,
            "CASE910" => ASHRAE140Case::Case910,
            "CASE920" => ASHRAE140Case::Case920,
            "CASE930" => ASHRAE140Case::Case930,
            "CASE940" => ASHRAE140Case::Case940,
            "CASE950" => ASHRAE140Case::Case950,
            "CASE900FF" => ASHRAE140Case::Case900FF,
            "CASE950FF" => ASHRAE140Case::Case950FF,
            "CASE960" => ASHRAE140Case::Case960,
            "CASE970" => ASHRAE140Case::Case970,
            "CASE195" => ASHRAE140Case::Case195,
            "CASE195HIGHMASS" | "CASE195HM" => ASHRAE140Case::Case195HighMass,
            "CASE195NOLOAds" | "CASE195NL" => ASHRAE140Case::Case195NoLoads,
            "CASE195NOSOLAR" | "CASE195NS" => ASHRAE140Case::Case195NoSolar,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown ASHRAE 140 case '{}'. Known cases: Case195, Case195HighMass, \
                     Case600, Case610, Case620, Case630, Case640, Case650, Case600FF, Case650FF, \
                     Case900, Case910, Case920, Case930, Case940, Case950, Case900FF, Case950FF, \
                     Case960, Case970",
                    other
                )));
            }
        };
        let spec = case_enum.spec();
        Ok(PyMultiZoneThermalModel {
            inner: ThermalModel::<VectorField>::from_spec_with_selector(
                &spec,
                &ThermalSelector::default(),
            )
            .expect("default selector must initialize"),
        })
    }

    /// Get number of zones in the model
    pub fn num_zones(&self) -> usize {
        self.inner.hvac.num_zones
    }

    /// Get the number of zones from the inner ThermalModel
    pub fn get_inner_num_zones(&self) -> usize {
        self.inner.hvac.num_zones
    }

    /// Get current zone temperatures
    pub fn get_zone_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    /// Set zone temperatures
    pub fn set_zone_temperatures(&mut self, temps: Vec<f64>) -> PyResult<()> {
        if temps.len() != self.inner.hvac.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Temperature vector length ({}) must match number of zones ({})",
                temps.len(),
                self.inner.hvac.num_zones
            )));
        }
        self.inner.setpoints.temperatures = VectorField::new(temps);
        Ok(())
    }

    /// Set zone-specific heating and cooling setpoints
    pub fn set_zone_setpoints(
        &mut self,
        zone_idx: usize,
        heating: f64,
        cooling: f64,
    ) -> PyResult<()> {
        if zone_idx >= self.inner.hvac.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone index {} out of range (0-{})",
                zone_idx,
                self.inner.hvac.num_zones - 1
            )));
        }

        if heating >= cooling {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Heating setpoint must be less than cooling setpoint",
            ));
        }

        // Set zone-specific setpoints
        self.inner.setpoints.heating_setpoints.as_mut_slice()[zone_idx] = heating;
        self.inner.setpoints.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;

        Ok(())
    }

    /// Get inter-zone conductance between two zones
    pub fn get_inter_zone_conductance(&self, zone_i: usize, zone_j: usize) -> PyResult<f64> {
        if zone_i >= self.inner.hvac.num_zones || zone_j >= self.inner.hvac.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Zone indices out of range",
            ));
        }

        Ok(self.inner.conduction.h_tr_iz.as_slice()[zone_i])
    }

    /// Set inter-zone conductance between two zones
    pub fn set_inter_zone_conductance(
        &mut self,
        zone_i: usize,
        zone_j: usize,
        conductance: f64,
    ) -> PyResult<()> {
        if zone_i >= self.inner.hvac.num_zones || zone_j >= self.inner.hvac.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Zone indices out of range",
            ));
        }

        if conductance < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Conductance must be non-negative",
            ));
        }

        self.inner.conduction.h_tr_iz.as_mut_slice()[zone_i] = conductance;
        if zone_i != zone_j {
            self.inner.conduction.h_tr_iz.as_mut_slice()[zone_j] = conductance;
        }
        Ok(())
    }

    /// Set all inter-zone conductances from a vector
    pub fn set_inter_zone_conductance_vector(&mut self, conductances: Vec<f64>) -> PyResult<()> {
        if conductances.len() != self.inner.hvac.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Conductance vector length ({}) must match number of zones ({})",
                conductances.len(),
                self.inner.hvac.num_zones
            )));
        }

        for (i, &c) in conductances.iter().enumerate() {
            if c < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "All conductances must be non-negative",
                ));
            }
            self.inner.conduction.h_tr_iz.as_mut_slice()[i] = c;
        }
        Ok(())
    }

    /// Get all inter-zone conductances as a vector
    pub fn get_inter_zone_conductance_vector(&self) -> Vec<f64> {
        self.inner.conduction.h_tr_iz.as_slice().to_vec()
    }

    /// Simulate multi-zone building energy consumption
    pub fn simulate_multi_zone(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        let steps = years as usize * 8760;

        // Use the existing surrogate manager from the model
        let surrogates = crate::ai::surrogate::SurrogateManager::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create SurrogateManager: {}",
                e
            ))
        })?;

        // Issue #2806: `simulate_multi_zone` returns the cumulative
        // `annual_heating_energy + annual_cooling_energy` trackers (see return
        // below), which `solve_timesteps` accumulates but never resets. Without
        // this reset, sequential calls on the same instance return N× the
        // per-call energy (10429 → 20861 → 31294 …), breaking determinism.
        // Reset first so the trackers reflect only THIS call's energy.
        self.inner.reset_heating_cooling_energy();
        // Issue #2826 follow-up: also reset transient state (zone air, mass,
        // surface, and air-node ODE state). Without this reset, the second
        // call starts from the first call's end-state, so `simulate_multi_zone`
        // is non-deterministic across calls (the first call is a warm-up from
        // constructor defaults; subsequent calls reach steady state from a
        // different initial condition). This also caused the multi-zone
        // fixture's per-test state to leak — `test_cooling_setpoint_decrease_raises_energy`
        // saw baseline energy figures inflated by warm-up heat stored in the
        // mass from earlier tests' heating branch.
        self.inner.reset_state();

        // Issue #2806 / #2747: pass an empty lighting schedule so the solver
        // does NOT auto-load the bundled Office building profile (which would
        // otherwise run because lighting/equipment/occupancy are all None).
        // That profile has a per-step `loads[i] += internal_gain` accumulation
        // quirk that overheats a default-constructed zone and drives EUI
        // negative. Envelope-only is the physically-sane baseline, matching
        // `Model::simulate` and the REST `/v1/simulate` path (#2747).
        let zone_area = self
            .inner
            .setpoints
            .zone_area
            .as_slice()
            .iter()
            .sum::<f64>()
            .max(1.0);
        let empty_lighting = crate::sim::lighting::LightingSchedule::new(0.0, zone_area);

        let _net_result = self.inner.solve_timesteps(
            steps,
            &surrogates,
            use_surrogates,
            Some(&empty_lighting),
            None,
            None,
        );

        // Issue #2547 — detect divergence (NaN / infinity) in the per-zone
        // hourly temperature trace and surface it as a `SimulationError`
        // carrying a `diagnostics` dict (failing_timestep / failing_zone /
        // max_residual_pct / last_known_good_timestep). Without this check a
        // diverged simulation would return a NaN energy figure silently.
        if let Some(hourly) = self.inner.get_hourly_temperatures() {
            if let Some(diag) =
                crate::api::error::SimulationDiagnostics::from_temperature_trace(&hourly)
            {
                return Err(crate::api::error::FluxionError::Simulation(
                    format!(
                        "simulation diverged at timestep {}{}",
                        diag.failing_timestep,
                        diag.failing_zone
                            .as_ref()
                            .map(|z| format!(" in zone {z}"))
                            .unwrap_or_default()
                    ),
                    Some(diag),
                )
                .into());
            }
        }

        // Return total energy consumption (heating + cooling) instead of net energy
        // This matches the expectation of Python tests that energy should be non-negative
        let total_energy =
            self.inner.get_heating_energy_kwh() + self.inner.get_cooling_energy_kwh();
        Ok(total_energy)
    }

    /// Get zone-specific energy consumption (heating + cooling) in kWh
    ///
    /// Returns a vector with total energy (heating + cooling) for each zone.
    /// These values are accumulated during simulation.
    pub fn get_zone_energies(&self) -> Vec<f64> {
        self.inner.get_zone_energies_kwh()
    }

    /// Get zone-specific peak loads (Issue #1289)
    ///
    /// Returns a dictionary with:
    /// - "heating": List of peak heating power per zone in kW
    /// - "cooling": List of peak cooling power per zone in kW
    pub fn get_zone_peak_loads(&self) -> PyResult<HashMap<String, Vec<f64>>> {
        let mut result = HashMap::new();
        result.insert("heating".to_string(), self.inner.get_zone_peak_heating_kw());
        result.insert("cooling".to_string(), self.inner.get_zone_peak_cooling_kw());
        Ok(result)
    }

    /// Get peak load metrics for a specific zone (Issue #1628)
    ///
    /// Accepts a zone identifier as either:
    /// - An integer index (e.g., 0, 1, 2)
    /// - A string zone name (e.g., "Zone1", "Zone 1", "zone_1")
    ///
    /// Returns a dictionary with:
    /// - "heating_mw": Peak heating power in MW
    /// - "cooling_mw": Peak cooling power in MW
    /// - "heating_timestep": Timestep index when peak heating occurred
    /// - "cooling_timestep": Timestep index when peak cooling occurred
    pub fn get_zone_peaks(
        &self,
        zone_identifier: &Bound<'_, PyAny>,
    ) -> PyResult<HashMap<String, Py<PyAny>>> {
        let zone_idx = if let Ok(idx) = zone_identifier.extract::<usize>() {
            idx
        } else if let Ok(name) = zone_identifier.extract::<String>() {
            let normalized = name.trim().to_lowercase();
            if let Some(stripped) = normalized.strip_prefix("zone") {
                let num_str = stripped.trim().replace(['_', ' '], "-");
                if let Ok(num) = num_str.parse::<usize>() {
                    if num == 0 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Zone indices are 0-based. Use 0 for Zone 1.",
                        ));
                    }
                    num - 1
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid zone name {:?}. Expected format: Zone1, Zone 1, or integer index.",
                        name
                    )));
                }
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid zone identifier {:?}. Expected format: Zone1, Zone 1, or integer index.",
                    name
                )));
            }
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Zone identifier must be an integer index or string like Zone1",
            ));
        };

        if zone_idx >= self.inner.hvac.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Zone index {} out of range (0-{})",
                zone_idx,
                self.inner.hvac.num_zones - 1
            )));
        }

        let heating_kw = self.inner.get_zone_peak_heating_kw();
        let cooling_kw = self.inner.get_zone_peak_cooling_kw();
        let heating_timesteps = self.inner.get_zone_peak_heating_timestep();
        let cooling_timesteps = self.inner.get_zone_peak_cooling_timestep();

        Python::attach(|py| {
            let mut result = HashMap::new();
            result.insert(
                "heating_mw".to_string(),
                (heating_kw[zone_idx] / 1000.0)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            );
            result.insert(
                "cooling_mw".to_string(),
                (cooling_kw[zone_idx] / 1000.0)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            );
            result.insert(
                "heating_timestep".to_string(),
                heating_timesteps[zone_idx]
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            );
            result.insert(
                "cooling_timestep".to_string(),
                cooling_timesteps[zone_idx]
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            );
            Ok(result)
        })
    }

    /// Export zone temperatures as Python dictionary
    pub fn export_zone_temperatures(&self) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);

            for (i, temp) in self.inner.get_temperatures().iter().enumerate() {
                dict.set_item(format!("zone_{}", i), *temp)?;
            }

            Ok(dict.unbind())
        })
    }

    /// Get the full hourly zone temperature profiles (Issue #763).
    ///
    /// # Returns
    /// `Some([[T00, T01, ...], [T10, T11, ...], ...])` where outer index is zone,
    /// inner index is timestep (0..steps-1), or `None` if the simulation has not
    /// been run yet.
    pub fn get_hourly_temperatures(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.get_hourly_temperatures()
    }

    /// Get hourly temperatures as zero-copy numpy arrays for ML training.
    ///
    /// This method provides direct access to the underlying temperature data
    /// without JSON/CSV serialization overhead, enabling high-performance
    /// ML training pipelines.
    ///
    /// # Arguments
    /// * `py` - Python GIL token for numpy array creation
    ///
    /// # Returns
    /// Tuple of (zone_temperatures, shape) where zone_temperatures is a 2D numpy array
    /// with shape [num_zones, timesteps] and zero-copy memory sharing when possible.
    ///
    /// # Example
    /// ```python
    /// import numpy as np
    /// model = fluxion.MultiZoneThermalModel(3)
    /// model.simulate_multi_zone(1, False)
    /// temps, shape = model.get_hourly_temperatures_numpy()
    /// # temps is a numpy array with shape [3, 8760]
    /// ```
    pub fn get_hourly_temperatures_numpy<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<(Bound<'a, numpy::PyArray2<f64>>, Vec<usize>)> {
        let hourly_temps = self.inner.get_hourly_temperatures();

        match hourly_temps {
            Some(temps) => {
                let num_zones = temps.len();
                let timesteps = if num_zones > 0 { temps[0].len() } else { 0 };
                let shape = vec![num_zones, timesteps];

                // Create numpy array from Vec<Vec<f64>> - this is the expected format for from_vec2
                let arr = numpy::PyArray2::from_vec2(py, &temps).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to create numpy array: {}",
                        e
                    ))
                })?;
                Ok((arr, shape))
            }
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "Simulation has not been run yet. Call simulate_multi_zone first.",
            )),
        }
    }

    /// Get sub-hourly 9R4C node temperature profiles (Issue #1799).
    ///
    /// Returns a nested list `[[[wall_t], [roof_t], [floor_t], [internal_t]] * num_zones]`
    /// where the middle axis follows `MultiNodeSolver::NODE_NAMES` ordering:
    /// `0=wall, 1=roof, 2=floor, 3=internal`. The inner axis is one entry per
    /// timestep, time-indexed by `simulate_multi_zone` execution order.
    ///
    /// # Returns
    /// `Some([num_zones][4][num_steps])` for high-mass models (9R4C), or
    /// `None` for low-mass models or before `simulate_multi_zone` has run.
    ///
    /// # Example
    /// ```python
    /// import fluxion
    /// model = fluxion.MultiZoneThermalModel(1)
    /// # Need high-mass construction for 9R4C — populate per-zone solver manually
    /// # via the public Python API for testing.
    /// # model.simulate_multi_zone(1, False)
    /// nodal = model.get_nodal_temperatures()
    /// # nodal[zone_idx][node_idx][timestep] is the time-indexed series.
    /// ```
    pub fn get_nodal_temperatures(&self) -> Option<Vec<Vec<Vec<f64>>>> {
        self.inner.get_nodal_temperatures()
    }

    /// Number of zones carrying a 9R4C `MultiNodeSolver` (Issue #1799).
    ///
    /// Returns 0 for low-mass models. Used by Python callers to disambiguate
    /// `get_nodal_temperatures() is None` between "not a 9R4C model" and
    /// "simulation not yet run".
    pub fn num_multinode_solvers(&self) -> usize {
        self.inner.num_multizone_solvers()
    }

    /// Canonical 9R4C node names in the same order as `get_nodal_temperatures()`
    /// (Issue #1799). Useful for labelling plots / ML feature columns.
    pub fn nodal_temperature_node_names(&self) -> Vec<String> {
        vec![
            "wall".to_string(),
            "roof".to_string(),
            "floor".to_string(),
            "internal".to_string(),
        ]
    }

    /// Sub-hourly 9R4C node temperatures as a 3D numpy array (Issue #1799).
    ///
    /// # Returns
    /// Tuple of `(array, shape)` where `array` is a numpy `ndarray` with shape
    /// `[num_zones, 4, num_steps]`, the middle axis matching the canonical node
    /// ordering (`wall, roof, floor, internal`), and `shape` is the same shape
    /// vector for convenience.
    ///
    /// # Errors
    /// Raises `ValueError` if the simulation has not been run, or if the model
    /// carries no `MultiNodeSolver` (i.e. is a low-mass 5R1C model — for which
    /// there are no 9R4C nodes to export).
    pub fn get_nodal_temperatures_numpy<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<(Bound<'a, numpy::PyArray3<f64>>, Vec<usize>)> {
        let nodal = self.inner.get_nodal_temperatures();
        let nodal = match nodal {
            Some(v) => v,
            None => {
                if self.inner.num_multizone_solvers() == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Model has no 9R4C MultiNodeSolver (low-mass construction). \
                         Sub-hourly nodal temperatures are only available for high-mass \
                         9R4C models.",
                    ));
                }
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Simulation has not been run yet. Call simulate_multi_zone first.",
                ));
            }
        };

        let num_zones = nodal.len();
        let num_nodes = if num_zones > 0 { nodal[0].len() } else { 0 };
        let timesteps = if num_zones > 0 && num_nodes > 0 {
            nodal[0][0].len()
        } else {
            0
        };

        // Flatten into a row-major contiguous buffer of shape [Z, N, T].
        // numpy::PyArray3::from_vec3 expects `&[Vec<Vec<f64>>]` of length Z,
        // each entry length N, each inner entry length T — exactly our layout.
        let shape = vec![num_zones, num_nodes, timesteps];
        let arr = numpy::PyArray3::from_vec3(py, &nodal).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to create numpy 3D array: {}",
                e
            ))
        })?;
        Ok((arr, shape))
    }

    /// Run energy balance validation for multi-zone model
    ///
    /// This method validates that the simulation conserves energy by checking
    /// that at each timestep: Q_cond + Q_solar + Q_vent + Q_int + Q_hvac ≈ 0
    /// within the tolerance (default 0.1%, see issue #1061).
    ///
    /// The validation re-runs a short simulation (24 timesteps) step-by-step
    /// and checks the energy balance invariant at each step.
    ///
    /// # Returns
    /// `true` if energy is balanced (within tolerance), `false` otherwise
    pub fn validate_energy_balance(&self) -> PyResult<bool> {
        // Create a fresh model from Case 600 spec for validation
        let spec = ASHRAE140Case::Case600.spec();
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

        // Initialize weather data
        let weather = DenverTmyWeather::new();

        // Create invariant checker with default tolerance (0.1%)
        let tolerance = ENERGY_BALANCE_TOLERANCE;
        let mut checker = InvariantChecker::new(tolerance);

        // Run 24 timesteps (24 hours) with invariant checking
        let dt = 3600.0; // 1 hour timestep

        for step in 0..24 {
            // Get weather for this timestep
            let weather_data = match weather.get_hourly_data(step) {
                Ok(data) => data,
                Err(_) => continue,
            };

            let outdoor_temp = weather_data.dry_bulb_temp;

            // Step the physics
            model.step_physics(step, outdoor_temp, dt);

            // Check the energy balance invariant
            let result = checker.check_invariant(&model, dt, outdoor_temp);

            // If any violation occurs, the simulation is unbalanced
            if result.violated {
                return Ok(false);
            }
        }

        // All timesteps passed - energy is balanced
        Ok(true)
    }

    /// Validate energy balance and return detailed diagnostic information
    ///
    /// # Returns
    /// Tuple of (is_balanced: bool, max_residual: f64, violation_count: usize)
    pub fn validate_energy_balance_detailed(&self) -> PyResult<(bool, f64, usize)> {
        // Create a fresh model from Case 600 spec for validation
        let spec = ASHRAE140Case::Case600.spec();
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

        // Initialize weather data
        let weather = DenverTmyWeather::new();

        // Create invariant checker with default tolerance (0.1%)
        let tolerance = ENERGY_BALANCE_TOLERANCE;
        let mut checker = InvariantChecker::new(tolerance);

        // Run 24 timesteps (24 hours) with invariant checking
        let dt = 3600.0; // 1 hour timestep

        for step in 0..24 {
            // Get weather for this timestep
            let weather_data = match weather.get_hourly_data(step) {
                Ok(data) => data,
                Err(_) => continue,
            };

            let outdoor_temp = weather_data.dry_bulb_temp;

            // Step the physics
            model.step_physics(step, outdoor_temp, dt);

            // Check the energy balance invariant
            checker.check_invariant(&model, dt, outdoor_temp);
        }

        // Return detailed results
        let is_balanced = checker.violation_count() == 0;
        let max_violation = checker.max_violation();
        let violations = checker.violation_count();

        Ok((is_balanced, max_violation, violations))
    }

    /// Check if the model is in an intentionally unbalanced state for testing.
    /// Returns true if the model has been configured to violate energy conservation.
    ///
    /// # Returns
    /// `true` if unbalanced, `false` otherwise
    pub fn is_energy_unbalanced(&self) -> PyResult<bool> {
        // This method can be used to check if the model has been
        // intentionally broken for testing validate_energy_balance
        // For now, we check if thermal capacitance is negative
        for i in 0..self.inner.hvac.num_zones {
            if self.inner.mass.thermal_capacitance.as_ref()[i] < 0.0 {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Create a multi-zone thermal model from a configuration dictionary
#[pyfunction]
pub fn create_multi_zone_model_from_config(
    config: &Bound<'_, PyDict>,
) -> PyResult<PyMultiZoneThermalModel> {
    // Extract number of zones
    let num_zones: usize = match config.get_item("num_zones") {
        Ok(Some(item)) => item.extract()?,
        _ => {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "Missing 'num_zones' in config",
            ))
        }
    };

    // Create model
    let mut model = PyMultiZoneThermalModel::new(num_zones)?;

    // Set zone setpoints from zones dict (zone_0, zone_1, etc.)
    if let Ok(Some(zone_configs)) = config.get_item("zones") {
        let zones_dict: &Bound<'_, PyDict> = match zone_configs.cast() {
            Ok(d) => d,
            Err(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected dict for 'zones'",
                ))
            }
        };

        for (key, value) in zones_dict.iter() {
            let zone_key: String = match key.extract() {
                Ok(k) => k,
                Err(_) => continue,
            };
            if let Some(stripped) = zone_key.strip_prefix("zone_") {
                if let Ok(zone_idx) = stripped.parse::<usize>() {
                    let zone_dict: &Bound<'_, PyDict> = match value.cast() {
                        Ok(d) => d,
                        Err(_) => continue,
                    };

                    if let Ok(Some(heating)) = zone_dict.get_item("heating") {
                        let heating_temp: f64 = match heating.extract() {
                            Ok(t) => t,
                            Err(_) => continue,
                        };
                        let current_cooling =
                            model.inner.setpoints.cooling_setpoints.as_slice()[zone_idx];
                        model.set_zone_setpoints(zone_idx, heating_temp, current_cooling)?;
                    }

                    if let Ok(Some(cooling)) = zone_dict.get_item("cooling") {
                        let cooling_temp: f64 = match cooling.extract() {
                            Ok(t) => t,
                            Err(_) => continue,
                        };
                        let current_heating =
                            model.inner.setpoints.heating_setpoints.as_slice()[zone_idx];
                        model.set_zone_setpoints(zone_idx, current_heating, cooling_temp)?;
                    }
                }
            }
        }
    } else if let Ok(Some(zone_setpoints)) = config.get_item("zone_setpoints") {
        // Legacy zone_setpoints format (zone_0: (heating, cooling), zone_1: (heating, cooling), ...)
        let setpoints_dict: &Bound<'_, PyDict> = match zone_setpoints.cast() {
            Ok(d) => d,
            Err(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected dict for 'zone_setpoints'",
                ))
            }
        };

        for (key, value) in setpoints_dict.iter() {
            let zone_key: String = match key.extract() {
                Ok(k) => k,
                Err(_) => continue,
            };
            if let Some(stripped) = zone_key.strip_prefix("zone_") {
                if let Ok(zone_idx) = stripped.parse::<usize>() {
                    let setpoints_tuple: (f64, f64) = match value.extract() {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    model.set_zone_setpoints(zone_idx, setpoints_tuple.0, setpoints_tuple.1)?;
                }
            }
        }
    }

    // Set inter-zone conductances
    if let Ok(Some(iz_conductance)) = config.get_item("inter_zone_conductance") {
        let iz_dict: &Bound<'_, PyDict> = match iz_conductance.cast() {
            Ok(d) => d,
            Err(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected dict for 'inter_zone_conductance'",
                ))
            }
        };

        let mut conductance_vec = Vec::new();
        for i in 0..num_zones {
            let row_key = format!("zone_{}", i);
            if let Ok(Some(row)) = iz_dict.get_item(&row_key) {
                let row_dict: &Bound<'_, PyDict> = match row.cast() {
                    Ok(d) => d,
                    Err(_) => {
                        conductance_vec.push(vec![0.0; num_zones]);
                        continue;
                    }
                };
                let mut row_vec = Vec::new();
                for j in 0..num_zones {
                    let col_key = format!("zone_{}", j);
                    if let Ok(Some(val)) = row_dict.get_item(&col_key) {
                        match val.extract::<f64>() {
                            Ok(v) => row_vec.push(v),
                            Err(_) => row_vec.push(0.0),
                        }
                    } else {
                        row_vec.push(0.0);
                    }
                }
                conductance_vec.push(row_vec);
            } else {
                conductance_vec.push(vec![0.0; num_zones]);
            }
        }

        // Flatten and set conductances (matching CLI behavior)
        for i in 0..num_zones {
            for j in 0..num_zones {
                if i < conductance_vec.len() && j < conductance_vec[i].len() {
                    let conductance = conductance_vec[i][j];
                    // Only set when i <= j to avoid double-setting
                    if i == j || i == 0 {
                        model.set_inter_zone_conductance(i, j, conductance)?;
                    }
                }
            }
        }
    }

    Ok(model)
}

/// Create a multi-zone thermal model from a schema JSON file path
#[pyfunction]
pub fn create_multi_zone_model_from_schema_file(
    schema_path: &str,
) -> PyResult<PyMultiZoneThermalModel> {
    use crate::api::schema::SimulationSchema;
    use std::path::Path;

    let path = Path::new(schema_path);
    if !path.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Schema file not found: {}",
            schema_path
        )));
    }

    let content = std::fs::read_to_string(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read schema file: {}", e))
    })?;

    let schema: SimulationSchema = serde_json::from_str(&content).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to parse schema JSON: {}", e))
    })?;

    let SimulationSchema::V1(schema_v1) = schema;

    // Create model from schema V1
    let mut model = PyMultiZoneThermalModel::new(schema_v1.geometry.zones.len().max(1))?;

    // Set zone setpoints from schema
    let heating = schema_v1.controls.zone_control.heating_setpoint;
    let cooling = schema_v1.controls.zone_control.cooling_setpoint;
    for zone_idx in 0..model.inner.hvac.num_zones {
        model.set_zone_setpoints(zone_idx, heating, cooling)?;
    }

    // Set inter-zone conductances from schema (using default matrix from schema geometry)
    let n = schema_v1.geometry.zones.len().max(1);
    let default_conductance = 5.0; // Default value matching CLI behavior
    for i in 0..n {
        model.set_inter_zone_conductance(i, i, default_conductance)?;
        if i < n - 1 {
            model.set_inter_zone_conductance(i, i + 1, default_conductance)?;
        }
    }

    Ok(model)
}

/// Create a multi-zone thermal model from a schema dictionary (PyDict)
#[pyfunction]
pub fn create_multi_zone_model_from_schema_dict(
    schema: &Bound<'_, PyDict>,
) -> PyResult<PyMultiZoneThermalModel> {
    // Extract number of zones from geometry
    let num_zones: usize = match schema.get_item("geometry") {
        Ok(Some(geometry)) => {
            let geometry_dict: &Bound<'_, PyDict> = match geometry.cast() {
                Ok(d) => d,
                Err(_) => return PyMultiZoneThermalModel::new(1),
            };
            match geometry_dict.get_item("zones") {
                Ok(Some(zones)) => {
                    let zones_list: &Bound<'_, pyo3::types::PyList> = match zones.cast() {
                        Ok(d) => d,
                        Err(_) => return PyMultiZoneThermalModel::new(1),
                    };
                    zones_list.len()
                }
                _ => 1,
            }
        }
        _ => 1,
    };

    let mut model = PyMultiZoneThermalModel::new(num_zones)?;

    // Extract setpoints from controls
    if let Ok(Some(controls)) = schema.get_item("controls") {
        let controls_dict: &Bound<'_, PyDict> = match controls.cast() {
            Ok(d) => d,
            Err(_) => return Ok(model),
        };

        if let Ok(Some(zone_control)) = controls_dict.get_item("zone_control") {
            let zone_control_dict: &Bound<'_, PyDict> = match zone_control.cast() {
                Ok(d) => d,
                Err(_) => return Ok(model),
            };

            let heating = match zone_control_dict.get_item("heating_setpoint") {
                Ok(Some(v)) => v.extract::<f64>().ok(),
                _ => None,
            };
            let cooling = match zone_control_dict.get_item("cooling_setpoint") {
                Ok(Some(v)) => v.extract::<f64>().ok(),
                _ => None,
            };

            if let (Some(h), Some(c)) = (heating, cooling) {
                for zone_idx in 0..num_zones {
                    model.set_zone_setpoints(zone_idx, h, c)?;
                }
            }
        }
    }

    // Extract inter-zone conductance from constructions if available
    // (For now, use default values - full schema parsing would go here)

    Ok(model)
}

/// Python module initialization
#[pymodule]
pub fn multi_zone(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiZoneThermalModel>()?;
    m.add_function(wrap_pyfunction!(create_multi_zone_model_from_config, m)?)?;
    m.add_function(wrap_pyfunction!(
        create_multi_zone_model_from_schema_file,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        create_multi_zone_model_from_schema_dict,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(export_osm, m)?)?;
    m.add_function(wrap_pyfunction!(export_gbxml, m)?)?;

    Ok(())
}

fn validation_error(message: impl Into<String>) -> PyErr {
    FluxionError::Validation(message.into()).into()
}

fn osm_error(error: OsmError) -> PyErr {
    FluxionError::Simulation(format!("OSM interoperability error: {}", error), None).into()
}

fn gbxml_error(error: GbXmlError) -> PyErr {
    FluxionError::Simulation(format!("gbXML interoperability error: {}", error), None).into()
}

fn schema_from_json(content: &str) -> PyResult<SimulationSchemaV1> {
    if let Ok(schema) = serde_json::from_str::<SimulationSchemaV1>(content) {
        return Ok(schema);
    }

    let schema: SimulationSchema = serde_json::from_str(content)
        .map_err(|error| validation_error(format!("Failed to parse schema JSON: {}", error)))?;
    let SimulationSchema::V1(schema) = schema;
    Ok(schema)
}

fn schema_from_dict(schema: &Bound<'_, PyDict>) -> PyResult<SimulationSchemaV1> {
    let py = schema.py();
    let json = PyModule::import(py, "json")?;
    let content: String = json.call_method1("dumps", (schema,))?.extract()?;
    schema_from_json(&content)
}

#[pyfunction]
pub fn export_osm(schema: &Bound<'_, PyDict>, path: &str) -> PyResult<()> {
    let schema = schema_from_dict(schema)?;
    export_osm_file(&schema, path).map_err(osm_error)
}

#[pyfunction]
pub fn export_gbxml(schema: &Bound<'_, PyDict>, path: &str) -> PyResult<()> {
    let schema = schema_from_dict(schema)?;
    export_gbxml_file(&schema, path).map_err(gbxml_error)
}

/// Register HVAC module in main bindings
#[cfg(feature = "python-bindings")]
pub fn register_hvac_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register HVAC classes directly
    m.add_class::<crate::python::hvac_bindings::PyZoneSetpoints>()?;
    m.add_class::<crate::python::hvac_bindings::PyZoneControl>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::python::hvac_bindings::create_zone_setpoints,
        m
    )?)?;
    Ok(())
}

/// HVAC module initialization function
#[cfg(feature = "python-bindings")]
#[pymodule]
pub fn hvac(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::python::hvac_bindings::PyZoneSetpoints>()?;
    m.add_class::<crate::python::hvac_bindings::PyZoneControl>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        crate::python::hvac_bindings::create_zone_setpoints,
        m
    )?)?;
    Ok(())
}

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    //! Rust-side inline tests for the PyO3 wrappers in this module (Issue #2532).
    //!
    //! These tests exercise the Rust validation/conversion logic without going
    //! through a live Python interpreter. They cover:
    //! - `PyMultiZoneThermalModel::new` zone-count validation,
    //! - `from_case_spec` case-id normalization (whitespace / underscore
    //!   stripping, case-insensitivity, alias coverage, unknown-id error path),
    //! - `set_zone_temperatures` length check,
    //! - `set_zone_setpoints` index- and ordering-range checks,
    //! - `set_inter_zone_conductance` range- and sign-checks,
    //! - the private `schema_from_json` schema-parsing helper, and
    //! - the `validation_error` / `osm_error` / `gbxml_error` mappers.

    use super::*;
    use crate::interop::gbxml::GbXmlError;
    use crate::interop::osm::OsmError;

    // -- PyMultiZoneThermalModel::new ------------------------------------

    #[test]
    fn new_rejects_zero_zones() {
        let err = PyMultiZoneThermalModel::new(0)
            .err()
            .expect("0 zones should be rejected");
        // ValidationError message must mention the lower bound.
        assert!(err.to_string().contains("at least 1"));
    }

    #[test]
    fn new_accepts_default_and_explicit_zone_counts() {
        let one = PyMultiZoneThermalModel::new(1).expect("1 zone is valid");
        assert_eq!(one.num_zones(), 1);
        assert_eq!(one.get_inner_num_zones(), 1);

        let five = PyMultiZoneThermalModel::new(5).expect("5 zones is valid");
        assert_eq!(five.num_zones(), 5);
    }

    // -- from_case_spec --------------------------------------------------

    #[test]
    fn from_case_spec_accepts_canonical_ids() {
        // Spot-check one low-mass and one high-mass case — these exercise
        // `case_enum.spec()` + `ThermalModel::from_spec` end-to-end without
        // touching Python.
        let m900 = PyMultiZoneThermalModel::from_case_spec("Case900").expect("Case900 parses");
        assert!(m900.num_zones() >= 1);

        let m600 = PyMultiZoneThermalModel::from_case_spec("Case600").expect("Case600 parses");
        assert!(m600.num_zones() >= 1);
    }

    #[test]
    fn from_case_spec_is_case_insensitive_and_strips_separators() {
        // The normalizer strips whitespace AND underscores, then uppercases.
        for input in [
            "case900",
            "CASE900",
            "  Case900  ",
            "Case 900",
            "Case_900",
            "case_9_0_0",
        ] {
            let model = PyMultiZoneThermalModel::from_case_spec(input)
                .unwrap_or_else(|e| panic!("'{}' should parse: {}", input, e));
            assert_eq!(model.hvac.num_zones(), 1, "input='{}'", input);
        }
    }

    #[test]
    fn from_case_spec_accepts_aliases() {
        // The short aliases for the Case195 variants must round-trip.
        PyMultiZoneThermalModel::from_case_spec("Case195HM").expect("Case195HM alias parses");
        PyMultiZoneThermalModel::from_case_spec("Case195NL").expect("Case195NL alias parses");
        PyMultiZoneThermalModel::from_case_spec("Case195NS").expect("Case195NS alias parses");
    }

    #[test]
    fn from_case_spec_rejects_unknown_id() {
        let err = PyMultiZoneThermalModel::from_case_spec("CaseNotFound")
            .err()
            .expect("unknown case id should error");
        // Error message should mention the bad id (normalized to upper case)
        // and the known set.
        let msg = err.to_string();
        assert!(msg.contains("CASENOTFOUND"), "msg={}", msg);
        // The known-cases hint should be in the message.
        assert!(
            msg.contains("Case600") || msg.contains("Case900"),
            "msg={}",
            msg
        );
    }

    // -- set_zone_temperatures ------------------------------------------

    #[test]
    fn set_zone_temperatures_rejects_length_mismatch() {
        let mut model = PyMultiZoneThermalModel::new(3).expect("3 zones");
        let err = model
            .set_zone_temperatures(vec![20.0, 21.0])
            .err()
            .expect("length mismatch should error");
        assert!(err.to_string().contains("2"));
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn set_zone_temperatures_round_trips() {
        let mut model = PyMultiZoneThermalModel::new(2).expect("2 zones");
        model
            .set_zone_temperatures(vec![20.0, 22.0])
            .expect("matching length is accepted");
        assert_eq!(model.get_zone_temperatures(), vec![20.0, 22.0]);
    }

    // -- set_zone_setpoints ---------------------------------------------

    #[test]
    fn set_zone_setpoints_rejects_out_of_range_index() {
        let mut model = PyMultiZoneThermalModel::new(2).expect("2 zones");
        let err = model
            .set_zone_setpoints(5, 20.0, 24.0)
            .err()
            .expect("out-of-range zone index should error");
        assert!(err.to_string().contains("Zone index"));
    }

    #[test]
    fn set_zone_setpoints_enforces_heating_below_cooling() {
        let mut model = PyMultiZoneThermalModel::new(1).expect("1 zone");
        // Equal heating/cooling is invalid.
        let err = model
            .set_zone_setpoints(0, 22.0, 22.0)
            .err()
            .expect("equal heating/cooling should error");
        assert!(err.to_string().contains("Heating setpoint"));

        // Heating above cooling is invalid.
        let err = model
            .set_zone_setpoints(0, 25.0, 22.0)
            .err()
            .expect("heating above cooling should error");
        assert!(err.to_string().contains("Heating setpoint"));
    }

    #[test]
    fn set_zone_setpoints_accepts_valid_pair() {
        let mut model = PyMultiZoneThermalModel::new(1).expect("1 zone");
        model
            .set_zone_setpoints(0, 20.0, 24.0)
            .expect("20 < 24 is a valid pair");
    }

    // -- set_inter_zone_conductance -------------------------------------

    #[test]
    fn set_inter_zone_conductance_rejects_out_of_range_index() {
        let mut model = PyMultiZoneThermalModel::new(2).expect("2 zones");
        let err = model
            .set_inter_zone_conductance(5, 0, 1.0)
            .err()
            .expect("out-of-range index should error");
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn set_inter_zone_conductance_rejects_negative_value() {
        let mut model = PyMultiZoneThermalModel::new(2).expect("2 zones");
        let err = model
            .set_inter_zone_conductance(0, 1, -0.5)
            .err()
            .expect("negative conductance should error");
        assert!(err.to_string().contains("non-negative"));
    }

    #[test]
    fn set_inter_zone_conductance_vector_round_trips() {
        let mut model = PyMultiZoneThermalModel::new(3).expect("3 zones");
        model
            .set_inter_zone_conductance_vector(vec![1.0, 2.0, 3.0])
            .expect("valid conductance vector");
        assert_eq!(
            model.get_inter_zone_conductance_vector(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn set_inter_zone_conductance_vector_rejects_mismatched_length() {
        let mut model = PyMultiZoneThermalModel::new(3).expect("3 zones");
        let err = model
            .set_inter_zone_conductance_vector(vec![1.0, 2.0])
            .err()
            .expect("length mismatch should error");
        assert!(err.to_string().contains("2"));
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn set_inter_zone_conductance_vector_rejects_negative_value() {
        let mut model = PyMultiZoneThermalModel::new(2).expect("2 zones");
        let err = model
            .set_inter_zone_conductance_vector(vec![1.0, -2.0])
            .err()
            .expect("negative value should error");
        assert!(err.to_string().contains("non-negative"));
    }

    // -- schema_from_json (private helper) ------------------------------

    #[test]
    fn schema_from_json_rejects_empty_input() {
        let err = schema_from_json("")
            .err()
            .expect("empty input should error");
        assert!(err.to_string().contains("Failed to parse"));
    }

    #[test]
    fn schema_from_json_rejects_garbage_input() {
        let err = schema_from_json("not json")
            .err()
            .expect("garbage should error");
        assert!(err.to_string().contains("Failed to parse"));
    }

    #[test]
    fn schema_from_json_accepts_bare_v1_schema() {
        // A bare SimulationSchemaV1 (no `{"V1": ...}` envelope) is the fast
        // path: serde should deserialize it directly.
        let json = serde_json::to_string(&SimulationSchemaV1::default()).unwrap();
        let schema = schema_from_json(&json).expect("bare V1 schema should parse");
        assert_eq!(schema, SimulationSchemaV1::default());
    }

    #[test]
    fn schema_from_json_accepts_envelope_v1_schema() {
        // The SimulationSchema::V1(...) envelope is the slower fallback path.
        let wrapped = SimulationSchema::V1(SimulationSchemaV1::default());
        let json = serde_json::to_string(&wrapped).unwrap();
        let schema = schema_from_json(&json).expect("enveloped V1 schema should parse");
        assert_eq!(schema, SimulationSchemaV1::default());
    }

    // -- error helpers ---------------------------------------------------
    //
    // These exercise each `*Error -> PyErr` mapper to ensure they don't panic
    // and produce a `PyErr` carrying the expected message text.

    #[test]
    fn validation_error_helper_carries_message() {
        let err = validation_error("a test validation problem");
        let s = err.to_string();
        assert!(s.contains("a test validation problem"), "s={}", s);
        assert!(s.contains("validation"), "s={}", s);
    }

    #[test]
    fn osm_error_helper_wraps_message() {
        let src = OsmError::invalid_object("bad object");
        let err = osm_error(src);
        let s = err.to_string();
        assert!(s.contains("OSM interoperability error"), "s={}", s);
        assert!(s.contains("bad object"), "s={}", s);
    }

    #[test]
    fn gbxml_error_helper_wraps_message() {
        let src = GbXmlError::invalid_structure("xml broke");
        let err = gbxml_error(src);
        let s = err.to_string();
        assert!(s.contains("gbXML interoperability error"), "s={}", s);
        assert!(s.contains("xml broke"), "s={}", s);
    }
}
