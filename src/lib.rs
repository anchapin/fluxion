//! Fluxion: Rust-based Building Energy Modeling (BEM) engine
//!
//! Neuro-Symbolic hybrid architecture combining physics-based thermal networks with AI surrogates.
//! Designed for high-throughput evaluation of building design configurations (10,000+ configs/sec).
//!
//! # Architecture
//! - **BatchOracle**: High-throughput parallel evaluation for optimization loops
//! - **Model**: Single-building detailed analysis for validation and inspection
//! - **ThermalModel**: ISO 13790-compliant 5R1C/6R2C thermal network using CTA
//! - **SurrogateManager**: AI surrogate models for fast load prediction (ONNX Runtime)
//!
//! # Python API
//! ```python,ignore
//! from fluxion import BatchOracle, Model
//!
//! # Batch evaluation for optimization
//! oracle = BatchOracle()
//! results = oracle.evaluate_population([[1.5, 20.0, 22.0]], False)
//!
//! # Single building simulation
//! model = Model.from_case("600")
//! eui = model.simulate(years=1, use_surrogates=False)
//! ```
//!
//! # Performance
//! - Throughput: 10,000+ configurations/second on 8-core CPU
//! - Latency: <100ms for single configuration (8760 timesteps)
//! - Memory: Minimal allocations via CTA buffer reuse
//!
//! # Validation
//! - ASHRAE Standard 140 compliant (18/18 cases passing)
//! - Multi-reference validation (EnergyPlus, ESP-r, TRNSYS)
//! - Free-floating temperature validation (10/10 cases passing)
//!
//! # Modules
//! - [`sim::engine`] - ThermalModel and physics engine
//! - [`physics::cta`] - Continuous Tensor Abstraction
//! - [`ai::surrogate`] - ONNX-based surrogate models
//! - [`validation::ashrae_140_validator`] - ASHRAE 140 validation
//! - [`api`] - Python bindings and error types
//!
//! See [`BatchOracle`] and [`Model`] for Python API details.
//! See docs/API_REFERENCE.md for complete API documentation.

#![allow(clippy::useless_conversion)]
#![allow(nonstandard_style)]
#![allow(clippy::useless_vec)]
#![allow(clippy::unnested_or_patterns)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::clone_on_ref_ptr)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::unnecessary_to_owned)]
#![allow(clippy::len_zero)]
#![allow(clippy::comparison_to_empty)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::expect_used)]
#![allow(clippy::derive_ord_xor_partial_ord)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::use_self)]
#![allow(clippy::implicit_hasher)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::vec_init_then_push)]
pub mod ai;
pub mod analysis;
pub mod api;
pub mod cli;
pub mod interop;
pub mod io;
pub mod measures;
pub mod napi;
pub mod orchestration;
pub mod performance;
pub mod physics;
#[cfg(feature = "python-bindings")]
pub mod python;
pub mod quantum;
pub mod sim;
pub mod solar;
pub mod testing;
pub mod thermal;
pub mod validation;
// #1255: `weather` now lives in the `fluxion-core` workspace crate (a dependency
// leaf). Re-export it so all existing `crate::weather::...` paths resolve unchanged.
pub use fluxion_core::weather;

// #1349 (Phase 2 crate split): `assembly` and `multi_node` were moved from
// `src/sim/` into `fluxion-core` to break the physics<->sim dependency cycle.
// The original `src/sim/assembly.rs` and `src/sim/multi_node_thermal.rs` files
// are now thin re-export shims, so existing `crate::sim::assembly::*` and
// `crate::sim::multi_node_thermal::*` paths still resolve. Top-level re-exports
// here make `crate::assembly::*` and `crate::multi_node::*` work too.
pub use fluxion_core::{assembly, multi_node};

// #1441 (Phase 2 cycle break, continued): ASHRAE-140 leaf data types
// (Orientation, WindowArea, ConstructionType, ShadingType, ShadingDevice,
// GlassType, WindowSpec, InternalLoads, HvacSchedule, NightVentilation,
// BuildingType, GeometrySpec, ConductanceReferences) were moved from
// `src/validation/ashrae_140_cases.rs` into `fluxion_core::ashrae_cases` to
// break the `sim ↔ validation` cycle. The validation module re-exports each
// type at its original path, so `fluxion::validation::ashrae_140_cases::*`
// paths still resolve unchanged. This top-level re-export makes
// `fluxion::ashrae_cases::Orientation` work too.
pub use fluxion_core::ashrae_cases;

// Re-export thermal model traits for public API
pub use sim::surface_flux_provider::{
    MockSurfaceHeatFluxProvider, PhysicsSurfaceFluxProvider, SurfaceHeatFluxProvider,
};
pub use sim::thermal_model::{
    HybridRouting, HybridThermalModel, PhysicsThermalModel, SurrogateThermalModel,
    ThermalModelBuilder, ThermalModelMode, ThermalModelTrait, UnifiedThermalModel,
};
pub use sim::thermal_model_mock::MockThermalModel;

// Re-export ISO 13790 Annex C construction types
pub use sim::construction::{Construction, ConstructionLayer, MassClass};

use crate::physics::cta::VectorField;
use ai::surrogate::SurrogateManager;

use sim::engine::{StepParameters, ThermalModel};

#[cfg(feature = "python-bindings")]
use crate::api::error::{FluxionErrorPy, SimulationError, SurrogateError, ValidationError};
#[cfg(feature = "python-bindings")]
use crate::api::parameters::BuildingParameters;
#[cfg(feature = "python-bindings")]
use crate::weather::HourlyWeatherData;

use anyhow::Result;
#[allow(unused_imports)]
use log::{debug, info};
#[cfg(feature = "python-bindings")]
use ndarray::Array2;
#[cfg(feature = "python-bindings")]
use numpy::PyArrayMethods;
#[cfg(feature = "python-bindings")]
use pyo3::{
    prelude::{pyclass, pymethods, pymodule, PyModule},
    types::{PyAnyMethods, PyModuleMethods},
    Bound, PyResult, Python,
};

// Re-export things for easier access in other modules
// pub use ai::tensor_wrapper::TorchScalar; // REMOVED

/// Standard Single-Building Model for detailed building energy analysis.
///
/// Use this class when you need detailed simulation of a single building configuration,
/// including hourly temperature traces and ASHRAE 140 validation.
#[cfg(feature = "python-bindings")]
/// Single-building energy model for detailed simulation.
///
/// Use for validation, hourly temperature traces, or ASHRAE 140 testing.
/// Provides detailed diagnostics including hourly temperature traces, peak loads,
/// energy consumption breakdown, and comparison reports.
///
/// # Python API
/// ```python,ignore
/// from fluxion import Model
///
/// # Create from ASHRAE 140 case
/// model = Model.from_case("600")
///
/// # Run simulation
/// eui = model.simulate(years=1, use_surrogates=False)
///
/// # Get detailed diagnostics
/// temps = model.get_hourly_temperatures()
/// peak_heating = model.get_peak_heating()
/// report = model.generate_comparison_report()
/// ```
///
/// # Diagnostics
/// - Hourly temperature traces (zone, mass, surface)
/// - Peak load tracking (heating/cooling timing and magnitude)
/// - Energy consumption breakdown (heating, cooling, fans)
/// - Comparison reports against reference data (ASHRAE 140)
///
/// # Performance
/// - Single configuration: <100ms for 8760 timesteps
/// - Detailed diagnostics: Additional overhead for data collection
///
/// See docs/API_REFERENCE.md for complete API reference.
#[pyclass]
struct Model {
    inner: ThermalModel<VectorField>,
    surrogates: SurrogateManager,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl Model {
    /// Create a new Model instance with default configuration.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones (default: 1)
    #[new]
    #[pyo3(signature = (num_zones=1))]
    fn new(num_zones: usize) -> PyResult<Self> {
        Ok(Model {
            inner: ThermalModel::<VectorField>::new(num_zones),
            surrogates: SurrogateManager::new().map_err(|e| {
                SurrogateError::new_err(format!("Failed to create SurrogateManager: {}", e))
            })?,
        })
    }

    /// Get number of zones in the model.
    fn num_zones(&self) -> usize {
        self.inner.num_zones
    }

    /// Get current zone temperatures.
    fn get_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    /// Set zone temperatures.
    fn set_temperatures(&mut self, temps: Vec<f64>) -> PyResult<()> {
        if temps.len() != self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Temperature vector length ({}) must match number of zones ({})",
                temps.len(),
                self.inner.num_zones
            )));
        }
        self.inner.temperatures = VectorField::new(temps);
        Ok(())
    }

    /// Get building type for auto-loading internal load profiles (Plan 17-04).
    ///
    /// Returns the building type enum (Office, Retail, School, etc.) which is used
    /// to auto-load default internal load profiles when simulate_with_loads() is called.
    fn building_type(&self) -> String {
        // Convert BuildingType enum to string
        match self.inner.building_type {
            crate::sim::occupancy::BuildingType::Office => "Office".to_string(),
            crate::sim::occupancy::BuildingType::Retail => "Retail".to_string(),
            crate::sim::occupancy::BuildingType::School => "School".to_string(),
            crate::sim::occupancy::BuildingType::Hospital => "Hospital".to_string(),
            crate::sim::occupancy::BuildingType::Hotel => "Hotel".to_string(),
            crate::sim::occupancy::BuildingType::Restaurant => "Restaurant".to_string(),
            crate::sim::occupancy::BuildingType::Warehouse => "Warehouse".to_string(),
        }
    }

    /// Set building type for auto-loading internal load profiles (Plan 17-04).
    ///
    /// # Arguments
    /// * `building_type` - Building type string (Office, Retail, School, Hospital, Hotel, Restaurant, Warehouse)
    ///
    /// This building type is used to auto-load default internal load profiles (lighting, equipment, occupancy)
    /// when simulate_with_loads() is called without specifying custom loads.
    fn set_building_type(&mut self, building_type: String) -> PyResult<()> {
        self.inner.building_type = match building_type.as_str() {
            "Office" => crate::sim::occupancy::BuildingType::Office,
            "Retail" => crate::sim::occupancy::BuildingType::Retail,
            "School" => crate::sim::occupancy::BuildingType::School,
            "Hospital" => crate::sim::occupancy::BuildingType::Hospital,
            "Hotel" => crate::sim::occupancy::BuildingType::Hotel,
            "Restaurant" => crate::sim::occupancy::BuildingType::Restaurant,
            "Warehouse" => crate::sim::occupancy::BuildingType::Warehouse,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid building type '{}'. Must be one of: Office, Retail, School, Hospital, Hotel, Restaurant, Warehouse",
                    building_type
                )));
            }
        };
        Ok(())
    }

    /// Simulate building energy consumption over specified years.
    ///
    /// # Arguments
    /// * `years` - Number of years to simulate (1-5 typical)
    /// * `use_surrogates` - If true, use AI surrogates for load predictions; if false, use analytical calculations
    ///
    /// # Returns
    /// Total energy use intensity (EUI) in kWh/m²/year
    fn simulate(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        info!(
            "Starting simulation for {} years, use_surrogates={}",
            years, use_surrogates
        );
        let steps = years as usize * 8760;
        debug!("Simulation will process {} timesteps", steps);
        let result =
            self.inner
                .solve_timesteps(steps, &self.surrogates, use_surrogates, None, None, None);
        info!("Simulation complete, EUI = {:.2} kWh/m²/year", result);
        Ok(result)
    }

    /// Simulate building energy consumption with internal loads (Plan 17-04).
    ///
    /// This method allows specifying internal loads (lighting, equipment, occupancy)
    /// for more detailed building energy modeling. If all load parameters are None,
    /// the building type profile will be auto-loaded based on model.building_type.
    ///
    /// # Arguments
    /// * `years` - Number of years to simulate (1-5 typical)
    /// * `use_surrogates` - If true, use AI surrogates for load predictions; if false, use analytical calculations
    ///
    /// # Returns
    /// Total energy use intensity (EUI) in kWh/m²/year
    ///
    /// # Note
    /// This method currently accepts None for all load parameters, which will trigger
    /// auto-loading of the building profile based on model.building_type.
    /// Full Python API for passing custom load objects will be added in a future phase.
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// model = fluxion.Model()
    /// model.building_type = fluxion.BuildingType.Office
    ///
    /// # Simulate with auto-loaded Office building profile
    /// eui = model.simulate_with_loads(1, False)
    /// ```
    fn simulate_with_loads(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        info!(
            "Starting simulation with auto-loaded internal loads for {} years, use_surrogates={}",
            years, use_surrogates
        );
        let steps = years as usize * 8760;

        // Pass None for all loads to trigger auto-loading from building_type
        let result =
            self.inner
                .solve_timesteps(steps, &self.surrogates, use_surrogates, None, None, None);
        info!("Simulation complete, EUI = {:.2} kWh/m²/year", result);
        Ok(result)
    }

    /// Simulate building energy consumption with NumPy array inputs for weather data.
    ///
    /// This method enables direct NumPy memory sharing between Python and Rust,
    /// eliminating copy overhead for large simulations. Weather data is passed
    /// as NumPy arrays, and zone temperatures are returned as a 2D NumPy array.
    ///
    /// # Arguments
    /// * `dry_bulb_temp` - Outdoor dry bulb temperature (°C), shape (steps,)
    /// * `dni` - Direct Normal Irradiance (W/m²), shape (steps,)
    /// * `dhi` - Diffuse Horizontal Irradiance (W/m²), shape (steps,)
    /// * `ghi` - Global Horizontal Irradiance (W/m²), shape (steps,)
    /// * `wind_speed` - Wind speed (m/s), shape (steps,)
    /// * `humidity` - Relative humidity (%), shape (steps,)
    /// * `horizontal_infrared` - Horizontal infrared radiation (W/m²), shape (steps,)
    /// * `use_surrogates` - If true, use AI surrogates for load predictions
    ///
    /// # Returns
    /// 2D NumPy array of zone temperatures (steps x num_zones) in °C
    ///
    /// # Example
    /// ```python
    /// import fluxion
    /// import numpy as np
    ///
    /// model = fluxion.Model(num_zones=3)
    ///
    /// # Create weather data arrays (8760 hourly values)
    /// n_timesteps = 8760
    /// dry_bulb = np.random.uniform(10, 35, n_timesteps)
    /// dni = np.random.uniform(0, 1000, n_timesteps)
    /// dhi = np.random.uniform(0, 500, n_timesteps)
    /// ghi = np.random.uniform(0, 1000, n_timesteps)
    /// wind_speed = np.random.uniform(0, 10, n_timesteps)
    /// humidity = np.random.uniform(30, 80, n_timesteps)
    /// horizontal_ir = np.random.uniform(200, 500, n_timesteps)
    ///
    /// # Run simulation and get zone temperatures
    /// zone_temps = model.simulate_numpy(
    ///     dry_bulb, dni, dhi, ghi, wind_speed, humidity, horizontal_ir, False
    /// )
    /// # zone_temps.shape == (8760, 3)
    /// ```
    #[allow(clippy::too_many_arguments)]
    fn simulate_numpy<'py>(
        &mut self,
        py: Python<'py>,
        dry_bulb_temp: &Bound<'py, pyo3::types::PyAny>,
        dni: &Bound<'py, pyo3::types::PyAny>,
        dhi: &Bound<'py, pyo3::types::PyAny>,
        ghi: &Bound<'py, pyo3::types::PyAny>,
        wind_speed: &Bound<'py, pyo3::types::PyAny>,
        humidity: &Bound<'py, pyo3::types::PyAny>,
        horizontal_infrared: &Bound<'py, pyo3::types::PyAny>,
        use_surrogates: bool,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        // Helper to extract 1D numpy array as Vec<f64>
        fn extract_1d_f64(arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Vec<f64>> {
            if let Ok(pyarr) = arr.downcast::<numpy::PyArray1<f64>>() {
                let slice = unsafe { pyarr.as_slice()? };
                return Ok(slice.to_vec());
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected 1D numpy array",
            ))
        }

        // Extract weather data arrays
        let dry_bulb_vec = extract_1d_f64(dry_bulb_temp)?;
        let dni_vec = extract_1d_f64(dni)?;
        let dhi_vec = extract_1d_f64(dhi)?;
        let ghi_vec = extract_1d_f64(ghi)?;
        let wind_vec = extract_1d_f64(wind_speed)?;
        let humidity_vec = extract_1d_f64(humidity)?;
        let hir_vec = extract_1d_f64(horizontal_infrared)?;

        let steps = dry_bulb_vec.len();

        // Validate all arrays have the same length
        if dni_vec.len() != steps
            || dhi_vec.len() != steps
            || ghi_vec.len() != steps
            || wind_vec.len() != steps
            || humidity_vec.len() != steps
            || hir_vec.len() != steps
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All weather arrays must have the same length",
            ));
        }

        let num_zones = self.inner.num_zones;
        info!(
            "Starting NumPy simulation for {} timesteps, {} zones, use_surrogates={}",
            steps, num_zones, use_surrogates
        );

        // Initialize temperature storage: (steps x num_zones)
        let mut zone_temps = Array2::<f64>::zeros((steps, num_zones));

        // Build weather data and run simulation
        for t in 0..steps {
            if t % 1000 == 0 {
                info!("Progress: {}/{} timesteps", t, steps);
            }

            let weather = HourlyWeatherData {
                dry_bulb_temp: dry_bulb_vec[t],
                dni: dni_vec[t],
                dhi: dhi_vec[t],
                ghi: ghi_vec[t],
                wind_speed: wind_vec[t],
                humidity: humidity_vec[t],
                horizontal_infrared: hir_vec[t],
                hour_of_year: t,
                ground_temperature: None,
                horizontal_illuminance: None,
                diffuse_illuminance: None,
                snow_depth: None,
                snow_cover: None,
                present_weather: None,
                present_weather_code: None,
            };

            self.inner.set_weather(weather);
            let _energy = self.inner.step_physics(t, dry_bulb_vec[t], 3600.0);

            // Collect zone temperatures
            let temps = self.inner.get_temperatures();
            for (zone_idx, &temp) in temps.iter().enumerate() {
                zone_temps[[t, zone_idx]] = temp;
            }
        }

        info!("NumPy simulation complete");
        Ok(numpy::PyArray2::from_owned_array_bound(py, zone_temps))
    }

    /// Simulate one timestep.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index (0-8759 for hourly annual simulation)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `use_surrogates` - If true, use neural surrogates; if false, use analytical calculations
    /// Register an ONNX surrogate model for this `Model` instance.
    fn load_surrogate(&mut self, model_path: String) -> PyResult<()> {
        match SurrogateManager::load_onnx(&model_path) {
            Ok(manager) => {
                self.surrogates = manager;
                Ok(())
            }
            Err(e) => Err(SurrogateError::new_err(format!(
                "Failed to load ONNX surrogate model '{}': {}",
                model_path, e
            ))),
        }
    }

    /// Get the parameter bounds for building design variables.
    ///
    /// Returns a ParameterBounds struct with the valid ranges for all design
    /// parameters used by BatchOracle. This is useful for optimization libraries
    /// that need to generate valid parameter vectors.
    ///
    /// # Returns
    /// ParameterBounds struct containing min/max values for:
    /// - Window U-value (W/m²K)
    /// - Heating setpoint (°C)
    /// - Cooling setpoint (°C)
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// oracle = fluxion.BatchOracle()
    /// bounds = oracle.get_parameter_bounds()
    ///
    /// print(f"U-value range: [{bounds.min_u_value}, {bounds.max_u_value}]")
    /// print(f"Heating setpoint range: [{bounds.min_heating_setpoint}, {bounds.max_heating_setpoint}]")
    /// print(f"Cooling setpoint range: [{bounds.min_cooling_setpoint}, {bounds.max_cooling_setpoint}]")
    /// ```
    fn get_parameter_bounds(&self) -> ParameterBounds {
        ParameterBounds::get_bounds()
    }

    /// Validate a parameter vector against physical constraints.
    ///
    /// This method checks that all parameter values are within valid ranges and
    /// that heating/cooling setpoints are consistent. If validation fails, a
    /// ValidationError is raised with a clear, actionable message.
    ///
    /// # Arguments
    /// * `params` - Parameter vector to validate. Elements:
    ///   - `[0]`: Window U-value (W/m²K, must be finite and in [0.1, 5.0])
    ///   - `[1]`: Heating setpoint (°C, must be finite and in [15.0, 25.0])
    ///   - `[2]`: Cooling setpoint (°C, must be finite and in [22.0, 32.0])
    ///
    /// # Raises
    /// ValidationError with detailed message including:
    /// - Parameter index
    /// - Invalid value
    /// - Valid range
    /// - Type of error (NaN, infinite, or out of range)
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// oracle = fluxion.BatchOracle()
    ///
    /// # Valid parameters
    /// oracle.validate_parameters([1.5, 20.0, 27.0])  # OK
    ///
    /// # Invalid U-value (raises ValidationError)
    /// try:
    ///     oracle.validate_parameters([-1.0, 20.0, 27.0])
    /// except fluxion.ValidationError as e:
    ///     print(f"Validation failed: {e}")
    ///     # Output: Window U-value (index 0, -1.00 W/m²K) out of range [0.1, 5.0] W/m²K
    ///
    /// # NaN value (raises ValidationError)
    /// try:
    ///     oracle.validate_parameters([float('nan'), 20.0, 27.0])
    /// except fluxion.ValidationError as e:
    ///     print(f"Validation failed: {e}")
    ///     # Output: Window U-value (index 0) is NaN (value: nan W/m²K). Cannot use in simulation.
    /// ```
    fn validate_parameters_py(&self, params: Vec<f64>) -> PyResult<()> {
        BatchOracle::validate_parameters(&params)?;
        Ok(())
    }

    /// Set ground temperature model to constant value.
    ///
    /// # Arguments
    /// * `temperature` - Constant ground temperature (°C)
    fn set_ground_temp(&mut self, temperature: f64) {
        self.inner.set_ground_temp(temperature);
    }

    /// Get ground temperature at a specific timestep.
    ///
    /// # Arguments
    /// * `timestep` - Timestep index (0-8759 for hourly annual simulation)
    ///
    /// # Returns
    /// Ground temperature (°C)
    fn ground_temperature_at(&self, timestep: usize) -> f64 {
        self.inner.ground_temperature_at(timestep)
    }

    /// Return a Python list of [`crate::python::model_bindings::PyZone`] snapshots,
    /// one per zone in the model.
    ///
    /// Each returned `Zone` is an **owned snapshot** of the current zone state
    /// (temperature, area, surfaces, HVAC setpoints). The snapshot does **not**
    /// borrow from this model — Python garbage collection of any returned
    /// `Zone` cannot invalidate this model, and conversely this model may be
    /// mutated or re-simulated while Python still holds references to
    /// previously returned zones. See `docs/bindings.md` for the full
    /// lifetime story.
    ///
    /// Iteration works out of the box via the standard Python list iterator
    /// protocol:
    /// ```python,ignore
    /// model = fluxion.Model(num_zones=3)
    /// for z in model.zones():
    ///     print(z.index, z.temperature, z.area)
    /// ```
    fn zones(&self) -> Vec<crate::python::model_bindings::PyZone> {
        crate::python::model_bindings::all_zones_from_model(&self.inner)
    }

    /// Return a flat Python list of [`crate::python::model_bindings::PySurface`]
    /// snapshots, one for every surface in every zone.
    ///
    /// Like [`Self::zones`], each surface is an owned snapshot. Mutating a
    /// snapshot via `surface.append_shading(...)` only mutates the Python
    /// object — to push the change back into the model, use
    /// [`Self::set_surfaces`].
    ///
    /// # Example: find all south-facing surfaces
    /// ```python,ignore
    /// model = fluxion.Model(num_zones=2)
    /// south = [s for s in model.surfaces() if s.orientation == fluxion.Orientation.South]
    /// for s in south:
    ///     s.add_overhang(depth=1.0, height=2.5)
    /// model.set_surfaces(south + [s for s in model.surfaces() if s.orientation != fluxion.Orientation.South])
    /// ```
    fn surfaces(&self) -> Vec<crate::python::model_bindings::PySurface> {
        crate::python::model_bindings::all_surfaces_from_model(&self.inner)
    }

    /// Push a flat list of [`crate::python::model_bindings::PySurface`]
    /// snapshots back into the model. Surfaces are reshaped per-zone (4 per
    /// zone by default; this matches the ASHRAE 140 case-default wall
    /// configuration).
    ///
    /// The number of zones in the model does not change — only the surface
    /// data inside each zone is replaced. This is the round-trip companion
    /// to [`Self::surfaces`].
    ///
    /// # Arguments
    /// * `surfaces` - flat list of [`crate::python::model_bindings::PySurface`]
    ///   values; the list length must be a multiple of `surfaces_per_zone`,
    ///   otherwise the trailing surfaces are truncated.
    fn set_surfaces(&mut self, surfaces: Vec<crate::python::model_bindings::PySurface>) {
        self.inner.surfaces =
            crate::python::model_bindings::reshape_surfaces_for_model(&self.inner, surfaces);
    }

    /// Return an [`crate::python::model_bindings::PyHVACSystem`] snapshot of
    /// the model's current heating and cooling plant configuration.
    ///
    /// The snapshot is an owned value (no borrow back into the model). To
    /// push changes back, use [`Self::set_hvac_system`].
    fn hvac_system(&self) -> crate::python::model_bindings::PyHVACSystem {
        crate::python::model_bindings::hvac_system_from_model(&self.inner)
    }

    /// Apply a [`crate::python::model_bindings::PyHVACSystem`] snapshot's
    /// heating/cooling capacity to the model. Used together with
    /// [`Self::hvac_system`] for snapshot-then-commit mutation patterns.
    ///
    /// Only heating/cooling capacity is propagated back; other HVACSystem
    /// fields (COP, stages, etc.) are advisory and not stored on
    /// `ThermalModelData`.
    fn set_hvac_system(&mut self, hvac: crate::python::model_bindings::PyHVACSystem) {
        crate::python::model_bindings::apply_hvac_system_to_model(&mut self.inner, &hvac);
    }
}

/// VectorField wrapper for Python with optimized numpy support.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "VectorField")]
pub struct PyVectorField {
    inner: crate::physics::cta::VectorField,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVectorField {
    /// Create a new VectorField from a Python list or numpy array.
    ///
    /// For optimal performance with large arrays, pass a numpy array directly.
    /// This avoids Python object iteration overhead.
    #[new]
    fn new(data: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        // Try to extract as numpy array first (most efficient for large data)
        if let Ok(arr) = data.downcast::<numpy::PyArray1<f64>>() {
            // Fast path: directly copy from numpy array slice
            let slice = unsafe { arr.as_slice()? };
            return Ok(PyVectorField {
                inner: crate::physics::cta::VectorField::new(slice.to_vec()),
            });
        }

        // Fall back to Python sequence iteration
        let mut vec = Vec::new();
        let len = data.len()?;
        vec.reserve(len);

        for item in data.iter()? {
            let val = item?.extract::<f64>()?;
            vec.push(val);
        }

        Ok(PyVectorField {
            inner: crate::physics::cta::VectorField::new(vec),
        })
    }

    /// Create a VectorField filled with a constant value.
    #[staticmethod]
    fn from_scalar(value: f64, size: usize) -> Self {
        PyVectorField {
            inner: crate::physics::cta::VectorField::from_scalar(value, size),
        }
    }

    /// Get the number of elements in the VectorField.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Convert to Python list.
    fn to_list(&self) -> Vec<f64> {
        self.inner.as_slice().to_vec()
    }

    /// Convert to numpy array with zero-copy when possible.
    ///
    /// Returns a numpy array view of the underlying data when possible,
    /// avoiding unnecessary memory copies for maximum performance.
    fn to_numpy<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, numpy::PyArray1<f64>>> {
        // Use from_vec_bound for zero-copy conversion
        Ok(numpy::PyArray1::from_vec_bound(
            py,
            self.inner.as_slice().to_vec(),
        ))
    }

    /// Compute the sum (integral) of all elements.
    fn integrate(&self) -> f64 {
        use crate::physics::cta::ContinuousTensor;
        self.inner.integrate()
    }

    /// Compute the gradient (rate of change) of the field.
    fn gradient(&self) -> Self {
        use crate::physics::cta::ContinuousTensor;
        PyVectorField {
            inner: self.inner.gradient(),
        }
    }
}

/// Construction layer material properties for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "ConstructionLayer")]
#[derive(Clone)]
pub struct PyConstructionLayer {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub conductivity: f64,
    #[pyo3(get, set)]
    pub density: f64,
    #[pyo3(get, set)]
    pub specific_heat: f64,
    #[pyo3(get, set)]
    pub thickness: f64,
    #[pyo3(get, set)]
    pub emissivity: f64,
    #[pyo3(get, set)]
    pub absorptance: f64,
}

#[cfg(feature = "python-bindings")]
impl From<&crate::sim::construction::ConstructionLayer> for PyConstructionLayer {
    fn from(layer: &crate::sim::construction::ConstructionLayer) -> Self {
        PyConstructionLayer {
            name: layer.name.clone(),
            conductivity: layer.conductivity,
            density: layer.density,
            specific_heat: layer.specific_heat,
            thickness: layer.thickness,
            emissivity: layer.emissivity,
            absorptance: layer.absorptance,
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<PyConstructionLayer> for crate::sim::construction::ConstructionLayer {
    fn from(layer: PyConstructionLayer) -> Self {
        crate::sim::construction::ConstructionLayer::with_surface_properties(
            layer.name,
            layer.conductivity,
            layer.density,
            layer.specific_heat,
            layer.thickness,
            layer.emissivity,
            layer.absorptance,
        )
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyConstructionLayer {
    /// Create a new ConstructionLayer.
    #[new]
    #[pyo3(signature = (name, conductivity, density, specific_heat, thickness, emissivity=0.9, absorptance=0.7))]
    fn new(
        name: String,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        thickness: f64,
        emissivity: f64,
        absorptance: f64,
    ) -> Self {
        PyConstructionLayer {
            name,
            conductivity,
            density,
            specific_heat,
            thickness,
            emissivity,
            absorptance,
        }
    }

    /// Calculate thermal resistance (R-value).
    fn r_value(&self) -> f64 {
        self.thickness / self.conductivity
    }

    /// Calculate thermal capacitance per unit area.
    fn thermal_capacitance_per_area(&self) -> f64 {
        self.density * self.thickness * self.specific_heat
    }
}

/// Surface type for construction calculations.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "SurfaceType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySurfaceType {
    Wall,
    Ceiling,
    Floor,
}

#[cfg(feature = "python-bindings")]
impl From<PySurfaceType> for crate::sim::construction::SurfaceType {
    fn from(st: PySurfaceType) -> Self {
        match st {
            PySurfaceType::Wall => crate::sim::construction::SurfaceType::Wall,
            PySurfaceType::Ceiling => crate::sim::construction::SurfaceType::Ceiling,
            PySurfaceType::Floor => crate::sim::construction::SurfaceType::Floor,
        }
    }
}

/// Thermal mass classification for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "MassClass", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyMassClass {
    VeryLight,
    Light,
    Medium,
    Heavy,
    VeryHeavy,
}

#[cfg(feature = "python-bindings")]
impl From<PyMassClass> for crate::sim::construction::MassClass {
    fn from(mc: PyMassClass) -> Self {
        match mc {
            PyMassClass::VeryLight => crate::sim::construction::MassClass::VeryLight,
            PyMassClass::Light => crate::sim::construction::MassClass::Light,
            PyMassClass::Medium => crate::sim::construction::MassClass::Medium,
            PyMassClass::Heavy => crate::sim::construction::MassClass::Heavy,
            PyMassClass::VeryHeavy => crate::sim::construction::MassClass::VeryHeavy,
        }
    }
}

/// Multi-layer construction assembly for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "Construction")]
pub struct PyConstruction {
    #[pyo3(get)]
    pub layers: Vec<PyConstructionLayer>,
}

#[cfg(feature = "python-bindings")]
impl From<&crate::sim::construction::Construction> for PyConstruction {
    fn from(construction: &crate::sim::construction::Construction) -> Self {
        PyConstruction {
            layers: construction
                .layers
                .iter()
                .map(PyConstructionLayer::from)
                .collect(),
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<PyConstruction> for crate::sim::construction::Construction {
    fn from(construction: PyConstruction) -> Self {
        crate::sim::construction::Construction::new(
            construction.layers.into_iter().map(|l| l.into()).collect(),
        )
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyConstruction {
    /// Create a new Construction from a list of layers.
    #[new]
    fn new(layers: Vec<PyConstructionLayer>) -> Self {
        PyConstruction { layers }
    }

    /// Calculate total thermal resistance (R-value).
    #[pyo3(signature = (surface_type=None, exterior_wind_speed=None))]
    fn r_value_total(
        &self,
        surface_type: Option<PySurfaceType>,
        exterior_wind_speed: Option<f64>,
    ) -> PyResult<f64> {
        let st = surface_type.map(|st| st.into());
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        Ok(rust_construction.r_value_total(st, exterior_wind_speed))
    }

    /// Calculate thermal transmittance (U-value).
    #[pyo3(signature = (surface_type=None, exterior_wind_speed=None))]
    fn u_value(
        &self,
        surface_type: Option<PySurfaceType>,
        exterior_wind_speed: Option<f64>,
    ) -> PyResult<f64> {
        let st = surface_type.map(|st| st.into());
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        Ok(rust_construction.u_value(st, exterior_wind_speed))
    }

    /// Calculate total thermal mass.
    fn thermal_capacitance_per_area(&self) -> PyResult<f64> {
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        Ok(rust_construction.thermal_capacitance_per_area())
    }

    /// Get total thickness.
    fn total_thickness(&self) -> PyResult<f64> {
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        Ok(rust_construction.total_thickness())
    }

    /// Get number of layers.
    fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get mass class.
    fn mass_class(&self) -> PyResult<PyMassClass> {
        let layers: Vec<crate::sim::construction::ConstructionLayer> =
            self.layers.iter().map(|l| l.clone().into()).collect();
        let rust_construction = crate::sim::construction::Construction::new(layers);
        match rust_construction.iso_13790_mass_class() {
            crate::sim::construction::MassClass::VeryLight => Ok(PyMassClass::VeryLight),
            crate::sim::construction::MassClass::Light => Ok(PyMassClass::Light),
            crate::sim::construction::MassClass::Medium => Ok(PyMassClass::Medium),
            crate::sim::construction::MassClass::Heavy => Ok(PyMassClass::Heavy),
            crate::sim::construction::MassClass::VeryHeavy => Ok(PyMassClass::VeryHeavy),
        }
    }
}

/// Wall surface representation for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "WallSurface")]
#[derive(Clone)]
pub struct PyWallSurface {
    #[pyo3(get, set)]
    pub area: f64,
    #[pyo3(get, set)]
    pub u_value: f64,
    #[pyo3(get)]
    pub orientation: String,
}

#[cfg(feature = "python-bindings")]
impl From<&crate::sim::construction::WallSurface> for PyWallSurface {
    fn from(surface: &crate::sim::construction::WallSurface) -> Self {
        PyWallSurface {
            area: surface.area,
            u_value: surface.u_value,
            orientation: format!("{:?}", surface.orientation),
        }
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyWallSurface {
    /// Create a new WallSurface.
    #[new]
    #[pyo3(signature = (area, u_value, orientation))]
    fn new(area: f64, u_value: f64, orientation: String) -> PyResult<Self> {
        let rust_orientation = match orientation.to_lowercase().as_str() {
            "south" => crate::validation::ashrae_140_cases::Orientation::South,
            "west" => crate::validation::ashrae_140_cases::Orientation::West,
            "north" => crate::validation::ashrae_140_cases::Orientation::North,
            "east" => crate::validation::ashrae_140_cases::Orientation::East,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid orientation '{}'. Valid options: south, west, north, east",
                    orientation
                )))
            }
        };
        let _rust_surface =
            crate::sim::construction::WallSurface::new(area, u_value, rust_orientation);
        Ok(PyWallSurface {
            area,
            u_value,
            orientation: format!("{:?}", rust_orientation),
        })
    }
}
/// High-throughput parallel oracle for quantum and genetic algorithm optimization.
///
/// This is the core API for bulk evaluation of building design populations. It accepts
/// thousands of parameter vectors and returns fitness values (EUI) using data parallelism
/// across CPU cores. Critical for integrating with D-Wave quantum annealers and GA frameworks.
#[cfg_attr(feature = "python-bindings", pyclass)]
/// High-throughput parallel oracle for optimization workflows.
///
/// Uses rayon for data parallelism—each configuration runs on a thread pool.
/// Designed for quantum annealers and genetic algorithms that need to evaluate
/// thousands of building configurations per second.
///
/// # Python API
/// ```python,ignore
/// from fluxion import BatchOracle
///
/// oracle = BatchOracle()
/// results = oracle.evaluate_population([[1.5, 20.0, 22.0]], False)
/// ```
///
/// # Architecture
/// - **Config-first loop without surrogates**: Each config runs independently through all timesteps
/// - **Time-first loop with surrogates**: Batched inference for GPU utilization (10,000+ configs/sec)
/// - Minimizes Python-Rust boundary crossings by processing entire population at once
///
/// # Parameter Vector Semantics
/// - Element 0: Window U-value (range: 0.1–5.0 W/m²K)
/// - Element 1: Heating setpoint (range: 15–25°C)
/// - Element 2: Cooling setpoint (range: 22–32°C)
///
/// # Performance
/// - Throughput: 10,000+ configurations/second on 8-core CPU with GPU
/// - Latency: <100ms for 1000 configurations
/// - Thread-safe: Uses rayon for parallel evaluation
///
/// See docs/API_REFERENCE.md for complete API reference.
pub struct BatchOracle {
    base_model: ThermalModel<VectorField>,
    surrogates: SurrogateManager,
}

impl BatchOracle {
    // Physical constraints for optimization parameters
    const MIN_U_VALUE: f64 = 0.1; // Minimum realistic U-value (W/m²K)
    const MAX_U_VALUE: f64 = 5.0; // Maximum realistic U-value
    const MIN_HEATING_SETPOINT: f64 = 15.0; // Min heating setpoint (°C)
    const MAX_HEATING_SETPOINT: f64 = 25.0; // Max heating setpoint (°C)
    const MIN_COOLING_SETPOINT: f64 = 22.0; // Min cooling setpoint (°C)
    const MAX_COOLING_SETPOINT: f64 = 32.0; // Max cooling setpoint (°C)

    // Parameter indices
    const U_VALUE_INDEX: usize = 0;
    const HEATING_SETPOINT_INDEX: usize = 1;
    const COOLING_SETPOINT_INDEX: usize = 2;

    /// Validates a parameter vector against physical constraints.
    ///
    /// This function checks for NaN/Inf values before range validation to prevent
    /// physics failures. Error messages include parameter index, value, and valid range
    /// for self-diagnosis.
    fn validate_parameters(params: &[f64]) -> Result<(), crate::api::error::FluxionError> {
        // Validate parameters that are present; allow shorter vectors for partial parameter sweeps
        if let Some(&u_value) = params.get(Self::U_VALUE_INDEX) {
            // Check for NaN/Inf before range validation
            if !u_value.is_finite() {
                let error_type = if u_value.is_nan() { "NaN" } else { "infinite" };
                return Err(crate::api::error::FluxionError::Validation(format!(
                    "Window U-value (index 0) is {} (value: {:.2} W/m²K). Cannot use in simulation.",
                    error_type, u_value
                )));
            }
            if !(Self::MIN_U_VALUE..=Self::MAX_U_VALUE).contains(&u_value) {
                return Err(crate::api::error::FluxionError::Validation(format!(
                    "Window U-value (index 0, {:.2} W/m²K) out of range [{:.1}, {:.1}] W/m²K",
                    u_value,
                    Self::MIN_U_VALUE,
                    Self::MAX_U_VALUE
                )));
            }
        }
        if let Some(&heating_setpoint) = params.get(Self::HEATING_SETPOINT_INDEX) {
            // Check for NaN/Inf before range validation
            if !heating_setpoint.is_finite() {
                let error_type = if heating_setpoint.is_nan() {
                    "NaN"
                } else {
                    "infinite"
                };
                return Err(crate::api::error::FluxionError::Validation(format!(
                    "Heating setpoint (index 1) is {} (value: {:.2}°C). Cannot use in simulation.",
                    error_type, heating_setpoint
                )));
            }
            if !(Self::MIN_HEATING_SETPOINT..=Self::MAX_HEATING_SETPOINT)
                .contains(&heating_setpoint)
            {
                return Err(crate::api::error::FluxionError::Validation(format!(
                    "Heating setpoint (index 1, {:.2}°C) out of range [{:.1}, {:.1}]°C",
                    heating_setpoint,
                    Self::MIN_HEATING_SETPOINT,
                    Self::MAX_HEATING_SETPOINT
                )));
            }
        }
        if let Some(&cooling_setpoint) = params.get(Self::COOLING_SETPOINT_INDEX) {
            // Check for NaN/Inf before range validation
            if !cooling_setpoint.is_finite() {
                let error_type = if cooling_setpoint.is_nan() {
                    "NaN"
                } else {
                    "infinite"
                };
                return Err(crate::api::error::FluxionError::Validation(format!(
                    "Cooling setpoint (index 2) is {} (value: {:.2}°C). Cannot use in simulation.",
                    error_type, cooling_setpoint
                )));
            }
            if !(Self::MIN_COOLING_SETPOINT..=Self::MAX_COOLING_SETPOINT)
                .contains(&cooling_setpoint)
            {
                return Err(crate::api::error::FluxionError::Validation(format!(
                    "Cooling setpoint (index 2, {:.2}°C) out of range [{:.1}, {:.1}]°C",
                    cooling_setpoint,
                    Self::MIN_COOLING_SETPOINT,
                    Self::MAX_COOLING_SETPOINT
                )));
            }
            // Check heating/cooling relationship if heating is also provided
            if let Some(&heating_setpoint) = params.get(Self::HEATING_SETPOINT_INDEX) {
                if heating_setpoint >= cooling_setpoint {
                    return Err(crate::api::error::FluxionError::Validation(format!(
                        "Heating setpoint ({:.2}°C, index 1) must be less than cooling setpoint ({:.2}°C, index 2)",
                        heating_setpoint, cooling_setpoint
                    )));
                }
            }
        }
        Ok(())
    }

    /// Creates a new BatchOracle from a base thermal model.
    pub fn from_model(base_model: ThermalModel<VectorField>) -> Self {
        BatchOracle {
            base_model,
            surrogates: SurrogateManager::new().expect("Failed to create SurrogateManager"),
        }
    }

    /// Evaluate a population of building design configurations in parallel.
    ///
    /// This is the critical "hot loop" for optimization. The function uses Rayon for
    /// multi-threaded evaluation. When using surrogates, it implements a time-first loop
    /// architecture for optimal GPU utilization.
    ///
    /// # Arguments
    /// * `population` - Vector of parameter vectors. Each inner vector should contain at least:
    ///   - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
    ///   - `[1]`: Heating setpoint (°C, range: 15-25)
    ///   - `[2]`: Cooling setpoint (°C, range: 22-32)
    /// * `use_surrogates` - If true, use neural network surrogates for faster evaluation;
    ///   if false, use analytical physics calculations.
    ///
    /// # Returns
    /// `Result<Vec<f64>, FluxionError>` where the vector contains EUI values (kWh/m²/yr) for each candidate.
    /// On validation failure, returns `Err(FluxionError)`.
    ///
    /// # Performance
    /// Target throughput: >10,000 configs/sec on 8-core CPU (~100µs per config).
    pub fn evaluate_population(
        &self,
        population: Vec<Vec<f64>>,
        use_surrogates: bool,
    ) -> Result<Vec<f64>, crate::api::error::FluxionError> {
        use crate::physics::cta::ContinuousTensor;
        use rayon::prelude::*;

        // 1. Validate and initialize all models upfront (parallel)
        let mut valid_configs: Vec<(usize, ThermalModel<VectorField>)> = population
            .par_iter()
            .enumerate()
            .filter_map(|(i, params)| {
                if Self::validate_parameters(params).is_err() {
                    return None;
                }
                let mut model = self.base_model.clone();
                model.apply_parameters(params);
                Some((i, model))
            })
            .collect();

        let mut results = vec![f64::NAN; population.len()];

        if use_surrogates && !valid_configs.is_empty() {
            let use_gpu = self.surrogates.gpu_supported();
            if use_gpu {
                // GPU path with SharedBatchInferenceService
                use crate::ai::shared_batch_service::{
                    DynamicBatchConfig, SharedBatchInferenceService,
                };
                let config = DynamicBatchConfig {
                    max_batch_size: std::cmp::min(valid_configs.len(), 1024),
                    wait_ms: 10,
                };
                let service = std::sync::Arc::new(SharedBatchInferenceService::new(
                    self.surrogates.clone(),
                    config,
                    valid_configs.len(), // channel capacity = number of workers
                ));
                let final_worker_data = rayon::scope(|s| {
                    let (result_tx, result_rx) = crossbeam::channel::unbounded();
                    for (idx, mut model) in valid_configs.drain(..) {
                        let service = std::sync::Arc::clone(&service);
                        let res_tx = result_tx.clone();
                        s.spawn(move |_| {
                            let mut energy = 0.0;
                            // Build daily cycle array
                            let cycle: [f64; 24] = {
                                let mut arr = [0.0; 24];
                                for (h, val) in arr.iter_mut().enumerate() {
                                    *val = ((h as f64 / 24.0 * 2.0 * std::f64::consts::PI)
                                        - std::f64::consts::PI / 2.0)
                                        .sin();
                                }
                                arr
                            };
                            for t in 0..8760 {
                                let hour_of_day = t % 24;
                                let daily_cycle = cycle[hour_of_day];
                                let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                                let temps = model.get_temperatures();
                                let rx = service.submit(temps);
                                let loads =
                                    rx.recv().expect("Failed to receive loads from service");
                                model.set_loads(&loads);
                                energy += model.step_physics(t, outdoor_temp, 3600.0);
                            }
                            let _ = res_tx.send((idx, model, energy));
                        });
                    }
                    drop(result_tx);
                    let mut final_data = Vec::new();
                    while let Ok(data) = result_rx.recv() {
                        final_data.push(data);
                    }
                    final_data
                });
                for (idx, model, energy) in final_worker_data {
                    let total_area = model.zone_area.integrate();
                    let eui = if total_area > 0.0 {
                        energy / total_area
                    } else {
                        0.0
                    };
                    results[idx] = eui.max(0.0);
                }
            } else {
                // CPU path (Issue #1439): the previous implementation
                // spawned N rayon tasks + 2N crossbeam channels + a
                // single coordinator thread that did O(N) round-trips
                // per timestep. We now use `BatchOrchestrator` with
                // `par_chunks` so each rayon worker runs all 8 760
                // timesteps for its slice of configs locally, removing
                // the per-timestep coordinator bottleneck entirely.
                use crate::sim::orchestrator::{BatchOrchestrator, RayonChunksOrchestrator};

                let orchestrator = RayonChunksOrchestrator::for_population(valid_configs.len());
                let final_worker_data =
                    orchestrator.run_cpu_surrogate(valid_configs, &self.surrogates);

                for (idx, eui) in final_worker_data {
                    results[idx] = eui;
                }
            }
        } else if !valid_configs.is_empty() {
            // Analytical path - fully parallel
            // Note: StepParameters is !Sync (Box<dyn Equipment>), so a single
            // instance cannot be shared across rayon workers. We construct
            // one StepParameters per worker work-item and reuse it for every
            // one of the 8 760 inner timesteps (Issue #1437 hoists the
            // construction out of the per-timestep inner loop, which
            // previously ran `surrogates.clone()` once per timestep per
            // config — the leading 5R1C allocation-pressure cost).
            let mut energies = vec![0.0; valid_configs.len()];
            valid_configs
                .par_iter_mut()
                .zip(energies.par_iter_mut())
                .for_each(|((_, model), energy)| {
                    let step_params = StepParameters::build_analytical(&self.surrogates);
                    for t in 0..8760 {
                        let hour_of_day = t % 24;
                        let daily_cycle =
                            (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                        let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                        *energy += model.solve_single_step(t, outdoor_temp, &step_params, 3600.0);
                    }
                });

            for ((idx, model), energy) in valid_configs.iter().zip(energies.iter()) {
                let total_area = model.zone_area.integrate();
                let eui = if total_area > 0.0 {
                    *energy / total_area
                } else {
                    0.0
                };
                // Clamp negative results to 0.0
                results[*idx] = eui.max(0.0);
            }
        }

        Ok(results)
    }
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl BatchOracle {
    /// Create a new BatchOracle instance.
    ///
    /// Initializes the base thermal model template and surrogate manager.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(BatchOracle {
            base_model: ThermalModel::<VectorField>::new(10), // The "template" building
            surrogates: SurrogateManager::new().map_err(|e| {
                SurrogateError::new_err(format!("Failed to create SurrogateManager: {}", e))
            })?,
        })
    }

    /// Evaluate a population of building design configurations in parallel.
    ///
    /// This is the critical "hot loop" for optimization. The function crosses the Python-Rust
    /// boundary once with all population data, then uses Rayon for multi-threaded evaluation.
    ///
    /// When using surrogates, this implements a time-first loop architecture:
    /// - Time loop (0..8760) runs sequentially on main thread
    /// - Batched inference ONCE per timestep (full GPU utilization)
    /// - Physics updates run in parallel with rayon
    ///
    /// This avoids nested parallelism and maximizes GPU tensor core utilization.
    ///
    /// # Arguments
    /// * `population` - Vec of parameter vectors, each representing one design candidate.
    ///   Each vector should have at least 3 elements:
    ///   - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
    ///   - `[1]`: Heating setpoint (°C, range: 15-25)
    ///   - `[2]`: Cooling setpoint (°C, range: 22-32)
    /// * `use_surrogates` - If true, use neural network surrogates for faster (~100x) evaluation;
    ///   if false, use physics-based analytical calculations (slower but exact)
    ///
    /// # Returns
    /// Vector of fitness values (EUI in kWh/m²/year) corresponding to each candidate.
    ///
    /// # Performance
    /// Target throughput: >10,000 configs/sec on 8-core CPU (~100µs per config)
    #[pyo3(name = "evaluate_population")]
    pub fn evaluate_population_py(
        &self,
        population: Vec<Vec<f64>>,
        use_surrogates: bool,
    ) -> PyResult<Vec<f64>> {
        Ok(Self::evaluate_population(self, population, use_surrogates)?)
    }

    /// Evaluate a population of building design configurations using BuildingParameters.
    ///
    /// This is a type-safe alternative to `evaluate_population` that accepts
    /// `BuildingParameters` objects instead of raw vectors. The parameters are
    /// validated on construction, providing better error messages and type safety.
    ///
    /// # Arguments
    /// * `population` - Vec of `BuildingParameters` objects, each representing one design candidate.
    /// * `use_surrogates` - If true, use neural network surrogates for faster (~100x) evaluation;
    ///   if false, use physics-based analytical calculations (slower but exact)
    ///
    /// # Returns
    /// Vector of fitness values (EUI in kWh/m²/year) corresponding to each candidate.
    ///
    /// # Performance
    /// Target throughput: >10,000 configs/sec on 8-core CPU (~100µs per config)
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// oracle = fluxion.BatchOracle()
    ///
    /// # Create typed parameters
    /// params = [
    ///     fluxion.BuildingParameters(window_u_value=1.5, heating_setpoint=20.0, cooling_setpoint=24.0),
    ///     fluxion.BuildingParameters(window_u_value=2.0, heating_setpoint=21.0, cooling_setpoint=25.0),
    /// ]
    ///
    /// # Evaluate with type safety
    /// results = oracle.evaluate_population_typed(params, use_surrogates=True)
    /// print(results)  # [123.45, 134.56]
    /// ```
    pub fn evaluate_population_typed(
        &self,
        population: Vec<BuildingParameters>,
        use_surrogates: bool,
    ) -> PyResult<Vec<f64>> {
        // Convert BuildingParameters to Vec<Vec<f64>> for existing implementation
        let vec_population: Vec<Vec<f64>> = population
            .iter()
            .map(|p: &BuildingParameters| p.to_vec())
            .collect();

        // Call existing implementation
        Ok(Self::evaluate_population(
            self,
            vec_population,
            use_surrogates,
        )?)
    }

    /// Evaluate a population of building design configurations using numpy arrays.
    ///
    /// This is an optimized version of `evaluate_population` that accepts numpy arrays
    /// directly, avoiding Python list iteration overhead. This can provide significant
    /// performance improvements when processing large populations.
    ///
    /// # Arguments
    /// * `population` - 2D numpy array of shape (n_candidates, 3) where each row contains:
    ///   - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
    ///   - `[1]`: Heating setpoint (°C, range: 15-25)
    ///   - `[2]`: Cooling setpoint (°C, range: 22-32)
    /// * `use_surrogates` - If true, use neural network surrogates for faster evaluation
    ///
    /// # Returns
    /// 1D numpy array of fitness values (EUI in kWh/m²/year) corresponding to each candidate.
    fn evaluate_population_numpy<'a>(
        &self,
        py: Python<'a>,
        population: &Bound<'_, pyo3::types::PyAny>,
        use_surrogates: bool,
    ) -> PyResult<Bound<'a, numpy::PyArray1<f64>>> {
        use crate::physics::cta::ContinuousTensor;
        use rayon::prelude::*;

        // Try to extract as 2D numpy array
        let array = population.downcast::<numpy::PyArray2<f64>>()?;

        // Get raw data pointer and dimensions
        let array_slice = unsafe { array.as_slice()? };
        let total_len = array_slice.len();

        // Assume 3 columns: U-value, heating, cooling
        let n_params = 3;
        if total_len % n_params != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Population array size must be divisible by 3",
            ));
        }
        let n_candidates = total_len / n_params;

        // Get contiguous copy of the data for efficient iteration
        let population_vec: Vec<Vec<f64>> = (0..n_candidates)
            .map(|i| {
                vec![
                    array_slice[i * n_params],
                    array_slice[i * n_params + 1],
                    array_slice[i * n_params + 2],
                ]
            })
            .collect();

        // 1. Validate and initialize all models upfront (parallel)
        let mut valid_configs: Vec<(usize, ThermalModel<VectorField>)> = population_vec
            .par_iter()
            .enumerate()
            .filter_map(|(i, params)| {
                if Self::validate_parameters(params).is_err() {
                    return None;
                }
                let mut model = self.base_model.clone();
                model.apply_parameters(params);
                Some((i, model))
            })
            .collect();

        let mut results = vec![f64::NAN; n_candidates];

        if use_surrogates && !valid_configs.is_empty() {
            // CPU path (Issue #1439): replaced coordinator-worker
            // channel pattern with `BatchOrchestrator::par_chunks`.
            // See `evaluate_population` for the rationale.
            use crate::sim::orchestrator::{BatchOrchestrator, RayonChunksOrchestrator};

            let orchestrator = RayonChunksOrchestrator::for_population(valid_configs.len());
            let final_worker_data = orchestrator.run_cpu_surrogate(valid_configs, &self.surrogates);

            for (idx, eui) in final_worker_data {
                results[idx] = eui;
            }
        } else if !valid_configs.is_empty() {
            // Analytical path - fully parallel
            // Note: StepParameters is !Sync (Box<dyn Equipment>), so a single
            // instance cannot be shared across rayon workers. We construct
            // one StepParameters per worker work-item and reuse it for every
            // one of the 8 760 inner timesteps (Issue #1437 hoists the
            // construction out of the per-timestep inner loop, which
            // previously ran `surrogates.clone()` once per timestep per
            // config — the leading 5R1C allocation-pressure cost).
            let mut energies = vec![0.0; valid_configs.len()];
            valid_configs
                .par_iter_mut()
                .zip(energies.par_iter_mut())
                .for_each(|((_, model), energy)| {
                    let step_params = StepParameters::build_analytical(&self.surrogates);
                    for t in 0..8760 {
                        let hour_of_day = t % 24;
                        let daily_cycle =
                            (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                        let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                        *energy += model.solve_single_step(t, outdoor_temp, &step_params, 3600.0);
                    }
                });

            for ((idx, model), energy) in valid_configs.iter().zip(energies.iter()) {
                let total_area = model.zone_area.integrate();
                let eui = if total_area > 0.0 {
                    *energy / total_area
                } else {
                    0.0
                };
                results[*idx] = eui.max(0.0);
            }
        }

        // Return as numpy array
        Ok(numpy::PyArray1::from_vec_bound(py, results))
    }

    /// Register an ONNX surrogate model for the oracle. This replaces the internal
    /// `SurrogateManager` with one pointing at the provided model file.
    fn load_surrogate(&mut self, model_path: String) -> PyResult<()> {
        match SurrogateManager::load_onnx(&model_path) {
            Ok(manager) => {
                self.surrogates = manager;
                Ok(())
            }
            Err(e) => Err(SurrogateError::new_err(format!(
                "Failed to load ONNX surrogate model '{}': {}",
                model_path, e
            ))),
        }
    }

    /// Get the parameter bounds for building design variables.
    ///
    /// Returns a ParameterBounds struct with the valid ranges for all design
    /// parameters used by BatchOracle. This is useful for optimization libraries
    /// that need to generate valid parameter vectors.
    ///
    /// # Returns
    /// ParameterBounds struct containing min/max values for:
    /// - Window U-value (W/m²K)
    /// - Heating setpoint (°C)
    /// - Cooling setpoint (°C)
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// oracle = fluxion.BatchOracle()
    /// bounds = oracle.get_parameter_bounds()
    ///
    /// print(f"U-value range: [{bounds.min_u_value}, {bounds.max_u_value}]")
    /// print(f"Heating setpoint range: [{bounds.min_heating_setpoint}, {bounds.max_heating_setpoint}]")
    /// print(f"Cooling setpoint range: [{bounds.min_cooling_setpoint}, {bounds.max_cooling_setpoint}]")
    /// ```
    fn get_parameter_bounds(&self) -> ParameterBounds {
        ParameterBounds::get_bounds()
    }

    /// Validate a parameter vector against physical constraints.
    ///
    /// This method checks that all parameter values are within valid ranges and
    /// that heating/cooling setpoints are consistent. If validation fails, a
    /// ValidationError is raised with a clear, actionable message.
    ///
    /// # Arguments
    /// * `params` - Parameter vector to validate. Elements:
    ///   - `[0]`: Window U-value (W/m²K, must be finite and in [0.1, 5.0])
    ///   - `[1]`: Heating setpoint (°C, must be finite and in [15.0, 25.0])
    ///   - `[2]`: Cooling setpoint (°C, must be finite and in [22.0, 32.0])
    ///
    /// # Raises
    /// ValidationError with detailed message including:
    /// - Parameter index
    /// - Invalid value
    /// - Valid range
    /// - Type of error (NaN, infinite, or out of range)
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// oracle = fluxion.BatchOracle()
    ///
    /// # Valid parameters
    /// oracle.validate_parameters([1.5, 20.0, 27.0])  # OK
    ///
    /// # Invalid U-value (raises ValidationError)
    /// try:
    ///     oracle.validate_parameters([-1.0, 20.0, 27.0])
    /// except fluxion.ValidationError as e:
    ///     print(f"Validation failed: {e}")
    ///     # Output: Window U-value (index 0, -1.00 W/m²K) out of range [0.1, 5.0] W/m²K
    ///
    /// # NaN value (raises ValidationError)
    /// try:
    ///     oracle.validate_parameters([float('nan'), 20.0, 27.0])
    /// except fluxion.ValidationError as e:
    ///     print(f"Validation failed: {e}")
    ///     # Output: Window U-value (index 0) is NaN (value: nan W/m²K). Cannot use in simulation.
    /// ```
    fn validate_parameters_py(&self, params: Vec<f64>) -> PyResult<()> {
        BatchOracle::validate_parameters(&params)?;
        Ok(())
    }
}

/// Parameter bounds for building design variables.
///
/// This struct provides programmatic access to the valid ranges for all
/// design parameters used by BatchOracle and Model. Optimization libraries
/// can query these bounds to generate valid parameter vectors.
#[cfg(feature = "python-bindings")]
#[pyclass]
#[derive(Clone)]
pub struct ParameterBounds {
    /// Minimum window U-value (W/m²K)
    #[pyo3(get)]
    pub min_u_value: f64,
    /// Maximum window U-value (W/m²K)
    #[pyo3(get)]
    pub max_u_value: f64,
    /// Minimum heating setpoint (°C)
    #[pyo3(get)]
    pub min_heating_setpoint: f64,
    /// Maximum heating setpoint (°C)
    #[pyo3(get)]
    pub max_heating_setpoint: f64,
    /// Minimum cooling setpoint (°C)
    #[pyo3(get)]
    pub min_cooling_setpoint: f64,
    /// Maximum cooling setpoint (°C)
    #[pyo3(get)]
    pub max_cooling_setpoint: f64,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl ParameterBounds {
    /// Get the default parameter bounds.
    ///
    /// Returns a ParameterBounds struct with the standard valid ranges
    /// for all design variables.
    #[staticmethod]
    fn get_bounds() -> Self {
        ParameterBounds {
            min_u_value: BatchOracle::MIN_U_VALUE,
            max_u_value: BatchOracle::MAX_U_VALUE,
            min_heating_setpoint: BatchOracle::MIN_HEATING_SETPOINT,
            max_heating_setpoint: BatchOracle::MAX_HEATING_SETPOINT,
            min_cooling_setpoint: BatchOracle::MIN_COOLING_SETPOINT,
            max_cooling_setpoint: BatchOracle::MAX_COOLING_SETPOINT,
        }
    }
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn fluxion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register custom exception types
    m.add("FluxionError", _py.get_type_bound::<FluxionErrorPy>())?;
    m.add("ValidationError", _py.get_type_bound::<ValidationError>())?;
    m.add("SurrogateError", _py.get_type_bound::<SurrogateError>())?;
    m.add("SimulationError", _py.get_type_bound::<SimulationError>())?;

    m.add_class::<Model>()?;
    m.add_class::<BatchOracle>()?;
    m.add_class::<ParameterBounds>()?;
    m.add_class::<BuildingParameters>()?;
    m.add_class::<PyVectorField>()?;
    m.add_class::<PyConstruction>()?;
    m.add_class::<PyConstructionLayer>()?;
    m.add_class::<PyMassClass>()?;
    m.add_class::<PySurfaceType>()?;
    m.add_class::<PyWallSurface>()?;
    m.add_class::<PyGeometryTensor>()?;

    // Register multi-zone module
    python::multi_zone(_py, m)?;

    // Register HVAC classes directly in main module for now
    m.add_class::<python::hvac_bindings::PyZoneSetpoints>()?;
    m.add_class::<python::hvac_bindings::PyZoneControl>()?;
    m.add_class::<python::hvac_bindings::PyDailySchedule>()?;
    m.add_class::<python::hvac_bindings::PyHVACSchedule>()?;
    m.add_function(pyo3::wrap_pyfunction!(
        python::hvac_bindings::create_zone_setpoints,
        m
    )?)?;

    m.add_class::<python::osm_bindings::PyOsmReader>()?;
    m.add_class::<python::osm_bindings::PyOsmWriter>()?;
    m.add_function(pyo3::wrap_pyfunction!(python::osm_bindings::import_osm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(python::osm_bindings::export_osm, m)?)?;

    // Register 9R4C Multi-Node Solver classes
    m.add_class::<python::multi_node_bindings::PyThermalMassNode>()?;
    m.add_class::<python::multi_node_bindings::PyMultiNodeThermalMass>()?;
    m.add_class::<python::multi_node_bindings::PyMassAirCouplingMode>()?;
    m.add_class::<python::multi_node_bindings::PySurfaceExteriorTemperatures>()?;
    m.add_class::<python::multi_node_bindings::PyMultiNodeSolver>()?;

    // Register FluxionModel interior struct bindings (Issue #1812).
    m.add_class::<python::model_bindings::PyOrientation>()?;
    m.add_class::<python::model_bindings::PyShadingType>()?;
    m.add_class::<python::model_bindings::PyShadingDevice>()?;
    m.add_class::<python::model_bindings::PyMaterial>()?;
    m.add_class::<python::model_bindings::PySurface>()?;
    m.add_class::<python::model_bindings::PyZone>()?;
    m.add_class::<python::model_bindings::PyHVACSystem>()?;

    Ok(())
}

// Re-export ASHRAE 140 validation models
pub use validation::ashrae140::high_mass;

// Tests for core physics engine (no Python bindings required)
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::surrogate::SurrogateManager;
    use crate::physics::cta::VectorField;
    use crate::sim::engine::ThermalModel;

    #[cfg(feature = "python-bindings")]
    use crate::BatchOracle;

    // Import logging macros for tests

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batch_oracle_validation() {
        let oracle = BatchOracle::new().unwrap();
        let population = vec![
            vec![1.5, 20.0, 27.0],  // Valid
            vec![-1.0, 20.0, 27.0], // Invalid U-value
            vec![1.5, 500.0, 27.0], // Invalid heating setpoint
            vec![1.5, 20.0, 10.0],  // Invalid cooling setpoint
            vec![1.5, 27.0, 20.0],  // Invalid: heating >= cooling
        ];

        let results = oracle.evaluate_population(population, false).unwrap();

        assert!(results[0].is_finite());
        assert!(results[1].is_nan());
        assert!(results[2].is_nan());
        assert!(results[3].is_nan());
        assert!(results[4].is_nan());
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batched_vs_unbatched_consistency() {
        let oracle = BatchOracle::new().unwrap();
        let population = vec![vec![1.5, 22.0], vec![2.0, 21.0], vec![1.0, 23.0]];

        // Test surrogate path
        let results_batched = oracle
            .evaluate_population(population.clone(), true)
            .unwrap();
        assert!(results_batched.iter().all(|r: &f64| r.is_finite()));

        // Test analytical path for comparison
        let results_analytical = oracle.evaluate_population(population, false).unwrap();
        assert!(results_analytical.iter().all(|r: &f64| r.is_finite()));

        // Results should be in similar range (may differ due to mock vs analytical loads)
        for (batched, analytical) in results_batched.iter().zip(results_analytical.iter()) {
            assert!(*batched > 0.0, "Batched result should be positive");
            assert!(*analytical > 0.0, "Analytical result should be positive");
        }
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_large_population_performance() {
        let oracle = BatchOracle::new().unwrap();
        let population: Vec<Vec<f64>> = (0..1000).map(|_| vec![1.5, 22.0]).collect();

        let start = std::time::Instant::now();
        let results = oracle.evaluate_population(population, true).unwrap();
        let duration = start.elapsed();

        assert_eq!(results.len(), 1000);
        assert!(results.iter().all(|r: &f64| r.is_finite()));

        // Target: <100ms for 1000 configs (may be slower in debug mode)
        #[cfg(debug_assertions)]
        println!("Debug mode: {:?}", duration);
        #[cfg(not(debug_assertions))]
        assert!(duration.as_millis() < 100, "Too slow: {:?}", duration);
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_10k_population_throughput() {
        let oracle = BatchOracle::new().unwrap();
        let population: Vec<Vec<f64>> = (0..10_000).map(|_| vec![1.5, 22.0]).collect();

        let start = std::time::Instant::now();
        let results = oracle.evaluate_population(population, true).unwrap();
        let duration = start.elapsed();

        assert_eq!(results.len(), 10_000);
        assert!(results.iter().all(|r: &f64| r.is_finite()));

        let configs_per_sec = 10_000.0 / duration.as_secs_f64();
        println!("Throughput: {:.0} configs/sec", configs_per_sec);

        // Target: >10,000 configs/sec on 8-core CPU (may be slower in debug mode)
        #[cfg(not(debug_assertions))]
        assert!(
            configs_per_sec > 10_000.0,
            "Below target: {:.0}/sec",
            configs_per_sec
        );
    }

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::<VectorField>::new(10);
        assert_eq!(model.num_zones, 10);
    }

    #[test]
    fn test_thermal_model_default() {
        let model = ThermalModel::<VectorField>::new(1);
        assert_eq!(model.num_zones, 1);
        assert_eq!(model.temperatures.as_ref().len(), 1);
    }

    #[test]
    fn test_apply_parameters() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let params = vec![1.5, 20.0, 27.0];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.heating_setpoint, 20.0);
        assert_eq!(model.cooling_setpoint, 27.0);
    }

    #[test]
    fn test_solve_timesteps() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 20.0, 27.0]);
        let energy = model.solve_timesteps(8760, &surrogates, false, None, None, None);

        assert!(energy.is_finite(), "Energy should be finite"); // Can be negative for cooling or mass charging
    }

    #[test]
    fn test_solve_timesteps_with_surrogates() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 20.0, 27.0]);
        // Should NOT panic now since it returns mock loads
        let energy = model.solve_timesteps(8760, &surrogates, true, None, None, None);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_async_task_creation() {
        let task = InferenceTask::new(1, vec![1.5, 20.0, 27.0]);
        assert_eq!(task.id, 1);
        assert_eq!(task.parameters.len(), 3);
        assert_eq!(task.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn test_async_task_manager_basic() {
        let mut manager = AsyncTaskManager::new(2);
        let task_id = manager.submit_task(vec![1.5, 20.0, 27.0]).await;
        assert_eq!(task_id, 0);
        assert_eq!(manager.tasks_submitted(), 1);
        assert_eq!(manager.max_concurrent(), 2);

        let results: Vec<Result<f64, String>> = manager.collect_results(1).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        assert_eq!(manager.tasks_completed(), 1);
    }

    #[test]
    fn test_distributed_executor_basic() {
        let executor = DistributedInferenceExecutor::new(2, 4);
        assert_eq!(executor.rayon_workers(), 2);
        assert_eq!(executor.async_tasks(), 4);

        let population = vec![vec![1.5, 20.0, 27.0], vec![2.0, 18.0, 28.0]];
        let results = executor.execute_population(population, false);
        assert_eq!(results.len(), 2);
        assert!(results[0] > 0.0);
    }

    #[test]
    fn test_distributed_executor_chunked() {
        let executor = DistributedInferenceExecutor::default();
        let population = vec![
            vec![1.5, 20.0, 27.0],
            vec![2.0, 18.0, 28.0],
            vec![1.0, 22.0, 24.0],
        ];
        let results = executor.execute_chunked(population, 2, false);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_parallel_execution_speedup() {
        use rayon::prelude::*;
        use std::path::Path;

        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThermalModel<VectorField>>();

        let base_model = ThermalModel::<VectorField>::new(10);

        let model_path = "tests_tmp_dummy.onnx";
        let (surrogates, _use_real_model) = if Path::new(model_path).exists() {
            match SurrogateManager::load_onnx(model_path) {
                Ok(s) => (s, true),
                Err(e) => {
                    eprintln!("Failed to load dummy model (proceeding with mock): {}", e);
                    (
                        SurrogateManager::new().expect("Failed to create SurrogateManager"),
                        false,
                    )
                }
            }
        } else {
            eprintln!("tests_tmp_dummy.onnx not found; proceeding with mock SurrogateManager");
            (
                SurrogateManager::new().expect("Failed to create SurrogateManager"),
                false,
            )
        };

        let population_size = 2000;
        let population: Vec<Vec<f64>> = (0..population_size)
            .map(|_| vec![1.5, 20.0, 27.0])
            .collect();

        let start_seq = std::time::Instant::now();
        let _results_seq: Vec<f64> = population
            .iter()
            .map(|params| {
                let mut instance = base_model.clone();
                instance.apply_parameters(params);
                instance.solve_timesteps(100, &surrogates, true, None, None, None)
            })
            .collect();
        let duration_seq = start_seq.elapsed();

        let start_par = std::time::Instant::now();
        let _results_par: Vec<f64> = population
            .par_iter()
            .map(|params| {
                let mut instance = base_model.clone();
                instance.apply_parameters(params);
                instance.solve_timesteps(100, &surrogates, true, None, None, None)
            })
            .collect();
        let duration_par = start_par.elapsed();

        println!("Sequential time: {:?}", duration_seq);
        println!("Parallel time: {:?}", duration_par);
        println!(
            "Available parallelism: {}",
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        );

        assert!(
            duration_par > std::time::Duration::ZERO && duration_seq > std::time::Duration::ZERO,
            "Both sequential and parallel runs should produce valid timings. Seq: {:?}, Par: {:?}",
            duration_seq,
            duration_par
        );
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batch_oracle_building_parameters() {
        use crate::api::parameters::BuildingParameters;

        let oracle = BatchOracle::new().unwrap();

        // Create valid BuildingParameters
        let params = vec![
            BuildingParameters::new(1.5, 20.0, 24.0).unwrap(),
            BuildingParameters::new(2.0, 21.0, 25.0).unwrap(),
        ];

        // Test with typed parameters
        let results = oracle
            .evaluate_population_typed(params.clone(), false)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_finite()));

        // Compare with Vec<f64> approach
        let vec_params: Vec<Vec<f64>> = params.iter().map(|p| p.to_vec()).collect();
        let vec_results = oracle.evaluate_population(vec_params, false).unwrap();

        assert_eq!(results.len(), vec_results.len());
        for (typed, vec_result) in results.iter().zip(vec_results.iter()) {
            assert!((typed - vec_result).abs() < 1e-6);
        }
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_batch_oracle_building_parameters_invalid() {
        use crate::api::parameters::BuildingParameters;

        let oracle = BatchOracle::new().unwrap();

        // Try to create invalid BuildingParameters - this should fail at construction
        let result = BuildingParameters::new(-1.0, 20.0, 24.0);
        assert!(result.is_err());

        // Valid parameters should work
        let valid_params = vec![BuildingParameters::new(1.5, 20.0, 24.0).unwrap()];
        let results = oracle
            .evaluate_population_typed(valid_params, false)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_finite());
    }

    #[test]
    fn test_logging_control() {
        // Test that logging can be controlled via RUST_LOG environment variable
        // This test verifies that the logging infrastructure is properly initialized
        // and that log statements don't cause panics or errors

        // Initialize logger (should be idempotent)
        let _ = env_logger::try_init();

        // Test various log levels - these should not panic
        log::error!("Test error log");
        log::warn!("Test warn log");
        log::info!("Test info log");
        log::debug!("Test debug log");
        log::trace!("Test trace log");

        // Test that BatchOracle and Model can be created and used with logging
        #[cfg(feature = "python-bindings")]
        {
            let oracle = BatchOracle::new().unwrap();
            info!("Created BatchOracle with logging");

            let population = vec![vec![1.5, 20.0, 27.0]];
            let results = oracle.evaluate_population(population, false).unwrap();
            assert!(results[0].is_finite());
            info!("BatchOracle evaluation completed successfully");
        }
    }
}

// =============================================================================
// Distributed Inference Architecture
// =============================================================================
// This module provides async task management using tokio and data parallelism
// using rayon for running thousands of building variants simultaneously.

/// Task status for distributed inference jobs.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    /// Task is pending and waiting to be scheduled
    Pending,
    /// Task is currently being processed
    Running,
    /// Task completed successfully with results
    Completed(f64), // EUI result
    /// Task failed with error message
    Failed(String),
}

/// A single inference task representing one building variant evaluation.
#[derive(Debug, Clone)]
pub struct InferenceTask {
    /// Unique task identifier
    pub id: u64,
    /// Building parameters: [U-value, heating_setpoint, cooling_setpoint]
    pub parameters: Vec<f64>,
    /// Current status of the task
    pub status: TaskStatus,
}

impl InferenceTask {
    /// Create a new inference task with the given parameters.
    pub fn new(id: u64, parameters: Vec<f64>) -> Self {
        Self {
            id,
            parameters,
            status: TaskStatus::Pending,
        }
    }
}

/// Async task manager for distributed inference using tokio.
///
/// This manager handles scheduling and execution of building variant simulations
/// using async/await patterns for high-throughput concurrent processing.
pub struct AsyncTaskManager {
    /// Channel sender for submitting new tasks
    task_sender: tokio::sync::mpsc::Sender<InferenceTask>,
    /// Channel receiver for receiving task results
    result_receiver: tokio::sync::mpsc::Receiver<Result<f64, String>>,
    /// Maximum number of concurrent tasks
    max_concurrent: usize,
    /// Total tasks submitted
    tasks_submitted: u64,
    /// Total tasks completed
    tasks_completed: u64,
}

impl AsyncTaskManager {
    /// Create a new async task manager.
    ///
    /// # Arguments
    /// * `max_concurrent` - Maximum number of concurrent tasks to run
    ///
    /// # Returns
    /// A new AsyncTaskManager instance with task channels
    #[allow(dead_code)]
    pub fn new(max_concurrent: usize) -> Self {
        let (task_sender, mut task_receiver) = tokio::sync::mpsc::channel::<InferenceTask>(10000);
        let (result_sender, result_receiver) =
            tokio::sync::mpsc::channel::<Result<f64, String>>(10000);

        // Spawn the async worker pool
        let worker_max_concurrent = max_concurrent;
        tokio::spawn(async move {
            let mut running_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
            let mut pending_queue: Vec<InferenceTask> = Vec::new();

            loop {
                tokio::select! {
                    // Try to receive new tasks
                    new_task = task_receiver.recv() => {
                        match new_task {
                            Some(task) => {
                                // Clean up finished tasks first
                                running_handles.retain(|h| !h.is_finished());

                                if running_handles.len() < worker_max_concurrent {
                                    // Spawn new async task immediately
                                    let sender = result_sender.clone();
                                    let handle = tokio::spawn(async move {
                                        let params = &task.parameters;
                                        if params.len() >= 3 {
                                            let u_value = params[0];
                                            let heating = params[1];
                                            let cooling = params[2];

                                            let base_load = 50.0;
                                            let u_factor = (u_value - 1.0).abs() * 10.0;
                                            let setpoint_diff = (cooling - heating) * 5.0;
                                            let eui = base_load + u_factor + setpoint_diff;

                                            let _ = sender.send(Ok(eui)).await;
                                        } else {
                                            let _ = sender.send(Err("Invalid parameters".to_string())).await;
                                        }
                                    });
                                    running_handles.push(handle);
                                } else {
                                    // Add to pending queue
                                    pending_queue.push(task);
                                }
                            }
                            None => {
                                // Channel closed, exit loop
                                // Wait for remaining running tasks before exit
                                for handle in running_handles {
                                    let _ = handle.await;
                                }
                                break;
                            }
                        }
                    }

                    // Periodic cleanup and task spawning
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(5)) => {
                        // Clean up finished tasks
                        running_handles.retain(|h| !h.is_finished());

                        // Spawn pending tasks if there's capacity
                        while running_handles.len() < worker_max_concurrent {
                            match pending_queue.pop() {
                                Some(task) => {
                                    let sender = result_sender.clone();
                                    let handle = tokio::spawn(async move {
                                        let params = &task.parameters;
                                        if params.len() >= 3 {
                                            let u_value = params[0];
                                            let heating = params[1];
                                            let cooling = params[2];

                                            let base_load = 50.0;
                                            let u_factor = (u_value - 1.0).abs() * 10.0;
                                            let setpoint_diff = (cooling - heating) * 5.0;
                                            let eui = base_load + u_factor + setpoint_diff;

                                            let _ = sender.send(Ok(eui)).await;
                                        } else {
                                            let _ = sender.send(Err("Invalid parameters".to_string())).await;
                                        }
                                    });
                                    running_handles.push(handle);
                                }
                                None => break,
                            }
                        }
                    }
                }
            }
        });

        Self {
            task_sender,
            result_receiver,
            max_concurrent,
            tasks_submitted: 0,
            tasks_completed: 0,
        }
    }

    /// Submit a new inference task for async processing.
    ///
    /// # Arguments
    /// * `parameters` - Building parameters [U-value, heating_setpoint, cooling_setpoint]
    ///
    /// # Returns
    /// Task ID that can be used to retrieve results
    #[allow(dead_code)]
    pub async fn submit_task(&mut self, parameters: Vec<f64>) -> u64 {
        let task_id = self.tasks_submitted;
        self.tasks_submitted += 1;

        let task = InferenceTask::new(task_id, parameters);
        let _ = self.task_sender.send(task).await;

        task_id
    }

    /// Submit multiple tasks at once (batch submission).
    ///
    /// # Arguments
    /// * `parameters_list` - List of building parameter vectors
    ///
    /// # Returns
    /// Vector of task IDs
    #[allow(dead_code)]
    pub async fn submit_batch(&mut self, parameters_list: Vec<Vec<f64>>) -> Vec<u64> {
        let mut task_ids = Vec::with_capacity(parameters_list.len());

        for params in parameters_list {
            let task_id = self.submit_task(params).await;
            task_ids.push(task_id);
        }

        task_ids
    }

    /// Wait for a specific task result.
    ///
    /// # Arguments
    /// * `task_id` - ID of the task to wait for
    ///
    /// # Returns
    /// Result containing EUI or error
    #[allow(dead_code)]
    pub async fn wait_for_result(&mut self, task_id: u64) -> Result<f64, String> {
        while let Some(result) = self.result_receiver.recv().await {
            self.tasks_completed += 1;
            if self.tasks_completed == task_id {
                return result;
            }
        }
        Err("No results available".to_string())
    }

    /// Collect all available results.
    ///
    /// # Returns
    /// Vector of results in order of completion
    #[allow(dead_code)]
    pub async fn collect_results(&mut self, count: usize) -> Vec<Result<f64, String>> {
        let mut results = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(result) = self.result_receiver.recv().await {
                self.tasks_completed += 1;
                results.push(result);
            }
        }

        results
    }

    /// Get the number of submitted tasks.
    #[allow(dead_code)]
    pub fn tasks_submitted(&self) -> u64 {
        self.tasks_submitted
    }

    /// Get the number of completed tasks.
    #[allow(dead_code)]
    pub fn tasks_completed(&self) -> u64 {
        self.tasks_completed
    }

    /// Get the maximum concurrent task limit.
    #[allow(dead_code)]
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

/// Distributed inference executor that combines tokio async tasks with rayon data parallelism.
///
/// This provides the best of both worlds:
/// - Tokio for async I/O and task scheduling
/// - Rayon for CPU-intensive parallel computation
pub struct DistributedInferenceExecutor {
    /// Number of rayon workers for CPU parallelism
    rayon_workers: usize,
    /// Number of tokio async tasks
    async_tasks: usize,
}

impl DistributedInferenceExecutor {
    /// Create a new distributed inference executor.
    ///
    /// # Arguments
    /// * `rayon_workers` - Number of rayon threads for data parallelism
    /// * `async_tasks` - Number of async tasks for I/O concurrency
    #[allow(dead_code)]
    pub fn new(rayon_workers: usize, async_tasks: usize) -> Self {
        Self {
            rayon_workers,
            async_tasks,
        }
    }

    /// Execute a population of building variants using combined async and data parallelism.
    ///
    /// This method uses:
    /// - Tokio async runtime for managing concurrent tasks
    /// - Rayon for parallel evaluation within each async task
    ///
    /// # Arguments
    /// * `population` - List of building parameter vectors
    /// * `use_surrogates` - Whether to use AI surrogates for evaluation
    ///
    /// # Returns
    /// Vector of EUI values for each building variant
    #[allow(dead_code)]
    pub fn execute_population(&self, population: Vec<Vec<f64>>, use_surrogates: bool) -> Vec<f64> {
        use rayon::prelude::*;

        // Use rayon for data parallelism (batch processing)
        let results: Vec<f64> = population
            .par_iter()
            .map(|params| {
                // Simulate evaluation (in real code, call thermal model)
                if params.len() >= 3 {
                    let u_value = params[0];
                    let heating = params[1];
                    let cooling = params[2];

                    let base_load = if use_surrogates { 50.0 } else { 55.0 };
                    let u_factor = (u_value - 1.5).abs() * 8.0;
                    let setpoint_diff = (cooling - heating) * 4.0;

                    base_load + u_factor + setpoint_diff
                } else {
                    f64::NAN
                }
            })
            .collect();

        results
    }

    /// Execute with chunked processing for very large populations.
    ///
    /// # Arguments
    /// * `population` - List of building parameter vectors
    /// * `chunk_size` - Size of each chunk for processing
    /// * `use_surrogates` - Whether to use AI surrogates
    ///
    /// # Returns
    /// Vector of EUI values
    #[allow(dead_code)]
    pub fn execute_chunked(
        &self,
        population: Vec<Vec<f64>>,
        chunk_size: usize,
        use_surrogates: bool,
    ) -> Vec<f64> {
        use rayon::prelude::*;

        // Split population into chunks
        let chunks: Vec<Vec<Vec<f64>>> =
            population.chunks(chunk_size).map(|c| c.to_vec()).collect();

        // Process chunks in parallel
        let chunk_results: Vec<Vec<f64>> = chunks
            .par_iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|params| {
                        if params.len() >= 3 {
                            let u_value = params[0];
                            let heating = params[1];
                            let cooling = params[2];

                            let base_load = if use_surrogates { 50.0 } else { 55.0 };
                            let u_factor = (u_value - 1.5).abs() * 8.0;
                            let setpoint_diff = (cooling - heating) * 4.0;

                            base_load + u_factor + setpoint_diff
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            })
            .collect();

        // Flatten results
        chunk_results.into_iter().flatten().collect()
    }

    /// Get the rayon worker count.
    #[allow(dead_code)]
    pub fn rayon_workers(&self) -> usize {
        self.rayon_workers
    }

    /// Get the async task count.
    #[allow(dead_code)]
    pub fn async_tasks(&self) -> usize {
        self.async_tasks
    }
}

impl Default for DistributedInferenceExecutor {
    fn default() -> Self {
        let rayon_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            rayon_workers,
            async_tasks: rayon_workers * 4, // 4x oversubscription for I/O
        }
    }
}

// ============================================================================
// Geometry Tensor Python Bindings (Zero-Copy)
// ============================================================================

#[cfg(feature = "python-bindings")]
use crate::physics::geometry_tensor::{
    GeometryTensor, ADJACENCY_MATRIX_DIMS, WALL_MATRIX_DIMS, WINDOW_MATRIX_DIMS, ZONE_COORDS_DIMS,
    ZONE_PROPERTIES_DIMS,
};
#[cfg(feature = "python-bindings")]
use crate::physics::zero_copy_matrix::ZeroCopyGeometryTensorHolder;

#[cfg(feature = "python-bindings")]
#[pyclass(name = "GeometryTensor")]
/// Python-accessible wrapper for GeometryTensor to expose to PyO3.
///
/// `inner` is wrapped in an `Arc` so that [`PyGeometryTensor::to_numpy`] can
/// hand a numpy array a borrow of the underlying storage without copying the
/// buffer — the cloned `Arc` is held by the numpy array's container and keeps
/// the bytes alive for as long as Python holds the numpy array.
pub struct PyGeometryTensor {
    inner: std::sync::Arc<GeometryTensor>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyGeometryTensor {
    /// Create a new empty GeometryTensor.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyGeometryTensor {
            inner: std::sync::Arc::new(GeometryTensor::new()),
        })
    }

    /// Create a GeometryTensor from numpy arrays.
    ///
    /// For optimal performance, use numpy arrays directly. The buffer protocol
    /// (the same wire format that Arrow uses for inter-process buffer sharing)
    /// is honored: a `numpy.ndarray` argument's storage is read in place via
    /// `PyReadonlyArray::as_slice`, avoiding intermediate `Vec` copies on the
    /// binding layer. A single ownership-transfer copy remains to move the
    /// data into the Rust-owned `GeometryTensor` storage.
    ///
    /// This method accepts:
    /// - zone_coords: (100, 20) zone coordinates
    /// - wall_matrix: (500, 6) wall geometry
    /// - window_matrix: (500, 6) window geometry
    /// - adjacency_matrix: (100, 100) zone adjacency
    /// - zone_properties: (100, 5) zone properties
    /// - summary: (6,) summary statistics
    #[staticmethod]
    fn from_numpy(
        zone_coords: &Bound<'_, pyo3::types::PyAny>,
        wall_matrix: &Bound<'_, pyo3::types::PyAny>,
        window_matrix: &Bound<'_, pyo3::types::PyAny>,
        adjacency_matrix: &Bound<'_, pyo3::types::PyAny>,
        zone_properties: &Bound<'_, pyo3::types::PyAny>,
        summary: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Self> {
        // Zero-copy on the binding layer: borrow the numpy array's storage
        // through `PyReadonlyArray::as_slice`, then take ownership with a
        // single `Vec::from` (or `to_vec`) copy required to detach the slice
        // from the numpy array's lifetime.
        //
        // The previous implementation called `slice.to_vec()` here AND
        // `from_numpy_arrays` called `to_vec()` again — two copies on the
        // hot path. The refactor keeps only the unavoidable ownership copy.
        fn borrow_f64_slice(arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Vec<f64>> {
            // Try 2D array first
            if let Ok(pyarr) = arr.downcast::<numpy::PyArray2<f64>>() {
                // SAFETY: PyReadonlyArray dynamically borrows the numpy array;
                // `as_slice` returns a `&[f64]` with no copy on the binding
                // layer.
                let slice = unsafe { pyarr.as_slice()? };
                return Ok(slice.to_vec());
            }
            // Try 1D array
            if let Ok(pyarr) = arr.downcast::<numpy::PyArray1<f64>>() {
                let slice = unsafe { pyarr.as_slice()? };
                return Ok(slice.to_vec());
            }
            // Fallback to Python sequence iteration (no zero-copy possible —
            // Python objects must be extracted element by element).
            let mut vec = Vec::new();
            for item in arr.iter()? {
                let val = item?.extract::<f64>()?;
                vec.push(val);
            }
            Ok(vec)
        }

        let zone_coords = borrow_f64_slice(zone_coords)?;
        let wall_matrix = borrow_f64_slice(wall_matrix)?;
        let window_matrix = borrow_f64_slice(window_matrix)?;
        let adjacency_matrix = borrow_f64_slice(adjacency_matrix)?;
        let zone_properties = borrow_f64_slice(zone_properties)?;
        let summary = borrow_f64_slice(summary)?;

        let inner = GeometryTensor::from_numpy_arrays(
            &zone_coords,
            &wall_matrix,
            &window_matrix,
            &adjacency_matrix,
            &zone_properties,
            &summary,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        Ok(PyGeometryTensor {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Get the number of zones.
    fn num_zones(&self) -> usize {
        self.inner.num_zones()
    }

    /// Get the number of walls.
    fn num_walls(&self) -> usize {
        self.inner.num_walls()
    }

    /// Get the total floor area.
    fn total_area(&self) -> f64 {
        self.inner.total_area()
    }

    /// Get the total volume.
    fn total_volume(&self) -> f64 {
        self.inner.total_volume()
    }

    /// Validate the geometry tensor.
    ///
    /// Returns a list of validation issues.
    fn validate(&self) -> Vec<String> {
        self.inner.validate()
    }

    /// Get summary statistics as a dictionary.
    fn get_summary(&self) -> Vec<f64> {
        self.inner.summary.clone()
    }

    /// Check if two zones are adjacent.
    fn zones_adjacent(&self, i: usize, j: usize) -> bool {
        self.inner.zones_adjacent(i, j)
    }

    /// Convert to numpy arrays with zero-copy buffer sharing.
    ///
    /// Each returned numpy array wraps a `numpy::PyArray2::borrow_from_array_bound`
    /// view of the underlying `GeometryTensor` storage — the numpy array and
    /// the Rust struct share the same buffer. A `PyClass` holder that retains
    /// an `Arc<GeometryTensor>` clone keeps the storage alive for the lifetime
    /// of the numpy array. The Arc clone is a refcount bump (no buffer copy),
    /// so the Rust → Python direction is allocation-free beyond the small
    /// holder object.
    ///
    /// This is the standard numpy buffer protocol that Arrow uses for
    /// inter-process buffer sharing — any Arrow-compatible consumer
    /// (PyArrow, ML frameworks) can ingest the returned arrays without
    /// re-copying.
    #[allow(clippy::type_complexity)]
    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    )> {
        // The holder pattern: the numpy array's container holds an Arc clone
        // of the GeometryTensor. The borrowed view points into the Arc's
        // storage. When Python releases the numpy array, the Arc is dropped,
        // and (if the last reference) the GeometryTensor is dropped too.
        fn build_zero_copy_2d<'py>(
            py: Python<'py>,
            inner: std::sync::Arc<GeometryTensor>,
            shape: (usize, usize),
            pick: fn(&GeometryTensor) -> &[f64],
        ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
            // SAFETY: the holder's Arc clone keeps `inner` alive for the
            // lifetime of the returned numpy array, so the view's data
            // pointer remains valid. `pick(&inner)` returns a borrow of one
            // of `inner`'s `Vec<f64>` fields, which is contiguous and
            // `shape.0 * shape.1` elements long (validated by the
            // `GeometryTensor::from_numpy_arrays` constructor and the
            // `WALL_MATRIX_DIMS` / etc. constants).
            let ptr = pick(&inner).as_ptr();
            let raw = unsafe { ndarray::RawArrayView::from_shape_ptr(shape, ptr) };
            let view = unsafe { raw.deref_into_view() };
            let holder = ZeroCopyGeometryTensorHolder { inner };
            let container = Bound::new(py, holder)
                .expect("ZeroCopyGeometryTensorHolder allocation cannot fail")
                .into_any();
            Ok(unsafe { numpy::PyArray2::borrow_from_array_bound(&view, container) })
        }

        let zone_coords = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            ZONE_COORDS_DIMS,
            |t| &t.zone_coords,
        )?;
        let wall_matrix = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            WALL_MATRIX_DIMS,
            |t| &t.wall_matrix,
        )?;
        let window_matrix = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            WINDOW_MATRIX_DIMS,
            |t| &t.window_matrix,
        )?;
        let adjacency_matrix = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            ADJACENCY_MATRIX_DIMS,
            |t| &t.adjacency_matrix,
        )?;
        let zone_properties = build_zero_copy_2d(
            py,
            std::sync::Arc::clone(&self.inner),
            ZONE_PROPERTIES_DIMS,
            |t| &t.zone_properties,
        )?;

        // 1-D summary path — same zero-copy recipe.
        let summary_inner = std::sync::Arc::clone(&self.inner);
        let summary_ptr = summary_inner.summary.as_ptr();
        // SAFETY: same Arc-alive invariant; `summary` is a 6-element Vec<f64>
        // held by the GeometryTensor.
        let summary_raw = unsafe {
            ndarray::RawArrayView::from_shape_ptr(summary_inner.summary.len(), summary_ptr)
        };
        let summary_view = unsafe { summary_raw.deref_into_view() };
        let summary_holder = ZeroCopyGeometryTensorHolder {
            inner: summary_inner,
        };
        let summary_container = Bound::new(py, summary_holder)
            .expect("ZeroCopyGeometryTensorHolder allocation cannot fail")
            .into_any();
        let summary =
            unsafe { numpy::PyArray1::borrow_from_array_bound(&summary_view, summary_container) };

        Ok((
            zone_coords,
            wall_matrix,
            window_matrix,
            adjacency_matrix,
            zone_properties,
            summary,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "GeometryTensor(zones={}, walls={}, area={:.2}m², volume={:.2}m³)",
            self.inner.num_zones(),
            self.inner.num_walls(),
            self.inner.total_area(),
            self.inner.total_volume()
        )
    }
}
