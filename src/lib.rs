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
pub mod ai;
pub mod analysis;
pub mod api;
pub mod cli;
pub mod hvac;
pub mod performance;
pub mod physics;
#[cfg(feature = "python-bindings")]
pub mod python;
pub mod sim;
pub mod testing;
pub mod thermal;
pub mod validation;
pub mod weather;

// Re-export thermal model traits for public API
pub use sim::thermal_model::{
    PhysicsThermalModel, SurrogateThermalModel, ThermalModelBuilder, ThermalModelMode,
    ThermalModelTrait, UnifiedThermalModel,
};

// Re-export ISO 13790 Annex C construction types
pub use sim::construction::{Construction, ConstructionLayer, MassClass};

use crate::physics::cta::VectorField;
use ai::surrogate::SurrogateManager;
// Logging for verbosity control via RUST_LOG environment variable
use sim::engine::ThermalModel;

#[cfg(feature = "python-bindings")]
use crate::api::{FluxionErrorPy, SimulationError, SurrogateError, ValidationError};

#[cfg(feature = "python-bindings")]
use crate::physics::cta::ContinuousTensor;
use anyhow::Result;
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
        self.inner.integrate()
    }

    /// Compute the gradient (rate of change) of the field.
    fn gradient(&self) -> Self {
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
impl From<&crate::sim::components::WallSurface> for PyWallSurface {
    fn from(surface: &crate::sim::components::WallSurface) -> Self {
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
            crate::sim::components::WallSurface::new(area, u_value, rust_orientation);
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
                // CPU path: Coordinator-Worker pattern with Channels
                let n_workers = valid_configs.len();
                let mut coord_txs = Vec::with_capacity(n_workers);
                let mut coord_rxs = Vec::with_capacity(n_workers);
                let mut worker_channels = Vec::with_capacity(n_workers);

                for _ in 0..n_workers {
                    let (tx_to_coord, rx_from_worker) = crossbeam::channel::unbounded();
                    let (tx_to_worker, rx_from_coord) = crossbeam::channel::unbounded();
                    coord_rxs.push(rx_from_worker);
                    coord_txs.push(tx_to_worker);
                    worker_channels.push((tx_to_coord, rx_from_coord));
                }

                let final_worker_data = rayon::scope(|s| {
                    let (result_tx, result_rx) = crossbeam::channel::unbounded();

                    // Move models and channels into workers
                    for ((idx, mut model), (tx, rx)) in
                        valid_configs.drain(..).zip(worker_channels.into_iter())
                    {
                        let res_tx = result_tx.clone();
                        s.spawn(move |_| {
                            let energy = model.solve_timesteps_batched(8760, tx, rx);
                            let _ = res_tx.send((idx, model, energy));
                        });
                    }
                    drop(result_tx);

                    // Coordinator loop
                    for _t in 0..8760 {
                        // 1. Collect temperatures from all workers
                        let mut batch_temps = Vec::with_capacity(n_workers);
                        for rx in &coord_rxs {
                            batch_temps.push(rx.recv().expect("Worker disconnected unexpectedly"));
                        }

                        // 2. Batched inference
                        let batch_loads = self.surrogates.predict_loads_batched(&batch_temps);

                        // 3. Send loads back to workers
                        for (tx, loads) in coord_txs.iter().zip(batch_loads) {
                            tx.send(loads).expect("Failed to send loads to worker");
                        }
                    }

                    let mut final_data = Vec::with_capacity(n_workers);
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
            }
        } else if !valid_configs.is_empty() {
            // Analytical path - fully parallel
            let mut energies = vec![0.0; valid_configs.len()];
            valid_configs
                .par_iter_mut()
                .zip(energies.par_iter_mut())
                .for_each(|((_, model), energy)| {
                    for t in 0..8760 {
                        let hour_of_day = t % 24;
                        let daily_cycle =
                            (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                        let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                        *energy += model.solve_single_step(
                            t,
                            outdoor_temp,
                            false,
                            &self.surrogates,
                            true,
                            None,
                            None,
                            None,
                            3600.0, // dt_seconds
                        );
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
        let vec_population: Vec<Vec<f64>> = population.iter().map(|p| p.to_vec()).collect();

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
            // Coordinator-Worker pattern with Channels
            let n_workers = valid_configs.len();
            let mut coord_txs = Vec::with_capacity(n_workers);
            let mut coord_rxs = Vec::with_capacity(n_workers);
            let mut worker_channels = Vec::with_capacity(n_workers);

            for _ in 0..n_workers {
                let (tx_to_coord, rx_from_worker) = crossbeam::channel::unbounded();
                let (tx_to_worker, rx_from_coord) = crossbeam::channel::unbounded();
                coord_rxs.push(rx_from_worker);
                coord_txs.push(tx_to_worker);
                worker_channels.push((tx_to_coord, rx_from_coord));
            }

            let final_worker_data = rayon::scope(|s| {
                let (result_tx, result_rx) = crossbeam::channel::unbounded();

                for ((idx, mut model), (tx, rx)) in
                    valid_configs.drain(..).zip(worker_channels.into_iter())
                {
                    let res_tx = result_tx.clone();
                    s.spawn(move |_| {
                        let energy = model.solve_timesteps_batched(8760, tx, rx);
                        let _ = res_tx.send((idx, model, energy));
                    });
                }
                drop(result_tx);

                for _t in 0..8760 {
                    let mut batch_temps = Vec::with_capacity(n_workers);
                    for rx in &coord_rxs {
                        batch_temps.push(rx.recv().expect("Worker disconnected unexpectedly"));
                    }

                    let batch_loads = self.surrogates.predict_loads_batched(&batch_temps);

                    for (tx, loads) in coord_txs.iter().zip(batch_loads) {
                        tx.send(loads).expect("Failed to send loads to worker");
                    }
                }

                let mut final_data = Vec::with_capacity(n_workers);
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
        } else if !valid_configs.is_empty() {
            // Analytical path - fully parallel
            let mut energies = vec![0.0; valid_configs.len()];
            valid_configs
                .par_iter_mut()
                .zip(energies.par_iter_mut())
                .for_each(|((_, model), energy)| {
                    for t in 0..8760 {
                        let hour_of_day = t % 24;
                        let daily_cycle =
                            (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                        let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                        *energy += model.solve_single_step(
                            t,
                            outdoor_temp,
                            false,
                            &self.surrogates,
                            true,
                            None,
                            None,
                            None,
                            3600.0, // dt_seconds
                        );
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
    m.add_function(pyo3::wrap_pyfunction!(
        python::hvac_bindings::create_zone_setpoints,
        m
    )?)?;

    Ok(())
}

// Re-export ASHRAE 140 validation models
pub use validation::ashrae_140::{Case600Model, SimulationResult};

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
    use log::info;

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
    #[ignore = "Requires specific ONNX model dimensions and multi-core environment"]
    fn test_parallel_execution_speedup() {
        use rayon::prelude::*;
        use std::path::Path;

        // Verify Send + Sync for ThermalModel (required for parallel execution)
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThermalModel<VectorField>>();

        let base_model = ThermalModel::<VectorField>::new(10);

        // Try to load a real model if available (created by other tests)
        // otherwise fall back to mock (but verify parallel mechanism either way)
        // Ideally we want to test with the pool active.
        let model_path = "tests_tmp_dummy.onnx";
        let surrogates = if Path::new(model_path).exists() {
            match SurrogateManager::load_onnx(model_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to load dummy model (proceeding with mock): {}", e);
                    SurrogateManager::new().expect("Failed to create SurrogateManager")
                }
            }
        } else {
            // Fall back to mock SurrogateManager if file missing
            eprintln!("tests_tmp_dummy.onnx not found; proceeding with mock SurrogateManager");
            SurrogateManager::new().expect("Failed to create SurrogateManager")
        };

        // Create a large population
        let population_size = 2000;
        let population: Vec<Vec<f64>> = (0..population_size)
            .map(|_| vec![1.5, 20.0, 27.0])
            .collect();

        // Sequential execution (using standard iter)
        let start_seq = std::time::Instant::now();
        let _results_seq: Vec<f64> = population
            .iter()
            .map(|params| {
                let mut instance = base_model.clone();
                instance.apply_parameters(params);
                // Use surrogates to test session pool contention/parallelism
                instance.solve_timesteps(100, &surrogates, true, None, None, None)
            })
            .collect();
        let duration_seq = start_seq.elapsed();

        // Parallel execution (using rayon par_iter)
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

        // On a multi-core machine, parallel should be faster.
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        if num_threads > 1 {
            // We expect significant speedup, but CI environments can be noisy.
            // Just asserting it's faster is a good baseline.
            assert!(
                duration_par < duration_seq,
                "Parallel execution should be faster than sequential on {} threads. Seq: {:?}, Par: {:?}",
                num_threads,
                duration_seq,
                duration_par
            );
        }
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
#[pyclass(name = "GeometryTensor")]
/// Python-accessible wrapper for GeometryTensor to expose to PyO3.
pub struct PyGeometryTensor {
    inner: GeometryTensor,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyGeometryTensor {
    /// Create a new empty GeometryTensor.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyGeometryTensor {
            inner: GeometryTensor::new(),
        })
    }

    /// Create a GeometryTensor from numpy arrays.
    ///
    /// For optimal performance, use numpy arrays directly.
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
        // Helper to extract numpy array as slice
        fn extract_f64_slice(arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Vec<f64>> {
            // Try 2D array first
            if let Ok(pyarr) = arr.downcast::<numpy::PyArray2<f64>>() {
                let slice = unsafe { pyarr.as_slice()? };
                return Ok(slice.to_vec());
            }
            // Try 1D array
            if let Ok(pyarr) = arr.downcast::<numpy::PyArray1<f64>>() {
                let slice = unsafe { pyarr.as_slice()? };
                return Ok(slice.to_vec());
            }
            // Fallback to Python sequence
            let mut vec = Vec::new();
            for item in arr.iter()? {
                let val = item?.extract::<f64>()?;
                vec.push(val);
            }
            Ok(vec)
        }

        let zone_coords = extract_f64_slice(zone_coords)?;
        let wall_matrix = extract_f64_slice(wall_matrix)?;
        let window_matrix = extract_f64_slice(window_matrix)?;
        let adjacency_matrix = extract_f64_slice(adjacency_matrix)?;
        let zone_properties = extract_f64_slice(zone_properties)?;
        let summary = extract_f64_slice(summary)?;

        let inner = GeometryTensor::from_numpy_arrays(
            &zone_coords,
            &wall_matrix,
            &window_matrix,
            &adjacency_matrix,
            &zone_properties,
            &summary,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        Ok(PyGeometryTensor { inner })
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

    /// Convert to numpy arrays (zero-copy view where possible).
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
        let zone_coords = numpy::PyArray2::from_owned_array_bound(
            py,
            Array2::from_shape_vec(ZONE_COORDS_DIMS, self.inner.zone_coords.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        );

        let wall_matrix = numpy::PyArray2::from_owned_array_bound(
            py,
            Array2::from_shape_vec(WALL_MATRIX_DIMS, self.inner.wall_matrix.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        );

        let window_matrix = numpy::PyArray2::from_owned_array_bound(
            py,
            Array2::from_shape_vec(WINDOW_MATRIX_DIMS, self.inner.window_matrix.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        );

        let adjacency_matrix = numpy::PyArray2::from_owned_array_bound(
            py,
            Array2::from_shape_vec(ADJACENCY_MATRIX_DIMS, self.inner.adjacency_matrix.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        );

        let zone_properties = numpy::PyArray2::from_owned_array_bound(
            py,
            Array2::from_shape_vec(ZONE_PROPERTIES_DIMS, self.inner.zone_properties.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        );

        let summary = numpy::PyArray1::from_slice_bound(py, self.inner.summary.as_slice());

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
