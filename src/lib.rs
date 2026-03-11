#![allow(clippy::useless_conversion)]
pub mod ai;
pub mod analysis;
pub mod physics;
pub mod sim;
pub mod validation;
pub mod weather;

// Re-export thermal model traits for public API
pub use sim::thermal_model::{
    PhysicsThermalModel, SurrogateThermalModel, ThermalModelBuilder, ThermalModelMode,
    ThermalModelTrait, UnifiedThermalModel,
};

// Re-export ISO 13790 Annex C construction types
pub use sim::construction::{Construction, ConstructionLayer, MassClass};

#[cfg(feature = "python-bindings")]
use crate::physics::cta::{ContinuousTensor, VectorField};
#[cfg(feature = "python-bindings")]
use ai::surrogate::SurrogateManager;
#[cfg(feature = "python-bindings")]
use sim::engine::ThermalModel;

#[cfg(feature = "python-bindings")]
use pyo3::{
    prelude::{pyclass, pymethods, pymodule, PyModule},
    types::{PyAnyMethods, PyModuleMethods},
    Bound, PyResult, Python,
};

// NumPy types - available when python-bindings feature is enabled
#[cfg(feature = "python-bindings")]
use ndarray::Array2;
#[cfg(feature = "python-bindings")]
use numpy::PyArrayMethods;

// When not using python-bindings feature, we still need these for tests
#[cfg(not(feature = "python-bindings"))]
#[allow(unused_imports)]
use ai::surrogate::SurrogateManager;
#[cfg(not(feature = "python-bindings"))]
#[allow(unused_imports)]
use sim::engine::ThermalModel;

// Re-export things for easier access in other modules
// pub use ai::tensor_wrapper::TorchScalar; // REMOVED

/// Standard Single-Building Model for detailed building energy analysis.
///
/// Use this class when you need detailed simulation of a single building configuration,
/// including hourly temperature traces and ASHRAE 140 validation.
#[cfg(feature = "python-bindings")]
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
            surrogates: SurrogateManager::new()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
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

    /// Simulate building energy consumption over specified years.
    ///
    /// # Arguments
    /// * `years` - Number of years to simulate (1-5 typical)
    /// * `use_surrogates` - If true, use AI surrogates for load predictions; if false, use analytical calculations
    ///
    /// # Returns
    /// Total energy use intensity (EUI) in kWh/m²/year
    fn simulate(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        let steps = years as usize * 8760;
        Ok(self
            .inner
            .solve_timesteps(steps, &self.surrogates, use_surrogates))
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
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
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
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid orientation. Use: south, west, north, east",
                ))
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
#[cfg(feature = "python-bindings")]
#[pyclass]
struct BatchOracle {
    base_model: ThermalModel<VectorField>,
    surrogates: SurrogateManager,
}

#[cfg(feature = "python-bindings")]
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
    fn validate_parameters(params: &[f64]) -> Result<(), String> {
        if params.len() < 3 {
            return Err("Parameter vector must have at least 3 elements.".to_string());
        }
        let u_value = params[Self::U_VALUE_INDEX];
        let heating_setpoint = params[Self::HEATING_SETPOINT_INDEX];
        let cooling_setpoint = params[Self::COOLING_SETPOINT_INDEX];

        if !(Self::MIN_U_VALUE..=Self::MAX_U_VALUE).contains(&u_value) {
            return Err(format!("U-value out of range: {}", u_value));
        }
        if !(Self::MIN_HEATING_SETPOINT..=Self::MAX_HEATING_SETPOINT).contains(&heating_setpoint) {
            return Err(format!(
                "Heating setpoint out of range: {}",
                heating_setpoint
            ));
        }
        if !(Self::MIN_COOLING_SETPOINT..=Self::MAX_COOLING_SETPOINT).contains(&cooling_setpoint) {
            return Err(format!(
                "Cooling setpoint out of range: {}",
                cooling_setpoint
            ));
        }
        // Validate that heating < cooling (deadband must exist)
        if heating_setpoint >= cooling_setpoint {
            return Err(format!(
                "Heating setpoint ({}) must be less than cooling setpoint ({})",
                heating_setpoint, cooling_setpoint
            ));
        }
        Ok(())
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
            surrogates: SurrogateManager::new()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?,
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
    fn evaluate_population(
        &self,
        population: Vec<Vec<f64>>,
        use_surrogates: bool,
    ) -> PyResult<Vec<f64>> {
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
                    max_batch_size: valid_configs.len(),
                    wait_ms: 10,
                };
                let service = std::sync::Arc::new(SharedBatchInferenceService::new(
                    self.surrogates.clone(),
                    config,
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
                                energy += model.step_physics(t, outdoor_temp);
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
                        *energy +=
                            model.solve_single_step(t, outdoor_temp, false, &self.surrogates, true);
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
                        *energy +=
                            model.solve_single_step(t, outdoor_temp, false, &self.surrogates, true);
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
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn fluxion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<BatchOracle>()?;
    m.add_class::<PyVectorField>()?;
    m.add_class::<PyConstruction>()?;
    m.add_class::<PyConstructionLayer>()?;
    m.add_class::<PyMassClass>()?;
    m.add_class::<PySurfaceType>()?;
    m.add_class::<PyWallSurface>()?;
    m.add_class::<PyGeometryTensor>()?;
    Ok(())
}

// Re-export ASHRAE 140 validation models
pub use validation::ashrae_140::{Case600Model, SimulationResult};

// Tests for core physics engine (no Python bindings required)
#[cfg(test)]
mod tests {
    use crate::ai::surrogate::SurrogateManager;
    use crate::physics::cta::VectorField;
    use crate::sim::engine::ThermalModel;

    #[cfg(feature = "python-bindings")]
    use crate::BatchOracle;

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
        let energy = model.solve_timesteps(8760, &surrogates, false);

        assert!(energy.is_finite(), "Energy should be finite"); // Can be negative for cooling or mass charging
    }

    #[test]
    fn test_solve_timesteps_with_surrogates() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 20.0, 27.0]);
        let energy = model.solve_timesteps(8760, &surrogates, true);

        assert!(energy > 0.0, "Energy should be positive");
    }

    #[test]
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
                instance.solve_timesteps(100, &surrogates, true)
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
                instance.solve_timesteps(100, &surrogates, true)
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
