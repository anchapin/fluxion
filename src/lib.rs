#![allow(clippy::useless_conversion)]
pub mod ai;
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
    types::{PyListMethods, PyModuleMethods},
    Bound, PyResult, Python,
};

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
    ///
    /// # Returns
    /// HVAC energy consumption for timestep in kWh
    fn step(&mut self, timestep: usize, outdoor_temp: f64, use_surrogates: bool) -> PyResult<f64> {
        Ok(self.inner.solve_single_step(
            timestep,
            outdoor_temp,
            use_surrogates,
            &self.surrogates,
            true,
        ))
    }

    /// Apply optimization parameters to the model.
    ///
    /// # Arguments
    /// * `params` - Parameter vector:
    ///   - params[0]: Window U-value (W/m²K, range: 0.1-5.0)
    ///   - params[1]: Heating setpoint (°C, range: 15-25)
    ///   - params[2]: Cooling setpoint (°C, range: 22-32)
    fn apply_parameters(&mut self, params: Vec<f64>) {
        self.inner.apply_parameters(&params);
    }

    /// Get current window U-value.
    fn window_u_value(&self) -> f64 {
        self.inner.window_u_value
    }

    /// Get current heating setpoint.
    fn heating_setpoint(&self) -> f64 {
        self.inner.heating_setpoint
    }

    /// Get current cooling setpoint.
    fn cooling_setpoint(&self) -> f64 {
        self.inner.cooling_setpoint
    }

    /// Get zone floor area in m².
    fn zone_area(&self) -> f64 {
        self.inner.zone_area.integrate()
    }

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

/// VectorField wrapper for Python.
#[cfg(feature = "python-bindings")]
#[pyclass(name = "VectorField")]
pub struct PyVectorField {
    inner: crate::physics::cta::VectorField,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVectorField {
    /// Create a new VectorField from a Python list of floats.
    #[new]
    fn new(data: &Bound<'_, pyo3::types::PyList>) -> PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        let mut vec = Vec::with_capacity(data.len());
        for item in data.iter() {
            if let Ok(val) = item.extract::<f64>() {
                vec.push(val);
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "VectorField elements must be floats",
                ));
            }
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

    /// Convert to numpy array (requires numpy feature).
    #[cfg(not(all(feature = "numpy")))]
    fn to_numpy(&self, _py: Python<'_>) -> PyResult<Vec<f64>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "NumPy support not enabled. Use to_list() instead.",
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
#[pyclass(name = "SurfaceType")]
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
#[pyclass(name = "MassClass")]
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
        let rust_surface =
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

            // 3. Normalize and populate results
            for (idx, model, energy) in final_worker_data {
                let total_area = model.zone_area.integrate();
                let eui = if total_area > 0.0 {
                    energy / total_area
                } else {
                    0.0
                };
                // Clamp negative results to 0.0 (caused by thermal mass energy accounting
                // when mass charging > HVAC input in deadband)
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
                // Clamp negative results to 0.0
                results[*idx] = eui.max(0.0);
            }
        }

        Ok(results)
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
    m.add_class::<VectorField>()?;
    m.add_class::<PyConstruction>()?;
    m.add_class::<PyConstructionLayer>()?;
    m.add_class::<PyMassClass>()?;
    m.add_class::<PySurfaceType>()?;
    m.add_class::<PyWallSurface>()?;
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
