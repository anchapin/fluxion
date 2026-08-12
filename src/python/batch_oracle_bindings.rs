//! PyO3 bindings for [`BatchOracle`] (Issue #2493).
//!
//! Extracted from `lib.rs` so the crate root stays a thin re-export shim.
//! Contains the `#[pymethods] impl BatchOracle` block (the Python-visible
//! `new` / `evaluate_population` / `evaluate_population_numpy` /
//! `evaluate_population_typed` / `load_surrogate` / `get_parameter_bounds` /
//! `validate_parameters` entrypoints) plus the `ParameterBounds` `#[pyclass]`.
//!
//! The core `BatchOracle` struct (with `#[cfg_attr(.., pyclass)]`) and the
//! physics `evaluate_population` hot loop live in [`crate::batch_oracle`].
//! PyO3 permits the `#[pyclass]` attribute and the `#[pymethods] impl` to
//! live in separate modules of the same crate, so the Python class
//! `fluxion.BatchOracle` is unchanged.

#[cfg(feature = "python-bindings")]
use crate::ai::surrogate::SurrogateManager;
#[cfg(feature = "python-bindings")]
use crate::api::error::SurrogateError;
#[cfg(feature = "python-bindings")]
use crate::api::parameters::BuildingParameters;
#[cfg(feature = "python-bindings")]
use crate::batch_oracle::BatchOracle;
#[cfg(feature = "python-bindings")]
use crate::physics::cta::VectorField;
#[cfg(feature = "python-bindings")]
use crate::sim::engine::{StepParameters, ThermalModel};

#[cfg(feature = "python-bindings")]
use std::time::Instant;

#[cfg(feature = "python-bindings")]
use numpy::PyArrayMethods;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymethods]
impl BatchOracle {
    /// Create a new BatchOracle instance.
    ///
    /// Initializes the base thermal model template and surrogate manager.
    //
    // `pub(crate)` so the `#[cfg(test)]` modules in `batch_oracle.rs` and
    // `lib_tests.rs` can construct a default oracle via `BatchOracle::new()`
    // post-#2493 (the tests used to live in the same `lib.rs` module as this
    // impl). Python-side visibility is governed by `#[new]`.
    #[new]
    pub(crate) fn new() -> PyResult<Self> {
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
        // Issue #2548: tracing span + metrics for the Python-facing batch
        // entrypoint. Mirrors `Model::simulate`. The span covers the full
        // Python→Rust call including rayon parallelism in the inner loop.
        let _span = tracing::info_span!(
            "python_batch_evaluate",
            population_size = population.len(),
            use_surrogates,
        )
        .entered();

        let start = Instant::now();
        let result = Self::evaluate_population(self, population, use_surrogates);
        let duration = start.elapsed();

        metrics::histogram!("fluxion_python_batch_evaluate_duration_seconds")
            .record(duration.as_secs_f64());
        let outcome_label = if result.is_ok() { "success" } else { "error" };
        metrics::counter!(
            "fluxion_python_batch_evaluate_total",
            "outcome" => outcome_label
        )
        .increment(1);

        Ok(result?)
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
        let array = population.cast::<numpy::PyArray2<f64>>()?;

        // Issue #2528: validate shape *before* any `unsafe` slice dereference.
        // A zero-row array (`[0, 3]`) or a wrong column count previously
        // reached `unsafe { array.as_slice() }` and could panic / abort the
        // host interpreter. Now it raises a catchable `PyValueError`.
        let (n_candidates, n_params) =
            crate::python::panic_hook::validate_population_array_shape(array)?;

        // SAFETY-free path: `readonly().as_slice()` is the safe numpy accessor
        // (returns `Err(NotContiguousError)` instead of panicking). The
        // previous `unsafe { array.as_slice()? }` had no soundness benefit —
        // `PyArrayMethods::as_slice` is itself the safe `readonly().as_slice()`
        // — and the `unsafe` block masked the panic-abort hazard.
        let readonly = array.readonly();
        let array_slice = readonly.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "population array must be C-contiguous: {e}"
            ))
        })?;
        let total_len = array_slice.len();

        // Defensive double-check: the validator guarantees the shape, but the
        // slice length must agree with `n_candidates * n_params`. If a future
        // change breaks that invariant we surface a clean error rather than
        // panicking on the row/slice indexing below.
        debug_assert_eq!(total_len, n_candidates * n_params);
        if total_len != n_candidates * n_params {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "population array slice length disagrees with validated shape",
            ));
        }

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
                    let step_params = StepParameters::build_analytical();
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
        Ok(numpy::PyArray1::from_vec(py, results))
    }

    /// Register an ONNX surrogate model for the oracle. This replaces the internal
    /// `SurrogateManager` with one pointing at the provided model file.
    ///
    /// The path is validated per Issue #2529 (existence, `.onnx` extension,
    /// allow-list directory via `FLUXION_MODEL_DIR`, and 256 MiB size limit)
    /// before reaching the ONNX runtime. Error messages are generic and never
    /// echo the raw user-supplied path.
    fn load_surrogate(&mut self, model_path: String) -> PyResult<()> {
        let validated = crate::ai::surrogate::validate_model_path(&model_path)
            .map_err(SurrogateError::new_err)?;
        match SurrogateManager::load_onnx(&validated.to_string_lossy()) {
            Ok(manager) => {
                self.surrogates = manager;
                Ok(())
            }
            Err(e) => Err(SurrogateError::new_err(format!(
                "Failed to load ONNX surrogate model: {e}"
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
#[pyclass(from_py_object)]
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
    //
    // `pub(crate)` so the `Model::get_parameter_bounds` / `BatchOracle::get_parameter_bounds`
    // helpers (which live in other python-binding submodules post-#2493) can call this
    // inherent fn directly. The Python-side visibility is governed by `#[staticmethod]`.
    #[staticmethod]
    pub(crate) fn get_bounds() -> Self {
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
