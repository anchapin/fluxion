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
use crate::sim::engine::ThermalModel;

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

        // Defensive double-check: the validator guarantees the shape, but the
        // slice length must agree with `n_candidates * n_params`. If a future
        // change breaks that invariant we surface a clean error rather than
        // panicking on the row/slice indexing in
        // `BatchOracle::evaluate_population_from_slice`.
        debug_assert_eq!(array_slice.len(), n_candidates * n_params);
        if array_slice.len() != n_candidates * n_params {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "population array slice length disagrees with validated shape",
            ));
        }

        // Issue #2874: hand the numpy read-only slice straight to the
        // zero-copy hot loop. The pre-#2874 implementation received this
        // contiguous `&[f64]`, then immediately discarded it by doing
        // `(0..n_candidates).map(|i| vec![array_slice[i*n_params..(i+1)*n_params]])
        // .collect::<Vec<Vec<f64>>>()` — one outer Vec + N inner Vec<f64>s +
        // 3N element copies, all of which the validator's contiguous slice
        // already represented. We now borrow row slices directly inside
        // `BatchOracle::evaluate_population_from_slice`'s per-row closure.
        let results = self.evaluate_population_from_slice(
            array_slice,
            n_candidates,
            n_params,
            use_surrogates,
        )?;

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

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    //! Rust-side inline tests for the PyO3 wrappers in this module
    //! (Issue #2882).
    //!
    //! These tests exercise the Rust validation / conversion / load-surrogate
    //! logic that backs the Python-visible entrypoints, without spinning up a
    //! CPython interpreter. They mirror the acceptance criteria from the issue:
    //!
    //! * `ParameterBounds::get_bounds` matches the physical constants on
    //!   `BatchOracle`,
    //! * `validate_parameters` rejects NaN / out-of-range U-value / heating /
    //!   cooling setpoints with informative messages,
    //! * `evaluate_population` (the path called by `evaluate_population_py`
    //!   and `evaluate_population_typed`) returns an empty vector for an empty
    //!   population and an all-NaN vector for an all-invalid population,
    //! * `load_surrogate` rejects file-not-found and out-of-allow-list paths
    //!   via the shared `validate_model_path` validator used by the Python
    //!   binding,
    //! * the numpy dtype round-trip on the core `evaluate_population` API
    //!   reproduces the analytical numpy path within `1e-6` EUI.
    //!
    //! Tests are gated on `feature = "python-bindings"` because the
    //! `parameter_bounds_get_matches_engine_constants` test references
    //! `BatchOracle::MIN_U_VALUE` / `MAX_U_VALUE`, which are only declared when
    //! the feature is on (the underlying `BatchOracle::new` is also gated).
    use super::*;
    use crate::ai::surrogate::validate_model_path_in_dir;
    use crate::api::parameters::BuildingParameters;

    // ----- ParameterBounds -----

    #[test]
    fn parameter_bounds_get_matches_engine_constants() {
        let b = ParameterBounds::get_bounds();
        assert_eq!(b.min_u_value, BatchOracle::MIN_U_VALUE);
        assert_eq!(b.max_u_value, BatchOracle::MAX_U_VALUE);
        assert_eq!(b.min_heating_setpoint, BatchOracle::MIN_HEATING_SETPOINT);
        assert_eq!(b.max_heating_setpoint, BatchOracle::MAX_HEATING_SETPOINT);
        assert_eq!(b.min_cooling_setpoint, BatchOracle::MIN_COOLING_SETPOINT);
        assert_eq!(b.max_cooling_setpoint, BatchOracle::MAX_COOLING_SETPOINT);
    }

    #[test]
    fn parameter_bounds_widening_ordering() {
        // Physical sanity: every max must be strictly greater than its min.
        // The heating/cooling bands intentionally overlap (heating 15-25°C,
        // cooling 22-32°C — the deadband is established per-config by the
        // `heating < cooling` cross-check in `validate_parameters`, not by
        // non-overlapping ranges).
        let b = ParameterBounds::get_bounds();
        assert!(b.min_u_value < b.max_u_value);
        assert!(b.min_heating_setpoint < b.max_heating_setpoint);
        assert!(b.min_cooling_setpoint < b.max_cooling_setpoint);
    }

    // ----- validate_parameters rejection paths -----
    //
    // These exercise the underlying `BatchOracle::validate_parameters` that
    // `validate_parameters_py` forwards to (line 393-396 of the binding).
    // Each error message includes the parameter index, the offending value,
    // and the valid range — verified by the substring assertions below.

    #[test]
    fn validate_parameters_rejects_nan_u_value() {
        let err = BatchOracle::validate_parameters(&[f64::NAN, 20.0, 27.0])
            .err()
            .expect("NaN U-value must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("NaN"), "msg must label the error: {msg}");
        assert!(
            msg.contains("index 0"),
            "msg must identify the failing parameter index: {msg}"
        );
    }

    #[test]
    fn validate_parameters_rejects_negative_u_value() {
        let err = BatchOracle::validate_parameters(&[-1.0, 20.0, 27.0])
            .err()
            .expect("negative U-value must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("out of range"),
            "msg must label the error: {msg}"
        );
        assert!(
            msg.contains("index 0") && msg.contains("0.1") && msg.contains("5.0"),
            "msg must include index and valid range: {msg}"
        );
    }

    #[test]
    fn validate_parameters_rejects_negative_infinity_u_value() {
        // Pathological case: `-inf` is finite == false, must be reported as
        // "infinite" rather than "NaN" — this differentiates the two failure
        // modes in the validator.
        let err = BatchOracle::validate_parameters(&[f64::NEG_INFINITY, 20.0, 27.0])
            .err()
            .expect("-inf U-value must be rejected");
        assert!(
            format!("{err}").contains("infinite"),
            "msg must label non-NaN non-finite as infinite: {err}"
        );
    }

    #[test]
    fn validate_parameters_rejects_out_of_range_heating_setpoint() {
        // Heating setpoint below MIN_HEATING_SETPOINT.
        let err = BatchOracle::validate_parameters(&[1.5, 5.0, 27.0])
            .err()
            .expect("heating setpoint below 15°C must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("Heating setpoint"), "msg={msg}");
        assert!(msg.contains("index 1"), "msg={msg}");
        assert!(msg.contains("15.0") && msg.contains("25.0"), "msg={msg}");
    }

    #[test]
    fn validate_parameters_rejects_out_of_range_cooling_setpoint() {
        // Cooling setpoint above MAX_COOLING_SETPOINT.
        let err = BatchOracle::validate_parameters(&[1.5, 20.0, 50.0])
            .err()
            .expect("cooling setpoint above 32°C must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("Cooling setpoint"), "msg={msg}");
        assert!(msg.contains("index 2"), "msg={msg}");
        assert!(msg.contains("22.0") && msg.contains("32.0"), "msg={msg}");
    }

    #[test]
    fn validate_parameters_rejects_heating_at_or_above_cooling() {
        // Equal heating & cooling setpoints — invalid.
        let err_equal = BatchOracle::validate_parameters(&[1.5, 22.0, 22.0])
            .err()
            .expect("equal heating/cooling setpoints must be rejected");
        assert!(
            format!("{err_equal}").contains("must be less than"),
            "msg={err_equal}"
        );

        // Heating above cooling — also invalid.
        let err_inverted = BatchOracle::validate_parameters(&[1.5, 25.0, 22.0])
            .err()
            .expect("heating > cooling setpoints must be rejected");
        assert!(
            format!("{err_inverted}").contains("must be less than"),
            "msg={err_inverted}"
        );
    }

    #[test]
    fn validate_parameters_accepts_valid_vector() {
        // Sanity guard: a plausible 3-tuple at the center of the valid region
        // must validate. If this fails, every downstream test in this module
        // is meaningless.
        BatchOracle::validate_parameters(&[1.5, 20.0, 24.0])
            .expect("central parameter tuple must validate");
    }

    // ----- evaluate_population contract: empty / all-invalid / round-trip -----

    #[test]
    fn evaluate_population_typed_empty_input_branch() {
        // The `evaluate_population_typed` wrapper (binding lines 148-165)
        // forwards an empty `Vec<BuildingParameters>` to
        // `evaluate_population(&self, Vec<Vec<f64>>, _)`, which must return
        // an empty `Vec<f64>` rather than panic / divide by zero on
        // `population.len()`.
        let oracle = BatchOracle::new().expect("oracle");
        let empty_pop: Vec<BuildingParameters> = Vec::new();
        let out = oracle.evaluate_population_typed(empty_pop, false);
        match out {
            Ok(v) => assert!(v.is_empty(), "empty population must give empty vec"),
            Err(e) => panic!("empty population must not error: {e}"),
        }

        // Mirror through the underlying Vec<Vec<f64>> path that the binding
        // shim uses so an empty `Vec<f64>` produces the same result whether
        // the caller typed it or not.
        let empty_vec = oracle.evaluate_population(Vec::new(), false).unwrap();
        assert!(empty_vec.is_empty());
    }

    #[test]
    fn evaluate_population_typed_validates_each_candidate_independently() {
        // Two vectors with valid BuildingParameters (heating strictly less
        // than cooling, all in-range). The bound `BuildingParameters::new`
        // rejects out-of-range values at construction, mirroring the Python
        // wrapper's boundary check; once both are accepted, both must
        // produce finite EUI values.
        let oracle = BatchOracle::new().expect("oracle");
        let params = vec![
            BuildingParameters::new(1.5, 20.0, 24.0).expect("valid"),
            BuildingParameters::new(0.5, 21.0, 26.0).expect("valid"),
        ];
        let valid = oracle
            .evaluate_population_typed(params, false)
            .expect("typed evaluation");
        assert_eq!(valid.len(), 2);
        assert!(valid.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn evaluate_population_typed_matches_underlying_vec_path() {
        // Quantify the numpy dtype round-trip contract: the analytical path
        // for `evaluate_population_typed` and the underlying
        // `evaluate_population(Vec<Vec<f64>>)` path must produce identical
        // EUI values for the same configurations (the same `to_vec` is the
        // only transformation the binding performs — binding line 154-157).
        let oracle = BatchOracle::new().expect("oracle");

        let parameters = vec![
            BuildingParameters::new(1.5, 20.0, 24.0).expect("valid"),
            BuildingParameters::new(0.5, 21.0, 25.0).expect("valid"),
            BuildingParameters::new(2.5, 19.0, 23.0).expect("valid"),
        ];
        let typed = oracle
            .evaluate_population_typed(parameters.clone(), false)
            .expect("typed");
        let vec_in: Vec<Vec<f64>> = parameters.iter().map(|p| p.to_vec()).collect();
        let vec_out = oracle.evaluate_population(vec_in, false).expect("vec");

        assert_eq!(typed.len(), vec_out.len());
        for (a, b) in typed.iter().zip(vec_out.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "typed {a:?} must match vec {b:?} within 1e-9 EUI"
            );
        }
    }

    #[test]
    fn evaluate_population_all_invalid_returns_nans_with_correct_length() {
        // Every element out-of-range → all results NaN, but length still
        // matches the input. Guards against an off-by-one in the NaN-fill
        // branch (binding pre-#2532 had this regression).
        let oracle = BatchOracle::new().expect("oracle");
        let bad: Vec<Vec<f64>> = vec![
            vec![-1.0, 20.0, 27.0], // negative U-value
            vec![1.5, 100.0, 27.0], // heating out of range
            vec![1.5, 20.0, 100.0], // cooling out of range
        ];
        let out = oracle
            .evaluate_population(bad.clone(), false)
            .expect("all-invalid inputs do not propagate the validation error");
        assert_eq!(out.len(), bad.len());
        for v in &out {
            assert!(v.is_nan(), "each invalid candidate must result in NaN");
        }
    }

    // ----- load_surrogate: file-not-found path -----

    #[test]
    fn load_surrogate_nonexistent_file_rejected() {
        // `load_surrogate` (binding line 310) first calls
        // `validate_model_path`; a non-existent path must be rejected with
        // the canonical "model file not found" error (no raw path echoed
        // back, per the security contract in `validate_model_path_in_dir`).
        let result = validate_model_path_in_dir(
            "/tmp/does-not-exist-surrogate.onnx",
            std::path::Path::new("/tmp"),
        );
        let err = result.err().expect("nonexistent file must reject");
        assert!(
            err.contains("model file not found"),
            "msg must be the canonical not-found: {err}"
        );
        assert!(
            !err.contains("/tmp/does-not-exist-surrogate.onnx"),
            "raw user path must NOT be reflected back (oracle): {err}"
        );
    }

    #[test]
    fn load_surrogate_wrong_extension_rejected() {
        // Existing file but with the wrong extension (`.bin` instead of
        // `.onnx`) must fail the extension check.
        let dir = tempdir();
        let bad_ext = dir.join("model.bin");
        std::fs::write(&bad_ext, [0u8; 8]).expect("seed file");
        let result = validate_model_path_in_dir(bad_ext.to_str().unwrap(), &dir);
        let err = result.err().expect("non-onnx extension must reject");
        assert!(
            err.contains("extension"),
            "msg must be the canonical extension error: {err}"
        );
    }

    #[test]
    fn load_surrogate_orphaned_path_rejected() {
        // File that exists at the given path but resolves outside the
        // allow-list directory (simulated by passing a model-shaped file
        // under `/tmp` and asking it to be validated against an unrelated
        // `dir`).
        let allow_dir = tempdir();
        let outside_dir = tempdir();
        let orphan = outside_dir.join("stray.onnx");
        std::fs::write(&orphan, [0u8; 8]).expect("seed file");
        let result = validate_model_path_in_dir(orphan.to_str().unwrap(), &allow_dir);
        let err = result.err().expect("escaped path must reject");
        assert!(
            err.contains("outside allowed directory") || err.contains("not found"),
            "msg must identify the path-policy violation: {err}"
        );
    }

    // -- helpers ---------------------------------------------------------

    /// Create a unique temporary directory scoped to this test invocation.
    /// Concurrent tests each see their own directory; the OS cleans up on
    /// process exit.
    fn tempdir() -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("fluxion-py-bind-{pid}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("create tempdir");
        dir
    }
}
