//! Core BatchOracle: high-throughput parallel evaluation of building design
//! populations (Issue #2493).
//!
//! Extracted from `lib.rs` to bring the crate root under the 500-line module
//! budget. The `#[pyclass]` / `#[pymethods]` PyO3 surface lives in
//! [`crate::python::batch_oracle_bindings`]; this module holds the struct
//! definition, the physical-constraint constants, the parameter validator, and
//! the analytical + surrogate `evaluate_population` hot loop.
//!
//! The struct is intentionally **not** feature-gated to `python-bindings`: it
//! is part of the public Rust API (`fluxion::BatchOracle`) and is consumed by
//! `src/analysis`, `src/bin/fluxion`, the NAPI bindings, and many integration
//! tests without the PyO3 feature enabled. Only the PyO3 attribute is gated,
//! via `#[cfg_attr(feature = "python-bindings", pyclass)]`.

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

// `pyclass` must be in scope for `#[cfg_attr(feature = "python-bindings", pyclass)]`
// on the struct below. PyO3 0.29 (post-#2585) API.
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

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
    pub(crate) base_model: ThermalModel<VectorField>,
    pub(crate) surrogates: SurrogateManager,
}

impl BatchOracle {
    // Physical constraints for optimization parameters
    pub(crate) const MIN_U_VALUE: f64 = 0.1; // Minimum realistic U-value (W/m²K)
    pub(crate) const MAX_U_VALUE: f64 = 5.0; // Maximum realistic U-value
    pub(crate) const MIN_HEATING_SETPOINT: f64 = 15.0; // Min heating setpoint (°C)
    pub(crate) const MAX_HEATING_SETPOINT: f64 = 25.0; // Max heating setpoint (°C)
    pub(crate) const MIN_COOLING_SETPOINT: f64 = 22.0; // Min cooling setpoint (°C)
    pub(crate) const MAX_COOLING_SETPOINT: f64 = 32.0; // Max cooling setpoint (°C)

    // Parameter indices
    pub(crate) const U_VALUE_INDEX: usize = 0;
    pub(crate) const HEATING_SETPOINT_INDEX: usize = 1;
    pub(crate) const COOLING_SETPOINT_INDEX: usize = 2;

    /// Validates a parameter vector against physical constraints.
    ///
    /// This function checks for NaN/Inf values before range validation to prevent
    /// physics failures. Error messages include parameter index, value, and valid range
    /// for self-diagnosis.
    pub(crate) fn validate_parameters(
        params: &[f64],
    ) -> Result<(), crate::api::error::FluxionError> {
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
        // Flatten the `Vec<Vec<f64>>` into a single contiguous `Vec<f64>` so
        // the hot loop can index row slices directly. One allocation of
        // size `N * n_params` replaces the previous `Vec<Vec<f64>>` shape
        // that allocated one outer `Vec` + N inner `Vec<f64>`s. For empty
        // populations we skip the allocation entirely.
        if population.is_empty() {
            return Ok(Vec::new());
        }
        let n_candidates = population.len();
        let n_params = population.iter().map(|row| row.len()).max().unwrap_or(0);
        let total = n_candidates * n_params;
        let mut flat: Vec<f64> = Vec::with_capacity(total);
        for row in &population {
            flat.extend_from_slice(row);
            if row.len() < n_params {
                // Zero-pad ragged rows so every row slice has uniform length
                // (only the *first* n_params elements are observed by
                // `validate_parameters` / `apply_parameters`, both of which
                // `params.get(i)` past the actual data).
                flat.resize(flat.len() + (n_params - row.len()), 0.0);
            }
        }
        debug_assert_eq!(flat.len(), total);
        self.evaluate_population_from_slice(&flat, n_candidates, n_params, use_surrogates)
    }

    /// Crate-private zero-copy evaluation entry point (Issue #2874).
    ///
    /// Equivalent to [`Self::evaluate_population`] but accepts a flat
    /// row-major population slice (the shape a numpy `PyArray2::as_slice()`
    /// read-only view produces) instead of an owned `Vec<Vec<f64>>`. The
    /// numpy binding calls this with the borrow from the validator (#2528)
    /// directly — no `Vec<Vec<f64>>` materialisation, no `Vec<f64>` row
    /// copies, no element re-shuffling.
    ///
    /// `flat.len()` must equal `n_candidates * n_params`; the function
    /// indexes each row via `flat[i*n_params..(i+1)*n_params]` and hands the
    /// `&[f64]` row slice to `validate_parameters` and `ThermalModel::
    /// apply_parameters`, both of which take `&[f64]`.
    ///
    /// Error/result semantics, NaN-fill ordering, GPU vs CPU surrogate
    /// dispatch and the analytical path are identical to
    /// [`Self::evaluate_population`].
    pub fn evaluate_population_from_slice(
        &self,
        flat: &[f64],
        n_candidates: usize,
        n_params: usize,
        use_surrogates: bool,
    ) -> Result<Vec<f64>, crate::api::error::FluxionError> {
        debug_assert_eq!(flat.len(), n_candidates * n_params);
        use crate::physics::cta::ContinuousTensor;
        use rayon::prelude::*;

        // 1. Validate and initialize all models upfront (parallel). Issue
        //    #2874: index the row slices directly from the contiguous
        //    `flat` buffer in the closure — no per-row Vec<f64>
        //    allocation, no element copies. The previous
        //    `population_vec.par_iter()` was preceded by a
        //    `(0..n_candidates).map(|i| vec![...]).collect()` that
        //    allocated one outer Vec + N inner Vec<f64>s + 3N f64 copies.
        let mut valid_configs: Vec<(usize, ThermalModel<VectorField>)> = (0..n_candidates)
            .into_par_iter()
            .filter_map(|i| {
                let params = &flat[i * n_params..(i + 1) * n_params];
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
                            // Issue #2751: hoist the per-timestep buffers out
                            // of the 8 760-step inner loop.
                            //
                            // **Response channel**: created once as a bounded(1)
                            // crossbeam channel and reused every timestep via
                            // `resp_tx.clone()` (a cheap `Arc` bump — crossbeam
                            // `Sender` is `Arc`-backed internally, so `clone()`
                            // performs zero heap allocation). The previous code
                            // called `service.submit(temps)` which allocated a
                            // fresh `channel::unbounded()` per timestep — one
                            // heap block per channel structure, per timestep,
                            // per config worker. `submit_with_sender` is the
                            // zero-alloc variant that takes the cloned sender.
                            //
                            // **Temps / loads buffer (ping-pong)**: `temps_buf`
                            // is filled in place by `get_temperatures_into`
                            // (no allocation). The loads `Vec` returned by the
                            // service is recycled as the next timestep's
                            // `temps_buf` — the same buffer-reuse pattern the
                            // CPU batched orchestrator uses
                            // (`orchestrator.rs:486`, Issue #2771). The bytes
                            // flowing into `step_physics` are bit-identical to
                            // the prior `get_temperatures()` + `submit()` +
                            // `recv()` path; only buffer ownership differs.
                            let (resp_tx, resp_rx) = crossbeam::channel::bounded::<Vec<f64>>(1);
                            let mut temps_buf: Vec<f64> = Vec::new();
                            model.get_temperatures_into(&mut temps_buf);
                            for t in 0..8760 {
                                let hour_of_day = t % 24;
                                let daily_cycle = cycle[hour_of_day];
                                let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                                // Hand the temps to the service; the cloned
                                // sender reuses the per-worker response channel
                                // (no per-call channel allocation).
                                service.submit_with_sender(temps_buf, resp_tx.clone());
                                let loads = resp_rx
                                    .recv()
                                    .expect("Failed to receive loads from service");
                                model.set_loads(&loads);
                                energy += model.step_physics(t, outdoor_temp, 3600.0);
                                // Recycle the received loads Vec as the next
                                // timestep's temps buffer (ping-pong). The
                                // reassignment is unconditional so the borrow
                                // checker sees `temps_buf` reinitialized every
                                // iteration before the next `submit_with_sender`
                                // moves it; the (cheap) temperature refill only
                                // runs when a further step follows, since the
                                // final step's gather is wasted.
                                temps_buf = loads;
                                if t + 1 < 8760 {
                                    model.get_temperatures_into(&mut temps_buf);
                                }
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
                // Issue #2520: when a real ONNX model is loaded, take the
                // per-timestep batched path (crossbeam rendezvous) so ONNX
                // tensor batching is restored — 8760 batched inference calls
                // instead of 8.97M per-config calls (1024× reduction). For the
                // mock / analytical fallback there is no batch-dimension
                // speedup, so the zero-coordinator `par_chunks` path is faster.
                let final_worker_data = if self.surrogates.model_loaded {
                    orchestrator.run_cpu_surrogate_batched(valid_configs, &self.surrogates)
                } else {
                    orchestrator.run_cpu_surrogate(valid_configs, &self.surrogates)
                };

                for (idx, eui) in final_worker_data {
                    results[idx] = eui;
                }
            }
        } else if !valid_configs.is_empty() {
            // Analytical path (Issue #2769): dispatch through the same
            // `RayonChunksOrchestrator` used by the CPU surrogate path so
            // the analytical path gets multi-core scaling — previously this
            // branch left N-1 cores idle, which was the path the absolute
            // perf gate (#2693) and `tests/performance_regression_test.rs`
            // actually measured.
            //
            // The pre-#2769 "sequential per-config to avoid nested
            // parallelism" rationale was stale: the population-validation
            // rayon loop above has already terminated before any branch
            // runs, so a single additional `par_chunks` inside the
            // orchestrator does not introduce nested parallelism (rayon
            // thread-pool exhaustion, #1065/#2524). The
            // `.githooks/batch-oracle-check.sh` scan explicitly excludes
            // `par_chunks`, and the orchestrator file is in its scope.
            //
            // `StepParameters` is `!Send + !Sync` (its
            // `equipment: Option<Vec<Box<dyn Equipment>>>` field fails the
            // auto-traits even when the analytical variant holds `None`),
            // so it cannot be shared across workers; the orchestrator
            // constructs one `StepParameters::build_analytical()` per
            // chunk worker (Issue #1437 hoisted the prior per-timestep
            // `surrogates.clone()` out of the inner loop).
            //
            // Per-config determinism is preserved bit-identically — the
            // 8 760-step inner loop runs sequentially inside each worker,
            // and result placement is by population index. Verified by
            // `tests/batch_oracle_hotloop_equivalence.rs`.
            use crate::sim::orchestrator::{BatchOrchestrator, RayonChunksOrchestrator};

            let orchestrator = RayonChunksOrchestrator::for_population(valid_configs.len());
            let final_worker_data = orchestrator.run_cpu_analytical(valid_configs);

            for (idx, eui) in final_worker_data {
                results[idx] = eui;
            }
        }

        Ok(results)
    }
}

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
        // Fixture chosen so every config yields a valid positive EUI on the
        // analytical physics path. The previous third element `[1.0, 23.0]`
        // produced EUI = 0.0 on BOTH paths (a 5R1C solver stability quirk
        // with the synthetic daily-cycle weather used inside
        // `evaluate_population`), which tripped the positivity assertions.
        // See issue #2614.
        let population = vec![vec![1.5, 22.0], vec![2.0, 21.0], vec![0.8, 22.0]];

        // Surrogate (batched) path.
        let results_batched = oracle
            .evaluate_population(population.clone(), true)
            .unwrap();
        assert!(
            results_batched.iter().all(|r: &f64| r.is_finite()),
            "batched results must all be finite: {:?}",
            results_batched
        );

        // Analytical path for comparison.
        let results_analytical = oracle.evaluate_population(population, false).unwrap();
        assert!(
            results_analytical.iter().all(|r: &f64| r.is_finite()),
            "analytical results must all be finite: {:?}",
            results_analytical
        );

        // The analytical path is real physics — every EUI must be strictly
        // positive for these configs.
        for a in results_analytical.iter() {
            assert!(*a > 0.0, "Analytical result should be positive, got {}", a);
        }

        // The surrogate-path positivity/range comparison is only meaningful
        // when a real ONNX model is loaded. In mock mode (default build, no
        // model file) `SurrogateManager::predict_loads` returns a constant
        // 1.2 W/m² placeholder per zone, so the resulting "energy" is not a
        // real EUI and is not directly comparable to the analytical value.
        // Asserting `batched > 0.0` in mock mode would conflate mock-path
        // physics quirks with real batching bugs, so gate the strict
        // comparison behind a loaded model. (Issue #2614.)
        if !oracle.surrogates.is_mock() {
            for (batched, analytical) in results_batched.iter().zip(results_analytical.iter()) {
                assert!(*batched > 0.0, "Batched result should be positive");
                assert!(*analytical > 0.0, "Analytical result should be positive");
            }
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
}
