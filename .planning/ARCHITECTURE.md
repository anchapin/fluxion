# Architecture

**Analysis Date:** 2026-03-08

## Pattern Overview

**Overall:** Neuro-Symbolic Hybrid Architecture with Batch Oracle Pattern

**Key Characteristics:**
- Two-tier PyO3 API: BatchOracle for high-throughput optimization, Model for detailed single-building analysis
- Physics-based 5R1C/6R2C thermal network compliant with ISO 13790 and ASHRAE 140 standards
- Continuous Tensor Abstraction (CTA) for unified tensor operations across CPU and future GPU backends
- AI surrogate integration via ONNX Runtime for fast thermal load predictions
- Thread-safe parallel execution using Rayon for population-level data parallelism
- Time-first vs Config-first parallelism strategies to maximize GPU utilization when using surrogates

## Layers

**Physics Engine Layer:**
- Purpose: Core building energy modeling with RC (Resistor-Capacitor) thermal networks
- Location: `src/sim/`
- Contains: ThermalModel, HVAC controllers, solar calculations, construction assemblies, shading, lighting, occupancy schedules
- Depends on: `src/physics/` (CTA), `src/weather/` (weather data), `src/validation/` (ASHRAE 140 cases)
- Used by: PyO3 bindings (Model, BatchOracle), validation tests, distributed inference

**Continuous Tensor Abstraction (CTA) Layer:**
- Purpose: Unified tensor operations for physics computations, enabling future GPU acceleration
- Location: `src/physics/`
- Contains: ContinuousTensor trait, VectorField (CPU implementation), continuous fields, geometry tensors
- Depends on: External crates (faer, ndarray, num-traits)
- Used by: Physics Engine (ThermalModel state variables), AI surrogates (neural fields)

**AI Surrogate Layer:**
- Purpose: Fast neural network inference for thermal load predictions (solar, infiltration, convection)
- Location: `src/ai/`
- Contains: SurrogateManager (ONNX Runtime session pool), NeuralScalarField (Fourier basis neural representation), context-aware inference, ensemble models, batch inference optimization
- Depends on: ort (ONNX Runtime 2.0.0-rc.10), PyO3 (Python bindings)
- Used by: ThermalModel (when use_surrogates=true), BatchOracle (batched inference)

**Validation Layer:**
- Purpose: Compliance testing against ASHRAE 140 standard (18 cases), cross-validation, diagnostics
- Location: `src/validation/`
- Contains: ASHRAE140Validator, CaseSpec definitions, diagnostic collectors, benchmark data, cross-validation framework, fault detection and diagnosis (FDD)
- Depends on: Physics Engine, Weather data
- Used by: CLI validation command, CI/CD pipeline, development workflow

**Weather Layer:**
- Purpose: Hourly meteorological data for building simulation
- Location: `src/weather/`
- Contains: HourlyWeatherData structure, EPW file parser, embedded TMY data (Denver), WeatherSource trait
- Depends on: serde (serialization)
- Used by: Physics Engine, Validation layer

**Python Binding Layer:**
- Purpose: Expose Rust functionality to Python via PyO3
- Location: `src/lib.rs` (feature-gated with python-bindings)
- Contains: BatchOracle class (parallel population evaluation), Model class (single-building simulation), PyVectorField wrapper
- Depends on: Physics Engine, AI Surrogates, CTA
- Used by: Python scripts, REST API (`api/main.py`), optimization workflows

**REST API Layer:**
- Purpose: Production-ready HTTP interface for remote building energy evaluation
- Location: `api/main.py`
- Contains: FastAPI application, population evaluation endpoints, distributed inference management, monitoring and BAS integration
- Depends on: Python bindings (fluxion), distributed inference manager
- Used by: External optimization tools, web clients, monitoring dashboards

## Data Flow

**BatchOracle Evaluation (Time-First with Surrogates):**

1. Python caller passes population matrix (Vec<Vec<f64>>) to `BatchOracle::evaluate_population()`
2. Rust receives population via PyO3, each row is a gene vector (window_u_value, setpoint, ...)
3. Clone base ThermalModel for each configuration (lightweight due to CTA)
4. **Time-first loop** (maximizes GPU utilization):
   - Loop timesteps 0..8760 on main thread
   - Collect all temperatures from all configurations
   - Single batched inference via `SurrogateManager::predict_loads_batched()`
   - Distribute loads via `set_loads()`, run `step_physics()` in parallel with `rayon::par_iter()`
5. Collect energy consumption (EUI) from each configuration
6. Return Vec<f64> to Python via PyO3

**Model Simulation (Config-First without Surrogates):**

1. Python caller creates `Model::new(num_zones)` and calls `simulate(years, use_surrogates)`
2. Rust receives configuration via PyO3
3. **Config-first loop** (each config runs independently):
   - Loop timesteps 0..8760 sequentially
   - Compute analytical loads (no surrogates)
   - Step physics forward using CTA operations
4. Return cumulative energy consumption (EUI) to Python

**ThermalModel Physics Step:**

1. Retrieve weather data for current timestep (dry bulb, solar radiation, etc.)
2. Calculate solar gains via `calculate_hourly_solar()` (considering orientation, shading, window properties)
3. Compute thermal loads:
   - If use_surrogates=true: Query `SurrogateManager::predict_loads()` (neural network)
   - If use_surrogates=false: Compute analytical loads (conduction, convection, infiltration)
4. Apply loads to CTA VectorField state variables
5. Solve 5R1C/6R2C algebraic system:
   - Calculate Ti_free (free-floating zone temperature)
   - Determine HVAC demand based on setpoints and deadband
   - Solve Ti_act (actual zone temperature with HVAC)
   - Update Tm_next (thermal mass temperature for next timestep)
6. Accumulate energy consumption for HVAC operation

**ASHRAE 140 Validation:**

1. `ASHRAE140Validator::validate()` iterates through 18 predefined cases (600 series, 900 series, FF cases)
2. For each case: Build ThermalModel from CaseSpec (geometry, construction, schedules)
3. Simulate 1 year (8760 timesteps) using Denver TMY weather
4. Compare results against benchmark data (annual energy, peak loads, monthly profiles)
5. Generate diagnostic report (temperature profiles, energy breakdown, peak timing)
6. Calculate MAE, RMSE, and pass/fail status for each metric

**State Management:**
- ThermalModel state (temperatures, mass_temperatures, loads) stored as CTA VectorFields
- VectorFields are Clone, enabling parallel processing without mutation conflicts
- HVAC controller maintains separate setpoint schedules and capacity limits
- SurrogateManager uses thread-safe SessionPool for concurrent ONNX inference

## Key Abstractions

**ContinuousTensor<T>:**
- Purpose: Unified interface for tensor operations across CPU and GPU backends
- Examples: `src/physics/cta.rs` (trait), `src/physics/cta.rs` (VectorField implementation)
- Pattern: Trait with element-wise operations (add, sub, mul, div), reduction (reduce, integrate), and spatial operations (gradient, elementwise_min/max)

**ThermalModelTrait:**
- Purpose: Modular interface for swapping physics-based and surrogate-based thermal models
- Examples: `src/sim/thermal_model.rs` (trait, PhysicsThermalModel, SurrogateThermalModel, UnifiedThermalModel)
- Pattern: Trait with methods for solving timesteps, applying parameters, querying temperatures and setpoints

**WeatherSource:**
- Purpose: Abstract different weather data sources (EPW files, embedded TMY)
- Examples: `src/weather/mod.rs` (trait), `src/weather/epw.rs` (EpwWeatherSource), `src/weather/denver.rs` (DenverTmyWeather)
- Pattern: Trait with method `get_hourly_data(hour_of_year) -> HourlyWeatherData`

**ContinuousField<T>:**
- Purpose: Represent continuous scalar fields for spatial calculations (solar radiation, view factors)
- Examples: `src/physics/continuous.rs` (trait), `src/physics/continuous.rs` (ConstantField), `src/ai/neural_field.rs` (NeuralScalarField)
- Pattern: Trait with methods `at(x, y)` (evaluate at point) and `integrate(x1, x2, y1, y2)` (integrate over domain)

**CaseSpec (ASHRAE 140):**
- Purpose: Declarative specification of building geometry, construction, schedules for validation cases
- Examples: `src/validation/ashrae_140_cases.rs` (CaseSpec struct, CaseBuilder)
- Pattern: Builder pattern with fluent API for configuring zone dimensions, wall areas, window areas, construction types, HVAC schedules

## Entry Points

**BatchOracle (Python):**
- Location: `src/lib.rs` (#[pyclass] BatchOracle)
- Triggers: Python instantiation `fluxion.BatchOracle()`, call to `evaluate_population(population, use_surrogates)`
- Responsibilities: High-throughput parallel evaluation of design populations, Rayon-based data parallelism, batched ONNX inference when surrogates enabled

**Model (Python):**
- Location: `src/lib.rs` (#[pyclass] Model)
- Triggers: Python instantiation `fluxion.Model(num_zones)`, call to `simulate(years, use_surrogates)`
- Responsibilities: Single-building detailed simulation, hourly temperature traces, ASHRAE 140 validation

**fluxion CLI (Rust):**
- Location: `src/bin/fluxion.rs`
- Triggers: Command-line invocation `fluxion validate --all`
- Responsibilities: CLI interface for ASHRAE 140 validation, benchmarking, testing

**REST API (Python):**
- Location: `api/main.py` (FastAPI application)
- Triggers: HTTP POST to `/evaluate_population` or `/simulate`
- Responsibilities: Remote population evaluation, distributed inference management, monitoring and BAS integration endpoints

**Unit Tests (Rust):**
- Location: `tests/` directory and inline #[test] modules
- Triggers: `cargo test` invocation
- Responsibilities: Testing individual components (ThermalModel, CTA operations, validation logic)

**Integration Tests (Python):**
- Location: `tests/*.py` files
- Triggers: `pytest` invocation
- Responsibilities: Testing Python bindings, ASHRAE 140 integration, surrogate integration

## Error Handling

**Strategy:** Result<T, Box<dyn Error + Send + Sync>> pattern for thermal model operations, panics for unrecoverable errors

**Patterns:**
- ThermalModelResult<T> alias in `src/sim/thermal_model.rs` wraps Box<dyn Error + Send + Sync>
- PyO3 converts Rust errors to Python exceptions (PyRuntimeError, PyValueError)
- Validation errors return ValidationError with diagnostic context
- SurrogateManager returns Result types for ONNX initialization and inference failures
- Assert checks for invariants (e.g., temperature bounds, parameter ranges) with descriptive messages

## Cross-Cutting Concerns

**Logging:** RUST_LOG environment variable controls verbosity (debug, info, warn, error), used throughout codebase for debugging physics calculations and validation

**Validation:**
- ASHRAE 140 compliance: All 18 cases must pass (600/900 series, FF cases)
- Parameter validation: `apply_parameters()` checks bounds (MIN_U_VALUE, MAX_U_VALUE, MIN_SETPOINT, MAX_SETPOINT)
- Energy conservation: Cross-validation checks for physical consistency

**Authentication:** Not applicable (local development and research tool)

**Parallelism:** Rayon for data parallelism (population-level), strict single-level parallelism enforced by pre-commit hook (batch-oracle-pattern), time-first vs config-first strategies based on surrogate usage

**Memory Management:** Clone semantics for ThermalModel enable safe parallelism, VectorField uses Vec<f64> internally, ONNX Runtime manages GPU memory for surrogates

**Configuration:** Environment variables (RUST_LOG), feature flags (python-bindings, cuda), runtime parameters (surrogate paths, weather data)

---

*Architecture analysis: 2026-03-08*
