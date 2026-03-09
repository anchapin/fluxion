# Codebase Structure

**Analysis Date:** 2026-03-08

## Directory Layout

```
fluxion/
├── src/                      # Rust source code (core engine)
│   ├── ai/                   # AI surrogate models and inference
│   ├── physics/              # Continuous Tensor Abstraction (CTA)
│   ├── sim/                  # Physics-based building simulation
│   ├── validation/           # ASHRAE 140 validation framework
│   ├── weather/             # Weather data handling (EPW, TMY)
│   ├── bin/                 # Rust CLI entry point
│   └── lib.rs              # PyO3 module definition (Python bindings)
├── api/                     # FastAPI REST API server
├── tests/                   # Integration tests (Rust and Python)
├── tools/                   # Python training and benchmarking scripts
├── benches/                 # Rust benchmarks (CTA, engine performance)
├── examples/                # Usage examples and demos
├── docs/                    # Documentation (architecture, guides)
├── models/                  # Trained ONNX models (gitignored)
├── .planning/              # Planning documents (this file)
├── Cargo.toml              # Rust package configuration
├── pyproject.toml          # Python packaging configuration
└── README.md               # Project overview
```

## Directory Purposes

**src/:**
- Purpose: Core Rust building energy modeling engine
- Contains: Physics engine, AI surrogates, validation framework, weather handling, Python bindings
- Key files: `src/lib.rs` (PyO3 bindings), `src/sim/engine.rs` (ThermalModel), `src/ai/surrogate.rs` (ONNX integration), `src/validation/ashrae_140_validator.rs` (validation)

**src/ai/:**
- Purpose: AI surrogate models and neural network inference for fast thermal load predictions
- Contains: SurrogateManager (ONNX Runtime), NeuralScalarField (Fourier basis), context-aware inference, ensemble models, batch inference optimization
- Key files: `src/ai/surrogate.rs` (main surrogate manager), `src/ai/neural_field.rs` (neural scalar field), `src/ai/context_aware.rs` (context-aware inference)

**src/physics/:**
- Purpose: Continuous Tensor Abstraction (CTA) for unified tensor operations
- Contains: ContinuousTensor trait, VectorField (CPU implementation), continuous fields, geometry tensors
- Key files: `src/physics/cta.rs` (CTA trait and VectorField), `src/physics/continuous.rs` (ContinuousField trait), `src/physics/geometry_tensor.rs` (geometry tensors)

**src/sim/:**
- Purpose: Physics-based building energy simulation with RC thermal networks
- Contains: ThermalModel (5R1C/6R2C), HVAC controllers, solar calculations, construction assemblies, shading, lighting, occupancy schedules, interzone heat transfer
- Key files: `src/sim/engine.rs` (ThermalModel implementation), `src/sim/thermal_model.rs` (ThermalModelTrait), `src/sim/solar.rs` (solar calculations), `src/sim/construction.rs` (construction assemblies)

**src/validation/:**
- Purpose: ASHRAE 140 compliance testing and diagnostic tools
- Contains: ASHRAE140Validator, CaseSpec definitions (18 cases), diagnostic collectors, benchmark data, cross-validation framework, fault detection and diagnosis (FDD)
- Key files: `src/validation/ashrae_140_validator.rs` (validator), `src/validation/ashrae_140_cases.rs` (case specifications), `src/validation/diagnostic.rs` (diagnostic tools)

**src/weather/:**
- Purpose: Hourly meteorological data for building simulation
- Contains: HourlyWeatherData structure, EPW file parser, embedded TMY data (Denver), WeatherSource trait
- Key files: `src/weather/mod.rs` (weather structures), `src/weather/epw.rs` (EPW parser), `src/weather/denver.rs` (Denver TMY)

**src/bin/:**
- Purpose: Rust CLI entry point for validation and benchmarking
- Contains: fluxion CLI tool
- Key files: `src/bin/fluxion.rs` (CLI implementation)

**api/:**
- Purpose: Production-ready FastAPI REST API for remote building energy evaluation
- Contains: FastAPI application, population evaluation endpoints, distributed inference management, monitoring and BAS integration
- Key files: `api/main.py` (FastAPI app), `api/distributed_inference.py` (distributed inference), `api/monitoring.py` (monitoring endpoints)

**tests/:**
- Purpose: Integration tests for Rust and Python functionality
- Contains: ASHRAE 140 validation tests, issue reproduction tests, thermal mass tests, Python binding tests
- Key files: `tests/test_python_bindings.py` (Python tests), `tests/test_thermal_mass_accounting.rs` (thermal mass tests), `tests/ashrae_140_free_floating.rs` (FF cases)

**tools/:**
- Purpose: Python training scripts and benchmarking tools
- Contains: ONNX model training, RL environment, batch inference benchmarks, distributed inference configuration
- Key files: `tools/train_surrogate.py` (ONNX training), `tools/benchmark_throughput.py` (performance benchmarks), `tools/gymnasium_env.py` (RL environment)

**benches/:**
- Purpose: Rust benchmarks for CTA and engine performance
- Contains: CTA performance benchmarks, engine throughput benchmarks
- Key files: `benches/cta_bench.rs` (CTA benchmarks), `benches/engine_bench.rs` (engine benchmarks)

**examples/:**
- Purpose: Usage examples and demonstration scripts
- Contains: Risk-aware optimization examples, construction examples, validation demos
- Key files: `examples/risk_aware_optimization.py` (optimization demo), `examples/construction_example.rs` (construction usage)

**docs/:**
- Purpose: Project documentation and guides
- Contains: Architecture deep dives, ASHRAE 140 validation guides, API reference, troubleshooting
- Key files: `docs/ARCHITECTURE.md` (architecture overview), `docs/ASHRAE140_RESULTS.md` (validation results), `docs/API_REFERENCE.md` (API documentation)

**models/:**
- Purpose: Trained ONNX models for surrogate inference
- Contains: ONNX model files (gitignored)
- Generated: True
- Committed: No (gitignored)

## Key File Locations

**Entry Points:**
- `src/lib.rs`: PyO3 module definition (BatchOracle, Model classes for Python)
- `src/bin/fluxion.rs`: Rust CLI entry point (fluxion validate --all)
- `api/main.py`: FastAPI REST API entry point (HTTP endpoints)

**Configuration:**
- `Cargo.toml`: Rust package configuration (dependencies, features, release profile)
- `pyproject.toml`: Python packaging configuration (dependencies, build system)
- `rust-toolchain.toml`: Rust toolchain version (1.83.0)
- `.pre-commit-config.yaml`: Pre-commit hooks (Rust and Python linting)

**Core Logic:**
- `src/sim/engine.rs`: ThermalModel implementation (4059 lines, 5R1C/6R2C physics)
- `src/sim/thermal_model.rs`: ThermalModelTrait and modular model types
- `src/physics/cta.rs`: Continuous Tensor Abstraction (VectorField, trait definitions)
- `src/ai/surrogate.rs`: SurrogateManager (ONNX Runtime integration, session pooling)

**Testing:**
- `tests/test_python_bindings.py`: Python integration tests
- `tests/ashrae_140_free_floating.rs`: ASHRAE 140 FF case tests
- `tests/test_thermal_mass_accounting.rs`: Thermal mass energy accounting tests
- `src/validation/ashrae_140_validator.rs`: Validation framework (1431 lines)

## Naming Conventions

**Files:**
- Rust source: `snake_case.rs` (e.g., `thermal_model.rs`, `surrogate_manager.rs`)
- Python source: `snake_case.py` (e.g., `main.py`, `train_surrogate.py`)
- Module directories: `snake_case/` (e.g., `src/sim/`, `src/ai/`)

**Directories:**
- Core modules: `src/` (Rust), `api/` (Python), `tools/` (Python scripts)
- Test directories: `tests/` (integration tests), `benches/` (Rust benchmarks)
- Documentation: `docs/` (markdown files), `examples/` (usage examples)

**Functions:**
- Rust: `snake_case` (e.g., `solve_timesteps`, `apply_parameters`, `predict_loads`)
- Python: `snake_case` (e.g., `evaluate_population`, `simulate`, `validate`)

**Types:**
- Rust structs: `PascalCase` (e.g., `ThermalModel`, `VectorField`, `SurrogateManager`)
- Rust enums: `PascalCase` (e.g., `HVACMode`, `ThermalModelType`, `InferenceBackend`)
- Rust traits: `PascalCase` (e.g., `ContinuousTensor`, `ThermalModelTrait`, `WeatherSource`)
- Python classes: `PascalCase` (e.g., `BatchOracle`, `Model`, `PopulationEvaluationRequest`)

**Constants:**
- Rust: `SCREAMING_SNAKE_CASE` (e.g., `MIN_U_VALUE`, `MAX_SETPOINT`, `HOURS_PER_YEAR`)
- Python: `SCREAMING_SNAKE_CASE` (e.g., `HOURS_PER_YEAR`, `DEFAULT_NUM_ZONES`)

**Modules:**
- Rust: `snake_case` (e.g., `pub mod ai;`, `pub mod sim;`, `pub mod validation;`)
- Python: `snake_case` (e.g., `import fluxion`, `from ai import surrogate`)

## Where to Add New Code

**New Feature (Physics):**
- Primary code: `src/sim/[feature_name].rs` (e.g., `src/sim/humidity.rs` for humidity modeling)
- Tests: `tests/test_[feature_name].rs` (e.g., `tests/test_humidity.rs`)
- Add to `src/sim/mod.rs`: `pub mod [feature_name];`

**New Feature (AI):**
- Primary code: `src/ai/[feature_name].rs` (e.g., `src/ai/transformer_surrogate.rs` for transformer-based models)
- Tests: `tests/test_[feature_name].rs` (e.g., `tests/test_transformer_surrogate.rs`)
- Add to `src/ai/mod.rs`: `pub mod [feature_name];`

**New Validation Case:**
- Implementation: `src/validation/ashrae_140/case_[case_id].rs` (e.g., `case_600.rs`)
- Integration: Add case to `src/validation/ashrae_140_cases.rs` (ASHRAE140Case enum)
- Tests: `tests/test_case_[case_id].rs` (e.g., `tests/test_case_600.rs`)

**New Component/Module:**
- Implementation: `src/[category]/[component_name].rs` (e.g., `src/physics/tensor_ops.rs` for new tensor operations)
- Tests: `tests/test_[component_name].rs` (e.g., `tests/test_tensor_ops.rs`)
- Add to module `mod.rs`: `pub mod [component_name];`

**New Python API Endpoint:**
- Implementation: `api/[endpoint_name].py` (e.g., `api/health.py` for health checks)
- Integration: Add router to `api/main.py` (app.include_router)
- Tests: `api/tests/test_[endpoint_name].py` (e.g., `api/tests/test_health.py`)

**New Training Script:**
- Implementation: `tools/train_[model_name].py` (e.g., `tools/train_gnn_surrogate.py` for GNN-based surrogates)
- Dependencies: Add to `tools/requirements.txt` or `requirements-dev.txt`
- Tests: `tools/tests/test_train_[model_name].py` (e.g., `tools/tests/test_train_gnn_surrogate.py`)

**Utilities:**
- Shared helpers: `src/utils/[utility_name].rs` (e.g., `src/utils/conversions.rs` for unit conversions)
- Add to `src/lib.rs`: `pub mod utils;` (if creating utils module)

**New Benchmark:**
- Implementation: `benches/[benchmark_name].rs` (e.g., `benches/solar_bench.rs` for solar calculation performance)
- Add to `Cargo.toml`: `[[bench]] name = "[benchmark_name]" harness = false`

## Special Directories

**models/:**
- Purpose: Trained ONNX models for surrogate inference
- Contains: ONNX model files (.onnx)
- Generated: Yes (trained by scripts in `tools/`)
- Committed: No (gitignored in `.gitignore`)

**.planning/:**
- Purpose: Planning documents and codebase analysis
- Contains: Codebase mapping (this file), project configuration, work plans
- Generated: Yes (by GSD agents)
- Committed: Yes (documentation)

**target/:**
- Purpose: Rust build artifacts (compiled binaries, libraries)
- Contains: Debug/release builds, dependencies
- Generated: Yes (by `cargo build`)
- Committed: No (gitignored in `.gitignore`)

**.venv/, venv/:**
- Purpose: Python virtual environments
- Contains: Installed Python packages
- Generated: Yes (by `python -m venv`)
- Committed: No (gitignored in `.gitignore`)

**.pytest_cache/, .ruff_cache/:**
- Purpose: Test runner and linter cache
- Contains: Cached test results, linting data
- Generated: Yes (by pytest, ruff)
- Committed: No (gitignored in `.gitignore`)

**.github/:**
- Purpose: GitHub Actions CI/CD workflows
- Contains: Workflow YAML files
- Generated: No
- Committed: Yes (CI/CD configuration)

**.githooks/:**
- Purpose: Custom Git hooks for code quality
- Contains: Shell scripts for batch-oracle-pattern enforcement, rust-doc-check
- Generated: No
- Committed: Yes (developer tooling)

**.jules/:**
- Purpose: Jules AI agent workspace
- Contains: Agent configuration and workspace data
- Generated: Yes
- Committed: No (gitignored in `.gitignore`)

**worktrees/:**
- Purpose: Git worktrees for parallel development
- Contains: Separate working directories
- Generated: Yes (by `git worktree add`)
- Committed: No (gitignored in `.gitignore`)

---

*Structure analysis: 2026-03-08*
