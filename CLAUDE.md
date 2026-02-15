# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in repository.

## Project Overview

**Fluxion** is a Rust-based Building Energy Modeling (BEM) engine with a **Neuro-Symbolic hybrid architecture**. It combines physics-based thermal networks with AI surrogates for 100x speedups, designed to serve as a high-throughput oracle for quantum optimization and genetic algorithms.

- **Core Language**: Rust (Edition 2021)
- **Python Bindings**: PyO3 + maturin
- **Key Use Case**: Evaluating 10,000+ building design configurations/second
- **Physics Model**: ISO 13790-compliant 5R1C Thermal Network using Continuous Tensor Abstraction (CTA)
- **Neural Fields**: Fourier basis neural representation (`NeuralScalarField`) for continuous field modeling

## Architecture: The "Batch Oracle" Pattern

Fluxion uses a **two-class** PyO3 API pattern critical to understand:

### 1. **BatchOracle** (the "hot loop")
- **Purpose**: High-throughput parallel evaluation of population vectors for optimization loops
- **Location**: `src/lib.rs` (exposed to Python)
- **Threading Model**: Uses `rayon` for data parallelism—each configuration runs on a thread pool
- **Key Method**: `evaluate_population(population: Vec<Vec<f64>>, use_surrogates: bool) -> Vec<f64>`
  - Accepts ~10,000 gene vectors (parameter arrays) from quantum optimizers or GA algorithms
  - Returns corresponding fitness scores (EUI - Energy Use Intensity)
  - **Critical**: Minimize Python-Rust boundary crossings; pass entire population at once

**Architecture with Surrogates (Time-First Loop)**:
When `use_surrogates=true`, the implementation uses a time-first loop to maximize GPU utilization:
1. Time loop (0..8760) runs sequentially on main thread
2. Collect all temperatures from all configurations → single batched inference
3. Distribute loads via `set_loads()`, run `step_physics()` in parallel with `rayon`

This avoids nested parallelism and maximizes GPU tensor core utilization for batched inference.

**Architecture without Surrogates (Config-First Loop)**:
When `use_surrogates=false`, each config runs independently through all timesteps in parallel.

- **Pattern**: Light clones of `ThermalModel`, apply parameters, solve in parallel, collect results

### 2. **Model** (detailed single-building analysis)
- **Purpose**: Single-configuration simulation for validation or detailed inspection
- **Use**: When engineers need hourly temperature traces or ASHRAE 140 validation
- **API**: `simulate(years: u32, use_surrogates: bool) -> f64`

## Core Physics Engine

### Thermal Network (`src/sim/engine.rs`)

**ThermalModel** is the physics backbone, implementing an **ISO 13790-compliant 5R1C Thermal Network** using **Continuous Tensor Abstraction (CTA)**.

```rust
pub struct ThermalModel {
    pub num_zones: usize,
    // State variables (CTA VectorFields)
    pub temperatures: VectorField,       // Zone air temperatures
    pub mass_temperatures: VectorField,  // Thermal mass temperatures
    pub loads: VectorField,              // Total thermal loads (Watts)

    // Design variables
    pub window_u_value: f64,
    pub hvac_setpoint: f64,

    // 5R1C Parameters (CTA VectorFields)
    pub h_tr_em: VectorField, // Transmission: Exterior -> Mass
    pub h_tr_ms: VectorField, // Transmission: Mass -> Surface
    pub h_tr_is: VectorField, // Transmission: Surface -> Interior
    pub h_tr_w: VectorField,  // Transmission: Exterior -> Interior (Windows)
    pub h_ve: VectorField,    // Ventilation: Exterior -> Interior
}
```

**Key Methods**:
- `apply_parameters(params: &[f64])`: Maps gene vector to model state and broadcasts 5R1C conductances to `VectorField`
  - `params[0]` → `window_u_value` (updates `h_tr_w`, `h_tr_em`)
  - `params[1]` → `hvac_setpoint`
- `solve_timesteps(steps, surrogates, use_ai)`: Core physics loop
  - Uses **CTA operations** (element-wise `+`, `*`, `/`) for vector-accelerated solving of the 5R1C algebraic system
  - Calculates `Ti_free` (free-floating temp), determines HVAC demand, and solves `Ti_act` and `Tm_next`
  - If `use_ai=true`: queries `SurrogateManager` for loads
  - If `use_ai=false`: computes analytical loads
  - Returns cumulative energy consumption

**Important**: `ThermalModel` is `Clone`—this enables the batch parallel pattern (each thread gets a copy to mutate).

## Continuous Tensor Abstraction (CTA)

The CTA (`src/physics/cta.rs`) provides a unified API for tensor-like operations used by the physics engine. It abstracts vector operations to enable future GPU acceleration.

```rust
use fluxion::physics::cta::VectorField;
let v = VectorField::new(vec![1.0, 2.0, 3.0]);
let g = v.gradient();
let integral = v.integrate();
```

**Key Point**: The physics engine uses CTA operations (`+`, `*`, `/`) on `VectorField` types, not raw `Vec<f64>`.

### Neural Scalar Fields (`src/ai/neural_field.rs`)

`NeuralScalarField<T>` represents a continuous scalar field using a Fourier basis neural network. It implements the `ContinuousField` trait for field evaluation and integration.

- **Basis**: Tensor product of 1D Fourier series (1, cos(kπx), sin(kπx), ...)
- **Domain**: [0, 1] × [0, 1]
- **ONNX Integration**: Can load from ONNX model weights via `from_onnx()`

```rust
use fluxion::ai::neural_field::NeuralScalarField;
// Create from weights (must be perfect square, (sqrt(n)-1) % 2 == 0)
let field = NeuralScalarField::new(vec![1.0, 0.1, 0.2, ...])?;

// Evaluate at point (0.5, 0.5)
let value = field.at(0.5, 0.5);

// Integrate over domain
let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
```

## AI Surrogates Integration

### SurrogateManager (`src/ai/surrogate.rs`)

- **Current Status**: Uses ONNX Runtime 2.0.0-rc.10 with thread-safe `SessionPool` for concurrent inference
- **Role**: Replaces expensive CFD/ray-tracing with pre-trained neural networks
  - Input: Current zone temperatures (`&[f64]`)
  - Output: Predicted thermal loads (solar radiation, infiltration, convection)
- **Design Philosophy**: Physics-Informed—neural predictions constrained by energy balance
- **Integration Point**: Called from `ThermalModel::solve_timesteps` when `use_ai=true`
- **Supported Backends**: CPU, CUDA, CoreML, DirectML, OpenVINO (via `InferenceBackend` enum)

**Key Methods**:
- `new()`: Creates manager with mock predictions (no model loaded)
- `load_onnx(path)`: Loads ONNX model with default CPU backend
- `with_gpu_backend(path, backend, device_id)`: Loads model with specific backend (CUDA, CoreML, etc.)
- `predict_loads(current_temps)`: Single prediction for one configuration
- `predict_loads_batched(batch_temps)`: Batched prediction for multiple configurations

**Batched Inference**: The `SessionPool` enables concurrent inference by providing session guards that return sessions to the pool when dropped. For maximum throughput, use `predict_loads_batched()` to process multiple temperature vectors in a single ONNX session run.

## ASHRAE Standard 140 Validation
Fluxion is fully validated against ASHRAE 140.
- **Run validation**: `fluxion validate --all`
- **Validation Report**: See `docs/ASHRAE140_RESULTS.md`
- **Status**: 18/18 cases passing (✅)

### Developer Workflows
- **Test execution**: `cargo test` runs all units and integration tests.
- **CI/CD**: Every PR triggers automated ASHRAE 140 validation.

## Build and Test Commands


| Task | Command |
|------|---------|
| **Setup local dev** | `cargo build && maturin develop` |
| **Run tests** | `cargo test` |
| **Code quality check** | `cargo fmt && cargo clippy && cargo test` |
| **Build wheel** | `maturin build --release` |
| **Quick syntax check** | `cargo check` |
| **Auto-fix clippy issues** | `cargo clippy --fix --allow-dirty` |
| **Run benchmarks** | `cargo bench --bench cta_bench` |
| **Quick smoke test** | `python -c "import fluxion; print(fluxion.BatchOracle())"` |
| **Test with specific backend** | `RUST_LOG=debug cargo test -- --nocapture` |

### Development Setup

```bash
# First-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
cargo build
maturin develop

# Typical iteration
cargo fmt && cargo clippy && cargo test
```

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_thermal_model_energy_conservation

# With output
cargo test -- --nocapture

# Single-threaded (for debugging race conditions)
cargo test -- --test-threads=1
```

## Critical Patterns & Conventions

### Batch Processing (The "Hot Loop")
- **NEVER** make nested calls across Python-Rust boundary during loops—collect all data, cross once
- **ALWAYS** use `rayon::par_iter()` for population-level parallelism; avoid nested parallelism inside `solve_timesteps`
- Clone base model → apply unique parameters → run physics → collect results
- **Pre-commit Hook**: The `batch-oracle-pattern` hook enforces single-level parallelism and rejects nested `par_iter()` in `evaluate_population`

### Parameter Vector Semantics
- The population format is **critical for external APIs** (D-Wave, GA libraries):
  - Element 0: Window U-value (range: 0.1–5.0 W/m²K) - defined by `MIN_U_VALUE` and `MAX_U_VALUE` constants
  - Element 1: HVAC setpoint (range: 15–30°C) - defined by `MIN_SETPOINT` and `MAX_SETPOINT` constants
  - Future elements: Thermal mass, infiltration rates, etc.
- Document any new design variables added to this vector
- `validate_parameters()` function enforces these constraints and returns `NaN` for invalid configs

### Simulation Timesteps
- 1 year = 8760 hours
- Internally represented as `VectorField` of hourly states
- Use `(years * 8760)` for step count

### Repository Hygiene
- **Only `.md` file in root should be `README.md`** — all other docs go in `docs/`
- Clean up temporary files before committing:
  - Delete `PRECOMMIT_*.md`, `plan.md`, `*.copilotmd` (or move to `tmp/`)
  - Verify: `ls -la | grep "\.md$"` should show only `README.md`

### Commit Message Convention
- **Format**: `<type>(<scope>): <subject>`
- **Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`
- **Examples**:
  - `feat(surrogate): integrate ONNX runtime session initialization`
  - `perf(engine): reduce memory allocations in solve_timesteps inner loop`
  - `test(batch-oracle): add population scaling validation`
  - `fix(physics): correct window U-value calculation units`

## Developer Workflows

### Adding a New Design Variable
1. Add field to `ThermalModel` struct (e.g., `thermal_mass: f64`)
2. Extend `apply_parameters()` to map new gene to field
3. Update physics in `solve_timesteps()` to use it
4. Document the parameter vector semantics
5. Test via `BatchOracle::evaluate_population` with sample populations

### Integrating a Pre-trained ONNX Model
1. Place model file in `models/` directory (gitignored)
2. In `SurrogateManager::new()`, initialize ONNX Runtime session
3. Implement `predict_loads()` to convert `&[f64]` → tensor → session.run() → output
4. Validate predictions against ground truth before enabling in production

### Creating Neural Scalar Fields
1. Prepare weight vector in correct format: length must be perfect square, `(sqrt(len) - 1) % 2 == 0`
2. Create field: `NeuralScalarField::new(weights)?`
3. For ONNX models: Use `NeuralScalarField::from_onnx(path, input_array)`
4. The field implements `ContinuousField` trait with `at(u, v)` and `integrate()` methods

### Training AI Surrogates
```bash
# Install dependencies
pip install -r requirements-dev.txt

# Train a surrogate model
python tools/train_surrogate.py --num-samples 50000 --epochs 100

# Models are saved to models/ (gitignored)
```

### Testing Physics Changes
- Use `Model::simulate(1, use_surrogates=false)` for single-year analytical validation
- Compare against baseline energies (documented in `docs/Fluxion_PRD.md`)
- Run population-level tests via `BatchOracle` to catch performance regressions

### Testing ONNX Surrogates
- Mock tests (no model): Use `SurrogateManager::new()` which returns `vec![1.2; ...]`
- Real model tests: Create dummy ONNX model (e.g., `tests_tmp_dummy.onnx`) for CI testing
- Tests should gracefully skip when ONNX model file is not present (check with `Path::exists()`)
- **ONNX Runtime Version**: Using `ort = "2.0.0-rc.10"` with `download-binaries` feature for easy setup

## Performance Requirements

### Critical Metrics
- **Per-Configuration Latency**: Single `solve_timesteps(8760)` should complete in <100ms
- **Throughput**: `BatchOracle::evaluate_population(1000)` should complete in <100ms total (100μs per config)
- **Target**: 10,000+ configurations/second on 8-core CPU

### Optimization Guidelines
- Use `rayon::par_iter()` only at population level
- Minimize Python-Rust boundary crossings
- Avoid allocations in inner loops
- Test with `--release` profile (aggressive optimizations: `lto=true`, `codegen-units=1`)
- Use `SessionPool` and `predict_loads_batched()` for concurrent ONNX inference
- For GPU inference: Use `SurrogateManager::with_gpu_backend(path, InferenceBackend::CUDA, device_id)`

## Common Pitfalls

1. **Cloning ThermalModel in inner loops**: Use `par_iter()` at the population level, not nested parallelism
2. **Ignoring Python boundary cost**: Test with realistic population sizes (~1000+) to catch FFI overhead
3. **Hardcoding parameter semantics**: Always document the gene-to-field mapping
4. **Forgetting `use_surrogates` flag**: Tests must validate both analytical and surrogate paths
5. **Not running `cargo fmt` before commit**: Linting failures block merge
6. **Skipping tests for "small" changes**: Physics changes can have downstream effects; always test
7. **Assuming parallel performance**: Profile with release builds; debug builds hide threading issues
8. **Using raw Vec instead of CTA**: Physics engine expects `VectorField` types for state variables
9. **Incorrect NeuralScalarField weight size**: Weights length must be perfect square where `(sqrt(len) - 1) % 2 == 0` (e.g., 1, 9, 25, 49, ...)
10. **Not using batched inference**: For large populations, `predict_loads_batched()` is significantly faster than repeated `predict_loads()` calls
11. **Assuming pre-commit hooks will skip**: The `batch-oracle-pattern` hook actively enforces parallelism patterns—fix violations before committing

## Key Files

- **Entrypoint**: `src/lib.rs` (PyO3 module definition, BatchOracle, Model classes)
- **Physics**: `src/sim/engine.rs` (ThermalModel, 5R1C thermal network, solve loop)
- **CTA**: `src/physics/cta.rs` (Continuous Tensor Abstraction, VectorField, ContinuousTensor trait)
- **AI Layer**:
  - `src/ai/surrogate.rs` (SurrogateManager, SessionPool, ONNX integration with multiple backends)
  - `src/ai/neural_field.rs` (NeuralScalarField for Fourier basis neural representation)
- **Hooks**: `.githooks/batch-oracle-check.sh` (enforces single-level parallelism), `.githooks/rust-doc-check.py` (doc comments)
- **Docs**: `docs/ARCHITECTURE.md` (detailed architecture), `docs/CONTRIBUTING.md` (development guidelines)
- **Config**: `Cargo.toml` (dependencies, release profile)

## Pre-commit Hooks

The project uses pre-commit hooks for code quality:
- **Rust**: `cargo fmt`, `cargo check`, `cargo audit`
- **Python**: `ruff-check`, `ruff-format`, `black`, `isort`, `mypy`
- **Custom**: `batch-oracle-pattern` (ensures BatchOracle pattern compliance), `rust-doc-check` (doc comment validation)
- **Commit Message**: `conventional-pre-commit` enforces `<type>(<scope>): <subject>` format

Enable with:
```bash
pre-commit install
pre-commit install --hook-type commit-msg -f
```

## External Integration Points

- **D-Wave Quantum Annealer**: Receives population vectors from solver, calls `evaluate_population`
- **Python GA Libraries**: Same interface—population arrays, energy costs back
- **ASHRAE 140 Validation**: Use `Model` class for detailed hourly traces
- **Future**: FMI 3.0 co-simulation (planned Phase 3)

## Debugging Tips

### Rust-side Debugging
```bash
# Enable debug symbols in release build
cargo build --release --debuginfo=full

# Check for FFI errors
python -c "import fluxion; oracle = fluxion.BatchOracle(); print(oracle.evaluate_population([[1.5, 21.0]], False))"
```

### Common Issues
- **"module not found"**: Run `maturin develop` after changes to Rust code
- **"thread panicked"**: Check physics calculations in `solve_timesteps` for NaN/Inf
- **"rayon panicked"**: Enable `--test-threads=1` to debug parallel issues sequentially
