# Fluxion: AI Copilot Instructions

## Project Overview

**Fluxion** is a Rust-based Building Energy Modeling (BEM) engine with a **Neuro-Symbolic hybrid architecture**. It combines physics-based thermal networks with AI surrogates for 100x speedups, designed to serve as a high-throughput oracle for quantum optimization and genetic algorithms.

- **Core Language**: Rust (Edition 2021)
- **Python Bindings**: PyO3 + maturin
- **Key Use Case**: Evaluating 10,000+ building design configurations/second

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
- **Pattern**: Light clones of `ThermalModel`, apply parameters, solve in parallel, collect results

### 2. **Model** (detailed single-building analysis)
- **Purpose**: Single-configuration simulation for validation or detailed inspection
- **Use**: When engineers need hourly temperature traces or ASHRAE 140 validation
- **API**: `simulate(years: u32, use_surrogates: bool) -> f64`

## Core Physics Engine

### Thermal Network (`src/sim/engine.rs`)

**ThermalModel** is the physics backbone, now implementing an **ISO 13790-compliant 5R1C Thermal Network** using **Continuous Tensor Abstraction (CTA)**.

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
- `apply_parameters(params: &[f64])`: Maps gene vector to model state and broadcasts 5R1C conductances to `VectorField`.
  - `params[0]` → `window_u_value` (updates `h_tr_w`, `h_tr_em`)
  - `params[1]` → `hvac_setpoint`
- `solve_timesteps(steps, surrogates, use_ai)`: Core physics loop
  - Uses **CTA operations** (element-wise `+`, `*`, `/`) for vector-accelerated solving of the 5R1C algebraic system.
  - Calculates `Ti_free` (free-floating temp), determines HVAC demand, and solves `Ti_act` and `Tm_next`.
  - If `use_ai=true`: queries `SurrogateManager` for loads.
  - If `use_ai=false`: computes analytical loads.
  - Returns cumulative energy consumption.

**Important**: `ThermalModel` is `Clone`—this enables the batch parallel pattern (each thread gets a copy to mutate).

## AI Surrogates Integration

### SurrogateManager (`src/ai/surrogate.rs`)

- **Current Status**: Placeholder with mock predictions (returns `vec![1.2; ...]`)
- **Future Implementation**: Will wrap ONNX Runtime sessions (`ort` crate)
- **Role**: Replaces expensive CFD/ray-tracing with pre-trained neural networks
  - Input: Current zone temperatures (`&[f64]`)
  - Output: Predicted thermal loads (solar radiation, infiltration, convection)
- **Design Philosophy**: Physics-Informed—neural predictions constrained by energy balance
- **Integration Point**: Called from `ThermalModel::solve_timesteps` when `use_ai=true`

## Build & Deployment

### Rust Core
```bash
cargo build --release  # LTO + single codegen-unit for optimal performance
cargo test             # Run unit tests
cargo bench            # Run benchmarks (when added)
cargo check            # Fast syntax/type check without compilation
cargo clippy           # Lint for common mistakes
cargo fmt              # Auto-format code (should be run before commits)
```

### Python Bindings
```bash
pip install maturin
maturin develop       # Build & install local development version
maturin build         # Generate publishable wheel
maturin build --release  # Optimized wheel for distribution
```

**Note**: The `release` profile has aggressive optimizations (`lto=true`, `codegen-units=1`) critical for the throughput requirements.

### Local Development Workflow
```bash
# First-time setup
cargo build           # Debug build for fast iteration
maturin develop      # Install Python bindings locally

# Typical iteration
cargo fmt && cargo clippy  # Format and lint
cargo test                 # Run unit tests
python -c "import fluxion; print(fluxion.BatchOracle())"  # Quick smoke test
```

## Critical Patterns & Conventions

### Batch Processing (The "Hot Loop")
- **Never** make nested calls across Python-Rust boundary during loops—collect all data, cross once
- **Always** use `rayon::par_iter()` for population-level parallelism; avoid nested parallelism inside `solve_timesteps`
- Clone base model → apply unique parameters → run physics → collect results

### Parameter Vector Semantics
- The population format is **critical for external APIs** (D-Wave, GA libraries):
  - Element 0: Window U-value (range: 0.5–3.0 W/m²K)
  - Element 1: HVAC setpoint (range: 19–24°C)
  - Future elements: Thermal mass, infiltration rates, etc.
- Document any new design variables added to this vector

### Simulation Timesteps
- 1 year = 8760 hours
- Internally represented as `Vec<f64>` of hourly states
- Use `(years * 8760)` for step count

## Developer Workflows

### Adding a New Design Variable
1. Add field to `ThermalModel` struct (e.g., `thermal_mass: f64`)
2. Extend `apply_parameters()` to map new gene to field
3. Update physics in `solve_timesteps()` to use it
4. Document the parameter vector semantics
5. Test via `BatchOracle::evaluate_population` with sample populations

### Integrating a Pre-trained ONNX Model
1. Place model file in project (e.g., `assets/loads_predictor.onnx`)
2. In `SurrogateManager::new()`, initialize ONNX Runtime session
3. Implement `predict_loads()` to convert `&[f64]` → tensor → session.run() → output
4. Validate predictions against ground truth before enabling in production

### Testing Physics Changes
- Use `Model::simulate(1, use_surrogates=false)` for single-year analytical validation
- Compare against baseline energies (documented in `docs/Fluxion_PRD.md`)
- Run population-level tests via `BatchOracle` to catch performance regressions

## External Integration Points

- **D-Wave Quantum Annealer**: Receives population vectors from solver, calls `evaluate_population`
- **Python GA Libraries**: Same interface—population arrays, energy costs back
- **ASHRAE 140 Validation**: Use `Model` class for detailed hourly traces
- **Future**: FMI 3.0 co-simulation (planned Phase 3)

## Key Files to Reference

- **Entrypoint**: `src/lib.rs` (PyO3 module definition)
- **Physics**: `src/sim/engine.rs` (ThermalModel, solve loop)
- **AI Layer**: `src/ai/surrogate.rs` (SurrogateManager, ONNX integration)
- **Docs**: `docs/Fluxion_PRD.md` (full architecture & roadmap)
- **Config**: `Cargo.toml` (dependencies, release profile)

## Testing Strategy

### Test Organization
- **Unit Tests**: Placed in the same file as implementation using `#[cfg(test)]` modules
- **Physics Validation**: Use `Model::simulate()` with `use_surrogates=false` against known baselines
- **Batch Performance**: Test `BatchOracle::evaluate_population()` with realistic population sizes (1000+) to catch FFI overhead
- **Surrogate Accuracy**: Compare `use_surrogates=true` vs `use_surrogates=false` to measure error bounds

### Test-Driven Development (TDD) Workflow
1. **Add Test First**: Write test case in `#[cfg(test)]` module with clear assertions
2. **Run `cargo test` to Fail**: Verify test fails before implementing
3. **Implement Feature**: Write minimal code to pass test
4. **Run Full Suite**: `cargo test` + `cargo clippy` + `cargo fmt`
5. **Commit**: Only commit when all tests pass and code is formatted

### Physics Test Template
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_model_energy_conservation() {
        let mut model = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().unwrap();

        // Analytical baseline (no AI)
        let energy_analytical = model.clone().solve_timesteps(8760, &surrogates, false);

        // Surrogate prediction (should be close)
        model.solve_timesteps(8760, &surrogates, true);

        // Verify energy conservation within tolerance
        assert!(energy_analytical.abs() > 0.0, "Energy should be non-zero");
    }

    #[test]
    fn test_apply_parameters_updates_model() {
        let mut model = ThermalModel::new(10);
        let params = vec![1.5, 22.0];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);
    }
}
```

### Batch Population Test Template
```rust
#[test]
fn test_batch_oracle_population_scaling() {
    let oracle = BatchOracle::new().unwrap();

    // Small population: 100 candidates
    let small_pop: Vec<Vec<f64>> = (0..100)
        .map(|i| vec![0.5 + (i as f64 * 0.01), 21.0])
        .collect();
    let small_results = oracle.evaluate_population(small_pop, false).unwrap();
    assert_eq!(small_results.len(), 100);

    // Larger population: 1000 candidates (tests FFI overhead)
    let large_pop: Vec<Vec<f64>> = (0..1000)
        .map(|i| vec![0.5 + (i as f64 * 0.001), 21.0])
        .collect();
    let large_results = oracle.evaluate_population(large_pop, false).unwrap();
    assert_eq!(large_results.len(), 1000);
}
```

### Running Tests
```bash
# All tests
cargo test

# Specific test
cargo test test_thermal_model_energy_conservation

# With output
cargo test -- --nocapture

# Threaded vs single-threaded (for debugging race conditions)
cargo test -- --test-threads=1
```

## Contributing Guidelines

### Code Style & Quality
- **Format**: Run `cargo fmt` before every commit
- **Linting**: Address all `cargo clippy` warnings; use `#[allow(...)]` sparingly with documentation
- **Documentation**: Add doc comments to public functions/structs:
  ```rust
  /// Predicts thermal loads using the neural network surrogate.
  ///
  /// # Arguments
  /// * `current_temps` - Zone temperatures in Celsius
  ///
  /// # Returns
  /// Vector of predicted loads (W/m²) per zone
  pub fn predict_loads(&self, current_temps: &[f64]) -> Vec<f64> { ... }
  ```

### Commit Message Conventions
- **Format**: `<type>(<scope>): <subject>`
- **Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`
- **Examples**:
  - `feat(surrogate): integrate ONNX runtime session initialization`
  - `perf(engine): reduce memory allocations in solve_timesteps inner loop`
  - `test(batch-oracle): add population scaling validation`
  - `fix(physics): correct window U-value calculation units`

### Pull Request Checklist
- [ ] Tests added/updated for new functionality
- [ ] All tests pass: `cargo test`
- [ ] Code formatted: `cargo fmt`
- [ ] No clippy warnings: `cargo clippy`
- [ ] Documentation updated (doc comments, README if applicable)
- [ ] Commit messages follow convention
- [ ] PR description explains "why" not just "what"
- [ ] Root directory cleaned up (see Repository Structure & Hygiene section below)

## Repository Structure & Hygiene

### Root Folder Cleanliness
The root directory must remain clean to maintain project clarity. The **only `.md` file in root should be `README.md`**. All other Markdown documentation belongs in the `docs/` folder.

**Allowed in root:**
- `README.md` (project overview only)
- `Cargo.toml`, `pyproject.toml` (package config)
- Hidden files: `.gitignore`, `.github/`, `.pre-commit-config.yaml`, etc.
- Build artifacts: `target/`, `.venv/`, etc. (only in `.gitignore`)

**Not allowed in root:**
- `PRECOMMIT_*.md`, `PRECOMMIT_*.py` (move to `docs/` or `tmp/`)
- `PLAN.md`, `SUMMARY.md`, `CHECKLIST.md` (move to `tmp/`)
- Any temporary analysis or debug files
- Any `.md` files other than `README.md`

### Pre-Commit Cleanup
**Before committing or pushing**, remove all temporary files and scripts generated during development:

**Examples of files to clean up:**
- `PRECOMMIT_*.md`, `PRECOMMIT_*.py` (setup scripts, checklists, summaries)
- `plan.md`, `deployment-plan.md`, `*.copilotmd` (AI planning documents)
- `.azure/plan.copilotmd`, `.azure/containerization.copilotmd` (Azure deployment plans)
- `todo.md`, `checklist.md`, `notes.md` (temporary notes)
- Any files generated by tools for analysis/debugging
- Session logs or intermediate outputs from optimization runs

**Directory structure after cleanup:**
- Permanent files: `docs/`, `src/`, `tests/`, `Cargo.toml`, `README.md`
- Temporary work: Move to `tmp/` folder if it needs to persist between sessions
- Ephemeral files: Delete entirely before commit

**Pre-commit workflow:**
1. Complete feature implementation and testing
2. Run final validation: `cargo fmt && cargo clippy && cargo test`
3. **Delete all temporary files** (or move to `tmp/` if they're utilities)
4. Verify clean root: `ls -la | grep "\.md$"` should show only `README.md`
5. Stage and commit: `git add . && git commit -m "..."`

### Review Criteria
1. **Physics Correctness**: Changes to `solve_timesteps` must validate against analytical baseline
2. **Performance**: Batch processing changes must test with 1000+ population size
3. **API Stability**: PyO3 changes must maintain backward compatibility or clearly document breaking changes
4. **Test Coverage**: New public functions require corresponding unit tests

## Performance Profiling & Optimization

### Identifying Bottlenecks
```bash
# Measure time before/after changes
cargo build --release && time cargo test --release

# Profile with perf (macOS: use Instruments or `cargo flamegraph`)
# On Linux:
cargo flamegraph --bin fluxion
```

### Critical Performance Metrics
- **Per-Configuration Latency**: Single `solve_timesteps(8760)` should complete in <100ms
- **Throughput**: `BatchOracle::evaluate_population(1000)` should complete in <100ms total (100μs per config)
- **Memory**: Avoid allocations in inner loops; benchmark with `--release` profile

### Optimization Checklist
- [ ] Inner loop uses `par_iter()` only at population level
- [ ] No unnecessary clones in `solve_timesteps`
- [ ] Python boundary crossed once per `evaluate_population` call
- [ ] Release profile enabled for all benchmarks

## Development Environment Setup

### Prerequisites
- **Rust**: Install via `rustup` (latest stable + nightly for tools)
  ```bash
  rustup update
  rustup component add rustfmt clippy
  ```
- **Python**: 3.10+ required (check `pyproject.toml`)
- **maturin**: `pip install maturin`

### IDE Setup (VS Code)
- **Rust Analyzer**: `rust-lang.rust-analyzer` extension
- **Extensions**: `GitHub.Copilot`, `serayuzgur.crates`, `vadimcn.vscode-lldb`
- **Settings**:
  ```json
  {
    "[rust]": {
      "editor.formatOnSave": true,
      "editor.defaultFormatter": "rust-lang.rust-analyzer"
    }
  }
  ```

### Common Tasks
| Task | Command |
|------|---------|
| Setup local dev | `cargo build && maturin develop` |
| Run tests | `cargo test` |
| Check code quality | `cargo fmt && cargo clippy && cargo test` |
| Build wheel | `maturin build --release` |
| Quick syntax check | `cargo check` |
| Auto-fix clippy issues | `cargo clippy --fix --allow-dirty` |

## Debugging

### Rust-side Debugging
```bash
# Enable debug symbols in release build
cargo build --release --debuginfo=full

# Run with debugger (LLDB on macOS)
lldb ./target/release/fluxion
(lldb) run
```

### Python-side Debugging
```bash
# Debug Python calling Rust
python -m pdb script.py

# Print debug info from Rust (use println! in non-release builds)
RUST_LOG=debug cargo test -- --nocapture

# Check for FFI errors
python -c "import fluxion; oracle = fluxion.BatchOracle(); print(oracle.evaluate_population([[1.5, 21.0]], False))"
```

### Common Issues
- **"module not found"**: Run `maturin develop` after changes to Rust code
- **"thread panicked"**: Check physics calculations in `solve_timesteps` for NaN/Inf
- **"rayon panicked"**: Enable `--test-threads=1` to debug parallel issues sequentially

## Common Pitfalls

1. **Cloning ThermalModel in inner loops**: Use `par_iter()` at the population level, not nested parallelism
2. **Ignoring Python boundary cost**: Test with realistic population sizes (~1000+) to catch FFI overhead
3. **Hardcoding parameter semantics**: Always document the gene-to-field mapping
4. **Forgetting `use_surrogates` flag**: Tests must validate both analytical and surrogate paths
5. **Not running `cargo fmt` before commit**: Linting failures block merge
6. **Skipping tests for "small" changes**: Physics changes can have downstream effects; always test
7. **Assuming parallel performance**: Profile with release builds; debug builds hide threading issues
