# Contributing to Fluxion

Thank you for your interest in contributing to Fluxion! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check existing issues to avoid duplicates
2. Provide a clear, descriptive title
3. Include steps to reproduce (for bugs)
4. Specify your environment (OS, Rust version, Python version)
5. Add relevant labels

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `develop`
2. **Follow the development workflow** (see Development Setup)
3. **Write or update tests** for your changes
4. **Ensure all checks pass**:
   ```bash
   cargo fmt && cargo clippy && cargo test
   ```
5. **Update documentation** if needed
6. **Write a clear PR description** explaining the "why" behind your changes
7. **Clean up temporary files** before committing (see Repository Hygiene)

**Note**: All PRs should be created against the `develop` branch. The `main` branch is reserved for releases.

## Development Setup

### Prerequisites

- **Rust**: Install via `rustup` (latest stable)
  ```bash
  rustup update
  rustup component add rustfmt clippy
  ```
- **Python**: 3.10+ required
- **maturin**: `pip install maturin`

### Local Development

#### First-Time Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fluxion.git
cd fluxion

# Ensure you're on the develop branch
git checkout develop

# Install Rust toolchain
rustup update
rustup component add rustfmt clippy

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg -f

# Build Rust code
cargo build

# Build and install Python bindings
maturin develop

# Quick smoke test
python -c "import fluxion; print(fluxion.BatchOracle())"
```

#### Typical Development Iteration

```bash
# Make your changes...

# Format code
cargo fmt

# Check for linting issues
cargo clippy

# Run tests
cargo test

# Build Python bindings
maturin develop

# Smoke test
python -c "import fluxion; print(fluxion.BatchOracle())"
```

#### Quick Smoke Test

After building Fluxion, verify installation with a quick smoke test:

```bash
python -c "import fluxion; print(fluxion.BatchOracle())"
```

Expected output: `BatchOracle object` (no errors).

If you see "module not found" or import errors, run `maturin develop` to rebuild Python bindings.

### Running CI locally with `act`

- **Purpose:** Run GitHub Actions workflows locally using the `act` CLI to reproduce CI jobs (useful for fast iterations and debugging).
- **Install:** Follow `act` installation instructions: https://github.com/nektos/act#installation
- **Example (macOS on Apple Silicon / ARM):**

```bash
act -j coverage \
  -W .github/workflows/code-coverage.yml \
  --container-architecture linux/arm64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

- **Notes:**
  - Use `--container-architecture linux/arm64` on Apple Silicon (M1/M2) to match the ARM environment.
  - The `-P ubuntu-latest=...` mapping sets the docker image `act` will use for the `ubuntu-latest` runner label; the `catthehacker/ubuntu:act-latest` image is commonly used and includes required tooling.
  - If you run into permission or sandboxing issues, ensure Docker Desktop is running and you have enough resources allocated.
  - For other jobs replace `-j coverage` with the job id from the workflow file or omit `-j` to run the default workflow.

#### Running Other Jobs

- **Run a different job by id:** find the job id under `jobs:` in the workflow YAML and pass it with `-j`:

```bash
act -j lint \
  -W .github/workflows/ci.yml \
  --container-architecture linux/arm64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

- **Run the entire workflow (all jobs):** omit `-j` and specify the workflow file:

```bash
act -W .github/workflows/code-coverage.yml \
  --container-architecture linux/arm64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

- **Provide an event payload:** simulate a specific event (e.g., `push`) with `-e` and a JSON file:

```bash
act -e tests/fixtures/push-event.json -W .github/workflows/ci.yml
```

- **Pass secrets and environment variables:** use `-s NAME=VALUE` for secrets or `--env-file .env` for environment variables:

```bash
act -j coverage -s GITHUB_TOKEN=ghp_xxx --env-file .secrets.env -W .github/workflows/code-coverage.yml
```

#### Troubleshooting `act`

- **Docker not running / connection errors:** ensure Docker Desktop is running and responsive. Restart Docker if mounts or networking fail.
- **Image/platform mismatches on Apple Silicon:** prefer `--container-architecture linux/arm64` and an ARM-compatible image (`-P ubuntu-latest=catthehacker/ubuntu:act-latest`). If an image is missing for ARM, pre-pull or choose an image that supports `linux/arm64`.
- **Permission / volume mount issues on macOS:** some actions rely on bind mounts that require Docker permissions — try enabling `--privileged` (note: increases privileges) or adjust Docker Desktop file sharing settings.
- **Slow or resource-heavy workflows:** increase Docker Desktop resources (CPUs, memory) or limit concurrent jobs. For heavy builds prefer running in the real CI runner.
- **Missing secrets or auth failures:** `act` does not automatically provide GitHub secrets. Supply required secrets with `-s` or `--env-file`, or create an `.actrc`/`.secrets.env` used locally.
- **Actions that rely on GitHub-hosted services (e.g., `actions/cache`, `setup-remote-docker`) may behave differently locally:** treat `act` as a debugging tool — final verification should still run in GitHub Actions.
- **Verbose logs:** add `-v` or `--verbose` to `act` to see more output and help diagnose failures.

If you hit an error you can't resolve locally, capture the `act` output and open an issue or include it in your PR so maintainers can reproduce it.


## Code Style & Quality

### Formatting
- **Rust**: Run `cargo fmt` before committing
- **Python**: Follow PEP 8 style guide

### Linting
- Address all `cargo clippy` warnings
- Use `#[allow(...)]` sparingly with documentation

### Documentation
- Add doc comments to public functions/structs:
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

**Documentation Guidelines:**
- All public APIs must have doc comments
- Include examples in doc comments for complex functions
- Use Markdown formatting in doc comments
- Reference related documentation where appropriate
- Update `docs/API_REFERENCE.md` for public Python APIs
- Update `docs/ARCHITECTURE.md` for architecture changes

**Documentation Files:**
- `README.md`: High-level project overview and installation guide
- `docs/API_REFERENCE.md`: Comprehensive API reference with examples
- `docs/ARCHITECTURE.md`: Architecture deep dive
- `docs/CONTRIBUTING.md`: This guide
- `docs/KNOWN_LIMITATIONS.md`: Known model limitations and accuracy constraints
- `docs/tutorials/`: Step-by-step tutorials (planned)
- `docs/ASHRAE140_RESULTS.md`: Validation results and analysis

## Testing Strategy

### Test Types

**Unit Tests:**
- Location: In the same file as implementation using `#[cfg(test)]` modules
- Purpose: Test individual functions and methods in isolation
- Example: Thermal model physics calculations, parameter validation

**Integration Tests:**
- Location: `tests/` directory
- Purpose: Test component interactions and end-to-end workflows
- Example: ASHRAE 140 validation, BatchOracle population evaluation

**Property-Based Tests:**
- Location: `tests/` directory using `proptest` crate
- Purpose: Test invariants across random inputs
- Example: Energy conservation, thermal stability

**Performance Regression Tests:**
- Location: `benches/` directory using `criterion` crate
- Purpose: Detect performance regressions in critical paths
- Example: Single config latency, population throughput

### Writing Tests

Place unit tests in the same file as implementation using `#[cfg(test)]` modules:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_model_energy_conservation() {
        let mut model = ThermalModel::new(10);
        // Test logic...
        assert!(condition, "failure message");
    }
}
```

**Best Practices:**
- Use descriptive test names that explain what is being tested
- Include assertions with clear failure messages
- Test edge cases (boundary conditions, invalid inputs)
- Use deterministic RNG seeding for tests involving randomness

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_thermal_model_energy_conservation

# With output (useful for debugging)
cargo test -- --nocapture

# Single-threaded (for debugging race conditions)
cargo test -- --test-threads=1

# Run tests in release mode (faster for large test suites)
cargo test --release

# Run specific module tests
cargo test tests::thermal_model
```

### ASHRAE 140 Validation

Fluxion includes comprehensive ASHRAE 140 validation to ensure physics correctness:

```bash
# Run full ASHRAE 140 validation suite
fluxion validate --all

# Run specific case
fluxion validate --case 900

# Run with detailed output
fluxion validate --all --verbose
```

**Validation Status:**
- Current pass rate: 18/18 cases (100%)
- See `docs/ASHRAE140_RESULTS.md` for detailed results
- See `docs/KNOWN_LIMITATIONS.md` for known 5R1C model limitations

### Deterministic Testing

Fluxion uses seeded RNG for deterministic test execution:

```rust
use rand::SeedableRng;
use rand::rngs::StdRng;

// Create seeded RNG for reproducible tests
let mut rng = StdRng::seed_from_u64(42);
let random_value = rng.gen::<f64>();
```

**Purpose:**
- Eliminate flaky tests caused by random number generation
- Ensure consistent test results across runs
- Enable reproducible debugging

**Verification:**
- Run tests multiple times to verify determinism
- Use `--test-threads=1` to catch race conditions

#### Cross-Platform Determinism CI Gate (Issue #1351)

A pull-request to `main` **cannot be merged** if the cross-platform
floating-point determinism check fails. The gate is wired in two
places:

1. `Cross-Platform Determinism CI` workflow
   (`.github/workflows/determinism_check.yml`) — runs Case 900
   determinism on `ubuntu-latest`, `windows-latest`, and
   `macos-latest` and compares the SHA-256 of the extracted values
   to confirm the simulation produces byte-identical output on all
   three operating systems.
2. `ASHRAE 140 CI Gate` workflow
   (`.github/workflows/ashrae_validation.yml`) — exposes a
   `workflow_run` listener job
   (`Fluxion Determinism Gate (Issue #1351)`) that fails the
   PR's checks list and posts a comment with the upstream run URL
   if the `Cross-Platform Determinism CI` workflow concluded
   `failure` / `cancelled` / `timed_out` for the same SHA. This
   listener is the canonical non-matrix required check that
   branch protection references.

The required status check list lives in
`release_gates.yaml::ci.required_checks` and is mirrored in
[`.github/BRANCH_PROTECTION.md`](../.github/BRANCH_PROTECTION.md)
alongside the manual admin steps to add the check to GitHub
branch protection.

**Common ways a PR can trip the gate (issue #1297 fix list):**

- A new `HashMap` / `HashSet` is used where a deterministic
  `BTreeMap` is required (non-deterministic iteration order across
  platforms).
- A non-deterministic `f32` reduction path (SIMD reordering,
  parallel reduction with non-associative orderings) is added
  without an explicit `BTreeMap`/sorted-iterator wrapper.
- A new dependency pulls in non-portable FP code. Rebuild against
  `--release --features wiring-tracing` to reproduce.

**Local repro recipe** (matches the upstream workflow's RUSTFLAGS):

```bash
RUSTFLAGS="-C opt-level=3 -C debug-assertions=no" \
  cargo test --test case_900_determinism --release -- --nocapture
```

The expected canonical hash for the three-OS matrix is published in
the `Determinism Check (ubuntu-latest)` step summary on `main`.

### Coverage Measurement

Fluxion uses `cargo-llvm-cov` for coverage measurement:

```bash
# Install llvm-cov
cargo install cargo-llvm-cov

# Generate coverage report
cargo llvm-cov --html

# Open report in browser
open target/llvm-cov/html/index.html
```

**Current Coverage:**
- Overall coverage: 69.36% (Phase 10 baseline)
- Target: >80% coverage
- See `docs/ASHRAE140_RESULTS.md` for detailed coverage analysis

### Physics Validation

- Use `Model::simulate(1, use_surrogates=false)` for single-year analytical validation
- Compare against baseline energies documented in `docs/Fluxion_PRD.md`
- Test batch operations with realistic population sizes (1000+)

## Commit Message Convention

Use semantic commit messages:

```
<type>(<scope>): <subject>

<body (optional)>
```

**Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`

**Examples**:
- `feat(surrogate): integrate ONNX runtime session initialization`
- `perf(engine): reduce memory allocations in solve_timesteps`
- `test(batch-oracle): add population scaling validation`
- `fix(physics): correct window U-value calculation units`

## Repository Hygiene

### Before Committing

1. **Delete temporary files** generated during development
2. **Keep root directory clean** — only `README.md` should be in root (besides config files)
3. **Move development artifacts to `tmp/`** if they need to persist

Examples of files to clean up:
- `PRECOMMIT_*.md`, planning documents
- `.azure/plan.copilotmd`, deployment plans
- Temporary scripts or debug files

### Pre-commit Checklist

- [ ] Code formatted: `cargo fmt`
- [ ] No clippy warnings: `cargo clippy`
- [ ] All tests pass: `cargo test`
- [ ] Temporary files removed or moved to `tmp/`
- [ ] Root directory clean (only `README.md` and config files)
- [ ] Commit message follows convention
- [ ] Documentation updated (if applicable)

## Pull Request Checklist

- [ ] Tests added/updated for new functionality
- [ ] All tests pass: `cargo test`
- [ ] Code formatted: `cargo fmt`
- [ ] No clippy warnings: `cargo clippy`
- [ ] Documentation updated (doc comments, README if applicable)
- [ ] Commit messages follow convention
- [ ] PR description explains "why" not just "what"
- [ ] Temporary files cleaned up
- [ ] No unnecessary `.md` files in root

## Architecture Overview

See `docs/Fluxion_PRD.md` for:
- System architecture (BatchOracle pattern)
- Physics engine details (ThermalModel)
- AI surrogate integration (SurrogateManager)
- API reference

## Parameter Vector Semantics

The parameter vector format is **critical for external APIs** (D-Wave, GA libraries, and other optimization frameworks). This section documents the complete specification of design variables, their bounds, and how they map to the thermal model.

### Population Format

When using `BatchOracle::evaluate_population()` or `Model::apply_parameters()`, parameters are passed as a vector of floating-point values:

```python
import fluxion

# Parameter vector: [window_u_value, hvac_setpoint]
params = [2.0, 21.0]

# Use with BatchOracle
oracle = fluxion.BatchOracle()
population = [params, [1.5, 20.0], [2.5, 22.0]]
results = oracle.evaluate_population(population, use_surrogates=False)

# Use with Model
model = fluxion.Model()
model.apply_parameters(params)
total_energy = model.simulate(1, use_surrogates=False)
```

### Element Mapping

#### Element 0: Window U-value
- **Range**: 0.1 – 5.0 W/m²K
- **Definition**: Thermal transmittance of glazing
- **Physical Meaning**: Rate of heat transfer through windows per degree temperature difference
- **Lower Values**: Better insulation, less heat transfer, reduced HVAC load
- **Higher Values**: Poorer insulation, more heat transfer, increased HVAC load
- **Constants**:
  - `MIN_U_VALUE = 0.1` (high-performance triple-glazed windows)
  - `MAX_U_VALUE = 5.0` (single-pane windows)
- **Model Impact**: Updates `h_tr_w` (window conductance) and `h_tr_em` (exterior transmission)

```rust
// In ThermalModel::apply_parameters()
self.window_u_value = params[0];

// Updates 5R1C conductances
self.h_tr_w = VectorField::repeat(window_area * self.window_u_value);
self.h_tr_em = VectorField::repeat(mass_coupling * self.window_u_value);
```

#### Element 1: HVAC Setpoint
- **Range**: 15 – 30°C
- **Definition**: Target indoor temperature maintained by HVAC system
- **Physical Meaning**: Comfort setpoint that triggers heating/cooling
- **Lower Values**: More heating demand, less cooling demand
- **Higher Values**: Less heating demand, more cooling demand
- **Constants**:
  - `MIN_SETPOINT = 15.0` (low setpoint for aggressive cooling)
  - `MAX_SETPOINT = 30.0` (high setpoint for aggressive heating)
- **Model Impact**: Determines HVAC activation threshold in `solve_timesteps()`

```rust
// In ThermalModel::apply_parameters()
self.hvac_setpoint = params[1];

// Used in solve_timesteps() to determine HVAC demand
if self.temperatures[step] < self.hvac_setpoint {
    // Heating needed
} else if self.temperatures[step] > self.hvac_setpoint {
    // Cooling needed
}
```

#### Future Elements (Planned)
The parameter vector is designed to be extensible. Future design variables will be added as additional elements:

- **Element 2**: Thermal mass capacitance (J/K) - Building material thermal inertia
- **Element 3**: Infiltration rate (1/hour) - Air exchange rate with outdoors
- **Element 4**: Solar Heat Gain Coefficient (SHGC) - Fraction of solar radiation transmitted through windows
- **Element 5**: Lighting power density (W/m²) - Internal heat gains from lighting

**Note**: Adding new elements requires:
1. Updating `MIN_*` and `MAX_*` constants
2. Extending `apply_parameters()` to map element to model field
3. Updating physics in `solve_timesteps()` to use the new parameter
4. Documenting the new element in this section
5. Testing via `BatchOracle::evaluate_population()` with sample populations

### Design Variable Documentation

| Design Variable | Element Index | Range | Unit | Impact |
|-----------------|---------------|-------|------|--------|
| Window U-value | 0 | 0.1 - 5.0 | W/m²K | Heat transfer through windows, HVAC load |
| HVAC Setpoint | 1 | 15 - 30 | °C | Heating/cooling demand, comfort |
| Thermal Mass | 2 (future) | TBD | J/K | Thermal inertia, temperature lag |
| Infiltration Rate | 3 (future) | TBD | 1/h | Ventilation heat loss/gain |
| SHGC | 4 (future) | 0 - 1 | - | Solar heat gain through windows |
| Lighting Density | 5 (future) | TBD | W/m² | Internal heat gains |

### Parameter Bounds and Validation

Fluxion enforces parameter bounds to ensure physically meaningful simulations.

#### Constants
```rust
// Current bounds (v0.3)
const MIN_U_VALUE: f64 = 0.1;
const MAX_U_VALUE: f64 = 5.0;
const MIN_SETPOINT: f64 = 15.0;
const MAX_SETPOINT: f64 = 30.0;
```

#### Validation Enforcement

**Python API:**
```python
import fluxion

oracle = fluxion.BatchOracle()

# validate_parameters() returns NaN for invalid configs
invalid_params = [0.05, 21.0]  # U-value too low
result = oracle.validate_parameters(invalid_params)
# Returns NaN with error message: "Window U-value 0.05 outside range [0.1, 5.0] W/m²K"

# Valid parameters return energy value
valid_params = [2.0, 21.0]
result = oracle.validate_parameters(valid_params)
# Returns: True or actual energy value
```

**Rust API:**
```rust
use fluxion::sim::engine::ValidationError;

// validate_parameters() returns Result<(), ValidationError>
let params = vec![2.0, 21.0];
match model.validate_parameters(&params) {
    Ok(_) => println!("Parameters valid"),
    Err(e) => println!("Validation error: {}", e),
}
```

#### Error Messages

When validation fails, detailed error messages help developers understand the issue:

- **Window U-value too low**: `Window U-value 0.05 outside range [0.1, 5.0] W/m²K`
- **Window U-value too high**: `Window U-value 6.0 outside range [0.1, 5.0] W/m²K`
- **HVAC setpoint too low**: `HVAC setpoint 10.0°C outside range [15, 30]°C`
- **HVAC setpoint too high**: `HVAC setpoint 35.0°C outside range [15, 30]°C`
- **Parameter count mismatch**: `Expected 2 parameters, got 3`
- **NaN detected**: `Parameter contains NaN or Inf value`

### Programmatic Access to Bounds

#### get_parameter_bounds() Method

Both `BatchOracle` and `Model` classes provide programmatic access to parameter bounds:

```python
import fluxion

oracle = fluxion.BatchOracle()
bounds = oracle.get_parameter_bounds()

# Returns: {"window_u_value": [0.1, 5.0], "hvac_setpoint": [15.0, 30.0]}
print(f"Window U-value bounds: {bounds['window_u_value']}")
print(f"HVAC setpoint bounds: {bounds['hvac_setpoint']}")
```

#### ParameterBounds Struct (Rust)

For Rust code, use the `ParameterBounds` struct:

```rust
use fluxion::api::parameters::ParameterBounds;

// Get bounds for all parameters
let bounds = ParameterBounds::get_bounds();
// Returns: Vec<(String, f64, f64)> with (name, min, max)

// Get bounds for specific parameter
let u_value_bounds = ParameterBounds::get_bounds_for("window_u_value");
// Returns: Some((0.1, 5.0))
```

### External Integration Points

The parameter vector format is designed for easy integration with external optimization frameworks.

#### D-Wave Quantum Annealer
```python
from dwave.system import DWaveSampler, EmbeddingComposite

# Define binary variables (discretized parameter space)
# U-value: 0.1 - 5.0 in 0.1 increments (50 values)
# Setpoint: 15 - 30 in 0.5°C increments (30 values)
# Total: 1500 possible combinations

# Use Fluxion BatchOracle for evaluation
import fluxion
oracle = fluxion.BatchOracle()

# Map quantum annealer solution to parameter vector
params = [u_value, setpoint]
energy = oracle.evaluate_population([params], use_surrogates=False)
```

#### Genetic Algorithm Libraries
```python
import numpy as np
from deap import base, creator, tools
import fluxion

# Define individual: parameter vector [u_value, setpoint]
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_u", np.random.uniform, 0.1, 5.0)
toolbox.register("attr_setpoint", np.random.uniform, 15.0, 30.0)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_u, toolbox.attr_setpoint), n=1)

# Use Fluxion BatchOracle for fitness evaluation
oracle = fluxion.BatchOracle()

def evaluate(individual):
    params = individual
    energy = oracle.evaluate_population([params], use_surrogates=False)[0]
    return (energy,)

toolbox.register("evaluate", evaluate)
```

### Cross-References

- **Parameter Vector Semantics (CLAUDE.md)**: See "Parameter Vector Semantics" section for implementation details
- **API Reference**: See `docs/API_REFERENCE.md` for BatchOracle and Model method documentation
- **Tutorial**: See `docs/tutorials/extending_fluxion.md` for a complete working example with custom parameters

### Adding New Design Variables

When adding a new design variable to the parameter vector, follow this checklist:

- [ ] Define `MIN_*` and `MAX_*` constants in `src/sim/engine.rs`
- [ ] Add field to `ThermalModel` struct (e.g., `thermal_mass: f64`)
- [ ] Extend `apply_parameters()` to map new element to field
- [ ] Update physics in `solve_timesteps()` to use the new parameter
- [ ] Add validation logic in `validate_parameters()`
- [ ] Update `ParameterBounds::get_bounds()` to include new parameter
- [ ] Update this section in CONTRIBUTING.md with element mapping and documentation
- [ ] Test via `BatchOracle::evaluate_population()` with sample populations
- [ ] Add example to tutorial showing new parameter usage

## Training AI Surrogates

### Development Workflow
Surrogate model development is an iterative process. The typical workflow is:
1.  **Generate synthetic data**: Use the analytical `fluxion` model (or the standalone generator in `tools/train_surrogate.py`) to create ground truth data.
2.  **Train neural network**: Run `train_surrogate.py` to train a PyTorch model.
3.  **Validate**: Check metrics (MAE, R²) against the analytical model on a held-out test set.
4.  **Export to ONNX**: The script automatically exports the best model.
5.  **Integrate**: Move the ONNX file to the appropriate location for the Rust `SurrogateManager`.

### Testing Surrogates
- **Accuracy**: Ensure MAE is within acceptable bounds (e.g., <5% error).
- **Speed**: Verify that inference speed meets performance targets (<100ms for 8760 timesteps).
- **Test Suite**: Add surrogate tests to the test suite to prevent regression.

### Model Versioning
- Track model versions and training configurations (e.g., in `assets/model_metrics.json`).
- **Do not commit large model files** to git. Use the `models/` directory (which is gitignored) or a separate model registry.

### Integration Guidelines
- Models must be in ONNX format.
- The Rust `SurrogateManager` (`src/ai/surrogate.rs`) is responsible for loading and running the model.
- Ensure the ONNX model's input/output shapes match what the Rust code expects.

### Performance Benchmarking
- Benchmark surrogate performance against the critical metrics mentioned in "Performance Considerations".
- Ensure that enabling surrogates provides the expected throughput boost (target: 10,000+ configs/sec).

## Performance Considerations

### Critical Metrics
- **Per-configuration latency**: <100ms for single `solve_timesteps(8760)`
- **Throughput**: <100ms total for `evaluate_population(1000)`

### Optimization Guidelines
- Use `rayon::par_iter()` only at population level, not nested
- Minimize Python-Rust boundary crossings
- Avoid allocations in inner loops
- Test with `--release` profile

## Getting Help

- Check existing documentation in `docs/`
- Review the Copilot instructions file for architecture details
- Ask questions in PR comments or open a discussion issue

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for contributing to Fluxion!
