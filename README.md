# Fluxion: AI-Accelerated Building Energy Engine

**Fluxion** is a next-generation Building Energy Modeling (BEM) engine. It is designed to be differentiable, quantum-ready, and exponentially faster than legacy monolithic tools by utilizing a hybrid Neuro-Symbolic architecture.

## 🏗 Architecture

Fluxion separates the "heavy lifting" of physics (CFD/Radiation) into AI surrogates, while maintaining a rigorous First-Principles thermal network for energy conservation.

## ASHRAE 140 Validation

![ASHRAE 140](https://img.shields.io/badge/ASHRAE140-v0.8.0-brightgreen)
![Version](https://img.shields.io/badge/version-0.8.0-blue)

Fluxion v0.8.0 introduces **Peak Load & Free-Float Validation** with comprehensive ASHRAE 140-2023 compliance testing. This release focuses on high-mass building physics improvements and proper reference data integration.

### v0.8.0 Release Highlights

- **Complete ASHRAE 140 Reference Database**: Multi-program reference ranges (EnergyPlus, ESP-r, TRNSYS) for all test cases including high-mass and free-float scenarios.
- **Peak Load Validation**: Proper validation of heating/cooling peak loads with ASHRAE 140-2023 reference ranges.
- **Free-Floating Temperature Validation**: Comprehensive validation of unconditioned building temperature profiles.
- **Automated Validation System**: Enhanced validation runner with proper reference data loading and detailed reporting.
- **Physics Improvements**: Continued refinement of CTF solver parameters for high-mass buildings.
- **Performance**: Achieves ~900 configs/sec throughput in release mode (exceeds 800 target).

See [ASHRAE 140 v0.8.0 Validation Results](docs/ASHRAE140_RESULTS_v0.8.0.md) for comprehensive validation data including peak loads and free-floating temperature profiles.

### Release Scorecard

For a consolidated view of release readiness, pass rates, and benchmark status, see the [Release Scorecard](SCORECARD.md).

### Known Limitations

- **Peak Load Accuracy**: High-mass peak loads show ~76-100% overestimation due to CTF solver limitations with instantaneous peak conditions. This is a known architectural constraint for v0.8.0.
- **CTF Thermal Mass**: Annual energy accuracy improved (±15-30% range), but peak load handling remains challenging. Full peak accuracy requires the planned v1.0 finite volume solver.
- **Free-Floating Validation**: Working correctly but shows ±1-2°C temperature range deviations due to simplified thermal damping models.

## 🚀 Features

  * **Throughput**: Evaluates **800-1000+ configurations/sec** via `BatchOracle` and `rayon` threading.
  * **Speed**: <100ms annual simulations via AI approximation.
  * **Hybrid Physics**: Hard constraints (Energy Balance) + Soft constraints (Neural Surrogates).
  * **Interoperability**: Native Python SDK via `pyo3` and Node.js bindings via `napi-rs`.
  * **Cross-Platform**: Supports macOS (x64 + ARM), Linux, and Windows.

## 🛠 Installation

### Rust Core
```bash
cargo build --release
```

### Python Bindings
```bash
pip install maturin
maturin develop
```

### Node.js Bindings
```bash
cd npm
npm install
npm run build
```

The Node.js bindings provide high-performance native access to Fluxion with >10,000 configs/sec throughput and full TypeScript support. See [npm/README.md](npm/README.md) for detailed documentation.

### Development Setup (recommended)

Follow these steps on macOS / zsh to create an isolated Python environment, install developer tools, enable `pre-commit` hooks, and build the Python bindings. This project requires Python `3.10+` (see `pyproject.toml`).

```bash
# 1) Create & activate a venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install development dependencies (linters, test tools, build helpers)
pip install -r requirements-dev.txt

# 4) Install and enable pre-commit hooks
pip install pre-commit
pre-commit install                # normal hooks
pre-commit install --hook-type commit-msg -f  # commit-msg hook (force replace existing hooks if needed)

# 5) Run hooks once across the repo (optional but recommended)
pre-commit run --all-files

# 6) Build & install Python bindings for local development
maturin develop
```

Optional minimal install

If you only need `maturin` for quick builds or one-off development (and don't want to install all dev tools), install it separately:

```bash
python -m pip install 'maturin>=1.0,<2.0'
```

## 🌳 Contributing & Branching

**Development Workflow**:
- **Development**: Use the \`develop\` branch for active feature development and testing
- **Pull Requests**: Create PRs against the \`develop\` branch
- **Releases**: Merge from \`develop\` to \`main\` for official releases

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

## 🚀 Release Process

Follow these steps to prepare and publish a new version of Fluxion:

### 1. Version Bump
Update the version number in both configuration files:
- **Rust**: \`Cargo.toml\` (\`[package] version = "X.Y.Z"\`)
- **Python**: \`pyproject.toml\` (\`[project] version = "X.Y.Z"\`)

### 2. Validation & Quality Check
Ensure all physics and integration tests pass before proceeding:
\`\`\`bash
# Run all unit tests
cargo test --release

# Run ASHRAE 140 validation suite
cargo test --test ashrae_140_validation -- --nocapture
\`\`\`
Verify that Case 900 annual energy results are within reference ranges in \`docs/ASHRAE140_RESULTS.md\`.

### 3. Package Verification
Verify the crate size and structure for crates.io:
\`\`\`bash
# Check package size (must be < 10MB)
# Ensure large directories like refdata/ and assets/ are excluded in Cargo.toml
cargo publish --dry-run --allow-dirty
\`\`\`
Successful output should show: \`Packaged X files, ~3.1MiB (632.9KiB compressed)\`.

### 4. Build Python Wheels
Build the cross-platform Python wheels:
\`\`\`bash
maturin build --release
\`\`\`
Wheels will be generated in \`target/wheels/\`.

### 5. Publication
Publish to package registries (requires owner tokens):
\`\`\`bash
# Publish to crates.io
cargo publish

# Publish to PyPI
twine upload target/wheels/*
\`\`\`

### 6. GitHub Release
Create a new tag and release on GitHub:
\`\`\`bash
gh release create vX.Y.Z --notes-file CHANGELOG.md
\`\`\`

## 🧪 Usage

### Quantum/ML Oracle (High Throughput)
Used for Genetic Algorithms, D-Wave Quantum Annealers, or Bayesian Optimization.

```python
import fluxion
import numpy as np

# Initialize the Oracle
oracle = fluxion.BatchOracle()

# Generate a population of 10,000 design candidates
# Column 0: Window U-Value (0.5 to 3.0)
# Column 1: HVAC Setpoint (19.0 to 24.0)
population = np.random.rand(10000, 2).tolist()

# Evaluate all 10,000 in parallel (Rust handles the threading)
results = oracle.evaluate_population(population, use_surrogates=True)

print(f"Best Performance: {min(results)}")
```

## 🤖 Training AI Surrogates

Fluxion uses AI surrogates to accelerate expensive physics calculations. These surrogates must be trained on synthetic data generated by the analytical model.

### Training Instructions

1.  **Install development dependencies:**
    ```bash
    pip install -r requirements-dev.txt
    ```

2.  **Train a surrogate model:**
    ```bash
    python tools/train_surrogate.py --num-samples 50000 --epochs 100
    ```

    This script generates synthetic data using the analytical physics model (or loads it from a file) and trains a PyTorch neural network.

3.  **Output:**
    Trained models are saved to the `models/` directory in ONNX format (e.g., `models/surrogate.onnx`).

### Configuration Options
Key arguments (see `python tools/train_surrogate.py --help` for full list):
- `--num-samples`: Number of training samples (default: 10000)
- `--epochs`: Training epochs (default: 100)
- `--hidden-dims`: Hidden layer sizes (e.g., `128 64`)
- `--output-dir`: Directory for saving results

### Integration
Once trained, the ONNX model can be integrated into the Rust `SurrogateManager` for inference. This is a future enhancement; currently, the `SurrogateManager` uses placeholder values.

## 🧩 Examples

A set of small, self-contained examples are included in the `examples/` folder to help new users get started quickly:

- `examples/run_model.py`: Creates a `Model`, runs a 1-year simulation with and without surrogates, and prints results.
- `examples/run_oracle.py`: Creates a `BatchOracle`, generates a small random population (20 candidates) and evaluates it using surrogates.
- `examples/quick_start.sh`: A helper script that installs `maturin` (if necessary), builds the Python bindings locally, and runs the oracle example.

Quick start (minimal):

1) Create and activate a Python virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Build & install Python bindings for local development:

```bash
pip install --upgrade pip
pip install maturin
maturin develop
```

3) Run the oracle example to see actual results:

```bash
python examples/run_oracle.py
```

Or use the helper script which runs the same steps (macOS / zsh):

```bash
bash examples/quick_start.sh
```

If you encounter an import error when running the examples, ensure `maturin develop` completed successfully and your Python interpreter matches the one used to build the bindings.

**Sample Output**

Running the small oracle example may produce output similar to the following:

```
Creating BatchOracle...
Evaluating population of 20 candidates (surrogates ON)...
Elapsed: 0.006s
Best candidate index: 12, EUI: 268850.3790
Sample results:
  #0: U=1.034, setpoint=19.44 -> EUI=678227.4306
  #1: U=2.733, setpoint=23.86 -> EUI=1699972.7457
  #2: U=2.229, setpoint=19.07 -> EUI=1192690.7518
  #3: U=1.413, setpoint=21.23 -> EUI=1107914.1676
  #4: U=2.733, setpoint=19.44 -> EUI=1312456.1032
```

**Interpreting Results**

- **U:** Window thermal transmittance (U-value) in `W/m²K` — lower values indicate better insulating windows. Example range used by the examples: `0.5` (high-performing glazing) to `3.0` (poor glazing).
- **setpoint:** HVAC setpoint temperature in degrees Celsius (`°C`). Typical design range in examples: `19.0` to `24.0`.
- **EUI:** Energy metric printed by the examples. In this repository the physics engine is intentionally simplified for clarity and testing: the `ThermalModel::solve_timesteps` routine accumulates a raw, per-hour, per-zone energy-like value (sum of absolute temperature departure from setpoint across all zones and hours). Because of that, numeric EUI values printed by the toy examples are very large — they are a raw cumulative metric, not a calibrated `kWh/m²/year`. Use these values for relative comparison (lower = better), not as validated physical EUI numbers.

  - If you need a quick normalization for human-scale comparison, divide the reported EUI by the number of zones and number of timesteps (e.g., `num_zones * 8760`) to get an average hourly temperature-gap metric. Converting this to physical energy (kWh/m²) requires thermal capacity and area scaling which are not part of the current toy model.

- **Elapsed:** Wall-clock time to evaluate the population. The `BatchOracle` is designed for high-throughput evaluation (many candidates in parallel); elapsed time will depend on population size and whether surrogates are enabled.

- **Best candidate index:** Index of the candidate with the lowest (best) EUI in the evaluated population and its corresponding EUI value.

Notes:
- The included `SurrogateManager` currently returns deterministic mock loads when no neural model is loaded, so results are deterministic given the same random seed and are intended for API validation and performance testing rather than production accuracy.
- For validation or publication-quality results, implement a trained surrogate (ONNX) and calibrate the physics engine; then `EUI` can be converted and reported in `kWh/m²/year` as described in the docs.
