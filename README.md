# Fluxion: AI-Accelerated Building Energy Engine

**Fluxion** is a next-generation Building Energy Modeling (BEM) engine. It is designed to be differentiable, quantum-ready, and exponentially faster than legacy monolithic tools by utilizing a hybrid Neuro-Symbolic architecture.

> **Status:** Fluxion is **in active development** — specifically mid-milestone on **v1.3 "Blind ASHRAE 140 Validation"** (physics-only, no calibration factors). It is **not** production-ready. Current ASHRAE 140-2023 validation pass rate is **14.3%** (see [Current Validation Status](#current-validation-status) below). Use it as a high-throughput research/oracle tool, not as a drop-in EnergyPlus replacement.

## 🏗 Architecture

Fluxion separates the "heavy lifting" of physics (CFD/Radiation) into AI surrogates, while maintaining a rigorous First-Principles thermal network for energy conservation. The thermal network, conduction solvers, and solar/ventilation models are organized as swap-point traits (`HeatConductionSolver`, `VentilationSchedule`, `ThermalModelTrait`) so that physics and AI-surrogate implementations are interchangeable. See [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`CODEBASE_MAP.md`](CODEBASE_MAP.md) for module boundaries and the full trait contracts.

## Current Validation Status

![ASHRAE 140](https://img.shields.io/badge/ASHRAE140-14.3%25%20pass-red)
![Version](https://img.shields.io/badge/status-in--development-orange)

Fluxion is **not yet ASHRAE 140-compliant**. The figures below come from the committed validation suite (generated 2026-08-07); see [`docs/ASHRAE140_RESULTS.md`](docs/ASHRAE140_RESULTS.md) for the full case-by-case breakdown and [`SCORECARD.md`](SCORECARD.md) for the consolidated, reproducible release-readiness view.

| Metric | Current | Target (release gate) | Status |
|--------|---------|-----------------------|--------|
| Pass rate (metric-level) | **14.3%** (12/84) | ≥ 60% | ❌ Fail |
| Mean Absolute Error (MAE) | **51.03%** | ≤ 50% | ❌ Fail |
| Cases fully passing | 0/18 (0.0%) | — | ❌ |
| Max single-case deviation | 470.11% | — | ℹ️ |

### v1.3 Milestone — Blind ASHRAE 140 Validation (Physics Only)

The current milestone removes all post-simulation correction factors and case-type hints, then fixes the underlying physics so the engine passes against **true** ASHRAE 140 reference values (not "calibrated for 5R1C" ranges). The milestone is structured in five phases (all currently in planning/baseline):

- **Phase A — Baseline Stripping:** Catalog and remove all correction infrastructure; measure the true physics-only baseline.
- **Phase B — Physics Fixes:** Solar distribution (ISO 13790), thermal-mass time constant, and free-floating temperature fixes across ~18 weeks.
- **Phase C — Benchmark Correction:** Replace calibrated ranges with true EnergyPlus/ESP-r/TRNSYS reference values.
- **Phase D — Blind Validation Pass:** Run the full blind suite targeting ≥80% pass.
- **Phase E — Sustained Validation:** CI gate + regression tracking to hold the pass rate as the codebase evolves.

See [`.planning/ROADMAP.md`](.planning/ROADMAP.md) and [`.planning/ASHRAE_140_BLIND_VALIDATION_PLAN.md`](.planning/ASHRAE_140_BLIND_VALIDATION_PLAN.md) for phase detail and requirements.

### Known Limitations

These are documented **structural failures** (also listed in `release_gates.yaml → validation.individual.known_failures` and `AGENTS.md`). Per `RULES.md`, the fix path is the underlying physics — **no parameter tuning** to make tests pass.

- **Baseline 600-series (low-mass):** All 6 cases FAIL. Simplified envelope model over-predicts peak loads (e.g. peak heating ~4.36 kW vs 2.80–3.80 kW reference band).
- **High-mass 900-series:** All 6 cases FAIL. Heating is over-predicted by ~**200%** due to a 5R1C/CTF thermal-mass limitation (e.g. Case 900 annual heating 5,449 kWh vs 1,170–2,040 kWh reference band).
- **Overall accuracy:** 51.03% MAE, driven by the high-mass annual-energy deviation above.
- **Peak load accuracy:** High-mass peak loads over-estimated; full peak accuracy awaits the planned gauge-solver / finite-volume work (Phase B / `gauge-solver` feature).

For the historical v0.8.0 snapshot (Peak Load & Free-Float Validation narrative), see [`docs/archive/ASHRAE140_RESULTS_v0.8.0.md`](docs/archive/ASHRAE140_RESULTS_v0.8.0.md) (archived; superseded by the current blind-validation figures above).

## 🚀 Features

- **Throughput:** ~900 configs/sec throughput in release mode via `BatchOracle` and `rayon` threading (≥150 configs/sec CI gate; see [`SCORECARD.md`](SCORECARD.md)).
- **Speed:** <100ms annual simulations via AI approximation (design target).
- **Hybrid Physics:** Hard constraints (Energy Balance) + Soft constraints (Neural Surrogates) at swap-point traits.
- **Interoperability:** Native Python SDK via `pyo3` and Node.js bindings via `napi-rs`.
- **Cross-Platform:** Supports macOS (x64 + ARM), Linux, and Windows.

### Feature Flags (default = none)

Most functionality is behind cargo feature flags; default builds skip the ONNX runtime. Notable flags:

- **Physics solvers:** `gauge-solver` (experimental/opt-in `GaugeZoneSolver` scaffolding, #2304 — does **not** replace 5R1C/9R4C; always `None`, see #2686), `debug-physics` (gates `eprintln!` in physics hot loops).
- **AI / surrog:** `ort` (alias `onnx`, ONNX inference), `cuda` (GPU inference; auto-downgrades to CPU if unavailable).
- **Acausal HVAC / fluid:** `fluid` (enables `fluxion-fluid` acausal HVAC/fluid port traits).
- **Advanced co-simulation:** `fluxion-cfd` (FFD/CFD airflow), `fluxion-city` (urban radiation), `dwave` (D-Wave quantum annealing SAPI).
- **Telemetry / concurrency:** `kafka` (rdkafka telemetry), `loom` (concurrency fuzzing; needs ~32 GB).
- **Bindings / interop:** `python-bindings`, `python-extension` (maturin wheel build, #2532), `napi-bindings`, `multi-zone`, `wiring-tracing`, `ashrae_140_v2021`, `dhat` (heap profiling).

See [`AGENTS.md`](AGENTS.md) §Toolchain Quirks for the complete, authoritative feature list and build commands.

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

The Node.js bindings provide high-performance native access to Fluxion with full TypeScript support. See [`npm/README.md`](npm/README.md) for detailed documentation.

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
- **Development**: Use the `develop` branch for active feature development and testing.
- **Pull Requests**: Create PRs against the `develop` branch (PR body must include `Closes #N` / `Fixes #N`).
- **Releases**: Merge from `develop` to `main` via a release PR (`--no-ff`). No direct pushes to `develop` or `main`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (short form) and [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) (long form) for detailed guidelines.

## 🚀 Release Process

Follow these steps to prepare and publish a new version of Fluxion.

> **Note on validation gating:** ASHRAE 140 validation is **not** currently a release-passing check — the strict ±15% annual-energy gate applies only to cases that are not documented structural failures (see [Current Validation Status](#current-validation-status)). Baseline **Case 600** and high-mass **Case 900** are excluded as known structural failures; do **not** treat "Case 900 annual energy within reference ranges" as a pre-release verification step — it is a known FAIL (5,449 kWh vs the 1,170–2,040 kWh band) being addressed by the v1.3 milestone.

### 1. Version Bump
Update the version number in both configuration files:
- **Rust**: `Cargo.toml` (`[package] version = "X.Y.Z"`)
- **Python**: `pyproject.toml` (`[project] version = "X.Y.Z"`)

### 2. Validation & Quality Check
Ensure all physics and integration tests pass before proceeding:
```bash
# Run all unit tests
cargo test --release

# Run ASHRAE 140 validation suite (informational until v1.3 lands)
cargo test --test ashrae_140_validation -- --nocapture
```
Check the current pass rate and known failures in [`docs/ASHRAE140_RESULTS.md`](docs/ASHRAE140_RESULTS.md) and [`SCORECARD.md`](SCORECARD.md). Required branch-protection checks are listed in `release_gates.yaml → ci.required_checks`.

### 3. Package Verification
Verify the crate size and structure for crates.io:
```bash
# Check package size (must be < 10MB)
# Ensure large directories like refdata/ and assets/ are excluded in Cargo.toml
cargo publish --dry-run --allow-dirty
```
Successful output should show: `Packaged X files, ~3.1MiB (632.9KiB compressed)`.

### 4. Build Python Wheels
Build the cross-platform Python wheels:
```bash
maturin build --release
```
Wheels will be generated in `target/wheels/`.

### 5. Publication
Publish to package registries (requires owner tokens):
```bash
# Publish to crates.io
cargo publish

# Publish to PyPI
twine upload target/wheels/*
```

### 6. GitHub Release
Create a new tag and release on GitHub:
```bash
gh release create vX.Y.Z --notes-file CHANGELOG.md
```

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

Fluxion uses AI surrogates to accelerate expensive physics calculations. These surrogates are trained on physics-extracted training data from the `data/training/` directory.

### Training Instructions

1.  **Install development dependencies:**
    ```bash
    pip install -r requirements-dev.txt
    ```

2.  **Train a surrogate model:**
    ```bash
    python tools/train_surrogate.py --num-samples 50000 --epochs 100
    ```

    This script extracts training data from physics simulations (or loads pre-extracted samples from `data/training/`) and trains a PyTorch neural network.

3.  **Output:**
    Trained models are saved to the `models/` directory in ONNX format (e.g., `models/surrogate.onnx`).

    **Production training source:** Pre-extracted physics samples live in `data/training/`. The training script loads from this directory (see `data/training/README.md` for the extraction pipeline).

### Configuration Options
Key arguments (see `python tools/train_surrogate.py --help` for full list):
- `--num-samples`: Number of training samples (default: 10000)
- `--epochs`: Training epochs (default: 100)
- `--hidden-dims`: Hidden layer sizes (e.g., `128 64`)
- `--output-dir`: Directory for saving results

### Integration
Once trained, the ONNX model is wired to the Rust `SurrogateManager` for inference (Issue #1285). The `SurrogateManager::load_onnx()` method loads the model and runs real ONNX inference at swap points.

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
- For validation or publication-quality results, implement a trained surrogate (ONNX) and complete the v1.3 blind-validation milestone; then `EUI` can be converted and reported in `kWh/m²/year` as described in the docs.
