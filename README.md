# Fluxion: AI-Accelerated Building Energy Engine

**Fluxion** is a next-generation Building Energy Modeling (BEM) engine. It is designed to be differentiable, quantum-ready, and exponentially faster than legacy monolithic tools by utilizing a hybrid Neuro-Symbolic architecture.

## 🏗 Architecture

Fluxion separates the "heavy lifting" of physics (CFD/Radiation) into AI surrogates, while maintaining a rigorous First-Principles thermal network for energy conservation.

## 🚀 Features

  * **Throughput**: Evaluates **10,000+ configurations/sec** via `BatchOracle` and `rayon` threading.
  * **Speed**: <100ms annual simulations via AI approximation.
  * **Hybrid Physics**: Hard constraints (Energy Balance) + Soft constraints (Neural Surrogates).
  * **Interoperability**: Native Python SDK via `pyo3`.

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
- **Development**: Use the `develop` branch for active feature development and testing
- **Pull Requests**: Create PRs against the `develop` branch
- **Releases**: Merge from `develop` to `main` for official releases

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

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
