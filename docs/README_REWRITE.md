# Fluxion

**A fast, open-source Building Energy Modeling (BEM) engine** — written in Rust, with Python and Node.js bindings. Evaluates 800–1,000+ building configurations per second. ASHRAE 140 validation in progress.

[![ASHRAE 140](https://img.shields.io/badge/ASHRAE140-v0.8.0--in--progress-yellow)](docs/ASHRAE140_RESULTS_v0.8.0.md)
[![Version](https://img.shields.io/badge/version-0.8.0-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What is Fluxion?

Legacy BEM tools (EnergyPlus, TRNSYS) are accurate but slow — a single simulation can take seconds, making large parametric sweeps or optimization loops impractical.

Fluxion solves this with a **hybrid neuro-symbolic architecture**: a rigorous first-principles thermal network for energy conservation, combined with AI surrogates that replace expensive CFD/radiation computations. The result is physically grounded simulation at machine-learning speeds.

**Use Fluxion when you need to:**
- Evaluate thousands of building design variants in an optimization loop (genetic algorithms, Bayesian optimization, quantum annealing)
- Develop and test against ASHRAE 140 / BESTEST — validation is active and improving each release
- Build a fast physics oracle for surrogate ML model training
- Embed BEM simulation in a Python or Node.js application

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Throughput (BatchOracle, release mode) | **800–1,000+ configs/sec** |
| Single annual simulation (surrogate mode) | **< 100 ms** |
| ASHRAE 140 case coverage | 600-series + high-mass + free-float |
| ASHRAE 140 pass rate (v0.8.0) | ~36% (physics fixes in progress; target 100% at v1.0) |
| Platforms | macOS (x64, ARM), Linux, Windows |

---

## Installation

### Python (recommended for most users)

```bash
pip install fluxion
```

Pre-built wheels are available for macOS, Linux, and Windows (Python 3.10+). No Rust toolchain required.

### Node.js / TypeScript

```bash
cd npm
npm install
npm run build
```

Full TypeScript support; >10,000 configs/sec throughput. See [`npm/README.md`](npm/README.md) for details.

### From source (Rust)

```bash
git clone https://github.com/anchapin/fluxion.git
cd fluxion
cargo build --release
```

For Python bindings from source:

```bash
pip install maturin
maturin develop
```

---

## Quick Start

### 1. Single simulation (Python)

```python
from fluxion import Model

model = Model(num_zones=1)
eui = model.simulate(years=1, use_surrogates=False)
print(f"EUI: {eui:.2f}")
```

### 2. High-throughput batch evaluation

```python
import fluxion
import numpy as np

oracle = fluxion.BatchOracle()

# 10,000 design candidates: [window_u_value, hvac_setpoint]
population = np.random.rand(10000, 2).tolist()

results = oracle.evaluate_population(population, use_surrogates=True)
print(f"Best candidate EUI: {min(results):.2f}")
print(f"Evaluated {len(results)} designs")
```

### 3. Using a trained surrogate

```python
from fluxion import Model

model = Model(num_zones=1)
model.load_surrogate("models/surrogate.onnx")  # ONNX format
eui = model.simulate(years=1, use_surrogates=True)
```

### 4. Command line

```bash
fluxion run config.json
fluxion run config.json --surrogate models/surrogate.onnx
fluxion serve   # REST API server
```

→ **See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for a full walkthrough including config schema.**

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Installation, first simulation, config format |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Full Python API (`Model`, `BatchOracle`, bindings) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Engine internals: CTF solver, thermal network, surrogate layer |
| [`docs/SCHEMA.md`](docs/SCHEMA.md) | Config JSON schema and field definitions |
| [`docs/EXAMPLES.md`](docs/EXAMPLES.md) | Worked examples: optimization loops, multi-zone, surrogates |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | How to contribute (Rust dev setup, PR process) |
| [`docs/compliance/`](docs/compliance/) | ASHRAE 140 compliance docs: SQT report, deviations register, standards roadmap |
| [`docs/ASHRAE140_VALIDATION.md`](docs/ASHRAE140_VALIDATION.md) | ASHRAE 140 validation methodology and current results |
| [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) | Current limitations and workarounds |
| [`npm/README.md`](npm/README.md) | Node.js / TypeScript bindings |
| [`docs/USER_PERSONAS.md`](docs/USER_PERSONAS.md) | Who uses Fluxion and why |

---

## Understanding EUI Output

> **Important:** The EUI values returned by the current engine are a raw cumulative metric (sum of absolute temperature departures), **not** calibrated kWh/m²/year. They are valid for **relative comparison** (lower = better) but should not be reported as physical EUI without calibration.
>
> See [`docs/EXAMPLES.md`](docs/EXAMPLES.md) for normalization guidance. Full kWh/m²/year calibration is planned for v1.0.

---

## ASHRAE 140 Validation Status

> **v0.8.0 is not yet ready for a compliance submission.** Validation is actively in progress — physics fixes are landing in Waves 1–5, targeting full compliance in v1.0.

**Current pass rate: ~36% of test cases within reference ranges.**

| Area | Status | Notes |
|------|--------|-------|
| Annual heating/cooling energy (low-mass) | ⚠️ Partial | Some cases within range; others outside |
| Annual heating/cooling energy (high-mass) | ⚠️ Outside range | ±15–30%+ deviation; CTF thermal mass limitations |
| Free-floating temperature profiles | ⚠️ Partial | ±1–2°C deviations; within some tolerances |
| Peak loads (all cases) | 🔴 Significant deviation | ~76–100%+ overestimation; CTF solver limitation |
| Section 8 output format | 🔴 Incomplete | Units, timestamps, hourly profiles not yet compliant |
| Weather data | 🔴 Non-normative | Synthetic data in use; normative Annex C file pending (#732) |

Full results: [`docs/ASHRAE140_RESULTS_v0.8.0.md`](docs/ASHRAE140_RESULTS_v0.8.0.md) | Scorecard: [`SCORECARD.md`](SCORECARD.md) | Known deviations: [`docs/compliance/deviations-register.md`](docs/compliance/deviations-register.md)

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│              User Interface                  │
│       Python (pyo3) · Node.js (napi-rs)     │
│            CLI · REST API                   │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           Simulation Engine (Rust)           │
│                                              │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │  Thermal Network │  │ Surrogate Layer  │ │
│  │  (CTF / 5R1C)    │◄─►  (ONNX Runtime) │ │
│  └──────────────────┘  └──────────────────┘ │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │         BatchOracle / rayon          │   │
│  │     (parallel multi-config eval)     │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

Fluxion uses an **ISO 13790 5R1C thermal network** as its physics core. The surrogate layer intercepts expensive CFD/radiation calls and replaces them with trained ONNX neural networks. The `BatchOracle` evaluates populations in parallel using `rayon` for throughput-oriented workloads.

→ Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Training AI Surrogates

The included `SurrogateManager` uses placeholder values by default. To train a real surrogate:

```bash
pip install -r requirements-dev.txt
python tools/train_surrogate.py --num-samples 50000 --epochs 100
# Output: models/surrogate.onnx
```

→ See [`docs/EXAMPLES.md`](docs/EXAMPLES.md) for surrogate integration details.

---

## Contributing

Contributions welcome. The main development branch is `develop`; all PRs target `develop`.

```bash
git clone https://github.com/anchapin/fluxion.git
cd fluxion
git checkout develop

# Rust setup
rustup update
rustup component add rustfmt clippy

# Python setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install

# Build and test
cargo build && cargo test
maturin develop
```

→ Full guide: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)

---

## Release Process

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md#release-process) for the full release checklist (version bump, validation, wheel build, publication to crates.io / PyPI, GitHub release).

---

## License

MIT — see [`LICENSE`](LICENSE).
