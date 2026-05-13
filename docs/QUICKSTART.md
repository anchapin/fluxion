# Fluxion Quickstart Guide

Get started with Fluxion in minutes.

## Installation

Fluxion's core is a Rust library with Python bindings built via [maturin](https://maturin.rs). There is no `pip install fluxion` yet — you build from source.

### Prerequisites

- Rust toolchain (stable) — [install rustup](https://rustup.rs)
- Python 3.10+
- `maturin`

### From source (recommended)

```bash
git clone https://github.com/anchapin/fluxion.git
cd fluxion

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install build tool and dependencies
pip install --upgrade pip
pip install maturin
pip install -r requirements-dev.txt

# Build and install Python bindings
maturin develop
```

To verify the build:

```python
import fluxion
print("Fluxion loaded OK")
```

---

## Quick Examples

> **Note on EUI values**: The raw value returned by `model.simulate()` and `oracle.evaluate_population()` is a cumulative energy-like metric — it is **not** calibrated `kWh/m²/year`. It is intended for **relative comparison** only (lower = better). See [Interpreting Results](#interpreting-results) below.

### 1. Single Building Simulation

```python
from fluxion import Model

# Create model from a JSON config file
model = Model("config.json")

# Run annual physics-based simulation (no surrogates)
result = model.simulate(years=1, use_surrogates=False)
print(f"Annual energy metric: {result:.2f}  (relative; lower = better)")
```

### 2. Using AI Surrogates (~100× faster)

```python
from fluxion import Model

model = Model("config.json")

# Load a trained ONNX surrogate
model.load_surrogate("loads_predictor.onnx")

# Run with surrogates — results are approximate but much faster
result = model.simulate(years=1, use_surrogates=True)
print(f"Annual energy metric: {result:.2f}  (surrogate-accelerated)")
```

### 3. Batch Optimization (Population Evaluation)

```python
from fluxion import BatchOracle
import numpy as np

oracle = BatchOracle()

# Define a population of building configurations
# Each row: [window_u_value, hvac_setpoint]
population = np.random.rand(10000, 2).tolist()

# Evaluate all configurations in parallel (Rust handles threading)
results = oracle.evaluate_population(population, use_surrogates=True)

best_idx = results.index(min(results))
print(f"Evaluated {len(results)} designs")
print(f"Best design index: {best_idx}, metric: {min(results):.2f}")
```

### 4. Run the Included Examples

```bash
# From the repo root with maturin develop already run:
python examples/run_model.py
python examples/run_oracle.py

# Or use the helper script (macOS/Linux):
bash examples/quick_start.sh
```

---

## Your First Configuration

Create `config.json`:

```json
{
  "zone_area": 48.0,
  "zone_volume": 129.6,
  "window_u_value": 1.5,
  "window_area": 12.0,
  "heating_setpoint": 20.0,
  "cooling_setpoint": 27.0,
  "infiltration_rate": 0.5,
  "internal_gains": 100.0
}
```

---

## Interpreting Results

The `EUI` / energy metric returned by fluxion in v0.8.x is a **raw cumulative value**, not a normalized `kWh/m²/year`. Specifically, the physics engine accumulates per-hour absolute temperature departure from setpoint across all zones. Because of this:

- **Use for relative comparison only** — lower value means a more energy-efficient design
- **Do not compare to published building benchmarks** without calibration
- Dividing by `num_zones × 8760` gives an average hourly temperature-gap metric — still not physical energy

Full normalization to `kWh/m²/year` requires thermal capacity and area scaling not yet in the current model. This will be addressed as the ASHRAE 140 physics Waves land.

---

## ASHRAE 140 Compliance

Fluxion's ASHRAE 140-2023 validation is actively in progress. The current pass rate (~36%) will improve with each physics Wave fix (Waves 1–5, targeting v1.0).

See [`docs/compliance/README.md`](compliance/README.md) for the full compliance status and known deviations.

---

## Next Steps

- [API Reference](API_REFERENCE.md) — Full Python API documentation
- [Architecture Deep Dive](ARCHITECTURE_DEEP_DIVE.md) — How Fluxion's physics engine works
- [Compliance Status](compliance/README.md) — ASHRAE 140 validation roadmap
- [Examples](../examples/) — More worked examples

---

## Getting Help

- **GitHub Issues**: https://github.com/anchapin/fluxion/issues
- **Discussions**: https://github.com/anchapin/fluxion/discussions
