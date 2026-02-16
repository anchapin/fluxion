# Fluxion Quickstart Guide

Get started with Fluxion in minutes.

## Installation

### From PyPI (recommended)

```bash
pip install fluxion
```

### From source

```bash
git clone https://github.com/anchapin/fluxion.git
cd fluxion
pip install -e .
```

### With Docker

```bash
docker run -p 8000:8000 fluxion-api
```

## Quick Examples

### 1. Basic Simulation

```python
from fluxion import Model

# Create model
model = Model("config.json")

# Run physics-based simulation
eui = model.simulate(years=1, use_surrogates=False)
print(f"EUI: {eui:.2f} kWh/m²/year")
```

### 2. Using Surrogates

```python
from fluxion import Model

model = Model("config.json")

# Load surrogate model for fast inference
model.load_surrogate("loads_predictor.onnx")

# Run with AI surrogates (~100x faster)
eui = model.simulate(years=1, use_surrogates=True)
print(f"EUI: {eui:.2f} kWh/m²/year")
```

### 3. Population Optimization

```python
from fluxion import BatchOracle

oracle = BatchOracle()

# Define 10,000 building configurations
population = [[1.5, 20.0, 27.0]] * 10000

# Evaluate in parallel
results = oracle.evaluate_population(population, use_surrogates=True)

print(f"Evaluated {len(results)} designs")
print(f"Best EUI: {min(results):.2f}")
```

### 4. Command Line

```bash
# Run a simulation
fluxion run config.json

# Run with surrogates
fluxion run config.json --surrogate model.onnx

# Run API server
fluxion serve
```

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

## Next Steps

- [API Reference](API_REFERENCE.md) - Full API documentation
- [Architecture Deep Dive](ARCHITECTURE_DEEP_DIVE.md) - Understanding Fluxion internals
- [Examples](../examples/) - More usage examples

## Getting Help

- GitHub Issues: https://github.com/anchapin/fluxion/issues
- Documentation: https://fluxion.readthedocs.io
