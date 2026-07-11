# Design Document: Python Bindings (PyO3) for Fluxion

> **Issue**: #782 - Python bindings (PyO3) for ecosystem tool integration
> **Status**: Design Document
> **Author**: Fluxion Team
> **Date**: 2026-06-16

## Executive Summary

This document describes the Python bindings architecture for Fluxion, a differentiable, AI-accelerated Building Energy Modeling (BEM) engine. Fluxion already has extensive PyO3 bindings that expose the core simulation engine to Python. This design document catalogs the current implementation, identifies gaps, and provides a roadmap for enhanced ecosystem integration.

## Current State

Fluxion's Python bindings are **production-ready** and provide comprehensive coverage of the simulation engine. The bindings are built using [PyO3](https://pyo3.rs/) with [maturin](https://www.maturin.rs/) as the build tool.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Ecosystem                          │
│   (scipy, pandas, numpy, optimization libs, ML frameworks)  │
└─────────────────────────────────────────────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │   src/lib.rs      │
                     │   (PyO3 module)   │
                     └─────────┬─────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
    │  python/  │       │   napi/  │       │  interop/ │
    │ bindings  │       │  napi    │       │    fmi    │
    └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
          │                    │                    │
    ┌─────▼────────────────────▼────────────────────▼─────┐
    │              Rust Core (rlib)                        │
    │   ThermalModel │ SurrogateManager │ Solvers        │
    └─────────────────────────────────────────────────────┘
```

### Feature Flag

Python bindings are controlled by the `python-bindings` feature in `Cargo.toml`:

```toml
python-bindings = ["dep:pyo3", "dep:pyo3-build-config", "dep:numpy"]
```

### Build Configuration

The project uses **maturin** for building Python extensions:

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
features = ["python-bindings", "pyo3/extension-module"]
module-name = "fluxion"
```

## Exposed Types

### Core Classes

| Rust Type | Python Class | Purpose |
|----------|-------------|---------|
| `Model` | `fluxion.Model` | Single-building detailed simulation |
| `BatchOracle` | `fluxion.BatchOracle` | High-throughput population evaluation |
| `BuildingParameters` | `fluxion.BuildingParameters` | Validated parameter wrapper |
| `ThermalModel<VectorField>` | `fluxion.MultiZoneThermalModel` | Multi-zone thermal model |
| `Construction` | `fluxion.Construction` | Wall construction assembly |
| `ConstructionLayer` | `fluxion.ConstructionLayer` | Individual material layer |
| `MassClass` | `fluxion.MassClass` | Thermal mass classification (light/medium/heavy) |
| `SurfaceType` | `fluxion.SurfaceType` | Surface type enumeration |
| `WallSurface` | `fluxion.WallSurface` | Wall surface with orientation |
| `GeometryTensor` | `fluxion.GeometryTensor` | Zone geometry as matrices |

### OSM Bindings

| Rust Type | Python Class | Purpose |
|----------|-------------|---------|
| `OsmReader` | `fluxion.OsmReader` | Import OpenStudio OSM files into schema dictionaries |
| `OsmWriter` | `fluxion.OsmWriter` | Export schema dictionaries to OpenStudio OSM files |
| `import_osm` | `fluxion.import_osm` | One-shot OSM import helper |
| `export_osm` | `fluxion.export_osm` | One-shot OSM export helper |

### HVAC Bindings

| Rust Type | Python Class | Purpose |
|----------|-------------|---------|
| `ZoneSetpoints` | `fluxion.ZoneSetpoints` | Zone temperature setpoints |
| `ZoneControl` | `fluxion.ZoneControl` | HVAC control logic |
| `DailySchedule` | `fluxion.DailySchedule` | Daily operating schedule |
| `HVACSchedule` | `fluxion.HVACSchedule` | Multi-day HVAC schedule |

### Exception Types

| Rust Type | Python Exception | Purpose |
|-----------|-----------------|---------|
| `FluxionError` | `fluxion.FluxionError` | Base exception |
| `ValidationError` | `fluxion.ValidationError` | Parameter validation errors |
| `SurrogateError` | `fluxion.SurrogateError` | AI surrogate errors |
| `SimulationError` | `fluxion.SimulationError` | Simulation runtime errors |

## API Reference

### BatchOracle

High-throughput parallel evaluation for optimization loops.

```python
from fluxion import BatchOracle, BuildingParameters

# Create oracle instance
oracle = BatchOracle()

# Raw vector API (backward compatible)
population = [[1.5, 20.0, 24.0], [2.0, 21.0, 25.0]]
results = oracle.evaluate_population(population, use_surrogates=False)

# Type-safe API with BuildingParameters
params = [
    BuildingParameters(window_u_value=1.5, heating_setpoint=20.0, cooling_setpoint=24.0),
    BuildingParameters(window_u_value=2.0, heating_setpoint=21.0, cooling_setpoint=25.0),
]
results = oracle.evaluate_population_typed(params, use_surrogates=True)

# NumPy-optimized API for large populations
import numpy as np
pop_array = np.array(population)  # shape (n, 3)
results = oracle.evaluate_population_numpy(pop_array, use_surrogates=False)
```

### Model

Single-building detailed analysis for validation and inspection.

```python
from fluxion import Model

# Create model
model = Model(num_zones=1)

# Set building type for auto-loading internal loads
model.building_type = "Office"

# Run simulation
eui = model.simulate(years=1, use_surrogates=False)

# Get diagnostics
temperatures = model.get_temperatures()
building_type = model.building_type
```

### NumPy Array API

Direct NumPy memory sharing between Python and Rust for weather data:

```python
import numpy as np
from fluxion import Model

model = Model(num_zones=3)

# Weather data arrays (8760 hourly values)
n_timesteps = 8760
dry_bulb = np.random.uniform(10, 35, n_timesteps)
dni = np.random.uniform(0, 1000, n_timesteps)
dhi = np.random.uniform(0, 500, n_timesteps)
ghi = np.random.uniform(0, 1000, n_timesteps)
wind_speed = np.random.uniform(0, 10, n_timesteps)
humidity = np.random.uniform(30, 80, n_timesteps)
horizontal_ir = np.random.uniform(200, 500, n_timesteps)

# Run simulation and get zone temperatures
zone_temps = model.simulate_numpy(
    dry_bulb, dni, dhi, ghi, wind_speed, humidity, horizontal_ir, False
)
# zone_temps.shape == (8760, 3)
```

## Performance Characteristics

| Metric | Value | Conditions |
|--------|-------|------------|
| Throughput | >10,000 configs/sec | 8-core CPU, surrogate mode |
| Latency | <100ms | Single configuration, 8760 timesteps |
| Memory | Minimal | CTA buffer reuse |

## Installation

### From PyPI (future)

```bash
pip install fluxion
```

### From source

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install
maturin develop --features python-bindings
```

### With CUDA support (GPU acceleration)

```bash
maturin develop --features "python-bindings cuda"
```

## Gaps and Roadmap

### Completed

- [x] Core PyO3 bindings infrastructure
- [x] Model class for single-building simulation
- [x] BatchOracle for high-throughput evaluation
- [x] BuildingParameters with validation
- [x] HVAC bindings (ZoneSetpoints, ZoneControl, Schedules)
- [x] NumPy array integration for weather data
- [x] Exception types with Python tracebacks

### In Progress

- [ ] PyPI release automation
- [ ] Comprehensive test coverage
- [ ] Documentation website (readthedocs.io)

### Future Enhancements

- [ ] Type stubs (.pyi) for IDE support
- [ ] Async API for concurrent simulations
- [ ] DataFrame integration for pandas
- [ ] Pre-trained surrogate model downloads
- [ ] Example notebooks (Jupyter)
- [ ] CI/CD with automated wheel building

## File Structure

```
src/
├── lib.rs                    # PyO3 module entry point
├── python/
│   ├── mod.rs               # Python module exports
│   ├── bindings.rs          # Multi-zone thermal model bindings
│   └── hvac_bindings.rs    # HVAC bindings
└── api/
    ├── mod.rs               # API support modules
    ├── error.rs            # Exception types
    ├── parameters.rs        # BuildingParameters
    └── schema.rs           # Simulation schema
```

## Testing

Python tests are located in `tests/python/`:

```bash
# Run Python tests
pytest tests/python/

# Run with coverage
pytest --cov=fluxion tests/python/
```

## References

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)
- [ASHRAE 140 Standard](https://www.ashrae.org/technical-resources/bookstore/standard-140)
- [ISO 13790 Energy Performance of Buildings](https://www.iso.org/standard/41905.html)
