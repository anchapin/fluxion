# Fluxion Codebase Map

> Comprehensive reference document for the Fluxion Building Energy Modeling engine

## Project Statistics

- **Source Files**: 49 Rust files in `src/`
- **Test Files**: 48 integration tests
- **Total Lines**: ~150,000+ (including generated code)
- **Documentation**: 35+ files in `docs/`

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python Layer                              │
│   BatchOracle (hot loop)  │  Model (detailed)  │  CLI         │
└─────────────────────────────────────────────────────────────────┘
                              │ FFI
┌─────────────────────────────────────────────────────────────────┐
│                    Rust Core (src/lib.rs)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   sim/   │  │    ai/   │  │ physics/ │  │validation│     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Breakdown

### `/src` - Core Source Code (49 files)

| Module | Files | Purpose |
|--------|-------|---------|
| `sim/` | 17 | Thermal modeling, HVAC, solar, boundaries |
| `ai/` | 8 | ONNX surrogates, neural fields, RL |
| `physics/` | 5 | CTA, VectorField, geometry tensors |
| `validation/` | 13 | ASHRAE 140, benchmarks, diagnostics |
| `weather/` | 3 | EPW parsing, weather data |

### `/tests` - Integration Tests (48 files)

| Category | Count | Examples |
|----------|-------|----------|
| ASHRAE 140 | 15 | `ashrae_140_integration.rs`, `ashrae_140_case_195_*.rs` |
| Issue investigations | 20 | `test_issue_273_*.rs`, `test_issue_274_*.rs` |
| Physics validation | 10 | `test_case_195_*.rs`, `test_6r2c_model.rs` |
| Debug/diagnostics | 3 | `debug_600.rs`, `diagnostic_demo.rs` |

### `/tools` - Python Scripts

| Script | Purpose |
|--------|---------|
| `train_surrogate.py` | Train ONNX surrogate models |
| `train_pinn.py` | Physics-Informed Neural Networks |
| `batch_rl_environment.py` | RL training environment |
| `geometry_extraction.py` | Extract building geometry |
| `benchmark_*.py` | Performance benchmarking |

### `/api` - FastAPI Layer

- `main.py` - REST API server
- `llm.py` - LLM integration
- `monitoring.py` - Metrics & monitoring
- `distributed_inference.py` - Distributed inference

### `/docs` - Documentation (35+ files)

Key docs:
- `ARCHITECTURE.md` - Deep dive into system design
- `ASHRAE140_RESULTS.md` - Validation results
- `CTA_USAGE.md` - Tensor abstraction guide
- `ONNX_INTEGRATION.md` - Surrogate setup
- `6R2C_IMPLEMENTATION.md` - Alternative model

### CI/CD (`.github/workflows/`)

| Workflow | Purpose |
|----------|---------|
| `ci.yml` | Main CI pipeline |
| `ashrae_140_validation.yml` | Validation suite |
| `python-bindings.yml` | PyO3 wheel building |
| `docker.yml` | Container builds |

## Key Source Files

### Entry Point: `src/lib.rs` (1746 lines)

```rust
// PyO3 Module: fluxion
pub mod ai;
pub mod physics;
pub mod sim;
pub mod validation;
pub mod weather;

// Core Classes
#[pyclass] struct Model { ... }      // Single building
#[pyclass] struct BatchOracle { ... } // Population optimization
#[pyclass] struct VectorField { ... }  // CTA wrapper
```

### Physics Engine: `src/sim/engine.rs` (4059 lines)

```rust
pub struct ThermalModel {
    pub temperatures: VectorField,       // Zone air temps
    pub mass_temperatures: VectorField, // Thermal mass
    pub loads: VectorField,            // Thermal loads
    pub window_u_value: f64,
    pub hvac_setpoint: f64,
    // 5R1C parameters...
}
```

### AI Surrogates: `src/ai/surrogate.rs` (793 lines)

```rust
pub struct SurrogateManager {
    session_pool: SessionPool,
    backend: InferenceBackend,
}

impl SurrogateManager {
    pub fn predict_loads(&self, temps: &[f64]) -> Vec<f64>;
    pub fn predict_loads_batched(&self, batch: &[Vec<f64>]) -> Vec<Vec<f64>>;
}
```

### CTA: `src/physics/cta.rs` (436 lines)

```rust
pub trait ContinuousTensor<T> { ... }

pub struct VectorField { data: Vec<f64> }
impl ContinuousTensor<f64> for VectorField { ... }
```

### Validation: `src/validation/ashrae_140_cases.rs` (71,808 lines)

```rust
pub enum CaseSpec {
    Case100, Case110, Case120, // Low mass
    Case600, Case610, Case620, // High mass  
    Case900, Case910, Case920, // Sunspace
    Case940, Case950, Case960, // Multi-zone
}
```

## Parameter Vector Semantics

**Used by BatchOracle for optimization:**

| Index | Parameter | Range | Units |
|-------|-----------|-------|-------|
| 0 | Window U-value | 0.1 - 5.0 | W/m²K |
| 1 | Heating setpoint | 15 - 25 | °C |
| 2 | Cooling setpoint | 22 - 32 | °C |

## Performance Targets

| Metric | Target | Current |
|--------|-------|---------|
| Per-config latency | <100ms | ~100ms |
| Population throughput | >10,000/sec | ~1,000/sec |
| GPU utilization (surrogates) | >80% | N/A |

## Build & Test Commands

```bash
# Development
cargo build && maturin develop
cargo test
cargo fmt && cargo clippy

# Validation
cargo test --test ashrae_140_validation
fluxion validate --all

# Benchmarks
cargo bench --bench cta_bench
cargo bench --bench engine_bench

# Python usage
python -c "import fluxion; print(fluxion.BatchOracle())"
```

## Dependency Graph

```
fluxion
├── rayon          (parallelism)
├── ort            (ONNX Runtime)
├── pyo3           (Python bindings)
├── ndarray        (arrays)
├── faer           (linear algebra)
├── tokio          (async)
├── crossbeam      (channels)
└── clap           (CLI)
```

## File Count Summary

```
src/                    49 files
├── lib.rs              1 file
├── sim/               17 files
├── ai/                 8 files  
├── physics/            5 files
├── validation/        13 files
├── weather/            3 files
└── bin/                1 file

tests/                 48 files
tools/                 25 files
api/                   10 files
docs/                  35 files
examples/              11 files
benches/                3 files

Total: ~180 source files
```

## Common Development Patterns

1. **Batch Oracle Pattern**: Pass full population across FFI once
2. **CTA Operations**: Use `VectorField` + element-wise ops
3. **Time-First Loop**: Sequential time, parallel configs
4. **ASHRAE 140**: All physics changes must validate

## Quick Reference

| Need | File |
|------|------|
| Add new physics | `src/sim/engine.rs` |
| Add new surrogate | `src/ai/surrogate.rs` |
| Add validation case | `src/validation/ashrae_140_cases.rs` |
| Fix Python FFI | `src/lib.rs` |
| Add CLI command | `src/bin/fluxion.rs` |
| Write test | `tests/` |
| Documentation | `docs/` |
