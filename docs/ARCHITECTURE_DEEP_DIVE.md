# Fluxion Architecture Deep Dive

## Overview

Fluxion is a differentiable, AI-accelerated Building Energy Modeling (BEM) engine. It combines physics-based thermal simulation with machine learning surrogates for high-throughput optimization.

## Architecture Layers

```
┌─────────────────────────────────────────┐
│           Python API Layer               │
│    (Model, BatchOracle, CLI)             │
├─────────────────────────────────────────┤
│           Rust Core Engine               │
│  (ThermalModel, Physics, Validation)    │
├─────────────────────────────────────────┤
│         ONNX Runtime Layer               │
│     (Surrogate Inference Engine)        │
└─────────────────────────────────────────┘
```

## Core Components

### 1. Thermal Model (`src/sim/engine.rs`)

The heart of Fluxion - a 5R1C thermal network solver.

```
┌──────────────┐     ┌──────────────┐
│  室外温度    │────▶│   外墙热阻   │
│  (Outdoor)   │     │     R_w      │
└──────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  室内温度    │
                    │     T_z      │
                    └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ 窗户得热 │  │ 内部得热 │  │ HVAC负荷 │
       │   Q_s   │  │   Q_i   │  │   Q_hc  │
       └──────────┘  └──────────┘  └──────────┘
```

**Key parameters:**
- R_w: Wall thermal resistance (m²K/W)
- C_m: Thermal mass (J/K)
- U: Overall heat transfer coefficient
- ACH: Air changes per hour

### 2. Surrogate Manager (`src/ai/surrogate.rs`)

Manages ONNX model loading and inference.

```rust
// Load surrogate
let surrogates = SurrogateManager::load_onnx("model.onnx")?;

// Batch prediction
let loads = surrogates.predict_loads_batched(&temperatures);
```

**Surrogate architecture:**
- Input: Temperature vector (8760 hourly values)
- Output: Heating/cooling loads (8760 values)
- Framework: ONNX Runtime (CPU/GPU)

### 3. Validation Framework (`src/validation/`)

ASHRAE 140 compliance testing.

```
tests/
├── case_600.rs    # Baseline low-mass
├── case_900.rs    # High-mass
├── case_960.rs    # Sunspace
└── ashrae_140_validator.rs
```

## Data Flow

### Physics Path

```
Config JSON → ThermalModel → solve_timesteps() → Energy
                │
                ▼
         Weather Data (8760h)
         - Outdoor temp
         - Solar radiation
         - Humidity
```

### Surrogate Path

```
Config → BatchOracle → Rayon Parallel → ONNX Runtime
                                        │
                                        ▼
                              GPU/Tensor Cores
                                        │
                                        ▼
                              EUI Results (Vector)
```

## Performance Optimizations

### 1. Vectorization

Uses SIMD (Single Instruction Multiple Data) for matrix operations.

### 2. Parallelism

- **Rayon**: Data parallelism for population evaluation
- **Crossbeam Channels**: Coordinator-worker pattern for batch inference

### 3. GPU Acceleration

- ONNX Runtime CUDA/CoreML backends
- Batch inference for 100K+ configs/sec

### 4. Caching

- Surrogate session pooling
- Weather data pre-loading

## Extension Points

### Adding New Physics Models

1. Implement `ThermalModel` trait
2. Add to `src/physics/`
3. Register in `src/lib.rs`

### Adding New Surrogates

1. Train ONNX model
2. Add to `src/ai/surrogate.rs`
3. Update `SurrogateManager`

### Adding New Validation Cases

1. Create case in `src/validation/ashrae_140/`
2. Add to test suite
3. Document in validation report

## Module Structure

```
src/
├── lib.rs              # Main entry point
├── ai/
│   ├── surrogate.rs    # ONNX model management
│   └── neural_field.rs # Neural field implementations
├── physics/
│   ├── cta.rs          # Continuous tensor algebra
│   └── continuous.rs   # PDE solvers
├── sim/
│   ├── engine.rs       # Thermal model solver
│   ├── construction.rs # Building construction
│   ├── boundary.rs     # Boundary conditions
│   ├── solar.rs       # Solar radiation
│   └── ventilation.rs # Air changes
└── validation/
    ├── ashrae_140/
    │   ├── case_600.rs
    │   └── ...
    └── validator.rs
```

## Comparison with EnergyPlus

| Feature | Fluxion | EnergyPlus |
|---------|---------|------------|
| Speed (single sim) | ~1ms | ~30s |
| Speed (10K configs) | ~1s | ~83h |
| Surrogate mode | Yes | No |
| Differentiable | Yes | No |
| GPU support | Yes | No |
| ASHRAE 140 | 18/18 cases | Full |

## Further Reading

- [Theory and Strategy](../docs/THEORY_AND_STRATEGY.md)
- [Performance Tuning](../docs/PERFORMANCE_TUNING.md)
- [ASHRAE 140 Validation](../docs/ASHRAE140_VALIDATION.md)
