# Fluxion - Building Energy Modeling Engine

## Project Overview

**Fluxion** is a Rust-based Building Energy Modeling (BEM) engine with a **Neuro-Symbolic hybrid architecture** that combines physics-based thermal networks with AI surrogates for 100x-1000x speedups. It's designed to serve as a high-throughput oracle for quantum optimization and genetic algorithms.

### Core Capabilities
- **High-throughput evaluation**: 10,000+ building configurations per second
- **Physics-based simulation**: ISO 13790-compliant 5R1C Thermal Network
- **AI acceleration**: ONNX Runtime surrogates for fast inference
- **Python bindings**: PyO3 + maturin for seamless integration

### Technology Stack
- **Language**: Rust (Edition 2021)
- **Python Bindings**: PyO3 + maturin
- **Physics Engine**: Continuous Tensor Abstraction (CTA)
- **AI/ML**: ONNX Runtime, PyTorch
- **Parallelism**: rayon for data-parallel evaluation
- **Validation**: ASHRAE Standard 140 (18/18 cases passing)

---

## Architecture

### Two-Class API Pattern

1. **BatchOracle** (the "hot loop")
   - High-throughput parallel evaluation of population vectors
   - Used by quantum optimizers and genetic algorithms
   - Key method: `evaluate_population(population: Vec<Vec<f64>>, use_surrogates: bool) -> Vec<f64>`

2. **Model** (detailed single-building analysis)
   - Single-configuration simulation for validation/inspection
   - API: `simulate(years: u32, use_surrogates: bool) -> f64`

### Core Modules
- `src/sim/` - Thermal modeling, HVAC, solar, boundaries
- `src/ai/` - ONNX surrogates, neural fields, RL
- `src/physics/` - CTA, VectorField, geometry tensors
- `src/validation/` - ASHRAE 140, benchmarks
- `src/weather/` - EPW parsing, weather data

---

## Current Status

### What's Working
- ✅ 5R1C thermal network implementation
- ✅ PyO3 Python bindings (BatchOracle, Model classes)
- ✅ ONNX Runtime surrogate integration
- ✅ ASHRAE 140 validation (18/18 cases)
- ✅ CTA VectorField operations
- ✅ CLI with validate/quantize commands

### Key Metrics
- Per-config latency: ~100ms target
- Population throughput: ~1,000/sec (target: 10,000/sec)
- Python-Rust FFI optimized for batch operations

### Known Issues / Technical Debt
- Multi-zone heat transfer (Case 960) needs improvement
- Peak load tracking and reporting
- GPU surrogate acceleration not fully utilized

---

## Goals

1. **Performance Optimization**
   - Achieve >10,000 configs/sec throughput
   - Implement GPU-accelerated surrogates
   - Reduce per-config latency to <50ms

2. **Validation Coverage**
   - Complete ASHRAE 140 case coverage
   - Add peak load validation
   - Improve multi-zone accuracy

3. **Extensibility**
   - Add 6R2C thermal model option
   - FMI 3.0 co-simulation support
   - Enhanced RL policy integration

---

## Recent Changes (2026)

- Implemented continuous tensor abstraction (CTA)
- Added Fourier basis neural fields
- Enhanced interzone radiation calculations
- Fixed multiple ASHRAE 140 validation issues

---

## Community

- **Documentation**: See `docs/` for architecture deep-dives
- **Examples**: See `examples/` for usage patterns
- **Tests**: Run `cargo test` for validation
- **CLI**: `fluxion validate --all` for ASHRAE 140
