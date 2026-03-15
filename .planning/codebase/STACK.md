# Technology Stack

**Analysis Date:** 2026-03-08

## Languages

**Primary:**
- Rust 2021 Edition - Core physics engine, CTA tensor abstraction, thermal model
- Python 3.10-3.12 - Python bindings via PyO3, training scripts, API server, ML tools

**Secondary:**
- YAML - Configuration files (Docker Compose, GitHub Actions)
- TOML - Project configuration (Cargo.toml, pyproject.toml)

## Runtime

**Environment:**
- Rust: rustc 1.93.1, cargo 1.93.1
- Python: 3.10+ (tested on 3.11, 3.12)

**Package Manager:**
- Rust: cargo 1.93.1
  - Lockfile: `Cargo.lock` (present)
- Python: pip via maturin
  - Lockfile: Not used (dev requirements specified)

## Frameworks

**Core:**
- PyO3 0.22 - Python-Rust FFI bindings
- maturin 1.0+ - Build and package Rust-Python extensions
- FastAPI 0.109+ - REST API server (`api/main.py`)
- Uvicorn 0.27+ - ASGI server for FastAPI
- Pydantic 2.5+ - Data validation

**Testing:**
- pytest 7.0+ - Python test runner
- cargo test - Rust test framework (built-in)
- criterion 0.5 - Rust benchmarking

**Build/Dev:**
- Docker - Multi-stage containerization
- GitHub Actions - CI/CD pipeline
- pre-commit - Git hooks (ruff, black, isort, mypy, cargo fmt, cargo check, cargo audit)

## Key Dependencies

**Critical:**
- ort 2.0.0-rc.10 - ONNX Runtime for AI surrogate inference
  - Download binaries automatically for easy setup
  - Supports multiple backends: CPU, CUDA, CoreML, DirectML, OpenVINO
- rayon 1.10 - Data parallelism for population evaluation
- tokio 1.40 - Async runtime for distributed inference
- ndarray 0.16 - Numerical arrays (Rust)
- faer 0.23.2 - Linear algebra library

**Infrastructure:**
- serde 1.0 - Serialization/deserialization
- anyhow 1.0 - Error handling
- thiserror 1.0 - Derive error types
- clap 4.5 - CLI argument parsing
- rand 0.8 / rand_distr 0.4 - Random number generation for uncertainty quantification
- crossbeam 0.8.4 - Concurrent data structures

**Python Runtime:**
- numpy 1.24+ - Numerical computing
- pandas 2.0+ - Data manipulation
- onnxruntime 1.15+ - Python ONNX runtime
- torch 2.0+ - PyTorch for model training
- scikit-learn 1.3+ - ML utilities
- matplotlib 3.7+ / seaborn 0.12+ - Visualization

**Python Dev Tools:**
- black 23.0+ - Code formatting
- isort 5.12+ - Import sorting
- ruff 0.1+ - Linter and formatter
- mypy 1.0+ - Static type checking

## Configuration

**Environment:**
- No environment configuration detected (no .env files found)
- Configuration via Python's logging module (api/main.py)
- RUST_LOG environment variable for Rust logging (set in Dockerfile)

**Build:**
- `Cargo.toml` - Rust package configuration
- `pyproject.toml` - Python package configuration (PEP 621)
- `requirements-dev.txt` - Python development dependencies
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `Dockerfile` - Multi-stage Docker build
- `docker-compose.yml` - Container orchestration

**Release Profile (Cargo.toml):**
```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
split-debuginfo = "packed"
strip = true
```

## Platform Requirements

**Development:**
- Rust toolchain (rustc, cargo)
- Python 3.10+ with pip
- maturin for building Python bindings
- pre-commit for git hooks
- Optional: CUDA toolkit for GPU inference

**Production:**
- Linux amd64/arm64 (Docker containers)
- Python 3.11+ runtime
- ONNX Runtime runtime libraries
- libgomp1, libssl3 system dependencies
- Optional: GPU for CUDA backend

---

*Stack analysis: 2026-03-08*
