# GEMINI.md - Fluxion Project Context

## Project Overview
**Fluxion** is a next-generation, AI-accelerated Building Energy Modeling (BEM) engine. It uses a **hybrid Neuro-Symbolic architecture**, combining first-principles physics (ISO 13790-compliant 5R1C thermal network) with high-performance AI surrogates (ONNX) to achieve 100x-1000x speedups over traditional tools.

The primary goal is to provide a high-throughput **Batch Oracle** capable of evaluating 10,000+ building configurations per second, making it ideal for genetic algorithms, Bayesian optimization, and quantum annealing.

### Main Technologies
- **Core Engine**: Rust (2021 edition) for performance and safety.
- **Physics**: Continuous Tensor Abstraction (CTA) for vector-accelerated thermal calculations.
- **AI/ML**: ONNX Runtime (via `ort`) for surrogate inference; PyTorch for training.
- **Parallelism**: `rayon` for data-parallel evaluation of building populations.
- **Bindings**: `PyO3` and `maturin` for seamless Python integration.
- **Validation**: Rigorous adherence to the **ASHRAE Standard 140** validation suite.

## Building and Running

### Prerequisites
- Rust toolchain (latest stable)
- Python 3.10+
- `maturin` (for building Python bindings)

### Key Commands
| Task | Command |
| :--- | :--- |
| **Build Rust Core** | `cargo build --release` |
| **Build Python Bindings** | `maturin develop` |
| **Run Unit Tests** | `cargo test` |
| **Run ASHRAE 140 Validation** | `cargo test --test ashrae_140_validation` |
| **Run Benchmarks** | `cargo bench` |
| **Train AI Surrogate** | `python tools/train_surrogate.py` |
| **Linting (Rust)** | `cargo fmt && cargo clippy` |
| **Linting (Python)** | `ruff check .` |

## Development Conventions

### Architecture Patterns
- **Batch Oracle Pattern**: Always prefer passing large batches of data (e.g., 10,000 configurations) across the Python-Rust FFI boundary at once. Avoid nested parallelism; use `rayon` at the population level.
- **CTA (Continuous Tensor Abstraction)**: Use `VectorField` and CTA operations (`+`, `*`, `/`) for physics state variables instead of raw `Vec<f64>` to enable future hardware acceleration.
- **Neuro-Symbolic Hybrid**: Physics-informed neural networks where surrogates predict loads/influences while the 5R1C thermal network maintains energy conservation.

### Coding Style
- **Rust**: Follow standard `idiomatic Rust` and `clippy` suggestions.
- **Python**: Adhere to `PEP 8`; use `ruff` for formatting and linting.
- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat(engine): ...`, `fix(physics): ...`).

### Testing & Validation
- **ASHRAE 140**: This is the "gold standard" for the project. Every physics change must be validated against the 18 cases in `src/validation/ashrae_140_cases.rs`.
- **Regression Testing**: Integration tests in `tests/` cover complex scenarios like sunspaces and night ventilation.

## Project Structure
- `src/sim/`: Core physics engine and thermal model logic.
- `src/ai/`: Surrogate management, ONNX integration, and neural field representations.
- `src/physics/`: Mathematical abstractions (CTA, VectorFields).
- `src/validation/`: ASHRAE 140 validation logic and benchmark data.
- `api/`: FastAPI-based monitoring and management layer.
- `tools/`: Python scripts for data generation, training, and benchmarking.
- `docs/`: Technical documentation, PRDs, and architecture deep-dives.

## Current Focus (as of Feb 2026)
- Stabilizing ASHRAE 140 annual energy metrics (currently high variance in controlled cases).
- Implementing Peak Load Tracking and reporting.
- Improving multi-zone heat transfer (Case 960).
- Automating validation reports via GitHub Actions.
