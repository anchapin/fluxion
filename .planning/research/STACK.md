# Stack Research

**Domain:** Building Energy Modeling (BEM) Validation with ASHRAE 140
**Researched:** 2026-03-08
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Rust** | 2021 Edition | Physics engine core | Provides memory safety, zero-cost abstractions, and excellent performance for numerical computing. Critical for achieving >10K evaluations/second throughput requirements. |
| **PyO3** | 0.22 | Python bindings | Established, well-maintained FFI bridge between Rust and Python. Essential for BatchOracle API pattern with minimal overhead. abi3-py310 compatibility ensures stable binary interface. |
| **rayon** | 1.10 | Data parallelism | Industry-standard data parallelism library for Rust. Critical for BatchOracle population-level parallelism without nested parallelism anti-patterns. |
| **ort (ONNX Runtime)** | 2.0.0-rc.10 | AI surrogate inference | Current stable release candidate for ONNX Runtime Rust bindings. Provides thread-safe SessionPool for concurrent inference, essential for batched surrogate predictions. |
| **tokio** | 1.40 | Async runtime | Production-grade async runtime with multi-threaded scheduler. Required for concurrent ONNX inference and future async operations. |
| **ndarray** | 0.16 | Numerical computing | Rust's de facto standard for n-dimensional arrays. Serde feature enables efficient serialization for diagnostic output. |
| **faer** | 0.23.2 | Linear algebra | High-performance linear algebra library used for CTA operations. Optimized for scientific computing with minimal dependencies. |

### Testing & Validation Infrastructure

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **pytest** | Latest | Python test framework | For ASHRAE 140 case generator tests and Python integration tests. Industry standard with excellent plugin ecosystem. |
| **cargo test** | Built-in | Rust unit/integration tests | Core testing framework for all Rust validation tests. Supports parallel test execution and benchmarking. |
| **criterion** | 0.5 | Rust benchmarks | For performance regression testing of thermal model evaluation. Critical for maintaining <100μs per-config latency targets. |
| **tempfile** | 3.10 | Test file management | For creating temporary test artifacts (hourly output CSVs, diagnostic reports). |

### Data Processing & Analysis

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **pandas** | 2.0+ | Data analysis | For analyzing validation results, comparing metrics across test runs, and generating comparison reports. |
| **numpy** | 1.24+ | Numerical computing | For statistical analysis of validation metrics and processing hourly simulation output. |
| **scikit-learn** | 1.3+ | ML utilities | For surrogate model training and validation metrics calculation (MSE, MAE, R²). |

### Visualization & Reporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **matplotlib** | 3.7+ | Plotting | For temperature profile visualizations, energy breakdown charts, and comparison plots. |
| **seaborn** | 0.12+ | Statistical plots | For enhanced visualizations of validation metrics distribution and correlation analysis. |

### Documentation & Configuration

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **serde** | 1.0 | Serialization | For JSON/YAML export of validation reports and diagnostic data. Derive feature simplifies implementation. |
| **serde_json** | 1.0 | JSON handling | For generating JSON validation reports for CI/CD integration. |
| **clap** | 4.5 | CLI parsing | For `fluxion validate --all` command-line interface. Derive feature enables declarative argument definition. |
| **anyhow** | 1.0 | Error handling | For ergonomic error handling in validation tests and diagnostic tools. |
| **thiserror** | 1.0 | Error types | For custom error types in validation infrastructure with helpful error messages. |

## Installation

```bash
# Core Rust dependencies (in Cargo.toml)
[dependencies]
rayon = "1.10"
tokio = { version = "1.40", features = ["rt-multi-thread", "sync", "time", "macros"] }
ort = { version = "2.0.0-rc.10", features = ["download-binaries"] }
pyo3 = { version = "0.22", features = ["extension-module", "auto-initialize", "abi3-py310"], optional = true }
ndarray = { version = "0.16", default-features = false, features = ["std", "serde"] }
faer = { version = "0.23.2", default-features = false, features = ["std"] }

# Dev dependencies for testing
[dev-dependencies]
criterion = "0.5"
tempfile = "3.10"

# Python development dependencies (in requirements-dev.txt)
maturin>=1.0,<2.0
pytest
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
onnx>=1.14.0
onnxruntime>=1.15.0
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **rayon** | crossbeam | Use crossbeam if you need fine-grained thread pool control or custom work-stealing algorithms. Rayon is preferred for data-parallel patterns typical in BatchOracle. |
| **ort (ONNX Runtime)** | tract | Use tract if you need pure Rust ML inference without external runtime dependencies. ONNX Runtime is preferred for broader model ecosystem support and GPU backends. |
| **ndarray** | nalgebra | Use nalgebra if you need extensive linear algebra operations with compile-time dimension checking. ndarray is preferred for dynamic, runtime-sized arrays typical in BEM. |
| **PyO3** | rust-cpython | Use rust-cpython only if you need to support Python <3.7. PyO3 has better Rust API ergonomics and broader community support. |
| **tokio** | async-std | Use async-std if you prefer a smaller, simpler async runtime. tokio is preferred for larger ecosystem, better testing support, and production maturity. |
| **criterion** | cargo bench built-in | Use built-in cargo bench only for simple benchmarks. criterion provides statistical analysis, HTML reports, and regression detection essential for performance validation. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **nested rayon par_iter()** | Violates BatchOracle pattern, causes thread pool contention, destroys performance. | Single-level par_iter() at population level only. Pre-commit hook enforces this. |
| **raw Vec<f64> for state variables** | Physics engine expects CTA VectorField types for GPU acceleration potential. | Use VectorField from src/physics/cta.rs for all state variables. |
| **serde < 1.0** | Older versions lack derive feature, requiring boilerplate code. | Use serde 1.0 with derive feature enabled. |
| **pyo3 < 0.20** | Missing abi3-py310 support, may cause binary compatibility issues. | Use pyo3 0.22 with abi3-py310 feature for stable Python bindings. |
| **manual test organization** | No structured test suites, hard to run specific validation categories. | Use cargo test with module organization: tests/ashrae_140_*.rs for case-specific tests. |
| **hard-coded reference values** | Makes maintenance difficult when ASHRAE standards update. | Use src/validation/ashrae_140_cases.rs with CaseSpec structs for maintainable case definitions. |
| **single-threaded validation runs** | Hides race conditions, wastes parallel processing capabilities. | Run `cargo test` with default parallel execution. Use `--test-threads=1` only for debugging specific failures. |
| **ignoring diagnostic output** | Makes debugging validation failures extremely difficult. | Enable diagnostic config: `ASHRAE_140_DEBUG=1 cargo test -- --nocapture` for detailed output. |
| **pytest without fixtures** | Leads to test code duplication and difficult maintenance. | Use pytest fixtures for common test setup (case generators, validation runners). |

## Stack Patterns by Variant

**If implementing new ASHRAE 140 test cases:**
- Use `ASHRAE140Case` enum in `src/validation/ashrae_140_cases.rs`
- Define CaseSpec with all required parameters (construction, HVAC, windows, etc.)
- Create corresponding test in `tests/ashrae_140_case_*.rs` following existing patterns
- Use DiagnosticConfig to enable detailed output for debugging
- Because this ensures consistent case specification and automated report generation

**If debugging validation failures:**
- Enable environment variables: `ASHRAE_140_DEBUG=1`, `ASHRAE_140_VERBOSE=1`
- Set `ASHRAE_140_HOURLY_OUTPUT=1` to dump hourly CSV data
- Run single test: `cargo test test_case_600_baseline -- --nocapture`
- Use DiagnosticCollector to capture hourly temperature profiles, energy breakdowns
- Because diagnostic infrastructure provides detailed visibility into simulation behavior

**If benchmarking performance:**
- Use criterion benches in benches/ directory
- Run with `cargo bench --bench cta_bench` for CTA operations
- Run with `cargo bench --bench engine_bench` for thermal model performance
- Compare baseline to after changes to detect regressions
- Because criterion provides statistical significance testing and HTML reports

**If training AI surrogates:**
- Use tools/train_surrogate.py with PyTorch 2.0+
- Export models to ONNX format for ONNX Runtime inference
- Use SessionPool in src/ai/surrogate.rs for concurrent inference
- Validate surrogate predictions against analytical model before production
- Because ONNX Runtime provides thread-safe session management and GPU backends

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| pyo3@0.22 | numpy@0.22 (Rust), Python 3.10+ | Must use same major version for numpy Rust bindings |
| ort@2.0.0-rc.10 | onnxruntime@1.15.0+ (Python) | Python version for training, Rust version for inference |
| tokio@1.40 | rayon@1.10 | No direct dependency, but avoid mixing async runtime and rayon in same context |
| ndarray@0.16 | faer@0.23.2 | Both use compatible array representations for zero-copy conversion |
| serde@1.0 | All serialization targets | JSON, YAML, CSV all supported via serde ecosystem |
| pyo3 0.22 (abi3-py310) | Python 3.10+ | Stable ABI ensures binary compatibility across Python versions |

## Sources

- **Existing codebase analysis** — HIGH confidence (verified directly in `/home/alex/Projects/fluxion`)
  - `src/validation/ashrae_140_validator.rs` - Validation framework
  - `src/validation/ashrae_140_cases.rs` - Case specifications
  - `src/validation/diagnostic.rs` - Diagnostic infrastructure
  - `src/validation/report.rs` - Report generation
  - `tests/ashrae_140_validation.rs` - Comprehensive validation tests
  - `Cargo.toml` - Current dependency versions
  - `requirements-dev.txt` - Python development dependencies

- **Domain knowledge (training data)** — MEDIUM confidence (not verified with 2025 sources due to web search limitations)
  - ASHRAE Standard 140 validation requirements
  - BEM validation best practices
  - Python/Rust interoperability patterns

**Note**: Web search services were unavailable during research (rate limiting errors). Recommendations are based on:
1. Direct analysis of existing Fluxion codebase (HIGH confidence)
2. Current versions in Cargo.toml and requirements-dev.txt (HIGH confidence)
3. Domain knowledge about BEM validation patterns (MEDIUM confidence, not verified with 2025 sources)

**Recommended verification for production use**:
- Cross-check ONNX Runtime Rust version with official documentation
- Verify PyO3 0.22 compatibility with Python 3.10+ using official docs
- Confirm rayon 1.10 stability with current Rust toolchain

---
*Stack research for: Building Energy Modeling (BEM) Validation with ASHRAE 140*
*Researched: 2026-03-08*
