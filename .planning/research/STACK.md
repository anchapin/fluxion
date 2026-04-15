# Technology Stack: v1.2 Testing and Validation

**Project:** Fluxion v1.2
**Researched:** 2026-04-07

## Recommended Stack

### Core Testing Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Rust native test | 1.70+ | Unit and integration testing | Built-in, no additional dependencies, excellent performance |
| rstest | 0.18+ | Parameterized testing | Comprehensive ASHRAE 140 case coverage with data-driven tests |
| criterion | 0.4+ | Benchmarking | Performance regression testing and validation |
| insta | 1.30+ | Snapshot testing | Validation report consistency checking |
| pytest | 7.0.0+ | Python testing | Python bindings validation and integration testing |

### Validation & Testing Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| approx | 0.5+ | Floating-point comparisons | Validation tests with tolerance-based assertions |
| proptest | 1.5+ | Property-based testing | Fuzz testing for edge cases and robustness |
| mockito | 1.7+ | HTTP mocking | API integration testing and external service mocking |
| dhat | 0.3+ | Heap profiling | Memory usage validation and optimization |

### Performance & Parallelism
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Rayon | 1.8+ | Data parallelism | Thread-safe parallel execution for validation suite |
| tokio | 1.0+ | Async runtime | Future-proofing for async validation operations |
| ONNX Runtime | 2.0.0-rc.10 | AI surrogate inference | Performance optimization for complex validation cases |

### Cross-Validation & External Tools
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| EnergyPlus | 23.1+ | Reference validation | Industry standard for building energy simulation |
| ESP-r | 12.6+ | Cross-validation | Alternative simulation tool for comparison |
| TRNSYS | 19+ | Cross-validation | Additional reference for validation diversity |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| **Testing Framework** | Rust native + rstest | JUnit, NUnit | Rust native provides better integration and performance |
| **Benchmarking** | criterion | bench-rs, iai | criterion has better statistical analysis and reporting |
| **Snapshot Testing** | insta | expectest, snapbox | insta has better Rust ecosystem support and features |
| **Parallelism** | Rayon | std::thread, crossbeam | Rayon's work-stealing provides better load balancing |
| **External Integration** | File-based exchange | Direct FFI | File-based avoids complex dependencies and licensing issues |

## Installation

```bash
# Core testing dependencies (already in Cargo.toml)
cargo add rstest@0.18
cargo add criterion@0.4 --dev
cargo add insta@1.30 --dev
cargo add approx@0.5
cargo add proptest@1.5 --dev
cargo add mockito@1.7 --dev
cargo add dhat@0.3 --dev

# Python testing dependencies (already in pyproject.toml)
pip install pytest>=7.0.0
pip install black>=23.0.0
pip install ruff>=0.1.0
pip install mypy>=1.0.0

# Performance optimization
cargo add rayon@1.8
cargo add onnxruntime@2.0.0-rc.10
```

## Testing Infrastructure

### Current Test Structure
- **Unit Tests:** Core functionality validation in `src/` modules
- **Integration Tests:** End-to-end validation in `tests/` directory
- **ASHRAE 140 Tests:** Comprehensive validation suite (`tests/ashrae_140_*.rs`)
- **Performance Tests:** Regression testing (`tests/performance_regression_test.rs`)
- **Cross-Validation:** External tool comparison framework
- **Python Tests:** Bindings validation (`tests/python/test_*.py`)

### Test Execution
```bash
# Run all Rust tests
cargo test --all-features

# Run specific test
cargo test test_ashrae_140_comprehensive_validation

# Run performance benchmarks
cargo bench

# Run Python tests
pytest tests/python/

# Run with coverage
cargo tarpaulin
```

## Sources

- Rust official testing documentation (HIGH confidence)
- rstest parameterized testing patterns (HIGH confidence)
- criterion benchmarking best practices (HIGH confidence)
- Existing Fluxion test infrastructure analysis (HIGH confidence)
- ASHRAE 140 validation methodology (MEDIUM confidence)
- EnergyPlus cross-validation patterns (MEDIUM confidence)
