# Technology Stack

**Project:** ASHRAE 140 Validation Expansion
**Researched:** 2026-04-07

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Rust | 1.70+ | Systems programming language | Memory safety, performance, excellent for numerical computing |
| PyO3 | 0.20+ | Python bindings for Rust | Enables Python API for building energy modeling community |
| Rayon | 1.8+ | Data parallelism library | Thread-safe parallel execution for population-level simulations |
| ONNX Runtime | 2.0.0-rc.10 | AI surrogate inference | Fast neural network predictions for thermal load calculations |

### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| faer | 0.15+ | Linear algebra | Matrix operations for thermal network solving |
| ndarray | 0.15+ | N-dimensional arrays | Tensor operations in Continuous Tensor Abstraction |
| serde | 1.0+ | Serialization | JSON/CSV export for validation reports |
| tokio | 1.0+ | Async runtime | Future-proofing for async I/O operations |

### Validation & Testing
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| criterion | 0.4+ | Benchmarking | Performance regression testing |
| insta | 1.30+ | Snapshot testing | Validation report consistency checking |
| rstest | 0.18+ | Parameterized tests | Comprehensive ASHRAE 140 case coverage |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| **Language** | Rust | Python, C++ | Rust provides memory safety without GC overhead, better performance than Python, safer than C++ |
| **Parallelism** | Rayon | std::thread, crossbeam | Rayon's work-stealing provides better load balancing for heterogeneous workloads |
| **AI Inference** | ONNX Runtime | TensorRT, PyTorch | ONNX Runtime has best cross-platform support and Rust bindings |
| **Serialization** | serde | bincode, rmp-serde | serde has widest ecosystem support and human-readable formats |

## Installation

```bash
# Core dependencies
cargo add rayon@1.8
cargo add onnxruntime@2.0.0-rc.10
cargo add faer@0.15
cargo add ndarray@0.15
cargo add serde@1.0 --features derive
cargo add tokio@1.0 --features full

# Dev dependencies
cargo add criterion@0.4 --dev
cargo add insta@1.30 --dev
cargo add rstest@0.18 --dev

# Python bindings (optional)
cargo add pyo3@0.20 --features extension-module
```

## Sources

- Rust official documentation (HIGH confidence)
- PyO3 documentation (HIGH confidence)
- Rayon performance benchmarks (MEDIUM confidence)
- ONNX Runtime Rust bindings (MEDIUM confidence)
- Existing Fluxion codebase patterns (HIGH confidence)
