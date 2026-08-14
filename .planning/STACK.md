# Technology Stack

**Analysis Date:** 2026-08-11
**Workspace:** 11 crates (1 root package + 10 workspace members) — see `STRUCTURE.md` for the layout.

## Languages

**Primary:**
- **Rust 2021 Edition** — Core physics engine, thermal networks, conduction solvers, validation, axum REST API, MCP server, WASM bindings
- **Python 3.10–3.12** — Python bindings via PyO3, training scripts, legacy FastAPI REST server (`api/`), ML tools

**Secondary:**
- **TypeScript / JavaScript** — Node.js bindings via NAPI-RS (`npm/`)
- **YAML** — Configuration files (Docker Compose, GitHub Actions, release gates)
- **TOML** — Project configuration (`Cargo.toml`, `pyproject.toml`, `rust-toolchain.toml`)
- **Ruby** — OpenStudio measures (`measures/`), EnergyPlus IDF scripting helpers
- **Python (scripts)** — Drift / cycle / docs-hygiene / release-gate CI guards (`scripts/check_*.py`, `scripts/release_gate_checker.py`)

## Runtime

**Environment:**
- Rust: stable toolchain, pinned via `rust-toolchain.toml` (channel = "stable" + rustfmt + clippy)
- Python: 3.10+ (tested on 3.11, 3.12)
- Node.js: for NAPI-RS bindings build (`npm/`)

**Package Manager:**
- Rust: cargo (workspace with shared `Cargo.lock` at workspace root)
- Python: pip via `maturin` (PEP 517 build backend)
- Node.js: npm (`npm/`)

## Workspace Crates

The fluxion workspace contains 11 crates. The root `fluxion` package is the main engine; the 10 workspace members (`members = [...]` in the root `Cargo.toml`) are either always-built siblings or feature-gated.

| Crate | Path | Purpose | Feature Gate |
|-------|------|---------|--------------|
| **fluxion** (root) | `.` / `src/` | The engine: `sim/` (thermal model, solar, ventilation), `physics/` (conduction solvers), `ai/` (ONNX surrogates), `validation/` (ASHRAE 140, energy balance), `api/` (axum REST), `python/` + `napi/` (bindings), `interop/` (OSM/gbXML/IFC/FMI), `cli/`, `bin/`. | default = none |
| **fluxion-core** | `fluxion-core/` | Dependency-light *leaf* modules: `weather/`, `assembly.rs`, `construction.rs`, `multi_node.rs`, `per_surface_conduction.rs`, `ashrae_cases.rs`, `physics_constants.rs`, `urban_radiation.rs`, `tensor.rs`, `earth_tube.rs`, `fluid/`. Built once & cached by cargo-mutators. Must not depend on `sim/physics/ai/validation` (cycle-breaking rule). | always built |
| **fluxion-grid** | `fluxion-grid/` | Grid-edge electrical network components: battery storage, bus nodes, power flow, PV, joint thermal-electrical convergence (`ThermalElectricalCoupler`). | always built; `fluxion-integration` / `fluid` features optional |
| **fluxion-behavior** | `fluxion-behavior/` | Thermal comfort & behavioral models: Fanger PMV/PPD, adaptive comfort, occupancy (Markov + deterministic), lighting, plug loads, internal gains, moisture, occupant triggers, TOON time encoder. | always built; default = `["ort"]` |
| **fluxion-wasm** | `fluxion-wasm/` | WebAssembly bindings over `fluxion-core` + `fluxion-fluid` via `wasm-bindgen`. | always built |
| **fluxion-city** | `fluxion-city/` | Urban radiation modeling with Nusselt-analog view factor computation. Parallel harness + ray tracing. | feature-gated on root: `fluxion-city` (Issue #2344); crate-local `parallel` feature |
| **fluxion-fluid** | `fluxion-fluid/` | Compile-time strongly typed fluid port traits for acausal HVAC / fluid DAE systems (Issue #1980 / ADR-0005). *Not* the same as `fluxion-core/src/fluid/`. | feature-gated on root: `fluid` |
| **fluxion-cfd** | `fluxion-cfd/` | GPU-accelerated Fast Fluid Dynamics (FFD) solver for building airflow simulation (Issue #2460). CPU / CUDA / OpenCL backends. | feature-gated on root: `fluxion-cfd` |
| **fluxion-mcp** | `fluxion-mcp/` | Model Context Protocol server for Rust-native BEM interface. Unconditionally depends on `fluxion` with `multi-zone` + `fluxion-fluid` + `fluxion-toon`. | always built; default = `["multi-zone"]` |
| **fluxion-toon** | `crates/fluxion-toon/` | Token-Oriented Object Notation (TOON) — compact, LLM-friendly serializer/deserializer. SPEC in `crates/fluxion-toon/SPEC.md`. | always built; `std` feature |
| **fluxion-twin** | `crates/fluxion-twin/` | Digital twin core — Unscented Kalman Filter for non-linear state estimation in thermal systems. MQTT telemetry consumer (TLS-only by default; see `FLUXION_MQTT_*` env vars). | always built |

**Workspace `default-members = ["."]`** — bare `cargo build` / `cargo test` build the root `fluxion` crate only, not the whole workspace.

### Cycle-Breaking Rules (CI-enforced)

- `fluxion-core/src/**/*.rs` must NOT import `crate::sim_*` / `crate::physics_*` / `crate::ai_*` / `crate::validation_*` — guard: `scripts/check_ashrae_cases_cycle.py` (#1441).
- No `use crate::sim::*` under `src/physics/**` or `use crate::physics::*` under `src/sim/construction.rs` / `src/sim/per_surface_conduction.rs` — guard: `scripts/check_physics_sim_cycle.py` (#2463).
- `sim::assembly` and `sim::multi_node_thermal` in `src/sim/` are thin re-export shims (kept for backwards compatibility).

## Feature Flags (root `fluxion` crate)

Default = none. Most functionality is behind cargo feature flags; default builds skip the ONNX runtime.

| Feature | Effect |
|---------|--------|
| `python-bindings` | Enable PyO3 Python bindings (links libpython) |
| `python-extension` | Build as a Python extension module (maturin wheel; #2532). Aliases `python-bindings` + `pyo3/extension-module` |
| `napi-bindings` | Enable Node.js / NAPI bindings |
| `ort` (alias `onnx`) | ONNX Runtime inference (AI surrogate); auto-download binaries |
| `cuda` | CUDA / TensorRT execution providers for `ort` (implies `ort`; auto-downgrades to CPU if feature not built) |
| `multi-zone` | Multi-zone thermal network support |
| `wiring-tracing` | Wiring tracing in integration tests |
| `ashrae_140_v2021` | ASHRAE 140 v2021 reference data set |
| `pr821-diag` | PR #821 hourly CSV diagnostics (600FF/650FF debugging) |
| `loom` | Loom concurrency fuzzing (needs ~32 GB RAM; #1065) |
| `dwave` | D-Wave quantum annealer SAPI REST integration (requires `DWAVE_API_TOKEN`) |
| `debug-physics` | Gate `eprintln!` in physics hot loops (#1967) |
| `tracing-subscriber-json` | Structured JSON tracing output from validation (#2500) |
| `kafka` | rdkafka Kafka telemetry consumer (#2056) |
| `fluid` | Enable `fluxion-fluid` acausal HVAC / fluid port traits (#1980 / ADR-0005) |
| `gauge-solver` | GaugeSolver as primary zone solver (replaces 5R1C/9R4C; #2304) |
| `fluxion-city` | Wire `fluxion-city` urban radiation solver (#2344) |
| `fluxion-cfd` | Wire `fluxion-cfd` FFD/CFD adapter (#2460) |
| `dhat` | Heap profiling via dhat (#2384) |

See `AGENTS.md` §Toolchain Quirks for the authoritative list and exact build commands.

## Frameworks

**Rust (primary):**
- **axum 0.7** — Rust-native REST API server (`src/api/server.rs`, binary `fluxion-rest`; default `0.0.0.0:8080`, healthcheck `/v1/healthz`). Auth modes: `off` / `token` / `tls` via `FLUXION_REST_AUTH`.
- **PyO3 0.22** — Python-Rust FFI bindings (`python-bindings` feature, `abi3-py310`)
- **ort 2.0.0-rc.10** — ONNX Runtime (`ort` feature; supports CPU / CUDA / CoreML / DirectML / OpenVINO backends via `FLUXION_ONNX_BACKEND`)
- **tokio 1.40** — Async runtime (REST server, distributed inference)
- **rayon 1.10** — Data parallelism for population evaluation (BatchOracle, population-level only)
- **clap 4.5** — CLI argument parsing
- **wasm-bindgen** — WebAssembly bindings (`fluxion-wasm`)
- **napi-rs 3** — Node.js native bindings (`napi-bindings`)

**Python (secondary):**
- **FastAPI 0.109+** — Legacy Python REST API server (`api/main.py`) — *superseded by the Rust-native axum server for production use*
- **Uvicorn 0.27+** — ASGI server for FastAPI
- **Pydantic 2.5+** — Data validation (Python API)
- **maturin 1.0+** — Build and package Rust-Python extensions

**Build/Dev:**
- **Docker** — Multi-stage containerization
- **GitHub Actions** — CI/CD pipeline (see `release_gates.yaml` for required branch-protection checks)
- **pre-commit** — Git hooks (ruff, black, isort, fmt, cargo-check, cargo-audit, batch-oracle-pattern, rust-doc-check, conventional-commit-msg)

**Testing:**
- `cargo test` — Rust test framework (built-in)
- `pytest 7.0+` — Python test runner
- `criterion 0.5` — Rust benchmarking
- `cargo mutants` — Mutation testing (advisory on PRs, nightly on `develop`; needs 32 GB RAM for full suite)

## Key Dependencies

**Critical (Rust):**
- `ort 2.0.0-rc.10` — ONNX Runtime for AI surrogate inference (auto-download binaries; multi-backend)
- `rayon 1.10` — Population-level data parallelism
- `tokio 1.40` — Async runtime
- `ndarray 0.16` — Numerical arrays
- `faer 0.23.2` — Linear algebra
- `axum 0.7` — REST API
- `serde 1.0` / `serde_json 1.0` — Serialization
- `anyhow 1.0` / `thiserror 1.0` — Error handling
- `crossbeam 0.8.4` — Concurrent data structures
- `rand 0.8` / `rand_distr 0.4` — Random number generation (uncertainty quantification)

**Python Runtime:**
- `numpy 1.24+` — Numerical computing
- `pandas 2.0+` — Data manipulation
- `onnxruntime 1.15+` — Python ONNX runtime
- `torch 2.0+` — PyTorch for model training
- `scikit-learn 1.3+` — ML utilities
- `matplotlib 3.7+` / `seaborn 0.12+` — Visualization

**Python Dev Tools:**
- `black 23.0+` — Code formatting
- `isort 5.12+` — Import sorting
- `ruff 0.1+` — Linter and formatter
- `mypy 1.0+` — Static type checking

## Configuration

**Environment Variables (REST API — `fluxion-rest`):**
- `FLUXION_REST_BIND` / `FLUXION_REST_PORT` — bind address (default `0.0.0.0:8080`)
- `FLUXION_REST_AUTH` — `off` (default) | `token` | `tls`
- `FLUXION_REST_AUTH_TOKEN` — bearer token for `token` / `tls` modes
- `FLUXION_REST_CORS_ORIGINS` — comma-separated origin allow-list (never `permissive()`)
- `FLUXION_REST_RATE_LIMIT_RPS` / `FLUXION_REST_RATE_LIMIT_BURST` — per-IP token-bucket (default 100 / 1000)
- `FLUXION_REST_ALLOW_INSECURE=1` — opt out of the release-build boot guard

**Environment Variables (ONNX / AI):**
- `FLUXION_ONNX_MODEL` — explicit ONNX model path (default `models/surrogate_zone_thermal.onnx`; mock fallback when unset)
- `FLUXION_ONNX_BACKEND` — `cpu | cuda | coreml | directml | openvino`
- `FLUXION_GPU=0` — force CPU inference

**Environment Variables (Digital Twin — `fluxion-twin`):**
- `FLUXION_MQTT_ALLOW_INSECURE` — permit plaintext `mqtt://` / `tcp://` brokers (default: TLS-only `mqtts://` port 8883)
- `FLUXION_MQTT_INSECURE` — skip TLS server-certificate validation (dangerous; local dev only)
- `DWAVE_API_TOKEN` — required at runtime for the `dwave` feature

**Environment Variables (CI / bindings):**
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` + `RUST_MIN_STACK=33554432` — Python-bindings CI (avoid linker SIGSEGV)
- `CARGO_BUILD_JOBS=1` — Clippy CI (peak RSS)

**Build files:**
- `Cargo.toml` — Rust workspace + root package configuration
- `pyproject.toml` — Python packaging (PEP 621)
- `requirements-dev.txt` — Python development dependencies
- `.pre-commit-config.yaml` — Pre-commit hooks
- `Dockerfile` / `docker-compose.yml` — Multi-stage Docker build
- `rust-toolchain.toml` — Stable toolchain pin (+ rustfmt + clippy)
- `.rustfmt.toml` — `edition = "2021"` (required; rustfmt falls back to 2015 without it)

**Release Profile (`Cargo.toml`):**
```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 4
split-debuginfo = "packed"
strip = true

[profile.ci]         # FAST iteration build (opt-level=1, codegen-units=256)
inherits = "dev"
opt-level = 1
debug = 0
codegen-units = 256
incremental = false
```

## Platform Requirements

**Development:**
- Rust stable toolchain (rustc, cargo, rustfmt, clippy)
- Python 3.10+ with pip + maturin
- Node.js + npm (for NAPI-RS bindings)
- pre-commit for git hooks
- Optional: CUDA toolkit for GPU inference
- Recommended: 32 GB+ RAM for full mutation testing / loom concurrency tests

**Production:**
- Linux amd64/arm64 (Docker containers)
- Python 3.11+ runtime (for Python bindings)
- ONNX Runtime runtime libraries
- libgomp1, libssl3 system dependencies
- Optional: GPU for CUDA backend

---

*Stack analysis: 2026-08-11 (refreshed from root `Cargo.toml` workspace members + per-crate `Cargo.toml` feature gates; see `STRUCTURE.md` for the directory layout).*
