# Codebase Structure

**Analysis Date:** 2026-08-11
**Layout:** Post-#1255 multi-crate Cargo workspace (1 root package + 10 workspace members).

## Workspace Layout

```
fluxion/                          # Cargo workspace root (also the main `fluxion` package)
├── src/                          # Root `fluxion` crate — the engine
│   ├── ai/                       # ONNX surrogate models and inference
│   ├── api/                      # Rust-native axum REST server (fluxion-rest)
│   ├── bin/                      # Rust CLI binaries (fluxion, fluxion_rest, export_csv, …)
│   ├── cli/                      # CLI module
│   ├── interop/                  # OSM / gbXML / IFC / FMI interoperability
│   ├── physics/                  # Conduction solvers (HeatConductionSolver trait)
│   ├── python/                   # PyO3 bindings (feature-gated: python-bindings)
│   ├── napi/                     # NAPI-RS Node.js bindings (feature-gated: napi-bindings)
│   ├── sim/                      # Physics-based building simulation (thermal model, solar, ventilation)
│   ├── validation/               # ASHRAE 140 validation framework + energy balance
│   └── lib.rs                    # PyO3 entrypoint — Model, BatchOracle
│
├── fluxion-core/                 # Dependency-light leaf modules (cycle-breaking crate, #1255)
│   └── src/
│       ├── weather/              # EPW / TMY weather parsing
│       ├── fluid/                # Core fluid types (NOT the same as fluxion-fluid crate)
│       ├── assembly.rs           # Construction assembly + ASHRAE 140 material constants
│       ├── construction.rs       # Construction layer modeling
│       ├── multi_node.rs         # Multi-node thermal network
│       ├── per_surface_conduction.rs
│       ├── ashrae_cases.rs       # ASHRAE 140 case definitions
│       ├── physics_constants.rs
│       ├── urban_radiation.rs
│       ├── earth_tube.rs
│       └── tensor.rs
│
├── fluxion-grid/                 # Grid-edge electrical: battery, bus, power flow, PV
├── fluxion-behavior/             # Thermal comfort (Fanger PMV/PPD, adaptive), occupancy, lighting
├── fluxion-wasm/                 # WebAssembly bindings (fluxion-core + fluxion-fluid)
├── fluxion-city/                 # Urban radiation modeling (feature-gated: fluxion-city, #2344)
├── fluxion-fluid/                # Acausal HVAC / fluid port traits (feature-gated: fluid, #1980)
├── fluxion-cfd/                  # Fast Fluid Dynamics solver (feature-gated: fluxion-cfd, #2460)
├── fluxion-mcp/                  # MCP server (depends on fluxion + multi-zone + fluid + toon)
│
├── crates/
│   ├── fluxion-toon/             # Token-Oriented Object Notation (LLM-friendly serializer)
│   └── fluxion-twin/             # Digital twin: Unscented Kalman Filter + MQTT telemetry
│
├── api/                          # Legacy Python FastAPI REST server (superseded by src/api/)
├── npm/                          # Node.js NAPI-RS binding sources + build (node build.js --release)
├── tests/                        # Integration tests + reference_data/ (EnergyPlus CSVs)
├── benches/                      # Rust benchmarks
├── tools/                        # Python training / benchmarking scripts
├── examples/                     # Usage examples
├── docs/                         # Documentation (architecture, ASHRAE 140 results, known issues)
├── scripts/                      # CI guards: drift / cycle / docs-hygiene / release-gate checks
├── models/                       # Trained ONNX models (gitignored)
├── refdata/                      # Reference data (excluded from published crate)
├── data/                         # Data files (excluded from published crate)
├── validation/                   # Validation harness + coverage baseline
├── measures/                     # OpenStudio measures (Ruby)
├── .planning/                    # Planning documents (this file)
├── .github/                      # GitHub Actions CI/CD workflows
├── .githooks/                    # Custom git hooks (batch-oracle-pattern, rust-doc-check)
│
├── Cargo.toml                    # Workspace + root package config (members, features, profiles)
├── Cargo.lock                    # Workspace lockfile (root)
├── rust-toolchain.toml           # Stable toolchain pin (+ rustfmt + clippy)
├── .rustfmt.toml                 # edition = "2021" (required)
├── pyproject.toml                # Python packaging (PEP 621)
├── release_gates.yaml            # Required branch-protection checks + thresholds
├── ARCHITECTURE.md               # Source-of-truth module boundaries + trait contracts
├── CODEBASE_MAP.md               # Cross-language FFI contracts (Rust/Python/Node)
├── RULES.md                      # Hard constraints (numerical-reasoning-via-code, energy balance)
├── CONTRIBUTING.md               # Workflow / PR / branch policy
├── SCORECARD.md                  # Auto-generated release-readiness scorecard
├── CHANGELOG.md                  # Versioned release notes
├── AGENTS.md                     # Agent instructions (this project)
└── README.md                     # Project overview
```

`default-members = ["."]` — bare `cargo build` / `cargo test` build the root `fluxion` crate only. Workspace `members = ["fluxion-core", "fluxion-city", "fluxion-fluid", "fluxion-grid", "fluxion-behavior", "fluxion-mcp", "fluxion-wasm", "crates/fluxion-toon", "crates/fluxion-twin", "fluxion-cfd"]`.

## Workspace Crate Purposes

### `fluxion` (root, `src/`)

- **Purpose:** Core Rust building energy modeling engine — the primary crate.
- **Contains:** Physics engine (`sim/`, `physics/`), AI surrogates (`ai/`), validation framework (`validation/`), axum REST API (`api/`), Python / Node bindings (`python/`, `napi/`), interop (`interop/`), CLI (`cli/`, `bin/`).
- **Key files:**
  - `src/lib.rs` — PyO3 entrypoint; re-exports `Model`, `BatchOracle`, thermal_model traits, assembly, multi_node, ashrae_cases
  - `src/physics/solver_trait.rs` — `HeatConductionSolver` swap-point trait (5R1C, CTF, FD, MultiNode)
  - `src/sim/thermal_model.rs` — `ThermalModelTrait` swap-point trait (physics, surrogate, hybrid)
  - `src/sim/ventilation.rs` — `VentilationSchedule` swap-point trait
  - `src/api/server.rs` — Rust-native axum REST server (binary `fluxion-rest`)
  - `src/bin/fluxion_rest.rs`, `src/bin/fluxion.rs`, `src/bin/run_ashrae_validation.rs` — CLI binaries

### `fluxion-core/`

- **Purpose:** Dependency-light *leaf* modules extracted to break the `sim` ↔ `validation` cycle (#1255). Built once & cached by cargo-mutants.
- **Cycle rule:** `fluxion-core/src/**/*.rs` must NOT import `crate::sim_*` / `crate::physics_*` / `crate::ai_*` / `crate::validation_*` — enforced by `scripts/check_ashrae_cases_cycle.py` (#1441).
- **Key files:** `weather/` (EPW / TMY), `assembly.rs` (inlined ASHRAE 140 material constants), `multi_node.rs`, `ashrae_cases.rs` (incl. `Orientation`), `per_surface_conduction.rs`, `physics_constants.rs`, `urban_radiation.rs`, `earth_tube.rs`, `tensor.rs`, `fluid/` (core fluid types).
- **Re-exports:** `crate::weather::*`, `crate::assembly::*`, `crate::multi_node::*`, `crate::ashrae_cases::*`, `crate::sim::assembly::*`, `crate::sim::multi_node_thermal::*`, `crate::validation::ashrae_140_cases::Orientation` are preserved from the root crate as thin shims.

### `fluxion-grid/`

- **Purpose:** Grid-edge electrical network components — battery storage, bus nodes, power flow, PV, joint thermal-electrical convergence (`ThermalElectricalCoupler`).
- **Features:** `fluxion-integration` (alias `fluxion`) for `Arc<dyn ThermalModelTrait>` coupling; `fluid` for `fluxion-fluid` HvacState coupling.

### `fluxion-behavior/`

- **Purpose:** Behavioral and thermal comfort models — Fanger PMV/PPD, adaptive comfort, Markov + deterministic occupancy, lighting, plug loads, internal gains, moisture, occupant triggers, TOON time encoder.
- **Features:** default = `["ort"]`.

### `fluxion-wasm/`

- **Purpose:** WebAssembly bindings over `fluxion-core` + `fluxion-fluid` via `wasm-bindgen`.

### `fluxion-city/` (feature-gated)

- **Purpose:** Urban radiation modeling with Nusselt-analog view factor computation (Issue #2344). Wires `UrbanRadiationSolver` into `PhysicsSurfaceFluxProvider` via `FluxionCitySurfaceFluxProvider`.
- **Root feature:** `fluxion-city = ["dep:fluxion-city"]`.
- **Crate features:** `parallel = ["rayon"]`.

### `fluxion-fluid/` (feature-gated)

- **Purpose:** Compile-time strongly typed fluid port traits for acausal HVAC / fluid DAE systems (Issue #1980 / ADR-0005). **Not** the same as `fluxion-core/src/fluid/`.
- **Root feature:** `fluid = ["dep:fluxion-fluid"]`.
- **Contains:** `port.rs`, `ports/`, `medium.rs`, `mediums/`, `properties.rs`, `energy.rs`, `hvac.rs`, `ecs/`, `autodiff/`, `pantelides.rs`.

### `fluxion-cfd/` (feature-gated)

- **Purpose:** GPU-accelerated Fast Fluid Dynamics (FFD) solver for building airflow simulation (Issue #2460). CPU / CUDA / OpenCL backends; wires `fluxion_cfd::FfdCfdSolver` into the loose-coupling `FfdSolver` trait.
- **Root feature:** `fluxion-cfd = ["dep:fluxion-cfd"]`.
- **Features:** default = `["cpu"]`; `cuda`, `opencl`.

### `fluxion-mcp/`

- **Purpose:** Model Context Protocol server for Rust-native BEM interface.
- **Dependency:** Unconditionally depends on `fluxion` with `multi-zone` + `fluxion-fluid` + `fluxion-toon`.
- **Features:** default = `["multi-zone"]` (gated behind the crate's own feature so workspace `--no-default-features` builds don't force `multi-zone` onto every member; see issue #2540).
- **Build:** `cargo build -p fluxion-mcp` / `cargo test -p fluxion-mcp`.

### `crates/fluxion-toon/`

- **Purpose:** Token-Oriented Object Notation (TOON) — compact, LLM-friendly serializer/deserializer. SPEC in `crates/fluxion-toon/SPEC.md`.
- **Features:** default = `[]`; `std`.

### `crates/fluxion-twin/`

- **Purpose:** Digital twin core — Unscented Kalman Filter for non-linear state estimation in thermal systems; MQTT telemetry consumer (TLS-only `mqtts://` port 8883 by default; plaintext gated on `FLUXION_MQTT_ALLOW_INSECURE`).

## Module Boundaries (root `src/`)

```
Weather (fluxion-core/src/weather/)  →  Solar (src/sim/solar.rs)      →  Zone Balance
                                     →  Ventilation (src/sim/ventilation.rs)
                                     →  Conduction (src/physics/solver_trait.rs)
```

**ML-surrogate swap-point traits** (see `ARCHITECTURE.md` for full contracts):
- `HeatConductionSolver` (`src/physics/solver_trait.rs`) — 5R1C, CTF, FD, MultiNode
- `VentilationSchedule` (`src/sim/ventilation.rs`) — constant, scheduled, weather-dependent
- `ThermalModelTrait` (`src/sim/thermal_model.rs`) — physics, surrogate, hybrid (`HybridThermalModel` + `HybridRouting`)

`ThermalModel` is `Clone`-by-design — `BatchOracle::evaluate_population` uses rayon `par_iter()` at the **population level only**. Nested parallelism in the inner loop causes thread-pool exhaustion; pre-commit hook `.githooks/batch-oracle-check.sh` enforces this on `lib.rs`.

## Key File Locations

**Entry Points:**
- `src/lib.rs` — PyO3 module definition (`BatchOracle`, `Model`)
- `src/bin/fluxion.rs` — Rust CLI (`fluxion validate --all`, etc.)
- `src/bin/fluxion_rest.rs` — Rust-native axum REST server (`fluxion-rest`, default `0.0.0.0:8080`)
- `fluxion-mcp/src/main.rs` — MCP server binary (`fluxion-mcp`)
- `api/main.py` — *Legacy* Python FastAPI server (superseded by `src/api/`)

**Configuration:**
- `Cargo.toml` — Workspace + root package (members, features, profiles)
- `pyproject.toml` — Python packaging (PEP 621)
- `rust-toolchain.toml` — Stable toolchain pin
- `.rustfmt.toml` — `edition = "2021"`
- `.pre-commit-config.yaml` — Pre-commit hooks
- `release_gates.yaml` — Required branch-protection checks + thresholds

**Source-of-truth contracts:**
- `ARCHITECTURE.md` — Module boundaries, trait contracts, data flow (checked by `scripts/check_architecture_drift.py`)
- `CODEBASE_MAP.md` — Cross-language FFI contracts (Rust / Python / Node), memory ownership, serialization

**Validation:**
- `src/validation/` — ASHRAE 140 validator, case specs, diagnostic tools
- `tests/ashrae_140_validation.rs`, `tests/zone_balance_eplus_isolation.rs` — Validation suites
- `tests/reference_data/` — EnergyPlus CSV reference data
- `validation/coverage_baseline.json` — Coverage ratchet baseline (#1932)

**Drift / cycle / docs-hygiene guards:**
- `scripts/check_architecture_drift.py` — `ARCHITECTURE.md` vs code
- `scripts/check_ashrae_cases_cycle.py` — `fluxion-core` cycle guard (#1441)
- `scripts/check_physics_sim_cycle.py` — physics ↔ sim cycle guard (#2463)
- `scripts/check_root_md_policy.py` / `scripts/check_docs_summaries.py` — docs hygiene (#2466)
- `scripts/check_known_issues_stale.py` — `docs/KNOWN_ISSUES.md` freshness (#1723)
- `scripts/release_gate_checker.py` — release-gate evaluation

## Naming Conventions

**Files:**
- Rust source: `snake_case.rs` (e.g., `thermal_model.rs`, `solver_trait.rs`)
- Python source: `snake_case.py` (e.g., `main.py`, `train_surrogate.py`)
- Module directories: `snake_case/` (e.g., `src/sim/`, `fluxion-core/src/weather/`)

**Types:**
- Rust structs / enums / traits: `PascalCase` (e.g., `ThermalModel`, `HeatConductionSolver`, `ThermalModelTrait`, `BatchOracle`)
- Python classes: `PascalCase` (e.g., `BatchOracle`, `Model`)

**Constants:**
- Rust / Python: `SCREAMING_SNAKE_CASE` (e.g., `EXTERIOR_FILM_COEFF`, `HOURS_PER_YEAR`)

## Where to Add New Code

**New physics feature:**
- Primary: `src/sim/[feature].rs` or `src/physics/[feature].rs`
- Tests: `tests/[feature].rs`
- Module wire-up: add `pub mod [feature];` to the relevant `mod.rs`
- ⚠️ Check `ARCHITECTURE.md` first — do NOT modify physics code without verifying the documented interfaces

**New leaf module (cycle-safe):**
- Add to `fluxion-core/src/[module].rs` (must not import `crate::sim_*` / `physics_*` / `ai_*` / `validation_*`)

**New workspace crate:**
- Add the crate directory, add to `members = [...]` in root `Cargo.toml`, and (if optional on the root) wire a feature flag in `[features]`

**New ASHRAE 140 case:**
- Implementation: `fluxion-core/src/ashrae_cases.rs`
- Tests: `tests/` (follow the existing `tests/ashrae_140_*` pattern)
- Reference data: `tests/reference_data/`

**New Rust CLI binary:**
- Source: `src/bin/[name].rs`
- Wire-up: add `[[bin]]` to root `Cargo.toml`

**New Rust benchmark:**
- Source: `benches/[name].rs`
- Wire-up: add `[[bench]] name = "[name]"` to root `Cargo.toml`

## Special Directories

**`models/`:** Trained ONNX models for surrogate inference (gitignored; mock fallback when unset).

**`.planning/`:** Planning documents and codebase analysis (this file). Not subject to the root-`.md` allow-list or docs-summary gate — it is project-internal planning.

**`target/`:** Rust build artifacts (gitignored).

**`.venv/`, `venv/`:** Python virtual environments (gitignored).

**`worktrees/`:** Git worktrees for parallel development (gitignored).

**`.github/`:** GitHub Actions CI/CD workflows (committed).

**`.githooks/`:** Custom git hooks (`batch-oracle-check.sh`, `rust-doc-check`; committed).

**`refdata/`, `data/`, `assets/`, `tests/`, `docs/`, `tools/`, `benches/`, `examples/`:** Excluded from the published crate via `Cargo.toml` `exclude` + `.cargoignore` (crate must stay <10 MB).

---

*Structure analysis: 2026-08-11 (post-#1255 multi-crate workspace layout; 11 crates total — 1 root + 10 members).*
