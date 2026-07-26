# Fluxion — Agent Instructions

## Required Reading (MANDATORY)

Before any non-trivial change, read these in order:

1. **`ARCHITECTURE.md`** (root) — source of truth for module boundaries, trait contracts, data flow. Feed the full file to the model on every new session.
2. **`CODEBASE_MAP.md`** (root) — MANDATORY for cross-language context (Rust/Python/Node FFI contracts, memory ownership, serialization formats).

**Rule**: Do NOT modify physics code without checking `ARCHITECTURE.md` first. If the code doesn't match the documented interfaces, update `ARCHITECTURE.md` to reflect reality OR fix the code to match the architecture.

Companion docs (read when relevant):
- `RULES.md` — hard constraints (numerical-reasoning-via-code, energy balance, ASHRAE 140)
- `CONTRIBUTING.md` — workflow / PR / branch policy
- `docs/KNOWN_ISSUES.md` — open physics limitations; CI gate `scripts/check_known_issues_stale.py` (issue #1723) fails if the `*Last Updated: YYYY-MM-DD*` line is >60 days old. The gate **skips (passes)** if the file is absent, so a missing file does not itself fail CI.
- `docs/ASHRAE140_RESULTS.md` and `docs/ASHRAE140_RESULTS_v0.8.0.md` — current validation pass rates

## Workspace Structure

Cargo workspace (root = main `fluxion` package, `default-members = ["."]`):

```
fluxion/                      # main engine crate (src/, benches/, tests/)
  src/
    sim/                      # ThermalModel, solar, ventilation, engine
    physics/                  # conduction solvers (5R1C, CTF, FD, MultiNode)
    ai/                       # ONNX surrogates, ensemble, batch inference
    validation/               # ASHRAE 140, energy balance, cross-validation
    api/                      # axum REST server (openapi.yaml, /v1/*)
    python/                   # PyO3 bindings (cfg(feature = "python-bindings"))
    napi/                     # Node.js bindings (cfg(feature = "napi-bindings"))
    interop/                  # OSM, gbXML, IFC, FMI 2.0 bridges
    cli/                      # fluxion-delta, validation, multi-zone
    bin/                      # fluxion-rest, fluxion, parallel-issue-workflow, …
fluxion-core/                 # workspace member (leaf modules, no sim/physics/ai/validation deps)
  src/weather/                # EPW/TMY3 parsing, psychrometrics, design-day, interpolation
  src/assembly.rs             # BuildingAssembly, AssemblyBuilder, MaterialLayer
  src/multi_node.rs           # ThermalMassNode, MultiNodeThermalMass
  src/ashrae_cases.rs         # Orientation, WindowArea, ConstructionType, … (pure data)
  src/tensor.rs               # geometry tensor types
fluxion-mcp/                  # STANDALONE crate (NOT in workspace) — Model Context Protocol server
```

**Cycle-breaking rule**: `fluxion-core/src/**/*.rs` must NOT import `crate::sim_*`, `crate::physics_*`, `crate::ai_*`, or `crate::validation_*`. CI enforces this via `scripts/check_ashrae_cases_cycle.py` (#1441). The `sim::assembly` and `sim::multi_node_thermal` paths in `src/sim/` are now thin re-export shims — keep them that way.

Re-export paths preserved across the crate split: `crate::weather::*`, `crate::assembly::*`, `crate::multi_node::*`, `crate::ashrae_cases::*`, `crate::sim::assembly::*`, `crate::sim::multi_node_thermal::*`, `crate::validation::ashrae_140_cases::Orientation`.

## Developer Commands

```bash
# Build & test
cargo build --release                       # primary build (lto="thin", opt-level=3)
cargo test                                  # all unit tests
cargo test -p fluxion <test_name>           # single test (e.g. multi_zone_n_zone_network)
cargo test --test ashrae_140_validation     # ASHRAE 140 validation suite
cargo test --test hvac_bestest               # HVAC BESTEST RP-865 scaffold/suite
cargo test --profile ci                     # FAST CI build (opt-level=1, codegen-units=256) — use for iteration
LOOM=1 cargo test --features loom           # concurrency/race-condition tests (needs ~32GB; issue #1065)
cargo test --features cuda --test surrogate_cuda_smoke  # GPU smoke (skips on CPU-only)

# Targeted validation
cargo test --test zone_balance_eplus_isolation                       # energy-conservation gate
cargo test --test ashrae_140_blind_validation -- --nocapture         # blind-mode ASHRAE 140
python scripts/release_gate_checker.py                                # release-gate evaluation
python3 scripts/check_architecture_drift.py                            # ARCHITECTURE.md vs code drift
python3 scripts/check_ashrae_cases_cycle.py                            # sim↔validation cycle regression

# Code quality (REQUIRED order)
cargo fmt -- --check                       # CI gate — omit --check to auto-fix
cargo clippy --all-targets                 # lint (CI runs `cargo clippy --lib -- -D warnings`)
cargo audit                               # also wired into pre-commit

# Python bindings (requires Python 3.10+)
maturin develop                            # local dev install (rebuilds + pip install -e)
maturin build --release                    # produces target/wheels/*.whl

# Node.js bindings (cd npm/, requires @napi-rs/cli)
npm run build                              # node build.js --release

# Pre-commit hooks (install once)
pip install pre-commit && pre-commit install && pre-commit install --hook-type commit-msg -f
pre-commit run --all-files                 # manual run (covers ruff, black, isort, fmt, cargo-check,
                                           #   cargo-audit, batch-oracle-pattern, rust-doc-check)
```

## Critical Physics Constants

- **`EXTERIOR_FILM_COEFF = 18.3 W/m²K`** (ASHRAE 140 v2023, vertical surfaces, ~3.4 m/s wind) — defined in `src/physics/constants/thermal/ashrae_140/v2023.rs`. The legacy `29.3 W/m²K` (6.7 m/s) must NOT appear in any computation path. Guard: `tests/regression_exterior_film_unification.rs`.
- ASHRAE 140 material constants (HW_CONCRETE_K, FOAM_BOARD_K, GYPSUM_K, EXTERIOR_SURFACE_ABSORPTANCE, …) are now inlined at `fluxion_core::assembly` call sites — they were previously imported from `crate::physics::constants::thermal::ashrae_140::materials` and that path is gone.
- ASHRAE 140 reference data lives in `tests/reference_data/` (EnergyPlus CSVs).

## Validation Strategy

**Bottom-up, module-isolated, EnergyPlus-comparable**:
1. **No ASHRAE 140 system-level testing** until individual modules pass E+ reference tests within 1% tolerance.
2. **No parameter tuning** to make system tests pass — fix the underlying math (RULES.md "must-never hardcode results").
3. Test order: Weather → Solar → Conduction → Ventilation → Zone Balance.
4. Module comparison tests: `tests/conduction_*_isolation.rs`, `tests/zone_balance_eplus_isolation.rs`, `tests/solar_*_tests.rs`.

Release gates (see `release_gates.yaml`):
- Validation min pass rate: **60%** (40% for patch releases). Known structural failures: cases **600** and **900**.
- Throughput ≥150 configs/sec, latency ≤10ms/config, MAE ≤50%.
- Strict ±15% annual-energy tolerance gate for Cases 600/900 is the **required branch-protection check** named `ASHRAE 140 Strict Energy Gate (Issue #1333)`. Cross-platform determinism (#1351) and performance (#1618) are also required checks.

## Module Boundaries

```
Weather (fluxion-core/src/weather/)  →  Solar (src/sim/solar.rs)      →  Zone Balance
                                     →  Ventilation (src/sim/ventilation.rs)
                                     →  Conduction (src/physics/solver_trait.rs)
```

ML-surrogate swap-point traits:
- `HeatConductionSolver` (`src/physics/solver_trait.rs`) — 5R1C, CTF, FD, MultiNode
- `VentilationSchedule` (`src/sim/ventilation.rs`) — constant, scheduled, weather-dependent
- `ThermalModelTrait` (`src/sim/thermal_model.rs`) — physics, surrogate, hybrid (HybridThermalModel + HybridRouting)

`ThermalModel` is `Clone`-by-design — `BatchOracle::evaluate_population` uses rayon `par_iter()` at the **population level only**. Nested parallelism in the inner loop causes thread-pool exhaustion; pre-commit hook `.githooks/batch-oracle-check.sh` enforces this on `lib.rs`.

## CI Gates (must stay green)

`rust-tests.yml` runs on every PR (Ubuntu only, fast signal) and on `main` push (full 3-OS matrix + release build). Required branch-protection checks:
- `Test` (matrix: no-default / wiring-tracing / multi-zone)
- `Energy Conservation` (Issue #1295) — grep for "violated energy conservation" in test output
- `Rustfmt` — `cargo fmt -- --check`
- `Clippy` — `cargo clippy --lib -- -D warnings`
- `Known Issues Stale Check` (Issue #1723)
- `Ashrae Cases Cycle Check` (Issue #1441) — runs `scripts/check_ashrae_cases_cycle.py`
- `CUDA Smoke Test` (Issue #1603) — `cargo build --features cuda` + `cargo test --test surrogate_cuda_smoke --features cuda`
- `ASHRAE 140 Strict Energy Gate (Issue #1333)` (4 tests, named explicitly in workflow)
- `Fluxion Determinism Gate (Issue #1351)` — listener on Cross-Platform Determinism CI workflow
- `Fluxion Performance Gate (Issue #1618)` — listener on Performance Dashboard workflow
- `Architecture Drift Detection` (nightly + on `src/**/*.rs` / `ARCHITECTURE.md` changes) — `scripts/check_architecture_drift.py`

Heavy Linux jobs honour `vars.FLUXION_LINUX_RUNNER` (self-hosted Hetzner fallback; see `docs/self-hosted-runners.md`).

## Environment Variables

- `FLUXION_REST_BIND` / `FLUXION_REST_PORT` — `fluxion-rest` binary (default `0.0.0.0:8080`, healthcheck `/v1/healthz`).
- `FLUXION_ONNX_MODEL` — explicit ONNX model path (`models/surrogate_zone_thermal.onnx` default; mock fallback when unset).
- `FLUXION_ONNX_BACKEND` — `cpu | cuda | coreml | directml | openvino`; auto-downgrades to `cpu` if `cuda` feature not built.
- `FLUXION_GPU` — `0`/`false` to force CPU inference.
- `DWAVE_API_TOKEN` — required at runtime for the `dwave` feature (D-Wave SAPI REST).
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` and `RUST_MIN_STACK=33554432` — set by Python-bindings CI to avoid linker SIGSEGV.
- `CARGO_BUILD_JOBS=1` — set by Clippy CI to keep peak RSS low.

## Toolchain Quirks

- **`rust-toolchain.toml`** pins **stable** + rustfmt + clippy. `.rustfmt.toml` sets `edition = "2021"` — without it rustfmt falls back to 2015 and breaks on `?`/`async`. Stable rustfmt does NOT support `exclude`; auto-generated fixture data must use `#[rustfmt::skip]` per-item (see `tests/per_tilt_per_azimuth_fixture_data.rs`).
- **Mutation testing (`cargo mutants`)**: requires **32 GB+ RAM**. `.cargo/mutants.toml` excludes combinatorial physics files (`state_space_ctf`, `multi_node_solver`, `ctf_coefficients`, `fd_*`, `ctf_*`, `geometry_tensor`, `cta`, `thermal_mass/**`) and the entire `src/validation/**` tree. Run manually: `cargo mutants --config .cargo/mutants.toml -p fluxion --jobs 2 --baseline skip`.
- **Feature flags** (default = none): `python-bindings`, `napi-bindings`, `ort` (alias `onnx`), `cuda`, `wiring-tracing`, `multi-zone`, `ashrae_140_v2021`, `pr821-diag`, `loom`, `dwave`. Default builds skip the ONNX runtime — opt in via `--features ort` for AI surrogate / mutation tests.
- **Crate size**: `Cargo.toml` `exclude` + `.cargoignore` strip `refdata/`, `data/`, `models/`, `assets/`, `tests/`, `docs/`, `target/`, `Cargo.lock`, etc. Published crate must stay under 10 MB.
- **Two CONTRIBUTING.md files** (root + `docs/CONTRIBUTING.md`) — root is the active short form; `docs/CONTRIBUTING.md` has the long-form guide.
- **`docs/CONTRIBUTING.md`** says `*always run tests, format, clippy*` — root CONTRIBUTING.md has the `cargo fmt --check` rustfmt-1.9 quirks and "avoid scope creep on CI failures" guidance that the docs file lacks.

## Mathematical Reasoning

**Always write Python code** (`ctx_execute language:"python"`) for calculations — LLMs are unreliable at arithmetic. Use for: unit conversions, formula verification, reference data comparison, solar angles, thermal resistances, statistical analysis. RULES.md makes this a hard "must-always" rule.

## Repository Hygiene

- **Root `.md` policy**: standing docs (`README.md`, `ARCHITECTURE.md`, `CODEBASE_MAP.md`, `CONTRIBUTING.md`, `RULES.md`, `CHANGELOG.md`, `AGENTS.md`) are the only root `.md` files that belong in a commit. Transient artifacts (analyses, session summaries, plans, `CASE_*.md`, `BATCH_*.md`, `*_REPORT.md`) must move to `tmp/` or `docs/` before commit — CI and `copilot-instructions.md` enforce this.
- All system docs in `docs/` must have a **7-line summary** at the top (lines 2–8). See `docs/doc-inventory.md`. AGENTS.md itself is exempt.
- Issue triage labels: see `docs/agents/triage-labels.md`. GitHub issue workflow: `gh issue create --title "..." --body "..." --label "..."` (per `docs/agents/issue-tracker.md`).

## Branch & PR Conventions

- **`develop`** is the default branch and integration branch.
  - All new feature branches must be created from `develop`: `git checkout develop && git pull && git checkout -b fix/issue-123`.
  - All PRs must target `develop`: `gh pr create --base develop`.
  - Direct pushes to `develop` are prohibited — all changes go through PR review.
- **`main`** is release-only.
  - No direct pushes to `main` — branch protection enforces PR-only flow.
  - PRs targeting `main` are **only permitted from the `develop` branch** (enforced by the `protect-main-branch.yml` workflow). Hotfixes targeting `main` directly from a feature branch will be automatically rejected by CI.
  - Releases are cut by merging `develop` → `main` via a release PR.
- `--no-ff` merges preserve history (CONTRIBUTING.md).
- Conventional commit messages: `fix(scope): …`, `feat(scope): …`, `refactor(scope): …`, `perf(scope): …`, `test(scope): …`, `docs(scope): …`.
- Never force-push `main` or `develop`. Hotfixes still go through PR review.

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Module boundaries, I/O contracts, trait hierarchies (~1000 lines, source of truth) |
| `CODEBASE_MAP.md` | Cross-language context: FFI contracts, module dependency graph |
| `src/lib.rs` | PyO3 entrypoint — `Model`, `BatchOracle` (reexports thermal_model traits, assembly, multi_node, ashrae_cases) |
| `src/physics/solver_trait.rs` | `HeatConductionSolver` trait (5R1C/CTF/FD/MultiNode) |
| `src/sim/thermal_model.rs` | `ThermalModelTrait` + `HybridRouting` |
| `src/sim/solar.rs` | Solar position and irradiance |
| `src/sim/ventilation.rs` | `VentilationSchedule` trait |
| `src/physics/multi_node_solver.rs` | 9R4C multi-node solver (ADR-002) |
| `src/ai/surrogate.rs` | `SurrogateManager`, ONNX runtime, env-var resolution (FLUXION_ONNX_*) |
| `src/api/server.rs` + `openapi.yaml` | axum REST API (port 8080, `/v1/healthz`, `/v1/metrics`) |
| `fluxion-core/src/weather/` | EPW/TMY3, psychrometrics, design-day, interpolation |
| `tests/reference_data/` | EnergyPlus CSV reference data for unit tests |
| `release_gates.yaml` | Required branch-protection checks + thresholds |
| `scripts/check_architecture_drift.py` | ARCHITECTURE.md vs source-code drift |
| `scripts/check_ashrae_cases_cycle.py` | `sim ↔ validation` cycle regression guard |
| `scripts/release_gate_checker.py` | Validates `release_gates.yaml` gates against current results |
