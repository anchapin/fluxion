# Fluxion — Agent Instructions

## Required Reading

Before modifying physics code or cross-module interfaces, read these (both are source-of-truth, kept in sync via `scripts/check_architecture_drift.py`):

1. **`ARCHITECTURE.md`** — module boundaries, trait contracts, data flow. Source of truth for swap-point traits (`HeatConductionSolver`, `VentilationSchedule`, `ThermalModelTrait`).
2. **`CODEBASE_MAP.md`** — cross-language context (Rust/Python/Node FFI contracts, memory ownership, serialization formats).

**Rule**: Do NOT modify physics code without checking `ARCHITECTURE.md` first. If code doesn't match the documented interfaces, update `ARCHITECTURE.md` to reflect reality OR fix the code.

Companion docs (read when relevant):
- `RULES.md` — hard constraints (numerical-reasoning-via-code, energy balance, ASHRAE 140)
- `CONTRIBUTING.md` — workflow / PR / branch policy (two copies exist — see Toolchain Quirks)
- `docs/KNOWN_ISSUES.md` — open physics limitations. CI gate `scripts/check_known_issues_stale.py` (#1723) fails if the `*Last Updated: YYYY-MM-DD*` line is >60 days old. **Skips (passes) if the file is absent.**
- `docs/ASHRAE140_RESULTS.md` and `docs/ASHRAE140_RESULTS_v0.8.0.md` — current validation pass rates
- `.github/copilot-instructions.md` — longer architecture overview (Batch Oracle pattern, thermal network); written for Copilot but applies to any agent

## Workspace Structure

Cargo workspace. **Root is also the main `fluxion` package** (`default-members = ["."]`), so bare `cargo build`/`cargo test` build the root crate, not the whole workspace.

- **`fluxion`** (root, `src/`) — the engine: `sim/` (thermal model, solar, ventilation), `physics/` (conduction solvers), `ai/` (ONNX surrogates), `validation/` (ASHRAE 140, energy balance), `api/` (axum REST), `python/` + `napi/` (bindings, feature-gated), `interop/` (OSM/gbXML/IFC/FMI), `cli/`, `bin/`.
- **`fluxion-core/`** — dependency-light *leaf* modules (`weather/`, `assembly.rs`, `construction.rs`, `multi_node.rs`, `per_surface_conduction.rs`, `ashrae_cases.rs`, `physics_constants.rs`, …). Built once & cached by cargo-mutants. **Must not depend on sim/physics/ai/validation** (cycle-breaking rule below).
- **Always-built siblings**: `fluxion-grid` (grid-edge electrical), `fluxion-behavior` (thermal comfort), `fluxion-wasm` (wasm-bindgen over fluxion-core + fluxion-fluid).
- **Feature-gated siblings**: `fluxion-cfd` (`--features fluxion-cfd`; FFD airflow), `fluxion-city` (`--features fluxion-city`; urban radiation), `fluxion-fluid` (`--features fluid`; acausal HVAC/fluid port traits — **not** the same as `fluxion-core/src/fluid/`).
- **`fluxion-mcp/`** — MCP server; unconditionally depends on `fluxion` with `features = ["multi-zone"]` + `fluxion-fluid` + `fluxion-toon`. `cargo build -p fluxion-mcp` / `cargo test -p fluxion-mcp`.
- **`crates/`**: `fluxion-toon` (Token-Oriented Object Notation, LLM-friendly; SPEC in `crates/fluxion-toon/SPEC.md`), `fluxion-twin` (digital twin UKF + MQTT telemetry).

**Cycle-breaking rules** (each enforced by a CI script):
- `fluxion-core/src/**/*.rs` must NOT import `crate::sim_*` / `crate::physics_*` / `crate::ai_*` / `crate::validation_*` — guard: `scripts/check_ashrae_cases_cycle.py` (#1441).
- No new `use crate::sim::*` under `src/physics/**` or `use crate::physics::*` under `src/sim/construction.rs` / `src/sim/per_surface_conduction.rs` — guard: `scripts/check_physics_sim_cycle.py` (#2463).
- The `sim::assembly` and `sim::multi_node_thermal` paths in `src/sim/` are thin re-export shims — keep them that way.

Re-export paths preserved across the crate split: `crate::weather::*`, `crate::assembly::*`, `crate::multi_node::*`, `crate::ashrae_cases::*`, `crate::sim::assembly::*`, `crate::sim::multi_node_thermal::*`, `crate::validation::ashrae_140_cases::Orientation`.

## Developer Commands

```bash
# Build & test
cargo build --release                       # primary build (lto="thin", opt-level=3)
cargo test                                  # all unit tests
cargo test -p fluxion <test_name>           # single test (e.g. multi_zone_n_zone_network)
cargo test --profile ci                     # FAST iteration build (opt-level=1, codegen-units=256)
cargo test --test ashrae_140_validation     # ASHRAE 140 validation suite
cargo test --test zone_balance_eplus_isolation                       # energy-conservation gate
LOOM=1 cargo test --features loom           # concurrency tests (needs ~32GB; #1065)
cargo test --features cuda --test surrogate_cuda_smoke               # GPU smoke (skips on CPU-only)

# Drift / cycle / gate checks
python3 scripts/check_architecture_drift.py                           # ARCHITECTURE.md vs code
python3 scripts/check_ashrae_cases_cycle.py                           # sim↔validation cycle
python3 scripts/check_physics_sim_cycle.py                            # physics↔sim cycle (#2463)
python scripts/release_gate_checker.py                                # release-gate evaluation

# Code quality (REQUIRED order)
cargo fmt -- --check                       # CI gate — omit --check to auto-fix
cargo clippy --lib -- -D warnings          # CI's exact clippy invocation
cargo audit                                # also wired into pre-commit

# Bindings (Python 3.10+ / Node)
maturin develop                             # Python: local dev install
(cd npm/ && npm run build)                  # Node: node build.js --release

# Pre-commit (install once)
pip install pre-commit && pre-commit install && pre-commit install --hook-type commit-msg -f
pre-commit run --all-files                  # ruff, black, isort, fmt, cargo-check, cargo-audit,
                                            # batch-oracle-pattern, rust-doc-check, conventional-commit-msg

# Local GitHub Actions (optional): https://github.com/nektos/act
```

**Pre-flight for orchestration / large ops**: run `./scripts/disk-space-check.sh` first. Minimum **10 GB free** (50 GB recommended) — exhaustion has caused credential-lock, PR-creation, and git-ref-lock failures.

## Critical Physics Constants

- **`EXTERIOR_FILM_COEFF = 18.3 W/m²K`** (ASHRAE 140 v2023, vertical surfaces, ~3.4 m/s wind) — in `src/physics/constants/thermal/ashrae_140/v2023.rs`. The legacy `29.3 W/m²K` (6.7 m/s) must NOT appear in any computation path. Guard: `tests/regression_exterior_film_unification.rs`.
- ASHRAE 140 material constants (HW_CONCRETE_K, FOAM_BOARD_K, GYPSUM_K, EXTERIOR_SURFACE_ABSORPTANCE, …) are **inlined at `fluxion_core::assembly` call sites** — the old `crate::physics::constants::thermal::ashrae_140::materials` import path is gone.
- ASHRAE 140 EnergyPlus reference data lives in `tests/reference_data/`.

## Validation Strategy

Bottom-up, module-isolated, EnergyPlus-comparable:
1. **No ASHRAE 140 system-level testing** until individual modules pass E+ reference tests within 1% tolerance.
2. **No parameter tuning** to make system tests pass — fix the underlying math (RULES.md "must-never hardcode results").
3. Test order: **Weather → Solar → Conduction → Ventilation → Zone Balance**.
4. Module comparison tests: `tests/conduction_*_isolation.rs`, `tests/zone_balance_eplus_isolation.rs`, `tests/solar_*_tests.rs`.

Release gates (`release_gates.yaml`): validation min pass rate **60%** (40% for patches; known structural failures: cases **600** and **900**); throughput ≥150 configs/sec; latency ≤10 ms/config; MAE ≤50%. The strict ±15% annual-energy gate for Cases 600/900 is the required branch-protection check `ASHRAE 140 Strict Energy Gate (Issue #1333)`.

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

`rust-tests.yml` runs on every PR (Ubuntu, fast signal) and on `main` push (full 3-OS matrix + release build). Required branch-protection checks (see `release_gates.yaml` → `ci.required_checks` for the canonical list):
- `Test` (matrix: no-default / wiring-tracing / multi-zone)
- `Energy Conservation` (#1295) — greps test output for "violated energy conservation"
- `Rustfmt` (`cargo fmt -- --check`) · `Clippy` (`cargo clippy --lib -- -D warnings`)
- `Known Issues Stale Check` (#1723) · `Ashrae Cases Cycle Check` (#1441) · `Physics-Sim-Cycle-Check` (#2463)
- `CUDA Smoke Test` (#1603) · `ASHRAE 140 Strict Energy Gate (Issue #1333)`
- `Fluxion Determinism Gate (Issue #1351)` · `Fluxion Performance Gate (Issue #1618)` (workflow_run listeners)
- `Architecture Drift Detection` (nightly + on `src/**/*.rs`/`ARCHITECTURE.md` changes)
- `Docs Hygiene Gate` (#2466) · `Code Coverage Gate` (#1932, ratchet; baseline in `validation/coverage_baseline.json`)
- `Mutation Testing (advisory)` (#1891, diff-scoped, non-blocking) — full suite runs nightly against `develop`

Heavy Linux jobs honour `vars.FLUXION_LINUX_RUNNER` (self-hosted Hetzner fallback).

## Environment Variables

- `FLUXION_REST_BIND` / `FLUXION_REST_PORT` — `fluxion-rest` (default `0.0.0.0:8080`; healthcheck `/v1/healthz`).
- `FLUXION_ONNX_MODEL` — explicit ONNX model path (default `models/surrogate_zone_thermal.onnx`; mock fallback when unset).
- `FLUXION_ONNX_BACKEND` — `cpu | cuda | coreml | directml | openvino`; auto-downgrades to `cpu` if `cuda` feature not built.
- `FLUXION_GPU` — `0`/`false` to force CPU inference.
- `DWAVE_API_TOKEN` — required at runtime for the `dwave` feature (D-Wave SAPI REST).
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` + `RUST_MIN_STACK=33554432` — set by Python-bindings CI to avoid linker SIGSEGV.
- `CARGO_BUILD_JOBS=1` — set by Clippy CI to keep peak RSS low.
- `FLUXION_MQTT_ALLOW_INSECURE` — `fluxion-twin` MQTT consumer. Set to a truthy value (`1`/`true`/`yes`/`on`) to permit plaintext broker URLs (`mqtt://`/`tcp://`). **Default: TLS-only** (`mqtts://`, port 8883); plaintext is rejected unless this is set. Local dev only.
- `FLUXION_MQTT_INSECURE` — `fluxion-twin` MQTT consumer. Set to a truthy value to skip TLS server-certificate validation (e.g. for self-signed brokers). **Dangerous** — disables all cert checking; logged as a warning. Local dev only.

## Toolchain Quirks

- **`rust-toolchain.toml`** pins **stable** + rustfmt + clippy. `.rustfmt.toml` sets `edition = "2021"` — without it rustfmt falls back to 2015 and breaks on `?`/`async`. **Stable rustfmt has no `exclude`**: auto-generated fixture data must use `#[rustfmt::skip]` per-item (see `tests/per_tilt_per_azimuth_fixture_data.rs`).
- **Mutation testing** (`cargo mutants`): full suite needs **32 GB+ RAM**; `.cargo/mutants.toml` excludes combinatorial physics files and all of `src/validation/**`. Diff-scoped advisory check on PRs, full suite nightly on 32 GB runner (skips gracefully <16 GB, #2130). Run manually: `cargo mutants --config .cargo/mutants.toml -p fluxion --jobs 2 --baseline skip`.
- **Feature flags (default = none)**: `python-bindings`, `extension-module` (maturin wheel builds, #2532), `napi-bindings`, `ort` (alias `onnx`), `cuda`, `wiring-tracing`, `multi-zone`, `ashrae_140_v2021`, `pr821-diag`, `loom`, `dwave`, `debug-physics`, `kafka`, `fluid`, `gauge-solver`, `fluxion-city`, `fluxion-cfd`, `dhat`. **Default builds skip the ONNX runtime** — opt in via `--features ort` for AI surrogate / mutation tests.
  - `debug-physics` gates `eprintln!` in physics hot loops (#1967) · `fluid` enables `fluxion-fluid` acausal HVAC (#1980/ADR-005) · `gauge-solver` replaces 5R1C/9R4C as primary zone solver (#2304) · `fluxion-city` wires urban radiation (#2344) · `fluxion-cfd` enables FFD/CFD co-simulation (#2460) · `kafka` enables rdkafka telemetry (#2056) · `dhat` enables heap profiling (#2384).
- **Crate size**: `Cargo.toml` `exclude` + `.cargoignore` strip `refdata/`, `data/`, `models/`, `assets/`, `tests/`, `docs/`, `target/`, `Cargo.lock`. Published crate must stay <10 MB.
- **Two `CONTRIBUTING.md` files** (root + `docs/CONTRIBUTING.md`): root is the active short form with the `cargo fmt --check` rustfmt-1.9 quirks and "avoid scope creep on CI failures" guidance; `docs/CONTRIBUTING.md` is the long-form guide. Edit the right one for the change.

## Mathematical Reasoning

**Always write and execute Python code** for calculations — LLMs are unreliable at arithmetic. Use for unit conversions, formula verification, reference-data comparison, solar angles, thermal resistances, statistical analysis. RULES.md makes this a hard **must-always** rule (constraint #0).

## Repository Hygiene

- **Root `.md` allow-list** (only these may live at repo root): `README.md`, `ARCHITECTURE.md`, `CODEBASE_MAP.md`, `CONTRIBUTING.md`, `RULES.md`, `CHANGELOG.md`, `AGENTS.md`. Move transient artifacts (`CASE_*.md`, `BATCH_*.md`, `ISSUE_*.md`, `*_REPORT.md`, session summaries) to `tmp/` or `docs/`. CI gate: `scripts/check_root_md_policy.py` (#2466). (`CLAUDE.md` is auto-generated per-session and `.gitignore`d — never commit it.)
- **`docs/**/*.md` must have a 7-line summary at the top (lines 2–8)**. CI gate: `scripts/check_docs_summaries.py` (#2466). AGENTS.md is exempt. After adding/removing files under `docs/`, run `scripts/generate_doc_inventory.py` to refresh the auto-table in `docs/doc-inventory.md`.
- Issue triage labels: `docs/agents/triage-labels.md`. Issue workflow: `docs/agents/issue-tracker.md`.

## Branch & PR Conventions

- **`develop`** is the default + integration branch. Branch from it (`git checkout develop && git pull && git checkout -b fix/issue-123`); **all PRs target `develop`** (`gh pr create --base develop`). No direct pushes to `develop`.
- **`main`** is release-only. No direct pushes. PRs targeting `main` are **only permitted from `develop`** (enforced by `protect-main-branch.yml` — hotfixes from a feature branch into `main` are auto-rejected). Releases cut by merging `develop` → `main` via a release PR. Use `--no-ff` merges.
- Conventional commit messages: `fix(scope): …`, `feat(scope): …`, `refactor(scope): …`, `perf(scope): …`, `test(scope): …`, `docs(scope): …`.
- **PR body must include `Closes #N` or `Fixes #N`** for the linked issue — orchestration depends on this keyword to auto-close issues.
- Never force-push `main` or `develop`. Hotfixes still go through PR review.

## Key Files (non-obvious)

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` · `CODEBASE_MAP.md` | Source-of-truth contracts (see Required Reading) |
| `src/lib.rs` | PyO3 entrypoint — `Model`, `BatchOracle` (re-exports thermal_model traits, assembly, multi_node, ashrae_cases) |
| `src/physics/solver_trait.rs` · `src/sim/thermal_model.rs` · `src/sim/ventilation.rs` | ML-surrogate swap-point traits |
| `release_gates.yaml` | Required branch-protection checks + thresholds |
| `tests/reference_data/` | EnergyPlus CSV reference data for unit tests |
| `scripts/check_{architecture_drift,ashrae_cases_cycle,physics_sim_cycle}.py` | Drift + cycle-regression guards |
| `scripts/check_{root_md_policy,docs_summaries}.py` · `scripts/generate_doc_inventory.py` | Docs-hygiene gates (#2466) |
| `scripts/release_gate_checker.py` | Validates gates against current results |
| `scripts/coverage_{critical_paths,baseline}.py` · `validation/coverage_baseline.json` | Coverage ratchet (#1932; `0.0` = unenforced) |
| `scripts/disk-space-check.sh` | Pre-flight before orchestration |
