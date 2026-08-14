# Fluxion — Agent Instructions

Building Energy Modeling engine (Rust + Python + Node bindings). Mid-milestone on v1.3 "Blind ASHRAE 140 Validation". See `README.md` for status (current pass rate ~20%) and `SCORECARD.md` for the consolidated release-readiness view.

## Required Reading

Before modifying physics or cross-module interfaces, read both (source of truth; synced via `scripts/check_architecture_drift.py`):

1. **`ARCHITECTURE.md`** — module boundaries, trait contracts, data flow. Canonical for swap-point traits (`HeatConductionSolver`, `VentilationSchedule`, `ThermalModelTrait`).
2. **`CODEBASE_MAP.md`** — cross-language FFI contracts, memory ownership, serialization formats.

**Rule:** do NOT modify physics code without checking `ARCHITECTURE.md` first. If code drifts, update the doc OR fix the code — never let them diverge.

Companion docs (read when relevant):
- `RULES.md` — hard constraints (numerical-reasoning-via-code, energy balance, ASHRAE 140).
- `CONTRIBUTING.md` (root) — workflow / PR / branch policy. There is also `docs/CONTRIBUTING.md` (long-form); edit the right one.
- `docs/KNOWN_ISSUES.md` — open physics limitations. `*Last Updated: YYYY-MM-DD*` line is checked by `scripts/check_known_issues_stale.py` (>60d → fail). Gate skips (passes) if the file is absent.
- `docs/ASHRAE140_RESULTS.md` · `docs/adr/` · `.github/copilot-instructions.md` — current pass rates, ADRs, longer architecture overview (Batch Oracle pattern, thermal network).

## Workspace Structure

Cargo workspace. The root `Cargo.toml` is also the `fluxion` package; bare `cargo build`/`cargo test` run against it (default-members is implicit — only `cargo test --workspace` or `cargo build -p <pkg>` reach siblings).

- **`fluxion`** (root, `src/`) — engine: `sim/` (thermal model, solar, ventilation), `physics/` (conduction solvers), `ai/` (ONNX surrogates), `validation/` (ASHRAE 140, energy balance), `api/` (axum REST), `python/` + `napi/` (bindings, feature-gated), `interop/` (OSM/gbXML/IFC/FMI), `cli/`, `bin/`. Binaries: `fluxion` (CLI; `src/bin/fluxion.rs`), `fluxion-rest` (axum API), `fluxion-delta` (diff tool), plus `export_csv` and four `run_{ashrae,cross,multi_zone}_validation` helpers — all via `cargo run --bin <name>`.
- **`fluxion-core/`** — dependency-light *leaf* modules (`weather/`, `assembly.rs`, `construction.rs`, `multi_node.rs`, `per_surface_conduction.rs`, `ashrae_cases.rs`, `physics_constants.rs`). **Must NOT import `crate::sim_*` / `crate::physics_*` / `crate::ai_*` / `crate::validation_*`** — guard: `scripts/check_ashrae_cases_cycle.py` (#1441).
- **Always-built siblings**: `fluxion-grid`, `fluxion-behavior`, `fluxion-wasm` (wasm-bindgen over `fluxion-core` + `fluxion-fluid`).
- **Feature-gated siblings**: `fluxion-cfd` (FFD airflow), `fluxion-city` (urban radiation), `fluxion-fluid` (acausal HVAC/fluid port traits — **not** the same as `fluxion-core/src/fluid/`).
- **`fluxion-mcp/`** — MCP server, separate package: `cargo build -p fluxion-mcp` / `cargo test -p fluxion-mcp` (do not use `--bin`). Depends on `fluxion` with `default-features = false`; enables `multi-zone` via its own default feature (#2540); unconditionally pulls `fluxion-fluid` + `fluxion-toon`.
- **`crates/`**: `fluxion-toon` (LLM-friendly token format; SPEC in `crates/fluxion-toon/SPEC.md`), `fluxion-twin` (digital twin UKF + MQTT).
- **Orchestration config at repo root**: `agent-orchestrator.yaml` (multi-agent task graph), `bernstein.yaml` (project context). `.opencode/plugins/` holds local OpenCode plugins (`agent-review`, `task-queue`) — agents on this machine may load them via `.opencode/package.json`.

Cycle-breaking rules (each enforced by CI):
- `fluxion-core/src/**/*.rs` may not import sim/physics/ai/validation — see above.
- No new `use crate::sim::*` under `src/physics/**`; no `use crate::physics::*` under `src/sim/construction.rs` or `src/sim/per_surface_conduction.rs` — guard: `scripts/check_physics_sim_cycle.py` (#2463).
- `src/sim/assembly.rs` and `src/sim/multi_node_thermal.rs` are thin re-export shims — keep them that way.

Re-export paths preserved across the crate split: `crate::weather::*`, `crate::assembly::*`, `crate::multi_node::*`, `crate::ashrae_cases::*`, `crate::sim::assembly::*`, `crate::sim::multi_node_thermal::*`, `crate::validation::ashrae_140_cases::Orientation`.

## Developer Commands

```bash
# Build & test
cargo build --release                       # primary build (lto="thin", opt-level=3)
cargo test                                  # all unit tests (root crate, default)
cargo test -p fluxion <test_name>           # single test
cargo test --profile ci                     # FAST iteration build (opt-level=1, codegen-units=256)
cargo test --test ashrae_140_validation     # ASHRAE 140 suite
cargo test --test zone_balance_eplus_isolation                       # energy-conservation gate
LOOM=1 cargo test --features loom           # concurrency tests (needs ~32 GB; #1065)
cargo test --features cuda --test surrogate_cuda_smoke               # GPU smoke (skips on CPU-only)
cargo test --features ort                   # opt in to ONNX runtime (off by default)

# Cycle / drift / gate checks
python3 scripts/check_architecture_drift.py                 # ARCHITECTURE.md vs code
python3 scripts/check_ashrae_cases_cycle.py                 # sim↔validation cycle
python3 scripts/check_physics_sim_cycle.py                  # physics↔sim cycle (#2463)
python3 scripts/check_root_hygiene.py                       # root-allow-list + scratch-blob gate (#2466/2814)
python3 scripts/check_known_issues_stale.py                 # KNOWN_ISSUES Last-Updated freshness (#1723)
python3 scripts/check_doc_inventory_fresh.py                # docs/doc-inventory.md freshness (#2765)
python3 scripts/check_strict_energy_gate_regression.py      # ±15% Cases 600/900 regression (#1333)
python  scripts/release_gate_checker.py                     # release-gate evaluation

# Code quality (REQUIRED order — mirrors CI)
cargo fmt -- --check                       # omit --check to auto-fix
cargo clippy --lib -- -D warnings          # CI's exact clippy invocation
cargo audit && cargo deny check            # supply-chain gates (#2699; config: deny.toml + .cargo/audit.toml)

# Bindings (Python 3.10+ / Node)
maturin develop                             # Python: local dev install
(cd npm/ && npm run build)                  # Node: node build.js --release

# Binaries
cargo run --bin fluxion                  # primary CLI (sim / validate / batch)
cargo run --bin fluxion-rest             # axum REST API (FLUXION_REST_* env; GET /v1/healthz)
cargo run --bin fluxion-delta            # diff tool (tools/fluxion_delta.rs)
cargo run -p fluxion-mcp                 # MCP server — SEPARATE package; do NOT use --bin

# Pre-commit (install once)
pip install pre-commit && pre-commit install && pre-commit install --hook-type commit-msg -f
pre-commit run --all-files                  # ruff, black, fmt, cargo-check, cargo-audit, batch-oracle-pattern,
                                            # rust-doc-check, conventional-commit-msg
```

**Pre-flight for orchestration / large ops**: `./scripts/disk-space-check.sh`. Minimum **10 GB free** (50 GB recommended) — exhaustion has caused credential-lock, PR-creation, and git-ref-lock failures.

## Critical Physics Constants

- **`EXTERIOR_FILM_COEFF = 18.3 W/m²K`** (ASHRAE 140 v2023, vertical surfaces, ~3.4 m/s wind) — `src/physics/constants/thermal/ashrae_140/v2023.rs`. The legacy `29.3 W/m²K` (6.7 m/s) must NOT appear in any computation path. Guard: `tests/regression_exterior_film_unification.rs`.
- ASHRAE 140 material constants (`HW_CONCRETE_K`, `FOAM_BOARD_K`, `GYPSUM_K`, `EXTERIOR_SURFACE_ABSORPTANCE`, …) are **inlined at `fluxion_core::assembly` call sites** — the old `crate::physics::constants::thermal::ashrae_140::materials` path is gone.
- ASHRAE 140 EnergyPlus reference data: `tests/reference_data/`.

## Validation Strategy

Bottom-up, module-isolated, EnergyPlus-comparable:
1. **No ASHRAE 140 system-level testing** until individual modules pass E+ reference tests within 1% tolerance.
2. **No parameter tuning** to make tests pass — fix the underlying math (RULES.md "must-never hardcode results"). The strict ±15% Case 600/900 cooling gap is a known structural failure; the baseline in `tests/reference_data/zone_balance/strict_energy_gate_baseline.json` must NEVER be raised to hide a regression.
3. Test order: **Weather → Solar → Conduction → Ventilation → Zone Balance**.
4. Module comparison tests: `tests/conduction_*_isolation.rs`, `tests/zone_balance_eplus_isolation.rs`, `tests/solar_*_tests.rs`.

Release gates (`release_gates.yaml`): validation min pass rate **60%** (40% for patches; known structural failures: cases **600** and **900**); throughput ≥150 cfg/s; latency ≤10 ms/config; multi-zone ≥10 cfg/s on 10-zone pop_1000 (#2772); MAE ≤50%. The strict ±15% annual-energy gate (#1333) compares each Case 600/900 metric's gap-from-band against the baseline (`tests/reference_data/zone_balance/strict_energy_gate_baseline.json`) and fails only on regression beyond `regression_tolerance_pp` (5 pp).

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

`ThermalModel` is `Clone`-by-design — `BatchOracle::evaluate_population` uses rayon `par_iter()` at the **population level only**. Nested parallelism in the inner loop exhausts the thread pool; pre-commit hook `.githooks/batch-oracle-check.sh` enforces this on `lib.rs`.

## CI Gates (must stay green)

`rust-tests.yml` runs on every PR (Ubuntu, fast signal) and on `main` push (full 3-OS matrix + release build). The canonical required-check list lives in `release_gates.yaml → ci.required_checks`. Headliners:

- `Test` (matrix: no-default / wiring-tracing / multi-zone) · `Energy Conservation` (#1295) · `Rustfmt` · `Clippy` (`-D warnings`)
- `Known Issues Stale Check` (#1723) · `Ashrae Cases Cycle Check` (#1441) · `Physics-Sim-Cycle-Check` (#2463)
- `ASHRAE 140 Strict Energy Gate (Issue #1333)` · `CUDA Smoke Test` (#1603)
- `Fluxion Determinism Gate (#1351)` · `Fluxion Performance Gate (#1618)` — relative 5% (workflow_run listeners)
- `Absolute Perf Gate (#2693)` — PR-blocking floor: ≥150 cfg/s AND ≤10 ms/config (median-of-3)
- `Multi-Zone Perf Gate (#2772)` — PR-blocking floor: ≥10 cfg/s on 10-zone pop_1000 analytical
- `Architecture Drift Detection` (nightly + on `src/**/*.rs`/`ARCHITECTURE.md` changes)
- `Docs Hygiene Gate` (#2466; freshness sub-check #2765) · `Code Coverage Gate` (#1932, ratchet; `validation/coverage_baseline.json`; `0.0` = unenforced)
- `Cargo Deny` (#2699) — supply-chain: licenses/duplicates/bans/sources (`deny.toml`); `ignore` list mirrored from `.cargo/audit.toml` (keep in sync)

## Environment Variables

- `FLUXION_REST_BIND` / `FLUXION_REST_PORT` — `fluxion-rest` (default `0.0.0.0:8080`; healthcheck `/v1/healthz`).
- `FLUXION_REST_AUTH` — `off` (default) | `token` | `tls`. Issue #2505 auth middleware on every `/v1/*` route except `/v1/healthz`. `token` requires `FLUXION_REST_AUTH_TOKEN`; `tls` expects the reverse proxy to set the verified-client header.
- `FLUXION_REST_AUTH_TOKEN` — bearer for `token|tls`.
- `FLUXION_REST_CORS_ORIGINS` — comma-separated allow-list (defaults to localhost dev origins; never `permissive()`).
- `FLUXION_REST_RATE_LIMIT_RPS` / `_BURST` / `_MAX_ENTRIES` — per-IP token-bucket governor (defaults `100`/`1000`/`100000`). Body capped at 16 MiB. `_MAX_ENTRIES` LRU-caps distinct per-IP buckets (#2688).
- `FLUXION_REST_TRUSTED_PROXIES` — CIDR allow-list of trusted reverse proxies. **When unset (default), `X-Forwarded-For`/`X-Real-IP` are IGNORED** and the limiter keys on the socket peer only (#2688 spoofing fix). When set, only rightmost-non-trusted hop is honoured.
- `FLUXION_REST_ALLOW_INSECURE=1` — opt out of the release-build boot guard that refuses `FLUXION_REST_BIND=0.0.0.0` + `FLUXION_REST_AUTH=off`.
- `FLUXION_ONNX_MODEL` — explicit ONNX model path (default `models/surrogate_zone_thermal.onnx`; mock fallback when unset).
- `FLUXION_ONNX_BACKEND` — `cpu|cuda|coreml|directml|openvino`; auto-downgrades to `cpu` if `cuda` feature not built.
- `FLUXION_GPU` — `0`/`false` to force CPU inference.
- `FLUXION_MQTT_ALLOW_INSECURE` — `fluxion-twin` MQTT consumer. Truthy (`1`/`true`/`yes`/`on`) opts into plaintext broker URLs AND bypasses the #2703 release-build boot guard that otherwise refuses insecure transports (plaintext or disabled cert validation). Debug builds skip the guard. **Default: TLS-only.**
- `FLUXION_MQTT_INSECURE` — truthy skips TLS server-cert validation (dangerous, warning-logged). Refused in release builds unless `FLUXION_MQTT_ALLOW_INSECURE=1` is also set.
- `DWAVE_API_TOKEN` — required at runtime for the `dwave` feature.
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` + `RUST_MIN_STACK=33554432` — set by Python-bindings CI to avoid linker SIGSEGV.
- `CARGO_BUILD_JOBS=1` — set by Clippy CI to keep peak RSS low.

## Toolchain Quirks

- **`rust-toolchain.toml`** pins **stable** + rustfmt + clippy. `.rustfmt.toml` sets `edition = "2021"` — without it rustfmt falls back to 2015 and breaks on `?`/`async`/2021+ fn syntax (PR #1387). **Stable rustfmt has no `exclude`**: auto-generated fixture data must use `#[rustfmt::skip]` per-item (see `tests/per_tilt_per_azimuth_fixture_data.rs`).
- **Mutation testing** (`cargo mutants`): full suite needs **32 GB+ RAM**; `.cargo/mutants.toml` excludes combinatorial physics files and all of `src/validation/**`. Diff-scoped advisory on PRs, full nightly on 32 GB runner (skips gracefully <16 GB, #2130). Manual: `cargo mutants --config .cargo/mutants.toml -p fluxion --jobs 2 --baseline skip`.
- **Feature flags (default = none)**: `python-bindings`, `python-extension` (maturin wheel builds, #2532), `napi-bindings`, `ort` (alias `onnx`), `cuda`, `wiring-tracing`, `multi-zone`, `ashrae_140_v2021`, `pr821-diag`, `loom`, `dwave`, `debug-physics`, `kafka`, `fluid`, `gauge-solver`, `fluxion-city`, `fluxion-cfd`, `dhat`, `tracing-subscriber-json` (#2500). **Default builds skip the ONNX runtime** — opt in via `--features ort` for AI surrogate / mutation tests.
  - `debug-physics` gates `eprintln!` in physics hot loops (#1967) · `fluid` enables `fluxion-fluid` acausal HVAC (#1980/ADR-0005) · **`gauge-solver` is experimental/opt-in scaffolding** for `GaugeZoneSolver` (#2304) — it does NOT replace 5R1C/9R4C: the field is feature-gated but always `None` (#2686); the live research path is the per-surface `GaugeSolver` in shadow mode via `PhysicsAdapter` (#1465/#1462) · `fluxion-city` wires urban radiation (#2344) · `fluxion-cfd` enables FFD/CFD co-simulation (#2460) · `kafka` enables rdkafka telemetry (#2056) · `dhat` enables heap profiling (#2384).
- **Crate size**: `.cargoignore` (33-line) strips `refdata/`, `data/`, `models/`, `assets/`, `tests/`, `benches/`, `docs/`, `target/`, `Cargo.lock`, `Dockerfile`, `.githooks/`, `.github/`, etc. from publish. Published crate must stay <10 MB.
- **Two `CONTRIBUTING.md` files** (root + `docs/CONTRIBUTING.md`): root is the active short form with rustfmt-1.9 quirks and "avoid scope creep on CI failures" guidance; `docs/CONTRIBUTING.md` is the long-form guide. Edit the right one for the change.

## Mathematical Reasoning

**Always write and execute Python code** for calculations — LLMs are unreliable at arithmetic (RULES.md constraint #0, must-always). Use for unit conversions, formula verification, reference-data comparison, solar angles, thermal resistances, statistical analysis.

## Repository Hygiene

- **Root allow-list** (only these `.md` files may live at repo root): `README.md`, `ARCHITECTURE.md`, `CODEBASE_MAP.md`, `CONTRIBUTING.md`, `RULES.md`, `CHANGELOG.md`, `AGENTS.md`, `SCORECARD.md` (auto-generated by `scripts/generate_scorecard.py`). Move transient artifacts (`CASE_*.md`, `BATCH_*.md`, `ISSUE_*.md`, `*_REPORT.md`, session summaries) to `tmp/` or `docs/`. Gate: `scripts/check_root_hygiene.py` (#2466 widened to #2814 — also catches `.txt/.csv/.rs/.py/.sh/.json/.zip` blobs, no-extension blobs, and scratch dirs like `fixes/`, `results/`). `scripts/check_root_md_policy.py` is now a backward-compat shim.
- **`docs/**/*.md` must have a 7-line summary at the top (lines 2–8)**. Gate: `scripts/check_docs_summaries.py`. AGENTS.md is exempt. After adding/removing files under `docs/`, regenerate the auto-table: `python3 scripts/generate_doc_inventory.py`. Drift is blocked by `scripts/check_doc_inventory_fresh.py` (#2765) — a content-based gate that diff-compares; if it fails locally or in CI, run the generator and commit `docs/doc-inventory.md`.
- **Local-only runtime dirs** (gitignored — never commit, never create at repo root): `.sdd/` (Bernstein agent-orchestration runtime, #2837), `.opencode/` (OpenCode plugin config), `.claude/` (skills), `.gitnexus/`, `.serena/`, `.automaker/`, `.jules/`, `.sisyphus/`, `.superset/`, `.planning/worktrees/`. `CLAUDE.md` is auto-generated per-session and `.gitignore`d.
- Issue triage labels: `docs/agents/triage-labels.md`. Issue workflow: `docs/agents/issue-tracker.md` (uses `gh issue create --label ...`).

## Branch & PR Conventions

- **`develop`** is the default + integration branch. Branch from it (`git checkout develop && git pull && git checkout -b fix/issue-123`); **all PRs target `develop`** (`gh pr create --base develop`). No direct pushes to `develop`.
- **`main`** is release-only. No direct pushes. PRs targeting `main` are **only permitted from `develop`** (`protect-main-branch.yml` auto-rejects hotfix branches). Releases cut by merging `develop` → `main` via a release PR. Use `--no-ff` merges.
- Conventional commit messages: `fix(scope): …`, `feat(scope): …`, `refactor(scope): …`, `perf(scope): …`, `test(scope): …`, `docs(scope): …`.
- **PR body must include `Closes #N` or `Fixes #N`** for the linked issue — orchestration depends on this keyword to auto-close.
- Never force-push `main` or `develop`. Hotfixes still go through PR review.

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` · `CODEBASE_MAP.md` | Source-of-truth contracts (see Required Reading) |
| `RULES.md` | Hard constraints (math via code, energy balance, ASHRAE 140, no tuning) |
| `Cargo.toml` · `.cargo/audit.toml` · `deny.toml` | Workspace + supply-chain config (advisories list mirrored across `audit.toml` ↔ `deny.toml`) |
| `src/lib.rs` · `src/python/` | PyO3 entrypoint — `Model` + re-exports of `thermal_model` traits, `assembly`, `multi_node`, `ashrae_cases`. `BatchOracle` bindings live in `src/python/` (extracted #2493; `lib.rs` kept <500 lines) |
| `src/physics/solver_trait.rs` · `src/sim/thermal_model.rs` · `src/sim/ventilation.rs` | ML-surrogate swap-point traits |
| `release_gates.yaml` | Required branch-protection checks + thresholds (canonical source of truth for CI gates) |
| `tests/reference_data/` | EnergyPlus CSV reference data + `zone_balance/strict_energy_gate_baseline.json` (#1333/#2506) |
| `scripts/check_{architecture_drift,ashrae_cases_cycle,physics_sim_cycle,root_hygiene,docs_summaries,doc_inventory_fresh}.py` | Drift + cycle + hygiene regression guards |
| `scripts/release_gate_checker.py` | Validates gates against current results |
| `scripts/coverage_{critical_paths,baseline}.py` · `validation/coverage_baseline.json` | Coverage ratchet (#1932; `0.0` = unenforced) |
| `scripts/disk-space-check.sh` · `scripts/disk-space-gate.sh` | Pre-flight before orchestration / CI |
| `.githooks/batch-oracle-check.sh` | Enforces population-level-only parallelism in `BatchOracle` |
