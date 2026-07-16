# Fluxion — Agent Instructions

## Required Reading (MANDATORY)

Before working on ANY issue, read `ARCHITECTURE.md` in the repository root. Feed the full file to the model on every new session — it is the **source of truth** for module boundaries, trait contracts, and data flow.

**Rule**: Do NOT modify physics code without checking ARCHITECTURE.md first. If the code doesn't match the documented interfaces, update ARCHITECTURE.md to reflect reality OR fix the code to match the architecture.

## Workspace Structure

- Root `fluxion` package: main engine (`src/`, physics, sim, AI, validation)
- `fluxion-core` crate: dependency-light leaf modules (`weather/`, `assembly/`, `multi_node/`, `ashrae_cases/`) — split for `cargo-mutants` caching
- `fluxion-mcp` crate: MCP server (`fluxion-mcp/`)
- Cycle-breaking rule: `fluxion-core/src/**/*.rs` must NOT import `crate::sim_*`, `crate::physics_*`, `crate::ai_*`, `crate::validation_*`

## Developer Commands

```bash
# Build & test
cargo build --release
cargo test --release                           # all unit tests
cargo test -p fluxion <test_name>             # single test (e.g. multi_zone_n_zone_network)
cargo test --test ashrae_140_validation       # ASHRAE 140 validation suite
LOOM=1 cargo test --features loom             # loom concurrency tests

# Code quality (required order)
cargo fmt
cargo clippy --all-targets
cargo audit

# Python bindings (requires Python 3.10+)
maturin develop       # local dev install
maturin build --release

# Pre-commit hooks
pre-commit run --all-files
```

## Validation Strategy

**Phase 1: Module Isolation**. Rules:
1. **No ASHRAE 140 system-level testing** until individual modules pass E+ reference tests
2. **No parameter tuning** to make system tests pass — fix the underlying math
3. **Each module must match EnergyPlus within 1% tolerance** on isolated scenarios
4. Test order: Weather -> Solar -> Conduction -> Ventilation -> Zone Balance

## Module Boundaries

```
Weather (fluxion-core/src/weather/)  -> Solar (src/sim/solar.rs)  -> Zone Balance
                                      -> Ventilation (src/sim/ventilation.rs)
                                      -> Conduction (src/physics/solver_trait.rs)
```

Trait hierarchy for ML surrogate swap points:
- `HeatConductionSolver` — conduction (5R1C, CTF, FD, MultiNode)
- `VentilationSchedule` — ventilation (constant, scheduled, weather-dependent)
- `ThermalModelTrait` — zone solver (physics, surrogate, hybrid)

## Critical Physics Constants

- **`EXTERIOR_FILM_COEFF = 18.3 W/m²K`** (ASHRAE 140 v2023 vertical surfaces, ~3.4 m/s wind) — defined in `src/physics/constants/thermal/ashrae_140/v2023.rs`. The legacy `29.3 W/m²K` (6.7 m/s) must NOT appear in any computation path. Guard: `tests/regression_exterior_film_unification.rs`.

## Branch & PR Conventions

- **`develop`** — default branch for all PRs. Create feature branches from `develop`.
- **`main`** — only for releases (merge `develop` → `main`). No direct PRs to `main` except for release merges.
- **Always use `--base develop`** when creating PRs via CLI (e.g., `gh pr create --base develop`).

**Why `develop`?** The `develop` branch is the main integration branch. `main` is reserved for releases. All wave orchestrator work, hotfixes, and agent-generated changes target `develop`.

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Module boundaries, I/O contracts, trait hierarchies, 1000+ line source of truth |
| `src/physics/solver_trait.rs` | HeatConductionSolver trait |
| `src/sim/thermal_model.rs` | ThermalModelTrait + HybridRouting |
| `src/sim/solar.rs` | Solar position and irradiance |
| `src/sim/ventilation.rs` | VentilationSchedule trait |
| `src/physics/multi_node_solver.rs` | 9R4C multi-node solver (ADR-002) |
| `tests/reference_data/` | EnergyPlus CSV reference data for unit tests |

## Mathematical Reasoning

**Always write Python code** (`ctx_execute language:"python"`) for calculations — LLMs are unreliable at arithmetic. Use for: unit conversions, formula verification, reference data comparison, solar angles, thermal resistances, statistical analysis.

## 7-Line Summary Convention

All system docs in `docs/` must have a **7-line summary** at the top (lines 2–8). See `docs/doc-inventory.md` after modifying any module. AGENTS.md itself is exempt.

## Cross-Agent Review

For multi-model PR review, use the `pr-review-merge` skill. See `docs/agent-review-guide.md` for per-phase review routing (Physics Auditor → Safety Engineer → etc.).

## Related Documentation

- Standard workflow: `@/docs/agent-workflow.md`
- Agent review guide: `@/docs/agent-review-guide.md`
- Doc inventory: `@/docs/doc-inventory.md`
- Contributing: `@/CONTRIBUTING.md`
- Architecture: `@/ARCHITECTURE.md`
