# Fluxion Cargo Feature Flags

Fluxion ships with **no default features** (`default = []`); every capability below is opt-in
via `cargo build --features <flag>` (combine several with commas). This document enumerates each
flag in `Cargo.toml`'s `[features]` table, cross-referenced with the toolchain notes in
`AGENTS.md` §"Toolchain Quirks" and the CI matrix in `.github/workflows/rust-tests.yml`.
The canonical source of truth is always `Cargo.toml` — if this file disagrees with it,
`Cargo.toml` wins and this doc is stale (please file a docs-hygiene issue).

*Last Updated: 2026-09-03*

## Summary Table

| Flag | Default | Enables | CI gate | Runtime config |
|------|:-------:|---------|---------|----------------|
| [`python-bindings`](#python-bindings) | off | PyO3 Python bindings (`pyo3`, `numpy`) | Python wheel build job | `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| [`napi-bindings`](#napi-bindings) | off | Node.js / NAPI bindings (`napi`, `napi-derive`) | Node wheel build job | `napi-build` at build time |
| [`ort`](#ort--onnx) | off | ONNX Runtime AI surrogate inference (#1294) | `--features ort` mutation tests | `FLUXION_ONNX_MODEL`, `FLUXION_ONNX_BACKEND` |
| [`onnx`](#ort--onnx) | off | Alias for `ort` (#1294) | same as `ort` | same as `ort` |
| [`cuda`](#cuda) | off | CUDA / TensorRT execution providers for `ort` (#1294) | `CUDA Smoke Test` (#1603) | `FLUXION_ONNX_BACKEND=cuda`, `FLUXION_GPU=1` |
| [`wiring-tracing`](#wiring-tracing) | off | Wiring tracing in integration tests (Plan 21-10) | `Test` matrix variant `wiring-tracing` | none |
| [`multi-zone`](#multi-zone) | off | Multi-zone thermal network coupling | `Test` matrix variant `multi-zone` | none |
| [`ashrae_140_v2021`](#ashrae_140_v2021) | off | ASHRAE 140 v2021 reference constants | advisory ASHRAE variant | none |
| [`pr821-diag`](#pr821-diag) | off | Hourly CSV diagnostics for 600FF/650FF (PR #821) | none (local debugging) | writes CSVs to CWD |
| [`loom`](#loom) | off | `loom` concurrency model tests (#1065) | manual only; ~32 GB RAM | `LOOM=1` |
| [`dwave`](#dwave) | off | D-Wave quantum annealer client (Phase 2c, #1609) | manual only | `DWAVE_API_TOKEN` (required at runtime) |
| [`debug-physics`](#debug-physics) | off | Unconditional `eprintln!` in physics hot loops (#1967) | none | none |
| [`kafka`](#kafka) | off | `rdkafka` telemetry consumer (#2056) | manual only | Kafka broker config |
| [`fluid`](#fluid) | off | Acausal HVAC / fluid network modeling via `fluxion-fluid` (ADR-0005, #1980) | `fluxion-mcp` build (unconditional) | none |
| [`gauge-solver`](#gauge-solver) | off | experimental `GaugeZoneSolver` scaffolding (#2304); not wired as primary — always `None` (#2686) | advisory ASHRAE variant | none |
| [`fluxion-city`](#fluxion-city) | off | Urban radiation solver wiring (#2344) | manual only | none |
| [`dhat`](#dhat) | off | `dhat` heap allocation profiling (#2384) | manual only | `DHAT_ANALYSIS=1` |
| [`fluxion-cfd`](#fluxion-cfd) | off | FFD / CFD loose-coupling co-simulation (#2460) | manual only | none |
| [`fast-math`](#fast-math) | off | algebraic-FP helper layer (`src/physics/fp_algebraic.rs`, #3322); **non-deterministic** | none — never in CI | none |
| [`fluxion`](#fluxion-internal-stub) | off | Internal stub for workspace feature resolution | none (never user-facing) | none |

**Total: 19 user-facing flags** (counting `ort` and its `onnx` alias once) plus 1 internal stub.
Default build (`cargo build`) enables none of them and skips the ONNX runtime, producing the
mock / analytical fallback in `src/ai/surrogate`.

---

## Build & Test Recipes

| Goal | Command |
|------|---------|
| Pure-Rust core build | `cargo build` |
| ONNX AI surrogate build | `cargo build --features ort` |
| GPU inference | `cargo build --features cuda` |
| Python bindings dev install | `cargo build --features python-bindings && maturin develop` |
| Node bindings | `(cd npm/ && npm run build)` — internal `napi-bindings` |
| Multi-zone test variant | `cargo test --features multi-zone` |
| Concurrency tests | `LOOM=1 cargo test --features loom --test loom_concurrency_tests` |
| Heap profile | `cargo build --features dhat` |
| ASHRAE 140 with GaugeSolver | `cargo test --features gauge-solver --test ashrae_140_case_600_series` |
| D-Wave client test | `cargo test --features dwave -p fluxion quantum::dwave_client` |
| Kafka consumer test | `cargo test --features kafka -p fluxion twin::kafka_telemetry_consumer` |
| Algebraic-FP helper smoke test | `cargo test --features fast-math -p fluxion physics::fp_algebraic` (**non-deterministic mode**) |

Combine flags with commas: `cargo test --features ort,multi-zone,fluid`.

---

## Per-Flag Details

### `python-bindings`

- **Enables:** PyO3 entrypoints in `src/lib.rs` — `Model`, `BatchOracle`, the ASHRAE cases
  re-export, and the thermal-model traits. Pulls in `pyo3`, `pyo3-build-config`, and `numpy`.
- **Build:** `cargo build --features python-bindings` then `maturin develop` (local install)
  or `maturin build --release` (wheel). Python 3.10+.
- **CI implication:** Built by the Python wheel job; the default `cargo test` matrix
  variant (`no-default`) deliberately leaves it off for pure-Rust testing. CI sets
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` and `RUST_MIN_STACK=33554432` to avoid linker SIGSEGV.
- **Default:** off.

### `napi-bindings`

- **Enables:** Node.js / NAPI bindings via `napi`, `napi-derive`, and `napi-build`. Code lives
  under `napi/`.
- **Build:** invoked by `npm/` — `(cd npm/ && npm run build)`. `napi-build` runs as a build
  script when the feature is active.
- **CI implication:** Node wheel build job; not in the default Rust test matrix.
- **Default:** off.

### `ort` / `onnx`

- **Enables:** ONNX Runtime inference for AI surrogates (`src/ai/surrogate`). When disabled,
  `src/ai/surrogate` exposes only the mock / analytical fallback. Requested by issue #1294.
- **`onnx` is an alias** for `ort` — pick whichever reads better; they activate the same code.
- **Build:** `cargo build --features ort` (or `--features onnx`).
- **CI implication:** Opt-in for AI surrogate tests and `cargo mutants` mutation testing
  (see `.cargo/mutants.toml`). Not in the default Rust test matrix.
- **Runtime config:** `FLUXION_ONNX_MODEL` (path, defaults to
  `models/surrogate_zone_thermal.onnx`; mock fallback when unset);
  `FLUXION_ONNX_BACKEND` (`cpu | cuda | coreml | directml | openvino`, auto-downgrades to
  `cpu` if `cuda` feature not built); `FLUXION_GPU=0` forces CPU inference.
  Silent CUDA→CPU downgrades now emit a one-shot `tracing::warn!` on target
  `fluxion::ai::surrogate::backend` from `SurrogateManager::resolve_backend_from_env`
  / `gpu_supported` (Issue #2920) — operators see the misconfiguration instead of paying
  the CPU throughput floor invisibly.
- **Default:** off.

### `cuda`

- **Enables:** CUDA / TensorRT execution providers for `ort`. Implies `ort` (do not combine
  redundantly). Used for GPU inference.
- **Build:** `cargo build --features cuda`.
- **CI implication:** Drives the `CUDA Smoke Test` (#1603) — `cargo test --features cuda
  --test surrogate_cuda_smoke`. Skips gracefully on CPU-only runners.
- **Runtime config:** set `FLUXION_ONNX_BACKEND=cuda` (or it falls back to `cpu`); set
  `FLUXION_GPU=0` / `FLUXION_GPU=false` to force CPU even when built with the feature.
- **Default:** off.

### `wiring-tracing`

- **Enables:** Wiring / harness tracing assertions in the integration test suite (Plan 21-10).
  No production code paths are affected.
- **Build:** `cargo test --features wiring-tracing`.
- **CI implication:** One of the three `Test` matrix variants in `rust-tests.yml`
  (`no-default` / `wiring-tracing` / `multi-zone`). Required branch-protection check.
- **Default:** off.

### `multi-zone`

- **Enables:** Multi-zone thermal network coupling — the N-zone analog of the 5R1C / 9R4C
  single-zone solver. See `tests/multi_zone_*` and `sim::multi_node_thermal`.
- **Build:** `cargo test --features multi-zone` (try `multi_zone_n_zone_network`).
- **CI implication:** One of the three `Test` matrix variants. Also note `fluxion-mcp`
  *unconditionally* depends on `fluxion` with `features = ["multi-zone"]` (+ `fluxion-fluid` +
  `fluxion-toon`), so `cargo build -p fluxion-mcp` always compiles it in.
- **Default:** off.

### `ashrae_140_v2021`

- **Enables:** ASHRAE 140 v2021 reference constants / boundary conditions. The default build
  uses the v2023 values (e.g. `EXTERIOR_FILM_COEFF = 18.3 W/m²K`); enable this flag to
  reproduce the legacy v2021 spec. Mutually exclusive in spirit with the v2023 defaults —
  do not mix in the same test run.
- **Build:** `cargo test --features ashrae_140_v2021 --test ashrae_140_validation`.
- **CI implication:** Advisory ASHRAE variant; the strict ±15 % energy gate
  (`ASHRAE 140 Strict Energy Gate`, Issue #1333) runs against the v2023 defaults.
- **Default:** off (v2023 is the active spec).

### `pr821-diag`

- **Enables:** Diagnostic CSV output for the 600FF / 650FF ASHRAE cases (PR #821). Writes
  hourly CSVs to the current working directory for offline convergence debugging.
- **Build:** `cargo test --features pr821-diag`.
- **CI implication:** None — local debugging only; never in the CI matrix.
- **Default:** off.

### `loom`

- **Enables:** `loom` concurrency-model tests for the lock-free hot paths
  (`BatchOracle::evaluate_population`, etc.). Requires **~32 GB RAM** (#1065).
- **Build:** `LOOM=1 cargo test --features loom --test loom_concurrency_tests`.
- **CI implication:** Never runs in `rust-tests.yml` — too memory-hungry for the default
  runners. Manual only.
- **Default:** off.

### `dwave`

- **Enables:** D-Wave quantum annealer client (Phase 2c, issue #1609). Uses the D-Wave SAPI
  REST API (the same API underlying Python `dimod`).
- **Build:** `cargo test --features dwave -p fluxion quantum::dwave_client`.
- **CI implication:** Manual only — requires live credentials.
- **Runtime config:** `DWAVE_API_TOKEN` is **required at runtime** for the `dwave` feature;
  calls fail without it.
- **Default:** off.

### `debug-physics`

- **Enables:** Unconditional `eprintln!` calls in physics hot loops (issue #1967). Without it,
  those debug prints are compiled out to keep the inner loop allocation-free and fast.
- **Build:** `cargo build --features debug-physics`.
- **CI implication:** None — off in CI so test output stays clean and timing gates
  (`Fluxion Performance Gate`, Issue #1618) reflect production paths.
- **Default:** off.

### `kafka`

- **Enables:** `rdkafka`-based Kafka telemetry consumer for enterprise-scale telemetry
  ingestion into the digital twin (`twin::kafka_telemetry_consumer`, issue #2056).
- **Build:** `cargo test --features kafka -p fluxion twin::kafka_telemetry_consumer`.
- **CI implication:** Manual only — pulls in the native `librdkafka` build dependency.
- **Runtime config:** standard Kafka broker connection (bootstrap servers, auth, etc.).
- **Default:** off.

### `fluid`

- **Enables:** Acausal HVAC / fluid network modeling via the `fluxion-fluid` crate, providing
  port-based fluid circuit traits (Issue #1980 / ADR-0005). This is **not** the same as
  `fluxion-core/src/fluid/`.
- **Build:** `cargo build --features fluid`.
- **CI implication:** `fluxion-mcp` unconditionally builds with `fluxion-fluid`, so the
  feature is exercised indirectly by `cargo build -p fluxion-mcp` and `cargo test -p
  fluxion-mcp`.
- **Default:** off (except when built via `fluxion-mcp`).

### `gauge-solver`

- **Enables:** the experimental `GaugeZoneSolver` zone-level scaffolding (Issue #2304).
  **Status (Issue #2686, 2026-08-11): opt-in and NOT wired as primary.** The
  `gauge_zone_solver` field on `ThermalModelData` is feature-gated but always initialized
  to `None`, and no construction path sets it to `Some`. Even with `--features gauge-solver`,
  the routing branch in `step_physics` is unreachable, so the legacy 5R1C / 9R4C
  lumped-capacitance networks remain the primary zone solver in ALL builds (default and
  feature-enabled). This stub wiring is preserved as WIP for whoever finishes #2304. The
  live gauge-theory research path is the separate per-surface `GaugeSolver` run in shadow
  mode via `PhysicsAdapter` (ARCHITECTURE.md Module 6, #1465 / #1462).
- **Build:** `cargo test --features gauge-solver --test ashrae_140_case_600_series`.
- **CI implication:** Advisory ASHRAE variant; the production solver path in CI is still the
  5R1C / 9R4C default so the strict energy gate (#1333) remains directly comparable to
  EnergyPlus reference data.
- **Default:** off.

### `fluxion-city`

- **Enables:** Urban radiation solver wiring (Issue #2344). Wires
  `UrbanRadiationSolver` into `PhysicsSurfaceFluxProvider` via
  `FluxionCitySurfaceFluxProvider`. Depends on the `fluxion-city` sibling crate.
- **Build:** `cargo build --features fluxion-city`.
- **CI implication:** Manual only — the `fluxion-city` crate is a feature-gated sibling and
  not in the default workspace build.
- **Default:** off.

### `dhat`

- **Enables:** `dhat` heap allocation tracking for memory profiling of large simulations
  (Issue #2384).
- **Build:** `cargo build --features dhat`.
- **CI implication:** Manual only — never in CI; adds significant runtime overhead.
- **Runtime config:** standard `dhat` analysis flow (set `DHAT_ANALYSIS=1` for the
  profiler-enabled run).
- **Default:** off.

### `fluxion-cfd`

- **Enables:** FFD / CFD loose-coupling co-simulation (Issue #2460). Wires
  `fluxion_cfd::FfdCfdSolver` into `crate::sim::loose_coupling::FfdSolver` via
  `crate::sim::ffd_cfd_adapter::FfdCfdAdapter`. This adapter is the production bring-up for
  the BES-FFD loose-coupling path (ARCHITECTURE.md §"Module N+2"). The default CPU solver in
  `fluxion-cfd/src/cpu/` is sufficient for the regression test; CUDA is **not** required.
- **Build:** `cargo build --features fluxion-cfd`.
- **CI implication:** Manual only — `fluxion-cfd` is a feature-gated sibling and not in the
  default workspace build.
- **Default:** off.

### `fast-math`

- **Enables:** routing of the `src/physics/fp_algebraic.rs` helper layer
  (`algebraic_add` / `algebraic_sub` / `algebraic_mul` / `algebraic_div` for
  `f32`/`f64`) to the Rust 1.98 std algebraic float methods, which permit operand
  reassociation and vectorization comparable to `-ffast-math` (Issue #3322). With the
  flag **off** (the default) every helper compiles to the plain IEEE 754 operator, so
  default builds are bit-identical to not using the module.
- **⚠️ Non-determinism:** the algebraic methods are non-deterministic by specification —
  results may differ from strict IEEE 754 at the last-ulp level across compiler versions,
  optimization levels, and targets. `determinism_check.yml` (bit-identical cross-platform
  output, #1297/#2549) and every ASHRAE 140 gate therefore **always run default features**.
- **Do NOT use in:** energy-balance-critical paths — `src/physics/ctf_solver.rs`,
  `src/physics/ctf_solver_wrapper.rs`, `src/physics/multi_node_solver.rs`,
  `src/physics/five_r1c_solver.rs`, `src/physics/fd_solver.rs`, and the zone-balance /
  thermal assembly (`src/sim/assembly.rs`, `src/sim/timestep_solver.rs`,
  `src/sim/thermal_model*.rs`). Intended consumers are the kernel-conversion issues
  #3324 (solar/irradiance reductions) and #3325 (AI batch metric reductions).
- **Build:** `cargo build --features fast-math`. Declared only on the root crate with an
  empty dependency list — no other feature or workspace member pulls it in (no feature
  unification, cf. the #2904 passthrough pattern).
- **CI implication:** none — intentionally absent from every workflow. Note that the
  nightly/manual `bench-all-features` job (`performance_dashboard.yml`) enables it
  implicitly via `--all-features`; benchmarks tolerate last-ulp noise by design.
- **Default:** off.

### `fluxion` (internal stub)

- **Enables:** Nothing user-facing. This is a stub feature that exists purely so
  `fluxion-grid` can depend on this crate via `dep:fluxion` and resolve `--features fluxion`
  at the workspace level. **Never set this from the command line.**
- **Default:** off (and irrelevant).

---

## CI Matrix at a Glance

The `rust-tests.yml` workflow runs three primary `Test` variants on every PR:

| Variant | Flags | Purpose |
|---------|-------|---------|
| `no-default` | *(none)* | Pure-Rust baseline; ONNX, bindings, and tracing all off |
| `wiring-tracing` | `--features wiring-tracing` | Integration test tracing harness (Plan 21-10) |
| `multi-zone` | `--features multi-zone` | Multi-zone thermal network coupling |

Feature-gated checks that run on every PR (or on schedule) regardless of the variants above:

- `CUDA Smoke Test` (#1603) — `--features cuda`
- `Energy Conservation` (#1295) — greps test output for energy-conservation violations
- `ASHRAE 140 Strict Energy Gate (Issue #1333)` — v2023 defaults, ±15 % annual energy

Manual / advisory (not branch-protection gates):

- `loom` concurrency suite (needs ~32 GB RAM)
- `dwave`, `kafka` client tests (need live services / credentials)
- `fluxion-city`, `fluxion-cfd`, `dhat`, `pr821-diag` (specialised analysis paths)
- `fast-math` (algebraic-FP helper layer — non-deterministic, never in validation CI)

## See Also

- `AGENTS.md` §"Toolchain Quirks" → "Feature flags" — the human-readable overview this
  document expands.
- `AGENTS.md` §"Environment Variables" — runtime configuration (`FLUXION_ONNX_*`,
  `FLUXION_REST_*`, `DWAVE_API_TOKEN`, …).
- `Cargo.toml` `[features]` — the authoritative machine-readable source.
- `release_gates.yaml` → `ci.required_checks` — which of the above are branch-protection gates.
- ARCHITECTURE.md §"Module N+2" — describes the `fluxion-cfd` FFD loose-coupling adapter.
