# Issue #2937: Mojo & MAX Surrogate Spike — Framework Investigation

**Issue:** [#2937](https://github.com/anchapin/fluxion/issues/2937)
**Date:** 2026-08-17
**Investigator:** architecture-tracking sub-agent
**Branch:** `fix/issue-2937-mojo-spike`
**Status:** ⏸ **Deferred** — see ADR-0013 §"Why this is still a spike" and §"Consequences / Negative"

> **Summary 1/7:** Investigation captures the spike's evaluation framework, acceptance criteria, and trait-gating contract for any future Mojo / MAX `ThermalModelTrait` backend, cross-referencing [ADR-0013](../adr/0013-mojo-surrogate-spike.md).
> **Summary 2/7:** All nine `ai-surrogate` critical issues listed in #2937's "Why This Is a Spike" body (#2923, #2924, #2922, #2921, #2919, #2920, #2925, #2906, #2905) are **CLOSED** as of 2026-08-17 — verified by `gh issue view <N> --json state` — so the spike's narrow gating condition has been met.
> **Summary 3/7:** The umbrella Mojo roadmap (ADR-0012, epic #2940) binds the spike to the broader v1.3 ASHRAE 140 release gate (`release_gates.yaml → validation.min_pass_rate = 60.0`); current metric pass rate is **14.3%** per `SCORECARD.md`, so the spike remains ⏸ Deferred until that gate clears.
> **Summary 4/7:** Future acceptance criteria are fixed by this investigation and ADR-0013: ONNX surrogate outputs reproduced within **1e-5** relative tolerance (no tuning allowed), microbenchmarks at batch sizes **1 / 100 / 1000 / 10000** on CPU SIMD AND GPU, microbenchmark report + integration write-up committed under `docs/investigations/`.
> **Summary 5/7:** Integration contract: any future `MojoThermalModel` (or equivalent) struct implements `ThermalModelTrait` (`src/sim/thermal_model.rs`); selected at the call site only (new `ThermalModelKind::Mojo` enum variant); never a parallel physics path; the Mojo shared library is loaded at runtime via `libloading` and is not a Cargo dependency.
> **Summary 6/7:** The Mojo toolchain is **not** installed by this PR; `bash scripts/check_mojo_toolchain.sh` stays an advisory detect gate (always exits 0 in default mode); no `Cargo.toml`, `deny.toml`, or `.cargo/audit.toml` change; the spike lives in its own evaluation directory, never in `src/`, `fluxion-core/`, `tests/`, or any binary.
> **Summary 7/7:** Per `RULES.md` ("must-never hardcode results") and `ADR-0001` (No-Parameter-Tuning Rule), the `1e-5` tolerance band is fixed and may not be relaxed to absorb a regression; the spike must reproduce the existing ONNX output, not improve it.

---

## 1. Background

Fluxion's hybrid neuro-symbolic architecture pairs a validated first-principles
thermal network with neural surrogates for CFD approximations and complex
radiation models. Today, ML inference bridges the Rust core through external
runtimes — ONNX Runtime (`src/ai/surrogate.rs`) by default and optionally
LibTorch — across FFI / binding layers that introduce serialization and
memory-boundary overhead. Issue #2937 proposes Modular's
[Mojo language][mojo] and the [MAX framework][max] as a candidate *second
backend* behind the existing ONNX surrogate path, on the hypothesis that
Mojo's MLIR-native tensor abstractions and MAX's graph-level operator
fusion can carry both surrogate inference and physical tensor operations in
a unified pipeline with lower FFI cost and direct hardware portability
across CPU SIMD and GPU backends.

[mojo]: https://mojolang.org/
[max]:  https://docs.modular.com/

The issue body is careful to label itself a **research artifact** rather
than an active task. Its "Why This Is a Spike (Not an Active Task)" section
enumerates **nine** `ai-surrogate` critical issues as the spike's narrow
gating dependencies and adds the closing rule:

> "this work does not start until the `ai-surrogate` critical issues above
> are closed."

This investigation captures the spike's evaluation framework, the
nine-blocker state-of-play, the broader umbrella release-gate that
ADR-0012 binds the spike to, and the fixed acceptance criteria a future
spike PR would have to satisfy to move the spike from ⏸ Deferred to
active. The investigation is the standalone companion to
[ADR-0013](../adr/0013-mojo-surrogate-spike.md) and the per-phase (Phase 2)
branch of the umbrella investigation
[`issue-2940-mojo-evaluation-roadmap.md`](./issue-2940-mojo-evaluation-roadmap.md).

## 2. State-of-play on the nine blockers (verified 2026-08-17)

The nine blockers from #2937's body are individually verified below using
`gh issue view <N> --json state`. The verification was performed in the
worktree for this PR (`fix/issue-2937-mojo-spike`); the closed-state values
are stable for the purpose of this investigation.

| # | Title | State on 2026-08-17 |
|---|-------|---------------------|
| #2923 | `SurrogateDriftToleranceGate` silently downgrades to advisory when no ONNX model is registered | **CLOSED** ✅ |
| #2924 | ASHRAE 140 surrogate-driven Cases 600–900 cooling has no MAE gate at the surrogate layer | **CLOSED** ✅ |
| #2922 | No PR-blocking throughput / latency gate for `HybridThermalModel` | **CLOSED** ✅ |
| #2921 | `HybridThermalModel::solve_timesteps` calls allocating `predict_loads_with_fallback` instead of zero-alloc `predict_loads_into` | **CLOSED** ✅ |
| #2919 | No PR-blocking cold-start latency gate for ONNX surrogate | **CLOSED** ✅ |
| #2920 | `FLUXION_GPU` / `FLUXION_ONNX_BACKEND` downgrade path is silent — operator misconfiguration invisible | **CLOSED** ✅ |
| #2925 | `HybridThermalModel` `clone` counter-preservation asymmetry | **CLOSED** ✅ |
| #2906 | Loaded ONNX model has no integrity verification (now `verify_onnx_signature` SHA-256 check) | **CLOSED** ✅ |
| #2905 | ONNX env-var path bypasses `validate_model_path` (#2529) in `SurrogateManager::new_with_auto_load` | **CLOSED** ✅ |

**9 / 9 closed.** The spike's narrow gating condition (its own closing
rule) has been met. This is a real change from the 0 / 9 state when
#2937 was opened.

The nine-blocker closure is necessary but **not sufficient** to start the
spike:

1. **ADR-0012 binds Phase 2 to the broader v1.3 ASHRAE 140 release gate**
   — `release_gates.yaml → validation.min_pass_rate = 60.0` (or 40% on a
   patch release via `release_requirements.patch.min_pass_rate`). The
   current metric-level pass rate is **14.3%** per `SCORECARD.md`
   (regenerated 2026-08-16 by `scripts/generate_scorecard.py`), so the
   umbrella gate is still red.
2. **The broader MAE gate is also red** at **51.03%** vs the 50% cap
   (`release_gates.yaml → validation.max_mae`).
3. **Only the throughput gate is currently green** (`benchmark.throughput
   .min_configs_per_sec ≥ 150`; current ~157 cfg/s).

A maintainer who wishes to revisit the deferral must independently clear
the umbrella gate. The nine specific blockers having closed is recorded
here as a state-of-play update — it does **not** by itself constitute
the v1.3 release-gate clearance that ADR-0012 §"Plan: Phase 2" requires.

## 3. Cross-phase umbrella (ADR-0012, issue #2940)

The spike is **Phase 2** of the 3-phase Mojo roadmap in
[ADR-0012](../adr/0012-mojo-evaluation-roadmap.md) (umbrella epic #2940).
The three phases run sequentially at the **go/no-go level** (each phase
needs the prior phase's measurement as input) and remain **mutually
independent at the implementation level**:

| Phase | Tracking issue | Focus | Status |
|-------|----------------|-------|--------|
| Phase 1 — Isolated physics & SIMD benchmarking | #2938 | Vectorized 5R1C + Perez Sky Model Mojo prototype | ⏸ Deferred |
| **Phase 2 — MAX surrogate integration & operator fusion (this issue)** | #2937 | Surrogate neural model in Mojo using MAX graph ops | ⏸ Deferred |
| Phase 3 — Ecosystem & interoperability assessment | (none yet) | Python interop + package management + CI/CD maturity | ⏸ Deferred |

All three phases are gated on the same umbrella release-gate condition
(`SCORECARD.md` ≥ 60% pass rate). Closing the nine blockers above does
**not** unblock Phase 1 either; #2938 likewise remains ⏸ Deferred until
the umbrella gate clears. The cross-phase umbrella is captured in
[`docs/investigations/issue-2940-mojo-evaluation-roadmap.md`](./issue-2940-mojo-evaluation-roadmap.md)
and [ADR-0012](../adr/0012-mojo-evaluation-roadmap.md); this
investigation scopes itself to Phase 2 (the surrogate spike) only.

## 4. The spike's evaluation framework

The spike's source-of-truth scope is the body of
[issue #2937](https://github.com/anchapin/fluxion/issues/2937). This
investigation restates the framework so the future implementer has a
single document to read.

### 4.1 Workload

A small surrogate neural network (3–4 dense layers) in Mojo predicting
one of:

- interior convection coefficients, **or**
- Perez radiation factors.

The choice between (a) and (b) is open — both are acceptable per
`RULES.md` as long as the Mojo port **reproduces** the existing ONNX
reference output within `1e-5` (the tolerance is fixed in §4.4 below;
it may not be relaxed to absorb a regression). The choice will be
recorded in the spike PR with a rationale anchored in
`models/surrogate_zone_thermal.onnx` (the model whose SHA-256 is
checked at load time per #2906) and `src/ai/surrogate.rs`.

### 4.2 Evaluation batch sizes

Four batch sizes per call — matching the issue body's "Proposed
Prototype Scope":

```text
batch_sizes = [1, 100, 1000, 10000]
```

For each batch size, the spike records throughput (calls / second),
p50 / p99 latency (microseconds / call), and per-call memory transfer
bytes for the microbenchmark report.

### 4.3 Backend matrix (hardware portability)

| Backend | Required? | CI posture |
|---------|-----------|------------|
| **CPU SIMD** (x86_64 AVX-512, aarch64 / Apple NEON) | **Yes** | Runs on every CI runner; primary data path |
| **GPU** (NVIDIA via the project's `cuda` feature flag) | **Yes — gated** | Opt-in; runs only on the GPU runner; CPU-only CI logs `SKIP` with a rationale |

The same Mojo source must compile and execute correctly on CPU SIMD
and GPU **without source-level divergence**. The presence of per-target
C++ / CUDA source code is a fail condition for the prototype. The
project's existing `cargo test --features cuda --test
surrogate_cuda_smoke` pattern is the parity-check template.

### 4.4 Acceptance criteria (fixed — do not tune)

These are operationalized on the spike in [ADR-0013](../adr/0013-mojo-surrogate-spike.md)
§"Plan — Acceptance criteria". Restated for the standalone investigation:

- **`1e-5` relative tolerance.** Mojo / MAX prototype reproduces the
  existing ONNX surrogate (`models/surrogate_zone_thermal.onnx`,
  SHA-256 verified per #2906's `verify_onnx_signature`) within **`1e-5`
  relative tolerance** across the full batch-size sweep.
  **No parameter tuning** is permitted to obtain parity — per
  `RULES.md` ("must-never hardcode results") and `ADR-0001`
  (No-Parameter-Tuning Rule), the tolerance band is fixed at `1e-5`
  and **may NOT be relaxed** to absorb a regression.
- **Microbenchmark report.** Throughput (calls / second) and memory
  footprint compared against:
  1. Existing ONNX Runtime backend via the
     `Fluxion ONNX` path in `src/ai/surrogate.rs`.
  2. Rust reference implementation exercising the same
     `predict_loads_into` path used by
     `HybridThermalModel::solve_timesteps` (#2921).
  The report is committed under `docs/investigations/` with a
  stable file name and cross-referenced from ADR-0013.
- **Integration write-up.** A technical note covering:
  1. How the Mojo backend would ship behind `ThermalModelTrait`
     (without adding a Cargo dependency).
  2. How PyO3 (`src/python/`) and Node bindings (`src/napi/`) would
     consume the new backend.
  3. What additional CI infrastructure would be required (the
     existing `bash scripts/check_mojo_toolchain.sh` advisory gate
     is necessary but not sufficient — a non-blocking *evaluation*
     workflow is sketched in the write-up).
- **Cross-backend numeric parity.** CPU SIMD and GPU produce identical
  numeric output within `1e-5`. The GPU parity check is opt-in via
  the `cuda` feature flag.
- **No parallel physics path.** A future `MojoThermalModel` integrates
  through `ThermalModelTrait` (`src/sim/thermal_model.rs`), selected
  at the call site only — never as a parallel physics path.

## 5. The trait-gating contract

Per `ARCHITECTURE.md` and the "ML Surrogate Path" section, ML inference
is a `ThermalModelTrait` implementation in the Rust core; the zone
solver doesn't know whether physics or ML is computing the result.
Any future Mojo backend must follow the same contract — see
[ADR-0013 §"Plan — Trait-gating contract"](../adr/0013-mojo-surrogate-spike.md).

Summary of the contract:

1. Define a `MojoThermalModel` (or equivalent) struct that implements
   `ThermalModelTrait`. The trait method signatures are fixed by
   `ARCHITECTURE.md §ThermalModelTrait in sim/thermal_model.rs`
   (`num_zones`, `get_temperatures`, `set_temperatures`, `mode`,
   `set_mode`, `solve_timesteps`, `apply_parameters`, `zone_area`,
   `heating_setpoint`, `cooling_setpoint`, `hvac_power_demand`,
   `is_valid`, `get_comfort_metrics`).
2. Add a `ThermalModelKind::Mojo` (or equivalent) enum variant and
   thread it through the existing `ThermalModelBuilder` /
   `ThermalModelMode` dispatch in `src/sim/thermal_model.rs` —
   **never** a parallel physics path that would require running the
   v1.3 ASHRAE 140 suite twice.
3. Delegate the heavy compute to a Mojo-compiled shared library
   (`.so` / `.dylib` / `.dll`) loaded via `libloading` (or equivalent)
   on the developer workstation. The Rust side stays in charge of
   validation, ASHRAE 140 dispatch, and test gating.
4. The Mojo shared library is **not** a `Cargo.toml` dependency.
   `.cargoignore` already strips `docs/`, `models/`, and `assets/`
   from publish; any Mojo build artifact under a dedicated evaluation
   directory outside `src/` stays out of the published crate
   (which remains < 10 MB per `AGENTS.md §Toolchain Quirks → Crate size`).

This contract is the **only** integration path. There is no
"evaluate Mojo as a parallel physics kernel" path because that would
require running the v1.3 ASHRAE 140 suite twice and would violate
`RULES.md` ("no parameter tuning") the moment one path drifts from the
other.

### 5.1 Adjacent swap-points (not in scope)

The umbrella Mojo roadmap in ADR-0012 considers three swap-points for
the candidate Mojo backend. Only one is in scope for Phase 2:

| Swap-point trait | Path | Phase 2 (this spike) | Phase 1 (#2938) | Phase 3 (none) |
|------------------|------|----------------------|-----------------|----------------|
| `ThermalModelTrait` | `src/sim/thermal_model.rs` | **In scope** | Out of scope | Out of scope |
| `HeatConductionSolver` | `physics/solver_trait.rs` | Out of scope | Candidate (Phase 1 deliverable) | Out of scope |
| `VentilationSchedule` | `sim/ventilation.rs` | Out of scope | Out of scope | Out of scope |
| Solar | (no trait — see `ARCHITECTURE.md §Note on Solar trait`) | Out of scope (no ML surrogate swap-point at the solar calculation layer) | Out of scope | Out of scope |

`VentilationSchedule` is excluded because there is no ML surrogate
swap-point at the ventilation layer — the energy-balance integrity
constraint per `RULES.md §2. Energy Balance Conservation` requires
deterministic physics for `h_ve_total`. The Phase 2 spike does NOT
introduce a `MojoHeatConductionSolver`; that path is conditional on
Phase 1 (#2938) demonstrating surrogate-grade SIMD gains.

## 6. Bindings write-up scope

The future spike PR must describe how the Mojo backend integrates
with the existing binding layers:

- **PyO3** (`src/python/`) — how the `Model` Python wrapper exposes
  the Mojo backend alongside the existing `physics` / `surrogate` /
  `hybrid` modes.
- **Node bindings** (`src/napi/`) — same question for the Node
  wrapper.
- **MCP server** (`fluxion-mcp/`, separate Cargo package with
  `default-features = false`) — does the Mojo backend survive the
  default-feature boundary (yes, if the shared library is loaded at
  runtime, not at build time).
- **Batch orchestration** (`src/sim/thermal_model.rs`,
  `BatchOracle` per #2493) — how `BatchOracle::evaluate_population`
  would dispatch to the Mojo backend at the **population level
  only** (not inner per-config), per the
  `.githooks/batch-oracle-check.sh` rule that pre-commit enforces.

## 7. CI / supply-chain posture (unchanged)

- `bash scripts/check_mojo_toolchain.sh` stays **non-blocking**
  (always exits 0 in default advisory mode). The spike does NOT
  make Mojo a CI requirement; CI that lacks Mojo runs the existing
  ONNX-only path. `WARN` lines on CI are expected and acceptable
  until the umbrella gate clears and the spike is actively pursued.
- The spike write-up sketches a future "evaluation" workflow
  (parallel to `.github/workflows/onnx-integrity.yml`, which is the
  precedent for SHA-256-integrity-gated ONNX evaluation per #2906)
  that runs the microbenchmark + tolerance check when the toolchain
  is present and is gated on `--strict` mode for agent pre-flight
  only.
- `cargo audit` / `cargo deny` supply-chain gates
  (`release_gates.yaml → ci.required_checks`, `deny.toml`,
  `.cargo/audit.toml`) are unaffected by the spike. Mojo never
  appears in `Cargo.toml`, `deny.toml`, or `.cargo/audit.toml`.

## 8. What this PR ships

1. **`docs/adr/0013-mojo-surrogate-spike.md`** — companion
   [ADR-0013](../adr/0013-mojo-surrogate-spike.md) that records
   the spike's evaluation framework, the trait-gating contract,
   and the state-of-play on the nine blockers. Status: Proposed.
2. **This investigation document** —
   `docs/investigations/issue-2937-mojo-surrogate-spike.md` —
   the standalone analysis with the nine-blocker state-of-play,
   the umbrella release-gate dependency, and the fixed acceptance
   criteria a future spike PR would have to satisfy.
3. **A tracking comment on Issue #2937** — the maintainer-facing
   record of the ADR + investigation links, with the
   state-of-play (9 / 9 blockers CLOSED, umbrella gate still red)
   surfaced inline.

This PR does **NOT**:

- Modify `src/`, `fluxion-core/`, `tests/`, or any binary
  (`src/bin/`, `tools/`, `fluxion-cfd/`, `fluxion-city/`,
  `fluxion-fluid/`, `fluxion-grid/`, `fluxion-behavior/`, `fluxion-wasm/`,
  `fluxion-mcp/`, `crates/fluxion-twin/`, `crates/fluxion-toon/`).
- Modify `ARCHITECTURE.md`, `RULES.md`, `AGENTS.md`,
  `CODEBASE_MAP.md`, `release_gates.yaml`, `deny.toml`,
  `.cargo/audit.toml`, `Cargo.toml`, or any Cargo workspace file.
- Add a Mojo crate, feature, workspace member, or workspace-level
  lockfile entry.
- Amend `scripts/check_mojo_toolchain.sh` or
  `docs/agents/mojo-setup.md`.
- Modify `docs/KNOWN_ISSUES.md`, `SCORECARD.md`, or `docs/ASHRAE140_RESULTS.md`.
- Bump any Cargo dependency, including `candle-onnx`, `ort-sys`,
  or any ONNX / ML crate.
- Install, configure, or activate the Mojo toolchain on any
  workstation or CI runner.
- Write any Mojo or MAX source code.
- Run any new benchmarks.

## 9. Recommendation (maintainer decision tree)

The decision is recorded in [ADR-0013](../adr/0013-mojo-surrogate-spike.md)
and is **deferred to maintainers**. The decision tree:

- **If the goal is "make the v1.3 release gate clear":** prioritize
  the architectural unblockers #3059 (aggressive-baseline cohort)
  and #1465 / #1462 (`GaugeSolver` production switchover). The
  nine-blocker closure is a state-of-play update; the umbrella
  gate (`release_gates.yaml → validation.min_pass_rate = 60.0`)
  is the active gating condition. Mojo work is a post-gate
  activity.
- **If the goal is "begin the Phase 2 spike anyway":** wait until
  the umbrella gate clears (or until maintainers carve out a
  patch-tier lane at 40% pass rate). Use the spike body
  (#2937) as the source-of-truth scope statement and ADR-0013
  for the trait-gating contract. The nine-blocker closure is
  recorded but does NOT by itself unblock the spike per
  ADR-0012 §"Plan: Phase 2".
- **If the goal is "adopt Mojo for production":** the trait-gating
  contract in ADR-0013 §"Plan — Trait-gating contract" (§5 of this
  investigation) is the only path. A follow-up ADR (Proposed or
  Accepted) is required before any `MojoThermalModel` lands in
  `src/`. The Mojo shared library must NOT appear in `Cargo.toml`.
- **Phase 1 (#2938) and Phase 3 (no tracking issue yet)** remain
  independently gated on the same umbrella release-gate condition.
  This investigation is the Phase 2 view only.

Per `AGENTS.md` / `RULES.md` / `ADR-0001`, this PR ships
**documentation and tracking only**; the spike implementation
decision is deferred to maintainers and tracked in
[Issue #2937](https://github.com/anchapin/fluxion/issues/2937) with
the companion ADR-0013.

## 10. Related issues and references

- **#2937** — origin and source-of-truth scope statement for the
  spike (this investigation's parent issue).
- **#2940** — umbrella Mojo roadmap epic (this investigation's
  parent epic; ADR-0012 source issue).
- **#2938** — Phase 1 (vectorized 5R1C + Perez Sky Model Mojo
  prototype), tracked separately under the same umbrella.
- **#2923, #2924, #2922, #2921, #2919, #2920, #2925, #2906,
  #2905** — the nine `ai-surrogate` critical issues that gated
  this spike; all **CLOSED** (verified 2026-08-17). The umbrella
  release-gate dependency remains the active gating condition.
- **#2979** — Mojo toolchain install guide; origin of
  `docs/agents/mojo-setup.md` and `scripts/check_mojo_toolchain.sh`.
- **#3059** — architectural unblocker coordination for the
  aggressive-baseline cohort. Independent of the Mojo spike but
  routed through the same release-gate dependency.
- **#1465 / #1462** — `GaugeSolver` production switchover; the
  per-surface path tracked in shadow mode via `PhysicsAdapter`.
- **#2906** — `verify_onnx_signature` SHA-256 integrity check in
  `src/ai/surrogate.rs`. The spike's tolerance check exercises
  the same `models/surrogate_zone_thermal.onnx` reference whose
  SHA-256 is checked at load time.
- **#1139** — v3.0 surrogate training and ONNX export; the
  upstream of the `SurrogateThermalModel` that
  `ThermalModelTrait` already dispatches to per `ARCHITECTURE.md
  §ML Surrogate Path`.
- **#1431** — `HybridRouting` (`use_surrogate_conduction`,
  `use_surrogate_ventilation`, `use_surrogate_loads`,
  `use_surrogate_hvac`, `use_ood_fallback`) — the routing
  flags the future `MojoThermalModel` would integrate alongside.
- **#2457 / #1892** — `use_ood_fallback` flag lineage; the
  fall-back path the Mojo backend would inherit.
- **#2921** — `predict_loads_into` zero-alloc path; the
  reference implementation the Mojo backend would integrate
  with.
- **#2529** — `validate_model_path` (the path-validation gate
  bypassed by the pre-#2905 ONNX env-var path).
- **[ADR-0013](../adr/0013-mojo-surrogate-spike.md)** —
  companion ADR: spike's evaluation framework + trait-gating
  contract.
- **[ADR-0012](../adr/0012-mojo-evaluation-roadmap.md)** —
  umbrella ADR: 3-phase Mojo roadmap and the v1.3 release-gate
  dependency.
- **[ADR-0001](../adr/0001-no-parameter-tuning-rule.md)** —
  operationalizes the `1e-5` tolerance band (the spike may not
  relax tolerance to absorb a regression).
- **`docs/investigations/issue-2940-mojo-evaluation-roadmap.md`**
  — the umbrella investigation that captures the cross-phase
  Mojo state and the umbrella release-gate dependency.

## 11. External references

- **`ARCHITECTURE.md`** — swap-point trait contracts;
  - `§ThermalModelTrait in sim/thermal_model.rs` is the canonical
    Mojo-backend entry point;
  - `§HeatConductionSolver in physics/solver_trait.rs` —
    consideration if Phase 1 (`#2938`) demonstrates surrogate-grade
    SIMD gains that warrant a follow-on `MojoHeatConductionSolver`;
    Phase 2 alone does NOT introduce one;
  - `§VentilationSchedule in sim/ventilation.rs` — NOT a candidate
    for the Mojo spike (no ML surrogate swap-point at the
    ventilation layer per `ARCHITECTURE.md §Note on Solar trait`).
- **`CODEBASE_MAP.md`** — cross-language FFI contracts, memory
  ownership, serialization formats (the spike write-up references
  the shared-library-load boundary).
- **`AGENTS.md` §"Mojo toolchain (advisory, optional)"** —
  operator-level install + verify guidance.
- **`AGENTS.md` §"Toolchain Quirks → Crate size"** — `.cargoignore`
  keeps the published crate < 10 MB; any Mojo build artifact
  outside `src/` stays out of publish.
- **`docs/agents/mojo-setup.md`** — three install paths (pixi,
  uv, legacy `modular` CLI), Windows / WSL notes,
  troubleshooting. **Read-only reference** for this PR; not
  modified by this PR.
- **`scripts/check_mojo_toolchain.sh`** — advisory detect gate.
- **`scripts/generate_scorecard.py`** — regenerates `SCORECARD.md`
  from committed sources; the `scorecard-drift` workflow enforces
  freshness on every PR.
- **`SCORECARD.md`** — current **14.3% pass rate** (12 / 84
  metrics); 51.03% MAE; throughput gate green at ≈ 157 cfg/s.
  The 60% floor in `release_gates.yaml →
  validation.min_pass_rate` is the precondition for all three
  Mojo phases.
- **`release_gates.yaml`** —
  - `validation.min_pass_rate = 60.0` (patch releases relax to
    40% via `release_requirements.patch.min_pass_rate`);
  - `validation.max_mae = 50.0` (current 51.03% → fails);
  - `benchmark.throughput.min_configs_per_sec = 150` (current
    ~157 cfg/s → passes);
  - the #2906-inspired `Surrogate Drift Gate` fail-closed check
    on `models/surrogate_zone_thermal.onnx` SHA-256 integrity.
- **`RULES.md`** — "no parameter tuning" + "must-never hardcode
  results" — operationalized on the spike via the fixed `1e-5`
  tolerance band.
- **`.cargoignore`** — strips `docs/`, `models/`, etc. from
  publish so the spike's evaluation directory does not bloat
  the published crate.
- **<https://mojolang.org/install/>** — upstream Mojo install
  reference.
- **<https://docs.modular.com/cli/>** — `max` CLI reference.
