# ADR-0013: Mojo & MAX Surrogate Spike — Tracking Stub (Issue #2937)

> **Summary 1/7:** Issue #2937 is a deferred evaluation-only spike on Modular's Mojo language and MAX framework as a *backend* for the `ThermalModelTrait` ML-surrogate swap-point; no implementation ships in this PR.
> **Summary 2/7:** Status remains ⏸ **Deferred** because the broader v1.3 release gate (`release_gates.yaml → validation.min_pass_rate = 60.0`) is still red — current 14.3% metric pass rate per `SCORECARD.md` — even though the 9 specific blockers listed in #2937's "Why This Is a Spike" section are now CLOSED (verified 2026-08-17).
> **Summary 3/7:** Phase 2 of the 3-phase umbrella in [ADR-0012](0012-mojo-evaluation-roadmap.md) (epic #2940); sibling to Phase 1 (#2938, vectorized 5R1C + Perez Sky Model) and Phase 3 (no tracking issue yet, ecosystem + Python interop).
> **Summary 4/7:** Acceptance criteria for any future implementation: ONNX surrogate outputs reproduced within **1e-5** tolerance (no tuning); CPU SIMD AND GPU portability demonstrated (GPU only when wired in CI); microbenchmarks at batch sizes **1, 100, 1000, 10000**; integration write-up with `fluxion-core` + Rust + PyO3 + Node bindings.
> **Summary 5/7:** Trait-gating contract: any future `MojoThermalModel` (or equivalent) implements `ThermalModelTrait` (`src/sim/thermal_model.rs`) and is selected at the call site only — never as a parallel physics path; this preserves the v1.3 ASHRAE 140 single-source-of-truth invariant per `ARCHITECTURE.md`.
> **Summary 6/7:** Mojo never ships in a production build path: no Cargo dependency is added, `deny.toml` / `.cargo/audit.toml` are untouched, the advisory `scripts/check_mojo_toolchain.sh` gate stays non-blocking, the Mojo prototypes live in their own evaluation directory (not `src/`, `fluxion-core/`, `tests/`, or any binary).
> **Summary 7/7:** Per `RULES.md` ("must-never hardcode results") + `ADR-0001` (No-Parameter-Tuning Rule), tolerance bands may not be relaxed to obtain a passing result; the spike must reproduce the existing ONNX output, not improve it.

- **Status:** Proposed (tracking stub only — no implementation recorded)
- **Date:** 2026-08-17
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** v1.3 ASHRAE 140 release gate clearing 60% pass rate (`release_gates.yaml → validation.min_pass_rate`); the 9 specific blockers from #2937 are all CLOSED but the umbrella-gate is still red so the spike remains deferred
- **Issue:** [#2937](https://github.com/anchapin/fluxion/issues/2937)
- **Related:** #2940 (umbrella epic), #2938 (Phase 1 — vectorized 5R1C + Perez Sky Model Mojo prototype), #2979 (Mojo toolchain install guide → `docs/agents/mojo-setup.md`), #3059 (architectural unblocker coordination), #1465 / #1462 (`GaugeSolver` production switchover), `AGENTS.md` §"Mojo toolchain (advisory, optional)"

---

## Context

Fluxion's hybrid neuro-symbolic architecture pairs a validated first-principles
thermal network with neural surrogates for CFD approximations and complex
radiation models. Today, ML inference bridges the Rust core through external
runtimes — ONNX Runtime (`src/ai/surrogate.rs`, the default) and optionally
LibTorch — across FFI / binding layers that introduce serialization and
memory-boundary overhead. Issue #2937 is a **research artifact** — explicitly
**deferred from v1.3** — proposing Modular's [Mojo language][mojo] and the
[MAX framework][max] as a candidate *second backend* behind the existing
ONNX surrogate path, on the hypothesis that Mojo's MLIR-native tensor
abstractions and MAX's graph-level operator fusion can carry both surrogate
inference and physical tensor operations in a unified pipeline with lower FFI
cost and direct hardware portability across CPU SIMD and GPU backends.

[mojo]: https://mojolang.org/
[max]:  https://docs.modular.com/

The spike is the second phase of a three-phase Mojo exploration recorded in
[ADR-0012](0012-mojo-evaluation-roadmap.md) (umbrella epic #2940). The three
phases run sequentially at the **go/no-go level** (each phase needs the prior
phase's measurement as input) while remaining **mutually independent at the
implementation level**:

| Phase | Tracking issue | Focus | Status |
|-------|----------------|-------|--------|
| Phase 1 — Isolated physics & SIMD benchmarking | #2938 | Vectorized 5R1C + Perez Sky Model Mojo prototype | ⏸ Deferred |
| Phase 2 — **MAX surrogate integration & operator fusion** (this ADR) | #2937 | Surrogate neural model in Mojo using MAX graph ops | ⏸ Deferred |
| Phase 3 — Ecosystem & interoperability assessment | (none yet) | Python interop + package management + CI/CD maturity | ⏸ Deferred |

### Why this is still a spike (not an active task)

The spike's own body — and ADR-0012 §"Plan: Phase 2" — explicitly states:

> "this work does not start until the `ai-surrogate` critical issues above
> are closed."

The body enumerates **nine** `ai-surrogate` critical issues as the spike's
direct gating dependencies. As of 2026-08-17 every one of the nine is
**CLOSED** (verified by `gh issue view <N> --json state`):

| # | Title | State (2026-08-17) |
|---|-------|--------------------|
| #2923 | `SurrogateDriftToleranceGate` silently downgrades to advisory when no ONNX model is registered | CLOSED |
| #2924 | ASHRAE 140 surrogate-driven Cases 600–900 cooling has no MAE gate at the surrogate layer | CLOSED |
| #2922 | No PR-blocking throughput / latency gate for `HybridThermalModel` | CLOSED |
| #2921 | `HybridThermalModel::solve_timesteps` calls allocating `predict_loads_with_fallback` instead of zero-alloc `predict_loads_into` | CLOSED |
| #2919 | No PR-blocking cold-start latency gate for ONNX surrogate | CLOSED |
| #2920 | `FLUXION_GPU` / `FLUXION_ONNX_BACKEND` downgrade path is silent — operator misconfiguration invisible | CLOSED |
| #2925 | `HybridThermalModel` `clone` counter-preservation asymmetry | CLOSED |
| #2906 | Loaded ONNX model has no integrity verification (now `verify_onnx_signature` SHA-256 check) | CLOSED |
| #2905 | ONNX env-var path bypasses `validate_model_path` (#2529) | CLOSED |

The closure of the nine blockers is necessary but **not sufficient** to
unblock the spike. The umbrella Mojo roadmap in ADR-0012 binds Phase 2 to
the broader v1.3 ASHRAE 140 release gate (`release_gates.yaml →
validation.min_pass_rate = 60.0` for major / minor releases; `40.0` for
patch releases). Per `SCORECARD.md` (regenerated 2026-08-16 by
`scripts/generate_scorecard.py`) the current metric-level pass rate is
**14.3%** (12 PASS / 8 WARN / 64 FAIL of 84 metrics) and MAE is
**51.03%** vs the 50% cap. Both headline gates fail. Only the throughput
gate (`benchmark.throughput.min_configs_per_sec ≥ 150`) is green (~157
cfg/s on the CI runner). **The umbrella gate is red, so this spike
remains ⏸ Deferred.**

A maintainer who wishes to revisit the deferral must independently clear
the umbrella gate. The nine specific blockers having closed is recorded
here as a state-of-play update — it does **not** by itself constitute the
v1.3 release-gate clearance that ADR-0012 §"Plan: Phase 2" requires.

## Decision

**No implementation is made in this PR.** This ADR remains **Proposed**
and records only the spike's evaluation framework, its acceptance
criteria, and its trait-level integration contract — nothing more. The
PR adds:

1. **This ADR** — `docs/adr/0013-mojo-surrogate-spike.md` — spike's
   evaluation framework and trait-gating decision contract.
2. **Companion investigation** —
   `docs/investigations/issue-2937-mojo-surrogate-spike.md` —
   captures the spike's framework in standalone form, enumerates the
   nine blockers (now CLOSED) and the umbrella release-gate dependency,
   and states the acceptance criteria a future spike PR would have to
   satisfy to move the spike from ⏸ Deferred to active.
3. **Tracking comment on Issue #2937** — the maintainer-facing record of
   the ADR + investigation links, with the current state-of-play
   (9 blockers CLOSED, umbrella gate still red) surfaced inline.

No production physics, validation logic, solver, ONNX surrogate,
ASHRAE 140 validator, `ARCHITECTURE.md`, `RULES.md`, `Cargo.toml`,
`deny.toml`, `.cargo/audit.toml`, `AGENTS.md`, or `release_gates.yaml`
change is part of this decision. No `MojoThermalModel` struct is
introduced. The Mojo toolchain is **not** installed or activated by this
PR; the advisory `bash scripts/check_mojo_toolchain.sh` gate continues
to be non-blocking and may keep producing `WARN` lines on CI.

## Plan — the spike evaluation framework

Once the v1.3 release gate clears 60% pass rate (or 40% on a patch
release per `release_requirements.patch.min_pass_rate`), the spike may
begin. Any future spike PR must satisfy all of the following:

### 1. Scope (matching #2937's "Proposed Prototype Scope")

- **Workload:** A small surrogate neural network (3–4 dense layers) in
  Mojo predicting one of (a) interior convection coefficients, (b)
  Perez radiation factors. The exact scope is open; the choice will be
  recorded in the spike PR with a rationale (the choice of target is a
  measurement question, not a tuning question — both targets are
  acceptable per RULES.md as long as the Mojo port reproduces the
  existing ONNX reference output, not improves it).
- **Evaluation batch sizes:** `1`, `100`, `1000`, `10000` evaluations
  per call. Throughput, p50 / p99 latency, and per-call memory
  transfer bytes are recorded at each batch size for the
  microbenchmark report.
- **Backend matrix:** CPU SIMD (AVX-512 on x86_64, NEON on
  aarch64 / Apple silicon) **and** GPU (NVIDIA via the project's
  `cuda` feature flag — the GPU side may run on the GPU runner only and
  skip on CPU-only CI per the existing `cargo test --features cuda
  --test surrogate_cuda_smoke` pattern).
- **Hardware portability:** No per-target C++ / CUDA source code; the
  same Mojo source must compile and execute correctly on CPU SIMD and
  GPU without source-level divergence.

### 2. Acceptance criteria (matching #2937's "Acceptance Criteria")

- **Numeric parity:** Mojo / MAX prototype reproduces the existing
  ONNX surrogate (`models/surrogate_zone_thermal.onnx`, SHA-256 verified
  per #2906's `verify_onnx_signature` in `src/ai/surrogate.rs`) within
  **`1e-5` relative tolerance** across the full batch-size sweep.
  **No parameter tuning** is permitted to obtain parity — per
  `RULES.md` ("must-never hardcode results") + `ADR-0001`
  (No-Parameter-Tuning Rule), the tolerance band is fixed at `1e-5`
  and may **not** be relaxed to absorb a regression.
- **Microbenchmark report:** Throughput (calls / second) and memory
  footprint compared against the existing ONNX Runtime backend (via the
  `Fluxion ONNX` path in `src/ai/surrogate.rs`) and against a Rust
  reference implementation that exercises the same `predict_loads_into`
  path used by `HybridThermalModel::solve_timesteps` (#2921). The
  report is committed under `docs/investigations/` with a stable file
  name and cross-referenced from this ADR.
- **Integration write-up:** A technical note on (a) how the Mojo
  backend would ship behind the `ThermalModelTrait` swap-point
  (without adding a Cargo dependency), (b) how PyO3 (`src/python/`)
  and Node bindings (`src/napi/`) would consume the new backend, (c)
  what additional CI infrastructure would be required (the existing
  `scripts/check_mojo_toolchain.sh` advisory gate is necessary but not
  sufficient — a non-blocking *evaluation* workflow is part of the
  write-up).
- **Cross-backend parity check:** CPU SIMD AND GPU produce identical
  numeric output within `1e-5`. The GPU parity check is opt-in via
  the `cuda` feature flag; the spike PR must record the parity result
  on the GPU runner (or `SKIP` with rationale on CPU-only CI).

### 3. Trait-gating contract (matching `ARCHITECTURE.md` swap-point patterns)

Per `ARCHITECTURE.md` and `ARCHITECTURE.md §`ML Surrogate Path``, the
ML-surrogate path is a `ThermalModelTrait` (`src/sim/thermal_model.rs`)
implementation selected at the call site; it is **never** a parallel
physics path. Any future Mojo backend must follow the same contract:

1. Define a `MojoThermalModel` (or equivalently named) struct that
   implements `ThermalModelTrait`. The trait method signatures are
   fixed by `ARCHITECTURE.md §ThermalModelTrait in
   sim/thermal_model.rs` (`num_zones`, `get_temperatures`,
   `set_temperatures`, `mode`, `set_mode`, `solve_timesteps`,
   `apply_parameters`, `zone_area`, `heating_setpoint`,
   `cooling_setpoint`, `hvac_power_demand`, `is_valid`,
   `get_comfort_metrics`).
2. Add a `ThermalModelKind::Mojo` (or equivalent) enum variant and
   thread it through the existing `ThermalModelBuilder` /
   `ThermalModelMode` dispatch in `src/sim/thermal_model.rs` —
   **never** a parallel physics path that would require running the
   v1.3 ASHRAE 140 suite twice.
3. Delegate the heavy compute (the trained surrogate weights, the
   per-zone inference step) to a Mojo-compiled shared library
   (`.so` / `.dylib` / `.dll`) loaded via `libloading` (or equivalent)
   on the developer workstation. The Rust side remains in charge of
   validation, ASHRAE 140 dispatch, and test gating.
4. The Mojo shared library is **not** a `Cargo.toml` dependency. The
   `.cargoignore` already strips `docs/`, `models/`, and `assets/`
   from publish; any Mojo build artifact under a dedicated evaluation
   directory outside `src/` stays out of the published crate.

This contract is the **only** integration path. There is no
"evaluate Mojo as a parallel physics kernel" path because that would
require running the v1.3 ASHRAE 140 suite twice and would violate
`RULES.md` ("no parameter tuning") the moment one path drifts from the
other.

### 4. Bindings write-up

The spike write-up must describe how the Mojo backend integrates with
the existing binding layers:

- **PyO3** (`src/python/`) — how the `Model` Python wrapper exposes
  the Mojo backend alongside the existing `physics` / `surrogate` /
  `hybrid` modes.
- **Node bindings** (`src/napi/`) — same question for the Node wrapper.
- **MCP server** (`fluxion-mcp/`, separate Cargo package) — does
  the Mojo backend survive the `default-features = false` boundary
  (yes, if the shared library is loaded at runtime, not at build
  time).
- **Batch orchestration** (`src/sim/thermal_model.rs`,
  `BatchOracle` per #2493) — how `BatchOracle::evaluate_population`
  would dispatch to the Mojo backend at the population level (not the
  inner per-config level, per the `BatchOracle` parallelism rule).

### 5. CI / supply-chain posture

- `bash scripts/check_mojo_toolchain.sh` stays **non-blocking**
  (always exits 0 in default advisory mode). The spike does NOT make
  Mojo a CI requirement; CI that lacks Mojo runs the existing
  ONNX-only path.
- The spike write-up sketches a future "evaluation" workflow
  (parallel to `.github/workflows/onnx-integrity.yml`) that runs the
  microbenchmark + tolerance check when the toolchain is present and
  is gated on `--strict` mode for agent pre-flight only.
- `cargo audit` / `cargo deny` supply-chain gates
  (`release_gates.yaml → ci.required_checks`, `deny.toml`,
  `.cargo/audit.toml`) are unaffected by the spike. Mojo never
  appears in `Cargo.toml`, `deny.toml`, or `.cargo/audit.toml`.

## Consequences

### Positive

- The v1.3 release-gate priority is preserved — this ADR records the
  state-of-play (9 / 9 listed blockers CLOSED, umbrella gate still
  red) without claiming a premature spike unblock.
- The trait-gating contract is recorded in advance so the future
  implementer (whenever the umbrella gate clears) has a single
  source-of-truth for the integration path.
- `RULES.md` ("must-never hardcode results") and `ADR-0001`
  (No-Parameter-Tuning Rule) are made operational on the spike:
  `1e-5` tolerance is fixed in the Acceptance Criteria and may not
  be relaxed to absorb a regression.
- The advisory `bash scripts/check_mojo_toolchain.sh` gate stays
  non-blocking, so a contributor laptop without Mojo never blocks
  CI.

### Negative

- The Mojo prototype directory referenced by #2937 is not in this PR.
  When the umbrella gate clears, the spike implementer is responsible
  for cross-referencing this ADR when their work begins; the body of
  the spike remains the canonical scope statement (the body is more
  current than this ADR for line-level scope questions).
- No quantitative evidence (yet) on whether Mojo will deliver the
  promised SIMD + operator-fusion benefits — that is the entire
  purpose of the spike and the reason the spike must run before any
  adoption decision.
- The 9-blocker-closure observation may be over-interpreted as
  "the spike is unblocked." It is not — the umbrella gate
  (`release_gates.yaml → validation.min_pass_rate`) is still red.
  This ADR records the state-of-play accurately but the spike's
  status remains ⏸ Deferred.

### Neutral

- **Phase 1 (#2938) and Phase 3 (no tracking issue yet)** remain
  independently gated on the same umbrella release-gate condition.
  This ADR scopes itself to Phase 2 only.
- **#3059 (aggressive-baseline cohort)**, **#1465 / #1462
  (`GaugeSolver` production switchover)**, and **#1139 (surrogate
  v3.0 training)** are independent architectural paths evaluated
  against the same v1.3 release gate. None pre-selects the Mojo
  backend over them.
- The `.cargoignore` already strips `docs/`, `models/`, and
  `assets/` from publish, so even when/if a Mojo prototype ends up
  under `docs/` or a sibling evaluation directory, it does not
  bloat the published crate (which stays < 10 MB per `AGENTS.md
  §Toolchain Quirks → Crate size`).
- The Mojo / MAX spike deliverable is **benchmarks + technical
  write-up**, not a Cargo dependency. The diffusion of "Mojo in
  production" requires a separate, future ADR that proves the
  measured gains in the spike justify the integration cost.

## References

- **Issue #2937** — origin and source scope statement. The issue
  body is the canonical scope statement; this ADR is the structural
  decision record.
- **Issue #2940** — umbrella Mojo roadmap epic (this ADR's parent).
- **Issue #2938** — Phase 1 (vectorized 5R1C + Perez Sky Model Mojo
  prototype), tracked separately under the same umbrella.
- **[ADR-0012](0012-mojo-evaluation-roadmap.md)** — companion
  structural-decision record for the 3-phase Mojo roadmap; this
  ADR is Phase 2's tracking stub.
- **[ADR-0001](0001-no-parameter-tuning-rule.md)** — the spike must
  not relax the `1e-5` tolerance band to obtain parity; tolerance is
  fixed.
- **Issue #2979** — Mojo toolchain install guide; origin of
  `docs/agents/mojo-setup.md` and `scripts/check_mojo_toolchain.sh`.
- **Issue #3059** — architectural unblocker coordination for the
  aggressive-baseline cohort (Case 600 / 900 high-mass /
  solar-coupling). Independent of the Mojo spike but routed through
  the same release-gate dependency.
- **Issue #1465 / #1462** — `GaugeSolver` production switchover; the
  per-surface path tracked in shadow mode via `PhysicsAdapter`.
- **Issue #2906** — `verify_onnx_signature` SHA-256 integrity check
  in `src/ai/surrogate.rs`. The spike's tolerance check exercises
  the same `models/surrogate_zone_thermal.onnx` reference whose
  SHA-256 is checked at load time.
- **Issue #2921 / #2922 / #2919 / #2923 / #2924 / #2925 / #2920 /
  #2905** — pre-existing surrogate / AI / supply-chain issues that
  gated the spike; all CLOSED (verified 2026-08-17). The umbrella
  release-gate dependency remains the active gating condition.
- **`docs/KNOWN_ISSUES.md`** — the §LIMIT / structural-failure
  index that the umbrella release-gate clears against.
- **`docs/investigations/issue-2940-mojo-evaluation-roadmap.md`** —
  the umbrella investigation that captures the cross-phase Mojo
  state and the umbrella release-gate dependency.
- **`docs/investigations/issue-2937-mojo-surrogate-spike.md`** —
  companion investigation (this PR's deliverable); the standalone
  spike-framework write-up.
- **`docs/agents/mojo-setup.md`** — three install paths (pixi, uv,
  legacy `modular` CLI), Windows / WSL notes, troubleshooting.
- **`scripts/check_mojo_toolchain.sh`** — advisory detect gate;
  always exits 0 in default mode; `--strict` for agent pre-flight.
- **`scripts/generate_scorecard.py`** — regenerates `SCORECARD.md`
  from committed sources; the `scorecard-drift` workflow enforces
  freshness on every PR.
- **`SCORECARD.md`** — current 14.3% pass rate; 51.03% MAE;
  throughput gate green at ≈ 157 cfg/s. The 60% floor in
  `release_gates.yaml → validation.min_pass_rate` is the
  precondition for all three Mojo phases.
- **`release_gates.yaml`** —
  - `validation.min_pass_rate = 60.0` (patch releases relax to 40%
    via `release_requirements.patch.min_pass_rate`);
  - `validation.max_mae = 50.0` (current 51.03% → fails);
  - `benchmark.throughput.min_configs_per_sec = 150` (current ~157
    cfg/s → passes);
  - the #2906-inspired `Surrogate Drift Gate` fail-closed check on
    `models/surrogate_zone_thermal.onnx` SHA-256 integrity.
- **`ARCHITECTURE.md`** — swap-point trait contracts;
  - `§ThermalModelTrait in sim/thermal_model.rs` is the canonical
    Mojo-backend entry point;
  - `§HeatConductionSolver in physics/solver_trait.rs` —
    consideration if Phase 1 (`#2938`) demonstrates surrogate-grade
    SIMD gains that warrant a follow-on `MojoHeatConductionSolver`;
    Phase 2 alone does not introduce one;
  - `§VentilationSchedule in sim/ventilation.rs` — not a candidate
    for the Mojo spike (no ML surrogate swap-point at the
    ventilation layer per `ARCHITECTURE.md §Note on Solar trait`).
- **`CODEBASE_MAP.md`** — cross-language FFI contracts, memory
  ownership, serialization formats (the spike write-up references
  the shared-library-load boundary).
- **`AGENTS.md`** §"Mojo toolchain (advisory, optional)" —
  operator-level install + verify guidance; `docs/agents/mojo-setup.md`
  is the expanded install guide.
- **`RULES.md`** — "no parameter tuning" + "must-never hardcode
  results" — operationalized on the spike via the fixed `1e-5`
  tolerance band.
- **`.cargoignore`** — strips `docs/`, `models/`, etc. from publish
  so the spike's evaluation directory does not bloat the published
  crate.
- **<https://mojolang.org/install/>** — upstream Mojo install
  reference.
- **<https://docs.modular.com/cli/>** — `max` CLI reference.
