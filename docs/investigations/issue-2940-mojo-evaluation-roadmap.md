# Issue #2940: Mojo Evaluation Multi-Phase Roadmap — Methodology Investigation

**Issue:** [#2940](https://github.com/anchapin/fluxion/issues/2940)
**Date:** 2026-08-17
**Investigator:** architecture-tracking sub-agent
**Branch:** `fix/issue-2940-mojo-epic`
**Status:** 🔄 **Investigation in progress — decision routed to maintainers**

## TL;DR

Issue #2940 is a **deferred** tracking epic for a 3-phase Mojo evaluation
plan. The Rust core remains the source of truth per `ARCHITECTURE.md`
and `RULES.md`; Mojo is evaluated as a **candidate backend** for specific
swap-point traits (`ThermalModelTrait`) — never a replacement, never a
parallel physics path. As of 2026-08-17 the v1.3 release gate is red
(**14.3% pass rate** vs 60% floor in `release_gates.yaml`,
`SCORECARD.md`), so all three phases remain `⏸ Deferred`. The
companion ADR (`docs/adr/0012-mojo-evaluation-roadmap.md`) records the
three-phase plan and the trait-gating contract; this document captures
the current Mojo state, the v1.3 release-gate dependency, and the
per-phase acceptance criteria that any future Mojo PR must satisfy.

## 1. Background

Building energy modeling is shifting toward differentiable simulations,
massive parametric batching, and neuro-symbolic workflows. The
[Mojo language](https://mojolang.org/) and its
[MAX framework](https://docs.modular.com/) promise SIMD-vectorized
Python-adjacent numerics, graph-level operator fusion between
surrogate inference and thermal state updates, and hardware
portability across CPU SIMD and GPU backends without a per-target
C++/CUDA stack. Issue #2940 captures a three-phase exploration plan
to assess whether Mojo can carry specific Fluxion swap-point traits
(`ThermalModelTrait`, possibly `HeatConductionSolver`) as a *second
backend* behind the validated Rust core.

The plan is intentionally conservative:

- The Rust core is the source of truth; Mojo never ships in a
  production build path.
- No Cargo dependency is added; Mojo is an evaluation-time
  developer tool only.
- The Mojo toolchain (`mojo`, `max`) is **never** required by CI;
  `bash scripts/check_mojo_toolchain.sh` is an **advisory** gate
  that always exits 0 in the default mode.
- The integration path, if Mojo is ever adopted, is a new
  `MojoThermalModel` struct that implements `ThermalModelTrait`
  (`src/sim/thermal_model.rs`) and is selected at the call site
  only — never a parallel physics path.

## 2. Current Mojo state (as of 2026-08-17)

### 2.1 Toolchain availability

- **Install guide** committed at `docs/agents/mojo-setup.md` (Issue
  #2979). Three install paths: `pixi` (recommended), `uv`
  (lightweight), legacy `modular` CLI (`curl get.modular.com | sh`).
  Windows users go through WSL2.
- **Advisory gate** at `scripts/check_mojo_toolchain.sh` —
  detects `mojo` and `max` on PATH, prints `PASS` / `WARN` / `FAIL`,
  always exits 0 in default (advisory) mode. Use `--strict` for
  agent pre-flight when a hard signal is needed.
- **Verify** after install: `mojo --version` and `max --version`
  must both print non-empty version strings in a fresh shell.
- **Binary locations**: `~/.modular/bin` (legacy installer) or
  inside the pixi / uv project venv (modern installs). The script
  does NOT assume a specific install path — it relies on
  `command -v` resolution.

### 2.2 Repository state

- `Cargo.toml` does NOT depend on `mojo` (any version). Mojo is not
  a crate, not a workspace member, not a feature.
- `deny.toml` and `.cargo/audit.toml` are NOT amended for Mojo.
  Cargo supply-chain gates ignore the Mojo ecosystem entirely.
- `.cargoignore` already strips `docs/`, `models/`, `tests/`, etc.
  from publish, so the published crate stays < 10 MB regardless of
  what Mojo prototype directory lands in the repo.
- `AGENTS.md` §"Mojo toolchain (advisory, optional)" is the
  operator-level entry point; the install guide is one click away.

### 2.3 Tracking issues (state of each phase)

| Phase | Tracking issue | Status | First action |
|-------|----------------|--------|--------------|
| Phase 1 — Isolated physics & SIMD benchmarking | #2938 | ⏸ Deferred | Create benchmark prototype once v1.3 gate clears 60% |
| Phase 2 — MAX surrogate integration & operator fusion | #2937 | ⏸ Deferred | Spike on MAX graph ops + ONNX parity once v1.3 gate clears 60% |
| Phase 3 — Ecosystem & interoperability assessment | (none) | ⏸ Deferred (no tracking issue) | Create tracking issue only when Phase 2 lands |

All three phases are explicitly marked `⏸ Deferred` in Issue #2940.
No work has started on any of them.

## 3. The v1.3 release-gate dependency

The 60% pass-rate gate is the explicit precondition for this epic.
`SCORECARD.md` (regenerated 2026-08-16 by
`scripts/generate_scorecard.py`) reports the current state:

| Metric | Current | Budget (gate) | Status | Source |
|--------|---------|---------------|--------|--------|
| ASHRAE 140 pass rate | **14.3%** (12/84 metrics) | ≥ 60% (`validation.min_pass_rate`) | ❌ Fail | `docs/ASHRAE140_RESULTS.md` |
| Mean Absolute Error (MAE) | **51.03%** | ≤ 50% (`validation.max_mae`) | ❌ Fail | `docs/ASHRAE140_RESULTS.md` |
| BatchOracle throughput | **157 (CI) / 900 (release)** cfg/s | ≥ 150 (`benchmark.throughput.min_configs_per_sec`) | ✅ Pass | `release_gates.yaml` + `README.md` |
| Max single-case deviation | 470.11% | (ref `individual.max_deviation` = 100%) | ℹ️ | `docs/ASHRAE140_RESULTS.md` |

Per `release_gates.yaml`:

- Major / minor releases require validation + benchmark + drift
  gates: `validation.min_pass_rate = 60.0`.
- Patch releases relax to **40%** (`release_requirements.patch` →
  `min_pass_rate: 40.0`).
- The known structural failures (Case 600 low-mass, Case 900
  high-mass) are excluded from the strict ±15% annual-energy gate
  and from the `extreme_deviation_limit` count.

Until the pass-rate gate clears, Mojo work is structurally out of
scope. The pre-existing surrogate / AI issues that actually block
the gate are the real priority:

- #2923, #2924, #2922, #2921, #2919, #2920, #2925 (multi-zone
  cold-start, hybrid perf, surrogate MAE)
- #2906, #2905, #2882 (ONNX surrogate & supply-chain)

These are listed in Issue #2940's "Critical Path Discipline"
section and are the dependencies that gate the Mojo exploration
back-end, not the other way around.

## 4. The three phases in detail

### 4.1 Phase 1 — Isolated physics & SIMD benchmarking (#2938)

**Focus:** Pure computational throughput on mathematical bottlenecks.

**Status:** ⏸ Deferred

**Key milestones:**

- Benchmark isolated solar radiation and 5R1C nodal calculation
  in Mojo (the two pure-compute kernels in `fluxion-core` and
  `src/sim/`).
- Compare compile times, binary footprint, and auto-vectorization
  against the existing Rust kernels.
- Reproduce exact numeric outputs of the existing Rust kernels
  within `1e-9` relative tolerance (no algorithmic drift).

**Acceptance criteria (from ADR-0012):**

- Numeric outputs reproduce the Rust kernels within `1e-9` relative
  tolerance.
- No algorithmic drift (the Mojo port is structurally identical to
  the Rust implementation, not a re-derivation).
- Per-kernel throughput ratio (Mojo / Rust) is reported in a CI
  workflow committed alongside the prototype.
- The prototype lives in its own evaluation directory, not in
  `src/` or `fluxion-core/`.

**Forbidden-by:** RULES.md "must-never hardcode results" (the
prototype is a measurement, not a tuning target).

### 4.2 Phase 2 — MAX surrogate integration & operator fusion (#2937)

**Focus:** Neuro-symbolic boundary and hardware portability.

**Status:** ⏸ Deferred

**Key milestones:**

- Implement a surrogate neural model in Mojo using MAX framework
  graph ops.
- Measure cross-kernel fusion benefits between surrogate inference
  and thermal state updates.
- Test portability across CPU SIMD and GPU backends.
- Reproduce ONNX reference outputs within `1e-5` tolerance (no
  tuning to match).

**Acceptance criteria (from ADR-0012):**

- ONNX reference outputs reproduced within `1e-5` tolerance.
- Cross-backend parity demonstrated on CPU SIMD **and** GPU (the
  latter only when a GPU runner is wired in CI; the project's
  `cuda` feature is opt-in).
- The MAX prototype is benchmarked against the existing
  `fluxion-zone-thermal.onnx` reference model (the model whose
  SHA-256 is checked at load time per #2906).

**Forbidden-by:** ADR-0001 (No-Parameter-Tuning Rule) — the
surrogate must reproduce the ONNX reference, not the
solver's tolerances.

### 4.3 Phase 3 — Ecosystem & interoperability assessment

**Focus:** Developer experience and integration with Python BEM
tooling.

**Status:** ⏸ Deferred (no tracking issue yet)

**Key milestones:**

- Evaluate Mojo's Python interop ergonomics compared to the
  existing `pyo3` bindings (`src/python/`).
- Assess maturity of package management, testing frameworks, and
  CI/CD automation.
- Cross-reference against existing PyO3 work (`src/python/`,
  `src/napi/`) and the `fluxion-mcp` package.
- Produce final recommendation on whether to adopt Mojo for
  specific modules (e.g., batch GPU solvers or surrogate
  backends) or retain pure Rust.

**Acceptance criteria (from ADR-0012):**

- A written recommendation is committed to `docs/investigations/`.
- A follow-up ADR (Proposed or Accepted) is recorded if adoption
  proceeds.
- The recommendation cites specific evidence from Phases 1 and 2
  (the Phase 3 writeup is the synthesis, not a primary measurement).

**Tracking:** None. A tracking issue is created only when Phase 2
lands and produces a Phase 3 input.

## 5. The trait-gating contract

If Mojo is ever adopted, integration is **gated behind a new trait
implementation** per `ARCHITECTURE.md` swap-point patterns. The
candidate entry point is `ThermalModelTrait` (`src/sim/thermal_model.rs`),
which is already the dispatch point for the `physics` / `surrogate`
/ `hybrid` (`HybridThermalModel + HybridRouting`) backends.

A future `MojoThermalModel` struct would:

1. Implement `ThermalModelTrait` (same `step` / `zone_state`
   / `set_zone_state` signatures).
2. Be selected at the call site only (e.g., a new
   `ThermalModelKind::Mojo` enum variant), never as a parallel
   physics path.
3. Delegate the heavy compute to a Mojo-compiled shared library
   (`.so` / `.dylib` / `.dll`) loaded via `libloading`; the Rust
   side stays in charge of validation, ASHRAE 140 dispatch, and
   test gating.
4. Never appear in `Cargo.toml` as a feature; the Mojo toolchain is
   a developer-workstation concern, not a build-time requirement.

This contract is recorded in `ADR-0012` §"Plan" and is the
canonical integration path. It is the only path; there is no
"evaluate Mojo as a parallel physics kernel" path, because that
path would require running the v1.3 ASHRAE 140 suite twice and
would violate the "no parameter tuning" rule the moment one path
drifts from the other.

## 6. What this PR ships

1. **`docs/adr/0012-mojo-evaluation-roadmap.md`** — the canonical
   ADR that records the three-phase plan, the trait-gating
   contract, and the v1.3 release-gate dependency. Status:
   Proposed.
2. **This investigation document** — the standalone analysis with
   the current Mojo state, the v1.3 release-gate dependency, and
   the per-phase acceptance criteria.
3. **A tracking comment on Issue #2940** — the maintainer-facing
   record of the ADR + investigation link, with the current
   SCORECARD.md pass rate surfaced inline.

This PR does **NOT**:

- Modify the production `ThermalModelTrait` (`src/sim/thermal_model.rs`).
- Modify the `Mojo` toolchain detection or install flow.
- Add a Mojo crate, feature, or workspace member.
- Amend `deny.toml` or `.cargo/audit.toml`.
- Touch `src/`, `fluxion-core/`, `tests/`, or any binary
  (`src/bin/`, `tools/`).
- Bump any Cargo dependency, including `candle-onnx`,
  `ort-sys`, or any ONNX / ML crate.

## 7. Recommendation

The decision is recorded in `ADR-0012` and is **deferred to
maintainers**. The three phases are mutually independent at the
implementation level but **sequentially dependent at the
go/no-go level**: Phase 2 needs Phase 1's prototype as a
reference; Phase 3 needs Phases 1 and 2's measurements as
inputs.

**Maintainer decision tree:**

- If the goal is "make the v1.3 release gate clear", prioritize
  the pre-existing surrogate / AI issues (#2923, #2924, #2922,
  #2921, #2919, #2920, #2925, #2906, #2905, #2882) before any
  Mojo work. The Mojo exploration is a post-gate activity.
- If the goal is "begin Phase 1 prototype work", wait until the
  pass-rate gate clears 60% (or 40% for a patch release). Use the
  tracking issue #2938 as the coordination point.
- If the goal is "adopt Mojo for production", the trait-gating
  contract in §5 above is the only path; a follow-up ADR is
  required before any `MojoThermalModel` lands in `src/`.

Per `AGENTS.md` / `RULES.md` / `ADR-0001`, this PR ships
**documentation and tracking only**; the implementation decision
is deferred to maintainers and tracked in Issue #2940 with the
companion ADR-0012.

## 8. Related issues and references

- **#2940** — Umbrella Mojo roadmap epic (this investigation's
  origin and the ADR-0012 source issue).
- **#2937** — Phase 2 tracking (Mojo & MAX framework evaluation
  for AI surrogate kernels).
- **#2938** — Phase 1 tracking (vectorized 5R1C + Perez Sky
  Model Mojo prototype).
- **#2979** — Mojo toolchain install guide; origin of
  `docs/agents/mojo-setup.md` and `scripts/check_mojo_toolchain.sh`.
- **#3059** — Architectural unblocker for the aggressive-baseline
  cohort (Case 600 / 900 high-mass / solar-coupling cohort).
- **#1465 / #1462** — `GaugeSolver` production switchover; the
  per-surface path tracked in shadow mode via `PhysicsAdapter`.
- **#2906** — SHA-256 integrity check for the ONNX surrogate
  model (`verify_onnx_signature` in `src/ai/surrogate.rs`).
- **#2923, #2924, #2922, #2921, #2919, #2920, #2925, #2905,
  #2882** — pre-existing surrogate / AI / supply-chain issues
  that block the v1.3 release gate and therefore gate the entire
  Mojo exploration.

## 9. External references

- `ARCHITECTURE.md` — `ThermalModelTrait` is the canonical
  swap-point trait; the Mojo backend is gated behind a new
  struct that implements this trait.
- `AGENTS.md` §"Mojo toolchain (advisory, optional)" — operator
  install + verify guidance.
- `docs/agents/mojo-setup.md` — three install paths (pixi, uv,
  legacy `modular` CLI), Windows / WSL notes, troubleshooting.
- `scripts/check_mojo_toolchain.sh` — advisory detect gate.
- `scripts/generate_scorecard.py` — regenerates `SCORECARD.md`
  from committed sources; the `scorecard-drift` workflow enforces
  freshness on every PR.
- `SCORECARD.md` — current 14.3% pass rate; 51.03% MAE; throughput
  gate green at ≈ 157 cfg/s. The 60% floor in
  `release_gates.yaml` → `validation.min_pass_rate` is the
  precondition for all three phases.
- `release_gates.yaml` → `validation.min_pass_rate = 60.0`;
  patch releases relax to 40%
  (`release_requirements.patch.min_pass_rate = 40.0`).
- `RULES.md` — no parameter tuning; no hardcoded physics results;
  ASHRAE 140 blind validation; numerical-reasoning-via-code.
- `ADR-0001` — No-Parameter-Tuning Rule (any future Mojo work
  must not relax tolerance bands to obtain a passing result).
- `ADR-0012` — Companion ADR: multi-phase Mojo evaluation
  roadmap, trait-gating contract, and v1.3 release-gate
  dependency.
- <https://mojolang.org/install/> — upstream Mojo install
  reference.
- <https://docs.modular.com/cli/> — `max` CLI reference.
