# ADR-0012: Mojo Evaluation Multi-Phase Roadmap (Issue #2940)
> **Summary 1/7:** Mojo is a candidate *backend* for swap-point traits (`ThermalModelTrait`), not a Rust replacement, gated behind the v1.3 ASHRAE 140 release gate.
> **Summary 2/7:** The 3-phase plan is **Deferred** as of 2026-08-17; current metric-level pass rate is 14.3% (12/84) vs the 60% release-gate floor in `release_gates.yaml`.
> **Summary 3/7:** Phase 1 (#2938) benchmarks vectorized 5R1C + Perez Sky Model in Mojo; Phase 2 (#2937) evaluates MAX surrogate kernel fusion; Phase 3 (no tracking issue) is ecosystem + Python interop.
> **Summary 4/7:** This PR records only the roadmap; no production physics, validation, solver, or `ARCHITECTURE.md` change ships here.
> **Summary 5/7:** If Mojo is ever adopted, integration is gated behind a new trait implementation per `ARCHITECTURE.md` swap-point patterns — never a parallel physics path.
> **Summary 6/7:** The advisory `bash scripts/check_mojo_toolchain.sh` gate stays non-blocking; Mojo is an evaluation-time developer tool, not a Cargo dependency.
> **Summary 7/7:** The epic closes only when all three phases produce a documented recommendation AND no Rust-core change merges without independent v1.3 ASHRAE 140 gate validation.

- **Status:** Proposed (tracking stub only — no implementation recorded)
- **Date:** 2026-08-17
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** v1.3 ASHRAE 140 release gate clearing 60% (40% for patches); Issue #3059 architectural unblocker; #1465 / #1462 GaugeSolver production switchover
- **Issue:** [#2940](https://github.com/anchapin/fluxion/issues/2940)
- **Related:** #2937 (Phase 2), #2938 (Phase 1), #2979 (Mojo toolchain install guide), #3059 (aggressive-baseline cohort), #1465 / #1462 (GaugeSolver), #1467 (swap-point patterns), `AGENTS.md` §"Mojo toolchain (advisory, optional)"

---

## Context

Building energy modeling is shifting toward differentiable simulations, massive
parametric batching, and neuro-symbolic workflows. The Modular Mojo language
and its MAX framework promise (a) SIMD-vectorized Python-adjacent numerics,
(b) graph-level operator fusion between surrogate inference and thermal state
updates, and (c) hardware portability across CPU SIMD and GPU backends without
a per-target C++/CUDA stack. Issue #2940 captures a three-phase exploration
plan to assess whether Mojo can carry specific Fluxion swap-point traits
(`ThermalModelTrait`, possibly `HeatConductionSolver`) as a *second backend*
behind the validated Rust core, while the Rust core remains the source of
truth per `ARCHITECTURE.md` and `RULES.md`.

The current state of the v1.3 release ("Blind ASHRAE 140 Validation") is
**14.3% metric-level pass rate** (12 PASS / 8 WARN / 64 FAIL of 84 metrics) vs
the 60% floor in `release_gates.yaml` → `validation.min_pass_rate`
(`SCORECARD.md`, regenerated 2026-08-16). MAE is **51.03%** vs the 50% cap.
Both headline gates fail. The throughput gate (≥ 150 cfg/s) is the only
release-gate row currently green on the CI runner (≈ 157 cfg/s).

The 60% pass-rate gate is the explicit precondition for this epic. While
the gate is red, no Mojo phase begins, and any pre-existing surrogate / AI
issues that block the gate (#2923, #2924, #2922, #2921, #2919, #2920, #2925,
#2906, #2905, #2882) are the real priority. The Mojo toolchain itself is
already installable on developer workstations (`docs/agents/mojo-setup.md`),
and `bash scripts/check_mojo_toolchain.sh` is wired as an **advisory** gate
that always exits 0 in the default mode. Mojo never ships in a production
build path; the `.cargoignore` already strips `docs/` from publish, and the
Mojo ecosystem never appears in `Cargo.toml` / `deny.toml` / `audit.toml`.

## Decision

**No implementation is made in this PR.** This ADR remains **Proposed** and
records only the structural roadmap, its dependencies, and the trait-level
integration path that any future Mojo adoption must follow. This PR adds:

1. **This ADR** — the multi-phase roadmap and the trait-gating decision
   contract.
2. **`docs/investigations/issue-2940-mojo-evaluation-roadmap.md`** — the
   standalone investigation that captures the current Mojo state, the
   three phases in detail, and the v1.3 release-gate dependency.

No production physics, validation, solver, ONNX surrogate, ASHRAE 140
validator, `ARCHITECTURE.md`, or `RULES.md` change is part of this decision.
No Cargo dependency is added; the `deny.toml` / `audit.toml` supply-chain
config is untouched.

## Plan

Once the v1.3 release gate clears 60% (or 40% for a patch release), the
three phases may run **in sequence**, each gated on the previous phase
producing a written go/no-go assessment recorded in `docs/investigations/`:

### Phase 1 — Isolated physics & SIMD benchmarking (#2938)

- **Goal:** Compare Mojo compile time, binary footprint, and
  auto-vectorization against Rust on pure-compute kernels
  (solar radiation, 5R1C nodal calculation).
- **Acceptance:** Numeric outputs reproduce the existing Rust kernels
  within `1e-9` relative tolerance; no algorithmic drift; per-kernel
  throughput ratio (Mojo/Rust) reported with a CI workflow committed
  alongside the prototype.
- **Status:** ⏸ Deferred until v1.3 gate clears.
- **Tracking:** #2938 — "[Benchmark] Prototype Vectorized 5R1C Thermal
  Network & Perez Sky Model in Mojo".

### Phase 2 — MAX surrogate integration & operator fusion (#2937)

- **Goal:** Implement a surrogate neural model in Mojo using MAX
  graph ops, measure cross-kernel fusion benefits with thermal state
  updates, and validate GPU / CPU portability.
- **Acceptance:** ONNX reference outputs reproduced within `1e-5`
  tolerance; cross-backend parity demonstrated on CPU SIMD *and* GPU
  (the latter only when a GPU runner is wired in CI).
- **Status:** ⏸ Deferred until v1.3 gate clears.
- **Tracking:** #2937 — "[Spike] Evaluate Mojo and MAX Framework for AI
  Surrogate Kernel Execution".

### Phase 3 — Ecosystem & interoperability assessment

- **Goal:** Evaluate Mojo's Python interop ergonomics compared to the
  existing `pyo3` bindings (`src/python/`), assess maturity of package
  management, testing, and CI/CD, and produce a final
  go/no-go recommendation on whether to adopt Mojo for batch GPU
  solvers or surrogate backends.
- **Acceptance:** Written recommendation committed to
  `docs/investigations/` and a follow-up ADR (proposed or accepted)
  recorded if adoption proceeds.
- **Status:** ⏸ Deferred (no tracking issue yet — created only when
  Phase 2 lands).
- **Tracking:** None (placeholder).

Any future Mojo implementation PR must satisfy all of the following:

- A new `MojoThermalModel` (or equivalent) struct implements the
  `ThermalModelTrait` (`src/sim/thermal_model.rs`) swap point and is
  selected at the call site only, never as a parallel physics path.
- The v1.3 ASHRAE 140 release gate remains green (≥ 60% pass rate)
  on the existing Rust core throughout the Mojo development window.
- Energy balance (`tests/test_energy_conservation.rs`) and
  module-isolation tests remain green for the Mojo backend with the
  same tolerance as the Rust backend.
- No case-specific parameter, hardcoded output, or relaxed reference
  assertion is used to obtain a passing result — per `RULES.md`
  ("must-never hardcode results") and `ADR-0001` (No-Parameter-Tuning
  Rule).
- `cargo deny` and `cargo audit` supply-chain gates are unaffected
  (Mojo never appears in `Cargo.toml`).

## Consequences

### Positive

- The v1.3 release-gate priority is preserved; the Mojo exploration is
  correctly recorded as a post-gate activity, not a parallel
  obligation.
- Future maintainers have a single source of truth (this ADR) for
  the three-phase plan, the trait-level integration contract, and
  the no-Cargo-dependency rule.
- The advisory `bash scripts/check_mojo_toolchain.sh` gate stays
  non-blocking, so absence of Mojo on a contributor laptop or CI
  runner never blocks CI.

### Negative

- The Mojo prototype directory referenced by #2938 is not in this PR
  and may diverge from this ADR if maintained in isolation. The
  tracking issue owners are responsible for cross-referencing this
  ADR when their phases begin.
- No quantitative evidence (yet) on whether Mojo will deliver the
  promised SIMD + fusion benefits; this is the entire purpose of
  Phases 1 and 2 and the reason the phases must run before any
  adoption decision.

### Neutral

- Issue #3059 (aggressive-baseline cohort) and #1465 / #1462
  (GaugeSolver) remain the architectural unblockers for the
  ASHRAE 140 pass-rate gate; this ADR does not pre-select GaugeSolver
  or Mojo over each other — they are independent back-end paths
  evaluated against the same validation gate.
- The `.cargoignore` already strips `docs/`, `models/`, and
  `assets/` from publish, so even if a Mojo prototype ends up under
  `docs/` or a sibling directory, it does not bloat the published
  crate.

## References

- Issue #2940 — Umbrella Mojo roadmap epic (this ADR's origin).
- Issue #2937 — Phase 2 tracking (Mojo & MAX framework evaluation for
  AI surrogate kernels).
- Issue #2938 — Phase 1 tracking (vectorized 5R1C + Perez Sky Model
  Mojo prototype).
- Issue #2979 — Mojo toolchain install guide; origin of
  `docs/agents/mojo-setup.md` and `scripts/check_mojo_toolchain.sh`.
- Issue #3059 — Architectural unblocker for the aggressive-baseline
  cohort (Case 600/900 high-mass / solar-coupling cohort).
- Issue #1465 / #1462 — `GaugeSolver` production switchover; the
  per-surface path tracked in shadow mode via `PhysicsAdapter`.
- ADR-0001 — No-Parameter-Tuning Rule (any future Mojo work must
  not relax tolerance bands to obtain a passing result).
- `ARCHITECTURE.md` — swap-point trait contracts; `ThermalModelTrait`
  is the canonical entry point for a Mojo backend.
- `AGENTS.md` §"Mojo toolchain (advisory, optional)" — operator-level
  install + verify guidance; `docs/agents/mojo-setup.md` is the
  expanded install guide.
- `docs/agents/mojo-setup.md` — three install paths (pixi, uv,
  legacy `modular` CLI), Windows/WSL notes, troubleshooting.
- `scripts/check_mojo_toolchain.sh` — advisory detect gate (always
  exits 0 in default mode; `--strict` for agent pre-flight).
- `SCORECARD.md` — current 14.3% pass rate; 51.03% MAE; throughput
  gate green at ≈ 157 cfg/s. The 60% floor in
  `release_gates.yaml` → `validation.min_pass_rate` is the
  precondition for all three phases.
- `release_gates.yaml` → `validation.min_pass_rate = 60.0`; patch
  releases relax to 40% (`release_requirements.patch`).
- `RULES.md` — no parameter tuning; no hardcoded physics results;
  ASHRAE 140 blind validation; numerical-reasoning-via-code.
