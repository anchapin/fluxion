# Mutation Testing & Crate Split (#1255)

## Goal

Enable `cargo-mutants` in CI. Previously, mutation testing was **disabled**
(`.github/workflows/mutation-testing.yml`) because `cargo-mutants 27.x` required
~28 GB of RAM to analyze fluxion's type hierarchies, far exceeding the 7 GB
available on standard GitHub Actions runners.

**Success criteria:** `cargo mutants --list` completes with < 4 GB RAM; all tests
pass.

## How cargo-mutants memory works

`cargo-mutants` generates many small source mutations and **recompiles the target
crate for each mutant** (then runs the test suite). Workspace *dependencies* are
built once and cached. Therefore:

> Memory is dominated by **per-mutant recompilation of the target crate**, not by
> the number of mutants.

The 28 GB figure came from recompiling, for every mutant, a single giant crate that
pulled in `ort` (ONNX Runtime, with `download-binaries`), `faer`, `nalgebra`, `ndarray`,
and the full physics/AI/validation type hierarchy.

The fix is structural: **shrink the target crate** so each mutant compile is cheap,
and push heavy code into cached dependencies.

## What was done (Phase 1 — this PR)

Converted the repository into a **Cargo workspace** and split out the genuinely
*leaf* modules into a new `fluxion-core` crate:

```
fluxion-wt-1255/
├── Cargo.toml              # [workspace] root + main `fluxion` [package]
├── src/                    # the `fluxion` crate (sim, physics, ai, validation, …)
└── fluxion-core/
    ├── Cargo.toml          # `fluxion-core` package
    └── src/
        ├── lib.rs
        └── weather/        # MOVED here (true dependency leaf)
```

### Why `weather` first

A dependency analysis (`use crate::<module>` references) showed `weather` is the only
large module with **zero upward dependencies** — it never references `sim`, `physics`,
`ai`, or `validation`. It is a clean leaf and could be moved with no cycle risk.

### The re-export shim (zero call-site churn)

Moving a module normally forces rewriting every `use crate::weather::…` across ~19
files. Instead, the main crate re-exports the moved module:

```rust
// src/lib.rs
pub use fluxion_core::weather;   // was: pub mod weather;
```

Because of this, **every existing `crate::weather::…` and `fluxion::weather::…`
path resolves unchanged.** No call-site edits were required; `cargo build`
(217 crates) passes.

Doctest paths inside the moved files were rewritten `fluxion::weather` →
`fluxion_core::weather` so `cargo test --doc -p fluxion-core` stays green.

## What is NOT yet moved, and why (dependency cycles)

The issue's proposed split was:

| Proposed `fluxion-core` | Proposed `fluxion` |
|------------------------|--------------------|
| `ai/`, `physics/`, `validation/` | `sim/`, `weather/`, … |

A static dependency analysis of `use crate::<module>` statements revealed
**bidirectional cycles** that make this verbatim split impossible today:

```
physics ──uses──▶ sim::assembly::BuildingAssembly        (19 sites)
sim     ──uses──▶ physics::HeatConductionSolver, …       (64 sites)

validation ──uses──▶ sim::engine::ThermalModel, …        (18 sites)
sim        ──uses──▶ validation::…                       (24 sites)

ai  ──uses──▶ sim::engine::ThermalModel                   (1 site)
sim ──uses──▶ ai::surrogate::SurrogateManager            (8 sites)
```

Concretely:
- `physics` conduction solvers (`five_r1c_solver.rs`, `solver_manager.rs`,
  `method_selector.rs`, `wall_properties.rs`, …) all take `sim::assembly::BuildingAssembly`.
- `sim` cannot compile without `physics::HeatConductionSolver` and friends.
- `validation` drives `sim::engine::ThermalModel`; `sim` calls back into `validation`.

Moving `{physics, validation, ai}` into `fluxion-core` would make `fluxion-core`
depend on `fluxion` (for `sim`) while `fluxion` depends on `fluxion-core` — a
**circular crate dependency**, which Cargo rejects. This is why only `weather`
(the sole acyclic leaf) was moved in Phase 1.

## The real memory hog: `ort`

`ort` (ONNX Runtime) is the single largest contributor to per-mutant compile memory.
It is currently an **unconditional** dependency in the root `Cargo.toml`, yet it is
used by only **three** files, all in `src/ai/`:

```
src/ai/surrogate.rs     # SurrogateManager + ort::session::Session pool
src/ai/neural_field.rs
src/ai/rl_policy.rs
```

The problem: `SurrogateManager` is not a leaf — it is embedded in the core per-timestep
struct `StepParameters { surrogates: SurrogateManager }` (`src/sim/timestep_solver.rs`)
and in `lib.rs`'s `Model`/`BatchOracle`. So `ort`'s types flow through the hottest
simulation paths and cannot simply be deleted.

## Phased plan to reach the < 4 GB target

### Phase 2 — break the `physics ↔ sim` cycle (extract shared domain types)
Move the *shared domain types* that both sides need into `fluxion-core`:
- `sim::assembly::{BuildingAssembly, AssemblyBuilder, ConcreteMaterial, …}`
- `sim::construction::{Construction, ConstructionLayer, …}`
- `sim::multi_node_thermal::{MultiNodeThermalMass, ThermalMassNode}`
- `sim::per_surface_conduction::{PerSurfaceConductionSolver, SurfaceKind}`

After this, `physics` can depend on `fluxion-core` (for the domain types) instead of
on `sim`, breaking the cycle. Then `physics` itself can move into `fluxion-core`.

This mirrors the existing architecture (`HeatConductionSolver` trait in
`physics/solver_trait.rs`) — the domain types are the natural seam.

### Phase 3 — decouple `ai`/`ort` via a trait (the big memory win)
Introduce a `SurrogateProvider` trait in `fluxion-core` (or a tiny `fluxion-traits`
crate) and make `StepParameters.surrogates: Box<dyn SurrogateProvider>`. The concrete
`ort`-backed `SurrogateManager` then lives in `fluxion-core` (compiled once), and the
main `fluxion` crate no longer mentions `ort` types at all. Gate `ort` behind a
non-default feature (`ai-inference`). Then `cargo mutants -p fluxion
--no-default-features` compiles **without `ort`** → per-mutant memory drops to the
target range.

### Phase 4 — break `validation ↔ sim`
`validation` is a consumer of `sim`; once `sim` is stable it can either join
`fluxion-core` or become its own `fluxion-validation` crate that depends on both.
Lower priority — validation is not on the per-mutant hot path once `ort` is gated.

## Verifying the current state

```bash
# Workspace builds:
cargo build                                  # builds fluxion (+ cached fluxion-core)

# fluxion-core compiles standalone:
cargo build -p fluxion-core

# Mutation testing targets only fluxion (fluxion-core is a cached dep).
# Always pass --config .cargo/mutants.toml so the canonical config is loaded
# explicitly (issue #1440 — the root-level mutants.toml has been removed):

# Full suite (requires 32 GB+):
cargo mutants --config .cargo/mutants.toml -p fluxion --baseline skip

# Diff-scoped (mirrors the PR CI check — any machine):
scripts/mutants_diff_files.sh origin/develop mutants_diff.patch
cargo mutants --config .cargo/mutants.toml -p fluxion \
  --baseline skip --in-diff mutants_diff.patch
```

## Summary

| 1. Workspace + move `weather` to `fluxion-core` | ✅ Done (#1255) | Establishes the seam; `weather` no longer recompiled per mutant |
| 2. Move shared domain types → `fluxion-core`, then `physics` | ✅ Done (#1349) | Removes physics type hierarchy from per-mutant compile |
| 3. `SurrogateProvider` trait + gate `ort` | ⏳ Tracked epic | **Removes `ort` from per-mutant compile — reaches < 4 GB target** |
| 4. Move `validation` | ⏳ Tracked epic | Completeness |

## Dual-Pipeline CI Strategy (#1891)

While Phases 3 & 4 remain as medium-term architectural fixes, Issue #1891
researched and selected a hybrid strategy to restore CI mutation coverage
**immediately** without the large refactor effort:

### Pipeline 1 — Diff-Scoped Advisory PR Check (`mutation-testing.yml`)

Runs on every PR that touches `src/**`. Uses `cargo mutants --in-diff` to
generate mutants **only in the changed lines** of the PR diff (produced by
`scripts/mutants_diff_files.sh`). Because only a handful of mutants are
generated per PR, the run completes in minutes on a standard 32 GB runner
(`ubuntu-latest-8-cores`).

**Advisory** (non-blocking): the job never fails on missed mutants. Results are
posted as a PR comment and uploaded as an artifact. The advisory status will be
revisited once flake rate is characterised (see Issue #1891, Section 6).

### Pipeline 2 — Nightly Full Suite (`mutation-nightly.yml`)

Runs the entire mutation suite (this config, no `--in-diff`) against `develop`
at 07:00 UTC on a 32 GB runner. Catches indirect mutations that diff-scoping
misses (e.g. a trait change rippling into an untouched solver). Prefers
self-hosted Hetzner runners when `vars.FLUXION_LINUX_RUNNER` is set.

### Why both

Diff-scoping is fast and cheap but can miss indirect mutations. The nightly full
run closes that gap at the cost of delayed feedback (overnight). Phase 3 (gate
`ort`) remains the durable root-cause fix: once per-mutant memory drops below
4 GB, the full suite can run on cheaper runners and eventually on PRs too.
