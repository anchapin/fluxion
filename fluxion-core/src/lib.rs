//! # fluxion-core
//!
//! Foundational, dependency-light modules shared across the Fluxion workspace.
//!
//! Split out from the main [`fluxion`] crate as part of
//! [issue #1255](https://github.com/anchapin/fluxion/issues/1255) ("Crate split to
//! enable cargo-mutants in CI") and extended in
//! [issue #1349](https://github.com/anchapin/fluxion/issues/1349) (Phase 2: break
//! the physics<->sim dependency cycle).
//!
//! ## Why a separate crate?
//!
//! `cargo-mutants` rebuilds the *target* crate for every generated mutant. When the
//! entire engine lived in a single crate, each mutant had to recompile the heavy AI
//! (`ort`/ONNX) and physics type hierarchies, requiring ~28 GB of RAM and making
//! mutation testing impossible in CI.
//!
//! By moving the genuinely *leaf* modules — those with **no** dependencies on the rest
//! of the engine (`sim`, `physics` solvers, `ai`, `validation`) — into `fluxion-core`,
//! they are compiled **once** and cached, while cargo-mutants mutates only the main
//! `fluxion` crate (`cargo mutants -p fluxion`).
//!
//! ## Re-export shim
//!
//! The main `fluxion` crate re-exports these modules, e.g.
//! `pub use fluxion_core::assembly;`, so existing `crate::assembly::...`,
//! `crate::sim::assembly::...`, and `fluxion::sim::assembly::...` paths are
//! **unchanged** — no call-site edits required.
//!
//! ## Current contents
//!
//! | Module        | Status | Notes |
//! |---------------|--------|-------|
//! | `weather`     | Moved (#1255, true leaf) | EPW/TMY3 parsing, psychrometrics, design-day, interpolation |
//! | `assembly`    | Moved (#1349) | `BuildingAssembly`, `AssemblyBuilder`, `MaterialLayer` trait, ASHRAE 140 material constants (inlined) |
//! | `construction`| Moved (#2462) | `ConstructionLayer`, `Construction`, `MassClass`, `Materials`, `Assemblies`, `SurfaceType`, ASHRAE 140 film/air constants (inlined). Breaks 3 of 5 `physics ↔ sim` cycle edges. |
//! | `multi_node`  | Moved (#1349) | `ThermalMassNode`, `MultiNodeThermalMass`, `MultiNodeModelType`, `MassAirCouplingMode` (pure data, zero deps) |
//! | `per_surface_conduction` | Moved (#2462) | `SurfaceKind`, `MassNode`, `SurfaceNode`, `PerSurfaceConductionSolver`. Breaks the remaining 2 `physics ↔ sim` cycle edges. |
//! | `physics_constants` | Moved (#2462) | `STEFAN_BOLTZMANN`. Hoisted out of `sim::sky_radiation` so `physics::multi_node_solver` no longer imports from `sim`. |
//! | `ashrae_cases`| Moved (#1441) | `Orientation`, `WindowArea`, `ConstructionType`, `ShadingType`, `ShadingDevice`, `GlassType`, `WindowSpec`, `InternalLoads`, `HvacSchedule`, `NightVentilation`, `BuildingType`, `GeometrySpec`, `ConductanceReferences` — pure-data leaf types from `validation::ashrae_140_cases`. Breaks the `sim ↔ validation` cycle (5 sim callers + 3 indirect sim callers). |
//!
//! ## Cycle break (#1349)
//!
//! After #1349 the `fluxion::physics -> fluxion::sim::assembly` edge is broken:
//!
//! ```text
//! fluxion::physics  ───> fluxion_core::assembly     (was: -> fluxion::sim::assembly)
//! fluxion::sim      ───> fluxion_core::assembly     (was: -> fluxion::sim::assembly, now self-loop via re-export shim)
//! fluxion::sim      ───> fluxion_core::multi_node   (was: -> fluxion::sim::multi_node_thermal, now self-loop via re-export shim)
//! ```
//!
//! `fluxion::sim::assembly::*` and `fluxion::sim::multi_node_thermal::*` remain
//! valid paths via thin re-export shims at `src/sim/assembly.rs` and
//! `src/sim/multi_node_thermal.rs`, so no downstream call-site edits are required.
//!
//! ## Not yet moved (blocked by remaining dependency edges)
//!
//! - `fluxion::physics::{wall_spec, wall_properties, method_selector}` reference
//!   `fluxion::physics::{ctf_coefficients, fd_discretization}` (heavy solver types).
//!   Moving the whole physics tree requires breaking these intra-crate edges first.
//!
//! ## Cycle break (#2462 — physics ↔ sim shared domain types → `fluxion-core`)
//!
//! Issue #2462 broke the last `physics ↔ sim` cycle (see ARCHITECTURE.md
//! §"Remaining cycles" and `docs/mutation_testing_crate_split.md` §"Phase 2")
//! by hoisting `ConstructionLayer` (+ `Construction`, `MassClass`, `Materials`,
//! `Assemblies`, `SurfaceType`, ASHRAE 140 film/air constants) and
//! `PerSurfaceConductionSolver` (+ `SurfaceKind`, `MassNode`, `SurfaceNode`)
//! into `fluxion_core`, plus lifting `STEFAN_BOLTZMANN` out of `sim::sky_radiation`
//! into the new `fluxion_core::physics_constants` leaf. `scripts/check_physics_sim_cycle.py`
//! baseline drops from 5+2 edges to 0; the gate is wired into CI as the
//! `Physics-Sim-Cycle-Check` listener (see release_gates.yaml).
//!
//! The `sim::construction ↔ validation::ashrae_140_cases::Orientation` cycle was
//! closed in #1441 by moving the leaf types into `fluxion_core::ashrae_cases`.
//!
//! Any doc-comment in this file that names a current cycle target is
//! diff-checked against the cycle baselines by `scripts/check_doc_drift.py`
//! (issue #2895); see ARCHITECTURE.md §"Remaining cycles" for the
//! source-of-truth list.

// Match the main `fluxion` crate's relaxed lint posture so leaf modules compile
// without the historical style warnings they carried before the split.
#![allow(nonstandard_style)]
#![allow(clippy::all)]

pub mod ashrae_cases;
pub mod assembly;
pub mod construction;
pub mod earth_tube;
pub mod fluid;
pub mod multi_node;
/// Parser size/depth/repetition limits — DoS hardening (issue #2527).
/// Dependency-light leaf; must not import sim/physics/ai/validation.
pub mod parser_limits;
pub mod per_surface_conduction;
pub mod physics_constants;
pub mod tensor;
pub mod urban_radiation;
pub mod weather;
