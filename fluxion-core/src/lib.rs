//! # fluxion-core
//!
//! Foundational, dependency-light modules shared across the Fluxion workspace.
//!
//! Split out from the main [`fluxion`] crate as part of
//! [issue #1255](https://github.com/anchapin/fluxion/issues/1255) ("Crate split to
//! enable cargo-mutants in CI").
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
//! `pub use fluxion_core::weather;`, so existing `crate::weather::...` and
//! `fluxion::weather::...` paths are **unchanged** — no call-site edits required.
//!
//! ## Current contents
//!
//! | Module   | Status | Notes |
//! |----------|--------|-------|
//! | `weather` | Moved (true leaf) | EPW/TMY3 parsing, psychrometrics, design-day, interpolation |
//!
//! ## Not yet moved (blocked by dependency cycles)
//!
//! `ai`, `physics`, and `validation` are coupled **bidirectionally** with `sim`
//! (e.g. `physics` uses `sim::assembly::BuildingAssembly`; `sim` uses
//! `physics::HeatConductionSolver`). Moving them verbatim would create a circular
//! crate dependency. See `docs/mutation_testing_crate_split.md` for the phased
//! cycle-breaking plan.

// Match the main `fluxion` crate's relaxed lint posture so leaf modules compile
// without the historical style warnings they carried before the split.
#![allow(nonstandard_style)]
#![allow(clippy::all)]

pub mod weather;
