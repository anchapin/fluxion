//! Small, dependency-light helpers shared across the root crate.
//!
//! Each submodule here is intentionally narrow: a single helper or a small
//! family of related helpers. Anything that grows beyond that should be
//! promoted to a proper module under `src/` (e.g. `src/io/`, `src/sim/`).
//!
//! # Why a `util` module?
//!
//! PR #2960 introduced a duplicated 6-line `sha256_hex` helper in six files
//! after the `sha2` 0.11 bump. Consolidating them here keeps the helper
//! definition in a single place while preserving the `fluxion-core` leaf
//! invariant (see `ARCHITECTURE.md` §"Cycle break"): the in-tree copy of
//! `sha256_hex` that lives in `fluxion-core/src/weather/tmy3.rs` cannot be
//! replaced by a `crate::util::sha256_hex` import, because `fluxion-core` is
//! a leaf and the root crate is not.
//!
//! ## Modules
//!
//! - [`sha256_hex`] — Format the bytes of a SHA-256 (or any
//!   `AsRef<[u8]>`) digest as a lowercase hex string. Replaces the
//!   `format!("{:x}", Sha256::digest(...))` pattern that the `sha2`
//!   0.11 / `generic-array` new release made unavailable.

pub mod sha256_hex;
