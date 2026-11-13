//! ASHRAE 140 §8 test target.
//!
//! Issue #4270 (foundation slice): the steady-state analytical verification
//! cases for the ground-coupled slab-on-grade domain model live in
//! `ground_slab_steady_state.rs`.
//!
//! Issue #4774: the transient half lives in `ground_transient.rs` and the
//! periodic half in `ground_periodic.rs`.
//!
//! Run with:
//!
//! ```text
//! cargo test --test ashrae_140 ground_slab
//! cargo test --test ashrae_140 ground_periodic
//! ```

mod ground_periodic;
mod ground_slab_steady_state;
mod ground_transient;
