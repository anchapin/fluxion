//! ASHRAE 140 §8 test target.
//!
//! Issue #4270 (foundation slice): the steady-state analytical verification
//! cases for the ground-coupled slab-on-grade domain model live in
//! `ground_slab_steady_state.rs`.
//!
//! Run with:
//!
//! ```text
//! cargo test --test ashrae_140 ground_slab
//! ```

mod ground_slab_steady_state;
