//! Thermal model 5R1C implementation.
//!
//! ISO 13790-compliant 5R1C thermal network implementation.
//! Contains the step_physics_5r1c method for single-zone thermal modeling.
//!
//! ## Module Responsibilities
//! - 5R1C model physics (surface-to-mass-to-air thermal network)
//! - Free-floating temperature calculation
//! - HVAC demand calculation based on thermal model
//! - CTF/FD envelope conduction integration
//!
//! ## Current Status
//! The 5R1C-specific physics are currently contained in `thermal_model_physics.rs`
//! within the `step_physics_5r1c` method. This module is a marker for future
//! extraction when the data architecture supports cleaner separation.
//!
//! ## Design Considerations for Future Extraction
//! 1. The step_physics_5r1c function directly accesses ThermalModel internal state
//! 2. The phi_m, phi_s, phi_st gain calculations are tightly coupled to the model
//! 3. Network conductance values (h_tr_is, h_tr_ms, h_tr_em) are pre-computed
//! 4. Moving these would require significant refactoring of thermal_model_core
