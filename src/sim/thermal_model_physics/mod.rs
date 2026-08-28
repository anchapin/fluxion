//! Thermal model physics module — split into focused submodules.
//!
//! This module hosts the `impl<T: ...> ThermalModel<T> { ... }` blocks for
//! the thermal physics step solvers, split across files for navigation and
//! review. All methods are merged into the single `ThermalModel<T>` type
//! because Rust unifies `impl` blocks defined in the same crate.
//!
//! ## Submodule Layout
//!
//! | File | Responsibility | Methods on `ThermalModel<T>` |
//! |------|----------------|-------------------------------|
//! | [`hvac`] | HVAC demand calculation using building heat transfer conductance | `compute_zone_hvac_load` |
//! | [`batched_solver`] | Batched coordinator–worker timestep solver | `solve_timesteps_batched` |
//! | [`solver_core`] | Top-level annual solver, timestep sizing, load accessors | `solve_timesteps`, `solve_timesteps_with_dt`, `calculate_timestep_seconds`, `estimate_time_constant_hours`, `get_temperatures`, `get_hourly_temperatures`, `calculate_analytical_loads`, `set_loads`, `set_weather` |
//! | [`step_dispatcher`] | Dispatcher routing to the correct physics model | `step_physics` |
//! | [`physics_impl`] | Physics step implementations (5R1C/6R2C/8R3C/9R4C) via submodules | `step_physics_5r1c`, `step_physics_6r2c`, `step_physics_8r3c`, `step_physics_9r4c` |
//!
//! ### physics_impl submodules (Issue #3220)
//!
//! | File | Responsibility |
//! |------|----------------|
//! | [`physics_impl::step_common`] | Shared helpers like `step_wall_surface_ode` |
//! | [`physics_impl::step_5r1c`] | 5R1C single thermal mass node model |
//! | [`physics_impl::step_6r2c`] | 6R2C two thermal mass node model |
//! | [`physics_impl::step_8r3c`] | 8R3C three thermal mass node model |
//! | [`physics_impl::step_9r4c`] | 9R4C four thermal mass node model |
//!
//! ## Public API
//!
//! The submodule `mod` declarations are intentionally private. The methods
//! they define are part of `ThermalModel<T>`, which itself is `pub` in
//! `sim::thermal_model_core`, so external callers access them as
//! `model.step_physics(...)`, `model.solve_timesteps(...)`, etc. — the
//! module path does not need to leak the internal split.
//!
//! ## Background
//!
//! The monolithic `src/sim/thermal_model_physics.rs` file (~2956 lines)
//! could not host a sibling directory of the same name, blocking Issue
//! #898's modular-split work. Splitting it into the
//! `thermal_model_physics/` directory (and removing the file) completes
//! the modular split without changing the public API (see Issue #902).
//!
//! Issue #3220 further splits `physics_impl.rs` (4370 lines) into focused
//! modules per physics variant, sharing code through a clean trait boundary.

mod batched_solver;
mod hvac;
mod physics_impl;
mod solver_core;
mod step_dispatcher;
