//! 5R1C/6R2C/8R3C/9R4C physics step implementations for `ThermalModel`.
//!
//! Issue #3220: This file has been refactored to extract each physics step
//! variant into its own module under `src/sim/thermal_model_physics/`:
//! - [`step_5r1c`] — 5R1C single mass node implementation (~1700 lines)
//! - [`step_6r2c`] — 6R2C two mass node implementation (~833 lines)
//! - [`step_8r3c`] — 8R3C three mass node implementation (~102 lines)
//! - [`step_9r4c`] — 9R4C four mass node implementation (~1486 lines)
//! - [`step_common`] — Shared helpers like `step_wall_surface_ode`
//!
//! Originally part of the monolithic `physics_impl.rs` (4370 lines),
//! further split in Issue #3220 to address single-responsibility principle violations.

#[allow(unused_imports)]
use crate::physics::cta::VectorField;

mod step_5r1c;
mod step_6r2c;
mod step_8r3c;
mod step_9r4c;
mod step_common;

#[cfg(test)]
mod scratch_pool_tests {
    //! Issue #2756 — prove `PhysicsScratchPool` is live (no longer dead code)
    //! and that its buffers are reused across timesteps rather than
    //! re-allocated per step.
    //!
    //! These tests live in-crate (not under `tests/`) because they must read
    //! `model.0.hvac.scratch_pool` — a `pub(crate)` field — to assert reuse. The
    //! dhat gate under `tests/dhat_step_physics_zero_alloc.rs` provides the
    //! independent heap-growth measurement.

    use crate::sim::construction::WallSurface;
    use crate::sim::solar::WindowProperties;
    use crate::sim::thermal_model_core::ThermalModel;
    use crate::weather::HourlyWeatherData;
    use fluxion_core::ashrae_cases::Orientation;
    // `VectorField` is already imported by the outer module (line 16). Reuse it
    // via `super::` rather than a fresh `use crate::physics::cta::VectorField` —
    // that would add a new sim→physics edge and trip the Physics-Sim-Cycle-Check
    // gate (#2463 / scripts/check_physics_sim_cycle.py).
    use super::VectorField;

    /// More than 4 zones so every `SmallVec<[f64; 4]>` field SPILLS to heap —
    /// that is the regime where per-step `new(num_zones)` allocation is
    /// observable and where the pool pays off. At ≤4 zones the fields are inline
    /// and the pool is a no-op (still correct, just not measurable by pointer
    /// stability).
    const NUM_ZONES: usize = 10;

    /// Mirrors the `dhat_zone_solar_gain_zero_alloc` fixture: a 10-zone model
    /// each with 5 windowed surfaces so `calc_analytical_loads` →
    /// `calculate_zone_solar_gain` is fully populated and `step_physics` runs the
    /// production 5R1C analytical path without panic.
    fn multizone_model() -> ThermalModel<VectorField> {
        let mut model = ThermalModel::<VectorField>::new(NUM_ZONES);
        model.solar.window_u_value = 1.5;
        model.setpoints.heating_setpoint = 20.0;
        model.setpoints.cooling_setpoint = 26.0;
        model.setpoints.temperatures = VectorField::from_scalar(20.0, NUM_ZONES);
        model.mass.mass_temperatures = VectorField::from_scalar(20.0, NUM_ZONES);
        model.setpoints.zone_area = VectorField::from_scalar(50.0, NUM_ZONES);

        let wp = WindowProperties::double_clear(8.0);
        model.solar.window_properties = vec![wp; NUM_ZONES];

        let surfaces_per_zone: Vec<Vec<WallSurface>> = (0..NUM_ZONES)
            .map(|_| {
                vec![
                    WallSurface::new(10.0, 0.5, Orientation::North).with_window(2.0),
                    WallSurface::new(10.0, 0.5, Orientation::East).with_window(2.0),
                    WallSurface::new(10.0, 0.5, Orientation::South).with_window(2.0),
                    WallSurface::new(10.0, 0.5, Orientation::West).with_window(2.0),
                    WallSurface::new(25.0, 0.3, Orientation::Up),
                ]
            })
            .collect();
        model.solar.surfaces = surfaces_per_zone;

        model
    }

    fn weather(hour: usize) -> HourlyWeatherData {
        HourlyWeatherData::new(20.0, 400.0, 100.0, 500.0, 2.0, 50.0, hour)
    }

    /// The pool must be populated by every 5R1C step, and the NON-`mem::take`'n
    /// fields (e.g. `wall_surface_new`) must keep the SAME heap buffer across
    /// steps — the direct proof that `checkout`/`fill_zero`/`return` reuses
    /// capacity instead of re-allocating.
    ///
    /// (`phi_ia`/`phi_st`/`phi_m`/`t_i_act`/`new_mass` are emptied each step by
    /// `mem::take` into a `VectorField`, so their pointer is expected to move;
    /// they are the residual per-step allocation tracked separately by the dhat
    /// gate. `wall_surface_new` and `wall_surface_correction` are written by
    /// index and never taken, so the pool reuses their buffer verbatim.)
    #[test]
    fn scratch_pool_5r1c_is_reused_across_timesteps() {
        let mut model = multizone_model();

        // Before any step the pool is empty (lazy).
        assert!(
            model.0.hvac.scratch_pool.r5r1c.is_none(),
            "pool must start empty"
        );

        // First step: checkout allocates, return restores into the pool.
        model.solar.weather = Some(weather(0));
        model.step_physics(0, 20.0, 3600.0);
        let pool = model
            .0
            .hvac
            .scratch_pool
            .r5r1c
            .as_ref()
            .expect("5R1C pool must be populated after step 1 (Issue #2756)");
        let ptr_warm = pool.wall_surface_new.as_ptr();
        let len_warm = pool.wall_surface_new.len();
        assert_eq!(len_warm, NUM_ZONES);

        // Drive 99 more steps — the pool must remain populated and the
        // wall_surface_new buffer must be the SAME allocation (pointer-stable).
        for step in 1..100 {
            model.solar.weather = Some(weather(step % 24));
            model.step_physics(step, 20.0, 3600.0);
        }
        let pool = model
            .0
            .hvac
            .scratch_pool
            .r5r1c
            .as_ref()
            .expect("5R1C pool must still be populated after 100 steps");
        let ptr_steady = pool.wall_surface_new.as_ptr();

        assert_eq!(
            ptr_warm, ptr_steady,
            "wall_surface_new buffer must be reused (same heap allocation) \
             across 100 timesteps — if this fails, the pool stopped reusing \
             the non-taken scratch fields and per-timestep allocation regressed."
        );
        assert_eq!(pool.wall_surface_new.len(), NUM_ZONES);
    }

    /// Same reuse property for the 9R4C pool. Exercises the dispatcher's 9R4C
    /// branch and the early-return path in `step_physics_9r4c` (free_float is
    /// false here, so the early return is not taken; the final return restores
    /// the pool). The `inter` flat buffer (num_zones × 7) is never `mem::take`'n
    /// — only sliced — so its pointer must be stable across steps.
    #[test]
    fn scratch_pool_9r4c_is_reused_across_timesteps() {
        let mut model = multizone_model();
        model.0.hvac.thermal_model_type =
            crate::sim::thermal_model_core::ThermalModelType::NineRFourC;

        assert!(model.0.hvac.scratch_pool.r9r4c.is_none());

        model.solar.weather = Some(weather(0));
        model.step_physics(0, 20.0, 3600.0);
        let inter_ptr_warm = model
            .0
            .hvac
            .scratch_pool
            .r9r4c
            .as_ref()
            .expect("9R4C pool populated after step 1")
            .inter
            .as_ptr();

        for step in 1..50 {
            model.solar.weather = Some(weather(step % 24));
            model.step_physics(step, 20.0, 3600.0);
        }
        let inter_ptr_steady = model
            .0
            .hvac
            .scratch_pool
            .r9r4c
            .as_ref()
            .expect("9R4C pool still populated")
            .inter
            .as_ptr();

        assert_eq!(
            inter_ptr_warm, inter_ptr_steady,
            "9R4C `inter` buffer must be reused across timesteps"
        );
    }

    /// The 9R4C free-float early-return path must ALSO restore the pool, else
    /// the next step re-allocates. Drives a free-float model and asserts the
    /// pool stays populated.
    #[test]
    fn scratch_pool_9r4c_restored_on_free_float_early_return() {
        let mut model = multizone_model();
        model.0.hvac.thermal_model_type =
            crate::sim::thermal_model_core::ThermalModelType::NineRFourC;
        model.0.hvac.free_float = true;

        for step in 0..20 {
            model.solar.weather = Some(weather(step % 24));
            model.step_physics(step, 20.0, 3600.0);
            // After EVERY step (each takes the early return), the pool must be
            // repopulated — otherwise the early return leaked the checkout.
            assert!(
                model.0.hvac.scratch_pool.r9r4c.is_some(),
                "9R4C pool must be restored on the free_float early-return path \
                 (step {step}); the pool leaked and the next step re-allocates."
            );
        }
    }

    /// Determinism: two fresh models with identical config + weather must
    /// produce bit-identical zone temperatures and cumulative HVAC energy over
    /// a multi-step run. The pool must not leak state across timesteps or across
    /// independent model instances — if it did, run A and run B would diverge.
    /// This is the "bit-identical vs. reference" guard for the wiring change.
    #[test]
    fn pooled_step_physics_is_deterministic() {
        fn run() -> (Vec<f64>, f64) {
            let mut model = multizone_model();
            let mut total = 0.0_f64;
            for step in 0..48 {
                model.solar.weather = Some(weather(step % 24));
                total += model.step_physics(step, 20.0, 3600.0);
            }
            (model.setpoints.temperatures.as_ref().to_vec(), total)
        }
        let (temps_a, energy_a) = run();
        let (temps_b, energy_b) = run();

        assert_eq!(temps_a.len(), temps_b.len());
        for (a, b) in temps_a.iter().zip(temps_b.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "zone temperatures diverged between identical runs — the pool \
                 leaked scratch state across timesteps or instances."
            );
        }
        assert_eq!(
            energy_a.to_bits(),
            energy_b.to_bits(),
            "cumulative HVAC energy diverged between identical runs"
        );
    }
}
