//! Shared helpers for physics step implementations.
//!
//! Contains [`step_wall_surface_ode`], the exponential ODE solver used by
//! the 5R1C, 6R2C, and 9R4C variants to evolve the interior wall-surface
//! temperature `T_si` before the air-node heat balance.

use crate::physics::five_r1c_solver::surface_time_constant_from_conductances;
use crate::sim::thermal_model_scratch::PhysicsScratch5r1c;

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_wall_surface_ode(
    dt: f64,
    h_tr_ms: &[f64],
    h_tr_is: &[f64],
    mass_temps: &[f64],
    zone_temps: &[f64],
    wall_surface_old: &[f64],
    thermal_cap: &[f64],
    scratch: &mut PhysicsScratch5r1c,
) {
    // Size the loop from the input slices, not from a scratch field —
    // by the time the caller invokes this helper, `phi_ia`/`phi_st`/`phi_m`
    // have already been moved out of the scratch via `mem::take`, so any
    // scratch-backed length probe would report zero. The caller guarantees
    // every input slice has the same length (`self.0.hvac.num_zones`).
    let n = wall_surface_old.len();
    debug_assert_eq!(h_tr_ms.len(), n);
    debug_assert_eq!(h_tr_is.len(), n);
    debug_assert_eq!(mass_temps.len(), n);
    debug_assert_eq!(zone_temps.len(), n);
    debug_assert_eq!(thermal_cap.len(), n);
    debug_assert_eq!(scratch.wall_surface_new.len(), n);
    debug_assert_eq!(scratch.wall_surface_correction.len(), n);

    let wall_surface_new = &mut scratch.wall_surface_new;
    let wall_surface_correction = &mut scratch.wall_surface_correction;
    for i in 0..n {
        let h_ms_i = h_tr_ms[i];
        let h_is_i = h_tr_is[i];
        if h_ms_i > 0.0 && h_is_i > 0.0 {
            // τ_si is the same quantity `FiveR1CSolver::surface_time_constant`
            // exposes, just expressed in the per-zone conductance basis the
            // physics consumer already has on hand. Delegates to the shared
            // free function so the formula lives in exactly one place.
            let tau_si = surface_time_constant_from_conductances(thermal_cap[i], h_ms_i, h_is_i);
            let t_m_i = mass_temps[i];
            let t_int_i = zone_temps[i];
            let t_si_eq = (t_int_i * h_is_i + t_m_i * h_ms_i) / (h_is_i + h_ms_i);
            let t_si_old_i = wall_surface_old[i];
            let t_si_new_i = if tau_si > 0.0 && dt > 0.0 {
                t_si_eq + (t_si_old_i - t_si_eq) * (-dt / tau_si).exp()
            } else {
                t_si_eq
            };
            wall_surface_new[i] = t_si_new_i;
            wall_surface_correction[i] = h_is_i * (t_si_new_i - t_si_eq);
        } else {
            wall_surface_new[i] = wall_surface_old[i];
            wall_surface_correction[i] = 0.0;
        }
    }
}
