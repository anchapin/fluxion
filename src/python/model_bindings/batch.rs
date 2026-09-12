//! Batch-evaluation bindings for [`Model`](super::model::Model).
//!
//! Extracted verbatim from the former monolithic
//! `src/python/model_bindings.rs`. This module holds the batch-evaluation /
//! optimisation entrypoints:
//!
//! - [`get_parameter_bounds`] — the [`ParameterBounds`] valid ranges used by
//!   [`BatchOracle`](crate::batch_oracle::BatchOracle),
//! - [`validate_parameters`] — physical-constraint validation of a parameter
//!   vector via `BatchOracle::validate_parameters`.
//!
//! These are `pub(crate)` free functions called by the one-line
//! `#[pymethods]` shims on [`Model`](super::model::Model) in
//! [`super::model`]. The shims must stay with the class: PyO3 0.29 rejects a
//! second `#[pymethods] impl Model` block in another module with E0119
//! (conflicting `PyMethods<Model>` impls), so the Python-visible methods —
//! names, signatures, and docstrings — are unchanged and only the logic
//! lives here.

use crate::batch_oracle::BatchOracle;
use crate::python::batch_oracle_bindings::ParameterBounds;
use pyo3::prelude::*;

/// Valid ranges for the building design variables used by [`BatchOracle`](crate::batch_oracle::BatchOracle).
///
/// Called by `Model.get_parameter_bounds`.
pub(crate) fn get_parameter_bounds() -> ParameterBounds {
    ParameterBounds::get_bounds()
}

/// Validate a parameter vector against physical constraints.
///
/// Called by `Model.validate_parameters_py`. Raises `ValidationError` (via
/// the `FluxionError` -> Python exception mapping) when a value is NaN,
/// infinite, or out of range.
pub(crate) fn validate_parameters(params: &[f64]) -> PyResult<()> {
    BatchOracle::validate_parameters(params)?;
    Ok(())
}

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    //! BatchOracle-contract regression test (Issue #2826 companion).
    //!
    //! `apply_parameters` must broadcast scalar setpoints to the per-zone
    //! vectors so the optimisation loop can steer the simulation; two
    //! scalar-driven runs with different setpoints must produce different
    //! energies.

    use crate::physics::cta::VectorField;
    use crate::python::model_bindings::populate_default_model_physics;
    use crate::sim::engine::ThermalModel;

    #[test]
    fn apply_parameters_scalar_broadcasts_to_per_zone_vectors_issue_2826() {
        // Companion regression: BatchOracle's `apply_parameters` historically
        // touched only the scalar setpoint fields. After the per-zone refactor,
        // `apply_parameters` must broadcast scalar → per-zone vector so the
        // optimisation loop can actually steer the simulation. This test
        // confirms that two scalar-driven runs with different setpoints
        // produce different energies (the BatchOracle contract).
        use crate::ai::surrogate::SurrogateManager;
        use crate::sim::lighting::LightingSchedule;

        let mut energies: Vec<f64> = Vec::with_capacity(2);
        for &(heat, cool) in &[(20.0, 27.0), (15.0, 30.0)] {
            let mut model = ThermalModel::<VectorField>::new(1);
            populate_default_model_physics(&mut model);
            model.apply_parameters(&[model.solar.window_u_value, heat, cool]);
            let surrogates = SurrogateManager::new().expect("SurrogateManager");
            model.reset_heating_cooling_energy();
            let zone_area = model
                .setpoints
                .zone_area
                .as_slice()
                .iter()
                .sum::<f64>()
                .max(1.0);
            let empty_lighting = LightingSchedule::new(0.0, zone_area);
            let _eui = model.solve_timesteps(
                24 * 30,
                &surrogates,
                false,
                Some(&empty_lighting),
                None,
                None,
            );
            let total = model.get_heating_energy_kwh() + model.get_cooling_energy_kwh();
            energies.push(total);
        }

        let lo = energies[0].min(energies[1]);
        let hi = energies[0].max(energies[1]);
        let tol = (lo.abs() * 5e-3).max(1e-3);
        assert!(
            (hi - lo) > tol,
            "scalar broadcast regression: apply_parameters should drive \
             different setpoints into different energies; got {:?} (tol = {tol})",
            energies
        );
    }
}
