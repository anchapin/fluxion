//! HVAC bindings: the [`PyHVACSystem`] snapshot pyclass and the
//! model<->HVAC-snapshot helpers.
//!
//! Extracted verbatim from the former monolithic
//! `src/python/model_bindings.rs` (Issue #1812). [`PyHVACSystem`] mirrors
//! [`crate::validation::ashrae140::HVACSystem`] with the most commonly-used
//! ASHRAE 140 fields exposed; [`hvac_system_from_model`] /
//! [`apply_hvac_system_to_model`] convert between the snapshot and the
//! model's HVAC storage. All items are re-exported through the
//! [`crate::python::model_bindings`] facade so existing paths are unchanged.

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::validation::ashrae140::HVACSystem;
use pyo3::prelude::*;
// =============================================================================
// HVACSystem (snapshot of validation::ashrae140::HVACSystem)
// =============================================================================

/// HVAC system configuration snapshot.
///
/// Mirrors [`crate::validation::ashrae140::HVACSystem`] (the validation-layer
/// HVAC type) with the most commonly-used ASHRAE 140 fields exposed. Used by
/// Measures that want to inspect or tweak the heating / cooling plant.
#[pyclass(name = "HVACSystem", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHVACSystem {
    /// Heating capacity (W).
    #[pyo3(get, set)]
    pub heating_capacity: f64,
    /// Cooling capacity (W).
    #[pyo3(get, set)]
    pub cooling_capacity: f64,
    /// Heating coefficient of performance (W_th/W_e).
    #[pyo3(get, set)]
    pub cop_heating: f64,
    /// Cooling coefficient of performance (W_th/W_e).
    #[pyo3(get, set)]
    pub cop_cooling: f64,
    /// Number of stages (1 = single-stage, 2 = two-stage, etc.).
    #[pyo3(get, set)]
    pub stages: u32,
    /// Minimum outdoor temperature for HVAC operation (°C).
    #[pyo3(get, set)]
    pub min_outdoor_temp: f64,
    /// Maximum outdoor temperature for HVAC operation (°C).
    #[pyo3(get, set)]
    pub max_outdoor_temp: f64,
    /// VAV (variable air volume) enabled flag.
    #[pyo3(get, set)]
    pub vav_enabled: bool,
    /// Economizer enabled flag.
    #[pyo3(get, set)]
    pub economizer_enabled: bool,
    /// Supply air temperature (°C).
    #[pyo3(get, set)]
    pub supply_air_temp: f64,
}

impl From<&HVACSystem> for PyHVACSystem {
    fn from(h: &HVACSystem) -> Self {
        Self {
            heating_capacity: h.heating_capacity,
            cooling_capacity: h.cooling_capacity,
            cop_heating: h.cop_heating,
            cop_cooling: h.cop_cooling,
            stages: h.stages,
            min_outdoor_temp: h.min_outdoor_temp,
            max_outdoor_temp: h.max_outdoor_temp,
            vav_enabled: h.vav_enabled,
            economizer_enabled: h.economizer_enabled,
            supply_air_temp: h.supply_air_temp,
        }
    }
}

#[pymethods]
impl PyHVACSystem {
    /// Create a new HVACSystem with default parameters.
    #[new]
    #[pyo3(signature = (
        heating_capacity=10000.0,
        cooling_capacity=8000.0,
        cop_heating=3.0,
        cop_cooling=3.2,
        stages=1,
        min_outdoor_temp=-10.0,
        max_outdoor_temp=40.0,
        vav_enabled=false,
        economizer_enabled=false,
        supply_air_temp=13.0,
    ))]
    // PyO3 `#[new]` constructors must accept a flat argument list (Python doesn't
    // expose keyword-only args for `__init__` ergonomically), so the 10-arg signature
    // is intentional and required by the bindings API contract.
    #[allow(clippy::too_many_arguments)]
    fn new(
        heating_capacity: f64,
        cooling_capacity: f64,
        cop_heating: f64,
        cop_cooling: f64,
        stages: u32,
        min_outdoor_temp: f64,
        max_outdoor_temp: f64,
        vav_enabled: bool,
        economizer_enabled: bool,
        supply_air_temp: f64,
    ) -> Self {
        Self {
            heating_capacity,
            cooling_capacity,
            cop_heating,
            cop_cooling,
            stages,
            min_outdoor_temp,
            max_outdoor_temp,
            vav_enabled,
            economizer_enabled,
            supply_air_temp,
        }
    }

    /// Returns the steady-state heating electrical input at full load (W_e).
    fn heating_electrical_input(&self) -> f64 {
        if self.cop_heating > 0.0 {
            self.heating_capacity / self.cop_heating
        } else {
            0.0
        }
    }

    /// Returns the steady-state cooling electrical input at full load (W_e).
    fn cooling_electrical_input(&self) -> f64 {
        if self.cop_cooling > 0.0 {
            self.cooling_capacity / self.cop_cooling
        } else {
            0.0
        }
    }

    /// Whether this HVAC system can operate at the given outdoor temperature (°C).
    fn can_operate_at(&self, outdoor_temp: f64) -> bool {
        outdoor_temp >= self.min_outdoor_temp && outdoor_temp <= self.max_outdoor_temp
    }

    fn __repr__(&self) -> String {
        format!(
            "HVACSystem(Q_h={:.0} W, Q_c={:.0} W, COP_h={:.2}, COP_c={:.2}, stages={})",
            self.heating_capacity,
            self.cooling_capacity,
            self.cop_heating,
            self.cop_cooling,
            self.stages
        )
    }
}
/// Build a [`PyHVACSystem`] snapshot from a model's heating / cooling capacity
/// and supply-air temperature.
pub fn hvac_system_from_model(model: &ThermalModel<VectorField>) -> PyHVACSystem {
    PyHVACSystem {
        heating_capacity: model.hvac.hvac_heating_capacity,
        cooling_capacity: model.hvac.hvac_cooling_capacity,
        cop_heating: 3.0,
        cop_cooling: 3.2,
        stages: 1,
        min_outdoor_temp: -10.0,
        max_outdoor_temp: 40.0,
        vav_enabled: false,
        economizer_enabled: false,
        supply_air_temp: 13.0,
    }
}

/// Apply a [`PyHVACSystem`] snapshot's heating/cooling capacity back to the model.
pub fn apply_hvac_system_to_model(model: &mut ThermalModel<VectorField>, hvac: &PyHVACSystem) {
    model.hvac.hvac_heating_capacity = hvac.heating_capacity;
    model.hvac.hvac_cooling_capacity = hvac.cooling_capacity;
}

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    //! Rust-side inline tests for the HVAC bindings (Issue #2532).
    //!
    //! Covers `PyHVACSystem` derived properties (electrical input,
    //! can_operate_at) and the model<->snapshot helpers
    //! (`hvac_system_from_model` / `apply_hvac_system_to_model`).

    use super::*;
    use crate::physics::cta::VectorField;
    use crate::sim::engine::ThermalModel;
    // ========================================================================
    // PyHVACSystem derived properties
    // ========================================================================

    #[test]
    fn hvac_heating_electrical_input_divides_capacity_by_cop() {
        let h = PyHVACSystem::new(
            10_000.0, 8_000.0, 4.0, 4.0, 1, -10.0, 40.0, false, false, 13.0,
        );
        assert!((h.heating_electrical_input() - 2_500.0).abs() < 1e-9);
    }

    #[test]
    fn hvac_cooling_electrical_input_divides_capacity_by_cop() {
        let h = PyHVACSystem::new(
            10_000.0, 8_000.0, 4.0, 4.0, 1, -10.0, 40.0, false, false, 13.0,
        );
        assert!((h.cooling_electrical_input() - 2_000.0).abs() < 1e-9);
    }

    #[test]
    fn hvac_electrical_input_zero_when_cop_is_zero() {
        let h = PyHVACSystem::new(
            10_000.0, 8_000.0, 0.0, 0.0, 1, -10.0, 40.0, false, false, 13.0,
        );
        assert_eq!(h.heating_electrical_input(), 0.0);
        assert_eq!(h.cooling_electrical_input(), 0.0);
    }

    #[test]
    fn hvac_can_operate_at_respects_min_max_bounds() {
        let h = PyHVACSystem::new(
            10_000.0, 8_000.0, 3.0, 3.2, 1, -10.0, 40.0, false, false, 13.0,
        );
        assert!(!h.can_operate_at(-15.0), "below min");
        assert!(h.can_operate_at(-10.0), "exactly min (inclusive)");
        assert!(h.can_operate_at(20.0), "mid-range");
        assert!(h.can_operate_at(40.0), "exactly max (inclusive)");
        assert!(!h.can_operate_at(45.0), "above max");
    }

    #[test]
    fn hvac_from_model_round_trip_through_apply() {
        // hvac_system_from_model snapshots capacities; apply_hvac_system_to_model
        // writes them back. Round-tripping should preserve the new values.
        let mut model = ThermalModel::<VectorField>::new(1);
        model.hvac.hvac_heating_capacity = 12_345.0;
        model.hvac.hvac_cooling_capacity = 6_789.0;
        let snap = hvac_system_from_model(&model);
        assert_eq!(snap.heating_capacity, 12_345.0);
        assert_eq!(snap.cooling_capacity, 6_789.0);

        // Apply modified values back.
        let mut updated = snap.clone();
        updated.heating_capacity = 99_999.0;
        updated.cooling_capacity = 88_888.0;
        apply_hvac_system_to_model(&mut model, &updated);
        assert_eq!(model.hvac.hvac_heating_capacity, 99_999.0);
        assert_eq!(model.hvac.hvac_cooling_capacity, 88_888.0);
    }

    #[test]
    fn hvac_from_model_uses_default_cop_and_stages() {
        // hvac_system_from_model hard-codes the default COPs / stages / temp
        // limits when constructing a snapshot — those should match the
        // PyHVACSystem::new defaults so Python sees consistent values.
        let model = ThermalModel::<VectorField>::new(1);
        let snap = hvac_system_from_model(&model);
        assert_eq!(snap.cop_heating, 3.0);
        assert_eq!(snap.cop_cooling, 3.2);
        assert_eq!(snap.stages, 1);
        assert_eq!(snap.min_outdoor_temp, -10.0);
        assert_eq!(snap.max_outdoor_temp, 40.0);
        assert!(!snap.vav_enabled);
        assert!(!snap.economizer_enabled);
        assert_eq!(snap.supply_air_temp, 13.0);
    }
}
