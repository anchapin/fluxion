//! Timestep solver module
//!
//! Timestep loop, convergence, and result accumulation.

use crate::ai::surrogate::SurrogateManager;
use crate::sim::equipment::Equipment;
use crate::sim::lighting::LightingSchedule;
use crate::sim::occupancy::OccupancyProfile;
use std::sync::Arc;

/// Per-timestep parameters passed by `&` into the inner physics step.
///
/// # Threading note (Issue #1437)
///
/// `equipment: Option<Vec<Box<dyn Equipment>>>` makes this type `!Send + !Sync`
/// because trait objects do not auto-implement the marker traits. This means a
/// single `&StepParameters` cannot be shared across rayon workers without
/// restructuring the equipment trait object (e.g. `Arc<dyn Equipment + Send +
/// Sync>`). Per-worker construction via [`StepParameters::build_analytical`]
/// skips the surrogate reference entirely on the analytical path.
pub struct StepParameters {
    pub use_ai: bool,
    pub surrogates: Option<Arc<SurrogateManager>>,
    pub use_analytical_gains: bool,
    pub lighting: Option<LightingSchedule>,
    pub equipment: Option<Vec<Box<dyn Equipment>>>,
    pub occupancy: Option<OccupancyProfile>,
}

impl Default for StepParameters {
    fn default() -> Self {
        Self {
            use_ai: false,
            surrogates: None,
            use_analytical_gains: false,
            lighting: None,
            equipment: None,
            occupancy: None,
        }
    }
}

impl StepParameters {
    pub fn clone_for_test(&self) -> Self {
        Self {
            use_ai: self.use_ai,
            surrogates: self.surrogates.clone(),
            use_analytical_gains: self.use_analytical_gains,
            lighting: self.lighting.clone(),
            equipment: None,
            occupancy: self.occupancy.clone(),
        }
    }
}

impl StepParameters {
    pub fn new() -> Self {
        Self {
            use_ai: false,
            surrogates: None,
            use_analytical_gains: false,
            lighting: None,
            equipment: None,
            occupancy: None,
        }
    }

    /// Build a `StepParameters` configured for analytical (non-ML) evaluation.
    ///
    /// `use_ai` is forced to `false`, `use_analytical_gains` to `true`, and
    /// `lighting` / `equipment` / `occupancy` are `None` — matching the
    /// historical behavior of `BatchOracle::evaluate_population` with
    /// `use_surrogates = false` (Issue #901 / #1437).
    ///
    /// No surrogate reference is retained because `use_ai` is false and the
    /// analytical timestep path never invokes surrogate inference.
    pub fn build_analytical() -> Self {
        Self {
            use_ai: false,
            surrogates: None,
            use_analytical_gains: true,
            lighting: None,
            equipment: None,
            occupancy: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StepParameters;

    #[test]
    fn analytical_parameters_do_not_retain_surrogate_manager() {
        let parameters = StepParameters::build_analytical();

        assert!(!parameters.use_ai);
        assert!(parameters.surrogates.is_none());
        assert!(parameters.use_analytical_gains);
        assert!(parameters.lighting.is_none());
        assert!(parameters.equipment.is_none());
        assert!(parameters.occupancy.is_none());
    }

    #[test]
    fn default_parameters_have_all_disabled() {
        let params = StepParameters::default();
        assert!(!params.use_ai);
        assert!(params.surrogates.is_none());
        assert!(!params.use_analytical_gains);
        assert!(params.lighting.is_none());
        assert!(params.equipment.is_none());
        assert!(params.occupancy.is_none());
    }

    #[test]
    fn new_parameters_match_default() {
        let from_new = StepParameters::new();
        let from_default = StepParameters::default();
        assert_eq!(from_new.use_ai, from_default.use_ai);
        assert!(from_new.surrogates.is_none() == from_default.surrogates.is_none());
        assert_eq!(
            from_new.use_analytical_gains,
            from_default.use_analytical_gains
        );
        assert_eq!(from_new.lighting.is_none(), from_default.lighting.is_none());
        assert_eq!(
            from_new.equipment.is_none(),
            from_default.equipment.is_none()
        );
        assert_eq!(
            from_new.occupancy.is_none(),
            from_default.occupancy.is_none()
        );
    }

    #[test]
    fn clone_for_test_preserves_fields_except_equipment() {
        let params = StepParameters {
            use_ai: true,
            surrogates: None,
            use_analytical_gains: true,
            lighting: None,
            equipment: None,
            occupancy: None,
        };
        let cloned = params.clone_for_test();
        assert_eq!(cloned.use_ai, params.use_ai);
        assert_eq!(cloned.use_analytical_gains, params.use_analytical_gains);
        assert_eq!(cloned.lighting.is_none(), params.lighting.is_none());
        assert_eq!(cloned.occupancy.is_none(), params.occupancy.is_none());
    }

    #[test]
    fn clone_for_test_strips_equipment() {
        let params = StepParameters {
            use_ai: false,
            surrogates: None,
            use_analytical_gains: true,
            lighting: None,
            equipment: Some(vec![]),
            occupancy: None,
        };
        let cloned = params.clone_for_test();
        assert!(
            cloned.equipment.is_none(),
            "clone_for_test should always strip equipment"
        );
    }

    #[test]
    fn build_analytical_all_fields_match_expected() {
        let p = StepParameters::build_analytical();
        assert!(!p.use_ai);
        assert!(p.surrogates.is_none());
        assert!(p.use_analytical_gains);
        assert!(p.lighting.is_none());
        assert!(p.equipment.is_none());
        assert!(p.occupancy.is_none());
    }

    #[test]
    fn new_and_default_produce_equivalent_structs() {
        // Both new() and default() should produce the same zero-state
        let from_new = StepParameters::new();
        let from_default = StepParameters::default();
        assert_eq!(from_new.use_ai, from_default.use_ai);
        assert_eq!(from_new.use_analytical_gains, from_default.use_analytical_gains);
        assert!(
            from_new.equipment.is_none() && from_default.equipment.is_none()
        );
    }

    #[test]
    fn step_parameters_with_ai_flag_enabled() {
        let params = StepParameters {
            use_ai: true,
            surrogates: None,
            use_analytical_gains: false,
            lighting: None,
            equipment: None,
            occupancy: None,
        };
        assert!(params.use_ai);
        assert!(!params.use_analytical_gains);
    }
}
