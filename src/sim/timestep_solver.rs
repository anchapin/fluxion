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
    }
}
