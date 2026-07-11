//! Timestep solver module
//!
//! Timestep loop, convergence, and result accumulation.

use crate::ai::surrogate::SurrogateManager;
use crate::sim::equipment::Equipment;
use crate::sim::lighting::LightingSchedule;
use crate::sim::occupancy::OccupancyProfile;

/// Per-timestep parameters passed by `&` into the inner physics step.
///
/// # Threading note (Issue #1437)
///
/// `equipment: Option<Vec<Box<dyn Equipment>>>` makes this type `!Send + !Sync`
/// because trait objects do not auto-implement the marker traits. This means a
/// single `&StepParameters` cannot be shared across rayon workers without
/// restructuring the equipment trait object (e.g. `Arc<dyn Equipment + Send +
/// Sync>`). Per-worker construction via [`StepParameters::build_analytical`]
/// is the current-correct hoist: each worker clones the cheap surrogate
/// reference once and reuses the same owned value for every one of the 8 760
/// hourly iterations on its work-item.
pub struct StepParameters {
    pub use_ai: bool,
    pub surrogates: SurrogateManager,
    pub use_analytical_gains: bool,
    pub lighting: Option<LightingSchedule>,
    pub equipment: Option<Vec<Box<dyn Equipment>>>,
    pub occupancy: Option<OccupancyProfile>,
}

impl Default for StepParameters {
    fn default() -> Self {
        Self {
            use_ai: false,
            surrogates: SurrogateManager::new().expect("Failed to create default SurrogateManager"),
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
            surrogates: SurrogateManager::new().expect("Failed to create SurrogateManager"),
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
    /// `surrogates` is cloned from the borrowed reference so the returned value
    /// is fully owned and can cross rayon worker boundaries by move. The
    /// helper exists primarily so that hot-loop call sites can name the
    /// intent once and reuse the same `StepParameters` for every timestep
    /// rather than rebuilding it inside the 8 760-iteration inner loop.
    pub fn build_analytical(surrogates: &SurrogateManager) -> Self {
        Self {
            use_ai: false,
            surrogates: surrogates.clone(),
            use_analytical_gains: true,
            lighting: None,
            equipment: None,
            occupancy: None,
        }
    }
}
