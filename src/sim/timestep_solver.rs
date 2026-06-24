//! Timestep solver module
//!
//! Timestep loop, convergence, and result accumulation.

use crate::ai::{SurrogateManager, SurrogateOpsBox};
use crate::sim::equipment::Equipment;
use crate::sim::lighting::LightingSchedule;
use crate::sim::occupancy::OccupancyProfile;

pub struct StepParameters {
    pub use_ai: bool,
    pub surrogates: SurrogateOpsBox,
    pub use_analytical_gains: bool,
    pub lighting: Option<LightingSchedule>,
    pub equipment: Option<Vec<Box<dyn Equipment>>>,
    pub occupancy: Option<OccupancyProfile>,
}

impl Default for StepParameters {
    fn default() -> Self {
        Self {
            use_ai: false,
            surrogates: SurrogateOpsBox::new(SurrogateManager::new().expect("Failed to create default SurrogateManager")),
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
            surrogates: SurrogateOpsBox::new(SurrogateManager::new().expect("Failed to create SurrogateManager")),
            use_analytical_gains: false,
            lighting: None,
            equipment: None,
            occupancy: None,
        }
    }
}
