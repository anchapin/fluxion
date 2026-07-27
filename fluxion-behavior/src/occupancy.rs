use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OccupancyType {
    Unoccupied,
    Occupied,
    Standby,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupancySchedule {
    pub schedule: Vec<OccupancyType>,
    pub timestep_minutes: u32,
}

impl OccupancySchedule {
    pub fn new(schedule: Vec<OccupancyType>, timestep_minutes: u32) -> Self {
        Self {
            schedule,
            timestep_minutes,
        }
    }

    pub fn get_occupancy(&self, timestep: usize) -> OccupancyType {
        self.schedule[timestep % self.schedule.len()]
    }

    pub fn is_occupied(&self, timestep: usize) -> bool {
        matches!(self.get_occupancy(timestep), OccupancyType::Occupied)
    }

    pub fn occupant_count(&self, timestep: usize, base_count: f64) -> f64 {
        match self.get_occupancy(timestep) {
            OccupancyType::Occupied => base_count,
            OccupancyType::Standby => base_count * 0.3,
            OccupancyType::Unoccupied => 0.0,
        }
    }
}

impl Default for OccupancyType {
    fn default() -> Self {
        OccupancyType::Unoccupied
    }
}
