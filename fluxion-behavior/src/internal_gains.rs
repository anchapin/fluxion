use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetabolicRate {
    Sleeping,
    SeatedQuiet,
    OfficeWork,
    LightActivity,
    Standing,
    Walking,
}

impl MetabolicRate {
    pub fn watts_per_kg(&self) -> f64 {
        match self {
            MetabolicRate::Sleeping => 1.0,
            MetabolicRate::SeatedQuiet => 1.2,
            MetabolicRate::OfficeWork => 1.4,
            MetabolicRate::LightActivity => 2.0,
            MetabolicRate::Standing => 1.7,
            MetabolicRate::Walking => 2.8,
        }
    }

    pub fn met(&self) -> f64 {
        self.watts_per_kg()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Co2Generation {
    pub co2_generation_rate_per_person: f64,
}

impl Default for Co2Generation {
    fn default() -> Self {
        Self::new()
    }
}

impl Co2Generation {
    pub fn new() -> Self {
        Self {
            co2_generation_rate_per_person: 0.005,
        }
    }

    pub fn with_rate(mut self, rate: f64) -> Self {
        self.co2_generation_rate_per_person = rate;
        self
    }

    pub fn calculate_co2_generation(
        &self,
        occupant_count: f64,
        metabolic_rate: MetabolicRate,
    ) -> f64 {
        let activity_factor = match metabolic_rate {
            MetabolicRate::Sleeping => 0.8,
            MetabolicRate::SeatedQuiet => 1.0,
            MetabolicRate::OfficeWork => 1.2,
            MetabolicRate::LightActivity => 1.6,
            MetabolicRate::Standing => 1.4,
            MetabolicRate::Walking => 2.0,
        };
        occupant_count * self.co2_generation_rate_per_person * activity_factor
    }
}

impl Default for MetabolicRate {
    fn default() -> Self {
        MetabolicRate::SeatedQuiet
    }
}

use chrono::{DateTime, Datelike, Timelike, Utc};
use std::sync::Arc;

use crate::lighting::{LightingModel, OccupantState};

pub trait OccupancyProvider: Send + Sync {
    fn occupant_state(&self, t: DateTime<Utc>) -> OccupantState;
    fn occupant_count(&self, t: DateTime<Utc>) -> f64;
}

pub trait PlugLoadProvider: Send + Sync {
    fn get_plug_load(&self, t: DateTime<Utc>) -> f64;
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct InternalGains {
    pub phi_sensible: f64,
    pub phi_latent: f64,
}

impl InternalGains {
    pub fn new(phi_sensible: f64, phi_latent: f64) -> Self {
        Self {
            phi_sensible,
            phi_latent,
        }
    }

    pub fn zero() -> Self {
        Self::default()
    }
}

pub struct DynamicInternalGainAdapter {
    occupancy: Arc<dyn OccupancyProvider>,
    plug_loads: Arc<dyn PlugLoadProvider>,
    lighting: LightingModel,
    sensible_per_occupant: f64,
    latent_per_occupant: f64,
}

impl DynamicInternalGainAdapter {
    pub fn new(
        occupancy: Arc<dyn OccupancyProvider>,
        plug_loads: Arc<dyn PlugLoadProvider>,
        lighting: LightingModel,
    ) -> Self {
        Self {
            occupancy,
            plug_loads,
            lighting,
            sensible_per_occupant: 70.0,
            latent_per_occupant: 30.0,
        }
    }

    pub fn with_metabolic_rates(
        mut self,
        sensible_per_occupant: f64,
        latent_per_occupant: f64,
    ) -> Self {
        self.sensible_per_occupant = sensible_per_occupant;
        self.latent_per_occupant = latent_per_occupant;
        self
    }

    pub fn compute_gains(&self, _zone_id: uuid::Uuid, t: DateTime<Utc>) -> InternalGains {
        let occupancy_state = self.occupancy.occupant_state(t);
        let n_occupants = self.occupancy.occupant_count(t);
        let plug_w = self.plug_loads.get_plug_load(t);
        let lighting_w = self.lighting.compute(t, occupancy_state);

        let occupant_sensible = n_occupants * self.sensible_per_occupant;
        let occupant_latent = n_occupants * self.latent_per_occupant;

        InternalGains {
            phi_sensible: occupant_sensible + plug_w + lighting_w,
            phi_latent: occupant_latent,
        }
    }
}

impl Default for DynamicInternalGainAdapter {
    fn default() -> Self {
        Self {
            occupancy: Arc::new(ScheduleOccupancyProvider::default()),
            plug_loads: Arc::new(ConstantPlugLoadProvider::default()),
            lighting: LightingModel::default(),
            sensible_per_occupant: 70.0,
            latent_per_occupant: 30.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheduleOccupancyProvider {
    pub hourly_counts: Vec<f64>,
}

impl ScheduleOccupancyProvider {
    pub fn new(hourly_counts: Vec<f64>) -> Self {
        Self { hourly_counts }
    }
}

impl OccupancyProvider for ScheduleOccupancyProvider {
    fn occupant_state(&self, t: DateTime<Utc>) -> OccupantState {
        let hour = t.hour() as usize;
        let count = self.hourly_counts.get(hour % 24).copied().unwrap_or(0.0);
        if count > 0.0 {
            OccupantState::PresentActive
        } else {
            OccupantState::Absent
        }
    }

    fn occupant_count(&self, t: DateTime<Utc>) -> f64 {
        let hour = t.hour() as usize;
        self.hourly_counts.get(hour % 24).copied().unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstantPlugLoadProvider {
    pub watts: f64,
}

impl ConstantPlugLoadProvider {
    pub fn new(watts: f64) -> Self {
        Self { watts }
    }
}

impl PlugLoadProvider for ConstantPlugLoadProvider {
    fn get_plug_load(&self, _t: DateTime<Utc>) -> f64 {
        self.watts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_internal_gains_default() {
        let gains = InternalGains::default();
        assert!((gains.phi_sensible - 0.0).abs() < 1e-10);
        assert!((gains.phi_latent - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_internal_gains_new() {
        let gains = InternalGains::new(100.0, 30.0);
        assert!((gains.phi_sensible - 100.0).abs() < 1e-10);
        assert!((gains.phi_latent - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_adapter_default() {
        let adapter = DynamicInternalGainAdapter::default();
        let t = Utc.with_ymd_and_hms(2024, 7, 15, 10, 0, 0).unwrap();
        let gains = adapter.compute_gains(uuid::Uuid::new_v4(), t);
        assert!(gains.phi_sensible >= 0.0);
        assert!(gains.phi_latent >= 0.0);
    }

    #[test]
    fn test_dynamic_adapter_zero_occupancy() {
        let adapter = DynamicInternalGainAdapter::default();
        let t = Utc.with_ymd_and_hms(2024, 7, 15, 3, 0, 0).unwrap();
        let gains = adapter.compute_gains(uuid::Uuid::new_v4(), t);
        assert!(gains.phi_latent <= 1e-10);
    }

    #[test]
    fn test_dynamic_adapter_with_custom_metabolic() {
        let occupancy = Arc::new(ScheduleOccupancyProvider::new(vec![1.0; 24]));
        let plug_loads = Arc::new(ConstantPlugLoadProvider::new(100.0));
        let lighting = LightingModel::default();

        let adapter = DynamicInternalGainAdapter::new(occupancy, plug_loads, lighting)
            .with_metabolic_rates(60.0, 40.0);

        let t = Utc.with_ymd_and_hms(2024, 7, 15, 10, 0, 0).unwrap();
        let gains = adapter.compute_gains(uuid::Uuid::new_v4(), t);

        assert!(gains.phi_sensible > 0.0);
        assert!((gains.phi_latent - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_schedule_occupancy_provider() {
        let provider = ScheduleOccupancyProvider::new(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let t_night = Utc.with_ymd_and_hms(2024, 7, 15, 3, 0, 0).unwrap();
        let t_day = Utc.with_ymd_and_hms(2024, 7, 15, 10, 0, 0).unwrap();

        assert!((provider.occupant_count(t_night) - 0.0).abs() < 1e-10);
        assert!((provider.occupant_count(t_day) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_constant_plug_load_provider() {
        let provider = ConstantPlugLoadProvider::new(200.0);
        let t = Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        assert!((provider.get_plug_load(t) - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_adapter_arc_traits() {
        let occupancy: Arc<dyn OccupancyProvider> = Arc::new(ScheduleOccupancyProvider::default());
        let plug_loads: Arc<dyn PlugLoadProvider> = Arc::new(ConstantPlugLoadProvider::new(150.0));
        let lighting = LightingModel::default();

        let adapter = DynamicInternalGainAdapter::new(occupancy, plug_loads, lighting);
        let t = Utc.with_ymd_and_hms(2024, 7, 15, 14, 0, 0).unwrap();
        let gains = adapter.compute_gains(uuid::Uuid::new_v4(), t);

        assert!(gains.phi_sensible >= 0.0);
        assert!(gains.phi_latent >= 0.0);
    }
}
