use serde::{Deserialize, Serialize};

use super::lighting::LightingModel;
use super::markov_occupancy::{MarkovOccupancyGenerator, OccupancyState};
use super::moisture::MoistureGeneration;
use super::plug_loads::MockPlugLoad;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalGains {
    pub sensible: f64,
    pub latent: f64,
    pub radiative: f64,
    pub convective: f64,
    pub co2_generation: f64,
    pub moisture_generation: f64,
}

impl Default for InternalGains {
    fn default() -> Self {
        Self::zero()
    }
}

impl InternalGains {
    pub fn zero() -> Self {
        Self {
            sensible: 0.0,
            latent: 0.0,
            radiative: 0.0,
            convective: 0.0,
            co2_generation: 0.0,
            moisture_generation: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicInternalGainAdapter {
    pub occupancy: MarkovOccupancyGenerator,
    pub lighting: LightingModel,
    pub plug_loads: MockPlugLoad,
    pub moisture: MoistureGeneration,
    pub person_heat: f64,
    pub latent_heat_fraction: f64,
}

impl Default for DynamicInternalGainAdapter {
    fn default() -> Self {
        Self::office()
    }
}

impl DynamicInternalGainAdapter {
    pub fn office() -> Self {
        Self {
            occupancy: MarkovOccupancyGenerator::from_ashrae90_1(),
            lighting: LightingModel::office(),
            plug_loads: MockPlugLoad::default(),
            moisture: MoistureGeneration::office(),
            person_heat: 100.0,
            latent_heat_fraction: 0.3,
        }
    }

    pub fn update_occupancy(&mut self, state: OccupancyState) {
        self.occupancy.initial_state = state;
    }

    pub fn gains_for_hour(
        &self,
        hour: f64,
        zone_area: f64,
        occupancy_count: f64,
    ) -> InternalGains {
        let daylight_illuminance = 500.0;

        let lighting_power = self.lighting.lighting_power(hour, zone_area, daylight_illuminance);
        let lighting_radiative = lighting_power * self.lighting.radiative_fraction();
        let lighting_convective = lighting_power * self.lighting.convective_fraction();

        let plug_radiative = self.plug_loads.radiative_gain(hour, zone_area, 1.0);
        let plug_convective = self.plug_loads.convective_gain(hour, zone_area, 1.0);

        let occupancy_fraction = if occupancy_count > 0.0 { 1.0 } else { 0.0 };
        let sensible_per_person = self.person_heat * (1.0 - self.latent_heat_fraction);
        let latent_per_person = self.person_heat * self.latent_heat_fraction;

        let person_sensible = sensible_per_person * occupancy_count * occupancy_fraction;
        let person_latent = latent_per_person * occupancy_count * occupancy_fraction;

        let moisture_gen = self.moisture.moisture_generation_rate(occupancy_count);
        let co2_gen = 0.005 * occupancy_count * occupancy_fraction;

        InternalGains {
            sensible: person_sensible + lighting_convective + plug_convective,
            latent: person_latent,
            radiative: person_sensible * 0.5 + lighting_radiative + plug_radiative,
            convective: person_sensible * 0.5 + lighting_convective + plug_convective,
            co2_generation: co2_gen,
            moisture_generation: moisture_gen,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gains_for_hour() {
        let adapter = DynamicInternalGainAdapter::default();
        let gains = adapter.gains_for_hour(10.0, 100.0, 5.0);

        assert!(gains.sensible > 0.0 || gains.radiative > 0.0 || gains.convective > 0.0);
    }

    #[test]
    fn test_gains_zero_occupancy() {
        let adapter = DynamicInternalGainAdapter::default();
        let gains = adapter.gains_for_hour(10.0, 100.0, 0.0);

        assert!(gains.co2_generation <= 0.0);
    }

    #[test]
    fn test_update_occupancy() {
        let mut adapter = DynamicInternalGainAdapter::default();
        adapter.update_occupancy(OccupancyState::Occupied);
        assert_eq!(adapter.occupancy.initial_state, OccupancyState::Occupied);
    }
}
