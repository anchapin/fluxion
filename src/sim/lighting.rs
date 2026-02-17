//! Daylighting and Lighting Control Models
//!
//! This module provides daylighting modeling and automated lighting controls
//! for commercial building energy simulations.

use serde::{Deserialize, Serialize};

/// Lighting control types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightingControlType {
    /// Manual on/off switching
    Manual,
    /// Continuous dimming based on daylight
    ContinuousDimming,
    /// Stepped dimming (multiple levels)
    SteppedDimming,
    /// Occupancy-based on/off
    OccupancySensing,
}

/// Represents a daylight zone for lighting control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaylightZone {
    /// Zone identifier
    pub id: String,
    /// Associated thermal zone
    pub thermal_zone_id: usize,
    /// Window area contributing daylight (m²)
    pub window_area: f64,
    /// Window height for daylight calculations (m)
    pub window_height: f64,
    /// Depth of daylight zone (m)
    pub daylight_zone_depth: f64,
    /// Average daylight factor (%)
    pub daylight_factor: f64,
    /// Illuminance threshold for dimming (lux)
    pub dimming_threshold: f64,
    /// Minimum lighting level when dimming (fraction 0-1)
    pub min_dimming_level: f64,
}

impl DaylightZone {
    /// Create a new daylight zone
    pub fn new(
        id: String,
        thermal_zone_id: usize,
        window_area: f64,
        window_height: f64,
    ) -> Self {
        Self {
            id,
            thermal_zone_id,
            window_area,
            window_height,
            daylight_zone_depth: window_height * 1.5, // Default depth based on window height
            daylight_factor: 5.0, // Default 5% DF
            dimming_threshold: 300.0, // lux
            min_dimming_level: 0.1, // 10% minimum
        }
    }

    /// Calculate interior daylight illuminance
    /// # Arguments
    /// * `exterior_illuminance` - Exterior horizontal illuminance (lux)
    /// * `sky_condition` - Sky condition factor (0-1, 1 = clear sky)
    pub fn interior_illuminance(&self, exterior_illuminance: f64, sky_condition: f64) -> f64 {
        // Interior illuminance = Exterior * DF * Sky Factor * Geometry Factor
        // Simplified: only using DF and sky condition
        exterior_illuminance * (self.daylight_factor / 100.0) * sky_condition
    }

    /// Calculate dimming level based on illuminance
    /// Returns fraction (0-1) of maximum lighting output
    pub fn dimming_level(&self, interior_illuminance: f64) -> f64 {
        if interior_illuminance >= self.dimming_threshold {
            // Fully daylit - minimum artificial lighting
            self.min_dimming_level
        } else {
            // Interpolate between min and max based on illuminance
            let fraction = interior_illuminance / self.dimming_threshold;
            self.min_dimming_level + fraction * (1.0 - self.min_dimming_level)
        }
    }

    /// Calculate energy savings from daylighting
    /// # Arguments
    /// * `baseline_power` - Lighting power without daylighting (W)
    /// * `hours_per_day` - Operating hours per day
    /// * `days_per_year` - Operating days per year
    pub fn annual_energy_savings(
        &self,
        baseline_power: f64,
        hours_per_day: f64,
        days_per_year: f64,
        average_illuminance: f64,
    ) -> f64 {
        let dimming = self.dimming_level(average_illuminance);
        let energy_reduction = 1.0 - dimming;
        baseline_power * hours_per_day * days_per_year * energy_reduction / 1000.0 // kWh/year
    }
}

/// Represents an automated shading system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadingControl {
    /// Shading device type
    pub shading_type: ShadingType,
    /// Position (0 = fully open, 1 = fully closed)
    pub position: f64,
    /// Solar irradiance threshold to deploy shading (W/m²)
    pub deployment_threshold: f64,
    /// Minimum outdoor temperature to allow shading (°C)
    pub min_temp_deployment: f64,
    /// Whether shading is currently deployed
    pub is_deployed: bool,
}

/// Types of shading devices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadingType {
    /// Interior blinds
    InteriorBlinds,
    /// Exterior blinds
    ExteriorBlinds,
    /// Roller shades
    RollerShades,
    /// Light shelves
    LightShelves,
}

impl ShadingControl {
    /// Create a new shading control
    pub fn new(shading_type: ShadingType) -> Self {
        Self {
            shading_type,
            position: 0.0,
            deployment_threshold: 300.0, // W/m²
            min_temp_deployment: 15.0,    // °C
            is_deployed: false,
        }
    }

    /// Determine shading deployment based on conditions
    pub fn update(&mut self, solar_irradiance: f64, outdoor_temp: f64) {
        // Deploy shading if irradiance exceeds threshold and temp is comfortable
        if solar_irradiance > self.deployment_threshold && outdoor_temp > self.min_temp_deployment {
            self.is_deployed = true;
            self.position = 1.0;
        } else {
            self.is_deployed = false;
            self.position = 0.0;
        }
    }

    /// Calculate solar heat gain coefficient reduction from shading
    pub fn shgc_reduction(&self) -> f64 {
        if !self.is_deployed {
            return 0.0;
        }

        // Different shading types have different effectiveness
        match self.shading_type {
            ShadingType::InteriorBlinds => 0.3 * self.position,
            ShadingType::ExteriorBlinds => 0.6 * self.position,
            ShadingType::RollerShades => 0.5 * self.position,
            ShadingType::LightShelves => 0.2 * self.position,
        }
    }
}

/// Represents an artificial lighting schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingSchedule {
    /// Hourly lighting schedule (0-23), values 0-1
    pub hourly_schedule: [f64; 24],
    /// Lighting power density (W/m²)
    pub power_density: f64,
    /// Zone area this schedule applies to (m²)
    pub zone_area: f64,
}

impl LightingSchedule {
    /// Create a new lighting schedule with default (off) values
    pub fn new(power_density: f64, zone_area: f64) -> Self {
        Self {
            hourly_schedule: [0.0; 24],
            power_density,
            zone_area,
        }
    }

    /// Create an office lighting schedule (8am - 6pm)
    pub fn office_schedule(power_density: f64, zone_area: f64) -> Self {
        let mut schedule = Self::new(power_density, zone_area);
        for hour in 8..=17 {
            schedule.hourly_schedule[hour] = 1.0;
        }
        schedule
    }

    /// Create a retail lighting schedule
    pub fn retail_schedule(power_density: f64, zone_area: f64) -> Self {
        let mut schedule = Self::new(power_density, zone_area);
        for hour in 9..=20 {
            schedule.hourly_schedule[hour] = 1.0;
        }
        schedule
    }

    /// Get lighting power for a specific hour
    pub fn lighting_power(&self, hour: usize) -> f64 {
        let h = hour % 24;
        self.power_density * self.zone_area * self.hourly_schedule[h]
    }

    /// Calculate annual lighting energy consumption (kWh)
    pub fn annual_energy(&self, operating_days: usize) -> f64 {
        let daily_energy: f64 = (0..24)
            .map(|h| self.lighting_power(h))
            .sum::<f64>()
            / 1000.0; // Convert to kWh
        daily_energy * operating_days as f64
    }
}

/// Combined lighting system with controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingSystem {
    /// Artificial lighting schedule
    pub schedule: LightingSchedule,
    /// Daylight zones
    pub daylight_zones: Vec<DaylightZone>,
    /// Shading controls per orientation
    pub shading_controls: Vec<ShadingControl>,
    /// Control type
    pub control_type: LightingControlType,
}

impl LightingSystem {
    /// Create a new lighting system
    pub fn new(power_density: f64, zone_area: f64) -> Self {
        Self {
            schedule: LightingSchedule::office_schedule(power_density, zone_area),
            daylight_zones: Vec::new(),
            shading_controls: Vec::new(),
            control_type: LightingControlType::ContinuousDimming,
        }
    }

    /// Add a daylight zone
    pub fn add_daylight_zone(&mut self, zone: DaylightZone) {
        self.daylight_zones.push(zone);
    }

    /// Add shading control for an orientation
    pub fn add_shading(&mut self, shading: ShadingControl) {
        self.shading_controls.push(shading);
    }

    /// Calculate effective lighting power with controls
    /// # Arguments
    /// * `hour` - Hour of day (0-23)
    /// * `exterior_illuminance` - Exterior illuminance (lux)
    /// * `sky_condition` - Sky factor (0-1)
    pub fn effective_lighting_power(
        &self,
        hour: usize,
        exterior_illuminance: f64,
        sky_condition: f64,
    ) -> f64 {
        let base_power = self.schedule.lighting_power(hour);

        // Apply daylighting dimming if applicable
        if self.control_type == LightingControlType::ContinuousDimming
            && !self.daylight_zones.is_empty()
        {
            // Average dimming across all daylight zones
            let avg_dimming: f64 = self
                .daylight_zones
                .iter()
                .map(|dz| dz.dimming_level(dz.interior_illuminance(exterior_illuminance, sky_condition)))
                .sum::<f64>()
                / self.daylight_zones.len() as f64;

            return base_power * avg_dimming;
        }

        base_power
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daylight_zone() {
        let zone = DaylightZone::new(
            "DZ-1".to_string(),
            0,
            10.0,
            2.0,
        );

        let illuminance = zone.interior_illuminance(10000.0, 0.8);
        assert!(illuminance > 0.0);

        let dimming = zone.dimming_level(500.0);
        assert!(dimming >= 0.1 && dimming <= 1.0);
    }

    #[test]
    fn test_shading_control() {
        let mut shading = ShadingControl::new(ShadingType::InteriorBlinds);
        
        // High irradiance should deploy shading
        shading.update(500.0, 25.0);
        assert!(shading.is_deployed);

        // Low irradiance should retract
        shading.update(100.0, 25.0);
        assert!(!shading.is_deployed);
    }

    #[test]
    fn test_lighting_schedule() {
        let schedule = LightingSchedule::office_schedule(10.0, 100.0);
        
        // During operating hours
        assert!(schedule.lighting_power(10) > 0.0);
        
        // Outside operating hours
        assert_eq!(schedule.lighting_power(2), 0.0);
    }

    #[test]
    fn test_lighting_system() {
        let mut system = LightingSystem::new(10.0, 100.0);
        
        let mut dz = DaylightZone::new("DZ-1".to_string(), 0, 10.0, 2.0);
        dz.dimming_threshold = 500.0;
        system.add_daylight_zone(dz);

        // Test with high daylight
        let power = system.effective_lighting_power(12, 10000.0, 0.8);
        assert!(power < 1000.0); // Should be dimmed
    }
}
