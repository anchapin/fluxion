use crate::comfort::{
    AdaptiveComfort, AdaptiveComfortStatus, ComfortMetrics, PmvComfort, PmvComfortStatus,
    TriggerType,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalComfortInput {
    pub air_temp: f64,
    pub radiant_temp: f64,
    pub rel_humidity: f64,
    pub air_velocity: f64,
    pub metabolic_rate: f64,
    pub clothing_level: f64,
}

impl ThermalComfortInput {
    pub fn new(
        air_temp: f64,
        radiant_temp: f64,
        rel_humidity: f64,
        air_velocity: f64,
        metabolic_rate: f64,
        clothing_level: f64,
    ) -> Self {
        Self {
            air_temp,
            radiant_temp,
            rel_humidity,
            air_velocity,
            metabolic_rate,
            clothing_level,
        }
    }
}
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortTrigger {
    pub timestamp: DateTime<Utc>,
    pub zone_id: Uuid,
    pub trigger_type: TriggerType,
    pub metrics: ComfortMetrics,
}

impl ComfortTrigger {
    pub fn new(zone_id: Uuid, metrics: ComfortMetrics) -> Self {
        Self {
            timestamp: Utc::now(),
            zone_id,
            trigger_type: metrics.trigger_type(),
            metrics,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComfortViolationType {
    PmvViolation,
    AdaptiveViolation,
    Co2ThresholdExceeded,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortViolation {
    pub violation_type: ComfortViolationType,
    pub zone_id: String,
    pub timestamp: usize,
    pub severity: f64,
    pub message: String,
}

impl ComfortViolation {
    pub fn new_pmv(
        zone_id: &str,
        timestamp: usize,
        status: PmvComfortStatus,
        severity: f64,
    ) -> Self {
        Self {
            violation_type: ComfortViolationType::PmvViolation,
            zone_id: zone_id.to_string(),
            timestamp,
            severity,
            message: format!("PMV comfort violation: {:?}", status),
        }
    }

    pub fn new_adaptive(
        zone_id: &str,
        timestamp: usize,
        status: AdaptiveComfortStatus,
        severity: f64,
    ) -> Self {
        Self {
            violation_type: ComfortViolationType::AdaptiveViolation,
            zone_id: zone_id.to_string(),
            timestamp,
            severity,
            message: format!("Adaptive comfort violation: {:?}", status),
        }
    }

    pub fn new_co2(zone_id: &str, timestamp: usize, co2_level: f64, threshold: f64) -> Self {
        Self {
            violation_type: ComfortViolationType::Co2ThresholdExceeded,
            zone_id: zone_id.to_string(),
            timestamp,
            severity: (co2_level / threshold - 1.0).max(0.0),
            message: format!(
                "CO2 threshold exceeded: {:.1} ppm > {:.1} ppm",
                co2_level, threshold
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupantComfortTriggersConfig {
    pub co2_threshold_ppm: f64,
    pub pmv_ppd_threshold: f64,
    pub adaptive_comfort_category: u8,
}

impl Default for OccupantComfortTriggersConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl OccupantComfortTriggersConfig {
    pub fn new() -> Self {
        Self {
            co2_threshold_ppm: 1000.0,
            pmv_ppd_threshold: 10.0,
            adaptive_comfort_category: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupantComfortTriggers {
    config: OccupantComfortTriggersConfig,
    pmv_evaluator: PmvComfort,
    adaptive_evaluator: AdaptiveComfort,
}

impl OccupantComfortTriggers {
    pub fn new(config: OccupantComfortTriggersConfig) -> Self {
        let pmv_evaluator = PmvComfort::default().with_ppd_threshold(config.pmv_ppd_threshold);
        Self {
            config,
            pmv_evaluator,
            adaptive_evaluator: AdaptiveComfort::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(OccupantComfortTriggersConfig::new())
    }

    pub fn evaluate_pmv(
        &self,
        zone_id: &str,
        timestep: usize,
        thermal: ThermalComfortInput,
    ) -> Option<ComfortViolation> {
        let pmv = self.pmv_evaluator.calculate_pmv(
            thermal.air_temp,
            thermal.radiant_temp,
            thermal.rel_humidity,
            thermal.air_velocity,
            thermal.metabolic_rate,
            thermal.clothing_level,
        );
        let status = self.pmv_evaluator.evaluate_status(pmv);
        let ppd = self.pmv_evaluator.calculate_ppd(pmv);

        if ppd > self.config.pmv_ppd_threshold {
            Some(ComfortViolation::new_pmv(
                zone_id,
                timestep,
                status,
                (ppd - self.config.pmv_ppd_threshold) / self.config.pmv_ppd_threshold,
            ))
        } else {
            None
        }
    }

    pub fn evaluate_adaptive(
        &self,
        zone_id: &str,
        timestep: usize,
        operative_temp: f64,
        running_mean_temp: f64,
    ) -> Option<ComfortViolation> {
        let status = self.adaptive_evaluator.evaluate_status(
            operative_temp,
            running_mean_temp,
            self.config.adaptive_comfort_category,
        );

        match status {
            AdaptiveComfortStatus::Comfortable => None,
            _ => {
                let severity = match status {
                    AdaptiveComfortStatus::SlightlyWarm | AdaptiveComfortStatus::SlightlyCool => {
                        0.3
                    }
                    AdaptiveComfortStatus::Warm | AdaptiveComfortStatus::Cool => 0.6,
                    AdaptiveComfortStatus::NoAssignmentPossible => 0.5,
                    _ => 0.0,
                };
                Some(ComfortViolation::new_adaptive(
                    zone_id, timestep, status, severity,
                ))
            }
        }
    }

    pub fn evaluate_co2(
        &self,
        zone_id: &str,
        timestep: usize,
        co2_level: f64,
    ) -> Option<ComfortViolation> {
        if co2_level > self.config.co2_threshold_ppm {
            Some(ComfortViolation::new_co2(
                zone_id,
                timestep,
                co2_level,
                self.config.co2_threshold_ppm,
            ))
        } else {
            None
        }
    }

    pub fn evaluate_all(
        &self,
        zone_id: &str,
        timestep: usize,
        thermal: ThermalComfortInput,
        operative_temp: f64,
        running_mean_temp: f64,
        co2_level: f64,
    ) -> Vec<ComfortViolation> {
        let mut violations = Vec::new();

        if let Some(v) = self.evaluate_pmv(zone_id, timestep, thermal) {
            violations.push(v);
        }

        if let Some(v) =
            self.evaluate_adaptive(zone_id, timestep, operative_temp, running_mean_temp)
        {
            violations.push(v);
        }

        if let Some(v) = self.evaluate_co2(zone_id, timestep, co2_level) {
            violations.push(v);
        }

        violations
    }
}

impl Default for OccupantComfortTriggers {
    fn default() -> Self {
        Self::with_defaults()
    }
}
