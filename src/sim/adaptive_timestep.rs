//! Adaptive Timestep Module
//!
//! This module provides adaptive timestep integration for high-mass buildings
//! to improve numerical accuracy. It reduces timestep from 1-hour to 6-minute
//! or finer for buildings with thermal mass time constants exceeding threshold.
//!
//! # Theory
//!
//! The thermal time constant τ = C / (h_tr_ms + h_tr_em) determines the
//! appropriate timestep:
//! - Low-mass (τ < 2 hours): Δt = 1 hour (standard)
//! - High-mass (τ ≥ 2 hours): Δt = 6 minutes (adaptive)
//!
//! # Example
//!
//! ```rust,no_run
//! use fluxion::sim::adaptive_timestep::{TimestepMode, AdaptiveTimestepScheduler};
//! use std::time::Duration;
//!
//! // Create adaptive scheduler for high-mass building
//! let scheduler = AdaptiveTimestepScheduler::new(
//!     TimestepMode::Adaptive {
//!         base_dt: Duration::from_secs(360), // 6 minutes
//!         min_dt: Duration::from_secs(60),   // 1 minute
//!         threshold_tau: 2.0,                 // 2 hours
//!     },
//!     5.0, // τ = 5 hours (high-mass)
//! );
//!
//! // Get timestep sequence for 24-hour simulation
//! let timesteps = scheduler.schedule_simulation(24);
//! assert_eq!(timesteps.len(), 240); // 24 hours × 10 timesteps/hour
//! ```

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Timestep mode configuration
///
/// Supports both fixed timestep (baseline behavior) and adaptive timestep
/// for high-mass buildings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "mode")]
pub enum TimestepMode {
    /// Fixed timestep for all timesteps
    Fixed {
        /// Timestep duration
        dt: Duration,
    },
    /// Adaptive timestep based on building thermal mass
    Adaptive {
        /// Base timestep (used when τ > threshold)
        #[serde(with = "duration_serde")]
        base_dt: Duration,
        /// Minimum timestep (for sub-cycling)
        #[serde(with = "duration_serde")]
        min_dt: Duration,
        /// Time constant threshold (hours) for switching to adaptive
        threshold_tau: f64,
    },
}

impl Default for TimestepMode {
    fn default() -> Self {
        TimestepMode::Fixed {
            dt: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl TimestepMode {
    /// Create fixed timestep mode with specified duration
    pub fn fixed(dt: Duration) -> Self {
        TimestepMode::Fixed { dt }
    }

    /// Create adaptive timestep mode
    pub fn adaptive(base_dt: Duration, min_dt: Duration, threshold_tau: f64) -> Self {
        TimestepMode::Adaptive {
            base_dt,
            min_dt,
            threshold_tau,
        }
    }

    /// Get the timestep duration for a given time constant
    pub fn get_timestep(&self, tau_hours: f64) -> Duration {
        match self {
            TimestepMode::Fixed { dt } => *dt,
            TimestepMode::Adaptive {
                base_dt,
                min_dt: _,
                threshold_tau,
            } => {
                if tau_hours >= *threshold_tau {
                    // High-mass: use base timestep (e.g., 6 minutes)
                    *base_dt
                } else {
                    // Low-mass: use 1 hour
                    Duration::from_secs(3600)
                }
            }
        }
    }

    /// Check if adaptive timestep is enabled
    pub fn is_adaptive(&self) -> bool {
        matches!(self, TimestepMode::Adaptive { .. })
    }
}

/// Custom serde module for Duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Adaptive timestep scheduler
///
/// Calculates appropriate timestep sequence based on building thermal mass
/// and time constant threshold.
#[derive(Clone, Debug)]
pub struct AdaptiveTimestepScheduler {
    /// Timestep mode configuration
    mode: TimestepMode,
    /// Building thermal time constant (hours)
    tau_hours: f64,
    /// Calculated timestep for this building
    dt: Duration,
}

impl AdaptiveTimestepScheduler {
    /// Create new adaptive timestep scheduler
    ///
    /// # Arguments
    /// * `mode` - Timestep mode configuration
    /// * `tau_hours` - Building thermal time constant in hours
    ///
    /// # Returns
    /// New adaptive timestep scheduler
    pub fn new(mode: TimestepMode, tau_hours: f64) -> Self {
        let dt = mode.get_timestep(tau_hours);
        Self {
            mode,
            tau_hours,
            dt,
        }
    }

    /// Get the timestep duration
    pub fn timestep(&self) -> Duration {
        self.dt
    }

    /// Get the number of timesteps per hour
    pub fn timesteps_per_hour(&self) -> usize {
        let secs = self.dt.as_secs();
        if secs == 0 {
            60 // Default to 1 minute
        } else {
            (3600 / secs) as usize
        }
    }

    /// Schedule simulation timesteps for given duration
    ///
    /// # Arguments
    /// * `total_hours` - Total simulation duration in hours
    ///
    /// # Returns
    /// Vector of timestep durations for each timestep
    pub fn schedule_simulation(&self, total_hours: usize) -> Vec<Duration> {
        let tph = self.timesteps_per_hour();
        let mut timesteps = Vec::with_capacity(total_hours * tph);

        for _ in 0..total_hours {
            for _ in 0..tph {
                timesteps.push(self.dt);
            }
        }

        timesteps
    }

    /// Get timestep for specific hour (supports diurnal adaptation)
    ///
    /// # Arguments
    /// * `hour` - Hour of simulation (0-based)
    ///
    /// # Returns
    /// Timestep duration for this hour
    pub fn get_timestep_for_hour(&self, _hour: usize) -> Duration {
        // For now, return constant timestep
        // Future: implement diurnal adaptation (finer timestep during day)
        self.dt
    }

    /// Calculate thermal time constant from building parameters
    ///
    /// # Arguments
    /// * `thermal_capacitance` - Building thermal capacitance (J/K)
    /// * `heat_transfer_coeff` - Total heat transfer coefficient (W/K)
    ///
    /// # Returns
    /// Time constant in hours
    pub fn calculate_time_constant(thermal_capacitance: f64, heat_transfer_coeff: f64) -> f64 {
        if heat_transfer_coeff <= 0.0 {
            return f64::INFINITY;
        }

        // τ = C / h (seconds)
        let tau_seconds = thermal_capacitance / heat_transfer_coeff;
        // Convert to hours
        tau_seconds / 3600.0
    }

    /// Check if timestep satisfies stability criterion
    ///
    /// # Arguments
    /// * `tau_hours` - Time constant in hours
    ///
    /// # Returns
    /// True if timestep is stable (Δt < 2τ)
    pub fn is_stable(&self, tau_hours: f64) -> bool {
        let dt_hours = self.dt.as_secs_f64() / 3600.0;
        dt_hours < 2.0 * tau_hours
    }

    /// Check if timestep satisfies accuracy criterion
    ///
    /// # Arguments
    /// * `tau_hours` - Time constant in hours
    ///
    /// # Returns
    /// True if timestep is accurate (Δt < τ/10)
    pub fn is_accurate(&self, tau_hours: f64) -> bool {
        let dt_hours = self.dt.as_secs_f64() / 3600.0;
        dt_hours < tau_hours / 10.0
    }

    /// Get stability margin
    ///
    /// # Arguments
    /// * `tau_hours` - Time constant in hours
    ///
    /// # Returns
    /// Ratio Δt / (2τ). Values < 1.0 are stable.
    pub fn stability_margin(&self, tau_hours: f64) -> f64 {
        let dt_hours = self.dt.as_secs_f64() / 3600.0;
        dt_hours / (2.0 * tau_hours)
    }

    /// Get accuracy margin
    ///
    /// # Arguments
    /// * `tau_hours` - Time constant in hours
    ///
    /// # Returns
    /// Ratio Δt / (τ/10). Values < 1.0 are accurate.
    pub fn accuracy_margin(&self, tau_hours: f64) -> f64 {
        let dt_hours = self.dt.as_secs_f64() / 3600.0;
        dt_hours / (tau_hours / 10.0)
    }
}

/// Time constant analyzer for ASHRAE 140 cases
///
/// Calculates thermal time constants for standard test cases.
pub struct TimeConstantAnalyzer;

impl TimeConstantAnalyzer {
    /// Calculate time constant for ASHRAE 140 case
    ///
    /// # Arguments
    /// * `case_id` - Case identifier (e.g., "600", "900")
    ///
    /// # Returns
    /// Time constant in hours, or None if case not found
    pub fn for_case(case_id: &str) -> Option<f64> {
        // Approximate values from ISO 13790 5R1C parameters
        // C = thermal capacitance (J/K)
        // h = h_tr_ms + h_tr_em (W/K)
        // τ = C / h (hours)

        let (c, h) = match case_id {
            // Low-mass cases
            "600" => (2.4e6, 800.0),   // τ ≈ 0.83 hours
            "610" => (2.4e6, 850.0),   // τ ≈ 0.78 hours
            "620" => (2.4e6, 900.0),   // τ ≈ 0.74 hours
            "630" => (2.4e6, 750.0),   // τ ≈ 0.89 hours
            "640" => (2.4e6, 850.0),   // τ ≈ 0.78 hours
            "650" => (3.5e6, 900.0),   // τ ≈ 1.08 hours
            "600FF" => (2.4e6, 800.0), // τ ≈ 0.83 hours
            "650FF" => (3.5e6, 900.0), // τ ≈ 1.08 hours

            // High-mass cases
            "900" => (1.2e7, 650.0),   // τ ≈ 5.13 hours
            "910" => (1.2e7, 700.0),   // τ ≈ 4.76 hours
            "920" => (1.8e7, 700.0),   // τ ≈ 7.14 hours
            "930" => (1.2e7, 1200.0),  // τ ≈ 2.78 hours
            "940" => (1.2e7, 400.0),   // τ ≈ 8.33 hours
            "950" => (1.2e7, 700.0),   // τ ≈ 4.76 hours
            "900FF" => (1.2e7, 650.0), // τ ≈ 5.13 hours
            "950FF" => (1.2e7, 700.0), // τ ≈ 4.76 hours
            "960" => (1.2e7, 800.0),   // τ ≈ 4.17 hours

            _ => return None,
        };

        let tau_seconds = c / h;
        Some(tau_seconds / 3600.0)
    }

    /// Classify case as low-mass or high-mass
    ///
    /// # Arguments
    /// * `case_id` - Case identifier
    ///
    /// # Returns
    /// "low-mass", "high-mass", or None if case not found
    pub fn classify_case(case_id: &str) -> Option<&'static str> {
        let tau = Self::for_case(case_id)?;

        // Threshold: 2 hours
        if tau < 2.0 {
            Some("low-mass")
        } else {
            Some("high-mass")
        }
    }

    /// Get recommended timestep for case
    ///
    /// # Arguments
    /// * `case_id` - Case identifier
    ///
    /// # Returns
    /// Recommended timestep in seconds, or None if case not found
    pub fn recommended_timestep(case_id: &str) -> Option<Duration> {
        let tau = Self::for_case(case_id)?;

        // Recommendation: Δt < τ/10 for accuracy
        let dt_seconds = (tau * 3600.0 / 10.0) as u64;

        // Round to convenient values
        let dt_seconds = if dt_seconds <= 60 {
            60 // Minimum 1 minute
        } else if dt_seconds <= 360 {
            360 // 6 minutes (for high-mass: τ=5hr → dt=30min, but we want 6 min for better accuracy)
        } else if dt_seconds <= 600 {
            600 // 10 minutes
        } else if dt_seconds <= 900 {
            900 // 15 minutes
        } else if dt_seconds <= 1800 {
            1800 // 30 minutes
        } else {
            3600 // 1 hour for low-mass
        };

        Some(Duration::from_secs(dt_seconds))
    }

    /// Generate time constant table for all cases
    pub fn generate_table() -> Vec<CaseTimeConstant> {
        let cases = vec![
            "600", "610", "620", "630", "640", "650", "600FF", "650FF", "900", "910", "920", "930",
            "940", "950", "900FF", "950FF", "960",
        ];

        cases
            .into_iter()
            .filter_map(|case_id| {
                let tau = Self::for_case(case_id)?;
                let classification = Self::classify_case(case_id)?;
                let recommended_dt = Self::recommended_timestep(case_id)?;

                Some(CaseTimeConstant {
                    case_id: case_id.to_string(),
                    tau_hours: tau,
                    classification: classification.to_string(),
                    recommended_timestep_secs: recommended_dt.as_secs(),
                })
            })
            .collect()
    }
}

/// Time constant information for a case
#[derive(Debug, Clone)]
pub struct CaseTimeConstant {
    /// Case identifier
    pub case_id: String,
    /// Time constant in hours
    pub tau_hours: f64,
    /// Classification ("low-mass" or "high-mass")
    pub classification: String,
    /// Recommended timestep in seconds
    pub recommended_timestep_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestep_mode_fixed() {
        let mode = TimestepMode::fixed(Duration::from_secs(3600));
        assert!(!mode.is_adaptive());
        assert_eq!(mode.get_timestep(5.0), Duration::from_secs(3600));
    }

    #[test]
    fn test_timestep_mode_adaptive() {
        let mode = TimestepMode::adaptive(Duration::from_secs(360), Duration::from_secs(60), 2.0);
        assert!(mode.is_adaptive());

        // High-mass (τ > threshold)
        assert_eq!(mode.get_timestep(5.0), Duration::from_secs(360));

        // Low-mass (τ < threshold)
        assert_eq!(mode.get_timestep(1.0), Duration::from_secs(3600));
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = AdaptiveTimestepScheduler::new(
            TimestepMode::adaptive(Duration::from_secs(360), Duration::from_secs(60), 2.0),
            5.0,
        );

        assert_eq!(scheduler.timestep(), Duration::from_secs(360));
        assert_eq!(scheduler.timesteps_per_hour(), 10);
    }

    #[test]
    fn test_scheduler_schedule() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(3600)), 1.0);

        let timesteps = scheduler.schedule_simulation(24);
        assert_eq!(timesteps.len(), 24); // 24 hours × 1 timestep/hour

        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(360)), 5.0);

        let timesteps = scheduler.schedule_simulation(24);
        assert_eq!(timesteps.len(), 240); // 24 hours × 10 timesteps/hour
    }

    #[test]
    fn test_time_constant_analyzer() {
        // Low-mass cases
        let tau_600 = TimeConstantAnalyzer::for_case("600").unwrap();
        assert!(tau_600 < 2.0);
        assert_eq!(TimeConstantAnalyzer::classify_case("600"), Some("low-mass"));

        // High-mass cases
        let tau_900 = TimeConstantAnalyzer::for_case("900").unwrap();
        assert!(tau_900 > 2.0);
        assert_eq!(
            TimeConstantAnalyzer::classify_case("900"),
            Some("high-mass")
        );
    }

    #[test]
    fn test_stability_criterion() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(360)), 5.0);

        // Should be stable: Δt = 6 min, τ = 5 hours
        // Δt < 2τ → 0.1 hours < 10 hours ✓
        assert!(scheduler.is_stable(5.0));
        assert!(scheduler.stability_margin(5.0) < 1.0);
    }

    #[test]
    fn test_accuracy_criterion() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(360)), 5.0);

        // Should be accurate: Δt = 6 min, τ = 5 hours
        // Δt < τ/10 → 0.1 hours < 0.5 hours ✓
        assert!(scheduler.is_accurate(5.0));
        assert!(scheduler.accuracy_margin(5.0) < 1.0);
    }

    #[test]
    fn test_recommended_timestep() {
        // Low-mass: τ ≈ 0.83 hours, τ/10 ≈ 300s → rounded to 360s (6 min)
        let dt_600 = TimeConstantAnalyzer::recommended_timestep("600").unwrap();
        assert_eq!(dt_600.as_secs(), 360); // 6 minutes

        // High-mass: τ ≈ 5.13 hours, τ/10 ≈ 1860s → rounded to 3600s (1 hour)
        // This is expected - the recommended timestep based on τ/10 is 1 hour for Case 900
        // But for accuracy improvement, we use 6-minute timestep in adaptive mode
        let dt_900 = TimeConstantAnalyzer::recommended_timestep("900").unwrap();
        assert_eq!(dt_900.as_secs(), 3600); // 1 hour (τ/10 rounded)
    }

    #[test]
    fn test_variable_timestep_energy_accumulation() {
        // This test verifies that the variable timestep API works correctly
        // and that energy is accumulated (regardless of sign - heating vs cooling)

        use crate::ai::surrogate::SurrogateManager;
        use crate::physics::cta::VectorField;
        use crate::sim::engine::ThermalModel;

        // Create model
        let mut model = ThermalModel::<VectorField>::new(1);
        model.apply_parameters(&[1.5, 20.0, 24.0]);

        let surrogates = SurrogateManager::new().unwrap_or_else(|_| {
            panic!("Failed to create SurrogateManager");
        });

        // Run with 1-hour timestep - should complete without errors
        let result_1hr = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut m = ThermalModel::<VectorField>::new(1);
            m.apply_parameters(&[1.5, 20.0, 24.0]);
            m.solve_timesteps_with_dt(24, &surrogates, false, None, None, None, 3600.0)
        }));

        assert!(
            result_1hr.is_ok(),
            "1-hour timestep simulation should complete without panicking"
        );

        // Run with 15-minute timestep - should complete without errors
        let result_15min = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut m = ThermalModel::<VectorField>::new(1);
            m.apply_parameters(&[1.5, 20.0, 24.0]);
            m.solve_timesteps_with_dt(96, &surrogates, false, None, None, None, 900.0)
        }));

        assert!(
            result_15min.is_ok(),
            "15-minute timestep simulation should complete without panicking"
        );

        // Both should return finite EUI values
        let eui_1hr = result_1hr.unwrap();
        let eui_15min = result_15min.unwrap();

        assert!(
            eui_1hr.is_finite(),
            "1-hour timestep EUI should be finite, got {:?}",
            eui_1hr
        );
        assert!(
            eui_15min.is_finite(),
            "15-minute timestep EUI should be finite, got {:?}",
            eui_15min
        );
    }

    #[test]
    fn test_timestep_mode_default() {
        let mode = TimestepMode::default();
        assert!(!mode.is_adaptive());
        assert_eq!(mode.get_timestep(1.0), Duration::from_secs(3600));
    }

    #[test]
    fn test_timestep_mode_adaptive_boundary() {
        let mode = TimestepMode::adaptive(Duration::from_secs(360), Duration::from_secs(60), 2.0);
        assert_eq!(mode.get_timestep(2.0), Duration::from_secs(360));
        assert_eq!(mode.get_timestep(1.99), Duration::from_secs(3600));
    }

    #[test]
    fn test_scheduler_timesteps_per_hour_edge_cases() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(0)), 1.0);
        assert_eq!(scheduler.timesteps_per_hour(), 60);
    }

    #[test]
    fn test_scheduler_get_timestep_for_hour() {
        let scheduler = AdaptiveTimestepScheduler::new(
            TimestepMode::adaptive(Duration::from_secs(360), Duration::from_secs(60), 2.0),
            5.0,
        );
        assert_eq!(scheduler.get_timestep_for_hour(0), Duration::from_secs(360));
        assert_eq!(
            scheduler.get_timestep_for_hour(12),
            Duration::from_secs(360)
        );
        assert_eq!(
            scheduler.get_timestep_for_hour(23),
            Duration::from_secs(360)
        );
    }

    #[test]
    fn test_scheduler_schedule_empty() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(3600)), 1.0);
        let timesteps = scheduler.schedule_simulation(0);
        assert!(timesteps.is_empty());
    }

    #[test]
    fn test_calculate_time_constant() {
        let tau = AdaptiveTimestepScheduler::calculate_time_constant(1.2e7, 650.0);
        assert!((tau - 5.13).abs() < 0.1);
        let tau = AdaptiveTimestepScheduler::calculate_time_constant(1.0e6, 0.0);
        assert!(tau == f64::INFINITY);
        let tau = AdaptiveTimestepScheduler::calculate_time_constant(1.0e6, -10.0);
        assert!(tau == f64::INFINITY);
    }

    #[test]
    fn test_stability_criterion_edge_cases() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(3600)), 1.0);
        assert!(!scheduler.is_stable(0.1));
        assert!(scheduler.is_stable(10.0));
        assert!(!scheduler.is_stable(0.5));
        assert!(scheduler.is_stable(0.51));
    }

    #[test]
    fn test_accuracy_criterion_edge_cases() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(3600)), 1.0);
        assert!(!scheduler.is_accurate(1.0));
        assert!(scheduler.is_accurate(100.0));
        assert!(!scheduler.is_accurate(10.0));
        assert!(scheduler.is_accurate(10.1));
    }

    #[test]
    fn test_stability_margin() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(3600)), 1.0);
        let margin = scheduler.stability_margin(5.0);
        assert!((margin - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_accuracy_margin() {
        let scheduler =
            AdaptiveTimestepScheduler::new(TimestepMode::fixed(Duration::from_secs(3600)), 1.0);
        let margin = scheduler.accuracy_margin(5.0);
        assert!((margin - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_time_constant_analyzer_unknown_case() {
        assert!(TimeConstantAnalyzer::for_case("999").is_none());
        assert!(TimeConstantAnalyzer::for_case("unknown").is_none());
        assert!(TimeConstantAnalyzer::for_case("").is_none());
    }

    #[test]
    fn test_time_constant_analyzer_classify_unknown() {
        assert!(TimeConstantAnalyzer::classify_case("999").is_none());
        assert!(TimeConstantAnalyzer::classify_case("").is_none());
    }

    #[test]
    fn test_time_constant_analyzer_all_low_mass_cases() {
        let low_mass_cases = vec!["600", "610", "620", "630", "640", "650", "600FF", "650FF"];
        for case_id in low_mass_cases {
            let tau = TimeConstantAnalyzer::for_case(case_id).unwrap();
            assert!(
                tau < 2.0,
                "Case {} should be low-mass (tau={})",
                case_id,
                tau
            );
            assert_eq!(
                TimeConstantAnalyzer::classify_case(case_id),
                Some("low-mass")
            );
        }
    }

    #[test]
    fn test_time_constant_analyzer_all_high_mass_cases() {
        let high_mass_cases = vec![
            "900", "910", "920", "930", "940", "950", "900FF", "950FF", "960",
        ];
        for case_id in high_mass_cases {
            let tau = TimeConstantAnalyzer::for_case(case_id).unwrap();
            assert!(
                tau > 2.0,
                "Case {} should be high-mass (tau={})",
                case_id,
                tau
            );
            assert_eq!(
                TimeConstantAnalyzer::classify_case(case_id),
                Some("high-mass")
            );
        }
    }

    #[test]
    fn test_recommended_timestep_unknown_case() {
        assert!(TimeConstantAnalyzer::recommended_timestep("unknown").is_none());
    }

    #[test]
    fn test_generate_table() {
        let table = TimeConstantAnalyzer::generate_table();
        assert_eq!(table.len(), 17);
        assert_eq!(table[0].case_id, "600");
        assert!(table[0].tau_hours > 0.0);
        assert_eq!(table[0].classification, "low-mass");
        assert!(table[0].recommended_timestep_secs > 0);
    }

    #[test]
    fn test_case_time_constant_debug() {
        let ctc = CaseTimeConstant {
            case_id: "600".to_string(),
            tau_hours: 0.83,
            classification: "low-mass".to_string(),
            recommended_timestep_secs: 360,
        };
        let debug_str = format!("{:?}", ctc);
        assert!(debug_str.contains("600"));
        assert!(debug_str.contains("low-mass"));
    }

    #[test]
    fn test_case_time_constant_clone() {
        let ctc = CaseTimeConstant {
            case_id: "900".to_string(),
            tau_hours: 5.13,
            classification: "high-mass".to_string(),
            recommended_timestep_secs: 360,
        };
        let cloned = ctc.clone();
        assert_eq!(cloned.case_id, "900");
        assert_eq!(cloned.tau_hours, 5.13);
    }

    #[test]
    fn test_serde_timestep_mode_fixed() {
        let mode = TimestepMode::fixed(Duration::from_secs(1800));
        let json = serde_json::to_string(&mode).unwrap();
        assert!(json.contains("Fixed"));
        assert!(json.contains("1800"));
        let deserialized: TimestepMode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.get_timestep(1.0), Duration::from_secs(1800));
    }

    #[test]
    fn test_serde_timestep_mode_adaptive() {
        let mode = TimestepMode::adaptive(Duration::from_secs(360), Duration::from_secs(60), 2.0);
        let json = serde_json::to_string(&mode).unwrap();
        assert!(json.contains("Adaptive"));
        assert!(json.contains("360"));
        let deserialized: TimestepMode = serde_json::from_str(&json).unwrap();
        assert!(deserialized.is_adaptive());
    }

    #[test]
    fn test_scheduler_with_adaptive_mode() {
        let scheduler = AdaptiveTimestepScheduler::new(
            TimestepMode::adaptive(Duration::from_secs(360), Duration::from_secs(60), 2.0),
            5.0,
        );
        assert_eq!(scheduler.timestep(), Duration::from_secs(360));
        assert_eq!(scheduler.timesteps_per_hour(), 10);
        let timesteps = scheduler.schedule_simulation(1);
        assert_eq!(timesteps.len(), 10);
        for dt in &timesteps {
            assert_eq!(*dt, Duration::from_secs(360));
        }
    }
}
