//! HVAC Cycling Loss Tracking
//!
//! This module models realistic equipment cycling behavior including startup
//! penalties, minimum runtime constraints, and part-load efficiency degradation.

use serde::{Deserialize, Serialize};

/// Cycling loss tracking for equipment.
///
/// Models realistic equipment cycling behavior:
/// 1. Startup penalty: Energy consumed during compressor/equipment startup
/// 2. Minimum runtime: Equipment must run minimum time after startup (prevents short-cycling)
/// 3. PLR degradation: Efficiency penalty at low part-load ratios
///
/// Combined approach per CONTEXT.md decision: startup penalty + minimum runtime + PLR degradation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CyclingTracker {
    /// Equipment state from previous timestep (true = on, false = off)
    pub was_on: bool,
    /// Cumulative runtime hours (for annual validation)
    pub cumulative_runtime_hours: f64,
    /// Startup count (for penalty calculation)
    pub startup_count: u32,
    /// Minimum runtime in timesteps (e.g., 5 minutes = 5 timesteps)
    pub minimum_runtime_timesteps: u32,
    /// Current runtime since last startup (timesteps)
    pub current_runtime_timesteps: u32,
    /// Energy penalty per startup (kWh)
    pub startup_penalty_kwh: f64,
    /// PLR degradation factor (e.g., 0.2 for +20% at 0% PLR)
    pub plr_degradation_factor: f64,
}

impl CyclingTracker {
    /// Create a new cycling tracker with default parameters.
    ///
    /// Defaults per CONTEXT.md (AHRI guidance):
    /// - Minimum runtime: 5 timesteps (5 hours in hourly simulation)
    /// - Startup penalty: 0.1 kWh
    /// - PLR degradation: 0.2 (+20% at 0% PLR)
    pub fn new() -> Self {
        Self {
            was_on: false,
            cumulative_runtime_hours: 0.0,
            startup_count: 0,
            minimum_runtime_timesteps: 5, // 5 hours (AHRI guidance)
            current_runtime_timesteps: 0,
            startup_penalty_kwh: 0.1,    // 0.1 kWh startup energy
            plr_degradation_factor: 0.2, // +20% at 0% PLR
        }
    }

    /// Calculate cycling loss for current timestep.
    ///
    /// Returns efficiency multiplier (1.0 = no degradation, >1.0 = efficiency penalty).
    ///
    /// # Arguments
    /// * `is_on` - Equipment state this timestep (true = on, false = off)
    /// * `plr` - Part-load ratio (0.0 to 1.0)
    ///
    /// # Returns
    /// Tuple of (efficiency_multiplier, startup_penalty_kwh)
    ///
    /// - `efficiency_multiplier`: Multiply power by this to apply PLR degradation
    /// - `startup_penalty_kwh`: Add this to energy consumption for this timestep
    pub fn calculate_cycling_loss(&mut self, is_on: bool, plr: f64) -> (f64, f64) {
        let mut startup_penalty = 0.0;
        let mut efficiency_multiplier = 1.0;

        // Detect startup event
        if is_on && !self.was_on {
            self.startup_count += 1;
            self.current_runtime_timesteps = 0;
            // Apply startup penalty
            startup_penalty = self.startup_penalty_kwh;
        }

        // Update state
        self.was_on = is_on;
        if is_on {
            self.current_runtime_timesteps += 1;
            self.cumulative_runtime_hours += 1.0; // 1 timestep = 1 hour

            // Check minimum runtime constraint
            let must_run = self.current_runtime_timesteps < self.minimum_runtime_timesteps;

            // PLR degradation: efficiency penalty at low PLR
            // Example: At PLR=0.3, degradation=0.2 → multiplier = 1.0 + 0.2 * 0.7 = 1.14
            if !must_run {
                let plr_penalty = self.plr_degradation_factor * (1.0 - plr);
                efficiency_multiplier = 1.0 + plr_penalty;
            }
            // If must_run (within minimum runtime), no PLR degradation
            // (equipment is in startup phase, penalized by startup_penalty_kwh)
        }

        (efficiency_multiplier, startup_penalty)
    }

    /// Reset tracking (for new simulation or year boundary).
    pub fn reset(&mut self) {
        self.was_on = false;
        self.cumulative_runtime_hours = 0.0;
        self.startup_count = 0;
        self.current_runtime_timesteps = 0;
    }

    /// Check if equipment must continue running (minimum runtime constraint).
    pub fn must_run(&self) -> bool {
        self.was_on && self.current_runtime_timesteps <= self.minimum_runtime_timesteps
    }
}

impl Default for CyclingTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycling_losses() {
        let mut tracker = CyclingTracker::new();

        // Initial state
        assert_eq!(tracker.startup_count, 0);
        assert_eq!(tracker.cumulative_runtime_hours, 0.0);

        // First startup
        let (_mult, penalty) = tracker.calculate_cycling_loss(true, 0.5);
        assert_eq!(tracker.startup_count, 1); // Detected startup
        assert_eq!(penalty, 0.1); // Startup penalty 0.1 kWh
        assert_eq!(tracker.current_runtime_timesteps, 1);

        // No startup (still on) - within minimum runtime, no PLR degradation
        let (mult, penalty) = tracker.calculate_cycling_loss(true, 0.5);
        assert_eq!(tracker.startup_count, 1); // No new startup
        assert_eq!(penalty, 0.0); // No penalty
        assert_eq!(mult, 1.0); // No PLR degradation (within minimum runtime)
        assert_eq!(tracker.current_runtime_timesteps, 2);

        // Shutdown
        let (_mult, penalty) = tracker.calculate_cycling_loss(false, 0.0);
        assert!(!tracker.was_on);
        assert_eq!(penalty, 0.0);
        assert_eq!(tracker.current_runtime_timesteps, 2); // Not incremented

        // Second startup
        let (_mult, penalty) = tracker.calculate_cycling_loss(true, 0.5);
        assert_eq!(tracker.startup_count, 2); // New startup detected
        assert_eq!(penalty, 0.1); // Startup penalty again
        assert_eq!(tracker.current_runtime_timesteps, 1); // Reset on startup
    }

    #[test]
    fn test_minimum_runtime_enforcement() {
        let mut tracker = CyclingTracker::new();

        // Startup
        tracker.calculate_cycling_loss(true, 0.5);
        assert_eq!(tracker.current_runtime_timesteps, 1);
        assert!(tracker.must_run()); // Must run for 5 timesteps

        // Within minimum runtime (timesteps 2-5)
        for _ in 2..6 {
            tracker.calculate_cycling_loss(true, 0.5);
            // Runtime is now: 2, 3, 4, 5
            // Minimum runtime is 5, so must_run should be true for runtime <= 5
            assert!(tracker.must_run());
        }

        // After the loop, we're at timestep 5 (within minimum runtime)
        assert_eq!(tracker.current_runtime_timesteps, 5);
        assert!(tracker.must_run()); // Still must run at timestep 5

        // One more timestep to pass minimum runtime
        tracker.calculate_cycling_loss(true, 0.5);
        assert_eq!(tracker.current_runtime_timesteps, 6);
        assert!(!tracker.must_run()); // Can shut down now (past minimum runtime)
    }

    #[test]
    fn test_plr_degradation() {
        let mut tracker = CyclingTracker::new();

        // Startup (no PLR degradation during minimum runtime)
        tracker.calculate_cycling_loss(true, 0.5);
        let (mult, _) = tracker.calculate_cycling_loss(true, 0.5);
        assert_eq!(mult, 1.0); // No degradation (within minimum runtime)

        // After minimum runtime (PLR degradation applies)
        for _ in 0..6 {
            tracker.calculate_cycling_loss(true, 0.5);
        }
        let (mult, _) = tracker.calculate_cycling_loss(true, 0.5);
        assert!(mult > 1.0); // PLR degradation now applies

        // Test different PLR values
        let mut tracker2 = CyclingTracker::new();
        for _ in 0..6 {
            tracker2.calculate_cycling_loss(true, 1.0); // 100% load
        }
        let (mult_100, _) = tracker2.calculate_cycling_loss(true, 1.0);
        assert_eq!(mult_100, 1.0); // No degradation at 100% PLR

        let mut tracker3 = CyclingTracker::new();
        for _ in 0..6 {
            tracker3.calculate_cycling_loss(true, 0.3); // 30% load
        }
        let (mult_30, _) = tracker3.calculate_cycling_loss(true, 0.3);
        let expected_degradation = 1.0 + 0.2 * (1.0 - 0.3); // 1.14
        assert!((mult_30 - expected_degradation).abs() < 0.01);
    }

    #[test]
    fn test_cumulative_runtime_tracking() {
        let mut tracker = CyclingTracker::new();

        // Run for 10 timesteps (10 hours)
        for _ in 0..10 {
            tracker.calculate_cycling_loss(true, 0.5);
        }

        assert_eq!(tracker.cumulative_runtime_hours, 10.0); // 10 hours
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = CyclingTracker::new();

        // Run for 5 timesteps
        for _ in 0..5 {
            tracker.calculate_cycling_loss(true, 0.5);
        }
        assert_eq!(tracker.startup_count, 1);
        assert_eq!(tracker.cumulative_runtime_hours, 5.0); // 5 hours

        // Reset
        tracker.reset();
        assert_eq!(tracker.startup_count, 0);
        assert_eq!(tracker.cumulative_runtime_hours, 0.0);
        assert_eq!(tracker.current_runtime_timesteps, 0);
        assert!(!tracker.was_on);
    }
}
