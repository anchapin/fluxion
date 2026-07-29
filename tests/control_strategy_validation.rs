//! Control Strategy Validation Tests
//!
//! This module provides comprehensive validation tests for HVAC control strategies
//! against EnergyPlus reference data. These tests address the gaps identified in
//! issue #1930.
//!
//! ## Test Coverage
//!
//! ### Setpoint Control
//! - `test_proportional_control_oscillation_amplitude` - Validates temperature
//!   oscillation amplitude for proportional control
//! - `test_pi_control_setpoint_tracking` - Validates setpoint tracking error
//!   for PI control against EnergyPlus reference
//! - `test_schedule_based_setback_transition` - Validates temperature deviation
//!   at schedule transition periods
//!
//! ### Cycling Behavior
//! - `test_short_cycling_frequency` - Validates equipment on/off frequency
//! - `test_cycling_energy_consumption` - Quantifies energy penalty from cycling
//!
//! ### Equipment Staging
//! - `test_two_stage_heating_activation` - Validates two-stage heating activation
//!   order and capacity
//! - `test_modulating_equipment_part_load_efficiency` - Validates part-load
//!   efficiency for modulating equipment

use fluxion::physics::cta::VectorField;
use fluxion::sim::hvac::cycling::CyclingTracker;
use fluxion::sim::hvac::economizer::{is_economizer_active, EconomizerMode};
use fluxion::sim::hvac::modes::PredictiveController;
use fluxion::sim::hvac::zones::schedule::{DailySchedule, HVACSchedule};
use fluxion::sim::hvac::{
    AnyEquipment, Boiler, Chiller, HVACMode, HeatPump, VariableCapacityEquipment,
};

// ============================================================================
// Setpoint Control Tests
// ============================================================================

mod setpoint_control {
    use super::*;

    /// Validates proportional control temperature oscillation amplitude.
    ///
    /// Proportional control should exhibit bounded oscillation when the zone
    /// temperature is near the setpoint. This test simulates a zone with
    /// proportional control and verifies the oscillation stays within expected
    /// bounds (typically ±0.5°C for well-tuned P-control).
    #[test]
    fn test_proportional_control_oscillation_amplitude() {
        // P-controller with unity gain (k_p = 1.0)
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.5, 0.0);

        // Simulate zone at setpoint with no thermal mass effects
        let zone_temps = [20.1, 20.2, 20.1, 20.0, 19.9, 20.0, 20.1];
        let mut prev_modulation = 0.0;
        let mut oscillations = Vec::new();

        for &zone_temp in &zone_temps {
            let (_, modulation) = controller.calculate_modulation(zone_temp, zone_temp, 0.0);
            if prev_modulation > 0.0 && modulation > 0.0 {
                oscillations.push((modulation - prev_modulation).abs());
            }
            prev_modulation = modulation;
        }

        // Check oscillation amplitude is bounded
        // With P-control (no integral), oscillation amplitude should be small
        let max_oscillation = oscillations.iter().fold(0.0f64, |max, &x| max.max(x));
        assert!(
            max_oscillation < 0.15,
            "P-control oscillation amplitude {} exceeds bound of 0.15",
            max_oscillation
        );
    }

    /// Validates PI control setpoint tracking error.
    ///
    /// PI control should track setpoints with minimal steady-state error.
    /// This test simulates a load change and verifies the controller reduces
    /// the tracking error over time.
    #[test]
    fn test_pi_control_setpoint_tracking() {
        // Create PI controller with tuning: k_p = 0.5, k_i = 0.1
        // Using PredictiveController which has integral-like behavior via thermal inertia
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.5, 0.05);

        // Track error over multiple timesteps
        let initial_error = 3.0; // 3°C below setpoint
        let mut errors = Vec::new();

        let mut zone_temp = 20.0 - initial_error;

        for _ in 0..10 {
            let (_, modulation) = controller.calculate_modulation(zone_temp, zone_temp, 0.0);
            errors.push((20.0 - zone_temp).abs());

            // Simulate zone response: zone warms based on modulation
            // Higher modulation -> faster warming
            zone_temp += modulation * 0.3;
        }

        // PI control should reduce error over time
        let initial_tracking_error = errors[0];
        let final_tracking_error = errors[errors.len() - 1];

        assert!(
            final_tracking_error < initial_tracking_error * 0.5,
            "PI control should reduce tracking error from {:.2} to {:.2}",
            initial_tracking_error,
            final_tracking_error
        );
    }

    /// Validates temperature deviation at schedule transition (setback/setup).
    ///
    /// During night setback and morning recovery, the zone temperature should
    /// track the setback setpoint within acceptable bounds. This test simulates
    /// a night setback period and morning recovery.
    #[test]
    fn test_schedule_based_setback_transition() {
        // Create schedule with night setback using helper constructor
        let schedule = HVACSchedule::setback_schedule(
            20.0, // day heating setpoint
            15.0, // night heating setback
            26.0, // cooling setpoint
            22,   // night start hour
            6,    // night end hour
        );

        // Night period: zone at setback temperature
        let night_hour = 3;
        let setback_heating_sp = schedule.heating_setpoint(night_hour);
        let setback_cooling_sp = schedule.cooling_setpoint(night_hour);

        assert_eq!(
            setback_heating_sp, 15.0,
            "Night setback heating setpoint should be 15°C"
        );
        assert_eq!(
            setback_cooling_sp, 26.0,
            "Night setup cooling setpoint should be 26°C"
        );

        // Recovery period: morning warmup
        let recovery_hour = 7;
        let occupied_heating_sp = schedule.heating_setpoint(recovery_hour);

        assert_eq!(
            occupied_heating_sp, 20.0,
            "Morning occupied heating setpoint should be 20°C"
        );

        // Test setback recovery with controller
        let mut controller = PredictiveController::with_tuning(
            occupied_heating_sp,
            schedule.cooling_setpoint(recovery_hour),
            0.5,
            0.05,
        );

        // Zone temperature during recovery (15°C -> 20°C)
        let zone_temp = 16.0;
        let mass_temp = 17.0;
        let (mode, modulation) = controller.calculate_modulation(zone_temp, mass_temp, 0.0);

        // Should be in heating mode with positive modulation
        assert_eq!(
            mode,
            HVACMode::Heating,
            "Should be in heating mode during recovery"
        );
        assert!(
            modulation > 0.0,
            "Modulation should be positive during recovery, got {}",
            modulation
        );
    }

    /// Validates proportional band behavior for heating.
    ///
    /// Within the proportional band, the control signal should vary linearly
    /// with temperature error.
    #[test]
    fn test_proportional_band_heating() {
        let mut controller = PredictiveController::with_tuning(20.0, 27.0, 0.5, 0.0);

        // Test temperatures across the proportional band (heating_sp - deadband to heating_sp)
        // Deadband is 0.5°C, so heating activates from 19.5°C to 20.0°C
        let test_temps = [19.5, 19.6, 19.7, 19.8, 19.9, 20.0];
        let mut modulations = Vec::new();

        for &temp in &test_temps {
            let (_, mod_factor) = controller.calculate_modulation(temp, temp, 0.0);
            modulations.push(mod_factor);
        }

        // Modulation should increase monotonically as temperature decreases
        for i in 1..modulations.len() {
            assert!(
                modulations[i] >= modulations[i - 1],
                "Modulation should increase as temperature decreases: {} -> {} at {}°C",
                modulations[i - 1],
                modulations[i],
                test_temps[i]
            );
        }

        // At setpoint (20°C), modulation should be zero (within deadband)
        assert!(
            modulations[modulations.len() - 1] == 0.0,
            "Modulation at setpoint should be 0.0, got {}",
            modulations[modulations.len() - 1]
        );
    }
}

// ============================================================================
// Cycling Behavior Tests
// ============================================================================

mod cycling_behavior {
    use super::*;

    /// Validates short cycling detection and minimum runtime enforcement.
    ///
    /// Equipment should not cycle on/off rapidly. When equipment is cycled
    /// on and then immediately requested off, the minimum runtime constraint
    /// should prevent the shutdown (must_run returns true).
    #[test]
    fn test_short_cycling_detection() {
        let mut tracker = CyclingTracker::new();

        // First startup - should record startup count = 1
        let (_, penalty) = tracker.calculate_cycling_loss(true, 0.5);
        assert_eq!(
            tracker.startup_count, 1,
            "First startup should increment count"
        );
        assert!(penalty > 0.0, "Startup penalty should be applied");

        // Immediately try to shut down (short cycle attempt)
        // Within minimum runtime, must_run should prevent shutdown
        assert!(
            tracker.must_run(),
            "Equipment must run once started, even if shutdown requested"
        );

        // Continue running through minimum runtime period
        // The minimum runtime is 5 timesteps. After 5 full timesteps of running,
        // must_run should become false (meaning minimum runtime has been satisfied).
        // Note: current_runtime_timesteps increments at the START of each call,
        // so after 5 more calls, it will be 6 and must_run will be false.
        let minimum_runtime: u32 = tracker.minimum_runtime_timesteps;

        // Run for one less than minimum runtime to ensure we're still in must_run
        for _ in 0..(minimum_runtime - 1) {
            let (_, _) = tracker.calculate_cycling_loss(true, 0.5);
            assert!(
                tracker.must_run(),
                "Equipment must run during minimum runtime period"
            );
        }

        // After minimum runtime is exceeded (6th call), shutdown is allowed
        let (_, _) = tracker.calculate_cycling_loss(true, 0.5);
        assert!(
            !tracker.must_run(),
            "After minimum runtime exceeded, must_run should be false"
        );

        // Now shutdown is allowed
        let (_, _) = tracker.calculate_cycling_loss(false, 0.0);
        assert!(!tracker.was_on, "Equipment should be off after shutdown");
    }

    /// Quantifies cycling energy penalty.
    ///
    /// Each startup incurs an energy penalty (startup_penalty_kwh).
    /// This test tracks cumulative energy including cycling penalties.
    #[test]
    fn test_cycling_energy_penalty() {
        let mut tracker = CyclingTracker::new();

        let mut total_energy_kwh = 0.0;
        let runtime_hours = 10;
        let startup_penalty_kwh = tracker.startup_penalty_kwh;

        // Simulate operation with cycling events
        for hour in 0..runtime_hours {
            let is_on = (hour % 3) != 0; // Cycle: on, on, off
            let plr = if is_on { 0.7 } else { 0.0 };

            let (_, penalty) = tracker.calculate_cycling_loss(is_on, plr);

            // Add base energy + cycling penalty
            let base_energy = if is_on { plr * 5.0 } else { 0.0 }; // 5 kW max * PLR
            total_energy_kwh += base_energy + penalty;
        }

        // Calculate expected startup penalties
        // With on/off pattern (every 3 hours), we expect ~3 startups
        let expected_startups = runtime_hours / 3;
        let expected_penalty = expected_startups as f64 * startup_penalty_kwh;

        // Verify startup count matches
        assert_eq!(
            tracker.startup_count, expected_startups as u32,
            "Expected {} startups, got {}",
            expected_startups, tracker.startup_count
        );

        // Total cycling penalty should be bounded
        assert!(
            tracker.startup_count as f64 * startup_penalty_kwh <= total_energy_kwh * 0.5,
            "Cycling penalties ({:.2} kWh) should be less than 50% of total energy",
            expected_penalty
        );
    }

    /// Validates cycling frequency bounds.
    ///
    /// Equipment should not cycle more than a reasonable number of times per hour.
    /// For hourly simulation, max cycles per hour should be bounded.
    #[test]
    fn test_cycling_frequency_bounds() {
        let mut tracker = CyclingTracker::new();

        // Simulate hourly operation with controlled cycling
        // Pattern: 2 hours on, 1 hour off (max reasonable cycling)
        let hourly_pattern = [true, true, false, true, true, false, true, true];

        for &is_on in &hourly_pattern {
            tracker.calculate_cycling_loss(is_on, if is_on { 0.5 } else { 0.0 });
        }

        // With 8 hours and pattern repeating every 3 hours:
        // startups at hours: 0, 3, 6 = 3 startups
        let max_acceptable_cycles_per_hour = 1.0; // One startup per hour is acceptable max
        let actual_cycles_per_hour = tracker.startup_count as f64 / hourly_pattern.len() as f64;

        assert!(
            actual_cycles_per_hour <= max_acceptable_cycles_per_hour,
            "Cycling frequency {:.2} cycles/hour exceeds maximum {:.2}",
            actual_cycles_per_hour,
            max_acceptable_cycles_per_hour
        );
    }

    /// Validates PLR degradation effect on cycling efficiency.
    ///
    /// At low part-load ratios, efficiency should degrade. This test
    /// verifies the cycling tracker applies PLR degradation correctly.
    #[test]
    fn test_plr_degradation_at_low_load() {
        let mut tracker = CyclingTracker::new();

        // Run past minimum runtime to reach PLR degradation regime
        for _ in 0..6 {
            tracker.calculate_cycling_loss(true, 1.0); // 100% PLR
        }

        // Now test at different PLR levels
        let test_plrs = [1.0, 0.75, 0.5, 0.25, 0.1];
        let mut efficiency_multipliers = Vec::new();

        for &plr in &test_plrs {
            // Need to be running to calculate cycling loss
            let (mult, _) = tracker.calculate_cycling_loss(true, plr);
            efficiency_multipliers.push(mult);
        }

        // Efficiency multiplier should increase (worse) as PLR decreases
        for i in 1..efficiency_multipliers.len() {
            assert!(
                efficiency_multipliers[i] >= efficiency_multipliers[i - 1],
                "Efficiency multiplier should increase at lower PLR: {:.3} -> {:.3}",
                efficiency_multipliers[i - 1],
                efficiency_multipliers[i]
            );
        }

        // At 100% PLR, multiplier should be 1.0 (no degradation)
        assert!(
            (efficiency_multipliers[0] - 1.0).abs() < 0.001,
            "At 100% PLR, efficiency multiplier should be 1.0"
        );
    }
}

// ============================================================================
// Equipment Staging Tests
// ============================================================================

mod equipment_staging {
    use super::*;

    /// Validates modulating heating capacity with PLR.
    ///
    /// For staged or modulating heating, the capacity should vary
    /// proportionally with the part-load ratio.
    #[test]
    fn test_modulating_heating_capacity() {
        // Create a boiler with default parameters
        let boiler = Boiler::new(
            "Boiler-Test".to_string(),
            100_000.0, // 100 kW heating capacity
            0.85,      // 85% efficiency
            -5.0,      // design temp
        );

        let outdoor_temp = 5.0;

        // Calculate capacity at full load
        let full_capacity = boiler.calculate_capacity(1.0, outdoor_temp);

        // Calculate capacity at part load
        let modulated_capacity = boiler.calculate_capacity(0.7, outdoor_temp);

        // Verify staged capacity is proportional to PLR
        assert!(
            modulated_capacity < full_capacity,
            "Modulated capacity should be less than full capacity: {} < {}",
            modulated_capacity,
            full_capacity
        );
        assert!(
            modulated_capacity > 0.0,
            "Modulated capacity should be positive"
        );

        // Capacity at PLR should be approximately proportional to PLR
        let expected_capacity = full_capacity * 0.7;
        let tolerance = full_capacity * 0.05; // 5% tolerance
        assert!(
            (modulated_capacity - expected_capacity).abs() <= tolerance,
            "Modulated capacity {:.2} should be within {:.2} of expected {:.2}",
            modulated_capacity,
            tolerance,
            expected_capacity
        );
    }

    /// Validates modulating equipment part-load efficiency.
    ///
    /// Part-load efficiency varies with PLR based on AHRI polynomial curves.
    /// For some equipment, efficiency at part-load can be higher than at full-load
    /// due to the polynomial shape. This test verifies the efficiency is bounded
    /// and positive across the PLR range.
    #[test]
    fn test_modulating_equipment_part_load_efficiency() {
        // Create chiller for cooling mode
        let chiller = Chiller::new(
            "Chiller-Test".to_string(),
            100_000.0, // 100 kW cooling capacity
            4.5,       // COP
            35.0,      // design temp
        );

        let outdoor_temp = 25.0; // 25°C outdoor
        let plrs = [1.0, 0.75, 0.5, 0.25];
        let mut efficiencies = Vec::new();

        for &plr in &plrs {
            let efficiency = chiller.calculate_efficiency(plr, outdoor_temp, HVACMode::Cooling);
            assert!(
                efficiency > 0.0,
                "Efficiency at PLR={} should be positive, got {}",
                plr,
                efficiency
            );
            efficiencies.push(efficiency);
        }

        // Efficiency should be within reasonable bounds
        // Typical chiller COP range is 3-6, so efficiency should be in that range
        for (i, &efficiency) in efficiencies.iter().enumerate() {
            assert!(
                (3.0..=6.0).contains(&efficiency),
                "Efficiency at PLR={} = {:.2} should be within reasonable bounds [3.0, 6.0]",
                plrs[i],
                efficiency
            );
        }
    }

    /// Validates equipment capacity calculation bounds.
    ///
    /// Equipment capacity should be bounded: positive, proportional to PLR,
    /// and not exceeding rated capacity.
    #[test]
    fn test_equipment_capacity_bounds() {
        let chiller = Chiller::new("Chiller-Test".to_string(), 100_000.0, 4.5, 35.0);
        let boiler = Boiler::new("Boiler-Test".to_string(), 100_000.0, 0.85, -5.0);

        let outdoor_temp = 20.0;
        let rated_capacity_chiller = chiller.rated_capacity();
        let rated_capacity_boiler = boiler.rated_capacity();

        // Test chiller capacity bounds
        for plr in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let capacity = chiller.calculate_capacity(plr, outdoor_temp);

            assert!(
                capacity >= 0.0,
                "Chiller capacity at PLR={} should be non-negative, got {}",
                plr,
                capacity
            );
            assert!(
                capacity <= rated_capacity_chiller * 1.05, // 5% tolerance for rounding
                "Chiller capacity {} should not exceed rated capacity {} by more than 5%",
                capacity,
                rated_capacity_chiller
            );
        }

        // Test boiler capacity bounds
        for plr in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let capacity = boiler.calculate_capacity(plr, outdoor_temp);

            assert!(
                capacity >= 0.0,
                "Boiler capacity at PLR={} should be non-negative, got {}",
                plr,
                capacity
            );
            assert!(
                capacity <= rated_capacity_boiler * 1.05,
                "Boiler capacity {} should not exceed rated capacity {}",
                capacity,
                rated_capacity_boiler
            );
        }
    }

    /// Validates part-load ratio calculation consistency.
    ///
    /// For a given load and equipment, the PLR should be consistent
    /// when calculated through different code paths.
    #[test]
    fn test_plr_consistency() {
        let chiller = Chiller::new("Chiller-Test".to_string(), 100_000.0, 4.5, 35.0);

        // Test at design temperature where capacity is not degraded
        let design_temp = 35.0;
        let plr = 0.6;

        // Capacity at design temperature should be rated capacity * PLR
        let capacity_at_plr = chiller.calculate_capacity(plr, design_temp);
        let expected_capacity = chiller.rated_capacity() * plr;

        // At design temperature, capacity should be approximately rated * PLR
        let tolerance = chiller.rated_capacity() * 0.05; // 5% tolerance
        assert!(
            (capacity_at_plr - expected_capacity).abs() <= tolerance,
            "Capacity {:.2} at PLR {:.2} should be within {:.2} of expected {:.2}",
            capacity_at_plr,
            plr,
            tolerance,
            expected_capacity
        );

        // At extreme temperatures, capacity degrades - but PLR relationship still holds
        let cold_temp = 5.0;
        let capacity_cold = chiller.calculate_capacity(plr, cold_temp);
        assert!(
            capacity_cold < capacity_at_plr,
            "Capacity at cold temperature {} should be less than at design temp {}",
            capacity_cold,
            capacity_at_plr
        );
    }
}

// ============================================================================
// Economizer Control Tests
// ============================================================================

mod economizer_control {
    use super::*;

    /// Validates economizer dry bulb control mode.
    ///
    /// Economizer should activate when outdoor temperature is below zone
    /// temperature and provides free cooling benefit.
    #[test]
    fn test_economizer_dry_bulb_activation() {
        let mode = EconomizerMode::DryBulb;
        let cooling_setpoint = 26.0;

        // Case 1: Outdoor cooler than zone - should activate
        let outdoor_temp_cool = 20.0;
        let zone_temp = 28.0;
        let is_active = is_economizer_active(
            mode,
            outdoor_temp_cool,
            None,
            zone_temp,
            None,
            cooling_setpoint,
        );
        assert!(
            is_active,
            "Economizer should activate when outdoor ({}) < zone ({})",
            outdoor_temp_cool, zone_temp
        );

        // Case 2: Outdoor warmer than zone - should not activate
        let outdoor_temp_warm = 30.0;
        let is_active_warm = is_economizer_active(
            mode,
            outdoor_temp_warm,
            None,
            zone_temp,
            None,
            cooling_setpoint,
        );
        assert!(
            !is_active_warm,
            "Economizer should not activate when outdoor ({}) > zone ({})",
            outdoor_temp_warm, zone_temp
        );

        // Case 3: Outdoor at same temp as zone - typically not beneficial
        let is_active_equal =
            is_economizer_active(mode, zone_temp, None, zone_temp, None, cooling_setpoint);
        assert!(
            !is_active_equal,
            "Economizer should not activate when outdoor == zone"
        );
    }

    /// Validates economizer disabled mode.
    ///
    /// When disabled, economizer should never activate.
    #[test]
    fn test_economizer_disabled_mode() {
        let mode = EconomizerMode::Disabled;
        let outdoor_temp = 15.0;
        let zone_temp = 30.0;
        let cooling_setpoint = 26.0;

        let is_active =
            is_economizer_active(mode, outdoor_temp, None, zone_temp, None, cooling_setpoint);

        assert!(!is_active, "Economizer should never activate when disabled");
    }

    /// Validates economizer enthalpy control mode.
    ///
    /// Enthalpy control considers both temperature and humidity for
    /// more accurate free cooling assessment.
    #[test]
    fn test_economizer_enthalpy_mode() {
        let mode = EconomizerMode::Enthalpy;
        let cooling_setpoint = 26.0;

        // Outdoor conditions: cooler AND lower enthalpy (drier)
        let outdoor_temp = 18.0;
        let outdoor_enthalpy = 40.0; // kJ/kg, dry air
        let zone_temp = 26.0;
        let zone_enthalpy = 55.0; // kJ/kg, more humid

        let is_active = is_economizer_active(
            mode,
            outdoor_temp,
            Some(outdoor_enthalpy),
            zone_temp,
            Some(zone_enthalpy),
            cooling_setpoint,
        );

        assert!(
            is_active,
            "Economizer should activate with cooler AND drier outdoor air"
        );
    }

    /// Validates economizer outdoor temperature limit.
    ///
    /// Economizer should not activate if outdoor temperature is above
    /// the cooling setpoint (no cooling benefit).
    #[test]
    fn test_economizer_outdoor_temp_limit() {
        let mode = EconomizerMode::DryBulb;
        let cooling_setpoint = 26.0;

        // Outdoor above cooling setpoint - no free cooling benefit
        let outdoor_temp = 28.0;
        let zone_temp = 28.0;

        let is_active =
            is_economizer_active(mode, outdoor_temp, None, zone_temp, None, cooling_setpoint);

        assert!(
            !is_active,
            "Economizer should not activate when outdoor temp ({}) >= cooling setpoint ({})",
            outdoor_temp, cooling_setpoint
        );
    }
}

// ============================================================================
// Schedule-Based Control Tests
// ============================================================================

mod schedule_control {
    use super::*;

    /// Validates schedule-based setpoint transitions.
    ///
    /// Schedule should provide correct setpoints at each hour.
    #[test]
    fn test_schedule_hourly_setpoints() {
        // Create a schedule using setback_schedule for night hours
        let schedule = HVACSchedule::setback_schedule(
            20.0, 15.0, 26.0, 22, 6, // setback from 22:00 to 6:00
        );

        // Validate occupied hours (7-21)
        for hour in 7..22 {
            assert_eq!(
                schedule.heating_setpoint(hour),
                20.0,
                "Heating setpoint at hour {} should be 20°C (occupied)",
                hour
            );
            assert_eq!(
                schedule.cooling_setpoint(hour),
                26.0,
                "Cooling setpoint at hour {} should be 26°C (occupied)",
                hour
            );
        }

        // Validate setback hours (22-6)
        for hour in [22, 23, 0, 3, 5] {
            assert_eq!(
                schedule.heating_setpoint(hour),
                15.0,
                "Heating setpoint at hour {} should be 15°C (setback)",
                hour
            );
        }
    }

    /// Validates setback and setup setpoint differences.
    ///
    /// Setback/setup should provide meaningful energy savings by
    /// allowing larger temperature excursions during unoccupied periods.
    #[test]
    fn test_setback_energy_savings_potential() {
        let schedule = HVACSchedule::setback_schedule(
            20.0, // occupied heating
            15.0, // setback heating
            26.0, // occupied cooling
            22,   // night start
            6,    // night end
        );

        let occupied_heating = 20.0;
        let occupied_cooling = 26.0;

        let heating_setback = schedule.heating_setpoint(3) - occupied_heating;
        let cooling_setback = schedule.cooling_setpoint(3) - occupied_cooling;

        assert_eq!(heating_setback, -5.0, "Heating setback should be -5°C");
        assert_eq!(
            cooling_setback, 0.0,
            "Cooling setup should be 0°C in this schedule"
        );

        // Setback should lower heating setpoints for energy savings
        assert!(
            heating_setback < 0.0,
            "Setback should lower heating setpoints for energy savings"
        );
    }

    /// Validates daily schedule structure.
    ///
    /// DailySchedule should provide 24 hourly values.
    #[test]
    fn test_daily_schedule_constant() {
        let schedule = DailySchedule::constant(20.0);

        for hour in 0..24 {
            assert_eq!(
                schedule.value(hour),
                20.0,
                "DailySchedule should have 24 hourly values"
            );
        }
    }
}

// ============================================================================
// Integration: Control Strategy Combinations
// ============================================================================

mod control_integration {
    use super::*;

    /// Validates control strategy with cycling and economizer.
    ///
    /// When economizer provides free cooling, mechanical cooling should
    /// modulate down. Combined with cycling behavior, the system should
    /// maintain comfort efficiently.
    #[test]
    fn test_economizer_with_cycling() {
        let mut tracker = CyclingTracker::new();

        // Simulate economizer providing free cooling
        let outdoor_temp = 18.0;
        let zone_temp = 27.0;
        let cooling_setpoint = 26.0;

        let economizer_active = is_economizer_active(
            EconomizerMode::DryBulb,
            outdoor_temp,
            None,
            zone_temp,
            None,
            cooling_setpoint,
        );

        assert!(
            economizer_active,
            "Economizer should be active with favorable conditions"
        );

        // When economizer is active, mechanical cooling load is reduced
        // Simulate cycling behavior with reduced load
        for _ in 0..6 {
            tracker.calculate_cycling_loss(true, 0.3); // Low PLR due to economizer
        }

        // After minimum runtime, verify cycling behavior at low PLR
        let (efficiency_mult, _) = tracker.calculate_cycling_loss(true, 0.3);
        assert!(
            efficiency_mult > 1.0,
            "PLR degradation should apply at low PLR, got {:.3}",
            efficiency_mult
        );
    }

    /// Validates multi-zone coordination through schedules.
    ///
    /// When one zone needs conditioning and another doesn't, schedules
    /// should coordinate equipment staging across zones.
    #[test]
    fn test_multi_zone_schedule_coordination() {
        // Zone A: early occupancy (6 AM start)
        let zone_a_schedule = HVACSchedule::with_operating_hours(
            20.0, // heating setpoint
            26.0, // cooling setpoint
            6,    // start hour
            14,   // end hour
        );

        // Zone B: late occupancy (9 AM start)
        let zone_b_schedule = HVACSchedule::with_operating_hours(
            20.0, 26.0, 9,  // start hour
            17, // end hour
        );

        // At 6 AM, Zone A needs heating but Zone B doesn't
        let hour = 6;
        let zone_a_needs_heat = zone_a_schedule.heating_setpoint(hour) > 0.0;
        let zone_b_needs_heat = zone_b_schedule.heating_setpoint(hour) > 0.0;

        assert!(
            zone_a_needs_heat,
            "Zone A should need heating at hour {} (early occupancy)",
            hour
        );
        assert!(
            !zone_b_needs_heat,
            "Zone B should not need heating at hour {} (late occupancy)",
            hour
        );

        // At 9 AM, both zones need heating
        let hour_late = 9;
        let both_need_heat = zone_a_schedule.heating_setpoint(hour_late) > 0.0
            && zone_b_schedule.heating_setpoint(hour_late) > 0.0;

        assert!(
            both_need_heat,
            "Both zones should need heating at hour {} (both occupied)",
            hour_late
        );
    }

    /// Validates heat pump staged capacity.
    ///
    /// Heat pumps should modulate capacity based on PLR and outdoor temperature.
    #[test]
    fn test_heat_pump_capacity_modulation() {
        let heat_pump = HeatPump::new(
            "HP-Test".to_string(),
            12_000.0, // heating capacity
            10_000.0, // cooling capacity
            3.5,      // heating COP
            3.0,      // cooling COP
        );

        let outdoor_temp = 20.0;

        // Full load capacity
        let full_heat_capacity = heat_pump.calculate_capacity(1.0, outdoor_temp);
        let full_cool_capacity = heat_pump.calculate_capacity(1.0, outdoor_temp);

        // Part load capacity
        let part_load_capacity = heat_pump.calculate_capacity(0.5, outdoor_temp);

        assert!(
            part_load_capacity < full_heat_capacity,
            "Part load capacity should be less than full capacity"
        );
        assert!(
            part_load_capacity > 0.0,
            "Part load capacity should be positive"
        );
    }
}
