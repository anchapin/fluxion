//! Comprehensive tests for ventilation and infiltration modeling.
//!
//! This test suite covers:
//! - Constant ventilation schedules
//! - Scheduled ventilation with night cooling
//! - ACH to conductance conversion
//! - Edge cases and boundary conditions

use fluxion::sim::ventilation::{
    ach_to_conductance, ConstantVentilation, ScheduledVentilation, VentilationSchedule,
};

// ============================================================================
// ConstantVentilation Tests
// ============================================================================

#[test]
fn test_constant_ventilation_creation() {
    let vent = ConstantVentilation::new(0.5);
    assert_eq!(vent.ach, 0.5);
}

#[test]
fn test_constant_ventilation_returns_same_ach() {
    let vent = ConstantVentilation::new(0.75);

    // Should return same ACH for any hour
    assert_eq!(vent.get_ach(0), 0.75);
    assert_eq!(vent.get_ach(6), 0.75);
    assert_eq!(vent.get_ach(12), 0.75);
    assert_eq!(vent.get_ach(23), 0.75);
}

#[test]
fn test_constant_ventilation_zero_ach() {
    let vent = ConstantVentilation::new(0.0);
    assert_eq!(vent.get_ach(0), 0.0);
    assert_eq!(vent.get_ach(12), 0.0);
}

#[test]
fn test_constant_ventilation_clone() {
    let vent = ConstantVentilation::new(1.0);
    let cloned = vent.clone_box();

    assert_eq!(cloned.get_ach(5), 1.0);
    assert_eq!(cloned.get_ach(15), 1.0);
}

// ============================================================================
// ScheduledVentilation Tests
// ============================================================================

#[test]
fn test_scheduled_ventilation_creation() {
    let vent = ScheduledVentilation::new(0.3, 2.0);
    assert_eq!(vent.base_ach, 0.3);
    assert_eq!(vent.fan_ach, 2.0);
    // Default schedule should be all false
    for hour in 0..24 {
        assert!(!vent.schedule[hour]);
    }
}

#[test]
fn test_scheduled_ventilation_night_cooling_same_hour() {
    // When start_hour == end_hour, fan should be on all 24 hours
    let vent = ScheduledVentilation::night_ventilation(0.3, 3.0, 20, 20);

    for hour in 0..24 {
        assert!(vent.schedule[hour], "Hour {} should be ON", hour);
        assert_eq!(vent.get_ach(hour), 0.3 + 3.0);
    }
}

#[test]
fn test_scheduled_ventilation_night_cooling_normal_range() {
    // Fan on from hour 22 to hour 6 (overnight)
    let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 22, 6);

    // Hours 22, 23 should be ON
    assert!(vent.schedule[22]);
    assert!(vent.schedule[23]);

    // Hours 0-5 should be ON
    for hour in 0..6 {
        assert!(vent.schedule[hour], "Hour {} should be ON", hour);
    }

    // Hours 6-21 should be OFF
    for hour in 6..22 {
        assert!(!vent.schedule[hour], "Hour {} should be OFF", hour);
    }

    // Verify ACH values
    assert_eq!(vent.get_ach(22), 0.3 + 2.0); // Fan ON
    assert_eq!(vent.get_ach(3), 0.3 + 2.0); // Fan ON
    assert_eq!(vent.get_ach(12), 0.3); // Fan OFF
}

#[test]
fn test_scheduled_ventilation_daytime_only() {
    // Fan on from hour 8 to hour 18 (daytime)
    let vent = ScheduledVentilation::night_ventilation(0.5, 1.5, 8, 18);

    // Hours 8-17 should be ON
    for hour in 8..18 {
        assert!(vent.schedule[hour], "Hour {} should be ON", hour);
    }

    // Hours 0-7 and 18-23 should be OFF
    for hour in 0..8 {
        assert!(!vent.schedule[hour], "Hour {} should be OFF", hour);
    }
    for hour in 18..24 {
        assert!(!vent.schedule[hour], "Hour {} should be OFF", hour);
    }

    // Verify ACH values
    assert_eq!(vent.get_ach(10), 0.5 + 1.5); // Fan ON
    assert_eq!(vent.get_ach(20), 0.5); // Fan OFF
}

#[test]
fn test_scheduled_ventilation_single_hour() {
    // Fan on for just one hour (14 to 15)
    let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 14, 15);

    assert!(vent.schedule[14]);
    assert!(!vent.schedule[13]);
    assert!(!vent.schedule[15]);

    assert_eq!(vent.get_ach(14), 0.3 + 2.0);
    assert_eq!(vent.get_ach(13), 0.3);
}

#[test]
fn test_scheduled_ventilation_clone() {
    let vent = ScheduledVentilation::night_ventilation(0.5, 1.0, 20, 6);
    let cloned = vent.clone_box();

    assert_eq!(cloned.get_ach(22), 0.5 + 1.0);
    assert_eq!(cloned.get_ach(3), 0.5 + 1.0);
    assert_eq!(cloned.get_ach(12), 0.5);
}

#[test]
fn test_scheduled_ventilation_zero_fan_ach() {
    // Fan adds no additional ventilation
    let vent = ScheduledVentilation::night_ventilation(0.5, 0.0, 20, 6);

    for hour in 20..24 {
        assert_eq!(vent.get_ach(hour), 0.5);
    }
    for hour in 0..6 {
        assert_eq!(vent.get_ach(hour), 0.5);
    }
}

// ============================================================================
// ACH to Conductance Conversion Tests
// ============================================================================

#[test]
fn test_ach_to_conductance_basic() {
    // Standard air properties
    let rho = 1.2; // kg/m³
    let cp = 1005.0; // J/kg·K

    // 1 ACH in 100 m³ volume
    let conductance = ach_to_conductance(1.0, 100.0, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();

    // Q = (1 * 100 * 1.2 * 1005) / 3600 = 33.5 W/K
    let expected = (1.0 * 100.0 * 1.2 * 1005.0) / 3600.0;
    assert!((conductance - expected).abs() < 1e-10);
    assert!((conductance - 33.5).abs() < 0.1);
}

#[test]
fn test_ach_to_conductance_zero_ach() {
    let conductance = ach_to_conductance(0.0, 100.0, 1.2, 1005.0)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();
    assert!(conductance.abs() < 1e-10);
}

#[test]
fn test_ach_to_conductance_zero_volume() {
    let conductance = ach_to_conductance(1.0, 0.0, 1.2, 1005.0)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();
    assert!(conductance.abs() < 1e-10);
}

#[test]
fn test_ach_to_conductance_proportional_to_ach() {
    let volume = 100.0;
    let rho = 1.2;
    let cp = 1005.0;

    let c1 = ach_to_conductance(0.5, volume, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();
    let c2 = ach_to_conductance(1.0, volume, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();
    let c3 = ach_to_conductance(2.0, volume, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();

    // Should be linear with ACH
    assert!((c2 - 2.0 * c1).abs() < 1e-10);
    assert!((c3 - 2.0 * c2).abs() < 1e-10);
}

#[test]
fn test_ach_to_conductance_proportional_to_volume() {
    let ach = 1.0;
    let rho = 1.2;
    let cp = 1005.0;

    let c1 = ach_to_conductance(ach, 50.0, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();
    let c2 = ach_to_conductance(ach, 100.0, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();
    let c3 = ach_to_conductance(ach, 200.0, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();

    // Should be linear with volume
    assert!((c2 - 2.0 * c1).abs() < 1e-10);
    assert!((c3 - 2.0 * c2).abs() < 1e-10);
}

#[test]
fn test_ach_to_conductance_typical_values() {
    // Typical residential values
    let volume = 250.0; // m³ (typical house)
    let ach = 0.5; // Typical infiltration rate
    let rho = 1.2;
    let cp = 1005.0;

    let conductance = ach_to_conductance(ach, volume, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();

    // Expected: (0.5 * 250 * 1.2 * 1005) / 3600 ≈ 41.9 W/K
    assert!((conductance - 41.9).abs() < 0.5);
}

#[test]
fn test_ach_to_conductance_large_building() {
    // Large commercial building
    let volume = 10000.0; // m³
    let ach = 1.0; // Higher ventilation rate
    let rho = 1.2;
    let cp = 1005.0;

    let conductance = ach_to_conductance(ach, volume, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();

    // Expected: (1.0 * 10000 * 1.2 * 1005) / 3600 ≈ 3350 W/K
    assert!((conductance - 3350.0).abs() < 10.0);
}

#[test]
fn test_ach_to_conductance_different_air_properties() {
    // Test with different air density (high altitude)
    let volume = 100.0;
    let ach = 1.0;
    let cp = 1005.0;

    let c_sea_level = ach_to_conductance(ach, volume, 1.2, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();
    let c_altitude = ach_to_conductance(ach, volume, 1.0, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();

    // Lower density should give lower conductance
    assert!(c_altitude < c_sea_level);

    // Ratio should match density ratio
    let ratio = c_sea_level / c_altitude;
    assert!((ratio - 1.2).abs() < 0.01);
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn test_ventilation_negative_ach_handling() {
    // Negative ACH doesn't make physical sense, but we should handle it gracefully
    let vent = ConstantVentilation::new(-0.5);
    let ach = vent.get_ach(0);

    // The implementation doesn't clamp to positive, so we just verify it returns the value
    assert_eq!(ach, -0.5);
}

#[test]
fn test_ventilation_very_high_ach() {
    // Very high ventilation rate (industrial)
    let vent = ConstantVentilation::new(50.0);
    assert_eq!(vent.get_ach(0), 50.0);

    let conductance = ach_to_conductance(50.0, 100.0, 1.2, 1005.0)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();
    assert!(conductance > 0.0);
    assert!(conductance < 10000.0); // Should be reasonable
}

#[test]
fn test_scheduled_ventilation_full_day_on() {
    // Explicitly set all hours to ON
    let mut vent = ScheduledVentilation::new(0.3, 1.0);
    vent.schedule = [true; 24];

    for hour in 0..24 {
        assert_eq!(vent.get_ach(hour), 0.3 + 1.0);
    }
}

#[test]
fn test_scheduled_ventilation_full_day_off() {
    // All hours OFF (only base infiltration)
    let vent = ScheduledVentilation::new(0.5, 2.0);
    // Default schedule is all false

    for hour in 0..24 {
        assert_eq!(vent.get_ach(hour), 0.5);
    }
}

#[test]
fn test_ventilation_trait_object_usage() {
    // Test that both types can be used as trait objects
    let schedules: Vec<Box<dyn VentilationSchedule>> = vec![
        Box::new(ConstantVentilation::new(0.5)),
        Box::new(ScheduledVentilation::night_ventilation(0.3, 1.0, 20, 6)),
    ];

    // Should be able to call get_ach on any schedule
    for schedule in &schedules {
        let ach = schedule.get_ach(12);
        assert!(ach >= 0.0);
    }
}

#[test]
fn test_ach_to_conductance_unit_consistency() {
    // Verify units: (1/hr) * m³ * (kg/m³) * (J/kg·K) / (s/hr) = W/K
    // = (1/3600 s) * m³ * kg/m³ * J/kg·K = J/s·K = W/K

    let ach = 1.0; // 1/hr
    let volume = 1.0; // m³
    let rho = 1.0; // kg/m³
    let cp = 3600.0; // J/kg·K (chosen to make math easy)

    let conductance = ach_to_conductance(ach, volume, rho, cp)
        .get::<uom::si::thermal_conductance::watt_per_kelvin>();

    // Expected: (1 * 1 * 1 * 3600) / 3600 = 1.0 W/K
    assert!((conductance - 1.0).abs() < 1e-10);
}

#[test]
fn test_scheduled_ventilation_boundary_hours() {
    // Test ventilation at hour boundaries
    let vent = ScheduledVentilation::night_ventilation(0.3, 1.0, 23, 1);

    // Hour 23 should be ON
    assert!(vent.schedule[23]);
    assert_eq!(vent.get_ach(23), 1.3);

    // Hour 0 should be ON
    assert!(vent.schedule[0]);
    assert_eq!(vent.get_ach(0), 1.3);

    // Hour 1 should be OFF (end_hour is exclusive)
    assert!(!vent.schedule[1]);
    assert_eq!(vent.get_ach(1), 0.3);

    // Hour 22 should be OFF
    assert!(!vent.schedule[22]);
    assert_eq!(vent.get_ach(22), 0.3);
}
