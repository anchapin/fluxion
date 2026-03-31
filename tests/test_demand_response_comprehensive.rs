//! Comprehensive demand response tests for building energy simulation.
//!
//! This module provides extensive test coverage for the demand response module,
//! including DR events, load shedding, real-time pricing, and the DR manager.

use fluxion::sim::demand_response::*;

// ============================================================================
// DR Event Tests
// ============================================================================

mod dr_event_tests {
    use super::*;

    #[test]
    fn test_dr_event_creation() {
        let event = DREvent::new(
            "DR-001".to_string(),
            DREventType::PeakShaving,
            14,    // 2pm
            4,     // 4 hours
            100.0, // 100 kW target
        );

        assert_eq!(event.id, "DR-001");
        assert_eq!(event.event_type, DREventType::PeakShaving);
        assert_eq!(event.start_hour, 14);
        assert_eq!(event.duration_hours, 4);
        assert_eq!(event.target_reduction, 100.0);
        assert_eq!(event.source, DRSignalSource::Manual);
        assert!(!event.is_active);
        assert_eq!(event.achieved_reduction, 0.0);
    }

    #[test]
    fn test_dr_event_min_reduction_calculation() {
        let event = DREvent::new("DR-002".to_string(), DREventType::Emergency, 10, 2, 50.0);

        // min_reduction should be 80% of target
        assert_eq!(event.min_reduction, 40.0);
    }

    #[test]
    fn test_dr_event_is_active_at_during_event() {
        let event = DREvent::new("DR-003".to_string(), DREventType::PeakShaving, 10, 4, 50.0);

        // Event runs from hour 10 to hour 14
        assert!(event.is_active_at(10), "Should be active at start hour");
        assert!(event.is_active_at(11), "Should be active during event");
        assert!(event.is_active_at(13), "Should be active at last hour");
        assert!(
            !event.is_active_at(14),
            "Should not be active at end hour (exclusive)"
        );
    }

    #[test]
    fn test_dr_event_is_active_at_before_event() {
        let event = DREvent::new("DR-004".to_string(), DREventType::PeakShaving, 14, 4, 50.0);

        assert!(!event.is_active_at(0), "Should not be active before event");
        assert!(!event.is_active_at(10), "Should not be active before event");
        assert!(!event.is_active_at(13), "Should not be active before event");
    }

    #[test]
    fn test_dr_event_is_active_at_after_event() {
        let event = DREvent::new("DR-005".to_string(), DREventType::PeakShaving, 10, 4, 50.0);

        assert!(!event.is_active_at(14), "Should not be active after event");
        assert!(!event.is_active_at(20), "Should not be active after event");
        assert!(!event.is_active_at(167), "Should not be active after event");
    }

    #[test]
    fn test_dr_event_start_and_end() {
        let mut event = DREvent::new("DR-006".to_string(), DREventType::Emergency, 10, 2, 50.0);

        assert!(!event.is_active);

        event.start();
        assert!(event.is_active);

        event.end();
        assert!(!event.is_active);
    }

    #[test]
    fn test_dr_event_target_met() {
        let mut event = DREvent::new("DR-007".to_string(), DREventType::PeakShaving, 10, 2, 100.0);

        // Target is 100 kW, min is 80 kW
        event.achieved_reduction = 0.0;
        assert!(!event.target_met());

        event.achieved_reduction = 79.0;
        assert!(!event.target_met());

        event.achieved_reduction = 80.0;
        assert!(event.target_met());

        event.achieved_reduction = 100.0;
        assert!(event.target_met());

        event.achieved_reduction = 120.0;
        assert!(event.target_met());
    }

    #[test]
    fn test_dr_event_all_event_types() {
        // Test all event types can be created
        let types = vec![
            DREventType::Emergency,
            DREventType::Voluntary,
            DREventType::AncillaryServices,
            DREventType::PeakShaving,
            DREventType::LoadShifting,
        ];

        for (i, event_type) in types.iter().enumerate() {
            let event = DREvent::new(format!("DR-{}", i), *event_type, 10, 2, 50.0);
            assert_eq!(event.event_type, *event_type);
        }
    }

    #[test]
    fn test_dr_event_all_signal_sources() {
        // Test all signal sources
        let sources = vec![
            DRSignalSource::Manual,
            DRSignalSource::OpenADR,
            DRSignalSource::RealTimePricing,
            DRSignalSource::TimeOfUse,
        ];

        for source in sources {
            let mut event =
                DREvent::new("DR-SRC".to_string(), DREventType::PeakShaving, 10, 2, 50.0);
            event.source = source;
            assert_eq!(event.source, source);
        }
    }

    #[test]
    fn test_dr_event_weekly_boundary() {
        // Event that spans end of week
        let event = DREvent::new(
            "DR-WEEK".to_string(),
            DREventType::PeakShaving,
            160,
            10,
            50.0,
        );

        // Event runs from hour 160 to 170, but week only has 168 hours
        assert!(event.is_active_at(160));
        assert!(event.is_active_at(167));
        // Hours 168-169 are outside the week but still "active" per the simple check
        assert!(event.is_active_at(168));
        assert!(!event.is_active_at(170));
    }
}

// ============================================================================
// Load Shedding Controller Tests
// ============================================================================

mod load_shedding_tests {
    use super::*;

    fn create_test_loads() -> Vec<Load> {
        vec![
            Load {
                id: "L1".to_string(),
                name: "HVAC".to_string(),
                power_kw: 50.0,
                priority: LoadPriority::Low,
                can_shed: true,
                is_shed: false,
            },
            Load {
                id: "L2".to_string(),
                name: "Lighting".to_string(),
                power_kw: 30.0,
                priority: LoadPriority::Medium,
                can_shed: true,
                is_shed: false,
            },
            Load {
                id: "L3".to_string(),
                name: "Receptacles".to_string(),
                power_kw: 20.0,
                priority: LoadPriority::High,
                can_shed: true,
                is_shed: false,
            },
            Load {
                id: "L4".to_string(),
                name: "Critical IT".to_string(),
                power_kw: 10.0,
                priority: LoadPriority::Critical,
                can_shed: false,
                is_shed: false,
            },
        ]
    }

    #[test]
    fn test_load_shedding_controller_creation() {
        let controller = LoadSheddingController::new();

        assert!(controller.loads.is_empty());
        assert_eq!(controller.total_shedding_capacity, 0.0);
        assert_eq!(controller.current_shed_load, 0.0);
    }

    #[test]
    fn test_add_load() {
        let mut controller = LoadSheddingController::new();

        let load = Load {
            id: "L1".to_string(),
            name: "HVAC".to_string(),
            power_kw: 50.0,
            priority: LoadPriority::Low,
            can_shed: true,
            is_shed: false,
        };

        controller.add_load(load);

        assert_eq!(controller.loads.len(), 1);
        assert_eq!(controller.total_shedding_capacity, 50.0);
    }

    #[test]
    fn test_add_non_sheddable_load() {
        let mut controller = LoadSheddingController::new();

        let load = Load {
            id: "L1".to_string(),
            name: "Critical IT".to_string(),
            power_kw: 10.0,
            priority: LoadPriority::Critical,
            can_shed: false,
            is_shed: false,
        };

        controller.add_load(load);

        assert_eq!(controller.loads.len(), 1);
        // Non-sheddable loads should not add to capacity
        assert_eq!(controller.total_shedding_capacity, 0.0);
    }

    #[test]
    fn test_available_capacity() {
        let mut controller = LoadSheddingController::new();

        controller.add_load(Load {
            id: "L1".to_string(),
            name: "HVAC".to_string(),
            power_kw: 50.0,
            priority: LoadPriority::Low,
            can_shed: true,
            is_shed: false,
        });

        controller.add_load(Load {
            id: "L2".to_string(),
            name: "Lighting".to_string(),
            power_kw: 30.0,
            priority: LoadPriority::Medium,
            can_shed: true,
            is_shed: false,
        });

        assert_eq!(controller.available_capacity(), 80.0);

        // Shed some load
        controller.shed_loads(30.0);
        assert_eq!(controller.available_capacity(), 50.0);
    }

    #[test]
    fn test_shed_loads_priority_order() {
        let mut controller = LoadSheddingController::new();

        for load in create_test_loads() {
            controller.add_load(load);
        }

        // Request 40 kW shed - should shed Low priority first (50 kW HVAC)
        let shed = controller.shed_loads(40.0);

        assert_eq!(shed, 40.0);

        // Find which load was shed
        let hvac = controller.loads.iter().find(|l| l.id == "L1").unwrap();
        let lighting = controller.loads.iter().find(|l| l.id == "L2").unwrap();
        let receptacles = controller.loads.iter().find(|l| l.id == "L3").unwrap();
        let critical = controller.loads.iter().find(|l| l.id == "L4").unwrap();

        assert!(hvac.is_shed, "Low priority HVAC should be shed first");
        assert!(
            !lighting.is_shed,
            "Medium priority lighting should not be shed"
        );
        assert!(
            !receptacles.is_shed,
            "High priority receptacles should not be shed"
        );
        assert!(!critical.is_shed, "Critical IT should never be shed");
    }

    #[test]
    fn test_shed_loads_multiple_priorities() {
        let mut controller = LoadSheddingController::new();

        for load in create_test_loads() {
            controller.add_load(load);
        }

        // Request 90 kW shed - should shed Low (50) + Medium (30) + part of High (10)
        let shed = controller.shed_loads(90.0);

        assert_eq!(shed, 90.0);

        let hvac = controller.loads.iter().find(|l| l.id == "L1").unwrap();
        let lighting = controller.loads.iter().find(|l| l.id == "L2").unwrap();
        let receptacles = controller.loads.iter().find(|l| l.id == "L3").unwrap();
        let critical = controller.loads.iter().find(|l| l.id == "L4").unwrap();

        assert!(hvac.is_shed);
        assert!(lighting.is_shed);
        assert!(receptacles.is_shed); // Partially shed
        assert!(!critical.is_shed);
    }

    #[test]
    fn test_shed_loads_exceeds_capacity() {
        let mut controller = LoadSheddingController::new();

        for load in create_test_loads() {
            controller.add_load(load);
        }

        // Request 200 kW shed - only 100 kW available (excluding critical)
        let shed = controller.shed_loads(200.0);

        // Should only shed what's available
        assert_eq!(shed, 100.0);
    }

    #[test]
    fn test_shed_loads_zero_target() {
        let mut controller = LoadSheddingController::new();

        for load in create_test_loads() {
            controller.add_load(load);
        }

        let shed = controller.shed_loads(0.0);
        assert_eq!(shed, 0.0);

        // No loads should be shed
        for load in &controller.loads {
            assert!(!load.is_shed);
        }
    }

    #[test]
    fn test_restore_all() {
        let mut controller = LoadSheddingController::new();

        for load in create_test_loads() {
            controller.add_load(load);
        }

        // Shed some loads
        controller.shed_loads(80.0);
        assert!(controller.current_shed_load > 0.0);

        // Restore all
        controller.restore_all();

        assert_eq!(controller.current_shed_load, 0.0);
        for load in &controller.loads {
            assert!(!load.is_shed);
        }
    }

    #[test]
    fn test_current_load() {
        let mut controller = LoadSheddingController::new();

        for load in create_test_loads() {
            controller.add_load(load);
        }

        // Total load = 50 + 30 + 20 + 10 = 110 kW
        assert_eq!(controller.current_load(), 110.0);

        // Shed 50 kW (HVAC)
        controller.shed_loads(50.0);

        // Current load should be 60 kW
        assert_eq!(controller.current_load(), 60.0);
    }

    #[test]
    fn test_shed_exact_load_amount() {
        let mut controller = LoadSheddingController::new();

        controller.add_load(Load {
            id: "L1".to_string(),
            name: "Load 1".to_string(),
            power_kw: 25.0,
            priority: LoadPriority::Low,
            can_shed: true,
            is_shed: false,
        });

        // Shed exactly the load amount
        let shed = controller.shed_loads(25.0);
        assert_eq!(shed, 25.0);
        assert_eq!(controller.current_shed_load, 25.0);
    }

    #[test]
    fn test_shed_smaller_than_smallest_load() {
        let mut controller = LoadSheddingController::new();

        controller.add_load(Load {
            id: "L1".to_string(),
            name: "Load 1".to_string(),
            power_kw: 50.0,
            priority: LoadPriority::Low,
            can_shed: true,
            is_shed: false,
        });

        // Shed 20 kW from a 50 kW load - should partially shed
        let shed = controller.shed_loads(20.0);
        assert_eq!(shed, 20.0);
        assert_eq!(controller.current_shed_load, 20.0);
    }
}

// ============================================================================
// Real-Time Pricing Tests
// ============================================================================

mod rtp_tests {
    use super::*;

    #[test]
    fn test_rtp_creation() {
        let rtp = RealTimePricing::new();

        assert_eq!(rtp.current_price, 0.10);
        assert_eq!(rtp.average_price, 0.10);
        assert_eq!(rtp.peak_price, 0.10);
        assert_eq!(rtp.off_peak_price, 0.10);

        // All hours should default to $0.10/kWh
        for hour in 0..24 {
            assert_eq!(rtp.hourly_prices[hour], 0.10);
        }
    }

    #[test]
    fn test_set_time_of_use() {
        let mut rtp = RealTimePricing::new();
        rtp.set_time_of_use(0.05, 0.10, 0.25);

        // Check off-peak hours (0-6, 23)
        for hour in [0, 1, 2, 3, 4, 5, 6, 23] {
            assert_eq!(rtp.hourly_prices[hour], 0.05);
        }

        // Check morning mid-peak (7-9)
        for hour in [7, 8, 9] {
            assert_eq!(rtp.hourly_prices[hour], 0.10);
        }

        // Check peak hours (10-16, 17-19)
        for hour in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] {
            assert_eq!(rtp.hourly_prices[hour], 0.25);
        }

        // Check evening mid-peak (20-22)
        for hour in [20, 21, 22] {
            assert_eq!(rtp.hourly_prices[hour], 0.10);
        }
    }

    #[test]
    fn test_rtp_statistics_after_tou() {
        let mut rtp = RealTimePricing::new();
        rtp.set_time_of_use(0.05, 0.10, 0.25);

        assert_eq!(rtp.off_peak_price, 0.05);
        assert_eq!(rtp.peak_price, 0.25);

        // Average should be weighted average of all hours
        // 8 off-peak @ 0.05 + 6 mid-peak @ 0.10 + 10 peak @ 0.25
        // = (8*0.05 + 6*0.10 + 10*0.25) / 24 = (0.4 + 0.6 + 2.5) / 24 = 3.5/24 ≈ 0.146
        let expected_avg = (8.0 * 0.05 + 6.0 * 0.10 + 10.0 * 0.25) / 24.0;
        assert!((rtp.average_price - expected_avg).abs() < 0.001);
    }

    #[test]
    fn test_rtp_update_current_price() {
        let mut rtp = RealTimePricing::new();
        rtp.set_time_of_use(0.05, 0.10, 0.25);

        // Test different hours
        rtp.update(3);
        assert_eq!(rtp.current_price, 0.05); // Off-peak

        rtp.update(8);
        assert_eq!(rtp.current_price, 0.10); // Mid-peak

        rtp.update(14);
        assert_eq!(rtp.current_price, 0.25); // Peak

        rtp.update(21);
        assert_eq!(rtp.current_price, 0.10); // Mid-peak
    }

    #[test]
    fn test_rtp_update_wraps_around() {
        let mut rtp = RealTimePricing::new();
        rtp.set_time_of_use(0.05, 0.10, 0.25);

        // Hour 25 should wrap to hour 1
        rtp.update(25);
        assert_eq!(rtp.current_price, 0.05);

        rtp.update(168);
        assert_eq!(rtp.current_price, 0.05); // 168 % 24 = 0
    }

    #[test]
    fn test_rtp_custom_pricing() {
        let mut rtp = RealTimePricing::new();

        // Set custom prices via set_time_of_use
        rtp.set_time_of_use(0.08, 0.15, 0.30);

        assert_eq!(rtp.off_peak_price, 0.08);
        assert_eq!(rtp.peak_price, 0.30);
    }

    #[test]
    fn test_rtp_extreme_prices() {
        let mut rtp = RealTimePricing::new();

        // Set extreme prices via set_time_of_use
        rtp.set_time_of_use(0.01, 0.50, 1.0);

        assert_eq!(rtp.off_peak_price, 0.01);
        assert_eq!(rtp.peak_price, 1.0);
    }
}

// ============================================================================
// DR Manager Tests
// ============================================================================

mod dr_manager_tests {
    use super::*;

    #[test]
    fn test_dr_manager_creation() {
        let manager = DRManager::new();

        assert!(manager.events.is_empty());
        assert!(!manager.enabled);
        assert!(manager.signal_url.is_none());
    }

    #[test]
    fn test_dr_manager_add_event() {
        let mut manager = DRManager::new();

        let event = DREvent::new("DR-1".to_string(), DREventType::PeakShaving, 14, 4, 100.0);
        manager.add_event(event);

        assert_eq!(manager.events.len(), 1);
    }

    #[test]
    fn test_dr_manager_active_events() {
        let mut manager = DRManager::new();

        // Add event 1: hours 10-14
        let event1 = DREvent::new("DR-1".to_string(), DREventType::PeakShaving, 10, 4, 50.0);
        manager.add_event(event1);

        // Add event 2: hours 18-22
        let event2 = DREvent::new("DR-2".to_string(), DREventType::PeakShaving, 18, 4, 75.0);
        manager.add_event(event2);

        // Check hour 12 - only event 1 active
        let active = manager.active_events(12);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "DR-1");

        // Check hour 20 - only event 2 active
        let active = manager.active_events(20);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "DR-2");

        // Check hour 16 - no events active
        let active = manager.active_events(16);
        assert!(active.is_empty());

        // Check hour 10 - event 1 active
        let active = manager.active_events(10);
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_dr_manager_total_reduction_target() {
        let mut manager = DRManager::new();

        // Add overlapping events
        let event1 = DREvent::new("DR-1".to_string(), DREventType::PeakShaving, 10, 4, 50.0);
        manager.add_event(event1);

        let event2 = DREvent::new("DR-2".to_string(), DREventType::Emergency, 12, 4, 75.0);
        manager.add_event(event2);

        // Hour 11 - only event 1 active (50 kW)
        assert_eq!(manager.total_reduction_target(11), 50.0);

        // Hour 13 - both events active (50 + 75 = 125 kW)
        assert_eq!(manager.total_reduction_target(13), 125.0);

        // Hour 15 - only event 2 active (75 kW)
        assert_eq!(manager.total_reduction_target(15), 75.0);
    }

    #[test]
    fn test_dr_manager_is_dr_active() {
        let mut manager = DRManager::new();

        let event = DREvent::new("DR-1".to_string(), DREventType::PeakShaving, 10, 4, 50.0);
        manager.add_event(event);

        assert!(manager.is_dr_active(12));
        assert!(!manager.is_dr_active(8));
        assert!(!manager.is_dr_active(16));
    }

    #[test]
    fn test_dr_manager_set_pricing() {
        let mut manager = DRManager::new();

        manager.set_pricing(0.05, 0.10, 0.25);

        assert_eq!(manager.rtp.off_peak_price, 0.05);
        assert_eq!(manager.rtp.peak_price, 0.25);
    }

    #[test]
    fn test_dr_manager_no_events() {
        let manager = DRManager::new();

        assert!(manager.active_events(12).is_empty());
        assert_eq!(manager.total_reduction_target(12), 0.0);
        assert!(!manager.is_dr_active(12));
    }

    #[test]
    fn test_dr_manager_multiple_overlapping_events() {
        let mut manager = DRManager::new();

        // Add 3 overlapping events
        // A: hours 8-14 (8,9,10,11,12,13), 30 kW
        // B: hours 10-14 (10,11,12,13), 50 kW
        // C: hours 12-16 (12,13,14,15), 25 kW
        manager.add_event(DREvent::new(
            "DR-A".to_string(),
            DREventType::PeakShaving,
            8,
            6,
            30.0,
        ));
        manager.add_event(DREvent::new(
            "DR-B".to_string(),
            DREventType::Emergency,
            10,
            4,
            50.0,
        ));
        manager.add_event(DREvent::new(
            "DR-C".to_string(),
            DREventType::Voluntary,
            12,
            4,
            25.0,
        ));

        // Hour 9 - only A active
        assert_eq!(manager.total_reduction_target(9), 30.0);

        // Hour 11 - A and B active (30 + 50 = 80)
        assert_eq!(manager.total_reduction_target(11), 80.0);

        // Hour 13 - all three active (30 + 50 + 25 = 105)
        assert_eq!(manager.total_reduction_target(13), 105.0);

        // Hour 15 - only C active (A ended at 14, B ended at 14)
        assert_eq!(manager.total_reduction_target(15), 25.0);

        // Hour 17 - none active
        assert_eq!(manager.total_reduction_target(17), 0.0);
    }
}

// ============================================================================
// Load Priority Tests
// ============================================================================

mod priority_tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        // Verify priority enum ordering
        assert!(LoadPriority::Low < LoadPriority::Medium);
        assert!(LoadPriority::Medium < LoadPriority::High);
        assert!(LoadPriority::High < LoadPriority::Critical);
    }

    #[test]
    fn test_priority_values() {
        assert_eq!(LoadPriority::Low as u8, 0);
        assert_eq!(LoadPriority::Medium as u8, 1);
        assert_eq!(LoadPriority::High as u8, 2);
        assert_eq!(LoadPriority::Critical as u8, 3);
    }

    #[test]
    fn test_shedding_respects_priority_order() {
        let mut controller = LoadSheddingController::new();

        // Add loads in random order
        controller.add_load(Load {
            id: "L1".to_string(),
            name: "Critical".to_string(),
            power_kw: 10.0,
            priority: LoadPriority::Critical,
            can_shed: true,
            is_shed: false,
        });

        controller.add_load(Load {
            id: "L2".to_string(),
            name: "Low".to_string(),
            power_kw: 50.0,
            priority: LoadPriority::Low,
            can_shed: true,
            is_shed: false,
        });

        controller.add_load(Load {
            id: "L3".to_string(),
            name: "High".to_string(),
            power_kw: 20.0,
            priority: LoadPriority::High,
            can_shed: true,
            is_shed: false,
        });

        controller.add_load(Load {
            id: "L4".to_string(),
            name: "Medium".to_string(),
            power_kw: 30.0,
            priority: LoadPriority::Medium,
            can_shed: true,
            is_shed: false,
        });

        // Shed 60 kW - should shed Low (50) + Medium (10 of 30)
        controller.shed_loads(60.0);

        let low = controller
            .loads
            .iter()
            .find(|l| l.priority == LoadPriority::Low)
            .unwrap();
        let medium = controller
            .loads
            .iter()
            .find(|l| l.priority == LoadPriority::Medium)
            .unwrap();
        let high = controller
            .loads
            .iter()
            .find(|l| l.priority == LoadPriority::High)
            .unwrap();
        let critical = controller
            .loads
            .iter()
            .find(|l| l.priority == LoadPriority::Critical)
            .unwrap();

        assert!(low.is_shed);
        assert!(medium.is_shed); // Partially
        assert!(!high.is_shed);
        assert!(!critical.is_shed);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_dr_event_lifecycle() {
        let mut manager = DRManager::new();
        manager.enabled = true;

        // Create and add event
        let mut event = DREvent::new(
            "DR-FULL".to_string(),
            DREventType::PeakShaving,
            14,
            4,
            100.0,
        );
        event.start();
        manager.add_event(event);

        // Verify event is active
        assert!(manager.is_dr_active(15));
        assert_eq!(manager.total_reduction_target(15), 100.0);

        // Set up load shedding
        manager.load_shedding.add_load(Load {
            id: "L1".to_string(),
            name: "HVAC".to_string(),
            power_kw: 60.0,
            priority: LoadPriority::Low,
            can_shed: true,
            is_shed: false,
        });

        manager.load_shedding.add_load(Load {
            id: "L2".to_string(),
            name: "Lighting".to_string(),
            power_kw: 40.0,
            priority: LoadPriority::Medium,
            can_shed: true,
            is_shed: false,
        });

        // Shed load to meet target
        let target = manager.total_reduction_target(15);
        let shed = manager.load_shedding.shed_loads(target);

        assert_eq!(shed, 100.0);
        assert_eq!(manager.load_shedding.current_shed_load, 100.0);
    }

    #[test]
    fn test_dr_with_real_time_pricing() {
        let mut manager = DRManager::new();

        // Set up TOU pricing
        manager.set_pricing(0.05, 0.10, 0.25);

        // Add event during peak hours
        manager.add_event(DREvent::new(
            "DR-PEAK".to_string(),
            DREventType::PeakShaving,
            14,
            2,
            50.0,
        ));

        // Verify pricing during event
        manager.rtp.update(14);
        assert_eq!(manager.rtp.current_price, 0.25);

        // DR event is active
        assert!(manager.is_dr_active(14));
    }

    #[test]
    fn test_load_shedding_cascade() {
        let mut controller = LoadSheddingController::new();

        // Add many small loads
        for i in 0..10 {
            controller.add_load(Load {
                id: format!("L{}", i),
                name: format!("Load {}", i),
                power_kw: 10.0,
                priority: LoadPriority::Low,
                can_shed: true,
                is_shed: false,
            });
        }

        // Shed 55 kW - should shed 5 loads fully and 1 partially
        let shed = controller.shed_loads(55.0);
        assert_eq!(shed, 55.0);

        let shed_count = controller.loads.iter().filter(|l| l.is_shed).count();
        assert!(shed_count >= 6); // At least 6 loads affected
    }
}
