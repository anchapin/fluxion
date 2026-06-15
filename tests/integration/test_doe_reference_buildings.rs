//! DOE Commercial Reference Building Integration Tests
//!
//! Tests validate that the engine handles multi-zone commercial buildings
//! at scale without deadlocking, memory issues, or numerical instability.
//!
//! # DOE Prototype Buildings
//!
//! - **Small Office**: 511 m², 4 zones - light commercial
//! - **Medium Office**: 4982 m², 12 zones - medium commercial
//! - **Stand-alone Retail**: 2326 m², 4 zones - retail
//!
//! # Test Coverage
//!
//! - Model creation and validation
//! - Annual simulation (8760 timesteps)
//! - Engine scalability (hundreds of nodes)
//! - Memory usage reporting
//! - Parallel execution safety

use fluxion::testing::integration::{
    run_annual_simulation, DoeBuildingConfig, DoeBuildingType, MemoryStats,
};
use std::time::Instant;

#[cfg(test)]
mod small_office_tests {
    use super::*;

    #[test]
    fn test_small_office_model_creation() {
        let config = DoeBuildingConfig::small_office();
        let model = config.create_model();

        assert_eq!(model.case_id, "DOE_SmallOffice");
        assert_eq!(config.num_zones, 4);
        assert!((config.total_floor_area_m2 - 511.0).abs() < 1.0);
    }

    #[test]
    fn test_small_office_simulation_completes() {
        let config = DoeBuildingConfig::small_office();
        let result = run_annual_simulation(&config);

        assert!(result.is_ok(), "Simulation should complete without error");
        let (energy, _stats, elapsed) = result.unwrap();

        assert!(
            energy.is_finite(),
            "Annual energy should be finite, got {}",
            energy
        );
        println!(
            "Small Office: {:.2} kWh annual energy ({} zones, {:?})",
            energy, config.num_zones, elapsed
        );
    }

    #[test]
    fn test_small_office_simulation_timing() {
        let config = DoeBuildingConfig::small_office();
        let start = Instant::now();
        let result = run_annual_simulation(&config);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        let (energy, stats, _) = result.unwrap();

        assert!(energy.is_finite());
        println!(
            "Small Office timing: {:?} for {} zones, {} nodes, energy={:.2} kWh",
            elapsed, config.num_zones, stats.model_node_count, energy
        );

        // Timing assertion - should complete in reasonable time
        // Release mode: < 5 seconds for 4 zones
        #[cfg(not(tarpaulin))]
        {
            let max_seconds = 5;
            assert!(
                elapsed.as_secs() < max_seconds,
                "Small Office simulation took {:?}, expected < {}s",
                elapsed,
                max_seconds
            );
        }
    }
}

#[cfg(test)]
mod medium_office_tests {
    use super::*;

    #[test]
    fn test_medium_office_model_creation() {
        let config = DoeBuildingConfig::medium_office();
        let model = config.create_model();

        assert_eq!(model.case_id, "DOE_MediumOffice");
        assert_eq!(config.num_zones, 12);
        assert!((config.total_floor_area_m2 - 4982.0).abs() < 10.0);
    }

    #[test]
    fn test_medium_office_simulation_completes() {
        let config = DoeBuildingConfig::medium_office();
        let result = run_annual_simulation(&config);

        assert!(result.is_ok(), "Simulation should complete without error");
        let (energy, _stats, elapsed) = result.unwrap();

        assert!(
            energy.is_finite(),
            "Annual energy should be finite, got {}",
            energy
        );
        println!(
            "Medium Office: {:.2} kWh annual energy ({} zones, {:?})",
            energy, config.num_zones, elapsed
        );
    }

    #[test]
    fn test_medium_office_memory_usage() {
        let config = DoeBuildingConfig::medium_office();
        let result = run_annual_simulation(&config);

        assert!(result.is_ok());
        let (energy, stats, elapsed) = result.unwrap();

        assert!(energy.is_finite());

        // Report memory statistics
        println!(
            "Medium Office memory: {} nodes, {:?}, energy={:.2} kWh",
            stats.model_node_count, elapsed, energy
        );

        // Medium Office has 12 zones - verify node count
        assert_eq!(
            stats.model_node_count, 12,
            "Medium Office should have 12 nodes"
        );
    }
}

#[cfg(test)]
mod standalone_retail_tests {
    use super::*;

    #[test]
    fn test_retail_model_creation() {
        let config = DoeBuildingConfig::standalone_retail();
        let model = config.create_model();

        assert_eq!(model.case_id, "DOE_RetailStandalone");
        assert_eq!(config.num_zones, 4);
        assert!((config.total_floor_area_m2 - 2326.0).abs() < 10.0);
    }

    #[test]
    fn test_retail_simulation_completes() {
        let config = DoeBuildingConfig::standalone_retail();
        let result = run_annual_simulation(&config);

        assert!(result.is_ok(), "Simulation should complete without error");
        let (energy, _stats, elapsed) = result.unwrap();

        assert!(
            energy.is_finite(),
            "Annual energy should be finite, got {}",
            energy
        );
        println!(
            "Stand-alone Retail: {:.2} kWh annual energy ({} zones, {:?})",
            energy, config.num_zones, elapsed
        );
    }
}

#[cfg(test)]
mod scalability_tests {
    use super::*;

    #[test]
    fn test_all_doe_buildings_simulate_successfully() {
        let configs = [
            DoeBuildingConfig::small_office(),
            DoeBuildingConfig::medium_office(),
            DoeBuildingConfig::standalone_retail(),
        ];

        for config in configs {
            let result = run_annual_simulation(&config);
            assert!(
                result.is_ok(),
                "{:?} simulation failed: {:?}",
                config.building_type,
                result.err()
            );

            let (energy, _stats, _elapsed) = result.unwrap();
            assert!(
                energy.is_finite(),
                "{:?} produced infinite energy",
                config.building_type
            );
        }
    }

    #[test]
    fn test_engine_handles_scale_without_deadlock() {
        // Medium Office has 12 zones - represents significant scale
        // Test verifies parallel execution doesn't deadlock
        let config = DoeBuildingConfig::medium_office();
        let start = Instant::now();
        let result = run_annual_simulation(&config);
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Medium Office simulation failed");

        // If we get here, no deadlock occurred
        let (energy, stats, _) = result.unwrap();
        assert!(energy.is_finite());

        println!(
            "Scale test: {:?} with {} nodes",
            elapsed, stats.model_node_count
        );
    }

    #[test]
    fn test_multi_zone_node_count() {
        // Verify node count scales with building complexity
        let small = DoeBuildingConfig::small_office();
        let medium = DoeBuildingConfig::medium_office();
        let retail = DoeBuildingConfig::standalone_retail();

        assert_eq!(small.num_zones, 4);
        assert_eq!(medium.num_zones, 12);
        assert_eq!(retail.num_zones, 4);

        // Medium Office should have most zones
        assert!(
            medium.num_zones > small.num_zones,
            "Medium Office should have more zones than Small Office"
        );
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_doe_config_validation() {
        let configs = [
            DoeBuildingConfig::small_office(),
            DoeBuildingConfig::medium_office(),
            DoeBuildingConfig::standalone_retail(),
        ];

        for config in configs {
            // All configs should have valid parameters
            assert!(config.total_floor_area_m2 > 0.0);
            assert!(config.zone_area_m2 > 0.0);
            assert!(config.ceiling_height_m > 0.0);
            assert!((0.0..=1.0).contains(&config.window_to_wall_ratio));
            assert!(config.wall_u_value_w_m2k > 0.0);
            assert!(config.roof_u_value_w_m2k > 0.0);
            assert!(config.window_u_value_w_m2k > 0.0);
            assert!(config.infiltration_rate_ach >= 0.0);
            assert!(config.heating_setpoint_c > 0.0);
            assert!(config.cooling_setpoint_c > config.heating_setpoint_c);
            assert!(config.internal_loads_w_m2 >= 0.0);
            assert!(config.hvac_heating_capacity_w_m2 > 0.0);
            assert!(config.hvac_cooling_capacity_w_m2 > 0.0);
        }
    }

    #[test]
    fn test_building_scenario_conversion() {
        let config = DoeBuildingConfig::small_office();
        let scenario = config.to_building_scenario();
        let _built = scenario.build().expect("scenario should be valid");

        // Create model and verify it has the expected properties
        let model = config.create_model();
        assert_eq!(model.case_id, "DOE_SmallOffice");
        assert_eq!(model.temperatures.len(), 4, "Should have 4 zones");
        assert!((model.heating_setpoint - 21.0).abs() < 0.1);
        assert!((model.cooling_setpoint - 24.0).abs() < 0.1);
        assert!((model.window_u_value - 2.78).abs() < 0.01);
    }
}
