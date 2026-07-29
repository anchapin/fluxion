//! Empirical Validation Integration Tests
//!
//! Tests for the empirical validation suite using monitored building data.
//! These tests verify:
//! - Statistical calculations (NMBE, CVRMSE, etc.)
//! - Data source registration and retrieval
//! - CSV parsing
//! - Report generation
//! - ASHRAE criteria validation

use fluxion::validation::empirical::{
    generate_empirical_report, get_ashrae_rp_sources, BuildingType, EmpiricalMetric,
    EmpiricalStatistics, EmpiricalValidationConfig, EmpiricalValidationReport,
    EmpiricalValidationResult, EmpiricalValidationStatus, MonitoredBuildingDatabase,
    MonitoredDataPoint, MonitoredDataSource,
};
use std::collections::HashMap;

/// Create test monitored data points
fn create_test_monitored_data(n: usize) -> Vec<MonitoredDataPoint> {
    (0..n)
        .map(|hour| MonitoredDataPoint {
            hour,
            T_outdoor: 10.0 + (hour as f64 * 0.01).sin() * 15.0, // Varies through day
            T_zone: 22.0 + (hour as f64 * 0.01).sin() * 2.0,     // Zone stays roughly constant
            Q_heat: if hour < 8 || hour > 18 { 5000.0 } else { 0.0 }, // Heating at night/morning
            Q_cool: if hour >= 12 && hour <= 16 {
                3000.0
            } else {
                0.0
            }, // Cooling afternoon
            Q_solar: (hour as f64 * 0.05).sin().max(0.0) * 2000.0,
            Q_internal: 1500.0,
            Q_ventilation: 800.0,
            Q_conduction: 1200.0,
        })
        .collect()
}

/// Create a test monitored data source
fn create_test_source() -> MonitoredDataSource {
    MonitoredDataSource {
        id: "test_source".to_string(),
        name: "Test Monitored Building".to_string(),
        source: "Test Source".to_string(),
        building_type: BuildingType::Office,
        climate_zone: "5A".to_string(),
        location: "Test City".to_string(),
        latitude: 42.0,
        longitude: -88.0,
        floor_area: 1000.0,
        num_floors: 2,
        zone_volume: 2700.0,
        u_wall: 0.35,
        u_roof: 0.25,
        u_window: 2.7,
        wwr: 0.30,
        infiltration_ach: 0.5,
        internal_gains_density: 15.0,
        time_resolution_hours: 1.0,
        num_data_points: 8760,
    }
}

#[test]
fn test_statistics_calculation_perfect_match() {
    let predicted = vec![20.0, 21.0, 22.0, 23.0, 24.0];
    let reference = vec![20.0, 21.0, 22.0, 23.0, 24.0];

    let stats = EmpiricalStatistics::calculate(&predicted, &reference);

    assert_eq!(stats.n, 5);
    assert!((stats.mbe - 0.0).abs() < 1e-10);
    assert!((stats.rmse - 0.0).abs() < 1e-10);
    assert!(stats.passes_ashrae_criteria());
}

#[test]
fn test_statistics_calculation_with_bias() {
    let predicted = vec![20.5, 21.5, 22.5, 23.5, 24.5]; // 0.5 offset
    let reference = vec![20.0, 21.0, 22.0, 23.0, 24.0];

    let stats = EmpiricalStatistics::calculate(&predicted, &reference);

    assert_eq!(stats.n, 5);
    assert!((stats.mbe - 0.5).abs() < 0.01);
    assert!((stats.nmbe - 2.27).abs() < 0.1); // 0.5/22 * 100
    assert!(stats.passes_ashrae_criteria()); // Still within ±10%
}

#[test]
fn test_statistics_calculation_exceeds_threshold() {
    let predicted = vec![25.0, 26.0, 27.0, 28.0, 29.0]; // 5.0 offset - exceeds 10%
    let reference = vec![20.0, 21.0, 22.0, 23.0, 24.0];

    let stats = EmpiricalStatistics::calculate(&predicted, &reference);

    assert_eq!(stats.n, 5);
    assert!((stats.mbe - 5.0).abs() < 0.01);
    assert!((stats.nmbe - 22.7).abs() < 0.1); // 5.0/22 * 100
    assert!(!stats.passes_ashrae_criteria()); // Exceeds ±10% NMBE but CV(RMSE) OK
    assert_eq!(
        stats.get_ashrae_status(),
        EmpiricalValidationStatus::Warning
    ); // Only NMBE fails
}

#[test]
fn test_statistics_calculation_cv_rmse_threshold() {
    // Small bias but high variance
    let predicted = vec![20.0, 25.0, 20.0, 25.0, 20.0, 25.0];
    let reference = vec![22.0, 22.0, 22.0, 22.0, 22.0, 22.0];

    let stats = EmpiricalStatistics::calculate(&predicted, &reference);

    // Mean reference is 22
    // RMSE should be around 3.16 (std dev of 2)
    // CV(RMSE) should be around 14.4%
    assert!(stats.cv_rmse > 10.0 && stats.cv_rmse < 20.0);
}

#[test]
fn test_statistics_empty_arrays() {
    let predicted: Vec<f64> = vec![];
    let reference: Vec<f64> = vec![];

    let stats = EmpiricalStatistics::calculate(&predicted, &reference);

    assert_eq!(stats.n, 0);
    assert!(stats.mbe.is_nan());
    assert!(stats.rmse.is_nan());
}

#[test]
fn test_statistics_single_point() {
    let predicted = vec![22.0];
    let reference = vec![21.0];

    let stats = EmpiricalStatistics::calculate(&predicted, &reference);

    assert_eq!(stats.n, 1);
    assert!((stats.mbe - 1.0).abs() < 0.01);
    // Single point - standard error undefined, CI undefined
    assert!(stats.standard_error.is_nan() || stats.standard_error == 0.0);
}

#[test]
fn test_monitored_building_database_registration() {
    let mut db = MonitoredBuildingDatabase::new();
    let source = create_test_source();

    db.register(source.clone());

    assert_eq!(db.sources.len(), 1);
    assert!(db.get("test_source").is_some());
    assert!(db.get("nonexistent").is_none());
}

#[test]
fn test_ashrae_rp_sources() {
    let db = get_ashrae_rp_sources();

    // Verify all standard sources are registered
    assert!(db.get("ashrae_rp1055_office").is_some());
    assert!(db.get("ashrae_rp1256_commercial").is_some());
    assert!(db.get("iea_ebc_annex60_office").is_some());
    assert!(db.get("nrel_cbm_small_office").is_some());
    assert!(db.get("nrel_cbm_medium_office").is_some());
    assert!(db.get("lbnl_flexlab_ashrae140").is_some());

    // Check building types
    let office = db.get("ashrae_rp1055_office").unwrap();
    assert_eq!(office.building_type, BuildingType::Office);

    let commercial = db.get("ashrae_rp1256_commercial").unwrap();
    assert_eq!(commercial.building_type, BuildingType::Commercial);

    // FLEXLAB dataset validation
    let flexlab = db.get("lbnl_flexlab_ashrae140").unwrap();
    assert_eq!(flexlab.climate_zone, "3C");
    assert_eq!(flexlab.location, "Berkeley, CA");
    assert!(flexlab.floor_area > 0.0);
    assert!(flexlab.zone_volume > 0.0);
}

#[test]
fn test_ashrae_rp_sources_metadata() {
    let db = get_ashrae_rp_sources();

    // ASHRAE RP-1055 Office
    let rp1055 = db.get("ashrae_rp1055_office").unwrap();
    assert_eq!(rp1055.source, "ASHRAE RP-1055");
    assert_eq!(rp1055.climate_zone, "5A");
    assert!(rp1055.floor_area > 0.0);
    assert!(rp1055.zone_volume > 0.0);

    // NREL CBM Small Office
    let nrel = db.get("nrel_cbm_small_office").unwrap();
    assert!(nrel.source.contains("NREL"));
    assert_eq!(nrel.building_type, BuildingType::Office);
    assert!(nrel.wwr > 0.0 && nrel.wwr < 1.0); // Valid WWR

    // LBNL FLEXLAB ASHRAE 140
    let flexlab = db.get("lbnl_flexlab_ashrae140").unwrap();
    assert!(flexlab.source.contains("FLEXLAB"));
    assert_eq!(flexlab.climate_zone, "3C");
    assert!(flexlab.floor_area > 0.0);
}

#[test]
fn test_building_types_all_variants() {
    let mut db = MonitoredBuildingDatabase::new();

    let types = vec![
        (BuildingType::Office, "office"),
        (BuildingType::Residential, "residential"),
        (BuildingType::Commercial, "commercial"),
        (BuildingType::Retail, "retail"),
        (BuildingType::Hotel, "hotel"),
        (BuildingType::Hospital, "hospital"),
        (BuildingType::School, "school"),
        (BuildingType::Warehouse, "warehouse"),
        (BuildingType::Other, "other"),
    ];

    for (building_type, _name) in types {
        let source = MonitoredDataSource {
            id: format!("test_{:?}", building_type).to_lowercase(),
            name: format!("{:?} Test Building", building_type),
            source: "Test".to_string(),
            building_type,
            climate_zone: "5A".to_string(),
            location: "Test".to_string(),
            latitude: 42.0,
            longitude: -88.0,
            floor_area: 1000.0,
            num_floors: 2,
            zone_volume: 2700.0,
            u_wall: 0.35,
            u_roof: 0.25,
            u_window: 2.7,
            wwr: 0.30,
            infiltration_ach: 0.5,
            internal_gains_density: 15.0,
            time_resolution_hours: 1.0,
            num_data_points: 8760,
        };
        db.register(source);
    }

    assert_eq!(db.sources.len(), 9);
}

#[test]
fn test_empirical_validation_config_defaults() {
    let config = EmpiricalValidationConfig::default();

    assert_eq!(config.nmbe_threshold, 10.0);
    assert_eq!(config.cv_rmse_threshold, 30.0);
    assert_eq!(config.near_zero_threshold, 1e-10);
    assert!(config.validate_hourly_temperatures);
    assert!(config.validate_daily_peaks);
    assert!(config.validate_annual_totals);
}

#[test]
fn test_empirical_validation_config_custom() {
    let config = EmpiricalValidationConfig {
        nmbe_threshold: 15.0,
        cv_rmse_threshold: 35.0,
        near_zero_threshold: 1e-8,
        validate_hourly_temperatures: false,
        validate_daily_peaks: true,
        validate_annual_totals: true,
    };

    assert_eq!(config.nmbe_threshold, 15.0);
    assert_eq!(config.cv_rmse_threshold, 35.0);
    assert!(!config.validate_hourly_temperatures);
}

#[test]
fn test_empirical_validation_result_serialization() {
    let result = EmpiricalValidationResult {
        source_id: "test_source".to_string(),
        metric: EmpiricalMetric::MeanZoneTemperature,
        predicted: 22.0,
        reference: 21.5,
        error: 0.5,
        percentage_error: 2.33,
        status: EmpiricalValidationStatus::Pass,
    };

    // Test JSON round-trip
    let json = serde_json::to_string(&result).unwrap();
    let parsed: EmpiricalValidationResult = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.source_id, result.source_id);
    assert_eq!(parsed.predicted, result.predicted);
    assert_eq!(parsed.reference, result.reference);
    assert_eq!(parsed.status, result.status);
}

#[test]
fn test_empirical_metrics_all_variants() {
    let metrics = vec![
        EmpiricalMetric::MeanZoneTemperature,
        EmpiricalMetric::MaxZoneTemperature,
        EmpiricalMetric::MinZoneTemperature,
        EmpiricalMetric::DailyPeakHeating,
        EmpiricalMetric::DailyPeakCooling,
        EmpiricalMetric::AnnualHeating,
        EmpiricalMetric::AnnualCooling,
        EmpiricalMetric::HourlyTemperature,
    ];

    for metric in metrics {
        let result = EmpiricalValidationResult {
            source_id: "test".to_string(),
            metric,
            predicted: 100.0,
            reference: 95.0,
            error: 5.0,
            percentage_error: 5.26,
            status: EmpiricalValidationStatus::Pass,
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: EmpiricalValidationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.metric, metric);
    }
}

#[test]
fn test_validation_status_all_variants() {
    let statuses = vec![
        EmpiricalValidationStatus::Pass,
        EmpiricalValidationStatus::Warning,
        EmpiricalValidationStatus::Fail,
    ];

    for status in statuses {
        let result = EmpiricalValidationResult {
            source_id: "test".to_string(),
            metric: EmpiricalMetric::MeanZoneTemperature,
            predicted: 100.0,
            reference: 95.0,
            error: 5.0,
            percentage_error: 5.26,
            status,
        };

        assert_eq!(result.status, status);
    }
}

#[test]
fn test_generate_empirical_report() {
    let source = create_test_source();
    let data = create_test_monitored_data(24);
    let fluxion_zone_temps: Vec<f64> = data.iter().map(|p| p.T_zone + 0.5).collect(); // 0.5 offset
    let config = EmpiricalValidationConfig::default();

    let report = generate_empirical_report(&source, &data, &fluxion_zone_temps, &config);

    assert_eq!(report.source.id, "test_source");
    assert!(report.statistics.n > 0);
    // With 0.5 offset on zone temps around 22, NMBE should be about 2.3%
    assert!(report.statistics.nmbe.abs() > 0.0 && report.statistics.nmbe.abs() < 5.0);
    assert!(
        report.status == EmpiricalValidationStatus::Pass
            || report.status == EmpiricalValidationStatus::Warning
    );
}

#[test]
fn test_generate_empirical_report_with_large_error() {
    let source = create_test_source();
    let data = create_test_monitored_data(24);
    // Fluxion predictions with 15°C offset - exceeds ASHRAE criteria
    let fluxion_zone_temps: Vec<f64> = data.iter().map(|p| p.T_zone + 15.0).collect();
    let config = EmpiricalValidationConfig::default();

    let report = generate_empirical_report(&source, &data, &fluxion_zone_temps, &config);

    assert_eq!(report.source.id, "test_source");
    assert!(!report.statistics.passes_ashrae_criteria());
    assert_eq!(report.status, EmpiricalValidationStatus::Fail);
}

#[test]
fn test_empirical_report_serialization() {
    let source = create_test_source();
    let data = create_test_monitored_data(24);
    let fluxion_zone_temps: Vec<f64> = data.iter().map(|p| p.T_zone + 0.5).collect();
    let config = EmpiricalValidationConfig::default();

    let report = generate_empirical_report(&source, &data, &fluxion_zone_temps, &config);

    // Test JSON serialization
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("test_source"));
    assert!(json.contains("statistics"));

    // Test deserialization
    let parsed: EmpiricalValidationReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.source.id, report.source.id);
    assert_eq!(parsed.statistics.n, report.statistics.n);
}

#[test]
fn test_monitored_data_point_structure() {
    let point = MonitoredDataPoint {
        hour: 12,
        T_outdoor: 25.0,
        T_zone: 22.0,
        Q_heat: 5000.0,
        Q_cool: 0.0,
        Q_solar: 1500.0,
        Q_internal: 1200.0,
        Q_ventilation: 800.0,
        Q_conduction: 1000.0,
    };

    assert_eq!(point.hour, 12);
    assert_eq!(point.T_outdoor, 25.0);
    assert_eq!(point.T_zone, 22.0);
    assert_eq!(point.Q_heat, 5000.0);
    assert_eq!(point.Q_cool, 0.0);
}

#[test]
fn test_monitored_data_source_structure() {
    let source = MonitoredDataSource {
        id: "test_id".to_string(),
        name: "Test Building".to_string(),
        source: "Test Source".to_string(),
        building_type: BuildingType::Commercial,
        climate_zone: "3C".to_string(),
        location: "Los Angeles, CA".to_string(),
        latitude: 34.05,
        longitude: -118.24,
        floor_area: 5000.0,
        num_floors: 3,
        zone_volume: 13500.0,
        u_wall: 0.45,
        u_roof: 0.30,
        u_window: 3.0,
        wwr: 0.40,
        infiltration_ach: 0.3,
        internal_gains_density: 20.0,
        time_resolution_hours: 1.0,
        num_data_points: 8760,
    };

    assert_eq!(source.id, "test_id");
    assert_eq!(source.building_type, BuildingType::Commercial);
    assert_eq!(source.climate_zone, "3C");
    assert!((source.latitude - 34.05).abs() < 0.01);
    assert!((source.longitude - (-118.24)).abs() < 0.01);
    assert_eq!(source.wwr, 0.40);
}

#[test]
fn test_monitored_data_source_derived_values() {
    let source = MonitoredDataSource {
        id: "test".to_string(),
        name: "Test".to_string(),
        source: "Test".to_string(),
        building_type: BuildingType::Office,
        climate_zone: "5A".to_string(),
        location: "Test".to_string(),
        latitude: 42.0,
        longitude: -88.0,
        floor_area: 1000.0,
        num_floors: 2,
        zone_volume: 2700.0,
        u_wall: 0.35,
        u_roof: 0.25,
        u_window: 2.7,
        wwr: 0.30,
        infiltration_ach: 0.5,
        internal_gains_density: 15.0,
        time_resolution_hours: 1.0,
        num_data_points: 8760,
    };

    // Check derived values make sense
    // With 1000m² floor area and 2.7m ceiling height per floor
    // Total volume should be roughly 1000 * 2.7 * 2 = 5400
    // But zone_volume is set to 2700, so one floor
    assert_eq!(source.floor_area, 1000.0);
    assert_eq!(source.num_floors, 2);
}

#[test]
fn test_empirical_validation_status_comparison() {
    // Test that EmpiricalValidationStatus derives Ord correctly for sorting
    let statuses = vec![
        EmpiricalValidationStatus::Fail,
        EmpiricalValidationStatus::Pass,
        EmpiricalValidationStatus::Warning,
    ];

    // Should be able to sort
    let mut sorted = statuses.clone();
    sorted.sort();
    sorted.dedup();

    assert_eq!(sorted.len(), 3);
}

#[test]
fn test_empirical_statistics_r_squared() {
    // Perfect predictions
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = EmpiricalStatistics::calculate(&predicted, &reference);
    assert!((stats.r_squared - 1.0).abs() < 1e-10);

    // Random predictions should give low R²
    let predicted_bad = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let stats_bad = EmpiricalStatistics::calculate(&predicted_bad, &reference);
    assert!(stats_bad.r_squared < 0.0); // Negative R² indicates predictions are worse than mean
}

#[test]
fn test_monitored_database_default() {
    let db = MonitoredBuildingDatabase::default();
    assert!(db.sources.is_empty());
}

#[test]
fn test_generate_empirical_report_empty_data() {
    let source = create_test_source();
    let data: Vec<MonitoredDataPoint> = vec![];
    let fluxion_zone_temps: Vec<f64> = vec![];
    let config = EmpiricalValidationConfig::default();

    let report = generate_empirical_report(&source, &data, &fluxion_zone_temps, &config);

    assert_eq!(report.source.id, "test_source");
    assert!(report.results.is_empty());
    // Empty data should result in fail status
    assert_eq!(report.status, EmpiricalValidationStatus::Fail);
}
