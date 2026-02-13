//! Benchmark data for ASHRAE 140 reference cases.
//!
//! This module provides reference data from EnergyPlus, ESP-r, TRNSYS, and DOE2
//! for all ASHRAE 140 test cases.

use super::report::BenchmarkData;
use std::collections::HashMap;

/// Returns benchmark data for all ASHRAE 140 test cases.
///
/// Reference data is from ASHRAE Standard 140-2023 and EnergyPlus BESTEST reports.
/// Values are the min/max across reference programs (EnergyPlus, ESP-r, TRNSYS, DOE2).
pub fn get_all_benchmark_data() -> HashMap<String, BenchmarkData> {
    let mut data = HashMap::new();

    // ==================== Low Mass Cases (600 Series) ====================

    // Case 600 - Baseline (Low Mass)
    data.insert(
        "600".to_string(),
        BenchmarkData {
            annual_heating_min: 4.30,  // MWh
            annual_heating_max: 5.71,
            annual_cooling_min: 6.14,  // MWh
            annual_cooling_max: 8.45,
            peak_heating_min: 4.20,    // kW
            peak_heating_max: 5.60,
            peak_cooling_min: 2.90,    // kW
            peak_cooling_max: 3.90,
            min_free_float_min: -18.8, // °C
            min_free_float_max: -15.6,
            max_free_float_min: 64.9,  // °C
            max_free_float_max: 75.1,
        },
    );

    // Case 610 - South Shading (Low Mass)
    data.insert(
        "610".to_string(),
        BenchmarkData {
            annual_heating_min: 4.36,
            annual_heating_max: 5.79,
            annual_cooling_min: 3.92,
            annual_cooling_max: 6.14,
            peak_heating_min: 4.30,
            peak_heating_max: 5.70,
            peak_cooling_min: 2.20,
            peak_cooling_max: 2.90,
            min_free_float_min: -19.2,
            min_free_float_max: -16.0,
            max_free_float_min: 60.2,
            max_free_float_max: 68.9,
        },
    );

    // Case 620 - East/West Windows (Low Mass)
    data.insert(
        "620".to_string(),
        BenchmarkData {
            annual_heating_min: 4.61,
            annual_heating_max: 5.94,
            annual_cooling_min: 3.42,
            annual_cooling_max: 5.48,
            peak_heating_min: 4.50,
            peak_heating_max: 5.90,
            peak_cooling_min: 2.10,
            peak_cooling_max: 2.80,
            min_free_float_min: -18.5,
            min_free_float_max: -15.3,
            max_free_float_min: 62.8,
            max_free_float_max: 71.5,
        },
    );

    // Case 630 - East/West Shading (Low Mass)
    data.insert(
        "630".to_string(),
        BenchmarkData {
            annual_heating_min: 5.05,
            annual_heating_max: 6.47,
            annual_cooling_min: 2.13,
            annual_cooling_max: 3.70,
            peak_heating_min: 4.70,
            peak_heating_max: 6.10,
            peak_cooling_min: 1.80,
            peak_cooling_max: 2.40,
            min_free_float_min: -18.0,
            min_free_float_max: -14.8,
            max_free_float_min: 58.5,
            max_free_float_max: 66.2,
        },
    );

    // Case 640 - Thermostat Setback (Low Mass)
    data.insert(
        "640".to_string(),
        BenchmarkData {
            annual_heating_min: 2.75,
            annual_heating_max: 3.80,
            annual_cooling_min: 5.95,
            annual_cooling_max: 8.10,
            peak_heating_min: 4.30,
            peak_heating_max: 5.70,
            peak_cooling_min: 2.80,
            peak_cooling_max: 3.70,
            min_free_float_min: -18.6,
            min_free_float_max: -15.4,
            max_free_float_min: 63.5,
            max_free_float_max: 72.8,
        },
    );

    // Case 650 - Night Ventilation (Low Mass)
    data.insert(
        "650".to_string(),
        BenchmarkData {
            annual_heating_min: 0.00,
            annual_heating_max: 0.00,
            annual_cooling_min: 4.82,
            annual_cooling_max: 7.06,
            peak_heating_min: 0.00,
            peak_heating_max: 0.00,
            peak_cooling_min: 1.90,
            peak_cooling_max: 2.50,
            min_free_float_min: -23.0,
            min_free_float_max: -21.0,
            max_free_float_min: 58.8,
            max_free_float_max: 67.5,
        },
    );

    // Case 600FF - Free Float (Low Mass)
    data.insert(
        "600FF".to_string(),
        BenchmarkData {
            annual_heating_min: 0.00,
            annual_heating_max: 0.00,
            annual_cooling_min: 0.00,
            annual_cooling_max: 0.00,
            peak_heating_min: 0.00,
            peak_heating_max: 0.00,
            peak_cooling_min: 0.00,
            peak_cooling_max: 0.00,
            min_free_float_min: -18.8,
            min_free_float_max: -15.6,
            max_free_float_min: 64.9,
            max_free_float_max: 75.1,
        },
    );

    // Case 650FF - Free Float with Night Ventilation (Low Mass)
    data.insert(
        "650FF".to_string(),
        BenchmarkData {
            annual_heating_min: 0.00,
            annual_heating_max: 0.00,
            annual_cooling_min: 0.00,
            annual_cooling_max: 0.00,
            peak_heating_min: 0.00,
            peak_heating_max: 0.00,
            peak_cooling_min: 0.00,
            peak_cooling_max: 0.00,
            min_free_float_min: -23.0,
            min_free_float_max: -21.0,
            max_free_float_min: 63.2,
            max_free_float_max: 73.5,
        },
    );

    // ==================== High Mass Cases (900 Series) ====================

    // Case 900 - Baseline (High Mass)
    data.insert(
        "900".to_string(),
        BenchmarkData {
            annual_heating_min: 1.17,
            annual_heating_max: 2.04,
            annual_cooling_min: 2.13,
            annual_cooling_max: 3.67,
            peak_heating_min: 1.80,
            peak_heating_max: 2.40,
            peak_cooling_min: 1.60,
            peak_cooling_max: 2.10,
            min_free_float_min: -6.4,
            min_free_float_max: -1.6,
            max_free_float_min: 41.8,
            max_free_float_max: 46.4,
        },
    );

    // Case 910 - South Shading (High Mass)
    data.insert(
        "910".to_string(),
        BenchmarkData {
            annual_heating_min: 1.51,
            annual_heating_max: 2.28,
            annual_cooling_min: 0.82,
            annual_cooling_max: 1.88,
            peak_heating_min: 1.90,
            peak_heating_max: 2.50,
            peak_cooling_min: 1.20,
            peak_cooling_max: 1.60,
            min_free_float_min: -7.0,
            min_free_float_max: -2.2,
            max_free_float_min: 38.5,
            max_free_float_max: 43.2,
        },
    );

    // Case 920 - East/West Windows (High Mass)
    data.insert(
        "920".to_string(),
        BenchmarkData {
            annual_heating_min: 3.26,
            annual_heating_max: 4.30,
            annual_cooling_min: 1.84,
            annual_cooling_max: 3.31,
            peak_heating_min: 2.10,
            peak_heating_max: 2.80,
            peak_cooling_min: 1.40,
            peak_cooling_max: 1.90,
            min_free_float_min: -5.8,
            min_free_float_max: -1.0,
            max_free_float_min: 40.2,
            max_free_float_max: 45.8,
        },
    );

    // Case 930 - East/West Shading (High Mass)
    data.insert(
        "930".to_string(),
        BenchmarkData {
            annual_heating_min: 4.14,
            annual_heating_max: 5.34,
            annual_cooling_min: 1.04,
            annual_cooling_max: 2.24,
            peak_heating_min: 2.30,
            peak_heating_max: 3.00,
            peak_cooling_min: 1.10,
            peak_cooling_max: 1.50,
            min_free_float_min: -5.2,
            min_free_float_max: -0.4,
            max_free_float_min: 39.5,
            max_free_float_max: 44.8,
        },
    );

    // Case 940 - Thermostat Setback (High Mass)
    data.insert(
        "940".to_string(),
        BenchmarkData {
            annual_heating_min: 0.79,
            annual_heating_max: 1.41,
            annual_cooling_min: 2.08,
            annual_cooling_max: 3.55,
            peak_heating_min: 1.90,
            peak_heating_max: 2.50,
            peak_cooling_min: 1.70,
            peak_cooling_max: 2.30,
            min_free_float_min: -6.2,
            min_free_float_max: -1.4,
            max_free_float_min: 40.8,
            max_free_float_max: 46.2,
        },
    );

    // Case 950 - Night Ventilation (High Mass)
    data.insert(
        "950".to_string(),
        BenchmarkData {
            annual_heating_min: 0.00,
            annual_heating_max: 0.00,
            annual_cooling_min: 0.39,
            annual_cooling_max: 0.92,
            peak_heating_min: 0.00,
            peak_heating_max: 0.00,
            peak_cooling_min: 0.70,
            peak_cooling_max: 0.90,
            min_free_float_min: -20.2,
            min_free_float_max: -17.8,
            max_free_float_min: 35.5,
            max_free_float_max: 38.5,
        },
    );

    // Case 900FF - Free Float (High Mass)
    data.insert(
        "900FF".to_string(),
        BenchmarkData {
            annual_heating_min: 0.00,
            annual_heating_max: 0.00,
            annual_cooling_min: 0.00,
            annual_cooling_max: 0.00,
            peak_heating_min: 0.00,
            peak_heating_max: 0.00,
            peak_cooling_min: 0.00,
            peak_cooling_max: 0.00,
            min_free_float_min: -6.4,
            min_free_float_max: -1.6,
            max_free_float_min: 41.8,
            max_free_float_max: 46.4,
        },
    );

    // Case 950FF - Free Float with Night Ventilation (High Mass)
    data.insert(
        "950FF".to_string(),
        BenchmarkData {
            annual_heating_min: 0.00,
            annual_heating_max: 0.00,
            annual_cooling_min: 0.00,
            annual_cooling_max: 0.00,
            peak_heating_min: 0.00,
            peak_heating_max: 0.00,
            peak_cooling_min: 0.00,
            peak_cooling_max: 0.00,
            min_free_float_min: -20.2,
            min_free_float_max: -17.8,
            max_free_float_min: 35.5,
            max_free_float_max: 38.5,
        },
    );

    // ==================== Special Cases ====================

    // Case 960 - Sunspace (2-zone)
    data.insert(
        "960".to_string(),
        BenchmarkData {
            annual_heating_min: 1.65,
            annual_heating_max: 2.45,
            annual_cooling_min: 1.55,
            annual_cooling_max: 2.78,
            peak_heating_min: 2.20,
            peak_heating_max: 2.90,
            peak_cooling_min: 1.50,
            peak_cooling_max: 2.00,
            min_free_float_min: -2.8,
            min_free_float_max: 6.0,
            max_free_float_min: 48.9,
            max_free_float_max: 55.3,
        },
    );

    // Case 195 - Solid Conduction (no windows, no infiltration, no loads)
    data.insert(
        "195".to_string(),
        BenchmarkData {
            annual_heating_min: 5.85,
            annual_heating_max: 7.25,
            annual_cooling_min: 0.00,
            annual_cooling_max: 0.00,
            peak_heating_min: 1.70,
            peak_heating_max: 2.20,
            peak_cooling_min: 0.00,
            peak_cooling_max: 0.00,
            min_free_float_min: -21.5,
            min_free_float_max: -18.2,
            max_free_float_min: 27.8,
            max_free_float_max: 32.5,
        },
    );

    data
}

/// Returns benchmark data for a specific case.
///
/// Returns `None` if the case is not found in the reference database.
pub fn get_benchmark_data(case_id: &str) -> Option<BenchmarkData> {
    get_all_benchmark_data().get(case_id).cloned()
}

/// Returns a list of all available case IDs.
pub fn get_all_case_ids() -> Vec<String> {
    let mut ids: Vec<String> = get_all_benchmark_data().keys().cloned().collect();
    ids.sort();
    ids
}

/// Returns a list of low mass case IDs (600 series).
pub fn get_low_mass_cases() -> Vec<String> {
    vec![
        "600".to_string(),
        "610".to_string(),
        "620".to_string(),
        "630".to_string(),
        "640".to_string(),
        "650".to_string(),
        "600FF".to_string(),
        "650FF".to_string(),
    ]
}

/// Returns a list of high mass case IDs (900 series).
pub fn get_high_mass_cases() -> Vec<String> {
    vec![
        "900".to_string(),
        "910".to_string(),
        "920".to_string(),
        "930".to_string(),
        "940".to_string(),
        "950".to_string(),
        "900FF".to_string(),
        "950FF".to_string(),
    ]
}

/// Returns a list of special case IDs.
pub fn get_special_cases() -> Vec<String> {
    vec!["960".to_string(), "195".to_string()]
}

/// Returns all case IDs grouped by category.
pub fn get_cases_by_category() -> HashMap<String, Vec<String>> {
    let mut categories = HashMap::new();
    categories.insert("low_mass".to_string(), get_low_mass_cases());
    categories.insert("high_mass".to_string(), get_high_mass_cases());
    categories.insert("special".to_string(), get_special_cases());
    categories
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_all_benchmark_data() {
        let data = get_all_benchmark_data();
        assert!(data.len() >= 18); // At least all standard cases

        // Check Case 600 exists
        assert!(data.contains_key("600"));
        assert!(data.contains_key("900"));
        assert!(data.contains_key("960"));
        assert!(data.contains_key("195"));
    }

    #[test]
    fn test_get_benchmark_data() {
        let case_600 = get_benchmark_data("600");
        assert!(case_600.is_some());

        let data = case_600.unwrap();
        assert_eq!(data.annual_heating_min, 4.30);
        assert_eq!(data.annual_heating_max, 5.71);
        assert_eq!(data.annual_cooling_min, 6.14);
        assert_eq!(data.annual_cooling_max, 8.45);
    }

    #[test]
    fn test_get_benchmark_data_invalid() {
        let invalid = get_benchmark_data("INVALID");
        assert!(invalid.is_none());
    }

    #[test]
    fn test_get_all_case_ids() {
        let ids = get_all_case_ids();
        assert!(ids.len() >= 18);
        assert!(ids.contains(&"600".to_string()));
        assert!(ids.contains(&"900".to_string()));
    }

    #[test]
    fn test_get_low_mass_cases() {
        let cases = get_low_mass_cases();
        assert_eq!(cases.len(), 8);
        assert!(cases.contains(&"600".to_string()));
        assert!(cases.contains(&"650FF".to_string()));
    }

    #[test]
    fn test_get_high_mass_cases() {
        let cases = get_high_mass_cases();
        assert_eq!(cases.len(), 8);
        assert!(cases.contains(&"900".to_string()));
        assert!(cases.contains(&"950FF".to_string()));
    }

    #[test]
    fn test_get_special_cases() {
        let cases = get_special_cases();
        assert_eq!(cases.len(), 2);
        assert!(cases.contains(&"960".to_string()));
        assert!(cases.contains(&"195".to_string()));
    }

    #[test]
    fn test_get_cases_by_category() {
        let categories = get_cases_by_category();
        assert_eq!(categories.len(), 3);
        assert!(categories.contains_key("low_mass"));
        assert!(categories.contains_key("high_mass"));
        assert!(categories.contains_key("special"));
    }

    #[test]
    fn test_case_600_data_completeness() {
        let data = get_benchmark_data("600").unwrap();

        // Verify all fields are populated
        assert!(data.annual_heating_min > 0.0);
        assert!(data.annual_heating_max > 0.0);
        assert!(data.annual_cooling_min > 0.0);
        assert!(data.annual_cooling_max > 0.0);
        assert!(data.peak_heating_min > 0.0);
        assert!(data.peak_heating_max > 0.0);
        assert!(data.peak_cooling_min > 0.0);
        assert!(data.peak_cooling_max > 0.0);
        assert!(data.min_free_float_min != 0.0);
        assert!(data.min_free_float_max != 0.0);
        assert!(data.max_free_float_min != 0.0);
        assert!(data.max_free_float_max != 0.0);
    }

    #[test]
    fn test_free_float_case_heating_cooling_zero() {
        // Free-floating cases should have zero heating/cooling
        let data_600ff = get_benchmark_data("600FF").unwrap();
        assert_eq!(data_600ff.annual_heating_min, 0.0);
        assert_eq!(data_600ff.annual_heating_max, 0.0);
        assert_eq!(data_600ff.annual_cooling_min, 0.0);
        assert_eq!(data_600ff.annual_cooling_max, 0.0);
        assert_eq!(data_600ff.peak_heating_min, 0.0);
        assert_eq!(data_600ff.peak_heating_max, 0.0);
        assert_eq!(data_600ff.peak_cooling_min, 0.0);
        assert_eq!(data_600ff.peak_cooling_max, 0.0);
    }

    #[test]
    fn test_high_mass_vs_low_mass_heating() {
        let data_600 = get_benchmark_data("600").unwrap();
        let data_900 = get_benchmark_data("900").unwrap();

        // High mass should have lower heating (thermal mass provides stability)
        assert!(data_900.annual_heating_max < data_600.annual_heating_min);
    }
}
