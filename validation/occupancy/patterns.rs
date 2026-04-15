// Standard occupancy patterns for building energy simulation
// Defines typical 24-hour occupancy schedules for different building types

use std::collections::HashMap;

/// Represents a 24-hour occupancy schedule with hourly values (0.0-1.0)
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancySchedule {
    pub name: String,
    pub hourly_values: [f64; 24],
    pub description: String,
}

impl OccupancySchedule {
    pub fn new(name: &str, hourly_values: [f64; 24], description: &str) -> Self {
        Self {
            name: name.to_string(),
            hourly_values,
            description: description.to_string(),
        }
    }

    /// Get occupancy value for specific hour (0-23)
    pub fn get_hourly_occupancy(&self, hour: usize) -> f64 {
        if hour < 24 {
            self.hourly_values[hour]
        } else {
            0.0
        }
    }

    /// Validate that all values are in valid range [0.0, 1.0]
    pub fn validate(&self) -> Result<(), String> {
        for (hour, &value) in self.hourly_values.iter().enumerate() {
            if value < 0.0 || value > 1.0 {
                return Err(format!("Invalid occupancy value {} at hour {} for pattern {}: must be between 0.0 and 1.0", value, hour, self.name));
            }
        }
        Ok(())
    }
}

/// Standard occupancy patterns for different building types
pub fn get_standard_occupancy_patterns() -> HashMap<String, OccupancySchedule> {
    let mut patterns = HashMap::new();

    // Residential occupancy pattern
    patterns.insert(
        "residential".to_string(),
        OccupancySchedule::new(
            "residential",
            [
                0.8, 0.8, 0.8, 0.8, 0.8, 0.8, // 0-5: night
                0.6, 0.4, 0.2, 0.1, 0.1, 0.1, // 6-11: morning
                0.1, 0.1, 0.1, 0.2, 0.4, 0.6, // 12-17: afternoon
                0.8, 0.9, 1.0, 1.0, 0.9, 0.8, // 18-23: evening
            ],
            "Typical residential occupancy with peaks in evening and morning",
        ),
    );

    // Commercial/office occupancy pattern
    patterns.insert(
        "commercial".to_string(),
        OccupancySchedule::new(
            "commercial",
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 0-5: night
                0.1, 0.3, 0.7, 0.9, 1.0, 1.0, // 6-11: morning
                1.0, 0.9, 0.8, 0.7, 0.5, 0.3, // 12-17: afternoon
                0.1, 0.0, 0.0, 0.0, 0.0, 0.0, // 18-23: evening
            ],
            "Commercial office occupancy with business hours peak",
        ),
    );

    // School occupancy pattern
    patterns.insert(
        "school".to_string(),
        OccupancySchedule::new(
            "school",
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 0-5: night
                0.0, 0.0, 0.8, 1.0, 1.0, 1.0, // 6-11: morning classes
                1.0, 1.0, 0.8, 0.5, 0.2, 0.0, // 12-17: afternoon classes
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 18-23: evening
            ],
            "School occupancy with class hours from 8am-4pm",
        ),
    );

    // Hospital occupancy pattern (24/7 with variations)
    patterns.insert(
        "hospital".to_string(),
        OccupancySchedule::new(
            "hospital",
            [
                0.7, 0.7, 0.7, 0.7, 0.7, 0.7, // 0-5: night shift
                0.8, 0.9, 1.0, 1.0, 1.0, 1.0, // 6-11: morning
                1.0, 1.0, 0.9, 0.8, 0.8, 0.8, // 12-17: afternoon
                0.8, 0.8, 0.8, 0.7, 0.7, 0.7, // 18-23: evening
            ],
            "Hospital occupancy with 24/7 operation and peak daytime hours",
        ),
    );

    // Retail occupancy pattern
    patterns.insert(
        "retail".to_string(),
        OccupancySchedule::new(
            "retail",
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 0-5: night
                0.2, 0.5, 0.8, 1.0, 1.0, 1.0, // 6-11: morning
                1.0, 0.9, 0.8, 0.7, 0.6, 0.4, // 12-17: afternoon
                0.3, 0.2, 0.1, 0.0, 0.0, 0.0, // 18-23: evening
            ],
            "Retail store occupancy with extended business hours",
        ),
    );

    patterns
}

/// Get a specific occupancy pattern by name
pub fn get_occupancy_pattern(name: &str) -> Option<OccupancySchedule> {
    get_standard_occupancy_patterns().get(name).cloned()
}

/// Validate all standard occupancy patterns
pub fn validate_all_patterns() -> Result<(), Vec<String>> {
    let patterns = get_standard_occupancy_patterns();
    let mut errors = Vec::new();

    for (name, pattern) in patterns.iter() {
        if let Err(err) = pattern.validate() {
            errors.push(format!("Pattern {}: {}", name, err));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residential_pattern() {
        let pattern = get_occupancy_pattern("residential").unwrap();
        assert_eq!(pattern.name, "residential");
        assert_eq!(pattern.hourly_values[0], 0.8); // Midnight
        assert_eq!(pattern.hourly_values[12], 0.1); // Noon
        assert_eq!(pattern.hourly_values[18], 0.8); // 6 PM
    }

    #[test]
    fn test_commercial_pattern() {
        let pattern = get_occupancy_pattern("commercial").unwrap();
        assert_eq!(pattern.name, "commercial");
        assert_eq!(pattern.hourly_values[8], 0.9); // 8 AM
        assert_eq!(pattern.hourly_values[12], 1.0); // Noon
        assert_eq!(pattern.hourly_values[17], 0.3); // 5 PM
    }

    #[test]
    fn test_pattern_validation() {
        let pattern = get_occupancy_pattern("residential").unwrap();
        assert!(pattern.validate().is_ok());
    }

    #[test]
    fn test_all_patterns_valid() {
        assert!(validate_all_patterns().is_ok());
    }
}
