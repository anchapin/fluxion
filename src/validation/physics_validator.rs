//! Physics Validator Module
//!
//! This module validates that surrogate predictions satisfy physical laws,
//! implementing Issue #173: Create physics validator module.
//!
//! ## Validations Performed:
//! - Energy balance validation
//! - Temperature bound checking
//! - Physical law validators

/// Result of physics validation
#[derive(Debug, Clone)]
pub struct PhysicsValidationResult {
    /// Whether all validations passed
    pub passed: bool,
    /// Energy balance error (W/m²)
    pub energy_balance_error: f64,
    /// Temperature bound violations
    pub temperature_violations: Vec<TemperatureViolation>,
    /// Detailed messages
    pub messages: Vec<String>,
}

/// Temperature bound violation
#[derive(Debug, Clone)]
pub struct TemperatureViolation {
    /// Zone index
    pub zone: usize,
    /// The temperature that violated bounds
    pub temperature: f64,
    /// Minimum allowed temperature (°C)
    pub min_temp: f64,
    /// Maximum allowed temperature (°C)
    pub max_temp: f64,
}

/// Physics validator for surrogate predictions
pub struct PhysicsValidator {
    /// Minimum physically reasonable temperature (°C)
    pub min_temperature: f64,
    /// Maximum physically reasonable temperature (°C)
    pub max_temperature: f64,
    /// Maximum reasonable thermal load (W/m²)
    pub max_thermal_load: f64,
    /// Energy balance tolerance (W/m²)
    pub energy_balance_tolerance: f64,
}

impl Default for PhysicsValidator {
    fn default() -> Self {
        Self {
            min_temperature: -50.0,         // Extreme cold
            max_temperature: 100.0,         // Extreme heat
            max_thermal_load: 500.0,        // W/m² - reasonable max for buildings
            energy_balance_tolerance: 10.0, // W/m² tolerance
        }
    }
}

impl PhysicsValidator {
    /// Create a new physics validator with custom bounds
    pub fn new(min_temp: f64, max_temp: f64, max_load: f64, tolerance: f64) -> Self {
        Self {
            min_temperature: min_temp,
            max_temperature: max_temp,
            max_thermal_load: max_load,
            energy_balance_tolerance: tolerance,
        }
    }

    /// Validate surrogate predictions against physical laws
    pub fn validate(
        &self,
        temperatures: &[f64],
        thermal_loads: &[f64],
        outdoor_temp: f64,
        zone_area: f64,
    ) -> PhysicsValidationResult {
        let mut passed = true;
        let mut energy_balance_error = 0.0;
        let mut temperature_violations = Vec::new();
        let mut messages = Vec::new();

        // 1. Temperature bound checking
        for (zone, &temp) in temperatures.iter().enumerate() {
            if temp < self.min_temperature {
                temperature_violations.push(TemperatureViolation {
                    zone,
                    temperature: temp,
                    min_temp: self.min_temperature,
                    max_temp: self.max_temperature,
                });
                passed = false;
                messages.push(format!(
                    "Zone {}: Temperature {}°C below minimum {}°C",
                    zone, temp, self.min_temperature
                ));
            } else if temp > self.max_temperature {
                temperature_violations.push(TemperatureViolation {
                    zone,
                    temperature: temp,
                    min_temp: self.min_temperature,
                    max_temp: self.max_temperature,
                });
                passed = false;
                messages.push(format!(
                    "Zone {}: Temperature {}°C above maximum {}°C",
                    zone, temp, self.max_temperature
                ));
            }
        }

        // 2. Thermal load validation
        for (zone, &load) in thermal_loads.iter().enumerate() {
            if load.abs() > self.max_thermal_load {
                passed = false;
                messages.push(format!(
                    "Zone {}: Thermal load {}W/m² exceeds maximum {}W/m²",
                    zone, load, self.max_thermal_load
                ));
            }
        }

        // 3. Energy balance validation
        // Q = U * A * (T_indoor - T_outdoor)
        // For steady state, the thermal load should be proportional to temperature difference
        if !temperatures.is_empty() && zone_area > 0.0 {
            let avg_indoor_temp: f64 = temperatures.iter().sum::<f64>() / temperatures.len() as f64;
            let delta_t = avg_indoor_temp - outdoor_temp;

            // Expected load based on simple conduction model
            // Using typical U-value of 1.0 W/m²K as reference
            let expected_load = delta_t * 1.0 * zone_area;
            let actual_load: f64 = thermal_loads.iter().sum::<f64>();

            energy_balance_error = (actual_load - expected_load).abs() / zone_area;

            if energy_balance_error > self.energy_balance_tolerance {
                passed = false;
                messages.push(format!(
                    "Energy balance error: {}W/m² exceeds tolerance {}W/m²",
                    energy_balance_error, self.energy_balance_tolerance
                ));
            }
        }

        if passed {
            messages.push("All physics validations passed".to_string());
        }

        PhysicsValidationResult {
            passed,
            energy_balance_error,
            temperature_violations,
            messages,
        }
    }

    /// Validate that predictions are physically reasonable
    pub fn validate_prediction(&self, temperatures: &[f64], thermal_loads: &[f64]) -> bool {
        // Quick validation check
        for &temp in temperatures {
            if temp < self.min_temperature || temp > self.max_temperature {
                return false;
            }
        }

        for &load in thermal_loads {
            if load.abs() > self.max_thermal_load {
                return false;
            }
        }

        true
    }
}

/// Generate a validation report
pub fn generate_validation_report(results: &PhysicsValidationResult) -> String {
    let mut report = String::new();
    report.push_str("=== Physics Validation Report ===\n\n");

    report.push_str(&format!(
        "Overall Status: {}\n\n",
        if results.passed { "PASSED" } else { "FAILED" }
    ));

    report.push_str(&format!(
        "Energy Balance Error: {:.2} W/m²\n\n",
        results.energy_balance_error
    ));

    if !results.temperature_violations.is_empty() {
        report.push_str("Temperature Violations:\n");
        for v in &results.temperature_violations {
            report.push_str(&format!(
                "  Zone {}: {}°C (valid range: {} to {}°C)\n",
                v.zone, v.temperature, v.min_temp, v.max_temp
            ));
        }
        report.push('\n');
    }

    report.push_str("Messages:\n");
    for msg in &results.messages {
        report.push_str(&format!("  - {}\n", msg));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_bounds_valid() {
        let validator = PhysicsValidator::default();
        let temps = vec![20.0, 21.0, 22.0];
        let loads = vec![10.0, 15.0, 12.0];

        let result = validator.validate(&temps, &loads, 10.0, 20.0);
        assert!(result.passed);
    }

    #[test]
    fn test_temperature_bounds_violation() {
        let validator = PhysicsValidator::default();
        let temps = vec![-60.0]; // Below minimum
        let loads = vec![10.0];

        let result = validator.validate(&temps, &loads, 10.0, 20.0);
        assert!(!result.passed);
        assert!(!result.temperature_violations.is_empty());
    }

    #[test]
    fn test_thermal_load_exceeds_max() {
        let validator = PhysicsValidator::default();
        let temps = vec![20.0];
        let loads = vec![600.0]; // Exceeds 500 W/m² max

        let result = validator.validate(&temps, &loads, 10.0, 20.0);
        assert!(!result.passed);
    }

    #[test]
    fn test_quick_validation() {
        let validator = PhysicsValidator::default();
        assert!(validator.validate_prediction(&[20.0, 21.0], &[10.0, 15.0]));
        assert!(!validator.validate_prediction(&[-60.0], &[10.0]));
    }
}
