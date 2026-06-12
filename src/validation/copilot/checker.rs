//! Rule-based BEM input validation checker
//!
//! This module provides rule-based validation checks for building energy model
//! configurations. These checks run without LLM assistance and catch common
//! setup errors.

use crate::validation::copilot::types::BemIssue;
#[allow(unused_imports)]
use crate::validation::copilot::types::BemIssueSeverity;
use serde_json::Value;

/// BEM input validation checker
pub struct BemChecker {
    // Could add configuration options here
}

impl BemChecker {
    /// Create a new BEM checker
    pub fn new() -> Self {
        Self {}
    }

    /// Check a BEM configuration for issues
    pub fn check(&self, config_json: &str) -> Vec<BemIssue> {
        let mut issues = Vec::new();

        // Parse the configuration
        let config: Value = match serde_json::from_str(config_json) {
            Ok(v) => v,
            Err(e) => {
                issues.push(BemIssue::error(
                    "parsing",
                    "configuration",
                    &format!("Failed to parse JSON configuration: {}", e),
                ));
                return issues;
            }
        };

        // Run all validation checks
        self.check_required_fields(&config, &mut issues);
        self.check_window_to_wall_ratio(&config, &mut issues);
        self.check_internal_loads(&config, &mut issues);
        self.check_material_properties(&config, &mut issues);
        self.check_hvac_configuration(&config, &mut issues);
        self.check_thermal_zones(&config, &mut issues);
        self.check_weather_data(&config, &mut issues);
        self.check_baseline_assumptions(&config, &mut issues);

        issues
    }

    /// Check for required fields
    fn check_required_fields(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        let required_fields = [
            ("building_type", "Building type must be specified"),
            ("climate_zone", "Climate zone must be specified"),
            ("floor_area", "Floor area must be specified"),
            (
                "window_wall_ratio",
                "Window-to-wall ratio must be specified",
            ),
        ];

        for (field, message) in required_fields {
            if !config.get(field).is_some() {
                issues.push(BemIssue::error("required_fields", field, message));
            }
        }

        // Check for location
        if !config.get("latitude").is_some() || !config.get("longitude").is_some() {
            issues.push(BemIssue::warning(
                "required_fields",
                "location",
                "Location (latitude/longitude) not specified - required for solar calculations",
            ));
        }
    }

    /// Check window-to-wall ratio for physical plausibility
    fn check_window_to_wall_ratio(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        let wwr = match config.get("window_wall_ratio") {
            Some(v) => v.as_f64().unwrap_or(-1.0),
            None => return, // Already reported as missing
        };

        // ASHRAE 90.1 typically limits WWR to 40% for baseline
        if wwr < 0.0 {
            issues.push(BemIssue::error(
                "window_to_wall_ratio",
                "window_wall_ratio",
                "Window-to-wall ratio must be non-negative",
            ));
            return;
        }

        if wwr > 0.95 {
            issues.push(BemIssue::error(
                "window_to_wall_ratio",
                "window_wall_ratio",
                &format!(
                    "Window-to-wall ratio of {:.1}% is physically impossible (>95%)",
                    wwr * 100.0
                ),
            ));
        } else if wwr > 0.70 {
            issues.push(BemIssue::error(
                "window_to_wall_ratio",
                "window_wall_ratio",
                &format!(
                    "Window-to-wall ratio of {:.1}% exceeds practical limits (>70%)",
                    wwr * 100.0
                ),
            ));
        } else if wwr > 0.50 {
            issues.push(BemIssue::warning(
                "window_to_wall_ratio",
                "window_wall_ratio",
                &format!(
                    "Window-to-wall ratio of {:.1}% is very high (typical range: 15-40%)",
                    wwr * 100.0
                ),
            ));
        } else if wwr > 0.40 {
            issues.push(BemIssue::hint(
                "window_to_wall_ratio",
                "window_wall_ratio",
                &format!("Window-to-wall ratio of {:.1}% is high (ASHRAE 90.1 baseline typically uses 20-30%)", wwr * 100.0),
            ));
        }

        // Check for zero WWR (no windows)
        if wwr == 0.0 {
            issues.push(BemIssue::info(
                "window_to_wall_ratio",
                "window_wall_ratio",
                "Window-to-wall ratio is 0 - no windows modeled (rare for real buildings)",
            ));
        }

        // Check window properties if WWR is reasonable
        if wwr > 0.0 && wwr <= 0.70 {
            self.check_window_properties(config, issues);
        }
    }

    /// Check window properties (U-factor, SHGC)
    fn check_window_properties(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        let window = match config.get("window") {
            Some(v) => v,
            None => {
                issues.push(BemIssue::warning(
                    "window_to_wall_ratio",
                    "window",
                    "Window properties not specified - will use defaults",
                ));
                return;
            }
        };

        // Check U-factor
        if let Some(u_factor) = window.get("u_factor").and_then(|v| v.as_f64()) {
            if u_factor <= 0.0 {
                issues.push(BemIssue::error(
                    "window_properties",
                    "u_factor",
                    "Window U-factor must be positive",
                ));
            } else if u_factor > 10.0 {
                issues.push(BemIssue::error(
                    "window_properties",
                    "u_factor",
                    &format!(
                        "Window U-factor of {:.2} W/m²K is unrealistically high (typical: 1.5-3.0)",
                        u_factor
                    ),
                ));
            } else if u_factor > 5.0 {
                issues.push(BemIssue::warning(
                    "window_properties",
                    "u_factor",
                    &format!("Window U-factor of {:.2} W/m²K is high (ASHRAE 90.1 max: ~2.4 for most climates)", u_factor),
                ));
            }
        }

        // Check SHGC
        if let Some(shgc) = window.get("shgc").and_then(|v| v.as_f64()) {
            if shgc < 0.0 || shgc > 1.0 {
                issues.push(BemIssue::error(
                    "window_properties",
                    "shgc",
                    "Solar Heat Gain Coefficient must be in range [0, 1]",
                ));
            } else if shgc > 0.9 {
                issues.push(BemIssue::warning(
                    "window_properties",
                    "shgc",
                    &format!("SHGC of {:.2} is very high (typical range: 0.2-0.7)", shgc),
                ));
            }
        }
    }

    /// Check internal loads and schedules
    fn check_internal_loads(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        // Check lighting power density
        if let Some(lpd) = config
            .get("lighting_power_density")
            .and_then(|v| v.as_f64())
        {
            if lpd <= 0.0 {
                issues.push(BemIssue::error(
                    "internal_loads",
                    "lighting_power_density",
                    "Lighting power density must be positive",
                ));
            } else if lpd > 50.0 {
                issues.push(BemIssue::error(
                    "internal_loads",
                    "lighting_power_density",
                    &format!(
                        "Lighting power density of {:.1} W/m² is unrealistically high",
                        lpd
                    ),
                ));
            } else if lpd > 20.0 {
                issues.push(BemIssue::warning(
                    "internal_loads",
                    "lighting_power_density",
                    &format!("Lighting power density of {:.1} W/m² is high (ASHRAE 90.1 allows ~10-15 for offices)", lpd),
                ));
            }
        }

        // Check equipment power density
        if let Some(epd) = config
            .get("equipment_power_density")
            .and_then(|v| v.as_f64())
        {
            if epd <= 0.0 {
                issues.push(BemIssue::error(
                    "internal_loads",
                    "equipment_power_density",
                    "Equipment power density must be positive",
                ));
            } else if epd > 30.0 {
                issues.push(BemIssue::warning(
                    "internal_loads",
                    "equipment_power_density",
                    &format!(
                        "Equipment power density of {:.1} W/m² is high (typical: 5-15 for offices)",
                        epd
                    ),
                ));
            }
        }

        // Check for occupancy density
        if let Some(occ) = config.get("occupancy_density").and_then(|v| v.as_f64()) {
            if occ <= 0.0 {
                issues.push(BemIssue::error(
                    "internal_loads",
                    "occupancy_density",
                    "Occupancy density must be positive",
                ));
            } else if occ > 1.0 {
                issues.push(BemIssue::warning(
                    "internal_loads",
                    "occupancy_density",
                    &format!(
                        "Occupancy density of {:.2} persons/m² is very high (typical: 0.05-0.2)",
                        occ
                    ),
                ));
            }
        }

        // Check if schedules are defined
        let schedule_fields = [
            "lighting_schedule",
            "occupancy_schedule",
            "equipment_schedule",
        ];
        for field in schedule_fields {
            if !config.get(field).is_some() {
                issues.push(BemIssue::warning(
                    "internal_loads",
                    field,
                    &format!("{} not defined - constant loads will be assumed", field),
                ));
            }
        }
    }

    /// Check material properties
    fn check_material_properties(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        // Check wall assembly if defined
        if let Some(wall) = config.get("wall") {
            self.check_assembly_layers(wall, "wall", issues);
        }

        // Check roof assembly if defined
        if let Some(roof) = config.get("roof") {
            self.check_assembly_layers(roof, "roof", issues);
        }

        // Check floor assembly if defined
        if let Some(floor) = config.get("floor") {
            self.check_assembly_layers(floor, "floor", issues);
        }
    }

    /// Check assembly material layers
    fn check_assembly_layers(
        &self,
        assembly: &Value,
        assembly_type: &str,
        issues: &mut Vec<BemIssue>,
    ) {
        let layers = match assembly.get("layers") {
            Some(v) => v,
            None => {
                issues.push(BemIssue::error(
                    "material_properties",
                    assembly_type,
                    &format!("{} assembly has no material layers defined", assembly_type),
                ));
                return;
            }
        };

        let layers_array = match layers.as_array() {
            Some(v) => v,
            None => {
                issues.push(BemIssue::error(
                    "material_properties",
                    &format!("{}.layers", assembly_type),
                    "Layers must be an array",
                ));
                return;
            }
        };

        if layers_array.is_empty() {
            issues.push(BemIssue::error(
                "material_properties",
                &format!("{}.layers", assembly_type),
                &format!("{} assembly has no material layers", assembly_type),
            ));
            return;
        }

        for (idx, layer) in layers_array.iter().enumerate() {
            let path = format!("{}.layers[{}]", assembly_type, idx);

            // Check thickness
            if let Some(thickness) = layer.get("thickness").and_then(|v| v.as_f64()) {
                if thickness <= 0.0 {
                    issues.push(BemIssue::error(
                        "material_properties",
                        &format!("{}.thickness", path),
                        "Material thickness must be positive",
                    ));
                } else if thickness > 1.0 {
                    issues.push(BemIssue::warning(
                        "material_properties",
                        &format!("{}.thickness", path),
                        &format!(
                            "Material thickness of {:.3} m seems unusually thick",
                            thickness
                        ),
                    ));
                }
            }

            // Check conductivity
            if let Some(conductivity) = layer.get("conductivity").and_then(|v| v.as_f64()) {
                if conductivity <= 0.0 {
                    issues.push(BemIssue::error(
                        "material_properties",
                        &format!("{}.conductivity", path),
                        "Thermal conductivity must be positive",
                    ));
                } else if conductivity > 5.0 {
                    issues.push(BemIssue::warning(
                        "material_properties",
                        &format!("{}.conductivity", path),
                        &format!("Thermal conductivity of {:.3} W/mK is very high (metal?), check if correct", conductivity),
                    ));
                }
            }

            // Check density
            if let Some(density) = layer.get("density").and_then(|v| v.as_f64()) {
                if density <= 0.0 {
                    issues.push(BemIssue::error(
                        "material_properties",
                        &format!("{}.density", path),
                        "Material density must be positive",
                    ));
                } else if density > 10000.0 {
                    issues.push(BemIssue::warning(
                        "material_properties",
                        &format!("{}.density", path),
                        &format!("Material density of {:.0} kg/m³ is extremely high", density),
                    ));
                }
            }
        }
    }

    /// Check HVAC configuration
    fn check_hvac_configuration(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        // Check for HVAC system definition
        if !config.get("hvac_system").is_some() {
            issues.push(BemIssue::info(
                "hvac_configuration",
                "hvac_system",
                "No HVAC system defined - assuming ideal loads",
            ));
            return;
        }

        let hvac = config.get("hvac_system").unwrap();

        // Check cooling setpoint
        if let Some(cooling) = hvac.get("cooling_setpoint").and_then(|v| v.as_f64()) {
            if cooling < 18.0 || cooling > 30.0 {
                issues.push(BemIssue::warning(
                    "hvac_configuration",
                    "cooling_setpoint",
                    &format!(
                        "Cooling setpoint of {:.1}°C is outside typical range (18-30°C)",
                        cooling
                    ),
                ));
            }
        }

        // Check heating setpoint
        if let Some(heating) = hvac.get("heating_setpoint").and_then(|v| v.as_f64()) {
            if heating < 10.0 || heating > 25.0 {
                issues.push(BemIssue::warning(
                    "hvac_configuration",
                    "heating_setpoint",
                    &format!(
                        "Heating setpoint of {:.1}°C is outside typical range (10-25°C)",
                        heating
                    ),
                ));
            }
        }

        // Check for setpoint conflicts
        let cooling = hvac
            .get("cooling_setpoint")
            .and_then(|v| v.as_f64())
            .unwrap_or(24.0);
        let heating = hvac
            .get("heating_setpoint")
            .and_then(|v| v.as_f64())
            .unwrap_or(20.0);
        if cooling <= heating {
            issues.push(BemIssue::error(
                "hvac_configuration",
                "setpoints",
                &format!(
                    "Cooling setpoint ({:.1}°C) must be greater than heating setpoint ({:.1}°C)",
                    cooling, heating
                ),
            ));
        }
    }

    /// Check thermal zone configuration
    fn check_thermal_zones(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        // Check zone definition
        if !config.get("zones").is_some() {
            issues.push(BemIssue::warning(
                "thermal_zones",
                "zones",
                "No zones defined - single zone model assumed",
            ));
            return;
        }

        let zones = match config.get("zones").unwrap().as_array() {
            Some(v) => v,
            None => {
                issues.push(BemIssue::error(
                    "thermal_zones",
                    "zones",
                    "Zones must be an array",
                ));
                return;
            }
        };

        if zones.is_empty() {
            issues.push(BemIssue::error(
                "thermal_zones",
                "zones",
                "At least one thermal zone must be defined",
            ));
        }

        // Check zone volume
        for (idx, zone) in zones.iter().enumerate() {
            if let Some(volume) = zone.get("volume").and_then(|v| v.as_f64()) {
                if volume <= 0.0 {
                    issues.push(BemIssue::error(
                        "thermal_zones",
                        &format!("zones[{}].volume", idx),
                        "Zone volume must be positive",
                    ));
                } else if volume > 100000.0 {
                    issues.push(BemIssue::warning(
                        "thermal_zones",
                        &format!("zones[{}].volume", idx),
                        &format!(
                            "Zone volume of {:.0} m³ is very large - verify this is intentional",
                            volume
                        ),
                    ));
                }
            }
        }
    }

    /// Check weather data configuration
    fn check_weather_data(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        // Check for weather file
        if !config.get("weather_file").is_some() {
            issues.push(BemIssue::warning(
                "weather_data",
                "weather_file",
                "No weather file specified - simulation may fail",
            ));
        }

        // Check climate zone
        if let Some(cz) = config.get("climate_zone").and_then(|v| v.as_str()) {
            // Validate climate zone format (ASHRAE format: e.g., "4A", "6B")
            if !cz.is_empty() {
                let valid = cz.len() >= 2
                    && cz.chars().next().unwrap().is_ascii_digit()
                    && ['A', 'B', 'a', 'b'].contains(&cz.chars().nth(1).unwrap_or(' '));

                if !valid && cz.len() < 5 {
                    issues.push(BemIssue::hint(
                        "weather_data",
                        "climate_zone",
                        &format!("Climate zone '{}' may not be in standard ASHRAE format (e.g., '4A', '6B')", cz),
                    ));
                }
            }
        }
    }

    /// Check ASHRAE 90.1 baseline assumptions
    fn check_baseline_assumptions(&self, config: &Value, issues: &mut Vec<BemIssue>) {
        // Check if this is meant to be a baseline comparison
        if config
            .get("is_baseline")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            // Baseline specific checks
            if let Some(wwr) = config.get("window_wall_ratio").and_then(|v| v.as_f64()) {
                if wwr > 0.40 {
                    issues.push(BemIssue::warning(
                        "baseline_assumptions",
                        "window_wall_ratio",
                        "ASHRAE 90.1 baseline typically uses 20-30% WWR for compliance",
                    ));
                }
            }
        }

        // Check for solar panel fraction (should not exceed roof area)
        if let Some(pv) = config.get("pv_fraction").and_then(|v| v.as_f64()) {
            if pv < 0.0 || pv > 1.0 {
                issues.push(BemIssue::error(
                    "baseline_assumptions",
                    "pv_fraction",
                    "PV fraction must be between 0 and 1",
                ));
            } else if pv > 0.7 {
                issues.push(BemIssue::warning(
                    "baseline_assumptions",
                    "pv_fraction",
                    &format!(
                        "PV fraction of {:.0}% is very high - verify roof area is sufficient",
                        pv * 100.0
                    ),
                ));
            }
        }
    }
}

impl Default for BemChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checker_empty_config() {
        let checker = BemChecker::new();
        let issues = checker.check("{}");

        // Should have multiple missing field errors
        let categories: Vec<_> = issues.iter().map(|i| i.category.as_str()).collect();
        assert!(categories.contains(&"required_fields"));
    }

    #[test]
    fn test_checker_invalid_wwr() {
        let checker = BemChecker::new();

        // Test very high WWR
        let issues = checker.check(r#"{"window_wall_ratio": 1.5}"#);
        let wwr_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.category == "window_to_wall_ratio")
            .collect();
        assert!(!wwr_issues.is_empty());
        assert_eq!(wwr_issues[0].severity, BemIssueSeverity::Error);

        // Test zero WWR
        let issues = checker.check(r#"{"window_wall_ratio": 0.0}"#);
        let wwr_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.category == "window_to_wall_ratio")
            .collect();
        assert!(!wwr_issues.is_empty());
        assert_eq!(wwr_issues[0].severity, BemIssueSeverity::Info);
    }

    #[test]
    fn test_checker_setpoint_conflict() {
        let checker = BemChecker::new();
        let config = r#"{
            "hvac_system": {
                "cooling_setpoint": 18.0,
                "heating_setpoint": 22.0
            }
        }"#;
        let issues = checker.check(config);

        let setpoint_issues: Vec<_> = issues.iter().filter(|i| i.field == "setpoints").collect();
        assert!(!setpoint_issues.is_empty());
    }

    #[test]
    fn test_checker_valid_config() {
        let checker = BemChecker::new();
        let config = r#"{
            "building_type": "office",
            "climate_zone": "4A",
            "floor_area": 1000.0,
            "window_wall_ratio": 0.3,
            "latitude": 39.7,
            "longitude": -105.0
        }"#;
        let issues = checker.check(config);

        // Should have no errors
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == BemIssueSeverity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "Expected no errors but got: {:?}",
            errors
        );
    }
}
