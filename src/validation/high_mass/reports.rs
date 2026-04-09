//! High-mass validation report generation.
//!
//! This module provides comprehensive report generation capabilities
//! for high-mass building validation, including detailed diagnostics,
//! construction analysis, and ASHRAE 140 compliance reporting.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::physics::thermal_mass::diagnostics::ThermalMassDiagnostics;
use crate::thermal::mass::types::ConstructionType;
use crate::validation::high_mass::metrics::HighMassMetrics;
use crate::validation::report::{MetricType, ValidationStatus};
use crate::validation::tolerance::ValidationTolerance;

/// Weather summary for validation reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherSummary {
    /// Location of weather data
    pub location: String,
    /// Weather data period
    pub period: String,
    /// Average outdoor temperature (°C)
    pub avg_temperature: f64,
    /// Total heating degree days (base 18°C)
    pub heating_degree_days: f64,
    /// Total cooling degree days (base 18°C)
    pub cooling_degree_days: f64,
}

impl Default for WeatherSummary {
    fn default() -> Self {
        Self {
            location: "ASHRAE 140 Reference".to_string(),
            period: "Annual".to_string(),
            avg_temperature: 12.0,
            heating_degree_days: 2500.0,
            cooling_degree_days: 1000.0,
        }
    }
}

/// Comprehensive high-mass validation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighMassValidationReport {
    /// ASHRAE 140 case identifier
    pub case_id: String,
    /// Building description and type
    pub building_description: String,
    /// Weather summary for the validation period
    pub weather_summary: WeatherSummary,
    /// Validation metrics
    pub metrics: HighMassMetrics,
    /// Thermal mass diagnostics
    pub diagnostics: ThermalMassDiagnostics,
    /// Construction type information
    pub construction_type: ConstructionType,
    /// Report generation timestamp
    pub timestamp: DateTime<Utc>,
    /// Overall pass/fail status
    pub passed: bool,
    /// Validation tolerance used
    pub tolerance: ValidationTolerance,
}

impl HighMassValidationReport {
    /// Create a new high-mass validation report from validation result.
    ///
    /// # Arguments
    /// * `result` - Validation result containing metrics and diagnostics
    ///
    /// # Returns
    /// A new HighMassValidationReport instance
    pub fn generate_report(
        case_id: &str,
        building_description: &str,
        weather_summary: WeatherSummary,
        metrics: HighMassMetrics,
        diagnostics: ThermalMassDiagnostics,
        construction_type: ConstructionType,
        tolerance: ValidationTolerance,
    ) -> Self {
        let passed = metrics.within_tolerance(&tolerance);

        Self {
            case_id: case_id.to_string(),
            building_description: building_description.to_string(),
            weather_summary,
            metrics,
            diagnostics,
            construction_type,
            timestamp: Utc::now(),
            passed,
            tolerance,
        }
    }

    /// Generate a comprehensive markdown report.
    ///
    /// # Returns
    /// Markdown-formatted string containing the full validation report
    pub fn generate_markdown(&self) -> String {
        let mut output = String::new();

        // Title
        output.push_str(&format!(
            "# High-Mass Validation Report: Case {}",
            self.case_id
        ));
        output.push_str("\n\n");

        // Case Information
        output.push_str("## Case Information\n\n");
        output.push_str(&format!("- **Case ID:** {}\n", self.case_id));
        output.push_str(&format!(
            "- **Building Type:** {}\n",
            self.building_description
        ));
        output.push_str(&format!(
            "- **Construction Type:** {:?}\n",
            self.construction_type
        ));
        output.push_str(&format!(
            "- **Timestamp:** {}\n",
            self.timestamp.to_rfc3339()
        ));
        output.push_str(&format!(
            "- **Status:** {}\n",
            if self.passed { "PASS ✓" } else { "FAIL ✗" }
        ));
        output.push_str("\n");

        // Weather Summary
        output.push_str("## Weather Summary\n\n");
        output.push_str(&format!(
            "- **Location:** {}\n",
            self.weather_summary.location
        ));
        output.push_str(&format!("- **Period:** {}\n", self.weather_summary.period));
        output.push_str(&format!(
            "- **Average Temperature:** {:.1}°C\n",
            self.weather_summary.avg_temperature
        ));
        output.push_str(&format!(
            "- **Heating Degree Days (18°C base):** {:.0}\n",
            self.weather_summary.heating_degree_days
        ));
        output.push_str(&format!(
            "- **Cooling Degree Days (18°C base):** {:.0}\n",
            self.weather_summary.cooling_degree_days
        ));
        output.push_str("\n");

        // Validation Metrics
        output.push_str("## Validation Metrics\n\n");
        output.push_str("| Metric | Value | Tolerance | Status |\n");
        output.push_str("|--------|-------|-----------|--------|\n");

        // NMBE metrics
        output.push_str(&format!(
            "| NMBE Heating | {:.2}% | ±{:.1}% | {}|\n",
            self.metrics.nmbe_heating,
            self.tolerance.nmbe_limit,
            self.get_metric_status(self.metrics.nmbe_heating.abs(), self.tolerance.nmbe_limit)
        ));
        output.push_str(&format!(
            "| NMBE Cooling | {:.2}% | ±{:.1}% | {}|\n",
            self.metrics.nmbe_cooling,
            self.tolerance.nmbe_limit,
            self.get_metric_status(self.metrics.nmbe_cooling.abs(), self.tolerance.nmbe_limit)
        ));

        // CV(RMSE) metrics
        output.push_str(&format!(
            "| CV(RMSE) Heating | {:.2}% | ≤{:.1}% | {}|\n",
            self.metrics.cv_rmse_heating,
            self.tolerance.cv_rmse_limit,
            self.get_metric_status(self.metrics.cv_rmse_heating, self.tolerance.cv_rmse_limit)
        ));
        output.push_str(&format!(
            "| CV(RMSE) Cooling | {:.2}% | ≤{:.1}% | {}|\n",
            self.metrics.cv_rmse_cooling,
            self.tolerance.cv_rmse_limit,
            self.get_metric_status(self.metrics.cv_rmse_cooling, self.tolerance.cv_rmse_limit)
        ));

        // MAE metrics
        output.push_str(&format!(
            "| MAE Heating | {:.4} kWh | ≤{:.2} kWh | {}|\n",
            self.metrics.mae_heating,
            self.tolerance.mae_limit,
            self.get_metric_status(self.metrics.mae_heating, self.tolerance.mae_limit)
        ));
        output.push_str(&format!(
            "| MAE Cooling | {:.4} kWh | ≤{:.2} kWh | {}|\n",
            self.metrics.mae_cooling,
            self.tolerance.mae_limit,
            self.get_metric_status(self.metrics.mae_cooling, self.tolerance.mae_limit)
        ));

        // Max Error metrics
        output.push_str(&format!(
            "| Max Error Heating | {:.4} kWh | - | INFO |\n",
            self.metrics.max_error_heating
        ));
        output.push_str(&format!(
            "| Max Error Cooling | {:.4} kWh | - | INFO |\n",
            self.metrics.max_error_cooling
        ));
        output.push_str("\n");

        // Thermal Mass Diagnostics
        output.push_str("## Thermal Mass Diagnostics\n\n");
        output.push_str(&format!(
            "- **Effective Capacitance:** {:.1} kJ/m²K\n",
            self.diagnostics.calculate_effective_capacitance()
        ));
        output.push_str(&format!(
            "- **Time Constant:** {:.1} hours\n",
            self.diagnostics.calculate_time_constant()
        ));
        output.push_str(&format!(
            "- **Damping Factor:** {:.3}\n",
            self.diagnostics.calculate_damping_factor()
        ));
        output.push_str(&format!(
            "- **Classification:** {}\n",
            self.diagnostics.classify_thermal_mass()
        ));
        output.push_str("\n");

        // Construction Analysis
        output.push_str("## Construction Analysis\n\n");
        output.push_str(&format!(
            "- **Construction Type:** {:?}\n",
            self.construction_type
        ));

        let props = self.construction_type.thermal_mass_properties();
        output.push_str(&format!(
            "- **Typical Capacitance:** {:.1} kJ/m²K\n",
            props.effective_capacitance
        ));
        output.push_str(&format!(
            "- **Typical Time Constant:** {:.1} hours\n",
            props.time_constant
        ));
        output.push_str(&format!(
            "- **Typical Damping Factor:** {:.3}\n",
            props.damping_factor
        ));
        output.push_str(&format!(
            "- **ISO 13790 Classification:** {}\n",
            self.construction_type.classification()
        ));
        output.push_str("\n");

        // Overall Assessment
        output.push_str("## Overall Assessment\n\n");
        if self.passed {
            output.push_str("✅ **VALIDATION PASSED**\n\n");
            output.push_str(
                "This high-mass building simulation meets all ASHRAE 140 validation criteria.\n",
            );
            output.push_str(
                "All metrics are within specified tolerance bands, indicating good agreement\n",
            );
            output.push_str("between Fluxion simulations and reference data.\n");
        } else {
            output.push_str("❌ **VALIDATION FAILED**\n\n");
            output.push_str("This high-mass building simulation does not meet all ASHRAE 140 validation criteria.\n");
            output.push_str("Review the metrics above to identify areas where performance deviates from expectations.\n");
        }

        output.push_str("\n");

        // Recommendations
        output.push_str("## Recommendations\n\n");
        if !self.passed {
            output.push_str("- **Review construction properties:** Verify material properties and layer thicknesses\n");
            output.push_str(
                "- **Check weather data:** Ensure weather data matches reference conditions\n",
            );
            output.push_str("- **Examine simulation parameters:** Review timestep, convergence criteria, and boundary conditions\n");
            output.push_str("- **Consider thermal mass tuning:** Adjust effective capacitance calculations if systematic biases are observed\n");
        } else {
            output.push_str("- **Proceed with confidence:** Validation results indicate reliable high-mass simulation performance\n");
            output.push_str(
                "- **Monitor performance:** Continue to track metrics during production use\n",
            );
        }

        output
    }

    /// Generate a JSON report for machine processing.
    ///
    /// # Returns
    /// JSON-formatted string containing the validation report
    pub fn generate_json(&self) -> serde_json::Value {
        serde_json::json!({
            "case_id": self.case_id,
            "building_description": self.building_description,
            "construction_type": format!("{:?}", self.construction_type),
            "timestamp": self.timestamp.to_rfc3339(),
            "passed": self.passed,
            "weather_summary": {
                "location": self.weather_summary.location,
                "period": self.weather_summary.period,
                "avg_temperature": self.weather_summary.avg_temperature,
                "heating_degree_days": self.weather_summary.heating_degree_days,
                "cooling_degree_days": self.weather_summary.cooling_degree_days
            },
            "metrics": {
                "nmbe_heating": self.metrics.nmbe_heating,
                "nmbe_cooling": self.metrics.nmbe_cooling,
                "cv_rmse_heating": self.metrics.cv_rmse_heating,
                "cv_rmse_cooling": self.metrics.cv_rmse_cooling,
                "mae_heating": self.metrics.mae_heating,
                "mae_cooling": self.metrics.mae_cooling,
                "max_error_heating": self.metrics.max_error_heating,
                "max_error_cooling": self.metrics.max_error_cooling
            },
            "diagnostics": {
                "effective_capacitance": self.diagnostics.calculate_effective_capacitance(),
                "time_constant": self.diagnostics.calculate_time_constant(),
                "damping_factor": self.diagnostics.calculate_damping_factor(),
                "classification": self.diagnostics.classify_thermal_mass()
            },
            "tolerance": {
                "nmbe_limit": self.tolerance.nmbe_limit,
                "cv_rmse_limit": self.tolerance.cv_rmse_limit,
                "mae_limit": self.tolerance.mae_limit
            }
        })
    }

    /// Save the report to a file.
    ///
    /// # Arguments
    /// * `path` - File path to save the report
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        let content = match path.extension().and_then(|e| e.to_str()) {
            Some("md") | Some("txt") => self.generate_markdown(),
            Some("json") => self.generate_json().to_string(),
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Unsupported file extension. Use .md, .txt, or .json",
                ));
            }
        };

        fs::write(path, content)
    }

    /// Get status string for a metric value against tolerance.
    ///
    /// # Arguments
    /// * `value` - Metric value
    /// * `limit` - Tolerance limit
    ///
    /// # Returns
    /// Status string (PASS/WARN/FAIL)
    fn get_metric_status(&self, value: f64, limit: f64) -> String {
        if value <= limit * 0.8 {
            "PASS ✓".to_string()
        } else if value <= limit {
            "WARN ⚠".to_string()
        } else {
            "FAIL ✗".to_string()
        }
    }
}

impl fmt::Display for HighMassValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "High-Mass Validation Report for Case {}", self.case_id)?;
        write!(f, "\nBuilding: {}", self.building_description)?;
        write!(f, "\nConstruction: {:?}", self.construction_type)?;
        write!(f, "\nStatus: {}", if self.passed { "PASS" } else { "FAIL" })?;
        write!(f, "\nTimestamp: {}", self.timestamp.to_rfc3339())?;

        let metrics = self.generate_json();
        write!(f, "\nMetrics: {}", metrics)?;

        Ok(())
    }
}

/// Combined report for multiple high-mass validation cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedHighMassReport {
    /// Individual case reports
    pub case_reports: Vec<HighMassValidationReport>,
    /// Summary statistics
    pub summary: HighMassSummary,
}

impl CombinedHighMassReport {
    /// Create a new combined report from individual case reports.
    ///
    /// # Arguments
    /// * `case_reports` - Vector of individual case reports
    ///
    /// # Returns
    /// A new CombinedHighMassReport instance
    pub fn new(case_reports: Vec<HighMassValidationReport>) -> Self {
        let summary = HighMassSummary::from_reports(&case_reports);

        Self {
            case_reports,
            summary,
        }
    }

    /// Generate a comprehensive markdown report for all cases.
    ///
    /// # Returns
    /// Markdown-formatted string containing the combined report
    pub fn generate_markdown(&self) -> String {
        let mut output = String::new();

        // Title
        output.push_str("# High-Mass Validation Suite Report\n\n");

        // Summary
        output.push_str("## Summary\n\n");
        output.push_str(&format!(
            "- **Total Cases:** {}\n",
            self.summary.total_cases
        ));
        output.push_str(&format!("- **Passed:** {}\n", self.summary.passed_cases));
        output.push_str(&format!("- **Failed:** {}\n", self.summary.failed_cases));
        output.push_str(&format!(
            "- **Pass Rate:** {:.1}%\n",
            self.summary.pass_rate()
        ));
        output.push_str(&format!(
            "- **Mean NMBE:** {:.2}%\n",
            self.summary.mean_nmbe
        ));
        output.push_str(&format!(
            "- **Mean CV(RMSE):** {:.2}%\n",
            self.summary.mean_cv_rmse
        ));
        output.push_str("\n");

        // Individual case summaries
        output.push_str("## Case Results\n\n");
        output.push_str("| Case | Building Type | Construction | Status | NMBE | CV(RMSE) |\n");
        output.push_str("|------|---------------|--------------|--------|------|---------|\n");

        for report in &self.case_reports {
            let avg_nmbe =
                (report.metrics.nmbe_heating.abs() + report.metrics.nmbe_cooling.abs()) / 2.0;
            let avg_cv_rmse =
                (report.metrics.cv_rmse_heating + report.metrics.cv_rmse_cooling) / 2.0;

            output.push_str(&format!(
                "| {} | {} | {:?} | {} | {:.1}% | {:.1}% |\n",
                report.case_id,
                report.building_description,
                report.construction_type,
                if report.passed {
                    "PASS ✓"
                } else {
                    "FAIL ✗"
                },
                avg_nmbe,
                avg_cv_rmse
            ));
        }

        output.push_str("\n");

        // Detailed reports
        for report in &self.case_reports {
            output.push_str(&format!("## Case {} Detailed Report\n\n", report.case_id));
            output.push_str(&report.generate_markdown());
            output.push_str("\n---\n\n");
        }

        // Overall assessment
        output.push_str("## Overall Assessment\n\n");
        if self.summary.pass_rate() >= 80.0 {
            output.push_str("✅ **VALIDATION SUITE PASSED**\n\n");
            output.push_str(&format!(
                "{} out of {} cases passed validation ({:.1}% pass rate).\n",
                self.summary.passed_cases,
                self.summary.total_cases,
                self.summary.pass_rate()
            ));
            output.push_str(
                "The high-mass validation suite demonstrates good overall performance.\n",
            );
        } else {
            output.push_str("❌ **VALIDATION SUITE FAILED**\n\n");
            output.push_str(&format!(
                "Only {} out of {} cases passed validation ({:.1}% pass rate).\n",
                self.summary.passed_cases,
                self.summary.total_cases,
                self.summary.pass_rate()
            ));
            output.push_str(
                "Review failed cases and consider improvements to high-mass simulation accuracy.\n",
            );
        }

        output
    }
}

/// Summary statistics for high-mass validation suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighMassSummary {
    /// Total number of validation cases
    pub total_cases: usize,
    /// Number of passed cases
    pub passed_cases: usize,
    /// Number of failed cases
    pub failed_cases: usize,
    /// Mean NMBE across all cases
    pub mean_nmbe: f64,
    /// Mean CV(RMSE) across all cases
    pub mean_cv_rmse: f64,
}

impl HighMassSummary {
    /// Create summary from vector of reports.
    ///
    /// # Arguments
    /// * `reports` - Vector of high-mass validation reports
    ///
    /// # Returns
    /// A new HighMassSummary instance
    pub fn from_reports(reports: &[HighMassValidationReport]) -> Self {
        let total_cases = reports.len();
        let passed_cases = reports.iter().filter(|r| r.passed).count();
        let failed_cases = total_cases - passed_cases;

        let total_nmbe: f64 = reports
            .iter()
            .map(|r| (r.metrics.nmbe_heating.abs() + r.metrics.nmbe_cooling.abs()) / 2.0)
            .sum();
        let mean_nmbe = if total_cases > 0 {
            total_nmbe / total_cases as f64
        } else {
            0.0
        };

        let total_cv_rmse: f64 = reports
            .iter()
            .map(|r| (r.metrics.cv_rmse_heating + r.metrics.cv_rmse_cooling) / 2.0)
            .sum();
        let mean_cv_rmse = if total_cases > 0 {
            total_cv_rmse / total_cases as f64
        } else {
            0.0
        };

        Self {
            total_cases,
            passed_cases,
            failed_cases,
            mean_nmbe,
            mean_cv_rmse,
        }
    }

    /// Calculate pass rate as percentage.
    ///
    /// # Returns
    /// Pass rate in percent
    pub fn pass_rate(&self) -> f64 {
        if self.total_cases == 0 {
            0.0
        } else {
            (self.passed_cases as f64 / self.total_cases as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::tolerance::ValidationTolerance;

    #[test]
    fn test_high_mass_validation_report_generation() {
        let weather = WeatherSummary::default();
        let metrics = HighMassMetrics::default();
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let construction = ConstructionType::HeavyWeight;
        let tolerance = ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        };

        let report = HighMassValidationReport::generate_report(
            "600",
            "Heavyweight residential",
            weather,
            metrics,
            diagnostics,
            construction,
            tolerance,
        );

        assert_eq!(report.case_id, "600");
        assert_eq!(report.building_description, "Heavyweight residential");
        assert!(report.passed); // Default metrics should pass default tolerance
        assert!(report.generate_markdown().contains("Case 600"));
    }

    #[test]
    fn test_markdown_generation() {
        let weather = WeatherSummary::default();
        let metrics = HighMassMetrics::default();
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let construction = ConstructionType::HeavyWeight;
        let tolerance = ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        };

        let report = HighMassValidationReport::generate_report(
            "900",
            "High-mass institutional",
            weather,
            metrics,
            diagnostics,
            construction,
            tolerance,
        );

        let markdown = report.generate_markdown();
        assert!(markdown.contains("# High-Mass Validation Report: Case 900"));
        assert!(markdown.contains("## Case Information"));
        assert!(markdown.contains("## Weather Summary"));
        assert!(markdown.contains("## Validation Metrics"));
        assert!(markdown.contains("## Thermal Mass Diagnostics"));
        assert!(markdown.contains("## Overall Assessment"));
    }

    #[test]
    fn test_json_generation() {
        let weather = WeatherSummary::default();
        let metrics = HighMassMetrics::default();
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let construction = ConstructionType::HeavyWeight;
        let tolerance = ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        };

        let report = HighMassValidationReport::generate_report(
            "650",
            "Medium-weight commercial",
            weather,
            metrics,
            diagnostics,
            construction,
            tolerance,
        );

        let json = report.generate_json();
        assert_eq!(json["case_id"], "650");
        assert_eq!(json["building_description"], "Medium-weight commercial");
        assert!(json["passed"].as_bool().unwrap());
    }

    #[test]
    fn test_combined_report() {
        let weather = WeatherSummary::default();
        let metrics = HighMassMetrics::default();
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let construction = ConstructionType::HeavyWeight;
        let tolerance = ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        };

        let report1 = HighMassValidationReport::generate_report(
            "600",
            "Case 600",
            weather,
            metrics.clone(),
            diagnostics.clone(),
            construction.clone(),
            tolerance.clone(),
        );

        let report2 = HighMassValidationReport::generate_report(
            "900",
            "Case 900",
            weather,
            metrics,
            diagnostics,
            construction,
            tolerance,
        );

        let combined = CombinedHighMassReport::new(vec![report1, report2]);
        assert_eq!(combined.case_reports.len(), 2);
        assert_eq!(combined.summary.total_cases, 2);
        assert_eq!(combined.summary.passed_cases, 2);

        let markdown = combined.generate_markdown();
        assert!(markdown.contains("# High-Mass Validation Suite Report"));
        assert!(markdown.contains("Case 600"));
        assert!(markdown.contains("Case 900"));
    }

    #[test]
    fn test_summary_calculations() {
        let weather = WeatherSummary::default();
        let diagnostics = ThermalMassDiagnostics::new(3600, 10.0);
        let construction = ConstructionType::HeavyWeight;
        let tolerance = ValidationTolerance {
            nmbe_limit: 5.0,
            cv_rmse_limit: 10.0,
            mae_limit: 0.1,
        };

        // Create metrics that will pass
        let passing_metrics = HighMassMetrics::default();
        let report1 = HighMassValidationReport::generate_report(
            "600",
            "Case 600",
            weather.clone(),
            passing_metrics.clone(),
            diagnostics.clone(),
            construction,
            tolerance.clone(),
        );

        let report2 = HighMassValidationReport::generate_report(
            "900",
            "Case 900",
            weather,
            passing_metrics,
            diagnostics,
            construction.clone(),
            tolerance,
        );

        let summary = HighMassSummary::from_reports(&[report1, report2]);
        assert_eq!(summary.total_cases, 2);
        assert_eq!(summary.passed_cases, 2);
        assert_eq!(summary.failed_cases, 0);
        assert_eq!(summary.pass_rate(), 100.0);
    }
}
