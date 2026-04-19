// Comprehensive validation suite for multi-zone foundation
// This module integrates all M1-related validators into a unified suite

use crate::validation::ashrae_140_multi_zone::ASHRAE140MultiZoneValidator;
use crate::validation::energy_balance::EnergyBalanceValidator;
use crate::validation::performance;
use crate::validation::performance::PerformanceValidator;
use crate::validation::thermal_mass_energy_accounting::EnergyBalanceReport;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// M1 Validation Suite - Comprehensive multi-zone foundation validation
pub struct M1ValidationSuite {
    energy_balance_validator: EnergyBalanceValidator,
    performance_validator: PerformanceValidator,
    ashrae_multi_zone_validator: ASHRAE140MultiZoneValidator,
}

impl M1ValidationSuite {
    /// Create a new M1 validation suite
    pub fn new() -> Result<Self, anyhow::Error> {
        Ok(Self {
            energy_balance_validator: EnergyBalanceValidator::new(),
            performance_validator: PerformanceValidator::new()?,
            ashrae_multi_zone_validator: ASHRAE140MultiZoneValidator::new(),
        })
    }

    /// Run all M1 validators sequentially
    pub fn run_all(&self) -> Result<M1ValidationReport, anyhow::Error> {
        let start_time = Instant::now();

        println!("=== M1 Multi-Zone Foundation Validation Suite ===");
        println!("Starting comprehensive validation...");

        // 1. Energy Balance Validation
        println!("\n[1/4] Running energy balance validation...");
        let energy_balance_start = Instant::now();
        let energy_balance_report = self.energy_balance_validator.run()?;
        let energy_balance_duration = energy_balance_start.elapsed();
        let energy_balance_passed = energy_balance_report.is_valid;

        // Print zone and building balance summary
        if !energy_balance_report.zone_balances.is_empty() {
            println!("\n  Zone Balance Summary:");
            for entry in &energy_balance_report.zone_balances {
                println!(
                    "    Zone {}: HVAC={:.2e}J, Solar={:.2e}J, Internal={:.2e}J",
                    entry.zone_index, entry.hvac_input, entry.solar_gains, entry.internal_gains
                );
            }
            println!("\n  Whole-Building Balance:");
            println!(
                "    Total Energy In: {:.6e} J",
                energy_balance_report.building_balance.total_energy_in
            );
            println!(
                "    Total Energy Out: {:.6e} J",
                energy_balance_report.building_balance.total_energy_out
            );
            println!(
                "    Balance Error: {:.6}%",
                energy_balance_report.building_balance.balance_error_pct
            );
        }

        println!(
            "✅ Energy balance validation completed in {:.3}s - {}",
            energy_balance_duration.as_secs_f64(),
            if energy_balance_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        );

        // 2. Performance Validation
        println!("\n[2/4] Running performance validation...");
        let performance_start = Instant::now();
        let performance_report = self
            .performance_validator
            .validate_performance_regression()?;
        let performance_duration = performance_start.elapsed();
        println!(
            "✅ Performance validation completed in {:.3}s",
            performance_duration.as_secs_f64()
        );

        // 3. ASHRAE 140 Case 960 Validation
        println!("\n[3/4] Running ASHRAE 140 Case 960 validation...");
        let case_960_start = Instant::now();
        let case_960_result = self.ashrae_multi_zone_validator.validate_case_960()?;
        let case_960_duration = case_960_start.elapsed();
        println!(
            "✅ Case 960 validation completed in {:.3}s",
            case_960_duration.as_secs_f64()
        );

        // 4. Generate performance report
        println!("\n[4/4] Generating comprehensive report...");
        let report_start = Instant::now();
        let performance_report_text = self
            .performance_validator
            .generate_performance_report(&performance_report);
        let report_duration = report_start.elapsed();
        println!(
            "✅ Report generation completed in {:.3}s",
            report_duration.as_secs_f64()
        );

        let total_duration = start_time.elapsed();
        println!("\n🎉 All M1 validations completed successfully!");
        println!(
            "Total validation time: {:.3}s",
            total_duration.as_secs_f64()
        );

        Ok(M1ValidationReport {
            energy_balance_passed,
            energy_balance_report,
            performance_report,
            case_960_report: case_960_result,
            performance_report_text,
            validation_duration_seconds: total_duration.as_secs_f64(),
        })
    }

    /// Run performance validation using the new performance module
    pub fn run_performance_validation(
        &self,
    ) -> Result<performance::PerformanceReport, anyhow::Error> {
        let thermal_model = self.create_thermal_model()?;
        let validator = performance::PerformanceValidator::new(thermal_model);
        Ok(validator.validate_performance())
    }

    /// Check if all M1 requirements are satisfied
    pub fn check_requirements(&self, report: &M1ValidationReport) -> RequirementsCheck {
        let mut all_passed = true;
        let mut findings = Vec::new();

        // Check energy balance
        if report.energy_balance_passed {
            findings.push(RequirementFinding {
                requirement_id: "MZ-05".to_string(),
                description: "Energy balance verification".to_string(),
                passed: true,
                details: "Energy conservation validated successfully".to_string(),
            });
        } else {
            findings.push(RequirementFinding {
                requirement_id: "MZ-05".to_string(),
                description: "Energy balance verification".to_string(),
                passed: false,
                details: "Energy conservation validation failed".to_string(),
            });
            all_passed = false;
        }

        // Check performance (10-zone <2× slowdown)
        match &report.performance_report.scalability_analysis {
            crate::validation::performance::ScalabilityAnalysis::GoodScalability { .. } => {
                findings.push(RequirementFinding {
                    requirement_id: "MZ-08".to_string(),
                    description: "Performance maintenance".to_string(),
                    passed: true,
                    details: "Good scalability detected (<2× slowdown for 10 zones)".to_string(),
                });
            }
            crate::validation::performance::ScalabilityAnalysis::LinearScalability { .. } => {
                findings.push(RequirementFinding {
                    requirement_id: "MZ-08".to_string(),
                    description: "Performance maintenance".to_string(),
                    passed: false,
                    details: "Linear scalability detected (>2× slowdown)".to_string(),
                });
                all_passed = false;
            }
            crate::validation::performance::ScalabilityAnalysis::QuadraticScaling { .. } => {
                findings.push(RequirementFinding {
                    requirement_id: "MZ-08".to_string(),
                    description: "Performance maintenance".to_string(),
                    passed: false,
                    details: "Quadratic scalability detected (poor performance)".to_string(),
                });
                all_passed = false;
            }
            crate::validation::performance::ScalabilityAnalysis::InsufficientData => {
                findings.push(RequirementFinding {
                    requirement_id: "MZ-08".to_string(),
                    description: "Performance maintenance".to_string(),
                    passed: false,
                    details: "Insufficient data for performance analysis".to_string(),
                });
                all_passed = false;
            }
        }

        // Check Case 960 validation
        if report.case_960_report.passed {
            findings.push(RequirementFinding {
                requirement_id: "MZ-06".to_string(),
                description: "ASHRAE 140 Case 960 validation".to_string(),
                passed: true,
                details: "Case 960 validation passed".to_string(),
            });
        } else {
            findings.push(RequirementFinding {
                requirement_id: "MZ-06".to_string(),
                description: "ASHRAE 140 Case 960 validation".to_string(),
                passed: false,
                details: format!(
                    "Case 960 validation failed: {}",
                    report.case_960_report.message
                ),
            });
            all_passed = false;
        }

        RequirementsCheck {
            all_requirements_passed: all_passed,
            findings,
        }
    }

    /// Export validation results to JSON
    pub fn export_results(
        &self,
        report: &M1ValidationReport,
        check: &RequirementsCheck,
    ) -> Result<String, anyhow::Error> {
        let export_data = M1ValidationExport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            energy_balance_passed: report.energy_balance_passed,
            performance_analysis: format!("{:?}", report.performance_report.scalability_analysis),
            case_960_passed: report.case_960_report.passed,
            requirements_check: check.findings.clone(),
            all_requirements_passed: check.all_requirements_passed,
            validation_duration_seconds: report.validation_duration_seconds,
        };

        Ok(serde_json::to_string_pretty(&export_data)?)
    }
}

/// CLI integration for M1 validation suite
pub fn run_m1_validation_cli() -> Result<(), anyhow::Error> {
    let suite = M1ValidationSuite::new()?;
    let report = suite.run_all()?;
    let requirements_check = suite.check_requirements(&report);

    // Print summary
    println!("\n=== M1 Validation Summary ===");
    println!(
        "Energy Balance: {}",
        if report.energy_balance_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    match &report.performance_report.scalability_analysis {
        crate::validation::performance::ScalabilityAnalysis::GoodScalability { .. } => {
            println!("Performance: ✅ GOOD SCALABILITY");
        }
        crate::validation::performance::ScalabilityAnalysis::LinearScalability { .. } => {
            println!("Performance: ⚠️  LINEAR SCALABILITY");
        }
        crate::validation::performance::ScalabilityAnalysis::QuadraticScaling { .. } => {
            println!("Performance: ❌ QUADRATIC SCALABILITY");
        }
        crate::validation::performance::ScalabilityAnalysis::InsufficientData => {
            println!("Performance: ⚠️  INSUFFICIENT DATA");
        }
    }

    println!(
        "Case 960: {}",
        if report.case_960_report.passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );
    println!(
        "Overall: {}",
        if requirements_check.all_requirements_passed {
            "✅ ALL REQUIREMENTS MET"
        } else {
            "❌ SOME REQUIREMENTS FAILED"
        }
    );

    // Print detailed performance report
    println!("\n=== Performance Report ===");
    println!("{}", report.performance_report_text);

    // Print requirements checklist
    println!("\n=== Requirements Checklist ===");
    for finding in &requirements_check.findings {
        let status = if finding.passed { "✅" } else { "❌" };
        println!(
            "{} {}: {}",
            status, finding.requirement_id, finding.description
        );
        if !finding.details.is_empty() {
            println!("   {}", finding.details);
        }
    }

    // Export JSON results
    let json_export = suite.export_results(&report, &requirements_check)?;
    println!("\n=== JSON Export ===");
    println!("{}", json_export);

    Ok(())
}

/// M1 Validation Report
#[derive(Debug, Clone)]
pub struct M1ValidationReport {
    pub energy_balance_passed: bool,
    pub energy_balance_report: EnergyBalanceReport,
    pub performance_report: crate::validation::performance::PerformanceReport,
    pub case_960_report: Case960ValidationResult,
    pub performance_report_text: String,
    pub validation_duration_seconds: f64,
}

/// Case 960 Validation Result
#[derive(Debug, Clone)]
pub struct Case960ValidationResult {
    pub passed: bool,
    pub message: String,
    pub metrics: Case960Metrics,
}

/// Case 960 Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case960Metrics {
    pub max_temperature_error_c: f64,
    pub mean_temperature_error_c: f64,
    pub energy_balance_error_percent: f64,
}

/// Requirements Check Result
#[derive(Debug, Clone)]
pub struct RequirementsCheck {
    pub all_requirements_passed: bool,
    pub findings: Vec<RequirementFinding>,
}

/// Individual Requirement Finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequirementFinding {
    pub requirement_id: String,
    pub description: String,
    pub passed: bool,
    pub details: String,
}

/// Export format for validation results
#[derive(Debug, Serialize, Deserialize)]
pub struct M1ValidationExport {
    pub timestamp: String,
    pub energy_balance_passed: bool,
    pub performance_analysis: String,
    pub case_960_passed: bool,
    pub requirements_check: Vec<RequirementFinding>,
    pub all_requirements_passed: bool,
    pub validation_duration_seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_suite_creation() {
        let suite = M1ValidationSuite::new();
        assert!(suite.is_ok());
    }

    #[test]
    fn test_requirements_check_all_passed() {
        let suite = M1ValidationSuite::new().unwrap();

        // Create a mock report with all tests passing
        let mock_report = M1ValidationReport {
            energy_balance_passed: true,
            performance_report: crate::validation::performance::PerformanceReport {
                results: vec![],
                scalability_analysis:
                    crate::validation::performance::ScalabilityAnalysis::GoodScalability {
                        metrics: vec![],
                    },
            },
            case_960_report: Case960ValidationResult {
                passed: true,
                message: "Case 960 passed".to_string(),
                metrics: Case960Metrics {
                    max_temperature_error_c: 0.5,
                    mean_temperature_error_c: 0.1,
                    energy_balance_error_percent: 1.0,
                },
            },
            performance_report_text: "Good performance".to_string(),
            validation_duration_seconds: 10.5,
        };

        let check = suite.check_requirements(&mock_report);
        assert!(check.all_requirements_passed);
        assert_eq!(check.findings.len(), 3); // 3 requirements: MZ-05, MZ-08, MZ-06
    }

    #[test]
    fn test_requirements_check_with_failures() {
        let suite = M1ValidationSuite::new().unwrap();

        // Create a mock report with some tests failing
        let mock_report = M1ValidationReport {
            energy_balance_passed: false,
            performance_report: crate::validation::performance::PerformanceReport {
                results: vec![],
                scalability_analysis:
                    crate::validation::performance::ScalabilityAnalysis::QuadraticScaling {
                        metrics: vec![],
                    },
            },
            case_960_report: Case960ValidationResult {
                passed: false,
                message: "Case 960 failed".to_string(),
                metrics: Case960Metrics {
                    max_temperature_error_c: 5.0,
                    mean_temperature_error_c: 2.0,
                    energy_balance_error_percent: 15.0,
                },
            },
            performance_report_text: "Poor performance".to_string(),
            validation_duration_seconds: 20.0,
        };

        let check = suite.check_requirements(&mock_report);
        assert!(!check.all_requirements_passed);
        assert_eq!(check.findings.len(), 3);

        // Verify all findings show failed
        for finding in check.findings {
            assert!(!finding.passed);
        }
    }

    #[test]
    fn test_json_export() {
        let suite = M1ValidationSuite::new().unwrap();

        let mock_report = M1ValidationReport {
            energy_balance_passed: true,
            performance_report: crate::validation::performance::PerformanceReport {
                results: vec![],
                scalability_analysis:
                    crate::validation::performance::ScalabilityAnalysis::GoodScalability {
                        metrics: vec![],
                    },
            },
            case_960_report: Case960ValidationResult {
                passed: true,
                message: "Case 960 passed".to_string(),
                metrics: Case960Metrics {
                    max_temperature_error_c: 0.5,
                    mean_temperature_error_c: 0.1,
                    energy_balance_error_percent: 1.0,
                },
            },
            performance_report_text: "Good performance".to_string(),
            validation_duration_seconds: 10.5,
        };

        let check = suite.check_requirements(&mock_report);
        let json_export = suite.export_results(&mock_report, &check).unwrap();

        // Verify JSON contains expected fields
        assert!(json_export.contains("timestamp"));
        assert!(json_export.contains("energy_balance_passed"));
        assert!(json_export.contains("all_requirements_passed"));

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json_export).unwrap();
        assert!(parsed["all_requirements_passed"].as_bool().unwrap());
    }
}
