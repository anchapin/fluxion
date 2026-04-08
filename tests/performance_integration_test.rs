#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::performance::PerformanceReport;
    use crate::validation::report::ValidationResult;
    use crate::validation::ValidationConfig;
    use chrono::Utc;
    use serde_json;

    #[test]
    fn test_integrated_validation_workflow() {
        // Setup validation suite with performance validation
        let config = ValidationConfig::standard();
        let validation_suite = ValidationSuite::new(config);
        let integrator = IntegratedPerformanceValidator::new(validation_suite);

        // Run full integrated validation
        let result = integrator.run_full_validation();

        // Verify standard validation passed
        assert!(result.standard.passed, "Standard validation should pass");

        // Verify performance validation succeeded
        assert!(
            result.performance.is_ok(),
            "Performance validation should succeed"
        );

        // Verify integration status
        assert!(result.integrated, "Integrated validation should pass");

        // Verify performance meets thresholds
        let performance_report = result.performance.unwrap();
        assert!(
            performance_report.metrics.timestep_duration_ms < 50.0,
            "Timestep duration should be under 50ms"
        );
        assert!(
            performance_report.metrics.memory_usage_bytes < 10_000_000,
            "Memory usage should be under 10MB"
        );
    }

    #[test]
    fn test_integrated_report_generation() {
        let config = ValidationConfig::standard();
        let validation_suite = ValidationSuite::new(config);
        let integrator = IntegratedPerformanceValidator::new(validation_suite);

        let result = integrator.run_full_validation();
        let report = integrator.generate_integrated_report(&result);

        // Verify report contains all required fields
        assert!(report.overall_status == "PASS" || report.overall_status == "FAIL");
        assert!(report.performance_validation.is_ok());

        // Verify JSON serialization works
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("timestamp"));
        assert!(json.contains("overall_status"));
    }

    #[test]
    fn test_performance_threshold_validation() {
        let config = ValidationConfig::standard();
        let validation_suite = ValidationSuite::new(config);
        let integrator = IntegratedPerformanceValidator::new(validation_suite);

        // Create a performance report that exceeds thresholds
        let bad_report = PerformanceReport {
            timestamp: Utc::now(),
            metrics: crate::validation::performance::PerformanceMetrics {
                timestep_duration_ms: 100.0,    // Exceeds 50ms threshold
                memory_usage_bytes: 20_000_000, // Exceeds 10MB threshold
                iterations_per_timestep: 100,
            },
            baseline_comparison: None,
        };

        // Verify threshold check fails
        let threshold_result = integrator.check_performance_thresholds(&bad_report);
        assert!(
            !threshold_result,
            "Should fail threshold check for poor performance"
        );
    }

    #[test]
    fn test_cli_integration_with_performance() {
        // Test CLI commands work with integrated validation
        let args = vec!["fluxion", "performance", "validate"];
        let matches = clap::Command::new("fluxion")
            .subcommand(crate::cli::performance::PerformanceCommand::augment())
            .try_get_matches_from(args)
            .unwrap();

        match matches.subcommand() {
            Some(("performance", sub_matches)) => {
                let command =
                    crate::cli::performance::PerformanceCommand::from_arg_matches(sub_matches)
                        .unwrap();
                match command {
                    crate::cli::performance::PerformanceCommand::Validate {
                        baseline,
                        threshold,
                    } => {
                        assert!(threshold > 0.0);
                        // Command parsed correctly
                    }
                    _ => panic!("Expected Validate command"),
                }
            }
            _ => panic!("Expected performance subcommand"),
        }
    }
}
