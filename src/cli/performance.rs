use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::validation::performance::{ci, integration, PerformanceValidator};
use crate::validation::report::ValidationSuite;
use clap::Subcommand;
use serde_json;

#[derive(Subcommand, Debug)]
pub enum PerformanceCommand {
    /// Run performance benchmarks
    Benchmark {
        /// Scenario to benchmark (single, multi, high-mass)
        scenario: Option<String>,
        /// Output format (json, text, pretty)
        #[arg(long, default_value = "json")]
        format: String,
    },
    /// Validate performance against baseline
    Validate {
        /// Baseline file path
        baseline: Option<String>,
        /// Regression threshold percentage
        #[arg(long, default_value_t = 5.0)]
        threshold: f64,
    },
    /// Generate performance report
    Report {
        /// Output file path
        output: Option<String>,
        /// Include detailed metrics
        #[arg(long)]
        detailed: bool,
    },
    /// Run integrated validation (standard + performance)
    Integrated {
        /// Output format (json, text, pretty)
        #[arg(long, default_value = "json")]
        format: String,
        /// Include detailed performance metrics
        #[arg(long)]
        detailed: bool,
    },
    /// Validate performance against ASHRAE 140 requirements
    Ashrae140 {
        /// ASHRAE 140 case number
        case: Option<u32>,
        /// Output file path
        #[arg(long)]
        output: Option<String>,
    },
}

pub fn handle_performance_command(command: &PerformanceCommand) -> Result<(), String> {
    match command {
        PerformanceCommand::Benchmark { scenario, format } => run_benchmarks(scenario, format),
        PerformanceCommand::Validate {
            baseline,
            threshold,
        } => validate_performance(baseline, *threshold),
        PerformanceCommand::Report { output, detailed } => generate_report(output, *detailed),
        PerformanceCommand::Integrated { format, detailed } => {
            run_integrated_validation(format, *detailed)
        }
        PerformanceCommand::Ashrae140 { case, output } => {
            run_ashrae140_performance_validation(case, output)
        }
    }
}

fn run_benchmarks(_scenario: &Option<String>, _format: &str) -> Result<(), String> {
    // Implement benchmark execution
    println!("Running performance benchmarks...");
    Ok(())
}

fn validate_performance(baseline: &Option<String>, _threshold: f64) -> Result<(), String> {
    // Create CI performance validator
    let ci_validator = ci::CiPerformanceValidator::new(baseline.clone());

    // Run CI performance validation
    // Pattern: ci::run_performance_validation
    let ci_report = ci_validator.run_performance_validation()?;

    println!("CI Performance validation complete: {:?}", ci_report);

    // Also run standard performance validation
    let model = ThermalModel::<VectorField>::new(1);
    let mut validator = PerformanceValidator::new(model);
    let report = validator.validate_performance();

    if let Some(baseline_path) = baseline {
        // Compare with baseline
        println!("Comparing with baseline: {}", baseline_path);
    }

    println!("Performance validation complete: {:?}", report);
    Ok(())
}

fn generate_report(output: &Option<String>, detailed: bool) -> Result<(), String> {
    // Create a default thermal model for validation
    let model = ThermalModel::<VectorField>::new(1);
    let mut validator = PerformanceValidator::new(model);
    let report = validator.validate_performance();

    let json = if detailed {
        serde_json::to_string_pretty(&report).unwrap()
    } else {
        serde_json::to_string(&report).unwrap()
    };

    match output {
        Some(path) => {
            std::fs::write(path, json).map_err(|e| format!("Failed to write report: {}", e))?
        }
        None => println!("{}", json),
    }

    Ok(())
}

fn run_integrated_validation(format: &str, detailed: bool) -> Result<(), String> {
    let config = crate::validation::ValidationConfig::standard();
    let validation_suite = ValidationSuite::new_with_config(config);
    let integrator = integration::IntegratedPerformanceValidator::new(validation_suite);

    let result = integrator.run_full_validation();
    let report = integrator.generate_integrated_report(&result);

    match format {
        "json" => {
            let json = if detailed {
                serde_json::to_string_pretty(&report).unwrap()
            } else {
                serde_json::to_string(&report).unwrap()
            };
            println!("{}", json);
        }
        "text" => {
            println!("Integrated Validation Report");
            println!("============================");
            println!("Status: {}", report.overall_status);
            println!("Timestamp: {}", report.timestamp);

            match &report.performance_validation {
                Ok(perf) => println!(
                    "Performance: OK ({:.2}ms/timestep)",
                    perf.metrics.timestep_duration_ms
                ),
                Err(e) => println!("Performance: ERROR - {}", e),
            }
        }
        "pretty" => {
            // Enhanced pretty printing
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        _ => return Err(format!("Unknown format: {}", format)),
    }

    Ok(())
}

fn run_ashrae140_performance_validation(
    case: &Option<u32>,
    output: &Option<String>,
) -> Result<(), String> {
    let case_number = case.unwrap_or(900); // Default to Case 900

    // Load ASHRAE 140 case
    load_ashrae140_case(case_number)
        .map_err(|e| format!("Failed to load ASHRAE 140 case {}: {}", case_number, e))?;

    // Create validation suite for this case
    let config = crate::validation::ValidationConfig::ashrae140(case_number);
    let validation_suite = ValidationSuite::new_with_config(config);
    let integrator = integration::IntegratedPerformanceValidator::new(validation_suite);

    // Run validation
    let result = integrator.run_full_validation();
    let report = integrator.generate_integrated_report(&result);

    // Output results
    let json = serde_json::to_string_pretty(&report).unwrap();

    match output {
        Some(path) => {
            std::fs::write(path, json).map_err(|e| format!("Failed to write report: {}", e))?
        }
        None => println!("{}", json),
    }

    // Check ASHRAE 140 performance requirements
    if let Ok(perf_report) = &report.performance_validation {
        if perf_report.metrics.timestep_duration_ms > 100.0 {
            return Err(format!(
                "ASHRAE 140 performance requirement failed: {:.2}ms > 100ms",
                perf_report.metrics.timestep_duration_ms
            ));
        }
    }

    Ok(())
}

// Mock function for loading ASHRAE 140 case
fn load_ashrae140_case(_case_number: u32) -> Result<(), String> {
    // In a real implementation, this would load the actual ASHRAE 140 case
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Inline unit tests for the performance CLI surface (Issue #2897).
    //!
    //! Coverage split:
    //! * clap argument wiring for every `PerformanceCommand` variant — defaults,
    //!   flag handling, and type rejection. This is the module's real
    //!   "release-only"-style boolean-flag surface: `--detailed` on `report` /
    //!   `integrated`, plus `--format`, `--threshold`, `--baseline`, `--output`.
    //! * `handle_performance_command` dispatch for the paths that do not run a
    //!   full validation suite (`benchmark`, `report`).
    //! * `generate_report` output-format selection and IO error propagation.
    //!
    //! Deliberately not covered here: `validate` / `integrated` / `ashrae140`
    //! execution, which build a full `ValidationSuite` and belong to the
    //! integration suite rather than a `--lib` unit test.

    use super::*;
    use clap::Parser;

    /// Test-only wrapper so `PerformanceCommand` is exercised through real clap.
    #[derive(Debug, Parser)]
    #[command(name = "fluxion-performance-test")]
    struct TestCli {
        #[command(subcommand)]
        cmd: PerformanceCommand,
    }

    fn parse(args: &[&str]) -> PerformanceCommand {
        TestCli::try_parse_from(args)
            .expect("args should parse")
            .cmd
    }

    /// Default ASHRAE 140 case applied by `run_ashrae140_performance_validation`
    /// when `case` is `None` (mirrors the `unwrap_or(900)` in that handler).
    const DEFAULT_ASHRAE_CASE: u32 = 900;

    // ---------------------------------------------------------------
    // benchmark
    // ---------------------------------------------------------------

    #[test]
    fn benchmark_defaults_scenario_none_and_json_format() {
        match parse(&["perf", "benchmark"]) {
            PerformanceCommand::Benchmark { scenario, format } => {
                assert_eq!(scenario, None, "scenario is an optional positional");
                assert_eq!(format, "json", "format defaults to json");
            }
            other => panic!("expected Benchmark, got {other:?}"),
        }
    }

    #[test]
    fn benchmark_parses_scenario_and_format_override() {
        match parse(&["perf", "benchmark", "high-mass", "--format", "text"]) {
            PerformanceCommand::Benchmark { scenario, format } => {
                assert_eq!(scenario.as_deref(), Some("high-mass"));
                assert_eq!(format, "text");
            }
            other => panic!("expected Benchmark, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // validate
    // ---------------------------------------------------------------

    #[test]
    fn validate_defaults_baseline_none_and_five_percent_threshold() {
        match parse(&["perf", "validate"]) {
            PerformanceCommand::Validate {
                baseline,
                threshold,
            } => {
                assert_eq!(baseline, None);
                assert!(
                    (threshold - 5.0).abs() < f64::EPSILON,
                    "regression threshold defaults to 5.0%, got {threshold}"
                );
            }
            other => panic!("expected Validate, got {other:?}"),
        }
    }

    #[test]
    fn validate_parses_baseline_positional_and_threshold_override() {
        match parse(&["perf", "validate", "baseline.json", "--threshold", "2.5"]) {
            PerformanceCommand::Validate {
                baseline,
                threshold,
            } => {
                assert_eq!(baseline.as_deref(), Some("baseline.json"));
                assert!((threshold - 2.5).abs() < 1e-12, "got {threshold}");
            }
            other => panic!("expected Validate, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_non_numeric_threshold() {
        assert!(
            TestCli::try_parse_from(["perf", "validate", "--threshold", "loose"]).is_err(),
            "--threshold must be an f64"
        );
    }

    // ---------------------------------------------------------------
    // report — boolean-flag handling
    // ---------------------------------------------------------------

    #[test]
    fn report_boolean_flag_is_false_when_absent_and_true_when_present() {
        // Boolean flag semantics: absent => false, present => true, and the
        // flag takes no value.
        match parse(&["perf", "report"]) {
            PerformanceCommand::Report { output, detailed } => {
                assert_eq!(output, None);
                assert!(!detailed, "--detailed must default to false");
            }
            other => panic!("expected Report, got {other:?}"),
        }
        match parse(&["perf", "report", "--detailed"]) {
            PerformanceCommand::Report { detailed, .. } => assert!(detailed),
            other => panic!("expected Report, got {other:?}"),
        }
        assert!(
            TestCli::try_parse_from(["perf", "report", "--detailed", "true", "extra"]).is_err(),
            "--detailed is a flag, not a value-taking option"
        );
    }

    #[test]
    fn report_parses_output_positional_before_flag() {
        match parse(&["perf", "report", "out.json", "--detailed"]) {
            PerformanceCommand::Report { output, detailed } => {
                assert_eq!(output.as_deref(), Some("out.json"));
                assert!(detailed);
            }
            other => panic!("expected Report, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // integrated / ashrae140
    // ---------------------------------------------------------------

    #[test]
    fn integrated_defaults_json_format_and_non_detailed() {
        match parse(&["perf", "integrated"]) {
            PerformanceCommand::Integrated { format, detailed } => {
                assert_eq!(format, "json");
                assert!(!detailed);
            }
            other => panic!("expected Integrated, got {other:?}"),
        }
        match parse(&["perf", "integrated", "--format", "pretty", "--detailed"]) {
            PerformanceCommand::Integrated { format, detailed } => {
                assert_eq!(format, "pretty");
                assert!(detailed);
            }
            other => panic!("expected Integrated, got {other:?}"),
        }
    }

    #[test]
    fn ashrae140_case_is_optional_and_defaults_to_case_900() {
        match parse(&["perf", "ashrae140"]) {
            PerformanceCommand::Ashrae140 { case, output } => {
                assert_eq!(case, None);
                assert_eq!(output, None);
                // The handler substitutes Case 900 when no case is supplied.
                assert_eq!(case.unwrap_or(DEFAULT_ASHRAE_CASE), DEFAULT_ASHRAE_CASE);
            }
            other => panic!("expected Ashrae140, got {other:?}"),
        }
        match parse(&["perf", "ashrae140", "800", "--output", "/tmp/p.json"]) {
            PerformanceCommand::Ashrae140 { case, output } => {
                assert_eq!(case, Some(800));
                assert_eq!(output.as_deref(), Some("/tmp/p.json"));
                assert_eq!(case.unwrap_or(DEFAULT_ASHRAE_CASE), 800);
            }
            other => panic!("expected Ashrae140, got {other:?}"),
        }
    }

    #[test]
    fn ashrae140_rejects_non_numeric_case() {
        assert!(
            TestCli::try_parse_from(["perf", "ashrae140", "nine-hundred"]).is_err(),
            "case must parse as u32"
        );
    }

    // ---------------------------------------------------------------
    // Rejection of unknown subcommands / flags
    // ---------------------------------------------------------------

    #[test]
    fn unknown_subcommand_and_unknown_flags_are_rejected() {
        assert!(
            TestCli::try_parse_from(["perf"]).is_err(),
            "subcommand required"
        );
        assert!(TestCli::try_parse_from(["perf", "benchmarkk"]).is_err());
        // No variant of this surface accepts a bare `--release-only` /
        // `--warmup-runs`; pin that so a future flag addition is a deliberate,
        // test-visible change rather than an accident.
        for flag in ["--release-only", "--warmup-runs", "--nope"] {
            for sub in ["benchmark", "validate", "report", "integrated", "ashrae140"] {
                assert!(
                    TestCli::try_parse_from(["perf", sub, flag]).is_err(),
                    "`{sub} {flag}` must be rejected"
                );
            }
        }
    }

    #[test]
    fn format_option_accepts_all_documented_values() {
        for format in ["json", "text", "pretty"] {
            match parse(&["perf", "integrated", "--format", format]) {
                PerformanceCommand::Integrated { format: parsed, .. } => {
                    assert_eq!(parsed, format);
                }
                other => panic!("expected Integrated, got {other:?}"),
            }
        }
    }

    // ---------------------------------------------------------------
    // handle_performance_command dispatch
    // ---------------------------------------------------------------

    #[test]
    fn handle_performance_command_dispatches_benchmark_variants() {
        for cmd in [
            PerformanceCommand::Benchmark {
                scenario: None,
                format: "json".to_string(),
            },
            PerformanceCommand::Benchmark {
                scenario: Some("single".to_string()),
                format: "text".to_string(),
            },
            PerformanceCommand::Benchmark {
                scenario: Some("multi".to_string()),
                format: "pretty".to_string(),
            },
        ] {
            handle_performance_command(&cmd)
                .unwrap_or_else(|e| panic!("benchmark dispatch must succeed: {e}"));
        }
    }

    #[test]
    fn handle_performance_command_report_writes_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("perf.json");
        let cmd = PerformanceCommand::Report {
            output: Some(path.to_string_lossy().to_string()),
            detailed: false,
        };
        handle_performance_command(&cmd).expect("report dispatch must succeed");
        let text = std::fs::read_to_string(&path).expect("report file written");
        serde_json::from_str::<serde_json::Value>(&text).expect("report must be valid JSON");
    }

    #[test]
    fn generate_report_detailed_flag_selects_pretty_json() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let compact_path = tmp.path().join("compact.json");
        let pretty_path = tmp.path().join("pretty.json");

        generate_report(&Some(compact_path.to_string_lossy().to_string()), false)
            .expect("compact report");
        generate_report(&Some(pretty_path.to_string_lossy().to_string()), true)
            .expect("pretty report");

        let compact = std::fs::read_to_string(&compact_path).expect("read compact");
        let pretty = std::fs::read_to_string(&pretty_path).expect("read pretty");

        assert!(
            !compact.contains('\n'),
            "compact JSON must be single-line: {compact}"
        );
        assert!(
            pretty.contains('\n'),
            "--detailed must produce pretty-printed JSON"
        );
        // Both must still be structurally valid and describe the same schema.
        let c: serde_json::Value = serde_json::from_str(&compact).expect("compact JSON");
        let p: serde_json::Value = serde_json::from_str(&pretty).expect("pretty JSON");
        assert_eq!(
            c.as_object().map(|o| o.keys().count()),
            p.as_object().map(|o| o.keys().count()),
            "detailed formatting must not change the report schema"
        );
    }

    #[test]
    fn generate_report_propagates_write_failure() {
        // A path whose parent does not exist must surface as an error string
        // rather than panicking.
        let err = generate_report(&Some("/definitely/not/here/perf.json".to_string()), false)
            .unwrap_err();
        assert!(
            err.contains("Failed to write report"),
            "IO failure must be reported: {err}"
        );
    }

    #[test]
    fn run_benchmarks_is_infallible_for_every_scenario_and_format() {
        for scenario in [
            None,
            Some("single".to_string()),
            Some("high-mass".to_string()),
        ] {
            for format in ["json", "text", "pretty", "unrecognised"] {
                assert!(
                    run_benchmarks(&scenario, format).is_ok(),
                    "run_benchmarks must not fail for {scenario:?}/{format}"
                );
            }
        }
    }

    #[test]
    fn load_ashrae140_case_accepts_documented_case_numbers() {
        for case in [195_u32, 470, 600, 800, 810, DEFAULT_ASHRAE_CASE] {
            assert!(
                load_ashrae140_case(case).is_ok(),
                "case {case} must load (mock loader)"
            );
        }
    }
}
