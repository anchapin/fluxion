use fluxion::validation::guardrails::{self, GuardrailBaseline};
use fluxion::validation::report::{BenchmarkReport, BenchmarkData, ValidationResult, MetricType, ValidationStatus};
use std::time::{Instant, Duration};

fn make_report(percent_errors: &[f64], statuses: &[ValidationStatus], duration_secs: f64) -> BenchmarkReport {
    let mut report = BenchmarkReport::new();
    for (i, &pe) in percent_errors.iter().enumerate() {
        report.results.push(ValidationResult {
            case_id: format!("case{}", i),
            metric: MetricType::AnnualHeating,
            fluxion_value: 0.0,
            ref_min: 1.0,
            ref_max: 2.0,
            percent_error: pe,
            status: statuses[i],
        });
    }
    // Add benchmark data for throughput (though not used by guardrails)
    report.benchmark_data.insert("case0".to_string(), BenchmarkData::new());
    // Set timing
    let start = Instant::now();
    report.start_time = Some(start);
    report.end_time = Some(start + Duration::from_secs_f64(duration_secs));
    report
}

#[test]
fn test_guardrail_mae_failure() {
    // Baseline mae=10.0, report mae=10.3 (>2%)
    let baseline = GuardrailBaseline {
        mae: 10.0,
        max_deviation: 1000.0,
        pass_rate: 100.0,
        validation_time_seconds: 50.0,
    };
    // One result with percent_error=10.3 → mae=10.3
    let report = make_report(&[10.3], &[ValidationStatus::Pass], 50.0);

    let (passed, failures) = guardrails::check(&report, &baseline);
    assert!(!passed);
    assert!(failures.iter().any(|f| f.contains("MAE")));
}

#[test]
fn test_guardrail_maxdev_failure() {
    // Baseline max_deviation=20.0, report maxdev=22.0 (>10%)
    let baseline = GuardrailBaseline {
        mae: 1000.0,
        max_deviation: 20.0,
        pass_rate: 100.0,
        validation_time_seconds: 50.0,
    };
    // One result with percent_error=22.0 → maxdev=22.0
    let report = make_report(&[22.0], &[ValidationStatus::Pass], 50.0);

    let (passed, failures) = guardrails::check(&report, &baseline);
    assert!(!passed);
    assert!(failures.iter().any(|f| f.contains("Max Deviation")));
}

#[test]
fn test_guardrail_passrate_failure() {
    // Baseline pass_rate=100.0, report pass_rate=94.0 (drop 6pp >5)
    let baseline = GuardrailBaseline {
        mae: 1000.0,
        max_deviation: 1000.0,
        pass_rate: 100.0,
        validation_time_seconds: 50.0,
    };
    // Two results: one Pass, one Fail → pass_rate = 50%
    let statuses = vec![ValidationStatus::Pass, ValidationStatus::Fail];
    let report = make_report(&[0.0, 0.0], &statuses, 50.0);

    let (passed, failures) = guardrails::check(&report, &baseline);
    assert!(!passed);
    assert!(failures.iter().any(|f| f.contains("Pass Rate")));
}

#[test]
fn test_guardrail_all_pass() {
    let baseline = GuardrailBaseline {
        mae: 10.0,
        max_deviation: 20.0,
        pass_rate: 90.0,
        validation_time_seconds: 100.0,
    };
    let report = make_report(&[5.0], &[ValidationStatus::Pass], 50.0); // duration 50 < 110

    let (passed, failures) = guardrails::check(&report, &baseline);
    assert!(passed);
    assert!(failures.is_empty());
}

#[test]
fn test_guardrail_time_warning_only() {
    let baseline = GuardrailBaseline {
        mae: 1000.0,
        max_deviation: 1000.0,
        pass_rate: 100.0,
        validation_time_seconds: 50.0,
    };
    // duration = 60 (>55 = 110% of 50) but within other thresholds
    let report = make_report(&[0.0], &[ValidationStatus::Pass], 60.0);

    let (passed, failures) = guardrails::check(&report, &baseline);
    assert!(passed, "Time overage should not cause failure");
    assert!(failures.is_empty());
}

#[test]
fn test_guardrail_multiple_failures() {
    // Baseline relatively low thresholds so this report violates MAE, MaxDev, PassRate
    let baseline = GuardrailBaseline {
        mae: 10.0,
        max_deviation: 10.0,
        pass_rate: 80.0,
        validation_time_seconds: 50.0,
    };
    // Two results: high error (causes MAE and MaxDev) and one Pass + one Fail to lower pass_rate
    // We'll have three results: two with high error and one pass to bring pass_rate = 33%? We need at least one fail to lower pass_rate.
    // Let's have two results: one with high error (status Pass), one with status Fail (any error). Then pass_rate = 50% (<75% if baseline 80? Actually baseline pass_rate=80, drop 30pp >5 -> fail.
    // MaxDev: max percent_error = 15 (>11 -> fail)
    // MAE: if we have both 15 and 15, mae = 15 >10.2 -> fail.
    let report = make_report(&[15.0, 15.0], &[ValidationStatus::Pass, ValidationStatus::Fail], 50.0);

    let (passed, failures) = guardrails::check(&report, &baseline);
    assert!(!passed);
    assert!(failures.len() >= 3); // expect MAE, MaxDev, PassRate failures
    assert!(failures.iter().any(|f| f.contains("MAE")));
    assert!(failures.iter().any(|f| f.contains("Max Deviation")));
    assert!(failures.iter().any(|f| f.contains("Pass Rate")));
}
