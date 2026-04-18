//! Known Issues Regression Tests
//!
//! This module implements QG-03: Turn known issues into tracked regression cases.
//! Each test corresponds to an issue in docs/KNOWN_ISSUES.md and:
//! - Has a reproducer that demonstrates the issue
//! - Documents severity and hypothesis
//! - Uses #[test] with #[ignore] to quarantine known failures
//!
//! Definition of done: No high-severity issue exists only as prose;
//! every critical issue is machine-traceable.

use fluxion::validation::report::{MetricType, ValidationStatus};
use fluxion::validation::ASHRAE140Validator;

mod solar_issues {
    use super::*;

    /// SOLAR-01: Peak Cooling Load Under-Prediction (Low-Mass FIXED, High-Mass OPEN)
    ///
    /// Severity: Critical (low-mass FIXED), High (high-mass still OPEN)
    /// Hypothesis: Solar distribution parameters insufficient for high-mass thermal dynamics
    /// GitHub Issue: #274
    ///
    /// Status: LOW-MASS (600 series) PASSES after Phase 7A BASE-05 fix
    ///         HIGH-MASS (900 series) STILL FAILS - peak cooling 2x reference
    #[test]
    #[ignore = "LIMIT-05: Known 5R1C model limitation for high-mass peak cooling (h_tr_ms additive issue)"]
    fn test_solar01_high_mass_peak_cooling_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== SOLAR-01: High-Mass Peak Cooling Regression ===");

        let cases = ["900", "910", "920", "930", "940", "950"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::PeakCooling))
            {
                let passed = matches!(result.status, ValidationStatus::Pass);
                println!(
                    "Case {}: {:.2} kW [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "SOLAR-01: Cases {:?} peak cooling fail",
            failures
        );
    }

    /// SOLAR-02: Annual Cooling Energy Under-Prediction (High-Mass)
    ///
    /// Severity: High
    /// Hypothesis: Solar gain timing and thermal mass coupling incorrect for high-mass
    /// GitHub Issue: #275
    #[test]
    #[ignore = "Known issue - SOLAR-02: High-mass annual cooling under-prediction"]
    fn test_solar02_high_mass_annual_cooling_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== SOLAR-02: High-Mass Annual Cooling Regression ===");

        let cases = ["900", "910", "920", "930", "940", "950"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::AnnualCooling))
            {
                let passed = matches!(result.status, ValidationStatus::Pass);
                println!(
                    "Case {}: {:.2} MWh [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "SOLAR-02: Cases {:?} annual cooling fail",
            failures
        );
    }

    /// SOLAR-03: Solar Shading Cases Not Sensitive to Shading Changes
    ///
    /// Severity: Medium
    /// Hypothesis: Shading coefficient application or solar distribution incorrect
    #[test]
    #[ignore = "Known issue - SOLAR-03: Shading sensitivity insufficient"]
    fn test_solar03_shading_sensitivity_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== SOLAR-03: Shading Sensitivity Regression ===");

        let shade_cases: [(&str, &str, &str); 4] = [
            ("610", "600", "South shading (low-mass)"),
            ("630", "600", "E/W shading (low-mass)"),
            ("910", "900", "South shading (high-mass)"),
            ("930", "900", "E/W shading (high-mass)"),
        ];

        let mut failures: Vec<&str> = Vec::new();

        for (shade_id, base_id, desc) in shade_cases {
            let base_result = report
                .results
                .iter()
                .find(|r| r.case_id == base_id && matches!(r.metric, MetricType::AnnualCooling));
            let shade_result = report
                .results
                .iter()
                .find(|r| r.case_id == shade_id && matches!(r.metric, MetricType::AnnualCooling));

            if let (Some(base), Some(shade)) = (base_result, shade_result) {
                let reduction_pct =
                    (base.fluxion_value - shade.fluxion_value) / base.fluxion_value * 100.0;
                let status = if reduction_pct > 20.0 { "PASS" } else { "FAIL" };
                println!(
                    "{} ({}): {:.1}% reduction - {}",
                    shade_id, desc, reduction_pct, status
                );

                if reduction_pct <= 20.0 {
                    failures.push(shade_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "SOLAR-03: Shading cases {:?} insufficient sensitivity",
            failures
        );
    }

    /// SOLAR-04: Night Ventilation Cooling Ineffective
    ///
    /// Severity: Medium
    /// Hypothesis: Ventilation air exchange not implemented correctly
    /// GitHub Issue: #276
    #[test]
    #[ignore = "Known issue - SOLAR-04: Night ventilation not effective"]
    fn test_solar04_night_ventilation_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== SOLAR-04: Night Ventilation Effectiveness Regression ===");

        let vent_cases: [(&str, &str, &str); 2] = [
            ("650", "600", "Low-mass night vent"),
            ("950", "900", "High-mass night vent"),
        ];

        let mut failures: Vec<&str> = Vec::new();

        for (vent_id, base_id, desc) in vent_cases {
            let base_result = report
                .results
                .iter()
                .find(|r| r.case_id == base_id && matches!(r.metric, MetricType::AnnualCooling));
            let vent_result = report
                .results
                .iter()
                .find(|r| r.case_id == vent_id && matches!(r.metric, MetricType::AnnualCooling));

            if let (Some(base), Some(vent)) = (base_result, vent_result) {
                let reduction_pct =
                    (base.fluxion_value - vent.fluxion_value) / base.fluxion_value * 100.0;
                let status = if reduction_pct > 15.0 { "PASS" } else { "FAIL" };
                println!(
                    "{} ({}): {:.1}% reduction - {}",
                    vent_id, desc, reduction_pct, status
                );

                if reduction_pct <= 15.0 {
                    failures.push(vent_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "SOLAR-04: Night ventilation cases {:?} show minimal effect",
            failures
        );
    }
}

mod free_floating_issues {
    use super::*;

    /// FREE-01: Maximum Free-Floating Temperature Under-Prediction (Low-Mass)
    ///
    /// Severity: High
    /// Hypothesis: Solar gain distribution or heat loss coefficients incorrect
    #[test]
    #[ignore = "Known issue - FREE-01: Low-mass free-float max temp under-predicted"]
    fn test_free01_low_mass_max_temp_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== FREE-01: Low-Mass Max Free-Float Temp Regression ===");

        let cases = [("600FF", 64.90, 75.10), ("650FF", 63.20, 73.50)];
        let mut failures: Vec<&str> = Vec::new();

        for (case_id, ref_min, ref_max) in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::MaxFreeFloat))
            {
                let passed = result.fluxion_value >= ref_min && result.fluxion_value <= ref_max;
                println!(
                    "{}: {:.1}°C [ref: {:.1}-{:.1}] - {:?}",
                    case_id, result.fluxion_value, ref_min, ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "FREE-01: Low-mass cases {:?} max temp outside reference",
            failures
        );
    }

    /// FREE-02: Minimum Free-Floating Temperature Over-Prediction (High-Mass)
    ///
    /// Severity: Medium
    /// Hypothesis: Inadequate heat loss or insufficient thermal mass responsiveness
    #[test]
    #[ignore = "Known issue - FREE-02: 950FF min temp still high"]
    fn test_free02_high_mass_min_temp_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== FREE-02: High-Mass Min Free-Float Temp Regression ===");

        let cases = [("900FF", -6.40, -1.60), ("950FF", -20.20, -17.80)];
        let mut failures: Vec<&str> = Vec::new();

        for (case_id, ref_min, ref_max) in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::MinFreeFloat))
            {
                let passed = result.fluxion_value >= ref_min && result.fluxion_value <= ref_max;
                println!(
                    "{}: {:.1}°C [ref: {:.1}-{:.1}] - {:?}",
                    case_id, result.fluxion_value, ref_min, ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - 950FF currently FAILS");
        assert!(
            failures.is_empty(),
            "FREE-02: High-mass cases {:?} min temp outside reference",
            failures
        );
    }

    /// FREE-03: Free-Floating Temperature Swings Reduced
    ///
    /// Severity: Medium
    /// Hypothesis: Thermal mass time constant too long or heat transfer coefficients too high
    #[test]
    #[ignore = "Known issue - FREE-03: Temperature swings reduced vs reference"]
    fn test_free03_temperature_swing_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== FREE-03: Temperature Swing Amplitude Regression ===");

        let cases = [
            ("600FF", 64.90, 75.10, -18.80, -15.60),
            ("650FF", 63.20, 73.50, -23.00, -21.00),
            ("900FF", 41.80, 46.40, -6.40, -1.60),
            ("950FF", 35.50, 38.50, -20.20, -17.80),
        ];

        let mut failures: Vec<&str> = Vec::new();

        for (case_id, max_lo, max_hi, min_lo, min_hi) in cases {
            let max_result = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::MaxFreeFloat));
            let min_result = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::MinFreeFloat));

            if let (Some(max_r), Some(min_r)) = (max_result, min_result) {
                let max_pass = max_r.fluxion_value >= max_lo && max_r.fluxion_value <= max_hi;
                let min_pass = min_r.fluxion_value >= min_lo && min_r.fluxion_value <= min_hi;

                println!(
                    "{}: max={:.1}°C min={:.1}°C swing={:.1}°C (max PASS={}, min PASS={})",
                    case_id,
                    max_r.fluxion_value,
                    min_r.fluxion_value,
                    max_r.fluxion_value - min_r.fluxion_value,
                    max_pass,
                    min_pass
                );

                if !max_pass || !min_pass {
                    failures.push(case_id);
                }
            }
        }

        println!("Status: REGRESSION TEST - Currently FAILS");
        assert!(
            failures.is_empty(),
            "FREE-03: Cases {:?} temperature swings outside reference",
            failures
        );
    }
}

mod model_limit_issues {
    use super::*;

    /// LIMIT-05: High-Mass Peak Cooling Overprediction (5R1C Model Limitation)
    ///
    /// Severity: Medium
    /// Hypothesis: 5R1C lumped thermal capacitance cannot handle high mass + large dt
    /// Root cause: h_tr_ms additive model overcounts thermal coupling
    #[test]
    #[ignore = "LIMIT-05: Known 5R1C model limitation - h_tr_ms additive overcounts thermal coupling"]
    fn test_limit05_high_mass_peak_cooling_model_limitation() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== LIMIT-05: High-Mass Peak Cooling Model Limitation ===");
        println!(
            "Root cause: h_tr_ms = h_tr_ms_wall + h_tr_ms_roof + h_tr_ms_floor (not additive)"
        );
        println!();

        let cases = ["900", "910", "920", "930", "940", "950"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::PeakCooling))
            {
                let passed = matches!(result.status, ValidationStatus::Pass);
                println!(
                    "{}: {:.2} kW [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        println!("\nHigh-mass peak cooling failures: {}", failures.len());
        println!("Solution: Requires Phase 6+ multi-node thermal model redesign");
        assert!(
            failures.is_empty(),
            "LIMIT-05: High-mass peak cooling deviations: {:?}",
            failures
        );
    }

    /// LIMIT-06: 600-Series Annual Heating Correction (Empirical - FIXED)
    ///
    /// Severity: Medium (FIXED)
    /// GitHub Issue: #522
    #[test]
    fn test_limit06_600_series_heating_no_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== LIMIT-06: 600-Series Annual Heating Regression Guard ===");

        let cases = ["600", "610", "620", "630", "640"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::AnnualHeating))
            {
                let passed = matches!(
                    result.status,
                    ValidationStatus::Pass | ValidationStatus::Warning
                );
                println!(
                    "{}: {:.2} MWh [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        assert!(
            failures.is_empty(),
            "LIMIT-06 REGRESSION: 600-series heating cases {:?} outside reference",
            failures
        );
    }

    /// LIMIT-06 UPDATE: 600-Series Annual Cooling (FIXED in Phase 36-04)
    ///
    /// Severity: Medium (FIXED)
    /// GitHub Issue: #531
    #[test]
    fn test_limit06_600_series_cooling_no_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== LIMIT-06 UPDATE: 600-Series Annual Cooling Regression Guard ===");

        let cases = ["600", "610", "620", "630", "640", "650"];
        let mut failures: Vec<&str> = Vec::new();

        for case_id in cases {
            if let Some(result) = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::AnnualCooling))
            {
                let passed = matches!(
                    result.status,
                    ValidationStatus::Pass | ValidationStatus::Warning
                );
                println!(
                    "{}: {:.2} MWh [{:.2}, {:.2}] - {:?}",
                    case_id, result.fluxion_value, result.ref_min, result.ref_max, result.status
                );

                if !passed {
                    failures.push(case_id);
                }
            }
        }

        assert!(
            failures.is_empty(),
            "LIMIT-06 UPDATE REGRESSION: 600-series cooling cases {:?} outside reference",
            failures
        );
    }
}

mod additional_regressions {
    use super::*;

    /// Issue #532: Case 195 producing zero annual energy
    ///
    /// Severity: High
    /// Hypothesis: Case 195 configuration error or validation bug
    #[test]
    #[ignore = "Issue #532: Case 195 zero annual energy investigation"]
    fn test_issue532_case195_energy_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== Issue #532: Case 195 Annual Energy Regression ===");

        let heating = report
            .results
            .iter()
            .find(|r| r.case_id == "195" && matches!(r.metric, MetricType::AnnualHeating));
        let cooling = report
            .results
            .iter()
            .find(|r| r.case_id == "195" && matches!(r.metric, MetricType::AnnualCooling));

        let h_val = heating.map(|r| r.fluxion_value).unwrap_or(0.0);
        let c_val = cooling.map(|r| r.fluxion_value).unwrap_or(0.0);

        println!("Case 195 Annual Heating: {:.2} MWh", h_val);
        println!("Case 195 Annual Cooling: {:.2} MWh", c_val);

        let is_zero = h_val < 0.01 && c_val < 0.01;

        assert!(
            !is_zero,
            "Issue #532 REGRESSION: Case 195 produces near-zero annual energy"
        );
        println!("Status: PASS - Case 195 produces valid energy values");
    }

    /// Issue #533: Case 600-series peak load underprediction
    ///
    /// Severity: High
    /// Hypothesis: Solar distribution or sensitivity calculation incorrect for 600 series
    #[test]
    #[ignore = "Issue #533: Case 600-series peak load underprediction investigation"]
    fn test_issue533_600_series_peak_load_regression() {
        let validator = ASHRAE140Validator::new();
        let report = validator.validate_analytical_engine();

        println!("\n=== Issue #533: 600-Series Peak Load Regression ===");

        let cases = ["600", "610", "620", "630", "640", "650"];
        let mut failures: Vec<(&str, &str)> = Vec::new();

        for case_id in cases {
            let peak_h = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::PeakHeating));
            let peak_c = report
                .results
                .iter()
                .find(|r| r.case_id == case_id && matches!(r.metric, MetricType::PeakCooling));

            let hp = peak_h
                .map(|r| {
                    let passed = matches!(r.status, ValidationStatus::Pass);
                    if !passed {
                        failures.push((case_id, "PeakHeating"));
                    }
                    format!("{:.2}kW - {:?}", r.fluxion_value, r.status)
                })
                .unwrap_or("N/A".to_string());

            let cp = peak_c
                .map(|r| {
                    let passed = matches!(r.status, ValidationStatus::Pass);
                    if !passed {
                        failures.push((case_id, "PeakCooling"));
                    }
                    format!("{:.2}kW - {:?}", r.fluxion_value, r.status)
                })
                .unwrap_or("N/A".to_string());

            println!("{}: Heat={} Cool={}", case_id, hp, cp);
        }

        assert!(
            failures.is_empty(),
            "Issue #533 REGRESSION: 600-series peak load failures: {:?}",
            failures
        );
    }
}
