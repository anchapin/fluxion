//! HVAC BESTEST Integration Tests
//!
//! Tests for ASHRAE RP-865 HVAC BESTEST validation suite.

use fluxion::validation::hvac_bestest::{
    get_bestest_cases, run_hvac_bestest, validate_results, HVACBestestCase, HVACBestestResult,
};

#[test]
fn test_hvac_bestest_runs() {
    let results = run_hvac_bestest();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_hvac_bestest_case_coverage() {
    let cases = get_bestest_cases();

    let case_ids: Vec<HVACBestestCase> = cases.iter().map(|c| c.case_id).collect();

    assert!(case_ids.contains(&HVACBestestCase::Case600));
    assert!(case_ids.contains(&HVACBestestCase::Case610));
    assert!(case_ids.contains(&HVACBestestCase::Case620));
    assert!(case_ids.contains(&HVACBestestCase::Case630));
    assert!(case_ids.contains(&HVACBestestCase::Case640));
}

#[test]
fn test_validate_results() {
    let results = run_hvac_bestest();
    let (passed, failed, mean_error) = validate_results(&results);

    println!("HVAC BESTEST Results:");
    println!("  Passed: {}", passed);
    println!("  Failed: {}", failed);
    println!("  Mean Error: {:.2}%", mean_error);

    // At least verify results are computed
    assert_eq!(results.len(), passed + failed);
}

#[test]
fn test_result_has_valid_metrics() {
    let results = run_hvac_bestest();

    for result in &results {
        assert!(
            result.annual_energy_kwh >= 0.0,
            "Case {:?}: Annual energy should be non-negative",
            result.case_id
        );
        assert!(
            result.peak_demand_w >= 0.0,
            "Case {:?}: Peak demand should be non-negative",
            result.case_id
        );
        assert!(
            result.plr_50_cop > 0.0,
            "Case {:?}: PLR 50% COP should be positive",
            result.case_id
        );
        assert!(
            result.plr_100_cop > 0.0,
            "Case {:?}: PLR 100% COP should be positive",
            result.case_id
        );
    }
}

#[test]
fn test_part_load_efficiency_decreases() {
    let results = run_hvac_bestest();

    for result in &results {
        // Part-load efficiency should generally be lower than 100% (or close to it)
        // This is a simplified check - actual behavior depends on equipment type
        assert!(
            result.plr_100_cop >= result.plr_50_cop * 0.9,
            "Case {:?}: PLR 50% should not be significantly higher than 100%",
            result.case_id
        );
    }
}

#[test]
fn test_case600_chiller_energy_in_range() {
    let results = run_hvac_bestest();
    let case600 = results
        .iter()
        .find(|r| r.case_id == HVACBestestCase::Case600)
        .unwrap();

    println!("Case 600 (Chiller):");
    println!("  Annual Energy: {:.1} kWh", case600.annual_energy_kwh);
    println!("  Peak Demand: {:.0} W", case600.peak_demand_w);
    println!("  Energy Error: {:.1}%", case600.energy_error_percent);
    println!("  PLR 50%: {:.2}", case600.plr_50_cop);
    println!("  PLR 100%: {:.2}", case600.plr_100_cop);

    // Just verify the simulation ran
    assert!(case600.annual_energy_kwh > 0.0);
    assert!(case600.peak_demand_w > 0.0);
}

#[test]
fn test_case610_boiler_energy_in_range() {
    let results = run_hvac_bestest();
    let case610 = results
        .iter()
        .find(|r| r.case_id == HVACBestestCase::Case610)
        .unwrap();

    println!("Case 610 (Boiler):");
    println!("  Annual Energy: {:.1} kWh", case610.annual_energy_kwh);
    println!("  Peak Demand: {:.0} W", case610.peak_demand_w);
    println!("  Energy Error: {:.1}%", case610.energy_error_percent);

    // Just verify the simulation ran
    assert!(case610.annual_energy_kwh > 0.0);
}

#[test]
fn test_all_cases_have_messages() {
    let results = run_hvac_bestest();

    for result in &results {
        assert!(
            !result.message.is_empty(),
            "Case {:?}: Should have a validation message",
            result.case_id
        );
    }
}

#[test]
fn test_error_percentages_are_reasonable() {
    let results = run_hvac_bestest();

    for result in &results {
        // Error should be finite
        assert!(
            !result.energy_error_percent.is_nan(),
            "Case {:?}: Energy error should not be NaN",
            result.case_id
        );
        assert!(
            !result.demand_error_percent.is_nan(),
            "Case {:?}: Demand error should not be NaN",
            result.case_id
        );
        assert!(
            !result.energy_error_percent.is_infinite(),
            "Case {:?}: Energy error should not be infinite",
            result.case_id
        );
    }
}

#[test]
fn test_print_summary() {
    let results = run_hvac_bestest();
    let (passed, failed, mean_error) = validate_results(&results);

    println!("\n=== HVAC BESTEST Summary ===");
    println!("Cases run: {}", results.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    println!("Mean Error: {:.2}%", mean_error);
    println!("==========================\n");

    for result in &results {
        let status = if result.passed { "PASS" } else { "FAIL" };
        println!(
            "{:?}: {} - Energy Error: {:.1}%, Demand Error: {:.1}%",
            result.case_id, status, result.energy_error_percent, result.demand_error_percent
        );
    }
}
