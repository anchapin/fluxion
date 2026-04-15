use fluxion::validation::{
    multi_reference::{CaseRefs, MultiReferenceDB, ProgramRange},
    BenchmarkReport, MetricType, ValidationStatus,
};
use std::collections::HashMap;

#[test]
fn test_multireference_status() {
    // Build a minimal multi-reference DB with two programs
    let mut cases = HashMap::new();
    let mut annual_heating = HashMap::new();
    annual_heating.insert(
        "EnergyPlus".to_string(),
        ProgramRange { min: 5.0, max: 5.5 },
    );
    annual_heating.insert("ESP-r".to_string(), ProgramRange { min: 6.0, max: 6.5 });

    let case_refs = CaseRefs {
        annual_heating: Some(annual_heating),
        annual_cooling: Some(HashMap::new()),
        peak_heating: Some(HashMap::new()),
        peak_cooling: Some(HashMap::new()),
        min_free_float: None,
        max_free_float: None,
    };
    cases.insert("600".to_string(), case_refs);

    let db = MultiReferenceDB {
        version: "test".to_string(),
        source: None,
        cases,
    };

    // Case 1: Fluxion value within EnergyPlus range -> overall PASS
    let mut report1 = BenchmarkReport::new();
    report1.add_result_with_multi("600", MetricType::AnnualHeating, 5.2, &db);
    let res1 = &report1.results[0];
    let per1 = res1.per_program.as_ref().unwrap();
    assert_eq!(per1["EnergyPlus"], ValidationStatus::Pass);
    assert_eq!(per1["ESP-r"], ValidationStatus::Fail);
    assert_eq!(res1.status, ValidationStatus::Pass);

    // Case 2: Fluxion within ESP-r but outside EnergyPlus -> overall WARN
    let mut report2 = BenchmarkReport::new();
    report2.add_result_with_multi("600", MetricType::AnnualHeating, 6.2, &db);
    let res2 = &report2.results[0];
    let per2 = res2.per_program.as_ref().unwrap();
    assert_eq!(per2["EnergyPlus"], ValidationStatus::Fail);
    assert_eq!(per2["ESP-r"], ValidationStatus::Pass);
    assert_eq!(res2.status, ValidationStatus::Warning);

    // Case 3: Fluxion outside all programs -> overall FAIL
    let mut report3 = BenchmarkReport::new();
    report3.add_result_with_multi("600", MetricType::AnnualHeating, 4.0, &db);
    let res3 = &report3.results[0];
    let per3 = res3.per_program.as_ref().unwrap();
    assert_eq!(per3["EnergyPlus"], ValidationStatus::Fail);
    assert_eq!(per3["ESP-r"], ValidationStatus::Fail);
    assert_eq!(res3.status, ValidationStatus::Fail);
}
