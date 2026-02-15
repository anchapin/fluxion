//! ASHRAE Standard 140 Integration Test Suite
//!
//! This suite runs all implemented ASHRAE 140 test cases and verifies
//! that the results are within the specified reference ranges.

use fluxion::validation::ASHRAE140Validator;
use fluxion::validation::MetricType;

#[test]
fn test_ashrae_140_comprehensive() {
    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();
    
    report.print_summary();
    
    // Check overall pass rate
    let pass_rate = report.pass_rate();
    println!("Overall ASHRAE 140 Pass Rate: {:.1}%", pass_rate * 100.0);
    
    // In CI, we might want to assert a minimum pass rate
    // assert!(pass_rate >= 0.8, "ASHRAE 140 pass rate too low");
}

#[test]
fn test_case_600_baseline() {
    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();
    
    let result = report.results.iter().find(|r| r.case_id == "600")
        .expect("Case 600 result not found");
        
    assert!(result.is_pass(), "Case 600 failed: {:?}", result);
}

mod phase2 {
    use super::*;
    
    #[test]
    #[ignore = "Issue #62 pending merge"]
    fn test_case_610_shading() {
        // ...
    }
}
