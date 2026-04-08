// validation/ashrae140/tests.rs
#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_case_enumeration() {
        let cases = cases::get_all_cases();
        assert!(!cases.is_empty(), "ASHRAE 140 cases should not be empty");
        assert!(cases.len() > 20, "Should have at least 20 ASHRAE 140 cases");
    }

    #[test]
    fn test_case_500() {
        let case = cases::ASHRAE140Case::Case500;
        assert_eq!(case.case_id(), "500");
        assert_eq!(
            case.description(),
            "Low mass baseline with alternative construction"
        );
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_699() {
        let case = cases::ASHRAE140Case::Case699;
        assert_eq!(case.case_id(), "699");
        assert_eq!(
            case.description(),
            "Low mass with comprehensive HVAC integration"
        );
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_series() {
        let series_600 = cases::get_cases_by_series("600");
        assert_eq!(series_600.len(), 8, "600 series should have 8 cases");

        let series_900 = cases::get_cases_by_series("900");
        assert_eq!(series_900.len(), 8, "900 series should have 8 cases");

        let series_500_699 = cases::get_cases_by_series("500-699");
        assert!(
            series_500_699.len() >= 12,
            "500-699 series should have at least 12 cases"
        );
    }

    #[test]
    fn test_validator_creation() {
        let validator = ashrae140::ASHRAE140Validator::new();
        // Just test that it can be created without panicking
        assert!(true, "ASHRAE140Validator should be creatable");
    }
}
