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
    fn test_case_501() {
        let case = cases::ASHRAE140Case::Case501;
        assert_eq!(case.case_id(), "501");
        assert_eq!(case.description(), "Low mass with north windows");
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_502() {
        let case = cases::ASHRAE140Case::Case502;
        assert_eq!(case.case_id(), "502");
        assert_eq!(case.description(), "Low mass with double glazing");
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_503() {
        let case = cases::ASHRAE140Case::Case503;
        assert_eq!(case.case_id(), "503");
        assert_eq!(case.description(), "Low mass with triple glazing");
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_504() {
        let case = cases::ASHRAE140Case::Case504;
        assert_eq!(case.case_id(), "504");
        assert_eq!(case.description(), "Low mass with reduced infiltration");
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_505() {
        let case = cases::ASHRAE140Case::Case505;
        assert_eq!(case.case_id(), "505");
        assert_eq!(case.description(), "Low mass with increased infiltration");
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_506() {
        let case = cases::ASHRAE140Case::Case506;
        assert_eq!(case.case_id(), "506");
        assert_eq!(
            case.description(),
            "Low mass with alternative roof construction"
        );
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_507() {
        let case = cases::ASHRAE140Case::Case507;
        assert_eq!(case.case_id(), "507");
        assert_eq!(
            case.description(),
            "Low mass with alternative floor construction"
        );
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_508() {
        let case = cases::ASHRAE140Case::Case508;
        assert_eq!(case.case_id(), "508");
        assert_eq!(case.description(), "Low mass with reduced window area");
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_509() {
        let case = cases::ASHRAE140Case::Case509;
        assert_eq!(case.case_id(), "509");
        assert_eq!(case.description(), "Low mass with increased window area");
        assert!(!case.is_free_floating());
    }

    #[test]
    fn test_case_510() {
        let case = cases::ASHRAE140Case::Case510;
        assert_eq!(case.case_id(), "510");
        assert_eq!(case.description(), "Low mass with alternative orientation");
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
        assert_eq!(
            series_500_699.len(),
            12,
            "500-699 series should have exactly 12 cases"
        );
    }

    #[test]
    fn test_expanded_cases() {
        let expanded_cases = cases::get_expanded_cases();
        assert_eq!(expanded_cases.len(), 12, "Should have 12 expanded cases");

        // Verify all expanded cases are present
        let case_ids: Vec<String> = expanded_cases.iter().map(|c| c.case_id()).collect();
        assert!(case_ids.contains(&"500".to_string()));
        assert!(case_ids.contains(&"501".to_string()));
        assert!(case_ids.contains(&"502".to_string()));
        assert!(case_ids.contains(&"503".to_string()));
        assert!(case_ids.contains(&"504".to_string()));
        assert!(case_ids.contains(&"505".to_string()));
        assert!(case_ids.contains(&"506".to_string()));
        assert!(case_ids.contains(&"507".to_string()));
        assert!(case_ids.contains(&"508".to_string()));
        assert!(case_ids.contains(&"509".to_string()));
        assert!(case_ids.contains(&"510".to_string()));
        assert!(case_ids.contains(&"699".to_string()));
    }

    #[test]
    fn test_is_expanded_case() {
        // Test that expanded cases are correctly identified
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case500));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case501));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case502));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case503));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case504));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case505));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case506));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case507));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case508));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case509));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case510));
        assert!(cases::is_expanded_case(&cases::ASHRAE140Case::Case699));

        // Test that non-expanded cases are not identified as expanded
        assert!(!cases::is_expanded_case(&cases::ASHRAE140Case::Case600));
        assert!(!cases::is_expanded_case(&cases::ASHRAE140Case::Case900));
    }

    #[test]
    fn test_validator_creation() {
        let validator = ashrae140::ASHRAE140Validator::new();
        // Just test that it can be created without panicking
        assert!(true, "ASHRAE140Validator should be creatable");
    }
}
