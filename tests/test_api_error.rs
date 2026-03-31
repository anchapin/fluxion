#[cfg(test)]
mod tests {
    use fluxion::api::error::FluxionError;

    #[test]
    fn test_fluxion_error_validation() {
        let error = FluxionError::Validation("Invalid parameter".to_string());
        let msg = format!("{}", error);
        assert!(msg.contains("Parameter validation error"));
        assert!(msg.contains("Invalid parameter"));
    }

    #[test]
    fn test_fluxion_error_surrogate() {
        let error = FluxionError::Surrogate("Model not found".to_string());
        let msg = format!("{}", error);
        assert!(msg.contains("Surrogate model error"));
        assert!(msg.contains("Model not found"));
    }

    #[test]
    fn test_fluxion_error_simulation() {
        let error = FluxionError::Simulation("NaN detected".to_string());
        let msg = format!("{}", error);
        assert!(msg.contains("Simulation error"));
        assert!(msg.contains("NaN detected"));
    }

    #[test]
    fn test_fluxion_error_debug() {
        let error = FluxionError::Validation("test".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("Validation"));
    }

    #[test]
    fn test_fluxion_error_is_validation() {
        let error = FluxionError::Validation("test".to_string());
        assert!(matches!(error, FluxionError::Validation(_)));
    }

    #[test]
    fn test_fluxion_error_is_surrogate() {
        let error = FluxionError::Surrogate("test".to_string());
        assert!(matches!(error, FluxionError::Surrogate(_)));
    }

    #[test]
    fn test_fluxion_error_is_simulation() {
        let error = FluxionError::Simulation("test".to_string());
        assert!(matches!(error, FluxionError::Simulation(_)));
    }

    #[test]
    fn test_fluxion_error_empty_message() {
        let error = FluxionError::Validation("".to_string());
        let msg = format!("{}", error);
        assert!(msg.contains("Parameter validation error"));
    }

    #[test]
    fn test_fluxion_error_long_message() {
        let long_msg = "a".repeat(1000);
        let error = FluxionError::Simulation(long_msg.clone());
        let msg = format!("{}", error);
        assert!(msg.contains(&long_msg));
    }

    #[test]
    fn test_fluxion_error_special_characters() {
        let error = FluxionError::Validation("Error: <invalid> & \"quotes\"".to_string());
        let msg = format!("{}", error);
        assert!(msg.contains("<invalid>"));
        assert!(msg.contains("quotes"));
    }

    #[test]
    fn test_fluxion_error_newline_message() {
        let error = FluxionError::Simulation("Line 1\nLine 2".to_string());
        let msg = format!("{}", error);
        assert!(msg.contains("Line 1"));
        assert!(msg.contains("Line 2"));
    }
}
