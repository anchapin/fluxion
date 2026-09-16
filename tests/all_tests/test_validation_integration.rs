#[cfg(test)]
mod tests {
    use fluxion::sim::engine::ThermalModel;

    #[test]
    fn test_new_with_validation_valid_inputs() {
        // Create ThermalModel with valid inputs
        let result = ThermalModel::new_with_validation(
            1,    // num_zones
            2.5,  // window_u_value
            20.0, // hvac_setpoint
            0.4,  // h_tr_em
            10.0, // h_tr_ms
            8.0,  // h_tr_is
            2.5,  // h_tr_w
            0.5,  // h_ve
        );
        assert!(result.is_ok(), "Should succeed with valid inputs");
    }

    #[test]
    fn test_new_with_validation_invalid_h_tr_em() {
        // Try to create with negative h_tr_em
        let result = ThermalModel::new_with_validation(1, 2.5, 20.0, -0.4, 10.0, 8.0, 2.5, 0.5);
        match result {
            Ok(_) => panic!("Should fail with negative h_tr_em"),
            Err(error) => {
                assert!(
                    error.contains("Invalid h_tr_em"),
                    "Error should mention h_tr_em: {}",
                    error
                );
            }
        }
    }

    #[test]
    fn test_new_with_validation_invalid_hvac_setpoint() {
        // Try to create with out-of-range HVAC setpoint
        let result = ThermalModel::new_with_validation(1, 2.5, 40.0, 0.4, 10.0, 8.0, 2.5, 0.5);
        match result {
            Ok(_) => panic!("Should fail with invalid HVAC setpoint"),
            Err(error) => {
                assert!(
                    error.contains("Invalid hvac_setpoint"),
                    "Error should mention hvac_setpoint: {}",
                    error
                );
            }
        }
    }

    #[test]
    fn test_new_with_validation_invalid_window_u_value() {
        // Try to create with out-of-range window U-value
        let result = ThermalModel::new_with_validation(1, 10.0, 20.0, 0.4, 10.0, 8.0, 2.5, 0.5);
        match result {
            Ok(_) => panic!("Should fail with invalid window U-value"),
            Err(error) => {
                assert!(
                    error.contains("Invalid window_u_value"),
                    "Error should mention window_u_value: {}",
                    error
                );
            }
        }
    }

    #[test]
    fn test_new_runtime_validation() {
        // Runtime validation in new() should not panic with valid defaults
        let model = ThermalModel::new(1);
        assert_eq!(model.hvac.num_zones, 1);
    }

    #[test]
    fn test_new_with_validation_zero_thermal_conductance() {
        // Test that zero thermal conductance is rejected
        let result = ThermalModel::new_with_validation(1, 2.5, 20.0, 0.0, 10.0, 8.0, 2.5, 0.5);
        match result {
            Ok(_) => panic!("Should fail with zero h_tr_em"),
            Err(error) => {
                assert!(
                    error.contains("must be positive"),
                    "Error should mention positive requirement: {}",
                    error
                );
            }
        }
    }

    #[test]
    fn test_new_with_validation_all_thermal_conductances() {
        // Test validation of all thermal conductances
        let result = ThermalModel::new_with_validation(1, 2.5, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(
            result.is_ok(),
            "Should succeed with all valid thermal conductances"
        );
    }
}
