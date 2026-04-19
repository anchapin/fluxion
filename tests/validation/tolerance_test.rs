use fluxion::validation::ValidationTolerance;

#[test]
fn test_validation_tolerance_default() {
    let tolerance = ValidationTolerance::default();
    assert_eq!(tolerance.nmbe_limit, 5.0);
    assert_eq!(tolerance.cv_rmse_limit, 10.0);
    assert_eq!(tolerance.mae_limit, 0.1);
}

#[test]
fn test_validation_tolerance_new() {
    let tolerance = ValidationTolerance::new(3.0, 8.0, 0.08);
    assert_eq!(tolerance.nmbe_limit, 3.0);
    assert_eq!(tolerance.cv_rmse_limit, 8.0);
    assert_eq!(tolerance.mae_limit, 0.08);
}

#[test]
fn test_validation_tolerance_strict() {
    let tolerance = ValidationTolerance::strict();
    assert_eq!(tolerance.nmbe_limit, 2.5);
    assert_eq!(tolerance.cv_rmse_limit, 5.0);
    assert_eq!(tolerance.mae_limit, 0.05);
}

#[test]
fn test_validation_tolerance_lenient() {
    let tolerance = ValidationTolerance::lenient();
    assert_eq!(tolerance.nmbe_limit, 10.0);
    assert_eq!(tolerance.cv_rmse_limit, 15.0);
    assert_eq!(tolerance.mae_limit, 0.2);
}

#[test]
fn test_within_nmbe_tolerance() {
    let tolerance = ValidationTolerance::default();
    assert!(tolerance.within_nmbe_tolerance(3.0));
    assert!(tolerance.within_nmbe_tolerance(5.0));
    assert!(!tolerance.within_nmbe_tolerance(6.0));
    assert!(tolerance.within_nmbe_tolerance(-4.0));
    assert!(!tolerance.within_nmbe_tolerance(-5.1));
}

#[test]
fn test_within_cv_rmse_tolerance() {
    let tolerance = ValidationTolerance::default();
    assert!(tolerance.within_cv_rmse_tolerance(5.0));
    assert!(tolerance.within_cv_rmse_tolerance(10.0));
    assert!(!tolerance.within_cv_rmse_tolerance(10.1));
}

#[test]
fn test_within_mae_tolerance() {
    let tolerance = ValidationTolerance::default();
    assert!(tolerance.within_mae_tolerance(0.05));
    assert!(tolerance.within_mae_tolerance(0.1));
    assert!(!tolerance.within_mae_tolerance(0.11));
}

#[test]
fn test_edge_cases() {
    // Test with zero values
    let tolerance = ValidationTolerance::new(0.0, 0.0, 0.0);
    assert!(tolerance.within_nmbe_tolerance(0.0));
    assert!(!tolerance.within_nmbe_tolerance(0.1));
    assert!(tolerance.within_cv_rmse_tolerance(0.0));
    assert!(!tolerance.within_cv_rmse_tolerance(0.1));
    assert!(tolerance.within_mae_tolerance(0.0));
    assert!(!tolerance.within_mae_tolerance(0.1));

    // Test with negative values for MAE and CV(RMSE) (should be treated as positive magnitudes)
    let tolerance = ValidationTolerance::default();
    // Note: CV(RMSE) and MAE checks don't use abs(), so negative values will fail
    assert!(!tolerance.within_cv_rmse_tolerance(-1.0)); // Negative CV(RMSE) should fail
    assert!(!tolerance.within_mae_tolerance(-0.05));   // Negative MAE should fail
}