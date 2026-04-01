//! Validator config tests for src/validation/ashrae_140_validator.rs

use fluxion::validation::ashrae_140_validator::ASHRAE140Validator;

#[test]
fn test_validator_new() {
    let validator = ASHRAE140Validator::new();
    assert!(!validator.is_skip_baseline_cases());
}

#[test]
fn test_validator_skip_baseline_cases() {
    let mut validator = ASHRAE140Validator::new();
    assert!(!validator.is_skip_baseline_cases());
    validator.skip_baseline_cases(true);
    assert!(validator.is_skip_baseline_cases());
    validator.skip_baseline_cases(false);
    assert!(!validator.is_skip_baseline_cases());
}

#[test]
fn test_validator_disable_diagnostics() {
    let mut validator = ASHRAE140Validator::new();
    validator.disable_diagnostics();
    assert!(!validator.is_skip_baseline_cases());
}

#[test]
fn test_validator_with_full_diagnostics() {
    let validator = ASHRAE140Validator::with_full_diagnostics();
    assert!(!validator.is_skip_baseline_cases());
}

#[test]
fn test_validator_add_diagnostic_case_range() {
    let mut validator = ASHRAE140Validator::new();
    validator.add_diagnostic_case_range("900-960".to_string());
}

#[test]
fn test_validator_with_diagnostics() {
    use fluxion::validation::diagnostic::DiagnosticConfig;
    let config = DiagnosticConfig::default();
    let validator = ASHRAE140Validator::with_diagnostics(config);
    assert!(!validator.is_skip_baseline_cases());
}

#[test]
fn test_diagnostic_config_default() {
    use fluxion::validation::diagnostic::DiagnosticConfig;
    let config = DiagnosticConfig::default();
    let _ = config.enabled;
}

#[test]
fn test_diagnostic_config_clone() {
    use fluxion::validation::diagnostic::DiagnosticConfig;
    let config = DiagnosticConfig::default();
    let cloned = config.clone();
    assert_eq!(config.enabled, cloned.enabled);
}

#[test]
fn test_orientation_enum() {
    use fluxion::validation::ashrae_140_cases::Orientation;
    assert_eq!(format!("{:?}", Orientation::South), "South");
    assert_eq!(format!("{:?}", Orientation::North), "North");
    assert_eq!(format!("{:?}", Orientation::East), "East");
    assert_eq!(format!("{:?}", Orientation::West), "West");
}

#[test]
fn test_orientation_clone_copy() {
    use fluxion::validation::ashrae_140_cases::Orientation;
    let o1 = Orientation::South;
    let o2 = o1;
    assert_eq!(o1, o2);
}

#[test]
fn test_ashrae140_case_enum() {
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    let _case600 = ASHRAE140Case::Case600;
    let _case900 = ASHRAE140Case::Case900;
}

#[test]
fn test_ashrae140_case_debug() {
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    assert_eq!(format!("{:?}", ASHRAE140Case::Case600), "Case600");
    assert_eq!(format!("{:?}", ASHRAE140Case::Case900), "Case900");
}

#[test]
fn test_ashrae140_case_clone_copy() {
    use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
    let c1 = ASHRAE140Case::Case600;
    let c2 = c1;
    assert_eq!(c1, c2);
}

#[test]
fn test_reference_program_enum() {
    use fluxion::validation::report::ReferenceProgram;
    assert_eq!(format!("{:?}", ReferenceProgram::EnergyPlus), "EnergyPlus");
    assert_eq!(format!("{:?}", ReferenceProgram::EspR), "EspR");
    assert_eq!(format!("{:?}", ReferenceProgram::TRNSYS), "TRNSYS");
}

#[test]
fn test_reference_program_clone() {
    use fluxion::validation::report::ReferenceProgram;
    let p1 = ReferenceProgram::EnergyPlus;
    let p2 = p1.clone();
    assert_eq!(p1, p2);
}

#[test]
fn test_reference_program_equality() {
    use fluxion::validation::report::ReferenceProgram;
    assert_eq!(ReferenceProgram::EnergyPlus, ReferenceProgram::EnergyPlus);
    assert_ne!(ReferenceProgram::EnergyPlus, ReferenceProgram::TRNSYS);
}

#[test]
fn test_window_spec_factories() {
    use fluxion::validation::ashrae_140_cases::WindowSpec;
    let clear = WindowSpec::double_clear_glass();
    assert!(clear.u_value > 0.0);

    let low_e = WindowSpec::double_low_e();
    assert!(low_e.u_value > 0.0);
}

#[test]
fn test_window_spec_clone() {
    use fluxion::validation::ashrae_140_cases::WindowSpec;
    let spec = WindowSpec::double_clear_glass();
    let cloned = spec.clone();
    assert!((spec.u_value - cloned.u_value).abs() < 0.01);
}

#[test]
fn test_window_spec_new() {
    use fluxion::validation::ashrae_140_cases::{GlassType, WindowSpec};
    let spec = WindowSpec::new(3.0, 0.5, 0.6, GlassType::DoubleClear);
    assert!((spec.u_value - 3.0).abs() < 0.01);
    assert!((spec.shgc - 0.5).abs() < 0.01);
}

#[test]
fn test_glass_type_enum() {
    use fluxion::validation::ashrae_140_cases::GlassType;
    let _clear = GlassType::DoubleClear;
    let _low_e = GlassType::DoubleLowE;
}

#[test]
fn test_construction_type_enum() {
    use fluxion::validation::ashrae_140_cases::ConstructionType;
    let _high = ConstructionType::HighMass;
    let _low = ConstructionType::LowMass;
}

#[test]
fn test_shading_type_enum() {
    use fluxion::validation::ashrae_140_cases::ShadingType;
    let _overhang = ShadingType::Overhang;
    let _fins = ShadingType::Fins;
}
