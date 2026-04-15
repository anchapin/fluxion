//! Unit tests for delta analysis module.
//!
//! Tests variant expansion, patch application, sweep generation,
//! and nested YAML manipulation.

use fluxion::analysis::delta::{
    apply_patch, expand_variants, generate_sweep_combinations, set_nested, DeltaConfig, Variant,
};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use std::collections::HashMap;

/// Create a minimal valid CaseSpec for testing delta analysis
fn create_test_case_spec() -> serde_yaml::Value {
    use fluxion::validation::ashrae_140_cases::{CaseBuilder, CaseSpec};
    let base_case: CaseSpec = CaseBuilder::new()
        .with_case_id("600".to_string())
        .with_description("Test case for delta analysis".to_string())
        .with_dimensions(8.0, 6.0, 2.7)
        .low_mass_construction()
        .with_south_window(12.0)
        .with_window_properties(
            fluxion::validation::ashrae_140_cases::WindowSpec::double_clear_glass(),
        )
        .with_internal_loads(fluxion::validation::ashrae_140_cases::InternalLoads::new(
            200.0, 0.6, 0.4,
        ))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_num_zones(1)
        .build()
        .expect("Failed to build base case");
    serde_yaml::to_value(&base_case).expect("Failed to serialize base case")
}

#[cfg(test)]
mod delta_unit_tests {
    use super::*;

    // ========================================================================
    // set_nested Tests
    // ========================================================================

    #[test]
    fn test_set_nested_top_level() {
        let mut value = serde_yaml::Value::Mapping(Default::default());
        set_nested(
            &mut value,
            "heating_setpoint",
            serde_yaml::Value::Number(serde_yaml::Number::from(20)),
        )
        .unwrap();

        let map = value.as_mapping().unwrap();
        assert_eq!(
            map.get(&serde_yaml::Value::String("heating_setpoint".to_string())),
            Some(&serde_yaml::Value::Number(serde_yaml::Number::from(20)))
        );
    }

    #[test]
    fn test_set_nested_two_levels() {
        let mut value = serde_yaml::Value::Mapping(Default::default());
        // First create the nested structure
        let mut inner_map = serde_yaml::Mapping::new();
        inner_map.insert(
            serde_yaml::Value::String("field".to_string()),
            serde_yaml::Value::Number(serde_yaml::Number::from(0)),
        );
        value.as_mapping_mut().unwrap().insert(
            serde_yaml::Value::String("hvac".to_string()),
            serde_yaml::Value::Mapping(inner_map),
        );

        set_nested(
            &mut value,
            "hvac.field",
            serde_yaml::Value::Number(serde_yaml::Number::from(42)),
        )
        .unwrap();

        let hvac = value
            .as_mapping()
            .unwrap()
            .get(&serde_yaml::Value::String("hvac".to_string()))
            .unwrap()
            .as_mapping()
            .unwrap();
        assert_eq!(
            hvac.get(&serde_yaml::Value::String("field".to_string())),
            Some(&serde_yaml::Value::Number(serde_yaml::Number::from(42)))
        );
    }

    #[test]
    fn test_set_nested_empty_path_error() {
        let mut value = serde_yaml::Value::Mapping(Default::default());
        let result = set_nested(&mut value, "", serde_yaml::Value::Null);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_nested_missing_key_error() {
        let mut value = serde_yaml::Value::Mapping(Default::default());
        let result = set_nested(
            &mut value,
            "missing.field",
            serde_yaml::Value::Number(serde_yaml::Number::from(1)),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_set_nested_not_object_error() {
        let mut value = serde_yaml::Value::Sequence(Default::default());
        let result = set_nested(
            &mut value,
            "field",
            serde_yaml::Value::Number(serde_yaml::Number::from(1)),
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // apply_patch Tests
    // ========================================================================

    #[test]
    fn test_apply_patch_single_field() {
        let base = serde_yaml::Value::Mapping(Default::default());
        let mut patch = HashMap::new();
        patch.insert(
            "heating_setpoint".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(20)),
        );

        let result = apply_patch(base, &patch).unwrap();
        let map = result.as_mapping().unwrap();
        assert!(map.contains_key(&serde_yaml::Value::String("heating_setpoint".to_string())));
    }

    #[test]
    fn test_apply_patch_multiple_fields() {
        let base = serde_yaml::Value::Mapping(Default::default());
        let mut patch = HashMap::new();
        patch.insert(
            "heating_setpoint".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(20)),
        );
        patch.insert(
            "cooling_setpoint".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(25)),
        );

        let result = apply_patch(base, &patch).unwrap();
        let map = result.as_mapping().unwrap();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_apply_patch_empty_patch() {
        let mut base = serde_yaml::Mapping::new();
        base.insert(
            serde_yaml::Value::String("existing".to_string()),
            serde_yaml::Value::Number(serde_yaml::Number::from(1)),
        );
        let base = serde_yaml::Value::Mapping(base);
        let patch = HashMap::new();

        let result = apply_patch(base, &patch).unwrap();
        let map = result.as_mapping().unwrap();
        assert_eq!(map.len(), 1);
    }

    // ========================================================================
    // generate_sweep_combinations Tests
    // ========================================================================

    #[test]
    fn test_sweep_single_parameter() {
        let base = serde_yaml::Value::Mapping(Default::default());
        let sweep_items = vec![("heating_setpoint".to_string(), vec![18.0, 20.0, 22.0])];
        let mut out = Vec::new();

        generate_sweep_combinations(&base, &sweep_items, 0, HashMap::new(), &mut out).unwrap();

        assert_eq!(out.len(), 3);
        assert!(out.iter().any(|(name, _)| name.contains("18.00")));
        assert!(out.iter().any(|(name, _)| name.contains("20.00")));
        assert!(out.iter().any(|(name, _)| name.contains("22.00")));
    }

    #[test]
    fn test_sweep_multiple_parameters() {
        let base = serde_yaml::Value::Mapping(Default::default());
        let sweep_items = vec![
            ("heating_setpoint".to_string(), vec![18.0, 20.0]),
            ("cooling_setpoint".to_string(), vec![24.0, 26.0]),
        ];
        let mut out = Vec::new();

        generate_sweep_combinations(&base, &sweep_items, 0, HashMap::new(), &mut out).unwrap();

        // Should generate 2x2 = 4 combinations
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_sweep_empty_sweep_items() {
        let base = serde_yaml::Value::Mapping(Default::default());
        let sweep_items: Vec<(String, Vec<f64>)> = vec![];
        let mut out = Vec::new();

        generate_sweep_combinations(&base, &sweep_items, 0, HashMap::new(), &mut out).unwrap();

        assert_eq!(out.len(), 1); // Single combination with no parameters
    }

    #[test]
    fn test_sweep_name_format() {
        let base = serde_yaml::Value::Mapping(Default::default());
        let sweep_items = vec![("param".to_string(), vec![1.5])];
        let mut out = Vec::new();

        generate_sweep_combinations(&base, &sweep_items, 0, HashMap::new(), &mut out).unwrap();

        assert_eq!(out.len(), 1);
        let (name, _): &(String, _) = &out[0];
        assert!(name.contains("param=1.50"));
    }

    // ========================================================================
    // expand_variants Tests
    // ========================================================================

    #[test]
    fn test_expand_variants_patch_only() {
        let base_case = create_test_case_spec();
        let mut yaml_map = serde_yaml::Mapping::new();
        yaml_map.insert(serde_yaml::Value::String("base".to_string()), base_case);

        let mut variants_seq = serde_yaml::Sequence::new();
        let mut variant_map = serde_yaml::Mapping::new();
        variant_map.insert(
            serde_yaml::Value::String("name".to_string()),
            serde_yaml::Value::String("higher_setpoint".to_string()),
        );
        let mut patch_map = serde_yaml::Mapping::new();
        patch_map.insert(
            serde_yaml::Value::String("heating_setpoint".to_string()),
            serde_yaml::Value::Number(serde_yaml::Number::from(22)),
        );
        variant_map.insert(
            serde_yaml::Value::String("patch".to_string()),
            serde_yaml::Value::Mapping(patch_map),
        );
        variants_seq.push(serde_yaml::Value::Mapping(variant_map));

        yaml_map.insert(
            serde_yaml::Value::String("variants".to_string()),
            serde_yaml::Value::Sequence(variants_seq),
        );

        let yaml_value = serde_yaml::Value::Mapping(yaml_map);
        let config: DeltaConfig = serde_yaml::from_value(yaml_value).unwrap();
        let results = expand_variants(&config).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "higher_setpoint");
    }

    #[test]
    fn test_expand_variants_patch_and_sweep() {
        let yaml_str = r#"
base:
  case_id: "600"
  description: "Test case for delta analysis"
  construction_type: "LowMass"
  num_zones: 1
  hvac: []
  windows: []
  night_ventilation: null
variants:
  - name: "combined"
    patch:
      cooling_setpoint: 26
    sweep:
      heating_setpoint: [18, 20]
"#;
        let config: DeltaConfig = serde_yaml::from_str(yaml_str).unwrap();
        let results = expand_variants(&config).unwrap();

        assert_eq!(results.len(), 2);
        // Names should include variant name and sweep values
        assert!(results[0].0.starts_with("combined"));
    }

    #[test]
    fn test_expand_variants_multiple_variants() {
        let yaml_str = r#"
base:
  case_id: "600"
  description: "Test case for delta analysis"
  construction_type: "LowMass"
  num_zones: 1
  hvac: []
  windows: []
  night_ventilation: null
variants:
  - name: "variant_a"
    patch:
      heating_setpoint: 22
  - name: "variant_b"
    patch:
      heating_setpoint: 18
"#;
        let config: DeltaConfig = serde_yaml::from_str(yaml_str).unwrap();
        let results = expand_variants(&config).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "variant_a");
        assert_eq!(results[1].0, "variant_b");
    }

    #[test]
    fn test_expand_variants_empty_variants() {
        let yaml_str = r#"
base:
  case_id: "600"
  description: "Test case for delta analysis"
  construction_type: "LowMass"
  num_zones: 1
  hvac: []
  windows: []
  night_ventilation: null
variants: []
"#;
        let config: DeltaConfig = serde_yaml::from_str(yaml_str).unwrap();
        let results = expand_variants(&config).unwrap();

        assert_eq!(results.len(), 0);
    }

    // ========================================================================
    // DeltaConfig Tests
    // ========================================================================

    #[test]
    fn test_delta_config_deserialization() {
        let yaml_str = r#"
base:
  case_id: "600"
  description: "Test case for delta analysis"
  construction_type: "LowMass"
  num_zones: 1
  hvac: []
  windows: []
  night_ventilation: null
variants:
  - name: "test_variant"
    patch:
      heating_setpoint: 22
"#;
        let config: DeltaConfig = serde_yaml::from_str(yaml_str).unwrap();
        assert_eq!(config.base.case_id, "600");
        assert_eq!(config.variants.len(), 1);
        assert_eq!(config.variants[0].name, "test_variant");
    }

    #[test]
    fn test_variant_clone() {
        let variant = Variant {
            name: "test".to_string(),
            patch: Some(HashMap::new()),
            sweep: None,
        };
        let cloned = variant.clone();
        assert_eq!(variant.name, cloned.name);
    }

    #[test]
    fn test_variant_debug() {
        let variant = Variant {
            name: "test".to_string(),
            patch: None,
            sweep: None,
        };
        let debug_str = format!("{:?}", variant);
        assert!(debug_str.contains("test"));
    }
}
