#[test]
fn generate_delta_yaml() {
    use fluxion::validation::{ASHRAE140Case, CaseSpec};
    use serde_yaml;

    let base_spec: CaseSpec = ASHRAE140Case::Case600.spec();

    // Build variants as separate mappings
    let variant1 = serde_yaml::Mapping::from_iter([
        (
            serde_yaml::Value::String("name".into()),
            serde_yaml::to_value("high_infil").unwrap(),
        ),
        (
            serde_yaml::Value::String("patch".into()),
            serde_yaml::to_value(&serde_yaml::Mapping::from_iter([(
                serde_yaml::Value::String("infiltration_ach".into()),
                serde_yaml::to_value(1.5).unwrap(),
            )]))
            .unwrap(),
        ),
        (
            serde_yaml::Value::String("sweep".into()),
            serde_yaml::Value::Null,
        ),
    ]);

    let variant2 = serde_yaml::Mapping::from_iter([
        (
            serde_yaml::Value::String("name".into()),
            serde_yaml::to_value("low_infil").unwrap(),
        ),
        (
            serde_yaml::Value::String("patch".into()),
            serde_yaml::to_value(&serde_yaml::Mapping::from_iter([(
                serde_yaml::Value::String("infiltration_ach".into()),
                serde_yaml::to_value(0.5).unwrap(),
            )]))
            .unwrap(),
        ),
        (
            serde_yaml::Value::String("sweep".into()),
            serde_yaml::Value::Null,
        ),
    ]);

    let variant3 = serde_yaml::Mapping::from_iter([
        (
            serde_yaml::Value::String("name".into()),
            serde_yaml::to_value("u_sweep").unwrap(),
        ),
        (
            serde_yaml::Value::String("patch".into()),
            serde_yaml::Value::Null,
        ),
        (
            serde_yaml::Value::String("sweep".into()),
            serde_yaml::to_value(&serde_yaml::Mapping::from_iter([(
                serde_yaml::Value::String("window_u_value".into()),
                serde_yaml::to_value(vec![2.0, 3.0, 4.0]).unwrap(),
            )]))
            .unwrap(),
        ),
    ]);

    let variants = vec![variant1, variant2, variant3];

    // Build the top-level config
    let delta_config = serde_yaml::Mapping::from_iter([
        (
            serde_yaml::Value::String("base".into()),
            serde_yaml::to_value(&base_spec).unwrap(),
        ),
        (
            serde_yaml::Value::String("variants".into()),
            serde_yaml::to_value(&variants).unwrap(),
        ),
    ]);

    let yaml = serde_yaml::to_string(&delta_config).unwrap();
    std::fs::write("delta_config.yaml", &yaml).unwrap();

    // Also print to stdout so we can capture
    println!("{}", yaml);
}
