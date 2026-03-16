use fluxion::validation::{ASHRAE140Case, CaseSpec};
use serde_yaml;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Case600 spec
    let base_spec: CaseSpec = ASHRAE140Case::Case600.spec();

    // Serialize to YAML
    let yaml = serde_yaml::to_string(&base_spec)?;

    // Write delta config with variants
    let delta_config = format!(
        r#"base:
{}
variants:
  - name: "high_infil"
    patch:
      infiltration_ach: 1.5
  - name: "low_infil"
    patch:
      infiltration_ach: 0.5
  - name: "u_sweep"
    sweep:
      window_u_value: [2.0, 3.0, 4.0]
"#,
        yaml
    );

    let mut file = File::create("delta_config.yaml")?;
    file.write_all(delta_config.as_bytes())?;

    println!("Delta config written to delta_config.yaml");
    Ok(())
}
