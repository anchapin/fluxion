use crate::interop::fmi::{FmiConfig, FmiExporter as CoreFmiExporter, ZoneVariables};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(untagged)]
enum ZoneInput {
    Name(String),
    Object { name: String },
}

#[napi_derive::napi]
pub struct FmiExporter;

#[napi_derive::napi]
impl FmiExporter {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self
    }

    #[napi(js_name = "exportFmu")]
    pub fn export_fmu(
        &self,
        path: String,
        zones_json: String,
        step_size_sec: f64,
    ) -> napi::bindgen_prelude::Result<()> {
        let mut config = FmiConfig::default();
        config.communication_timestep = step_size_sec;

        let zones = parse_zones(&zones_json)?;
        let exporter = CoreFmiExporter::with_config(config)
            .map_err(|e| {
                napi::bindgen_prelude::Error::from_reason(format!("FMI export failed: {e}"))
            })?
            .with_zones(zones);

        exporter.export_fmu(&PathBuf::from(path)).map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("FMI export failed: {e}"))
        })
    }
}

impl Default for FmiExporter {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_zones(zones_json: &str) -> napi::bindgen_prelude::Result<Vec<ZoneVariables>> {
    let inputs: Vec<ZoneInput> = serde_json::from_str(zones_json).map_err(|e| {
        napi::bindgen_prelude::Error::from_reason(format!("Invalid zones JSON: {e}"))
    })?;

    if inputs.is_empty() {
        return Ok(vec![ZoneVariables::default_zone()]);
    }

    inputs
        .into_iter()
        .map(|zone| match zone {
            ZoneInput::Name(name) | ZoneInput::Object { name } => {
                if name.trim().is_empty() {
                    Err(napi::bindgen_prelude::Error::from_reason(
                        "Zone names must not be empty".to_string(),
                    ))
                } else {
                    Ok(ZoneVariables::new(name))
                }
            }
        })
        .collect()
}
