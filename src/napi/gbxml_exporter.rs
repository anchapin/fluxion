use crate::api::schema::SimulationSchemaV1;
use crate::interop::gbxml;
use std::path::PathBuf;

#[napi_derive::napi]
pub struct GbXmlExporter;

#[napi_derive::napi]
impl GbXmlExporter {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self
    }

    #[napi(js_name = "exportGbXml")]
    pub fn export_gbxml(
        &self,
        schema_json: String,
        path: String,
    ) -> napi::bindgen_prelude::Result<()> {
        let schema: SimulationSchemaV1 = serde_json::from_str(&schema_json).map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("Invalid schema JSON: {e}"))
        })?;
        gbxml::export_gbxml(&schema, PathBuf::from(path)).map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("gbXML export failed: {e}"))
        })
    }
}

impl Default for GbXmlExporter {
    fn default() -> Self {
        Self::new()
    }
}
