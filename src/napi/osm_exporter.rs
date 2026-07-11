use crate::api::schema::SimulationSchemaV1;
use crate::interop::osm;
use std::path::PathBuf;

#[napi_derive::napi]
pub struct OsmExporter;

#[napi_derive::napi]
impl OsmExporter {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self
    }

    #[napi(js_name = "exportOsm")]
    pub fn export_osm(
        &self,
        schema_json: String,
        path: String,
    ) -> napi::bindgen_prelude::Result<()> {
        let schema: SimulationSchemaV1 = serde_json::from_str(&schema_json).map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("Invalid schema JSON: {e}"))
        })?;
        osm::export_osm(&schema, PathBuf::from(path)).map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("OSM export failed: {e}"))
        })
    }
}

impl Default for OsmExporter {
    fn default() -> Self {
        Self::new()
    }
}
