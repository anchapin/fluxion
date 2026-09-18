use crate::api::schema::SimulationSchemaV1;
use crate::api::security::validate_export_path;
use crate::interop::osm;

#[napi_derive::napi]
pub struct OsmExporter;

#[napi_derive::napi]
impl OsmExporter {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self
    }

    /// Export a `SimulationSchemaV1` to an OpenStudio Model (`.osm`) file
    /// on the local filesystem.
    ///
    /// Issue #3728 — the write path is confined to the operator-configured
    /// `FLUXION_EXPORT_DIR` allow-list (default `exports/`, relative to the
    /// process working directory). The extension is pinned to `.osm`. Set
    /// `FLUXION_EXPORT_ALLOW_UNRESTRICTED=1` to opt out (extension pin
    /// still applies). See `docs/SECURITY.md` §"Exporter write-path
    /// confinement (Issue #3728)" for the full policy.
    #[napi(js_name = "exportOsm")]
    pub fn export_osm(
        &self,
        schema_json: String,
        path: String,
    ) -> napi::bindgen_prelude::Result<()> {
        let schema: SimulationSchemaV1 = serde_json::from_str(&schema_json).map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("Invalid schema JSON: {e}"))
        })?;
        let validated_path = validate_export_path(&path, "osm").map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("OSM export path rejected: {e}"))
        })?;
        osm::export_osm(&schema, validated_path).map_err(|e| {
            napi::bindgen_prelude::Error::from_reason(format!("OSM export failed: {e}"))
        })
    }
}

impl Default for OsmExporter {
    fn default() -> Self {
        Self::new()
    }
}
