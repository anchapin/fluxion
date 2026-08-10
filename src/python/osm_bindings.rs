use crate::api::error::FluxionError;
use crate::api::schema::{SimulationSchema, SimulationSchemaV1};
use crate::interop::osm::{export_osm as export_osm_file, import_osm as import_osm_file, OsmError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

fn simulation_error(message: impl Into<String>) -> PyErr {
    FluxionError::Simulation(message.into(), None).into()
}

fn validation_error(message: impl Into<String>) -> PyErr {
    FluxionError::Validation(message.into()).into()
}

fn osm_error(error: OsmError) -> PyErr {
    simulation_error(format!("OSM interoperability error: {error}"))
}

fn schema_from_json(content: &str) -> PyResult<SimulationSchemaV1> {
    if let Ok(schema) = serde_json::from_str::<SimulationSchemaV1>(content) {
        return Ok(schema);
    }

    let schema: SimulationSchema = serde_json::from_str(content)
        .map_err(|error| validation_error(format!("Failed to parse schema JSON: {error}")))?;
    let SimulationSchema::V1(schema) = schema;
    Ok(schema)
}

fn schema_from_dict(schema: &Bound<'_, PyDict>) -> PyResult<SimulationSchemaV1> {
    let py = schema.py();
    let json = PyModule::import_bound(py, "json")?;
    let content: String = json.call_method1("dumps", (schema,))?.extract()?;
    schema_from_json(&content)
}

fn schema_to_dict(py: Python<'_>, schema: &SimulationSchemaV1) -> PyResult<Py<PyDict>> {
    let json = PyModule::import_bound(py, "json")?;
    let content = serde_json::to_string(schema)
        .map_err(|error| validation_error(format!("Failed to serialize schema: {error}")))?;
    let value = json.call_method1("loads", (content,))?;
    Ok(value.downcast_into::<PyDict>()?.unbind())
}

#[pyclass(name = "OsmReader")]
pub struct PyOsmReader {
    schema: SimulationSchemaV1,
}

#[pymethods]
impl PyOsmReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        Self::from_path(path)
    }

    #[staticmethod]
    pub fn from_path(path: &str) -> PyResult<Self> {
        let schema = import_osm_file(path).map_err(osm_error)?;
        Ok(Self { schema })
    }

    pub fn to_schema_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        schema_to_dict(py, &self.schema)
    }
}

#[pyclass(name = "OsmWriter")]
pub struct PyOsmWriter {
    schema: SimulationSchemaV1,
}

#[pymethods]
impl PyOsmWriter {
    #[new]
    pub fn new(schema: &Bound<'_, PyDict>) -> PyResult<Self> {
        Self::from_schema_dict(schema)
    }

    #[staticmethod]
    pub fn from_schema_dict(schema: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            schema: schema_from_dict(schema)?,
        })
    }

    #[staticmethod]
    pub fn from_schema_file(path: &str) -> PyResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|error| simulation_error(format!("Failed to read schema file: {error}")))?;
        Ok(Self {
            schema: schema_from_json(&content)?,
        })
    }

    pub fn export(&self, path: &str) -> PyResult<()> {
        export_osm_file(&self.schema, path).map_err(osm_error)
    }

    pub fn to_schema_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        schema_to_dict(py, &self.schema)
    }
}

#[pyfunction]
pub fn import_osm(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    let schema = import_osm_file(path).map_err(osm_error)?;
    schema_to_dict(py, &schema)
}

#[pyfunction]
pub fn export_osm(schema: &Bound<'_, PyDict>, path: &str) -> PyResult<()> {
    let schema = schema_from_dict(schema)?;
    export_osm_file(&schema, path).map_err(osm_error)
}

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    //! Rust-side inline tests for the PyO3 wrappers in this module (Issue #2532).
    //!
    //! These tests exercise the schema-parsing helper (`schema_from_json`),
    //! which is the same parser used by every public entry point in this
    //! module (`PyOsmReader::from_path`, `PyOsmWriter::from_schema_dict`,
    //! `PyOsmWriter::from_schema_file`, `import_osm`, `export_osm`), plus the
    //! three private error-mapping helpers (`simulation_error`,
    //! `validation_error`, `osm_error`). They do not touch a live Python
    //! interpreter.

    use super::*;

    // -- schema_from_json ------------------------------------------------

    #[test]
    fn schema_from_json_rejects_empty_input() {
        let err = schema_from_json("")
            .err()
            .expect("empty input should error");
        assert!(err.to_string().contains("Failed to parse"));
    }

    #[test]
    fn schema_from_json_rejects_garbage_input() {
        let err = schema_from_json("{ not json")
            .err()
            .expect("garbage should error");
        assert!(err.to_string().contains("Failed to parse"));
    }

    #[test]
    fn schema_from_json_accepts_bare_v1_schema() {
        // The fast path: a bare SimulationSchemaV1 deserializes directly.
        let json = serde_json::to_string(&SimulationSchemaV1::default()).unwrap();
        let schema = schema_from_json(&json).expect("bare V1 schema should parse");
        assert_eq!(schema, SimulationSchemaV1::default());
    }

    #[test]
    fn schema_from_json_accepts_envelope_v1_schema() {
        // The slower fallback path: a `{"V1": ...}` envelope.
        let wrapped = SimulationSchema::V1(SimulationSchemaV1::default());
        let json = serde_json::to_string(&wrapped).unwrap();
        let schema = schema_from_json(&json).expect("enveloped V1 schema should parse");
        assert_eq!(schema, SimulationSchemaV1::default());
    }

    #[test]
    fn schema_from_json_round_trip_preserves_version() {
        // Round-trip a non-default schema version through serialize → parse.
        let mut schema = SimulationSchemaV1::default();
        schema.version = crate::api::schema::SchemaVersion::V1;
        let json = serde_json::to_string(&schema).unwrap();
        let parsed = schema_from_json(&json).expect("round-trip should parse");
        assert_eq!(parsed.version, schema.version);
    }

    // -- error helpers ---------------------------------------------------
    //
    // `osm_bindings.rs` keeps three mappers: `simulation_error`, `validation_error`,
    // and `osm_error`. Each must convert cleanly into a `PyErr` and carry the
    // source message.

    #[test]
    fn simulation_error_helper_carries_message() {
        let err = simulation_error("sim failed");
        let s = err.to_string();
        assert!(s.contains("sim failed"), "s={}", s);
        assert!(s.contains("Simulation"), "s={}", s);
    }

    #[test]
    fn validation_error_helper_carries_message() {
        let err = validation_error("schema missing field");
        let s = err.to_string();
        assert!(s.contains("schema missing field"), "s={}", s);
        // The exception type is registered as `ValidationError` (capital V).
        assert!(s.to_lowercase().contains("validation"), "s={}", s);
    }

    #[test]
    fn osm_error_helper_wraps_message() {
        // Map every OsmError variant to ensure the helper doesn't panic on
        // any branch of the underlying Display impl.
        let cases = [
            OsmError::InvalidObject("bad object".to_string()),
            OsmError::parse_error(42, "boom"),
            OsmError::missing_field("Surface", "Area"),
            OsmError::UnknownObjectType("Mystery".to_string()),
            OsmError::ConversionError("units".to_string()),
            OsmError::ExportError("write fail".to_string()),
            OsmError::IoError(std::io::Error::new(std::io::ErrorKind::NotFound, "gone")),
        ];
        for case in cases {
            let err = osm_error(case);
            let s = err.to_string();
            assert!(
                s.contains("OSM interoperability error"),
                "missing prefix; s={}",
                s
            );
        }
    }

    #[test]
    fn osm_error_helper_preserves_underlying_message() {
        let src = OsmError::invalid_object("specific failure text");
        let err = osm_error(src);
        let s = err.to_string();
        assert!(s.contains("specific failure text"), "s={}", s);
    }
}
