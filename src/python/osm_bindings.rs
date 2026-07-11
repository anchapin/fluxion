use crate::api::error::FluxionError;
use crate::api::schema::{SimulationSchema, SimulationSchemaV1};
use crate::interop::osm::{export_osm as export_osm_file, import_osm as import_osm_file, OsmError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

fn simulation_error(message: impl Into<String>) -> PyErr {
    FluxionError::Simulation(message.into()).into()
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
