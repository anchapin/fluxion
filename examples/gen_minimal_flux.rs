//! Generate a minimal valid `.flux` model file for CLI smoke-testing.
//!
//! Run: `cargo run --example gen_minimal_flux`
//! Writes: `examples/minimal.flux`

use fluxion::api::schema::{
    ConstructionSet, ControlSet, Geometry, ScheduleSet, SchemaVersion, SimulationOutput,
    SimulationSchema, SimulationSchemaV1, WeatherData,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = SimulationSchema::V1(SimulationSchemaV1 {
        version: SchemaVersion::V1,
        metadata: Default::default(),
        geometry: Geometry::default(),
        constructions: ConstructionSet::default(),
        schedules: ScheduleSet::default(),
        weather: WeatherData::default(),
        controls: ControlSet::default(),
        output: SimulationOutput::default(),
    });
    let json = serde_json::to_string_pretty(&schema)?;
    std::fs::write("examples/minimal.flux", &json)?;
    println!("Wrote examples/minimal.flux ({} bytes)", json.len());
    Ok(())
}
