# Design Document: OpenStudio OSM Import/Export Support

## Issue
[#779](https://github.com/anchapin/fluxion/issues/779) - OpenStudio OSM import/export support

## Context

Fluxion is a building energy modeling (BEM) engine that needs interoperability with the broader BEM ecosystem. OpenStudio is the industry-standard GUI for EnergyPlus and uses `.osm` (OpenStudio Model) files. Supporting OSM import/export enables:

1. **Workflow Integration**: Users can create models in OpenStudio GUI and use Fluxion for high-throughput analysis
2. **Validation**: Compare Fluxion results against established OpenStudio/EnergyPlus models
3. **Migration Path**: Gradually port models from OpenStudio to Fluxion

## OSM Format Overview

OpenStudio Model (`.osm`) files are XML documents following the OpenStudio schema. Key object types:

| OSM Object | Fluxion Equivalent | Purpose |
|------------|-------------------|---------|
| `OS:Building` | `Geometry` | Building-level metadata |
| `OS:ThermalZone` | `ZoneGeometry` | Zone thermal properties |
| `OS:Space` | (implicit in zone) | Space with geometry |
| `OS:Surface` | (implicit in zone) | Walls, floors, roofs |
| `OS:SubSurface` | `WindowSpec` | Windows, doors |
| `OS:Construction` | `SurfaceConstruction` | Layered assemblies |
| `OS:Material` | `ConstructionLayer` | Material properties |
| `OS:Schedule` | `ScheduleSet` | Time-based operations |
| `OS:Site` | `WeatherData` | Location data |
| `OS:WeatherFile` | `WeatherData::EpwFile` | EPW reference |
| `OS:ThermostatSetpointDualSetpoint` | `ControlConfig` | HVAC setpoints |

## Architecture

### Module Location
```
src/interop/
├── mod.rs          # Module exports
├── fmi/            # Existing FMI support
└── osm/           # NEW: OSM import/export
    ├── mod.rs      # Module exports
    ├── reader.rs   # OSM → SimulationSchema
    ├── writer.rs   # SimulationSchema → OSM
    ├── types.rs    # OSM-specific types
    └── error.rs    # OSM errors
```

### Key Traits

```rust
/// Trait for reading OSM files into SimulationSchema
pub trait OsmReader: Send + Sync {
    fn from_path(&mut self, path: &Path) -> Result<SimulationSchema, OsmError>;
    fn from_str(&mut self, xml: &str) -> Result<SimulationSchema, OsmError>;
}

/// Trait for writing SimulationSchema to OSM files
pub trait OsmWriter: Send + Sync {
    fn write(&self, schema: &SimulationSchema, path: &Path) -> Result<(), OsmError>;
    fn to_string(&self, schema: &SimulationSchema) -> Result<String, OsmError>;
}
```

## Implementation Details

### OSM Reader

1. **Parse Strategy**: Use `quick-xml` for streaming XML parsing (memory-efficient for large models)
2. **Object Mapping**:
   - Extract building geometry → `Geometry`
   - Extract thermal zones → `Vec<ZoneGeometry>`
   - Extract constructions → `ConstructionSet`
   - Extract schedules → `ScheduleSet`
   - Extract weather reference → `WeatherData`
   - Extract thermostat → `ControlConfig`

3. **Handle Defaults**: OSM often has implicit defaults; provide sensible defaults for missing data
4. **Multi-Zone Support**: Parse all `OS:ThermalZone` objects into zone vector

### OSM Writer

1. **Serialization Strategy**: Build XML using `quick-xml` `Writer`
2. **Object Mapping**:
   - Write `Geometry` → `OS:Building` + `OS:Site`
   - Write `ZoneGeometry` → `OS:ThermalZone` + `OS:Space`
   - Write `ConstructionSet` → `OS:Construction` + `OS:Material`
   - Write `ScheduleSet` → `OS:Schedule:*`
   - Write `ControlConfig` → `OS:ThermostatSetpointDualSetpoint`

3. **Round-Trip Fidelity**: Preserve data that Fluxion doesn't use (via `OS:Extension` fields)

## Data Flow

### Import (OSM → Fluxion)
```
OSM File → OsmReader → SimulationSchema → Fluxion Engine
```

### Export (Fluxion → OSM)
```
Fluxion Engine → SimulationSchema → OsmWriter → OSM File
```

## Dependencies

Add to `Cargo.toml`:
```toml
quick-xml = "0.31"  # XML parsing/serialization
```

## API Surface

```rust
// In src/interop/mod.rs
pub mod osm;

pub use osm::{OsmConfig, OsmReader, OsmWriter};

// Simple API
impl SimulationSchema {
    pub fn from_osm(path: &Path) -> Result<Self, OsmError>;
    pub fn to_osm(&self, path: &Path) -> Result<(), OsmError>;
}
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum OsmError {
    #[error("Failed to parse OSM XML: {0}")]
    Parse(String),

    #[error("Missing required object: {0}")]
    MissingRequired(String),

    #[error("Unsupported OSM version: {0}")]
    UnsupportedVersion(String),

    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

## Limitations

1. **Partial Initial Support** (IO-02 spike):
   - Single-zone and multi-zone thermal models
   - Rectangular geometry only (Shoebox models)
   - Standard constructions and materials
   - Simple schedules (constant values)

2. **Not Supported Initially**:
   - HVAC systems (use Ideal Air Loads)
   - Complex geometries
   - Daylighting controls
   - Custom plugins/measures
   - Zone equipment

## Testing Strategy

1. **Round-Trip Tests**: Import OSM → Export OSM → Re-import → Compare schemas
2. **Reference Files**: Use ASHRAE 140 test case OSM files
3. **Golden Tests**: Compare against known-good OSM outputs

## Future Extensions

- Support complex geometries via `OS:Space`
- HVAC system import/export
- Schedule library support
- Full building geometry preservation
- OpenStudio Measure integration
