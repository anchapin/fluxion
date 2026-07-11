# Design Document: EnergyPlus IDF / epJSON Import Support

> **Issue**: #778 - EnergyPlus IDF / epJSON import support  
> **Status**: Draft  
> **Created**: 2026-06-16

## 1. Problem Statement

Fluxion needs to import building energy models from EnergyPlus to enable:
- Migration of existing EnergyPlus models to Fluxion
- Validation of Fluxion against ASHRAE 140 reference tests (which use IDF format)
- User workflow support for engineers familiar with EnergyPlus

## 2. Format Analysis

### 2.1 IDF Format Structure

The Input Data File (IDF) format is a text-based format with the following characteristics:

```
Version, 25.2;

ObjectType,
  Field1,    ! Comment
  Field2,
  Field3;
```

**Key objects for Fluxion import:**

| Object | Purpose | Fluxion Mapping |
|--------|---------|-----------------|
| `Version` | EnergyPlus version | Validation |
| `SimulationControl` | Simulation settings | RunPeriod, Timestep |
| `RunPeriod` | Simulation date range | `SimulationSchema` metadata |
| `Timestep` | Simulation timestep | `SimulationSchema` metadata |
| `Building` | Building properties | `Geometry` |
| `Zone` | Zone definitions | `Geometry.zones` |
| `Material` | Material properties | `ConstructionLayer` |
| `Construction` | Layer assembly | `SurfaceConstruction` |
| `BuildingSurface:Detailed` | Surface geometry | `ZoneGeometry` |
| `Site:GroundTemperature:BuildingSurface` | Ground boundary | `GroundTemperature` |
| `GlobalGeometryRules` | Coordinate system | Geometry rules |

### 2.2 epJSON Format

epJSON is the JSON representation of IDF with the structure:

```json
{
  "Version": {
    "Version 1": {
      "version_identifier": "25.2"
    }
  },
  "Building": {
    "Building 1": {
      "name": "RefBox",
      "north_axis": 0.0,
      ...
    }
  }
}
```

### 2.3 IDF Parsing Challenges

1. **Field counting**: Objects end with `;` but fields may contain commas within quotes
2. **Comments**: Lines starting with `!` are comments; field comments after `!`
3. **Extensibility**: IDF allows extensible objects (extra fields not in IDD)
4. **Case insensitivity**: Object and field names are case-insensitive
5. **Missing fields**: Optional fields may be omitted (`, ,` syntax)
6. **Alpha vs. Numeric**: Fields can be A1 (string) or N1 (numeric) types

## 3. Architecture Design

### 3.1 Module Structure

```
src/
  io/
    idf/
      mod.rs                    # Public API
      lexer.rs                  # IDF tokenization
      parser.rs                 # IDF parsing
      epjson.rs                 # epJSON parsing  
      convert.rs                # IDF -> SimulationSchema conversion
      error.rs                  # Error types
```

### 3.2 Key Types

```rust
/// IdfObject represents a parsed IDF object
pub struct IdfObject {
    pub object_type: String,
    pub name: Option<String>,
    pub fields: Vec<IdfValue>,
}

/// IdfValue represents a single field value
pub enum IdfValue {
    String(String),
    Real(f64),
    Integer(i64),
    Empty,
}

/// Parsed IDF file
pub struct IdfFile {
    pub version: Option<String>,
    pub objects: Vec<IdfObject>,
}
```

### 3.3 Conversion Flow

```
IDF File → Lexer → Parser → IdfFile → Converter → SimulationSchema
                ↓
             epJSON → JSON Parser → IdfFile → Converter → SimulationSchema
```

## 4. Implementation Plan

### 4.1 Phase 1: Core Parsing (MVP)

**Scope**: Parse IDF files sufficient for ASHRAE 140 test cases

| Object | Fields to Parse |
|--------|-----------------|
| `Version` | version_identifier |
| `Timestep` | number_of_timesteps_per_hour |
| `RunPeriod` | name, begin_month, begin_day, end_month, end_day |
| `Building` | name, north_axis, terrain, zone_inside_convection_algorithm |
| `Zone` | name, direction_of_relative_north, x_origin, y_origin, z_origin, volume |
| `Material` | name, roughness, thickness, conductivity, density, specific_heat |
| `Construction` | name, outside_layer, layer_2, ... |
| `BuildingSurface:Detailed` | name, surface_type, construction_name, zone_name, outside_boundary_condition, sun_exposure, wind_exposure, number_of_vertices, vertices |
| `GlobalGeometryRules` | starting_vertex_position, vertex_entry_direction |
| `Site:GroundTemperature:BuildingSurface` | monthly_ground_temperature (12 values) |

**Deliverable**: `IdfParser::from_file(path) -> Result<IdfFile>`

### 4.2 Phase 2: epJSON Support

Add JSON parsing via existing `serde_json` infrastructure.

**Deliverable**: `IdfParser::from_epjson(path) -> Result<IdfFile>`

### 4.3 Phase 3: Conversion to SimulationSchema

Implement `TryFrom<IdfFile> for SimulationSchema` conversion.

**Deliverable**: `idf_file.try_to_simulation_schema() -> Result<SimulationSchema>`

### 4.4 Phase 4: CLI Integration

Add `fluxion import` subcommand:

```bash
fluxion import <input.idf> [--output <schema.json>]
```

## 5. Key Design Decisions

### 5.1 Lexer-First Parsing

IDF parsing requires proper tokenization because:
- Fields can contain commas within quoted strings: `"Hello, World!",`
- Semicolons terminate objects but may appear in strings
- Comments must be skipped

### 5.2 Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum IdfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("Conversion error: {0}")]
    Conversion(String),

    #[error("Unsupported object type: {0}")]
    UnsupportedObject(String),
}
```

### 5.3 Extensibility

For MVP, unknown object types are silently ignored. This allows:
- Forward compatibility with newer EnergyPlus versions
- Ignoring HVAC/control objects not yet supported

### 5.4 Geometry Coordinate System

IDF uses `UpperLeftCorner` or `LowerLeftCorner` starting position. Fluxion uses a standard 3D coordinate system. The conversion must:
1. Apply `GlobalGeometryRules` transformation
2. Convert vertices to Fluxion's zone geometry format
3. Calculate zone volumes and floor areas from surface geometry

## 6. Validation Strategy

### 6.1 Unit Tests

- Test lexer on edge cases (quoted commas, multi-line strings)
- Test parser on each object type
- Test converter against known IDF files

### 6.2 Integration Tests

- Parse existing test IDF files in `tests/reference_data/energyplus_models/`
- Verify `SimulationSchema` round-trip where applicable

### 6.3 Reference Validation

Run Fluxion simulation on IDF-imported ASHRAE 140 cases and verify results match original EnergyPlus output.

## 7. Dependencies

No new crate dependencies required. Uses existing:
- `serde` and `serde_json` for JSON/epJSON
- `thiserror` for error handling
- `anyhow` for ergonomic error propagation

## 8. File Inventory

| File | Purpose |
|------|---------|
| `src/io/idf/mod.rs` | Public API: `IdfParser`, `IdfFile`, `IdfError` |
| `src/io/idf/lexer.rs` | Tokenization of IDF text |
| `src/io/idf/parser.rs` | Object and field parsing |
| `src/io/idf/epjson.rs` | epJSON parsing |
| `src/io/idf/convert.rs` | IdfFile → SimulationSchema conversion |
| `src/io/idf/error.rs` | Error types |
| `tests/idf_parser_tests.rs` | Unit tests |

## 9. API Design

```rust
/// Parse an IDF file into an intermediate representation
pub struct IdfParser;

impl IdfParser {
    /// Parse IDF text
    pub fn from_str(input: &str) -> Result<IdfFile, IdfError>;

    /// Parse IDF file
    pub fn from_path(path: &Path) -> Result<IdfFile, IdfError>;

    /// Parse epJSON file
    pub fn from_epjson_path(path: &Path) -> Result<IdfFile, IdfError>;
}

impl TryFrom<IdfFile> for SimulationSchema {
    type Error = IdfError;
    fn try_from(idf: IdfFile) -> Result<Self, Self::Error>;
}
```

## 10. Out of Scope (Future Work)

Resolved in #1435 (`TryFrom<IdfFile> for SimulationSchemaV1`, design §4.3):

- ~~Full IDD (Input Data Dictionary) support for validation~~ — relaxed
  version validation (allow-list `24-2`, `25-1`, `25-2`) lives in
  `src/io/idf/convert.rs::SUPPORTED_VERSIONS`.
- ~~`TryFrom<IdfFile> for SimulationSchema` conversion (design §4.3)~~
  — landed in `src/io/idf/convert.rs`.

Still out of scope (each is a follow-up issue):

- **Schedule:* object import** — `Schedule:Compact`, `ScheduleTypeLimits`,
  `ZoneControl:Thermostat`, `ThermostatSetpoint:DualSetpoint`. The
  ASHRAE 140 acceptance test in `tests/idf_ashrae_140_acceptance.rs`
  reads these directly from the `IdfFile` (bypassing `SimulationSchemaV1`)
  via the `case_spec_from_idf` helper, but they are not yet typed
  accessors on the schema.
- **FenestrationSurface:Detailed import** — windows / doors. Same as
  schedules: read on demand by `case_spec_from_idf`, not yet first-class
  on `SimulationSchemaV1`.
- **IDF export** — only import is in the MVP; export (e.g. round-trip
  `SimulationSchemaV1 → IDF` for debugging or downstream tooling) is
  deferred.

Other follow-ups not addressed by #1435:

- Full IDD (Input Data Dictionary) support for full per-field validation.
- HVAC system objects import (`ZoneHVAC:*`, `Boiler:*`, `Chiller:*`, …).
- Zone equipment import (`ZoneHVAC:EquipmentConnections`,
  `ZoneHVAC:IdealLoadsAirSystem`).
- epJSON parsing (design §4.2).

## 11. References

- [EnergyPlus IDF Format](https://bigladdersoftware.com/epx/docs/24-1/input-output-reference/)
- [EnergyPlus epJSON Schema](https://energyplus.readthedocs.io/en/latest/schema.html)
- [eplusr R Package IDF Parsing](https://hongyuanjia.github.io/eplusr/reference/Idf.html)
