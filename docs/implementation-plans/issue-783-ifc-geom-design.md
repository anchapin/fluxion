# Issue #783: IFC/BIM Geometry Import — Design Document

**Epic**: #783 — IFC/BIM geometry import for building envelope and zone extraction
**Date**: 2026-06-16
**Author**: Agent
**Status**: DESIGN

---

## 1. Problem Statement

Fluxion currently requires building geometry to be specified via JSON schema or programmatic construction. There is no path to ingest building geometry directly from Industry Foundation Classes (IFC) BIM files — the standard exchange format for architectural and engineering design data.

Users want to import a BIM model exported from Revit, ArchiCAD, or similar tools and automatically derive:
- Zone layout (spaces → thermal zones)
- Envelope geometry (walls, roofs, floors with areas and orientations)
- Window/door fenestration (areas, U-values, SHGC)
- Construction assemblies (material layer sets → fluxion `ConstructionLayer`)

This is a **design document only** — no implementation code is committed in this phase.

---

## 2. IFC Format Overview

### 2.1 Standard

IFC is defined by buildingSMART in **ISO 16739-1:2022** (IFC4 Addendum 2). The schema is expressed in EXPRESS (ISO 10303-11). Files are typically serialized as:
- **IFC-SPF** (`.ifc`): ISO-8859-1 text, STEP format
- **IFC-XML** (`.ifcxml`): XML representation
- **IFC-ZIP** (`.ifczip`): compressed

The most common exchange format is IFC-SPF. Fluxion's importer targets **IFC4** (the current standard) with fallback to **IFC2x3**.

### 2.2 Core Entity Hierarchy for Building Geometry

The IFC spatial structure follows this containment hierarchy:

```
IfcSite
  └─ IfcBuilding
       └─ IfcBuildingStorey (floor level)
            └─ IfcSpace (thermal zone — may also be IfcZone for grouped spaces)
                 └─ IfcBuildingElement (wall, roof, floor, window, door)
```

### 2.3 Key IFC Entities for Fluxion

| IFC Entity | Fluxion Mapping | Notes |
|---|---|---|
| `IfcSite` | — | Latitude/longitude for solar position |
| `IfcBuilding` | — | Building name, global warming settings |
| `IfcBuildingStorey` | `Geometry::floor_height`, `number_of_floors` | One storey per floor |
| `IfcSpace` | `ZoneGeometry` | Thermal zone boundary; volume, floor area |
| `IfcZone` | Groups of `ZoneGeometry` | Multiple spaces per zone |
| `IfcWall` | Surface area + orientation | Exterior walls only |
| `IfcRoof` | Roof surface area + tilt | Flat or pitched |
| `IfcFloor` (IfcSlab) | Floor area | Ground floor, intermediate, roof slab |
| `IfcWindow` | `WindowSpec` | Window area, U-value, SHGC |
| `IfcDoor` | Fenestration fraction | Less critical for energy |
| `IfcMaterialLayerSet` | `ConstructionLayer[]` | Wall/roof/floor construction |
| `IfcMaterialLayer` | Single `ConstructionLayer` | Material, thickness, thermal properties |
| `IfcRelContainedInSpatialStructure` | Zone membership | Links elements to spaces |
| `IfcRelBindsTo` | Storey grouping | Links spaces to storeys |
| `IfcRelAssociatesMaterial` | Construction assignment | Links elements to material sets |
| `IfcProductRepresentation` | Geometry | Boundary representation (IfcBoundingBox, IfcFacetedBRep) |
| `IfcSurfaceStyle` | Emissivity, absorptivity | Visual/thermal surface properties |
| `IfcWindowLiningProperties` / `IfcWindowPanelProperties` | Window U-value, SHGC | Derived or from IfcPropertySet |
| `IfcPropertySet` | Thermal transmittance, gap conductivity | Custom thermal properties |
| `IfcBuildingElementProxy` | Generic element | Used when no standard type available |

### 2.4 Geometry Representation

IFC supports multiple geometry representation types. For thermal simulation, only a subset is relevant:

| Representation Type | Use for Fluxion | Notes |
|---|---|---|
| `IfcBoundingBox` | Rough volume estimate | Fallback when B-Rep unavailable |
| `IfcFacetedBRep` | Surface area, orientation | Best for thermal calcs |
| `IfcManifoldSolidBrep` | Precise surface geometry | Full B-Rep with topology |
| `IfcMappedRepresentation` | Reused element geometry | Instance of shared type |
| `IfcExtrudedAreaSolid` | Simple wall/roof geometry | Common in BIM |

For fluxion's needs, **surface area** and **orientation** (surface normal azimuth) are the primary geometric outputs. Volume is secondary.

### 2.5 Construction Representation

Material properties in IFC follow this structure:

```
IfcMaterialLayerSet
  └─ IfcMaterialLayer (layer 1) → IfcMaterial → IfcThermalMaterialProperties
  └─ IfcMaterialLayer (layer 2) → IfcMaterial → IfcThermalMaterialProperties
  ...
```

`IfcThermalMaterialProperties` contains:
- `ThermalConductivity` [W/m·K]
- `SpecificHeatCapacity` [J/kg·K]
- `Density` [kg/m³]

These map directly to fluxion's `ConstructionLayer`: `(name, conductivity, density, specific_heat, thickness)`.

---

## 3. Geometry Extraction Requirements

### 3.1 Zone Extraction

**Goal**: Map `IfcSpace` entities to `ZoneGeometry`.

**Required outputs per zone**:
| Field | IFC Source | Notes |
|---|---|---|
| `name` | `IfcSpace.Name` | Zone identifier |
| `floor_area` | Computed from `IfcSpace.Representation` | Sum of floor polygons at zone boundary |
| `volume` | `IfcSpace.NetVolume` or computed | From IFC property or B-Rep |
| `height` | `IfcBuildingStorey.Elevation` + `IfcSpace.Height` | Zone height |

**Algorithm**:
1. Traverse `IfcBuilding` → `IfcBuildingStorey[]` → `IfcSpace[]`
2. For each `IfcSpace`, extract `NetVolume` attribute (IFC4) or compute from B-Rep
3. Floor area = zone floor polygon area from `IfcProductRepresentation`
4. Assign zone name from `IfcSpace.Name` or `LongName`

### 3.2 Building Envelope Extraction

**Goal**: Map `IfcBuildingElement` (wall, roof, floor) to surfaces with area and orientation.

**Required outputs per surface**:
| Field | IFC Source | Notes |
|---|---|---|
| Surface type | Entity type (`IfcWall`, `IfcRoof`, `IfcSlab`) | Wall / Roof / Floor classification |
| Area | `IfcProduct.Representation` | Computed from B-Rep face area |
| Azimuth | `IfcElementarySurface.Position.Orientation` | Surface normal in world coords |
| Tilt | Computed from surface normal | 0° = horizontal, 90° = vertical |
| Zone membership | `IfcRelContainedInSpatialStructure` | Which `IfcSpace` this element bounds |

**Algorithm**:
1. For each `IfcBuildingStorey`, find contained elements via `IfcRelContainedInSpatialStructure`
2. Filter to envelope elements (exclude internal partitions unless interzone)
3. Compute surface area from `IfcFacetedBRep` or `IfcExtrudedAreaSolid`
4. Extract surface normal orientation to determine azimuth/tilt

### 3.3 Window/Fenestration Extraction

**Goal**: Map `IfcWindow` and `IfcDoor` to `WindowSpec`.

**Required outputs**:
| Field | IFC Source | Notes |
|---|---|---|
| `window_area` | `IfcWindow.Area` or computed from B-Rep | Glazed opening area |
| `window_u_value` | `IfcWindowLiningProperties.Transmittance` or `IfcPropertySet` | Thermal transmittance |
| `window_shgc` | Computed from `IfcSolarSpaceDeviceType` or property | SHGC (solar heat gain coefficient) |

**Algorithm**:
1. Find `IfcWindow` entities in each space via `IfcRelContainedInSpatialStructure`
2. Extract area from `IfcWindow.Area` (if defined) or compute from opening geometry
3. Extract U-value from `IfcWindowLiningProperties.Transmittance` or `IfcPropertySet`
4. SHGC may require estimation from glazing type or be absent in IFC (common gap)

### 3.4 Construction Extraction

**Goal**: Map `IfcMaterialLayerSet` to `ConstructionLayer[]`.

**Required outputs per layer**:
| Field | IFC Source | Notes |
|---|---|---|
 Material name | `IfcMaterial.Name` | Display name |
| `conductivity` | `IfcThermalMaterialProperties.ThermalConductivity` | W/m·K |
| `density` | `IfcThermalMaterialProperties.Density` | kg/m³ |
| `specific_heat` | `IfcThermalMaterialProperties.SpecificHeatCapacity` | J/kg·K |
| `thickness` | `IfcMaterialLayer.LayerThickness` | m |

**Algorithm**:
1. For each envelope element, traverse `IfcRelAssociatesMaterial` → `IfcMaterialLayerSet`
2. Extract `IfcMaterialLayer[]` with thickness from each
3. For each layer, get `IfcMaterial` → `IfcThermalMaterialProperties`
4. Build `ConstructionLayer` struct from properties

---

## 4. Parser Selection

### 4.1 Options

| Parser | Language | Maturity | IFC Coverage | Notes |
|---|---|---|---|---|
| **ifc-spec-folder** | Python | High | IFC2x3, IFC4 | buildingSMART reference; slow |
| **ifcjs** | JavaScript | Medium | IFC4 | Web-oriented; geometry parsing complex |
| **ifcopenshell** | Python/C++ | High | IFC2x3, IFC4 | Most complete; complex install |
| **ifc-rs** | Rust | Early | IFC4 | Rust-native; incomplete geometry |
| **ifc24** | Rust | Early | IFC4 | Newer; active development |

**Recommended**: `ifcopenshell` (Python) for the **spike phase** due to:
- Most complete IFC4 support
- Mature geometry extraction (`getTopLevelBoundingBox`, `getBrepBoundingBox`)
- Large user base and extensive documentation
- Used in production by major BIM tools

**Long-term**: `ifc-rs` or `ifc24` (Rust-native) to eliminate Python dependency.

### 4.2 Python Spike Implementation

The initial implementation uses a **Python subprocess** called from Rust, following the FMI interop pattern in `src/interop/`. This allows rapid prototyping before committing to a Rust-native solution.

```
src/interop/
  ifc/                    # NEW: IFC import module
    lib.rs
    parser.rs             # Python IPC wrapper
    geometry.rs           # Geometry extraction
    construction.rs       # Material/construction mapping
    schema.rs             # → SimulationSchema conversion
  fmi/
    ...
```

The Python parser reads `.ifc` files and emits a JSON intermediate format consumed by the Rust side:

```json
{
  "zones": [
    {
      "name": "Office_1",
      "floor_area": 48.0,
      "volume": 129.6,
      "height": 2.7,
      "surfaces": [
        {
          "type": "wall",
          "name": "ExtWall_South",
          "area": 12.0,
          "azimuth": 180.0,
          "tilt": 90.0,
          "construction": "Wall_200mm_Concrete",
          "windows": [{"area": 4.0, "u_value": 1.8, "shgc": 0.35}]
        }
      ]
    }
  ],
  "materials": {
    "Wall_200mm_Concrete": [
      {"name": "Concrete", "thickness": 0.2, "conductivity": 1.7, "density": 2300, "specific_heat": 840}
    ]
  },
  "metadata": {
    "latitude": 39.739,
    "longitude": -104.984,
    "building_name": "Office Building"
  }
}
```

---

## 5. Design

### 5.1 Module Location

```
src/interop/ifc/
  mod.rs               # Module root, re-exports
  parser.rs            # IfcParser struct — subprocess launcher + JSON parsing
  geometry.rs          # GeometryExtraction trait + IfcGeometryExtractor
  construction.rs      # ConstructionExtraction trait + IfcConstructionExtractor
  schema_builder.rs    # → SimulationSchema conversion
  error.rs             # IfcImportError enum
```

### 5.2 Core Traits

Two extraction traits define the interface, matching the interop pattern used by `FMI`:

```rust
/// Extracts geometry (zones, surfaces, orientations) from an IFC file.
pub trait GeometryExtraction {
    fn extract_zones(&self, path: &Path) -> Result<Vec<ZoneGeometry>, IfcImportError>;
    fn extract_surfaces(&self, path: &Path) -> Result<Vec<SurfaceGeometry>, IfcImportError>;
    fn extract_windows(&self, path: &Path) -> Result<Vec<WindowSpec>, IfcImportError>;
}

/// Extracts material constructions from an IFC file.
pub trait ConstructionExtraction {
    fn extract_constructions(&self, path: &Path) -> Result<ConstructionSet, IfcImportError>;
    fn extract_material_layers(&self, path: &Path) -> Result<HashMap<String, Vec<ConstructionLayer>>, IfcImportError>;
}
```

### 5.3 Python Parser Interface

The Python parser is invoked as a subprocess:

```rust
pub struct IfcParser {
    python_path: PathBuf,
    script_path: PathBuf,
}

impl IfcParser {
    pub fn new() -> Self {
        let script = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .join("interop/ifc/parser.py");
        Self {
            python_path: "python3".into(),
            script_path: script,
        }
    }

    pub fn parse(&self, ifc_path: &Path) -> Result<IfcParseResult, IfcImportError> {
        let output = Command::new(&self.python_path)
            .arg(&self.script_path)
            .arg(ifc_path)
            .output()
            .map_err(|e| IfcImportError::ParserError(e.to_string()))?;

        let result: IfcParseResult = serde_json::from_slice(&output.stdout)
            .map_err(|e| IfcImportError::ParseResultError(e.to_string()))?;
        Ok(result)
    }
}
```

### 5.4 SimulationSchema Assembly

```rust
impl IfcParser {
    pub fn to_simulation_schema(&self, ifc_path: &Path) -> Result<SimulationSchema, IfcImportError> {
        let parse_result = self.parse(ifc_path)?;

        // Build ZoneGeometry from parse_result.zones
        let zones: Vec<ZoneGeometry> = parse_result.zones
            .into_iter()
            .map(|z| ZoneGeometry {
                name: z.name,
                floor_area: z.floor_area,
                volume: z.volume,
                height: z.height,
            })
            .collect();

        let geometry = Geometry {
            zones,
            total_floor_area: zones.iter().map(|z| z.floor_area).sum(),
            total_volume: zones.iter().map(|z| z.volume).sum(),
            number_of_floors: zones.len(),
            floor_height: zones.first().map(|z| z.height).unwrap_or(2.7),
        };

        // Build ConstructionSet from parse_result.materials
        let constructions = self.extract_constructions(ifc_path)?;

        Ok(SimulationSchema::v1(SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata: SchemaMetadata {
                name: parse_result.metadata.building_name,
                description: format!("Imported from IFC: {}", ifc_path.display()),
                schema_version: SchemaVersion::V1,
                ..Default::default()
            },
            geometry,
            constructions,
            schedules: ScheduleSet::default(),
            weather: WeatherData::default(),
            controls: ControlSet::default(),
            output: Default::default(),
        }))
    }
}
```

### 5.5 Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum IfcImportError {
    #[error("IFC parser not found: {0}")]
    ParserNotFound(String),

    #[error("IFC parsing failed: {0}")]
    ParserError(String),

    #[error("Failed to parse IFC JSON output: {0}")]
    ParseResultError(String),

    #[error("No zones found in IFC file")]
    NoZonesFound,

    #[error("No construction data found for surface: {0}")]
    NoConstructionFound(String),

    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),

    #[error("IFC version not supported: {0}")]
    UnsupportedIfcVersion(String),
}
```

---

## 6. Implementation Phases

### Phase 1: Spike — Python IFC Parser (1–2 days)

**Goal**: Prove the IFC → JSON → SimulationSchema path works.

- Set up `src/interop/ifc/` module structure
- Write `parser.py` using `ifcopenshell` to extract zones, surfaces, materials
- Implement `IfcParser` Rust wrapper calling the Python script
- Connect to `SimulationSchema` builder
- Test with a sample `.ifc` file from a real BIM tool

**Deliverable**: Working end-to-end import of one real IFC file.

### Phase 2: Rust-Native Parser (1 week)

**Goal**: Replace Python dependency with Rust-native IFC reading.

- Evaluate `ifc-rs` or `ifc24` for geometry coverage
- Implement `GeometryExtraction` and `ConstructionExtraction` traits in Rust
- Remove Python subprocess; `IfcParser` becomes a pure Rust struct
- Add comprehensive error messages for malformed IFC

### Phase 3: Full IFC Coverage (1 week)

- Support IFC2x3 in addition to IFC4
- Extract `IfcWindow` U-values and SHGC from `IfcPropertySet`
- Extract building site coordinates (latitude/longitude) from `IfcSite`
- Handle `IfcZone` (grouped spaces) → multiple `ZoneGeometry`
- Extract shading geometry from `IfcSite` → `IfcBuildingElementProxy` (overhangs, fins)

### Phase 4: Validation (2–3 days)

- Test with IFC files from Revit, ArchiCAD, SketchUp exports
- Verify zone floor areas match manually calculated values
- Verify construction R-values match manufacturer data sheets
- ASHRAE 140 qualification: compare IFC-derived model vs reference case inputs

---

## 7. Assumptions and Known Gaps

| Assumption | Risk | Mitigation |
|---|---|---|
| IFC files use metric units | Medium — US projects may use feet | Detect units via `IfcUnitAssignment` and convert |
| `IfcSpace.NetVolume` is defined | High — many exporters omit it | Fall back to B-Rep volume computation |
| Material thermal properties are present | Medium — many IFC files omit `IfcThermalMaterialProperties` | Map to ASHRAE 90.1 material defaults by name |
| Surface orientation is computable from B-Rep | Medium — some BIM tools export nonmanifold geometry | Fall back to `IfcBoundingBox` with default orientation |
| Window SHGC is in IFC | High — rarely stored in IFC | Estimate from window type or use 0.3 default |

---

## 8. Out of Scope (This Design)

- **IFC geometry repair** — assumes input IFC is valid geometry
- **HVAC system import** — mechanical systems in IFC (`IfcHvacSystem`) are not mapped in this phase
- **IFC to gbXML translation** — gbXML is a separate standard; no automatic conversion
- **IFC schema validation** — assumes IFC files pass `ifc-check` or equivalent
- **EnergyPlus IDF generation** — this imports INTO fluxion, not export

---

## 9. Example Usage

```rust
use fluxion::interop::ifc::IfcParser;

let parser = IfcParser::new();
let schema = parser.to_simulation_schema("/path/to/building.ifc")?;

let engine = fluxion::sim::engine::Engine::new(schema);
engine.run()?;
```

---

## 10. References

- ISO 16739-1:2022 — Industry Foundation Classes (IFC) for data sharing
- buildingSMART IFC Specification: https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2/
- ifcopenshell: https://github.com/IfcOpenShell/IfcOpenShell
- Fluxion Architecture: `docs/ARCHITECTURE.md`
- Fluxion Schema: `src/api/schema.rs`
