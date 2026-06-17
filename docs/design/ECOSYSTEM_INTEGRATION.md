# Epic: Ecosystem Integration and Interoperability for Fluxion

> **Issue**: [#777](https://github.com/FluxionProject/fluxion/issues/777) — Ecosystem Integration and Interoperability: BEM File Formats and Workflow Compatibility

## Context

Fluxion's adoption hinges on seamless interoperability with existing BEM workflows. Modelers invest years mastering specific tools and formats; forcing format migration is a non-starter. Instead, Fluxion must speak the languages their tools already use.

## Vision

Fluxion as the **universal translator** in the BEM ecosystem:
- Import models from any major BEM format
- Export to any target tool without data loss
- Preserve semantic fidelity (not just geometry)
- Enable hybrid workflows where Fluxion accelerates specific subsystems

---

## Ecosystem Landscape

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BEM TOOL ECOSYSTEM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │Revit     │    │SketchUp  │    │ Rhino    │    │ ArchiCAD │    │
│   │(BIM)     │    │(建模)    │    │ (BIM)    │    │ (BIM)    │    │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘    │
│        │               │               │               │           │
│        ▼               ▼               ▼               ▼           │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │                    gbXML                                │     │
│   │  (Geometry + Construction + Basic Thermal Zones)        │     │
│   └───────────────────────┬─────────────────────────────────┘     │
│                           │                                         │
│                           ▼                                         │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────┐ │
│   │ EnergyPlus│    │ OpenStudio   │    │ IESVE    │    │ Heedy   │ │
│   │ (IDF)    │    │ (OSM)       │    │ (IFC)    │    │ (Python)│ │
│   └──────────┘    └──────────────┘    └──────────┘    └─────────┘ │
│        │                 │                 │                │      │
│        └─────────────────┴────────┬────────┴────────────────┘      │
│                                  │                                  │
│                                  ▼                                  │
│                    ┌─────────────────────────┐                    │
│                    │      FLUXION            │                    │
│                    │   (ML-Accelerated BEM)  │                    │
│                    └─────────────────────────┘                    │
│                                  │                                  │
│                                  ▼                                  │
│                    ┌─────────────────────────┐                    │
│                    │   FMI Co-Simulation     │                    │
│                    │   (Functional Mockup)    │                    │
│                    └─────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Format Support Matrix

| Format | Type | Direction | Priority | Status | Issue |
|--------|------|-----------|----------|--------|-------|
| **EPW** | Weather Data | Import | P0 | ✅ Complete | - |
| **IDF** | EnergyPlus Input | Import/Export | P0 | 🔄 In Progress | #778 |
| **OSM** | OpenStudio Model | Import/Export | P0 | 🔄 In Progress | #779 |
| **gbXML** | BIM Exchange | Import/Export | P1 | 🔄 In Progress | #780 |
| **IFC** | Industry Foundation Classes | Import | P2 | 📋 Planned | #783 |
| **FMI** | Co-Simulation | Export | P1 | ✅ Partial | - |
| **JSON** | Fluxion Native | Import/Export | P0 | ✅ Complete | - |
| **CSV** | Reference Data | Import/Export | P0 | ✅ Complete | - |

---

## Format Deep Dive

### 1. EPW (Weather Data) — ✅ Complete

**Priority**: P0 (Blocking)

EnergyPlus Weather files contain:
- Location metadata (latitude, longitude, elevation)
- Design day parameters
- Hourly observations (8760 or 8784 records)

**Fluxion Integration**: `src/weather/epw.rs`

```rust
// Current capability
pub fn parse_epw(path: &Path) -> Result<WeatherData, EpwError>;
pub struct HourlyRecord {
    pub dry_bulb: f64,        // °C
    pub dew_point: f64,       // °C
    pub wind_speed: f64,      // m/s
    pub dni: f64,             // W/m²
    pub dhi: f64,             // W/m²
    pub ghi: f64,             // W/m²
}
```

**Gap Analysis**: None — EPW support is complete.

---

### 2. IDF (EnergyPlus Input Data File) — 🔄 In Progress

**Priority**: P0
**Issue**: [#778](https://github.com/FluxionProject/fluxion/issues/778)

The IDF format is the de-facto standard for energy modeling. Support enables:
- Direct migration from EnergyPlus workflows
- Validation against EnergyPlus reference tests
- Hybrid simulations where Fluxion replaces only the zone solver

**Schema Scope** (Initial Import):

| IDF Object | Fluxion Mapping | Priority |
|------------|-----------------|----------|
| `Site:Location` | Geometry.lat/lon | P0 |
| `Building` | Geometry | P0 |
| `Zone` | ZoneGeometry | P0 |
| `BuildingSurface:Detailed` | WallSurface[] | P0 |
| `Construction` | Construction | P0 |
| `Material` | ConstructionLayer | P0 |
| `Window` | WindowSurface | P1 |
| `WindowMaterial:*` | WindowMaterial | P1 |
| `Schedule:*` | Schedules | P2 |
| `People` | InternalGains | P1 |
| `Lights` | InternalGains | P1 |
| `Equipment` | HVAC Systems | P2 |
| `Output:*` | Reporting | P2 |

**Data Mapping** (IDF → Fluxion SimulationSchema):

```
IDF:text
  Site:Location,
    My Site,             → geometry.name
    39.739,              → geometry.latitude
    -104.984,            → geometry.longitude
    5,                   → geometry.timezone (Mountain)
    1609;                → geometry.elevation (ft→m)

  Building,
    My Building,         → schema.name
    30.0,                → (floor area, derived from surfaces)
    0.0,                 → (height, derived from surfaces)
    0.0,                 → (rotation)
    City,                → geometry.location_name
    Building;            → geometry.building_type

  Zone,
    Zone 1,              → zones[0].name
    0.0, 0.0, 0.0,      → zones[0].origin (x, y, z)
    1,                   → zones[0].floor
    Yes,                 → zones[0].is_conditioned
    25.0, 20.0;          → zones[0].cooling_setpoint, heating_setpoint

  BuildingSurface:Detailed,
    West Wall,           → surfaces[0].name
    Wall,                → surfaces[0].surface_type
    ExtWall,             → surfaces[0].construction_name
    Zone 1,              → surfaces[0].zone_name
    Outdoors,             → surfaces[0].boundary_condition
    ,                    → (sun exposure)
    ,                    → (wind exposure)
    4,                   → surfaces[0].num_vertices
    0.0, 0.0, 3.0,      → vertices[0]
    0.0, 0.0, 0.0,      → vertices[1]
    0.0, 5.0, 0.0,      → vertices[2]
    0.0, 5.0, 3.0;      → vertices[3]

  Construction,
    ExtWall,             → constructions[0].name
    ExtFin,              → layers[0].material_name
    Concrete,             → layers[1].material_name
    Gypsum;              → layers[2].material_name

  Material,
    Concrete,             → materials[0].name
    MediumSmooth,         → (roughness, not in fluxion)
    0.15,                → materials[0].thickness (m)
    1.4,                 → materials[0].conductivity (W/m·K)
    2400.0,              → materials[0].density (kg/m³)
    840.0,               → materials[0].specific_heat (J/kg·K)
    0.9,                 → materials[0].thermal_absorptance
    0.7,                 → materials[0].solar_absorptance
    0.7;                 → materials[0].visible_absorptance
```

**Implementation Tasks**:

1. [ ] `src/interop/idf/` module scaffold
2. [ ] IDF parser using `pest` or `logos`
3. [ ] Object-class-to-fluxion mapping
4. [ ] Surface geometry reconstruction from vertices
5. [ ] Construction assembly builder
6. [ ] Round-trip test with ASHRAE 140 reference IDF
7. [ ] Multi-zone support
8. [ ] Schedule import

**Key Dependencies**: `pest` crate for IDF grammar parsing.

---

### 3. OSM (OpenStudio Model) — 🔄 In Progress

**Priority**: P0
**Issue**: [#779](https://github.com/FluxionProject/fluxion/issues/779)

OSM is an XML-based format used by OpenStudio/SketchUp plugin. It contains:
- Full building geometry (vertices, surfaces)
- Thermal zone assignment
- HVAC systems (detailed)
- Schedules and loads
- Metering and reporting

**Current Status**: `src/interop/osm/` module exists but is **disabled** due to compilation errors.

**Module Structure** (when fixed):
```
src/interop/osm/
├── mod.rs          # ✅ Module exports (disabled)
├── reader.rs       # ⚠️ 18.7K - needs compilation fix
├── writer.rs       # ⚠️ 17.3K - needs compilation fix
├── types.rs        # ⚠️ 5.7K - needs compilation fix
└── error.rs       # ✅ 1.0K
```

**Critical Path**: Fix OSM module compilation to enable OpenStudio workflow.

**Data Mapping** (OSM → Fluxion SimulationSchema):

```
OSM:XML
  <:OS:Building>
    <:Name>Office Building</:Name>     → schema.name
    <BuildingStory任命>                           → zones[].floor
  </:OS:Building>

  <OS:ThermalZone>
    <Name>Zone 1</Name>                 → zones[0].name
    <ZoneInsideFaceareance>48.0</ZoneInsideFaceareance> → zones[0].floor_area
    <Volume>129.6</Volume>             → zones[0].volume
  </OS:ThermalZone>

  <OS:BuildingStory>
    <Name>Floor 1</Name>               → building_stories[0].name
    <NominalZcoordinate>3.0</NominalZcoordinate> → height
  </OS:BuildingStory>

  <OS:Construction>
    <Name>ExtWall</Name>               → constructions[0].name
    <OS:Layer任命>...</OS:Layer>
  </OS:Construction>
```

**OSM-Specific Challenges**:

1. **Namespace handling**: OSM uses OS: prefix, fluxion uses clean names
2. **Geometry serialization**: OSM uses `Vertex` objects, fluxion uses raw coordinates
3. **HVAC complexity**: OSM has full HVAC systems; fluxion only needs envelope + loads
4. **Reverse translation**: Converting fluxion results back to OSM for OpenStudio comparison

**Implementation Tasks**:

1. [ ] Fix compilation errors in `src/interop/osm/reader.rs`
2. [ ] Verify round-trip: OSM → fluxion → OSM
3. [ ] Handle OS: namespace stripping
4. [ ] Zone-to-ThermalZone mapping
5. [ ] Surface-to-Gen-Geometry映射
6. [ ] Construction to OS:Construction translation

---

### 4. gbXML (BIM Exchange) — 🔄 In Progress

**Priority**: P1
**Issue**: [#780](https://github.com/FluxionProject/fluxion/issues/780)

**Current Status**: Basic import/export implemented in `src/interop/gbxml/`

**gbXML Coverage**:

| Feature | Import | Export | Notes |
|---------|--------|--------|-------|
| Campus/Building | ✅ | ✅ | Single building only |
| BuildingStorey | ✅ | ✅ | Maps to floor number |
| Space (Zone) | ✅ | ✅ | Basic properties |
| Surface geometry | ✅ | ✅ | Rectangular only |
| Material properties | ✅ | ✅ | Full thermal props |
| Window geometry | ❌ | ❌ | Not implemented |
| HVAC | ❌ | ❌ | Not in scope |
| Shading | ❌ | ❌ | Not implemented |
| Schedules | ❌ | ❌ | Not implemented |

**Implementation Tasks**:

1. [ ] Enable rectangular surface export to gbXML
2. [ ] Space boundary → zone volume derivation
3. [ ] CADBuildingSurfaceType → fluxion surface type mapping
4. [ ] Window geometry (if surface has openings)
5. [ ] gbXML 7.0 vs 8.01 version handling

---

### 5. IFC (Industry Foundation Classes) — 📋 Planned

**Priority**: P2
**Issue**: [#783](https://github.com/FluxionProject/fluxion/issues/783)

IFC is the universal BIM format (ISO 16739). It is:
- Extremely expressive (1000+ entity types)
- Complex to parse correctly
- The lingua franca for BIM tools

**IFC Schema Entities for BEM**:

| IFC Entity | BEM Relevance |
|------------|---------------|
| `IfcBuilding` | Root building object |
| `IfcBuildingStorey` | Floor levels |
| `IfcSpace` | Thermal zones |
| `IfcBuildingElement` | Walls, roofs, slabs |
| `IfcMaterial` | Thermal properties |
| `IfcWindow` | Window geometry |
| `IfcThermalMaterial` | Thermal properties (extension) |
| `IfcEnergyProperties` | Energy analysis properties |

**Parsing Strategy**: Use `ifc-rs` or `ifc-spec` crate

**Fluxion Mapping**:

```
IFC:EXPRESS
  ENTITY IfcSpace
    Name            → zone.name
    Description     → zone.description
    ObjectType      → zone.zone_type
    ObjectPlacement → zone.origin (from IfcLocalPlacement)
    -- Geomtry      → zone.shape (from IfcProductDefinitionShape)
    -- ContainsElements → zone.surfaces[]

  ENTITY IfcBuildingElement
    Name            → surface.name
    PredefinedType  → surface.surface_type
    -- Geometry      → surface.vertices[]
    -- Material      → surface.construction
```

**Implementation Approach**:

1. **Phase 1**: Basic geometry extraction (walls, floors, roofs)
2. **Phase 2**: Space → zone mapping
3. **Phase 3**: Material property extraction
4. **Phase 4**: Complex geometry (curved surfaces)

**Known Challenges**:
- IFC geometry can be parametric or explicit
- `IfcOpenShell` and `IfcBooleanResults` require geometry processing
- Material assignment via `IfcRelAssociatesMaterial`

---

### 6. FMI (Functional Mock-up Interface) — ✅ Partial

**Priority**: P1

**Current Status**: Basic FMI 2.0 co-simulation export scaffold exists in `src/interop/fmi/mod.rs`

**FMI Variables** (Current):

| Variable | Direction | Unit | Description |
|----------|-----------|------|-------------|
| `outdoor_temperature` | Input | K | Outdoor dry bulb |
| `direct_normal_solar` | Input | W/m² | Solar beam |
| `diffuse_horizontal_solar` | Input | W/m² | Solar diffuse |
| `internal_gains` | Input | W | Internal loads |
| `zone_temperature` | Output | K | Zone air temp |
| `heating_load` | Output | W | Heating demand |
| `cooling_load` | Output | W | Cooling demand |

**Limitations (IO-01 Spike)**:
- Single-zone thermal network only
- Fixed 1-hour communication timestep
- No actual FMU binary generation
- No Model Exchange import

**Implementation Tasks**:

1. [ ] Generate actual FMU binary (ZIP with `modelDescription.xml` + shared library)
2. [ ] Multi-zone FMU support
3. [ ] Variable timestep support
4. [ ] FMU import (Model Exchange)
5. [ ] Master algorithm implementation

---

## Unified Interoperability Architecture

### Trait Hierarchy

```rust
// Core interop trait
pub trait BemFileFormat: Send + Sync {
    fn name(&self) -> &str;
    fn extension(&self) -> &str;
    fn import_schema(&self, path: &Path) -> Result<SimulationSchema, InteropError>;
    fn export_schema(&self, schema: &SimulationSchema, path: &Path) -> Result<(), InteropError>;
}

// Specific format readers
pub trait IdfReader { /* ... */ }
pub trait OsmReader { /* ... */ }
pub trait GbXmlReader { /* ... */ }
pub trait IfcReader { /* ... */ }

// Unified factory
pub enum FileFormat {
    IDF,
    OSM,
    gbXML,
    IFC,
    FMI,
    JSON,
    EPW,
}

impl FileFormat {
    pub fn detect_from_extension(ext: &str) -> Option<Self> { ... }
    pub fn detect_from_content(content: &str) -> Option<Self> { ... }
    pub fn reader(&self) -> Box<dyn BemFileFormat> { ... }
}
```

### Interop Module Structure

```
src/interop/
├── mod.rs              # Re-exports, FileFormat enum
├── error.rs            # InteropError union
│
├── idf/                # EnergyPlus IDF (P0)
│   ├── mod.rs
│   ├── parser.rs       # Pest-based IDF grammar
│   ├── reader.rs       # IDF → SimulationSchema
│   ├── writer.rs       # SimulationSchema → IDF
│   ├── types.rs        # IDF object types
│   └── tests/
│
├── osm/                # OpenStudio OSM (P0) — disabled, needs fix
│   ├── mod.rs          # ✅
│   ├── reader.rs       # ⚠️ compilation errors
│   ├── writer.rs       # ⚠️
│   ├── types.rs        # ⚠️
│   └── error.rs        # ✅
│
├── gbxml/              # BIM Exchange (P1) — ✅ basic impl
│   ├── mod.rs          # ✅
│   ├── reader.rs       # ✅
│   ├── writer.rs       # ✅
│   ├── types.rs        # ✅
│   ├── error.rs        # ✅
│   └── tests/
│
├── ifc/                # IFC BIM (P2) — 📋 planned
│   ├── mod.rs
│   ├── reader.rs
│   ├── geometry.rs     # IFC geometry processing
│   ├── mapper.rs       # IFC entity → fluxion mapping
│   └── tests/
│
├── fmi/                # FMI Co-Simulation (P1) — partial
│   ├── mod.rs          # ✅ scaffold
│   ├── exporter.rs     # FMU generation
│   ├── importer.rs     # FMU import (future)
│   └── schema.rs       # FMI modelDescription
│
├── json/               # Fluxion native (P0)
│   └── mod.rs          # ✅ already complete
│
└── epw/                # Weather (P0) — in weather module
    └── mod.rs          # ✅ already complete
```

---

## Workflow Integration Patterns

### Pattern 1: Direct Import

```
User: "Run fluxion on my EnergyPlus model"
┌─────────────────────────────────────────────┐
│  fluxion run model.idf --weather Denver.epw │
└─────────────────────────────────────────────┘
        │
        ▼
  IDF Reader parses
        │
        ▼
  SimulationSchema created
        │
        ▼
  Fluxion physics engine runs
        │
        ▼
  Results output
```

### Pattern 2: BIM to Simulation

```
User: "Import my Revit model"
┌─────────────────────────────────────────────┐
│  fluxion import --format gbxml model.xml    │
└─────────────────────────────────────────────┘
        │
        ▼
  gbXML Reader parses
        │
        ▼
  Geometry + Construction extracted
        │
        ▼
  SimulationSchema created
        │
        ▼
  (Optional) Editor enriches with HVAC/schedules
        │
        ▼
  Fluxion runs
```

### Pattern 3: Hybrid Co-Simulation

```
┌─────────────────────────────────────────────┐
│        EnergyPlus (HVAC) + Fluxion (Zone)  │
│                 Co-Simulation              │
└─────────────────────────────────────────────┘
        │
        ▼
  FMI Master Algorithm
        │
        ├──────────────────────┐
        ▼                      ▼
┌───────────────┐    ┌───────────────────┐
│  EnergyPlus   │◄──►│     Fluxion      │
│  FMU          │    │  FMU             │
│  (HVAC sys)   │    │  (Zone physics)  │
└───────────────┘    └───────────────────┘
        │                      │
        └──────────────────────┘
              FMI Exchange:
              - T_zone
              - Q_heating
              - Q_cooling
```

---

## Validation & Testing Strategy

### Reference Files

| Format | Source | Purpose |
|--------|--------|---------|
| `ASHRAE140_600.idf` | ASHRAE | IDF round-trip validation |
| `denver_office.osm` | OpenStudio | OSM compatibility |
| `revit_export.gbxml` | Revit | gbXML import test |
| `office.ifc` | IFC Sample files | IFC parsing test |

### Validation Criteria

1. **Round-trip fidelity**: Import → Export → Import should yield equivalent schemas
2. **Semantic preservation**: Zone volumes, surface areas, material properties must match source
3. **EnergyPlus comparison**: ASHRAE 140 test cases should produce similar results when run in Fluxion vs E+
4. **Performance**: Import should complete in < 1 second for typical building models

---

## Roadmap

### Phase 1: Foundation (Issues #778, #779)

- [ ] **#778**: IDF import working with ASHRAE 140 test cases
- [ ] **#779**: OSM import/export compilation fixed

**Deliverable**: Fluxion can read EnergyPlus and OpenStudio models

### Phase 2: BIM Integration (Issues #780, #783)

- [ ] **#780**: gbXML full support (windows, shading)
- [ ] **#783**: IFC basic geometry extraction

**Deliverable**: Fluxion can import from Revit via gbXML/IFC

### Phase 3: Co-Simulation (Issue TBD)

- [ ] FMI full implementation
- [ ] EnergyPlus co-simulation working

**Deliverable**: Hybrid simulations with EnergyPlus HVAC + Fluxion zones

---

## Open Questions

1. **IDF schedule handling**: How to map E+ schedules to fluxion internal gains?
2. **IFC geometry complexity**: Should we support nurbs and B-Rep, or only simplified geometry?
3. **OSM HVAC**: Should fluxion export HVAC systems back to OSM, or only envelope?
4. **Unit conversion**: Should we preserve original units or normalize to SI?

---

## References

- [EnergyPlus IDF Format](https://energyplus.net/sites/all/modules/custom/ietd/IEtd_docs/InputOutputReference.pdf)
- [OpenStudio SDK Documentation](https://nrel.github.io/OpenStudio-user-documentation/)
- [gbXML Schema 8.01](https://www.gbxml.org/schema_doc/8.01/GreenBuildingXML_Ver8.01.html)
- [IFC4 Documentation](https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2/HTML/)
- [FMI 2.0 Specification](https://fmi-standard.org/)
- [ASHRAE 140-2022](https://www.ashrae.org/technical-resources/standards-and-guidelines/ashrae-140)
