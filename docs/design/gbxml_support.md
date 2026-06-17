# gbXML Import/Export Support for Fluxion

## Issue
Issue #780: Enable fluxion to read and write gbXML files for BIM tool integration.

## Overview

gbXML (Green Building XML) is an industry-standard schema for exchanging building information between BIM tools (Revit, AutoCAD) and energy analysis software. Adding gbXML support allows fluxion to:

1. **Import** building geometry and construction data from BIM tools
2. **Export** fluxion simulation models to other energy analysis tools

## gbXML Schema Overview

The gbXML schema (current version 8.01) defines these key elements:

| gbXML Element | Description | Fluxion Mapping |
|--------------|-------------|-----------------|
| `Campus` | Top-level container (site) | N/A - single building |
| `Building` | Building with location | `Geometry` |
| `BuildingStorey` | Floor level | `number_of_floors` |
| `Space` | Thermal zone | `ZoneGeometry` |
| `Surface` | Wall/roof/floor with geometry | `WallSurface` |
| `Construction` | Layer assembly | `Construction` |
| `Layer` | Material layer (outside→inside) | `ConstructionLayer` |
| `Material` | Thermal properties | `ConstructionLayer` |
| `CartesianPoint` | 3D coordinates | Geometry vertices |

### Surface Types (CADBuildingSurfaceType)
- `InteriorWall`, `ExteriorWall`, `Roof`, `Floor`, `Ceiling`, `InteriorFloor`, `UndergroundWall`, `UndergroundSlab`, `SlabOnGrade`, `FreestandingColumn`, `EmbeddedColumn`

## Implementation Plan

### 1. Module Structure

Create `src/interop/gbxml/` with:

```
src/interop/gbxml/
├── mod.rs              # Module exports
├── reader.rs           # gbXML → fluxion schema
├── writer.rs           # fluxion schema → gbXML
├── types.rs            # gbXML Rust structs (de/serialization)
├── error.rs            # Error types
└── tests/
    └── test_gbxml.rs  # Round-trip tests
```

### 2. Dependencies

Add to `Cargo.toml`:

```toml
[dependencies]
quick-xml = "0.37"  # XML parsing with serde support
```

### 3. Data Mapping

#### Import (gbXML → SimulationSchema)

```
Campus.Building[0]
  └── BuildingStorey[]
        └── Space[] → ZoneGeometry[]
              ├── name → zone.name
              ├── volume → zone.volume
              ├── floorArea → zone.floor_area
              └── SpaceBoundary[]
                    └── Surface[] → WallSurface[]
                          ├── name → surface.name
                          ├── area → surface.area
                          ├── CADBuildingSurfaceType → surface.type
                          ├── constructionIdRef → Construction lookup
                          └── RectangularGeometry → surface coordinates

Construction[ID]
  └── Layer[]
        └── MaterialReference → ConstructionLayer[]
              ├── layerIdRef → Layer[ID]
              └── Material[]
                    ├── name → layer.name
                    ├── thickness → layer.thickness
                    ├── conductivity → layer.conductivity
                    ├── density → layer.density
                    └── specificHeat → layer.specific_heat
```

#### Export (SimulationSchema → gbXML)

Reverse the above mapping.

### 4. Key Traits

```rust
/// Import a building model from gbXML
pub fn import_gbxml(path: &Path) -> Result<SimulationSchema, GbXmlError>;

/// Export a building model to gbXML
pub fn export_gbxml(schema: &SimulationSchema, path: &Path) -> Result<(), GbXmlError>;

/// Parse gbXML into intermediate representation
pub fn parse_gbxml(xml: &str) -> Result<GbXmlDocument, GbXmlError>;
```

### 5. Validation Requirements

- All required gbXML elements must be present
- Surface geometry must form valid closed loops
- Material properties must be positive values
- Zone volumes must be consistent with surface areas

## Limitations (Initial Implementation)

1. **Single building only** - Campus with multiple buildings not supported
2. **Rectangular geometry only** - Complex CAD surfaces simplified
3. **No HVAC systems** - Only building envelope exported
4. **No schedules** - Default schedules assumed
5. **No shading devices** - Overhangs/fins not exported

## Testing

1. **Round-trip test**: Import sample gbXML → Export → Compare
2. **Reference file test**: Load `tests/gbxml/denver_office.xml` and verify ZoneGeometry
3. **Validation test**: Ensure exported gbXML passes gbXML validator

## Sample gbXML Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<gbXML xmlns="http://www.gbxml.org/schema" version="8.01">
  <Campus id="c1" name="Main Campus">
    <Location>
      <Name>Denver, CO</Name>
      <Latitude>39.739</Latitude>
      <Longitude>-104.984</Longitude>
    </Location>
    <Building id="b1" name="Office Building">
      <BuildingStorey id="bs1" name="Floor 1" level="0">
        <Space id="s1" name="Zone 1">
          <Area>48.0</Area>
          <Volume>129.6</Volume>
          <Surface id="surf1" surfaceType="ExteriorWall">
            <Name>West Wall</Name>
            <Area>12.0</Area>
            <ConstructionIdRef>c1</ConstructionIdRef>
            <RectangularGeometry>
              <CartesianPoint>
                <Coordinate>0.0</Coordinate>
                <Coordinate>0.0</Coordinate>
                <Coordinate>0.0</Coordinate>
              </CartesianPoint>
            </RectangularGeometry>
          </Surface>
        </Space>
      </BuildingStorey>
    </Building>
  </Campus>
  <Construction id="c1" layerCount="2">
    <Name>ExtWall</Name>
    <LayerIdRef>layer1</LayerIdRef>
    <LayerIdRef>layer2</LayerIdRef>
  </Construction>
  <Layer id="layer1">
    <MaterialIdRef>mat1</MaterialIdRef>
  </Layer>
  <Material id="mat1" name="Concrete">
    <Thickness>0.1</Thickness>
    <Conductivity>1.4</Conductivity>
    <Density>2300</Density>
    <SpecificHeat>840</SpecificHeat>
  </Material>
</gbXML>
```

## References

- [gbXML Schema Documentation](https://www.gbxml.org/schema_doc/6.01/GreenBuildingXML_Ver6.01.html)
- [gbXML Official Site](https://www.gbxml.org/)
- [gbXML GitHub (sample files)](https://github.com/GreenBuildingXML)
