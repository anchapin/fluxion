# Fluxion Simulation Schema v1.0

## Overview

The Fluxion Simulation Schema is a **versioned, unified contract** for building energy simulation data. It consolidates geometry, constructions, schedules, weather, controls, and outputs into a single schema to ensure consistency between CLI and Python pathways.

## Schema Version

Current version: **1.0**

The schema uses a tagged union approach (`SimulationSchema`) to allow forward compatibility when the schema evolves.

## Core Types

### SimulationSchema

The top-level container for all simulation data:

```rust
pub enum SimulationSchema {
    V1(SimulationSchemaV1),
}
```

### SchemaMetadata

Metadata about the simulation:

```rust
pub struct SchemaMetadata {
    pub name: String,
    pub description: String,
    pub author: Option<String>,
    pub created_at: Option<String>,
    pub schema_version: SchemaVersion,
}
```

### Geometry

Building geometry specification:

```rust
pub struct Geometry {
    pub zones: Vec<ZoneGeometry>,
    pub total_floor_area: f64,
    pub total_volume: f64,
    pub number_of_floors: usize,
    pub floor_height: f64,
}

pub struct ZoneGeometry {
    pub name: String,
    pub floor_area: f64,
    pub volume: f64,
    pub height: f64,
}
```

### ConstructionSet

Building envelope constructions:

```rust
pub struct ConstructionSet {
    pub wall: SurfaceConstruction,
    pub roof: SurfaceConstruction,
    pub floor: SurfaceConstruction,
    pub interzone: Option<SurfaceConstruction>,
}

pub struct SurfaceConstruction {
    pub name: String,
    pub layers: Vec<ConstructionLayer>,
    pub window: Option<WindowSpec>,
}

pub struct ConstructionLayer {
    pub name: String,
    pub conductivity: f64,  // W/m·K
    pub density: f64,        // kg/m³
    pub specific_heat: f64, // J/kg·K
    pub thickness: f64,      // m
    pub emissivity: f64,     // 0.0 to 1.0
    pub absorptance: f64,  // 0.0 to 1.0
}

pub struct WindowSpec {
    pub window_area: f64,
    pub window_u_value: f64,
    pub window_shgc: f64,
}
```

### ScheduleSet

Time-based schedules:

```rust
pub struct ScheduleSet {
    pub occupancy: DailySchedule,
    pub lighting: DailySchedule,
    pub hvac: HVACSchedule,
    pub infiltration: Option<DailySchedule>,
}
```

### WeatherData

Weather data specification (supports multiple sources):

```rust
pub enum WeatherData {
    EpwFile { path: PathBuf },
    TmyLocation { location: String },
    Inline { hourly_data: Vec<HourlyWeatherData> },
}
```

### ControlSet

HVAC control configurations:

```rust
pub struct ControlSet {
    pub zone_control: ControlConfig,
    pub global_control: Option<ControlConfig>,
}

pub struct ControlConfig {
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub deadband_tolerance: f64,
    pub heating_capacity: f64,
    pub cooling_capacity: f64,
}
```

### SimulationOutput

Simulation results:

```rust
pub struct SimulationOutput {
    pub eui: f64,                    // kWh/m²/year
    pub total_energy: f64,            // kWh
    pub peak_heating_load: f64,       // W
    pub peak_cooling_load: f64,       // W
    pub heating_energy: f64,           // kWh
    pub cooling_energy: f64,           // kWh
    pub zone_temperatures: Option<Vec<f64>>,
}
```

## JSON Schema Example

```json
{
  "V1": {
    "version": "V1",
    "metadata": {
      "name": "Office Building Simulation",
      "description": "Standard office building with ASHRAE 140 geometry",
      "author": "Fluxion Team",
      "created_at": "2026-04-17",
      "schema_version": "V1"
    },
    "geometry": {
      "zones": [
        {
          "name": "Main Zone",
          "floor_area": 48.0,
          "volume": 129.6,
          "height": 2.7
        }
      ],
      "total_floor_area": 48.0,
      "total_volume": 129.6,
      "number_of_floors": 1,
      "floor_height": 2.7
    },
    "constructions": {
      "wall": {
        "name": "Low Mass Wall",
        "layers": [
          {"name": "Plasterboard", "conductivity": 0.16, "density": 950.0, "specific_heat": 840.0, "thickness": 0.012, "emissivity": 0.9, "absorptance": 0.7},
          {"name": "Fiberglass", "conductivity": 0.04, "density": 12.0, "specific_heat": 840.0, "thickness": 0.066, "emissivity": 0.9, "absorptance": 0.7},
          {"name": "Wood siding", "conductivity": 0.14, "density": 500.0, "specific_heat": 1300.0, "thickness": 0.009, "emissivity": 0.9, "absorptance": 0.7}
        ],
        "window": {
          "window_area": 12.0,
          "window_u_value": 1.5,
          "window_shgc": 0.3
        }
      },
      "roof": { ... },
      "floor": { ... },
      "interzone": null
    },
    "schedules": {
      "occupancy": { ... },
      "lighting": { ... },
      "hvac": {
        "heating": { "schedule_type": "Weekly", "values": { ... } },
        "cooling": { "schedule_type": "Weekly", "values": { ... } }
      },
      "infiltration": null
    },
    "weather": {
      "type": "tmy",
      "location": "Denver, CO"
    },
    "controls": {
      "zone_control": {
        "heating_setpoint": 20.0,
        "cooling_setpoint": 24.0,
        "deadband_tolerance": 0.5,
        "heating_capacity": 100000.0,
        "cooling_capacity": 100000.0
      },
      "global_control": null
    },
    "output": {
      "eui": 150.5,
      "total_energy": 7224.0,
      "peak_heating_load": 5000.0,
      "peak_cooling_load": 4500.0,
      "heating_energy": 3500.0,
      "cooling_energy": 3724.0,
      "zone_temperatures": null
    }
  }
}
```

## CLI Usage

The CLI uses this schema for configuration:

```bash
# Run simulation with schema-based config
fluxion multi-zone simulate --config building_schema.json

# Validate a schema file
fluxion multi-zone validate --schema building_schema.json
```

## Python API Usage

```python
from fluxion import SimulationSchema, SchemaMetadata, Geometry, ConstructionSet

# Create schema
schema = SimulationSchema.V1(
    metadata=SchemaMetadata(
        name="My Building",
        description="Office building simulation"
    ),
    geometry=Geometry.default(),
    constructions=ConstructionSet.default(),
    ...
)

# Serialize to JSON
json_str = schema.to_json()

# Load from file
schema = SimulationSchema.from_json(open("building.json").read())
```

## Schema Evolution

When the schema needs to change:

1. Add a new variant to `SchemaVersion` (e.g., `V2`)
2. Create a new struct (e.g., `SimulationSchemaV2`)
3. Add the new variant to `SimulationSchema` enum
4. Implement migration functions if needed

This approach ensures:
- **Backward compatibility**: Old schemas continue to work
- **Forward compatibility**: New schemas can be read by old code (with unknown fields ignored)
- **Clear versioning**: Each schema explicitly declares its version
