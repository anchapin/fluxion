# CaseBuilder Quickstart Guide

## Introduction

Building energy modeling (BEM) with Fluxion begins with defining a **case specification** – a description of your building's geometry, construction, HVAC systems, and weather. While ASHRAE 140 provides standard test cases, you often need to create custom configurations for your own building designs.

This guide walks you through creating custom cases using the extended `CaseBuilder` API. You'll learn how to:

- Define simple rectangular zones with optional names
- Connect multiple zones with common walls
- Select construction assemblies from a reusable library
- Use custom EPW weather files
- Run simulations and interpret results

By the end, you'll be able to build a custom multi-zone house and evaluate its energy performance without needing deep expertise in ASHRAE 140 details.

## Prerequisites

- Rust toolchain installed (`rustc`, `cargo`)
- Fluxion library built (`cargo build`)
- Familiarity with basic Rust syntax and `cargo` commands

## Getting Started

### The CaseBuilder Pattern

`CaseBuilder` is a fluent builder for constructing `CaseSpec` objects. It provides sensible defaults and validates the configuration when you call `build()`.

```rust
use fluxion::validation::ashrae_140_cases::{CaseBuilder, CaseSpec};

let spec = CaseBuilder::new()
    .with_case_id("my_case".to_string())
    .with_description("My custom building".to_string())
    // ... configuration methods ...
    .build()
    .unwrap();
```

If the configuration is invalid (e.g., mismatched zone counts), `build()` returns an error.

### Step 1: Define Geometry

The simplest way to create zones is with `rectangular_zone`. This method adds a rectangular zone with the given dimensions.

- `length`: Zone length along one floor axis (meters)
- `width`: Zone width along the other floor axis (meters)
- `height`: Ceiling height (meters)
- `name`: An optional identifier; if omitted, a name like `"zone0"`, `"zone1"` is auto-generated

**Example: Single-zone house**

```rust
let spec = CaseBuilder::new()
    .rectangular_zone(8.0, 6.0, 2.7, Some("living"))
    .with_south_window(12.0)  // 12 m² of south-facing windows
    .with_hvac_setpoints(20.0, 27.0)
    .build()
    .unwrap();
```

Zone dimensions are stored in the `GeometrySpec` struct, which also provides helper methods like `floor_area()`, `volume()`, and `wall_area()`.

**Example: Two-zone house with a connecting wall**

```rust
let spec = CaseBuilder::new()
    .rectangular_zone(8.0, 6.0, 2.7, Some("main"))
    .rectangular_zone(5.0, 4.0, 2.7, Some("bedroom"))
    .add_common_wall("main", "bedroom", 15.0, 2.5)  // 15 m² wall, R-2.5
    .with_south_window(12.0)
    .with_hvac_setpoints(20.0, 27.0)
    .build()
    .unwrap();
```

The `add_common_wall` method connects two zones:

- `zone1_id`, `zone2_id`: The names of the zones (must match those given to `rectangular_zone`)
- `area`: Surface area of the common wall (m²)
- `r_value`: Thermal resistance of the wall materials (m²K/W). The method constructs a simple single-layer wall to achieve that resistance.

> **Note:** Common walls use a default interior film coefficient on both sides, which is automatically added during simulation to compute the total conductance.

### Step 2: Choose Constructions

For OOP-style convenience, you can either:

- Use `low_mass_construction()` or `high_mass_construction()` to select predefined assemblies (ASHRAE defaults)
- Use the assembly library to pick a named construction for each surface type
- Provide explicit `Construction` objects via `with_construction(wall, roof, floor)`

#### Using the Assembly Library

The assembly library loads assembly definitions from a YAML file. The default file is `config/assemblies.yaml`, but you can load any custom YAML.

```rust
use fluxion::validation::assembly_library::AssemblyLibrary;

// Load from default location (relative to project root)
let lib = AssemblyLibrary::from_file("config/assemblies.yaml").unwrap();

// Or load from a custom path
// let lib = AssemblyLibrary::from_file("my_assemblies.yaml").unwrap();

// List available assemblies
for name in lib.list() {
    println!("Available: {}", name);
}

// Retrieve a construction (e.g., for walls)
let wood_frame = lib.get("wood_frame_wall").unwrap();
let roof = lib.get("insulated_roof").unwrap();
let floor = lib.get("insulated_slab").unwrap();

// Apply to your builder
let spec = CaseBuilder::new()
    .rectangular_zone(8.0, 6.0, 2.7, None)
    .with_construction(wood_frame.clone(), roof.clone(), floor.clone())
    .with_south_window(12.0)
    .with_hvac_setpoints(20.0, 27.0)
    .build()
    .unwrap();
```

The `with_construction` method takes three arguments: wall, roof, and floor `Construction` objects. You can mix and match assemblies from the library.

#### Assembly YAML Format

Each assembly in the YAML file is defined by its layers:

```yaml
my_wall:
  layers:
    - material: "Gypsum Board"
      thickness: 0.0127    # meters
      conductivity: 0.16    # W/m·K
      density: 950.0        # kg/m³
      specific_heat: 840.0  # J/kg·K
    - material: "Insulation"
      thickness: 0.066
      conductivity: 0.04
      density: 12.0
      specific_heat: 840.0
```

Optional `emissivity` and `absorptance` may be provided; they default to 0.9 and 0.7 respectively.

### Step 3: Set Windows and Internal Loads

Windows are defined by area and orientation. Use helpers:

```rust
.with_south_window(12.0)       // South-facing, area 12 m²
.with_ew_windows(6.0)          // East + West each 6 m²
.with_zone_window(1, 8.0, Orientation::North)  // Zone index 1
```

Window properties (U-value, SHGC, etc.) can be changed via `with_window_properties(WindowSpec)`.

Internal loads (lighting, equipment, people) are set per zone:

```rust
.with_internal_loads(InternalLoads::new(total_watts, radiative_fraction, convective_fraction))
```

The fractions must sum to 1.0.

### Step 4: HVAC and Infiltration

HVAC schedules are per-zone. For constant setpoints:

```rust
.with_hvac_setpoints(heating_setpoint, cooling_setpoint)  // applies to first zone
```

For more complex schedules, use `HvacSchedule::with_setback(...)`.

Infiltration rate in air changes per hour (ACH):

```rust
.with_infiltration(0.5)  // default is 0.5 ACH
```

### Step 5: Custom EPW Weather

By default, validation uses embedded Denver TMY weather. To use your own EPW file:

```rust
.with_weather_epw("path/to/your.epw")
```

The path is stored in the resulting `CaseSpec.epw_path`. When you later run a simulation (outside of this builder, in your own code), you would load that EPW and pass the weather source to the simulation engine.

### Step 6: Building and Running

After building your `CaseSpec`, you can run a simulation using the `Model` class from Fluxion.

```rust
use fluxion::Model;

let model = Model::new(spec).expect("Failed to create model");
let annual_energy_mwh = model.simulate(1, false);  // 1 year, without AI surrogates
println!("Annual energy: {:.2} MWh", annual_energy_mwh);
```

### Full Example: Custom Two-Zone House

```rust
use fluxion::validation::ashrae_140_cases::{CaseBuilder, Orientation};
use fluxion::validation::assembly_library::AssemblyLibrary;
use fluxion::Model;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load assembly library
    let lib = AssemblyLibrary::from_file("config/assemblies.yaml")?;

    // Retrieve constructions
    let wall = lib.get("wood_frame_wall").unwrap();
    let roof = lib.get("insulated_roof").unwrap();
    let floor = lib.get("insulated_slab").unwrap();

    // Build case specification
    let spec = CaseBuilder::new()
        .with_case_id("my_house".to_string())
        .with_description("Two-zone custom house".to_string())
        .rectangular_zone(8.0, 6.0, 2.7, Some("main"))
        .rectangular_zone(5.0, 4.0, 2.7, Some("bedroom"))
        .add_common_wall("main", "bedroom", 15.0, 2.5)
        .with_construction(
            wall.clone(),
            roof.clone(),
            floor.clone(),
        )
        .with_zone_window(0, 6.0, Orientation::South)
        .with_zone_window(1, 3.0, Orientation::East)
        .with_internal_loads(fluxion::validation::ashrae_140_cases::InternalLoads::new(200.0, 0.6, 0.4))
        .with_hvac_setpoints(20.0, 27.0)
        .with_infiltration(0.5)
        .with_weather_epw("weather/denver.epw")
        .build()?;

    // Run simulation
    let model = Model::new(spec)?;
    let energy = model.simulate(1, false);
    println!("Annual energy use: {:.3} MWh", energy);

    Ok(())
}
```

## Advanced Customization

### Direct Field Access

All builder methods merely set fields on `CaseSpec`. You can set any field directly after `build()` if you need finer control. For example, to adjust the opaque absorptance:

```rust
let mut spec = CaseBuilder::new().rectangular_zone(8.0, 6.0, 2.7, None).build().unwrap();
spec.opaque_absorptance = 0.7;
```

### Multi-Zone Patterns

For more than two zones, simply call `rectangular_zone` multiple times. Connect them with `add_common_wall` as needed. The `common_walls` can reference any zones by their assigned names.

### Using Different Assembly Files

You can maintain multiple assembly YAML files for different projects:

```rust
let lib = AssemblyLibrary::from_file("projects/old_house/assemblies.yaml")?;
```

The library is lightweight; create it once and reuse.

## Tips and Units

- **Lengths**: All dimensions are in meters (m).
- **Thermal properties**:
  - R-value: m²K/W (higher = better insulation)
  - U-value: W/m²K (lower = better)
  - Conductivity: W/m·K
- **Weather**: EPW files must have 8760 hourly records. The format is standard; see EnergyPlus documentation.
- **Energy results**: `Model::simulate` returns annual energy in MWh (including both heating and cooling).

## Troubleshooting

- **"Geometry must be specified"**: You forgot to call `rectangular_zone` or `with_dimensions`.
- **"Zone 'X' not found"**: `add_common_wall` referenced a zone name that hasn't been added. Ensure you called `rectangular_zone` with matching names first.
- **Invalid EPW file**: Path is incorrect or file malformed. Verify using an EPW viewer or the `fluxion weather validate` CLI (if available).
- **Assembly not found**: Check spelling in `lib.get(...)` and ensure the YAML key matches exactly.

## Next Steps

- Explore the ASHRAE 140 cases in `src/validation/ashrae_140_cases.rs` for inspiration on advanced patterns (night ventilation, shading, free-floating).
- Try modifying the window area, orientation, or properties to see their impact on energy.
- Create your own assembly library tailored to your climate and construction practices.
- Integrate with optimization loops using `BatchOracle` for high-throughput design space exploration.

## API Reference

For detailed type definitions, see:

- `CaseSpec` and `CaseBuilder` in `src/validation/ashrae_140_cases.rs`
- `AssemblyLibrary` in `src/validation/assembly_library.rs`
- `Construction`, `ConstructionLayer` in `src/sim/construction.rs`
- `Model` in `src/lib.rs`

Happy modeling!
