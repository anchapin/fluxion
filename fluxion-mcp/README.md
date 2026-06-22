# Fluxion MCP Server

Model Context Protocol (MCP) server for Rust-native Building Energy Modeling interface.

## Overview

Fluxion-MCP provides an AI-native interface to the Fluxion BEM engine via the Model Context Protocol. It enables AI assistants (Claude, Copilot, etc.) to interact with building energy simulations through 10 carefully designed tools.

## Tools

### 1. `load_building_model`
Load and validate a fluxion thermal network model from construction definitions.

**Parameters:**
- `num_zones` (integer, required): Number of thermal zones (1-100)
- `zone_area` (number, required): Zone floor area in m² (min: 10.0)
- `window_u_value` (number): Window U-value in W/m²K (0.5-3.0)
- `heating_setpoint` (number): Heating setpoint in °C (15.0-25.0)
- `cooling_setpoint` (number): Cooling setpoint in °C (22.0-32.0)

### 2. `run_simulation`
Execute an annual or period simulation with weather data.

**Parameters:**
- `timesteps` (integer, required): Number of hourly timesteps (8760 = annual)
- `use_surrogates` (boolean): Use AI surrogate models for load prediction

### 3. `get_zone_temperatures`
Return hourly zone temperatures from the last simulation.

**Parameters:**
- `zone_index` (integer): Zone index (0-based)
- `start_hour` (integer): Start hour (0-8759)
- `end_hour` (integer): End hour (exclusive)

### 4. `get_hvac_energy`
Return heating and cooling energy by period.

**Parameters:**
- `period_start` (integer): Period start hour (0-based)
- `period_end` (integer): Period end hour (exclusive)

### 5. `get_solar_gains`
Return incident and transmitted solar radiation by surface.

**Parameters:**
- `surface_index` (integer): Surface index (0-based)

### 6. `list_construction_assemblies`
Enumerate walls, roofs, and floors with R-values.

**Parameters:**
- `mass_class` (string): Filter by mass class (VeryLight, Light, Medium, Heavy, VeryHeavy)

### 7. `get_ashrae140_results`
Return BESTEST test case outputs for ASHRAE 140 validation.

**Parameters:**
- `case_id` (string, required): ASHRAE 140 case ID (e.g., '600', '650', '900')

### 8. `set_parameter`
Mutate a simulation parameter.

**Parameters:**
- `name` (string, required): Parameter name (window_u_value, heating_setpoint, cooling_setpoint)
- `value` (number, required): New parameter value

### 9. `describe_model`
Return structured summary of zones, surfaces, and HVAC.

**Parameters:** None

### 10. `compare_to_reference`
Compare simulation output against ASHRAE 140 reference bands.

**Parameters:**
- `case_id` (string, required): ASHRAE 140 case ID to compare against
- `metric` (string, required): Metric to compare (annual_heating, annual_cooling, peak_heating, peak_cooling)

## Docker Installation

### Claude Desktop

1. Build the Docker image:
```bash
cd fluxion-mcp
docker build -t fluxion-mcp .
```

2. Add to Claude Desktop config (`~/.claude.json`):
```json
{
  "mcpServers": {
    "fluxion": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "fluxion-mcp"]
    }
  }
}
```

3. Restart Claude Desktop

### VS Code / Cursor

Add to your IDE's MCP settings:
```json
{
  "mcpServers": {
    "fluxion": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "fluxion-mcp"]
    }
  }
}
```

## Local Development

```bash
# Build
cargo build --release

# Run
./target/release/fluxion-mcp

# Test with JSON-RPC
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | ./target/release/fluxion-mcp
```

## Example Session

```json
// Initialize
{"jsonrpc":"2.0","id":1,"method":"initialize"}

// List tools
{"jsonrpc":"2.0","id":2,"method":"tools/list"}

// Load model
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"load_building_model","arguments":{"num_zones":1,"zone_area":20.0}}}

// Run simulation
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"run_simulation","arguments":{"timesteps":24}}}

// Get results
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_zone_temperatures","arguments":{"zone_index":0}}}
```

## License

Apache-2.0