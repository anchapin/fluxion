# SKILL.wave-5.md — Interactive Simulation Controls

## Summary

Add a "Controls" tab to the fluxion-tauri frontend with sliders for simulation parameter adjustment.

## Changes

- **commands.rs**: Added `SimulationParameters` struct and `update_simulation_parameters` / `get_simulation_parameters` commands
- **main.rs**: Registered new commands with Tauri handler
- **index.html**: Added Controls tab with sliders for:
  - Heating setpoint (°C): 10-25
  - Cooling setpoint (°C): 20-35
  - Lighting load (W/m²): 0-30
  - Equipment load (W/m²): 0-30
  - Occupancy (pers/m²): 0-1
  - Ventilation rate (ach): 0-5
  - Wall U-value (W/m²K): 0.1-2.0
  - Roof U-value (W/m²K): 0.1-2.0

## Testing

- `cargo check -p fluxion-tauri` passes

## Resolves

- Closes #3177
