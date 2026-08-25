# REFERENCE.wave-5.md — Interactive Simulation Controls

## Files Modified

### fluxion-tauri/src-tauri/src/commands.rs
- Added `SimulationParameters` struct with fields: `heating_setpoint`, `cooling_setpoint`, `lighting_load`, `equipment_load`, `occupancy`, `ventilation_rate`, `wall_u_value`, `roof_u_value`
- Added `update_simulation_parameters(params: SimulationParameters)` command
- Added `get_simulation_parameters()` command

### fluxion-tauri/src-tauri/src/main.rs
- Registered `commands::update_simulation_parameters` and `commands::get_simulation_parameters` in invoke handler

### fluxion-tauri/src-tauri/src/index.html
- Added "Controls" tab to tabs container
- Added `#controls-panel` div for slider content
- Added CSS styles for `.control-group`, `.slider-item`, `.slider-header`, `.slider-label`, `.slider-value`, `.slider-unit`, range input styling, `.apply-btn`, `#param-status`
- Added `renderControlsPanel()` function to render sliders and wire up Tauri invoke
- Updated tab click handler to show controls panel and call `renderControlsPanel()`
- Sliders wired to `invoke('update_simulation_parameters', { params: currentParams })`

## API

```rust
#[tauri::command]
pub async fn update_simulation_parameters(params: SimulationParameters) -> Result<SimulationParameters, String>

#[tauri::command]
pub async fn get_simulation_parameters() -> Result<SimulationParameters, String>
```

## Frontend Usage

```javascript
const { invoke } = await import('@tauri-apps/api/core');
await invoke('update_simulation_parameters', { params: currentParams });
```
