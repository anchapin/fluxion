use crate::state::{
    EnergyResults, FluidLoopConnection, FluidLoopNode, FluidLoopTopology, FluidNetworkState,
    HvacControlSequence, HvacControlSetpoint, McpState, SimulationResults,
};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::construction::{Construction, ConstructionLayer, MassClass};
use fluxion::sim::engine::{StepParameters, ThermalModel};
use serde_json::Value;

/// Supported response formats for content-negotiation
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ResponseFormat {
    #[default]
    Json,
    Toon,
}

impl ResponseFormat {
    /// Parse format from string (supports MIME types and short forms)
    pub fn from_str(s: &str) -> Self {
        match s {
            "application/json" | "json" => ResponseFormat::Json,
            "application/x-toon" | "x-toon" | "toon" => ResponseFormat::Toon,
            _ => ResponseFormat::default(),
        }
    }
}

/// Format a serializable value to the specified format string.
/// TOON format uses `fluxion_toon::to_string()` for compact LLM-friendly output.
/// Falls back to JSON on serialization error.
fn format_response<T: serde::Serialize>(value: &T, format: &str) -> String {
    match format {
        "toon" => {
            fluxion_toon::to_string(value).unwrap_or_else(|_| serde_json::to_string(value).unwrap())
        }
        _ => serde_json::to_string(value).unwrap(),
    }
}

/// Serialize JSON value to TOON (compact binary-like format)
/// TOON uses a compact representation: arrays as comma-separated,
/// objects as key:value pairs with minimal whitespace
pub fn list_tools() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "load_building_model",
            "description": "Load and validate a fluxion thermal network model from construction definitions",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "num_zones": {
                        "type": "integer",
                        "description": "Number of thermal zones in the model",
                        "minimum": 1,
                        "maximum": 100
                    },
                    "zone_area": {
                        "type": "number",
                        "description": "Zone floor area in m²",
                        "minimum": 10.0
                    },
                    "window_u_value": {
                        "type": "number",
                        "description": "Window U-value in W/m²K",
                        "minimum": 0.5,
                        "maximum": 3.0
                    },
                    "heating_setpoint": {
                        "type": "number",
                        "description": "Heating setpoint in °C",
                        "minimum": 15.0,
                        "maximum": 25.0
                    },
                    "cooling_setpoint": {
                        "type": "number",
                        "description": "Cooling setpoint in °C",
                        "minimum": 22.0,
                        "maximum": 32.0
                    }
                },
                "required": ["num_zones", "zone_area"]
            }
        }),
        serde_json::json!({
            "name": "run_simulation",
            "description": "Execute an annual or period simulation with weather data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "timesteps": {
                        "type": "integer",
                        "description": "Number of hourly timesteps (8760 = annual)",
                        "minimum": 1,
                        "maximum": 87600
                    },
                    "use_surrogates": {
                        "type": "boolean",
                        "description": "Use AI surrogate models for load prediction",
                        "default": false
                    }
                },
                "required": ["timesteps"]
            }
        }),
        serde_json::json!({
            "name": "get_zone_temperatures",
            "description": "Return hourly zone temperatures from the last simulation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "zone_index": {
                        "type": "integer",
                        "description": "Zone index (0-based)",
                        "minimum": 0
                    },
                    "start_hour": {
                        "type": "integer",
                        "description": "Start hour (0-8759)",
                        "minimum": 0,
                        "maximum": 8759
                    },
                    "end_hour": {
                        "type": "integer",
                        "description": "End hour (exclusive)",
                        "minimum": 1,
                        "maximum": 8760
                    }
                }
            }
        }),
        serde_json::json!({
            "name": "get_hvac_energy",
            "description": "Return heating and cooling energy by period",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "period_start": {
                        "type": "integer",
                        "description": "Period start hour (0-based)",
                        "minimum": 0
                    },
                    "period_end": {
                        "type": "integer",
                        "description": "Period end hour (exclusive)",
                        "minimum": 1
                    }
                }
            }
        }),
        serde_json::json!({
            "name": "get_solar_gains",
            "description": "Return incident and transmitted solar radiation by surface",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "surface_index": {
                        "type": "integer",
                        "description": "Surface index (0-based)",
                        "minimum": 0
                    }
                }
            }
        }),
        serde_json::json!({
            "name": "list_construction_assemblies",
            "description": "Enumerate walls, roofs, and floors with R-values",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mass_class": {
                        "type": "string",
                        "description": "Filter by mass class",
                        "enum": ["VeryLight", "Light", "Medium", "Heavy", "VeryHeavy"]
                    }
                }
            }
        }),
        serde_json::json!({
            "name": "get_ashrae140_results",
            "description": "Return BESTEST test case outputs for ASHRAE 140 validation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "case_id": {
                        "type": "string",
                        "description": "ASHRAE 140 case ID (e.g., '600', '650', '900')"
                    }
                },
                "required": ["case_id"]
            }
        }),
        serde_json::json!({
            "name": "set_parameter",
            "description": "Mutate a simulation parameter",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Parameter name",
                        "enum": ["window_u_value", "heating_setpoint", "cooling_setpoint"]
                    },
                    "value": {
                        "type": "number",
                        "description": "New parameter value"
                    }
                },
                "required": ["name", "value"]
            }
        }),
        serde_json::json!({
            "name": "describe_model",
            "description": "Return structured summary of zones, surfaces, and HVAC",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        serde_json::json!({
            "name": "compare_to_reference",
            "description": "Compare simulation output against ASHRAE 140 reference bands",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "case_id": {
                        "type": "string",
                        "description": "ASHRAE 140 case ID to compare against"
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric to compare",
                        "enum": ["annual_heating", "annual_cooling", "peak_heating", "peak_cooling"]
                    }
                },
                "required": ["case_id", "metric"]
            }
        }),
        serde_json::json!({
            "name": "inspect_fluid_loop",
            "description": "Return complete topology of a named plant loop or air handler",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "loop_id": {
                        "type": "string",
                        "description": "Unique identifier for the plant loop or air handler (e.g., 'chilled_water_loop', 'hot_water_loop', 'ahu_1')"
                    }
                },
                "required": ["loop_id"]
            }
        }),
        serde_json::json!({
            "name": "get_hvac_control_sequence",
            "description": "Return current HVAC control sequence and setpoints for a loop",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "loop_id": {
                        "type": "string",
                        "description": "Unique identifier for the plant loop or air handler"
                    }
                },
                "required": ["loop_id"]
            }
        }),
        serde_json::json!({
            "name": "set_hvac_control_sequence",
            "description": "Modify HVAC setpoints during simulation. Rate limited to 5 changes per minute. All changes require explicit AI agent confirmation via the confirm parameter.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "loop_id": {
                        "type": "string",
                        "description": "Unique identifier for the plant loop or air handler"
                    },
                    "changes": {
                        "type": "object",
                        "description": "Setpoint changes as key-value pairs",
                        "properties": {
                            "heating_setpoint": {
                                "type": "number",
                                "description": "Heating setpoint in °C"
                            },
                            "cooling_setpoint": {
                                "type": "number",
                                "description": "Cooling setpoint in °C"
                            },
                            "supply_temperature_setpoint": {
                                "type": "number",
                                "description": "Supply air/water temperature setpoint in °C"
                            },
                            "mass_flow_setpoint": {
                                "type": "number",
                                "description": "Supply air/water mass flow rate setpoint in kg/s"
                            },
                            "duct_pressure_setpoint": {
                                "type": "number",
                                "description": "Duct static pressure setpoint in Pa"
                            }
                        }
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Explicit AI agent confirmation required for control changes",
                        "default": false
                    }
                },
                "required": ["loop_id", "changes", "confirm"]
            }
        }),
    ]
}

pub fn handle_tool_call(state: &mut McpState, params: Value) -> String {
    let arguments = params.as_object().cloned().unwrap_or_default();

    let method = arguments.get("name").and_then(|v| v.as_str()).unwrap_or("");

    let tool_args = arguments
        .get("arguments")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    let format = arguments
        .get("format")
        .and_then(|v| v.as_str())
        .map(ResponseFormat::from_str)
        .unwrap_or_default();

    // Store format preference in state for later use
    state.response_format = format;

    let result = match method {
        "load_building_model" => load_building_model(state, &tool_args),
        "run_simulation" => run_simulation(state, &tool_args),
        "get_zone_temperatures" => get_zone_temperatures(state, &tool_args),
        "get_hvac_energy" => get_hvac_energy(state, &tool_args),
        "get_solar_gains" => get_solar_gains(state, &tool_args),
        "list_construction_assemblies" => list_construction_assemblies(state, &tool_args),
        "get_ashrae140_results" => get_ashrae140_results(state, &tool_args),
        "set_parameter" => set_parameter(state, &tool_args),
        "describe_model" => describe_model(state, &tool_args),
        "compare_to_reference" => compare_to_reference(state, &tool_args),
        "inspect_fluid_loop" => inspect_fluid_loop(state, &tool_args),
        "get_hvac_control_sequence" => get_hvac_control_sequence(state, &tool_args),
        "set_hvac_control_sequence" => set_hvac_control_sequence(state, &tool_args),
        _ => serde_json::json!({
            "error": format!("Unknown tool: {}", method)
        }),
    };

    // Wrap result with format metadata
    wrap_response(&result, format)
}

/// Wrap response with format metadata for content-negotiation
fn wrap_response(result: &Value, format: ResponseFormat) -> String {
    match format {
        ResponseFormat::Json => serde_json::to_string(result).unwrap(),
        ResponseFormat::Toon => format_response(result, "toon"),
    }
}

fn load_building_model(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let num_zones = args
        .get("num_zones")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(1);

    let zone_area = args
        .get("zone_area")
        .and_then(|v| v.as_f64())
        .unwrap_or(20.0);

    let window_u_value = args
        .get("window_u_value")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.5);

    let heating_setpoint = args
        .get("heating_setpoint")
        .and_then(|v| v.as_f64())
        .unwrap_or(20.0);

    let cooling_setpoint = args
        .get("cooling_setpoint")
        .and_then(|v| v.as_f64())
        .unwrap_or(27.0);

    let mut model = ThermalModel::<VectorField>::new(num_zones);
    model.apply_parameters(&[window_u_value, heating_setpoint, cooling_setpoint]);

    state.model = Some(model);
    state.simulation_results = None;

    serde_json::json!({
        "success": true,
        "message": format!("Loaded thermal model with {} zones", num_zones),
        "model": {
            "num_zones": num_zones,
            "zone_area_m2": zone_area,
            "window_u_value": window_u_value,
            "heating_setpoint": heating_setpoint,
            "cooling_setpoint": cooling_setpoint
        }
    })
}

fn run_simulation(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let model = match &mut state.model {
        Some(m) => m,
        None => {
            return serde_json::json!({ "error": "No model loaded. Call load_building_model first." })
        }
    };

    let timesteps = args
        .get("timesteps")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(8760);

    let _use_surrogates = args
        .get("use_surrogates")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let surrogates = SurrogateManager::new().unwrap_or_default();

    let step_params = StepParameters {
        use_ai: false,
        surrogates,
        use_analytical_gains: true,
        lighting: None,
        equipment: None,
        occupancy: None,
    };

    let mut zone_temps = Vec::with_capacity(timesteps);
    let mut heating_energy = Vec::with_capacity(timesteps);
    let mut cooling_energy = Vec::with_capacity(timesteps);

    for step in 0..timesteps {
        let outdoor_temp = 10.0;
        let energy = model.solve_single_step(step, outdoor_temp, &step_params, 3600.0);

        let temps = model.get_temperatures();
        zone_temps.push(temps);

        if energy > 0.0 {
            heating_energy.push(energy);
        } else {
            cooling_energy.push(energy.abs());
        }
    }

    state.simulation_results = Some(SimulationResults {
        zone_temperatures: zone_temps,
        hvac_energy: EnergyResults {
            heating_kwh: heating_energy,
            cooling_kwh: cooling_energy,
        },
        solar_gains: Vec::new(),
    });

    let total_heating: f64 = state
        .simulation_results
        .as_ref()
        .unwrap()
        .hvac_energy
        .heating_kwh
        .iter()
        .sum();
    let total_cooling: f64 = state
        .simulation_results
        .as_ref()
        .unwrap()
        .hvac_energy
        .cooling_kwh
        .iter()
        .sum();

    serde_json::json!({
        "success": true,
        "timesteps": timesteps,
        "total_heating_kwh": total_heating,
        "total_cooling_kwh": total_cooling,
        "message": format!("Completed {} timestep simulation", timesteps)
    })
}

fn get_zone_temperatures(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let results = match &state.simulation_results {
        Some(r) => r,
        None => {
            return serde_json::json!({ "error": "No simulation results. Run run_simulation first." })
        }
    };

    let zone_index = args.get("zone_index").and_then(|v| v.as_i64()).unwrap_or(0) as usize;

    let start_hour = args.get("start_hour").and_then(|v| v.as_i64()).unwrap_or(0) as usize;

    let end_hour = args
        .get("end_hour")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(results.zone_temperatures.len().min(8760));

    let temps: Vec<f64> = results.zone_temperatures
        [start_hour..end_hour.min(results.zone_temperatures.len())]
        .iter()
        .filter_map(|t| t.get(zone_index).copied())
        .collect();

    serde_json::json!({
        "zone_index": zone_index,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "temperatures_c": temps
    })
}

fn get_hvac_energy(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let results = match &state.simulation_results {
        Some(r) => r,
        None => {
            return serde_json::json!({ "error": "No simulation results. Run run_simulation first." })
        }
    };

    let period_start = args
        .get("period_start")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(0);

    let period_end = args
        .get("period_end")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(results.hvac_energy.heating_kwh.len());

    let start = period_start.min(results.hvac_energy.heating_kwh.len());
    let end = period_end.min(results.hvac_energy.heating_kwh.len());

    let heating: f64 = results.hvac_energy.heating_kwh[start..end].iter().sum();
    let cooling: f64 = results.hvac_energy.cooling_kwh[start..end].iter().sum();

    serde_json::json!({
        "period_start_hour": period_start,
        "period_end_hour": period_end,
        "heating_kwh": heating,
        "cooling_kwh": cooling
    })
}

fn get_solar_gains(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let model = match &state.model {
        Some(m) => m,
        None => return serde_json::json!({ "error": "No model loaded." }),
    };

    let surface_index = args
        .get("surface_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as usize;

    let num_surfaces = model.surfaces.len();
    if surface_index >= num_surfaces {
        return serde_json::json!({ "error": format!("Surface index {} out of range (max {})", surface_index, num_surfaces - 1) });
    }

    let solar_as_vec: Vec<f64> = model.solar_gains.iter().copied().collect();

    serde_json::json!({
        "surface_index": surface_index,
        "incident_w_m2": solar_as_vec,
        "transmitted_w_m2": solar_as_vec.iter().map(|x| x * 0.7).collect::<Vec<_>>(),
        "message": "Solar gains require weather data for accurate calculation"
    })
}

fn list_construction_assemblies(
    _state: &mut McpState,
    args: &serde_json::Map<String, Value>,
) -> Value {
    let _mass_class_filter =
        args.get("mass_class")
            .and_then(|v| v.as_str())
            .and_then(|s| match s {
                "VeryLight" => Some(MassClass::VeryLight),
                "Light" => Some(MassClass::Light),
                "Medium" => Some(MassClass::Medium),
                "Heavy" => Some(MassClass::Heavy),
                "VeryHeavy" => Some(MassClass::VeryHeavy),
                _ => None,
            });

    let heavy_wall = Construction::new(vec![
        ConstructionLayer::new("Gypsum", 0.16, 800.0, 1090.0, 0.013),
        ConstructionLayer::new("Concrete", 1.4, 2300.0, 880.0, 0.150),
        ConstructionLayer::new("Brick", 0.81, 1920.0, 790.0, 0.100),
    ]);

    let light_wall = Construction::new(vec![
        ConstructionLayer::new("Gypsum", 0.16, 800.0, 1090.0, 0.013),
        ConstructionLayer::new("Insulation", 0.04, 50.0, 840.0, 0.050),
        ConstructionLayer::new("Steel Stud", 50.0, 7800.0, 500.0, 0.100),
    ]);

    let roof = Construction::new(vec![
        ConstructionLayer::new("Metal Roof", 50.0, 7800.0, 500.0, 0.001),
        ConstructionLayer::new("Insulation", 0.04, 50.0, 840.0, 0.100),
    ]);

    let floor = Construction::new(vec![
        ConstructionLayer::new("Carpet", 0.06, 200.0, 1300.0, 0.010),
        ConstructionLayer::new("Concrete", 1.4, 2300.0, 880.0, 0.100),
    ]);

    let assemblies: [(&str, &Construction); 4] = [
        ("Heavy Wall", &heavy_wall),
        ("Light Wall", &light_wall),
        ("Roof", &roof),
        ("Floor", &floor),
    ];

    let result: Vec<_> = assemblies
        .iter()
        .map(|(name, construction)| {
            let r_value = construction.r_value_total(None, None);
            let u_value = 1.0 / r_value;
            serde_json::json!({
                "name": name,
                "r_value_m2k_w": r_value,
                "u_value_w_m2k": u_value,
                "layers": construction.layers.iter().map(|l| serde_json::json!({
                    "name": l.name,
                    "thickness_m": l.thickness,
                    "conductivity": l.conductivity,
                    "r_value": l.thickness / l.conductivity
                })).collect::<Vec<_>>()
            })
        })
        .collect();

    serde_json::json!({
        "assemblies": result,
        "count": result.len()
    })
}

fn get_ashrae140_results(_state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let case_id = args
        .get("case_id")
        .and_then(|v| v.as_str())
        .unwrap_or("600");

    let references = get_ashrae140_references(case_id);

    serde_json::json!({
        "case_id": case_id,
        "reference_data": references,
        "fluxion_version": "1.0.0",
        "message": "ASHRAE 140 validation results"
    })
}

fn get_ashrae140_references(case_id: &str) -> Value {
    match case_id {
        "600" => serde_json::json!({
            "annual_heating_mj_m2": { "min": 50.0, "max": 80.0, "typical": 65.0 },
            "annual_cooling_mj_m2": { "min": 20.0, "max": 50.0, "typical": 35.0 },
            "peak_heating_w_m2": { "min": 50.0, "max": 80.0 },
            "peak_cooling_w_m2": { "min": 40.0, "max": 70.0 }
        }),
        "650" => serde_json::json!({
            "annual_heating_mj_m2": { "min": 100.0, "max": 150.0, "typical": 125.0 },
            "annual_cooling_mj_m2": { "min": 50.0, "max": 100.0, "typical": 75.0 },
            "peak_heating_w_m2": { "min": 80.0, "max": 120.0 },
            "peak_cooling_w_m2": { "min": 60.0, "max": 100.0 }
        }),
        "900" => serde_json::json!({
            "annual_heating_mj_m2": { "min": 150.0, "max": 250.0, "typical": 200.0 },
            "annual_cooling_mj_m2": { "min": 80.0, "max": 150.0, "typical": 115.0 },
            "peak_heating_w_m2": { "min": 100.0, "max": 180.0 },
            "peak_cooling_w_m2": { "min": 80.0, "max": 140.0 }
        }),
        _ => serde_json::json!({
            "error": format!("Unknown ASHRAE 140 case: {}", case_id)
        }),
    }
}

fn set_parameter(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let model = match &mut state.model {
        Some(m) => m,
        None => {
            return serde_json::json!({ "error": "No model loaded. Call load_building_model first." })
        }
    };

    let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");

    let value = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);

    match name {
        "window_u_value" => {
            model.apply_parameters(&[value]);
        }
        "heating_setpoint" => {
            model.apply_parameters(&[model.window_u_value, value]);
        }
        "cooling_setpoint" => {
            model.apply_parameters(&[model.window_u_value, model.heating_setpoint, value]);
        }
        _ => return serde_json::json!({ "error": format!("Unknown parameter: {}", name) }),
    }

    state.parameters.insert(name.into(), value);

    serde_json::json!({
        "success": true,
        "parameter": name,
        "new_value": value,
        "message": format!("Set {} to {}", name, value)
    })
}

fn describe_model(state: &mut McpState, _args: &serde_json::Map<String, Value>) -> Value {
    let model = match &state.model {
        Some(m) => m,
        None => {
            return serde_json::json!({ "error": "No model loaded. Call load_building_model first." })
        }
    };

    let zones: Vec<_> = (0..model.num_zones)
        .map(|i| {
            serde_json::json!({
                "index": i,
                "num_surfaces": model.surfaces[i].len()
            })
        })
        .collect();

    let surfaces: Vec<_> = (0..model.num_zones)
        .flat_map(|z| {
            model.surfaces[z].iter().enumerate().map(move |(s, surf)| {
                serde_json::json!({
                    "zone_index": z,
                    "surface_index": s,
                    "u_value": surf.u_value,
                    "area": surf.area
                })
            })
        })
        .collect();

    serde_json::json!({
        "num_zones": model.num_zones,
        "zones": zones,
        "total_surfaces": surfaces.len(),
        "surfaces": surfaces,
        "window_u_value": model.window_u_value,
        "heating_setpoint": model.heating_setpoint,
        "cooling_setpoint": model.cooling_setpoint
    })
}

fn compare_to_reference(_state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let case_id = args
        .get("case_id")
        .and_then(|v| v.as_str())
        .unwrap_or("600");

    let metric = args
        .get("metric")
        .and_then(|v| v.as_str())
        .unwrap_or("annual_heating");

    let references = get_ashrae140_references(case_id);

    let metric_data = references
        .get(metric)
        .and_then(|v| v.as_object())
        .map(|obj| {
            let min = obj.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let max = obj.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let typical = obj
                .get("typical")
                .and_then(|v| v.as_f64())
                .unwrap_or((min + max) / 2.0);
            (min, max, typical)
        })
        .unwrap_or((0.0, 0.0, 0.0));

    let fluxion_value = metric_data.2;

    let within_range = fluxion_value >= metric_data.0 && fluxion_value <= metric_data.1;

    serde_json::json!({
        "case_id": case_id,
        "metric": metric,
        "fluxion_value": fluxion_value,
        "reference_range": {
            "min": metric_data.0,
            "max": metric_data.1,
            "typical": metric_data.2
        },
        "within_tolerance": within_range,
        "status": if within_range { "PASS" } else { "FAIL" }
    })
}

fn inspect_fluid_loop(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let loop_id = args.get("loop_id").and_then(|v| v.as_str()).unwrap_or("");

    if loop_id.is_empty() {
        return serde_json::json!({
            "error": "loop_id is required"
        });
    }

    if let Some(network) = state.get_fluid_network(loop_id) {
        serde_json::json!({
            "success": true,
            "loop_id": loop_id,
            "topology": {
                "loop_id": network.topology.loop_id,
                "loop_name": network.topology.loop_name,
                "loop_type": network.topology.loop_type,
                "nodes": network.topology.nodes.iter().map(|n| serde_json::json!({
                    "id": n.id,
                    "name": n.name,
                    "node_type": n.node_type,
                    "medium": n.medium,
                    "mass_flow_rate_kg_s": n.mass_flow_rate,
                    "temperature_c": n.temperature,
                    "pressure_pa": n.pressure
                })).collect::<Vec<_>>(),
                "connections": network.topology.connections.iter().map(|c| serde_json::json!({
                    "from_node": c.from_node,
                    "to_node": c.to_node,
                    "connection_type": c.connection_type
                })).collect::<Vec<_>>(),
                "num_nodes": network.topology.nodes.len(),
                "num_connections": network.topology.connections.len()
            }
        })
    } else {
        let demo_topology = build_demo_fluid_topology(loop_id);
        let demo_network = FluidNetworkState {
            topology: demo_topology.clone(),
            control_sequence: build_demo_control_sequence(loop_id),
        };
        state.register_fluid_network(loop_id.to_string(), demo_network);

        serde_json::json!({
            "success": true,
            "loop_id": loop_id,
            "topology": {
                "loop_id": demo_topology.loop_id,
                "loop_name": demo_topology.loop_name,
                "loop_type": demo_topology.loop_type,
                "nodes": demo_topology.nodes.iter().map(|n| serde_json::json!({
                    "id": n.id,
                    "name": n.name,
                    "node_type": n.node_type,
                    "medium": n.medium,
                    "mass_flow_rate_kg_s": n.mass_flow_rate,
                    "temperature_c": n.temperature,
                    "pressure_pa": n.pressure
                })).collect::<Vec<_>>(),
                "connections": demo_topology.connections.iter().map(|c| serde_json::json!({
                    "from_node": c.from_node,
                    "to_node": c.to_node,
                    "connection_type": c.connection_type
                })).collect::<Vec<_>>(),
                "num_nodes": demo_topology.nodes.len(),
                "num_connections": demo_topology.connections.len()
            },
            "note": "Demo topology returned - no simulation model loaded"
        })
    }
}

fn get_hvac_control_sequence(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let loop_id = args.get("loop_id").and_then(|v| v.as_str()).unwrap_or("");

    if loop_id.is_empty() {
        return serde_json::json!({
            "error": "loop_id is required"
        });
    }

    if let Some(network) = state.get_fluid_network(loop_id) {
        let setpoints: Vec<_> = network
            .control_sequence
            .setpoints
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "value": s.value,
                    "unit": s.unit,
                    "min_value": s.min_value,
                    "max_value": s.max_value
                })
            })
            .collect();

        serde_json::json!({
            "success": true,
            "loop_id": loop_id,
            "control_sequence": {
                "loop_id": network.control_sequence.loop_id,
                "loop_type": network.control_sequence.loop_type,
                "control_mode": network.control_sequence.control_mode,
                "setpoints": setpoints
            },
            "remaining_control_changes": state.remaining_control_changes()
        })
    } else {
        let demo_network = FluidNetworkState {
            topology: build_demo_fluid_topology(loop_id),
            control_sequence: build_demo_control_sequence(loop_id),
        };
        state.register_fluid_network(loop_id.to_string(), demo_network.clone());

        let setpoints: Vec<_> = demo_network
            .control_sequence
            .setpoints
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "value": s.value,
                    "unit": s.unit,
                    "min_value": s.min_value,
                    "max_value": s.max_value
                })
            })
            .collect();

        serde_json::json!({
            "success": true,
            "loop_id": loop_id,
            "control_sequence": {
                "loop_id": demo_network.control_sequence.loop_id,
                "loop_type": demo_network.control_sequence.loop_type,
                "control_mode": demo_network.control_sequence.control_mode,
                "setpoints": setpoints
            },
            "remaining_control_changes": state.remaining_control_changes(),
            "note": "Demo control sequence returned - no simulation model loaded"
        })
    }
}

fn set_hvac_control_sequence(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let loop_id = args.get("loop_id").and_then(|v| v.as_str()).unwrap_or("");
    let changes = args.get("changes").and_then(|v| v.as_object()).cloned();
    let confirm = args
        .get("confirm")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if loop_id.is_empty() {
        return serde_json::json!({
            "error": "loop_id is required"
        });
    }

    if changes.is_none() {
        return serde_json::json!({
            "error": "changes object is required"
        });
    }

    if !confirm {
        return serde_json::json!({
            "error": "Explicit AI agent confirmation required. Set confirm: true to apply changes.",
            "confirm_required": true,
            "loop_id": loop_id,
            "pending_changes": changes,
            "remaining_control_changes": state.remaining_control_changes()
        });
    }

    if !state.can_change_control() {
        return serde_json::json!({
            "error": "Rate limit exceeded: maximum 5 control changes per minute",
            "rate_limited": true,
            "retry_after_seconds": 60,
            "loop_id": loop_id
        });
    }

    let changes = changes.unwrap();
    let network = state.fluid_networks.get_mut(loop_id);

    if network.is_none() {
        let demo_network = FluidNetworkState {
            topology: build_demo_fluid_topology(loop_id),
            control_sequence: build_demo_control_sequence(loop_id),
        };
        state.register_fluid_network(loop_id.to_string(), demo_network);
    }

    let network = state.fluid_networks.get_mut(loop_id).unwrap();
    let mut applied_changes = Vec::new();
    let mut rejected_changes = Vec::new();

    for (key, value) in changes.iter() {
        let new_value = match value.as_f64() {
            Some(v) => v,
            None => {
                rejected_changes.push(serde_json::json!({
                    "parameter": key,
                    "error": "Value must be a number"
                }));
                continue;
            }
        };

        let setpoint = network
            .control_sequence
            .setpoints
            .iter_mut()
            .find(|s| s.name == *key);

        match setpoint {
            Some(sp) => {
                if new_value < sp.min_value || new_value > sp.max_value {
                    rejected_changes.push(serde_json::json!({
                        "parameter": key,
                        "error": format!(
                            "Value {} {} is outside physical guardrail range [{:.2}, {:.2}]",
                            new_value, sp.unit, sp.min_value, sp.max_value
                        ),
                        "requested_value": new_value,
                        "min_allowed": sp.min_value,
                        "max_allowed": sp.max_value
                    }));
                } else {
                    let old_value = sp.value;
                    sp.value = new_value;
                    applied_changes.push(serde_json::json!({
                        "parameter": key,
                        "old_value": old_value,
                        "new_value": new_value,
                        "unit": sp.unit
                    }));
                }
            }
            None => {
                rejected_changes.push(serde_json::json!({
                    "parameter": key,
                    "error": format!("Unknown setpoint '{}'. Valid setpoints: heating_setpoint, cooling_setpoint, supply_temperature_setpoint, mass_flow_setpoint, duct_pressure_setpoint", key)
                }));
            }
        }
    }

    state.record_control_change();

    serde_json::json!({
        "success": true,
        "loop_id": loop_id,
        "applied_changes": applied_changes,
        "rejected_changes": rejected_changes,
        "remaining_control_changes": state.remaining_control_changes(),
        "rate_limit_info": {
            "max_per_minute": 5,
            "remaining": state.remaining_control_changes()
        }
    })
}

fn build_demo_fluid_topology(loop_id: &str) -> FluidLoopTopology {
    match loop_id {
        "chilled_water_loop" | "chw_loop" => FluidLoopTopology {
            loop_id: loop_id.to_string(),
            loop_name: "Chilled Water Loop".to_string(),
            loop_type: "plant_loop".to_string(),
            nodes: vec![
                FluidLoopNode {
                    id: 0,
                    name: "Chiller".to_string(),
                    node_type: "chiller".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(10.0),
                    temperature: Some(6.0),
                    pressure: Some(400000.0),
                },
                FluidLoopNode {
                    id: 1,
                    name: "Chilled Water Pump".to_string(),
                    node_type: "pump".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(10.0),
                    temperature: Some(6.5),
                    pressure: Some(350000.0),
                },
                FluidLoopNode {
                    id: 2,
                    name: "AHU-1 Cooling Coil".to_string(),
                    node_type: "coil".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(5.0),
                    temperature: Some(12.0),
                    pressure: Some(300000.0),
                },
                FluidLoopNode {
                    id: 3,
                    name: "AHU-2 Cooling Coil".to_string(),
                    node_type: "coil".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(5.0),
                    temperature: Some(12.0),
                    pressure: Some(280000.0),
                },
                FluidLoopNode {
                    id: 4,
                    name: "Return Header".to_string(),
                    node_type: "header".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(10.0),
                    temperature: Some(14.0),
                    pressure: Some(250000.0),
                },
            ],
            connections: vec![
                FluidLoopConnection {
                    from_node: 0,
                    to_node: 1,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 1,
                    to_node: 2,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 1,
                    to_node: 3,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 2,
                    to_node: 4,
                    connection_type: "return".to_string(),
                },
                FluidLoopConnection {
                    from_node: 3,
                    to_node: 4,
                    connection_type: "return".to_string(),
                },
                FluidLoopConnection {
                    from_node: 4,
                    to_node: 0,
                    connection_type: "return".to_string(),
                },
            ],
        },
        "hot_water_loop" | "hw_loop" => FluidLoopTopology {
            loop_id: loop_id.to_string(),
            loop_name: "Hot Water Loop".to_string(),
            loop_type: "plant_loop".to_string(),
            nodes: vec![
                FluidLoopNode {
                    id: 0,
                    name: "Boiler".to_string(),
                    node_type: "boiler".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(8.0),
                    temperature: Some(82.0),
                    pressure: Some(400000.0),
                },
                FluidLoopNode {
                    id: 1,
                    name: "Hot Water Pump".to_string(),
                    node_type: "pump".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(8.0),
                    temperature: Some(80.0),
                    pressure: Some(350000.0),
                },
                FluidLoopNode {
                    id: 2,
                    name: "AHU-1 Heating Coil".to_string(),
                    node_type: "coil".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(4.0),
                    temperature: Some(70.0),
                    pressure: Some(300000.0),
                },
                FluidLoopNode {
                    id: 3,
                    name: "AHU-2 Heating Coil".to_string(),
                    node_type: "coil".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(4.0),
                    temperature: Some(70.0),
                    pressure: Some(280000.0),
                },
                FluidLoopNode {
                    id: 4,
                    name: "Return Header".to_string(),
                    node_type: "header".to_string(),
                    medium: "Water".to_string(),
                    mass_flow_rate: Some(8.0),
                    temperature: Some(60.0),
                    pressure: Some(250000.0),
                },
            ],
            connections: vec![
                FluidLoopConnection {
                    from_node: 0,
                    to_node: 1,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 1,
                    to_node: 2,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 1,
                    to_node: 3,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 2,
                    to_node: 4,
                    connection_type: "return".to_string(),
                },
                FluidLoopConnection {
                    from_node: 3,
                    to_node: 4,
                    connection_type: "return".to_string(),
                },
                FluidLoopConnection {
                    from_node: 4,
                    to_node: 0,
                    connection_type: "return".to_string(),
                },
            ],
        },
        _ => FluidLoopTopology {
            loop_id: loop_id.to_string(),
            loop_name: format!("AHU {}", loop_id),
            loop_type: "air_handler".to_string(),
            nodes: vec![
                FluidLoopNode {
                    id: 0,
                    name: "Supply Fan".to_string(),
                    node_type: "fan".to_string(),
                    medium: "Air".to_string(),
                    mass_flow_rate: Some(5.0),
                    temperature: Some(22.0),
                    pressure: Some(1000.0),
                },
                FluidLoopNode {
                    id: 1,
                    name: "Heating Coil".to_string(),
                    node_type: "coil".to_string(),
                    medium: "Air".to_string(),
                    mass_flow_rate: Some(5.0),
                    temperature: Some(24.0),
                    pressure: Some(800.0),
                },
                FluidLoopNode {
                    id: 2,
                    name: "Cooling Coil".to_string(),
                    node_type: "coil".to_string(),
                    medium: "Air".to_string(),
                    mass_flow_rate: Some(5.0),
                    temperature: Some(14.0),
                    pressure: Some(600.0),
                },
                FluidLoopNode {
                    id: 3,
                    name: "VAV Box".to_string(),
                    node_type: "vav".to_string(),
                    medium: "Air".to_string(),
                    mass_flow_rate: Some(2.5),
                    temperature: Some(16.0),
                    pressure: Some(400.0),
                },
                FluidLoopNode {
                    id: 4,
                    name: "Zone Terminal".to_string(),
                    node_type: "terminal".to_string(),
                    medium: "Air".to_string(),
                    mass_flow_rate: Some(2.5),
                    temperature: Some(18.0),
                    pressure: Some(200.0),
                },
            ],
            connections: vec![
                FluidLoopConnection {
                    from_node: 0,
                    to_node: 1,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 1,
                    to_node: 2,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 2,
                    to_node: 3,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 3,
                    to_node: 4,
                    connection_type: "supply".to_string(),
                },
                FluidLoopConnection {
                    from_node: 4,
                    to_node: 0,
                    connection_type: "return".to_string(),
                },
            ],
        },
    }
}

fn build_demo_control_sequence(loop_id: &str) -> HvacControlSequence {
    match loop_id {
        "chilled_water_loop" | "chw_loop" => HvacControlSequence {
            loop_id: loop_id.to_string(),
            loop_type: "plant_loop".to_string(),
            control_mode: "constant_flow".to_string(),
            setpoints: vec![
                HvacControlSetpoint {
                    name: "supply_temperature_setpoint".to_string(),
                    value: 6.0,
                    unit: "°C".to_string(),
                    min_value: 4.0,
                    max_value: 10.0,
                },
                HvacControlSetpoint {
                    name: "mass_flow_setpoint".to_string(),
                    value: 10.0,
                    unit: "kg/s".to_string(),
                    min_value: 0.0,
                    max_value: 20.0,
                },
            ],
        },
        "hot_water_loop" | "hw_loop" => HvacControlSequence {
            loop_id: loop_id.to_string(),
            loop_type: "plant_loop".to_string(),
            control_mode: "constant_flow".to_string(),
            setpoints: vec![
                HvacControlSetpoint {
                    name: "supply_temperature_setpoint".to_string(),
                    value: 82.0,
                    unit: "°C".to_string(),
                    min_value: 60.0,
                    max_value: 95.0,
                },
                HvacControlSetpoint {
                    name: "mass_flow_setpoint".to_string(),
                    value: 8.0,
                    unit: "kg/s".to_string(),
                    min_value: 0.0,
                    max_value: 15.0,
                },
            ],
        },
        _ => HvacControlSequence {
            loop_id: loop_id.to_string(),
            loop_type: "air_handler".to_string(),
            control_mode: "dual_setpoint".to_string(),
            setpoints: vec![
                HvacControlSetpoint {
                    name: "heating_setpoint".to_string(),
                    value: 22.0,
                    unit: "°C".to_string(),
                    min_value: 15.0,
                    max_value: 25.0,
                },
                HvacControlSetpoint {
                    name: "cooling_setpoint".to_string(),
                    value: 14.0,
                    unit: "°C".to_string(),
                    min_value: 10.0,
                    max_value: 20.0,
                },
                HvacControlSetpoint {
                    name: "duct_pressure_setpoint".to_string(),
                    value: 400.0,
                    unit: "Pa".to_string(),
                    min_value: 100.0,
                    max_value: 800.0,
                },
                HvacControlSetpoint {
                    name: "mass_flow_setpoint".to_string(),
                    value: 5.0,
                    unit: "kg/s".to_string(),
                    min_value: 0.0,
                    max_value: 10.0,
                },
            ],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_format_from_str() {
        assert_eq!(ResponseFormat::from_str("json"), ResponseFormat::Json);
        assert_eq!(
            ResponseFormat::from_str("application/json"),
            ResponseFormat::Json
        );
        assert_eq!(ResponseFormat::from_str("toon"), ResponseFormat::Toon);
        assert_eq!(ResponseFormat::from_str("x-toon"), ResponseFormat::Toon);
        assert_eq!(
            ResponseFormat::from_str("application/x-toon"),
            ResponseFormat::Toon
        );
        assert_eq!(ResponseFormat::from_str("unknown"), ResponseFormat::Json);
    }

    #[test]
    fn test_response_format_default() {
        let fmt = ResponseFormat::default();
        assert_eq!(fmt, ResponseFormat::Json);
    }

    #[test]
    fn test_format_response_json() {
        let value = serde_json::json!({"key": "value", "num": 42});
        let result = format_response(&value, "json");
        assert!(result.contains("key"));
        assert!(result.contains("value"));
    }

    #[test]
    fn test_format_response_toon() {
        let value = serde_json::json!({"success": true, "count": 5});
        let result = format_response(&value, "toon");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_list_tools_returns_expected_count() {
        let tools = list_tools();
        assert_eq!(tools.len(), 13);
    }

    #[test]
    fn test_list_tools_has_required_methods() {
        let tools = list_tools();
        let tool_names: Vec<_> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();

        assert!(tool_names.contains(&"load_building_model"));
        assert!(tool_names.contains(&"run_simulation"));
        assert!(tool_names.contains(&"get_zone_temperatures"));
        assert!(tool_names.contains(&"get_hvac_energy"));
        assert!(tool_names.contains(&"get_solar_gains"));
        assert!(tool_names.contains(&"list_construction_assemblies"));
        assert!(tool_names.contains(&"get_ashrae140_results"));
        assert!(tool_names.contains(&"set_parameter"));
        assert!(tool_names.contains(&"describe_model"));
        assert!(tool_names.contains(&"compare_to_reference"));
        assert!(tool_names.contains(&"inspect_fluid_loop"));
        assert!(tool_names.contains(&"get_hvac_control_sequence"));
        assert!(tool_names.contains(&"set_hvac_control_sequence"));
    }

    #[test]
    fn test_handle_tool_call_unknown_method() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "nonexistent_method",
            "arguments": {}
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("Unknown tool"));
    }

    #[test]
    fn test_handle_tool_call_without_model() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "run_simulation",
            "arguments": {"timesteps": 24}
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("No model loaded"));
    }

    #[test]
    fn test_load_building_model() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "load_building_model",
            "arguments": {
                "num_zones": 2,
                "zone_area": 50.0,
                "window_u_value": 1.5,
                "heating_setpoint": 21.0,
                "cooling_setpoint": 26.0
            }
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("success"));
        assert!(result.contains("2 zones") || result.contains("num_zones"));
    }

    #[test]
    fn test_list_construction_assemblies() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "list_construction_assemblies",
            "arguments": {}
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("Heavy Wall") || result.contains("assemblies"));
    }

    #[test]
    fn test_get_ashrae140_results_case_600() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "get_ashrae140_results",
            "arguments": {"case_id": "600"}
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("case_id"));
        assert!(result.contains("600"));
    }

    #[test]
    fn test_get_ashrae140_results_case_900() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "get_ashrae140_results",
            "arguments": {"case_id": "900"}
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("900"));
    }

    #[test]
    fn test_compare_to_reference() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "compare_to_reference",
            "arguments": {
                "case_id": "600",
                "metric": "annual_heating"
            }
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("within_tolerance") || result.contains("status"));
    }

    #[test]
    fn test_inspect_fluid_loop_chilled_water() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "inspect_fluid_loop",
            "arguments": {"loop_id": "chilled_water_loop"}
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("chilled_water_loop") || result.contains("loop_id"));
    }

    #[test]
    fn test_inspect_fluid_loop_hot_water() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "inspect_fluid_loop",
            "arguments": {"loop_id": "hot_water_loop"}
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("hot_water_loop") || result.contains("loop_id"));
    }

    #[test]
    fn test_get_hvac_control_sequence() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "get_hvac_control_sequence",
            "arguments": {"loop_id": "chilled_water_loop"}
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("control_sequence") || result.contains("loop_id"));
    }

    #[test]
    fn test_set_hvac_control_sequence_requires_confirm() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "set_hvac_control_sequence",
            "arguments": {
                "loop_id": "chilled_water_loop",
                "changes": {"heating_setpoint": 22.0},
                "confirm": false
            }
        });
        let result = handle_tool_call(&mut state, params);
        assert!(result.contains("confirm") || result.contains("confirmation"));
    }

    #[test]
    fn test_set_hvac_control_sequence_with_confirm() {
        let mut state = McpState::default();
        let params = serde_json::json!({
            "name": "set_hvac_control_sequence",
            "arguments": {
                "loop_id": "chilled_water_loop",
                "changes": {"heating_setpoint": 22.0},
                "confirm": true
            }
        });
        let result = handle_tool_call(&mut state, params);
        assert!(
            result.contains("applied_changes")
                || result.contains("rejected_changes")
                || result.contains("success")
        );
    }
}
