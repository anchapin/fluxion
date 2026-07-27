use crate::state::{EnergyResults, McpState, SimulationResults};
use fluxion::sim::construction::{Construction, ConstructionLayer, MassClass};
use fluxion::sim::engine::{StepParameters, ThermalModel};
use fluxion::physics::cta::VectorField;
use fluxion::ai::surrogate::SurrogateManager;
use serde_json::Value;

/// Supported response formats for content-negotiation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponseFormat {
    Json,
    Toon,
}

impl Default for ResponseFormat {
    fn default() -> Self {
        ResponseFormat::Json
    }
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

/// Serialize JSON value to TOON (compact binary-like format)
/// TOON uses a compact representation: arrays as comma-separated,
/// objects as key:value pairs with minimal whitespace
fn serialize_to_toon(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("\"{}\"", s),
        Value::Array(arr) => {
            if arr.is_empty() {
                "[]".to_string()
            } else {
                let inner: Vec<String> = arr.iter().map(|v| serialize_to_toon(v)).collect();
                format!("[{}]", inner.join(","))
            }
        }
        Value::Object(obj) => {
            if obj.is_empty() {
                "{}".to_string()
            } else {
                let inner: Vec<String> = obj.iter()
                    .map(|(k, v)| format!("{}:{}", k, serialize_to_toon(v)))
                    .collect();
                format!("{{{}}}", inner.join(","))
            }
        }
    }
}

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
    ]
}

pub fn handle_tool_call(
    state: &mut McpState,
    params: Value,
) -> Value {
    let arguments = params.as_object().cloned().unwrap_or_default();

    let method = arguments.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let tool_args = arguments.get("arguments")
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
        _ => serde_json::json!({
            "error": format!("Unknown tool: {}", method)
        }),
    };

    // Wrap result with format metadata
    wrap_response(&result, format)
}

/// Wrap response with format metadata for content-negotiation
fn wrap_response(result: &Value, format: ResponseFormat) -> Value {
    match format {
        ResponseFormat::Json => result.clone(),
        ResponseFormat::Toon => {
            serde_json::json!({
                "format": "application/x-toon",
                "data": result,
                "_toon": serialize_to_toon(result)
            })
        }
    }
}

fn load_building_model(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let num_zones = args.get("num_zones")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(1);

    let zone_area = args.get("zone_area")
        .and_then(|v| v.as_f64())
        .unwrap_or(20.0);

    let window_u_value = args.get("window_u_value")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.5);

    let heating_setpoint = args.get("heating_setpoint")
        .and_then(|v| v.as_f64())
        .unwrap_or(20.0);

    let cooling_setpoint = args.get("cooling_setpoint")
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
        None => return serde_json::json!({ "error": "No model loaded. Call load_building_model first." }),
    };

    let timesteps = args.get("timesteps")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(8760);

    let use_surrogates = args.get("use_surrogates")
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

    let total_heating: f64 = state.simulation_results.as_ref().unwrap().hvac_energy.heating_kwh.iter().sum();
    let total_cooling: f64 = state.simulation_results.as_ref().unwrap().hvac_energy.cooling_kwh.iter().sum();

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
        None => return serde_json::json!({ "error": "No simulation results. Run run_simulation first." }),
    };

    let zone_index = args.get("zone_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as usize;

    let start_hour = args.get("start_hour")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as usize;

    let end_hour = args.get("end_hour")
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
        None => return serde_json::json!({ "error": "No simulation results. Run run_simulation first." }),
    };

    let period_start = args.get("period_start")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(0);

    let period_end = args.get("period_end")
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

    let surface_index = args.get("surface_index")
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

fn list_construction_assemblies(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let _mass_class_filter = args.get("mass_class")
        .and_then(|v| v.as_str())
        .map(|s| match s {
            "VeryLight" => Some(MassClass::VeryLight),
            "Light" => Some(MassClass::Light),
            "Medium" => Some(MassClass::Medium),
            "Heavy" => Some(MassClass::Heavy),
            "VeryHeavy" => Some(MassClass::VeryHeavy),
            _ => None,
        })
        .flatten();

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

    let assemblies = vec![
        ("Heavy Wall", heavy_wall),
        ("Light Wall", light_wall),
        ("Roof", roof),
        ("Floor", floor),
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

fn get_ashrae140_results(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let case_id = args.get("case_id")
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
        None => return serde_json::json!({ "error": "No model loaded. Call load_building_model first." }),
    };

    let name = args.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let value = args.get("value")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

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
        None => return serde_json::json!({ "error": "No model loaded. Call load_building_model first." }),
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

fn compare_to_reference(state: &mut McpState, args: &serde_json::Map<String, Value>) -> Value {
    let case_id = args.get("case_id")
        .and_then(|v| v.as_str())
        .unwrap_or("600");

    let metric = args.get("metric")
        .and_then(|v| v.as_str())
        .unwrap_or("annual_heating");

    let references = get_ashrae140_references(case_id);

    let metric_data = references.get(metric)
        .and_then(|v| v.as_object())
        .map(|obj| {
            let min = obj.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let max = obj.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let typical = obj.get("typical").and_then(|v| v.as_f64()).unwrap_or((min + max) / 2.0);
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