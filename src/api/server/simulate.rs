// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `/v1/simulate` endpoint — request / response types, the `ValidatedJson`
//! extractor, `parse_selector_from_options`, the synchronous `run_simulation`
//! runner, the schema→physics `build_model_from_schema` wiring, and the
//! `simulate` + `simulate_stream` handlers.
//!
//! Decomposed from the legacy `server.rs` so the simulate path is one
//! focused submodule (Issue #3457 / #3543 — module-size ratchet).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_stream::stream;
use axum::{
    extract::{FromRequest, Request, State},
    http::StatusCode,
    response::Response,
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::ai::surrogate::SurrogateManager;
use crate::api::error::SimulationDiagnostics;
use crate::api::metrics;
use crate::api::schema::{SimulationOutput, SimulationSchema, SimulationSchemaV1};
use crate::api::server::api_error::ApiError;
use crate::api::server::constants::{MAX_BATCH_SIMULATIONS, MAX_CAMPAIGN_STEPS, MAX_YEARS};
use crate::api::server::state::AppState;
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::sim::thermal_selector::ThermalSelector;

use super::X_REQUEST_ID;

/// Optional knobs attached to a simulation request. Defaults match the
/// existing `bindings.rs` path so the REST result matches an in-process call
/// within numerical noise.
#[derive(Debug, Clone, Deserialize)]
pub struct SimulateOptions {
    /// Number of years to simulate. Default: `1`. Bounded to `1..=MAX_YEARS`
    /// at deserialisation (Issue #2530) so a `{"years": u32::MAX}` payload is
    /// rejected as a 400 before `steps = years * 8760` is ever computed.
    #[serde(default = "default_years", deserialize_with = "validate_years")]
    pub years: u32,
    /// Whether to use the ONNX surrogate path. Default: `false`.
    #[serde(default)]
    pub use_surrogates: bool,
    /// Zone solver selection (Issue #3281). One of `"gauge"` (default),
    /// `"5r1c"`, `"9r4c"`. Validated by [`parse_selector_from_options`]
    /// *after* deserialisation so the rejection message can name the
    /// experimental gate (`FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1`) for the
    /// reserved `"6r2c"` / `"8r3c"` identifiers. `None` ⇒
    /// [`ThermalSelector::default()`].
    #[serde(default)]
    pub zone_solver: Option<String>,
    /// Conduction algorithm selection (Issue #3281). One of `"default"`
    /// (default), `"ctf"`, `"fd"`. Validated alongside `zone_solver`.
    #[serde(default)]
    pub conduction_solver: Option<String>,
    /// Optional opaque id; if present, the request's schema is stored under
    /// this id *and* the id is returned for retrieval via
    /// `GET /v1/schema/{id}`.
    #[serde(default)]
    pub store_as: Option<String>,
}

impl Default for SimulateOptions {
    fn default() -> Self {
        SimulateOptions {
            years: default_years(),
            use_surrogates: false,
            zone_solver: None,
            conduction_solver: None,
            store_as: None,
        }
    }
}

fn default_years() -> u32 {
    1
}

/// Serde validator for `SimulateOptions.years` (Issue #2530). Rejects `0` and
/// any value above [`MAX_YEARS`] at deserialisation so the `Json` extractor
/// surfaces a structured 400 (via [`ValidatedJson`]) rather than letting a
/// multi-trillion-step payload reach the synchronous solver. `#[serde(default
/// = "default_years")]` is still honoured when the field is *absent* — this
/// function only runs when the caller supplies an explicit value.
fn validate_years<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = u32::deserialize(deserializer)?;
    if v == 0 {
        return Err(serde::de::Error::custom(format!(
            "options.years must be between 1 and {MAX_YEARS} (got 0)"
        )));
    }
    if v > MAX_YEARS {
        return Err(serde::de::Error::custom(format!(
            "options.years must be between 1 and {MAX_YEARS} (got {v})"
        )));
    }
    Ok(v)
}

/// Translate the optional `zone_solver` / `conduction_solver` request fields
/// into a [`ThermalSelector`] (Issue #3281).
pub fn parse_selector_from_options(options: &SimulateOptions) -> Result<ThermalSelector, ApiError> {
    let zone_solver = match &options.zone_solver {
        Some(s) => {
            let parsed = crate::sim::thermal_selector::parse_zone_solver(s)
                .map_err(ApiError::InvalidRequest)?;
            if parsed == crate::sim::thermal_selector::ZoneSolverKind::Gauge {
                return Err(ApiError::InvalidRequest(
                    "explicit zone_solver \"gauge\" is not supported over REST: the REST schema \
                     does not carry per-surface construction detail (wall_spec), so the gauge \
                     solver cannot initialise on this path and the request would silently fall \
                     through to 5R1C. Omit zone_solver or request \"5r1c\" / \"9r4c\" (fail-closed \
                     per issue #3305)"
                        .to_string(),
                ));
            }
            parsed
        }
        // Issue #3508: the REST path builds via `ThermalModel::new` (line 1250)
        // which does NOT initialise the gauge backend. Defaulting to
        // `ThermalSelector::default()` (Gauge) would panic at step time under
        // `--features gauge-solver`. Route the default REST request to the
        // legacy 5R1C path explicitly.
        None => crate::sim::thermal_selector::ThermalSelector::legacy().zone_solver,
    };
    let conduction_solver = match &options.conduction_solver {
        Some(s) => crate::sim::thermal_selector::parse_conduction_solver(s)
            .map_err(ApiError::InvalidRequest)?,
        None => crate::sim::thermal_selector::ThermalSelector::legacy().conduction_solver,
    };
    Ok(ThermalSelector {
        zone_solver,
        conduction_solver,
    })
}

/// Request body for `POST /v1/simulate`.
#[derive(Debug, Clone, Deserialize)]
pub struct SimulateRequest {
    /// The simulation schema. Accepts either a bare `SimulationSchemaV1`
    /// or the version-tagged `SimulationSchema` envelope.
    #[serde(flatten)]
    pub schema: SimulationSchemaBody,
    #[serde(default)]
    pub options: SimulateOptions,
}

/// Helper for the polymorphic schema payload (bare V1 or `{ "version": ... }`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SimulationSchemaBody {
    V1(SimulationSchemaV1),
    Enveloped(SimulationSchema),
}

impl SimulationSchemaBody {
    /// Unwrap to the V1 schema regardless of which wire form was supplied.
    pub fn into_v1(self) -> SimulationSchemaV1 {
        match self {
            SimulationSchemaBody::V1(v) => v,
            SimulationSchemaBody::Enveloped(SimulationSchema::V1(v)) => v,
        }
    }
}

/// Validating JSON extractor (Issue #2530).
///
/// Wraps [`axum::Json`] so that any deserialisation failure surfaces as a
/// structured [`ApiError::InvalidRequest`] (HTTP 400) instead of axum's
/// default `JsonRejection` (which is a bare 422 with a non-`error`-enveloped
/// body).
pub struct ValidatedJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(j) => Ok(ValidatedJson(j.0)),
            // `body_text()` preserves the inner serde error chain (e.g. the
            // `validate_years` "options.years must be between 1 and 10"
            // message) so the client can see *why* the body was rejected.
            Err(rejection) => Err(ApiError::InvalidRequest(rejection.body_text())),
        }
    }
}

/// Response body for `POST /v1/simulate`.
#[derive(Debug, Clone, Serialize)]
pub struct SimulateResponse {
    pub schema_id: Option<String>,
    pub output: SimulationOutput,
}

/// Build a [`ThermalModel`] from a [`SimulationSchemaV1`], mirroring the
/// schema→physics wiring that `ThermalModel::from_spec` performs for the
/// ASHRAE 140 validation path.
///
/// This is the root-cause fix for issue #2747 / LIMIT-07: previously
/// `run_simulation` and `/v1/simulate/stream` called `ThermalModel::new`
/// and set only the heating/cooling setpoints, leaving `thermal_capacitance`
/// at its `1.0 J/K` placeholder.
pub(crate) fn build_model_from_schema(schema: &SimulationSchemaV1) -> ThermalModel<VectorField> {
    use crate::sim::construction::{Construction, SurfaceType};

    let num_zones = schema.geometry.zones.len().max(1);
    let mut model = ThermalModel::<VectorField>::new(num_zones);

    let heating = schema.controls.zone_control.heating_setpoint;
    let cooling = schema.controls.zone_control.cooling_setpoint;

    // Constants — ρ_air and cp_air at sea level.
    const AIR_DENSITY: f64 = 1.2; // kg/m³
    const AIR_SPECIFIC_HEAT: f64 = 1005.0; // J/(kg·K)
    const DEFAULT_INFILTRATION_ACH: f64 = 0.5;
    const H_MS_COEFF_LOW_MASS: f64 = 2.0; // W/(m²·K)
    const H_SI: f64 = 3.45; // W/(m²·K)

    let mut zone_area_vec = Vec::with_capacity(num_zones);
    let mut ceiling_height_vec = Vec::with_capacity(num_zones);
    let mut zone_volume_vec = Vec::with_capacity(num_zones);
    let mut wall_area_vec = Vec::with_capacity(num_zones);
    let mut roof_area_vec = Vec::with_capacity(num_zones);
    let mut floor_area_vec = Vec::with_capacity(num_zones);
    let mut window_ratio_vec = Vec::with_capacity(num_zones);
    let mut infiltration_vec = Vec::with_capacity(num_zones);

    let wall_c: Construction = Construction::new(schema.constructions.wall.layers.clone());
    let roof_c: Construction = Construction::new(schema.constructions.roof.layers.clone());
    let floor_c: Construction = Construction::new(schema.constructions.floor.layers.clone());

    let wall_u_value = wall_c.u_value(Some(SurfaceType::Wall), None);
    let roof_u_value = roof_c.u_value(Some(SurfaceType::Ceiling), None);
    let floor_u_value = floor_c.u_value(Some(SurfaceType::Floor), None);
    let window_u_value = schema
        .constructions
        .wall
        .window
        .as_ref()
        .map(|w| w.window_u_value)
        .unwrap_or(2.5);

    for zone in &schema.geometry.zones {
        let floor_area = zone.floor_area.max(1.0);
        let height = zone.height.max(1.0);
        let volume = if zone.volume > 0.0 {
            zone.volume
        } else {
            floor_area * height
        };
        // Square-footprint approximation: perimeter = 4·√(A).
        let perimeter = 4.0 * floor_area.sqrt();
        let gross_wall_area = perimeter * height;
        let window_area = schema
            .constructions
            .wall
            .window
            .as_ref()
            .map(|w| w.window_area)
            .filter(|a| *a > 0.0 && *a <= gross_wall_area)
            .unwrap_or(0.15 * gross_wall_area);
        let window_ratio = if gross_wall_area > 0.0 {
            window_area / gross_wall_area
        } else {
            0.0
        };

        zone_area_vec.push(floor_area);
        ceiling_height_vec.push(height);
        zone_volume_vec.push(volume);
        wall_area_vec.push(gross_wall_area);
        roof_area_vec.push(floor_area); // flat-roof assumption
        floor_area_vec.push(floor_area);
        window_ratio_vec.push(window_ratio);
        infiltration_vec.push(DEFAULT_INFILTRATION_ACH);
    }

    while zone_area_vec.len() < num_zones {
        zone_area_vec.push(*zone_area_vec.last().unwrap_or(&48.0));
        ceiling_height_vec.push(*ceiling_height_vec.last().unwrap_or(&2.7));
        zone_volume_vec.push(*zone_volume_vec.last().unwrap_or(&129.6));
        wall_area_vec.push(*wall_area_vec.last().unwrap_or(&64.8));
        roof_area_vec.push(*roof_area_vec.last().unwrap_or(&48.0));
        floor_area_vec.push(*floor_area_vec.last().unwrap_or(&48.0));
        window_ratio_vec.push(*window_ratio_vec.last().unwrap_or(&0.15));
        infiltration_vec.push(DEFAULT_INFILTRATION_ACH);
    }

    model.setpoints.zone_area = VectorField::new(zone_area_vec.clone());
    model.setpoints.ceiling_height = VectorField::new(ceiling_height_vec.clone());
    model.setpoints.zone_volume = VectorField::new(zone_volume_vec.clone());
    model.setpoints.wall_area = VectorField::new(wall_area_vec.clone());
    model.setpoints.roof_area = VectorField::new(roof_area_vec.clone());
    model.setpoints.floor_area = VectorField::new(floor_area_vec.clone());
    model.setpoints.window_ratio = VectorField::new(window_ratio_vec.clone());
    model.setpoints.aspect_ratio = VectorField::from_scalar(1.0, num_zones);
    model.setpoints.infiltration_rate = VectorField::new(infiltration_vec.clone());
    model.setpoints.air_density = VectorField::from_scalar(AIR_DENSITY, num_zones);
    model.setpoints.heat_capacity = VectorField::from_scalar(AIR_SPECIFIC_HEAT, num_zones);

    model.setpoints.wall_u_value = wall_u_value;
    model.setpoints.roof_u_value = roof_u_value;
    model.setpoints.floor_u_value = floor_u_value;
    model.solar.window_u_value = window_u_value;

    let mut thermal_cap_vec = Vec::with_capacity(num_zones);
    let mut air_thermal_cap_vec = Vec::with_capacity(num_zones);
    let mut h_tr_ms_vec = Vec::with_capacity(num_zones);
    let mut h_tr_em_vec = Vec::with_capacity(num_zones);
    let mut h_tr_me_vec = Vec::with_capacity(num_zones);

    let wall_cap_per_area = wall_c.thermal_capacitance_per_area();
    let roof_cap_per_area = roof_c.thermal_capacitance_per_area();
    let floor_cap_per_area = floor_c.thermal_capacitance_per_area();

    for zone_idx in 0..num_zones {
        let zone_floor_area = zone_area_vec[zone_idx];
        let zone_wall_area = wall_area_vec[zone_idx];
        let zone_volume = zone_volume_vec[zone_idx];
        let window_area = window_ratio_vec[zone_idx] * zone_wall_area;
        let opaque_wall_area = (zone_wall_area - window_area).max(0.0);

        let wall_cap = wall_cap_per_area * opaque_wall_area;
        let roof_cap = roof_cap_per_area * zone_floor_area;
        let floor_cap = floor_cap_per_area * zone_floor_area;
        let total_thermal_cap = (wall_cap + roof_cap + floor_cap).max(1.0e3);
        thermal_cap_vec.push(total_thermal_cap);

        let air_cap = zone_volume * AIR_DENSITY * AIR_SPECIFIC_HEAT;
        air_thermal_cap_vec.push(air_cap);

        let a_m = 2.5 * zone_floor_area;
        let h_ms = H_MS_COEFF_LOW_MASS * a_m;
        h_tr_ms_vec.push(h_ms);

        let h_op = wall_u_value * opaque_wall_area + roof_u_value * zone_floor_area;
        let h_em = if h_op > 0.0 && h_op < h_ms {
            (1.0 / (1.0 / h_op - 1.0 / h_ms)).max(0.1)
        } else {
            h_op.max(0.1)
        };
        h_tr_em_vec.push(h_em);

        h_tr_me_vec.push(9.1 * 0.5 * zone_floor_area);

        let _h_tr_is_check = H_SI * zone_floor_area;
    }

    model.mass.thermal_capacitance = VectorField::new(thermal_cap_vec);
    model.mass.air_thermal_capacitance = VectorField::new(air_thermal_cap_vec);
    model.conduction.h_tr_ms = VectorField::new(h_tr_ms_vec);
    model.conduction.h_tr_em = VectorField::new(h_tr_em_vec);
    model.mass.h_tr_me = VectorField::new(h_tr_me_vec);
    model.conduction.h_tr_is = VectorField::from_scalar(0.0, num_zones);

    use crate::validation::ashrae_140_cases::Orientation;
    let orientations = [
        Orientation::South,
        Orientation::West,
        Orientation::North,
        Orientation::East,
    ];
    let mut surfaces = Vec::with_capacity(num_zones);
    for zone_idx in 0..num_zones {
        let gross_wall_area = wall_area_vec[zone_idx];
        let per_orientation = gross_wall_area / 4.0;
        let total_window_area = window_ratio_vec[zone_idx] * gross_wall_area;
        let window_per_orientation = total_window_area / 4.0;
        let mut zone_surfaces = Vec::with_capacity(orientations.len());
        for &orientation in &orientations {
            let surface = crate::sim::construction::WallSurface::new(
                per_orientation,
                wall_u_value,
                orientation,
            )
            .with_window(window_per_orientation);
            zone_surfaces.push(surface);
        }
        surfaces.push(zone_surfaces);
    }
    model.solar.surfaces = surfaces;

    model.setpoints.heating_setpoint = heating;
    model.setpoints.cooling_setpoint = cooling;
    model.setpoints.heating_setpoints = VectorField::from_scalar(heating, num_zones);
    model.setpoints.cooling_setpoints = VectorField::from_scalar(cooling, num_zones);
    model.hvac.hvac_enabled = VectorField::from_scalar(1.0, num_zones);
    model.hvac.hvac_heating_capacity = schema.controls.zone_control.heating_capacity.max(1.0);
    model.hvac.hvac_cooling_capacity = schema.controls.zone_control.cooling_capacity.max(1.0);
    model.setpoints.heating_schedule = schema.schedules.hvac.heating.clone();
    model.setpoints.cooling_schedule = schema.schedules.hvac.cooling.clone();

    model.update_derived_parameters();

    model
}

/// Run a simulation synchronously and return the structured output.
#[tracing::instrument(
    skip(schema),
    fields(request_id = %request_id, num_zones = schema.geometry.zones.len().max(1), years),
)]
pub fn run_simulation(
    schema: &SimulationSchemaV1,
    years: u32,
    use_surrogates: bool,
    selector: ThermalSelector,
    request_id: &str,
) -> Result<SimulationOutput, ApiError> {
    let num_zones = schema.geometry.zones.len().max(1);

    let heating = schema.controls.zone_control.heating_setpoint;
    let cooling = schema.controls.zone_control.cooling_setpoint;
    if heating >= cooling {
        return Err(ApiError::InvalidSchema(format!(
            "heating_setpoint ({heating}) must be < cooling_setpoint ({cooling})"
        )));
    }
    if schema.geometry.zones.is_empty() {
        return Err(ApiError::InvalidSchema(
            "geometry.zones must contain at least one zone".to_string(),
        ));
    }

    let years = years.clamp(1, MAX_YEARS);

    let solve_started = std::time::Instant::now();
    let empty_lighting =
        crate::sim::lighting::LightingSchedule::new(0.0, schema.geometry.total_floor_area);
    let solve_result: Result<SimulationOutput, ApiError> = (|| {
        let mut model = build_model_from_schema(schema);
        model.hvac.thermal_selector = selector;
        for zone_idx in 0..model.hvac.num_zones {
            model.setpoints.heating_setpoints.as_mut_slice()[zone_idx] = heating;
            model.setpoints.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;
        }

        let steps = years as usize * 8760;
        let surrogates = SurrogateManager::new().map_err(|e| {
            ApiError::SimulationFailed(format!("failed to create SurrogateManager: {e}"), None)
        })?;

        if use_surrogates && !surrogates.model_loaded {
            tracing::warn!(
                backend = surrogates.backend.as_str(),
                "surrogate requested but no ONNX model loaded — using analytical fallback"
            );
        }

        let _ = model.solve_timesteps(
            steps,
            &surrogates,
            use_surrogates,
            Some(&empty_lighting),
            None,
            None,
        );

        if let Some(hourly) = model.get_hourly_temperatures() {
            if let Some(diag) = SimulationDiagnostics::from_temperature_trace(&hourly) {
                return Err(ApiError::SimulationFailed(
                    format!(
                        "simulation diverged at timestep {}{}",
                        diag.failing_timestep,
                        diag.failing_zone
                            .as_ref()
                            .map(|z| format!(" in zone {z}"))
                            .unwrap_or_default()
                    ),
                    Some(diag),
                ));
            }
        }

        let heating_energy = model.get_heating_energy_kwh();
        let cooling_energy = model.get_cooling_energy_kwh();
        let total_energy = heating_energy + cooling_energy;
        let floor_area = schema.geometry.total_floor_area.max(1.0);
        let eui = total_energy / floor_area;

        let peak_heating_load = model.get_peak_heating_power_kw() * 1000.0;
        let peak_cooling_load = model.get_peak_cooling_power_kw() * 1000.0;

        let hourly_zone_temperatures = model.get_hourly_temperatures();
        let zone_temperatures = model.get_temperatures();

        Ok(SimulationOutput {
            eui,
            total_energy,
            peak_heating_load,
            peak_cooling_load,
            heating_energy,
            cooling_energy,
            zone_temperatures: Some(zone_temperatures),
            hourly_zone_temperatures,
            effective_solver: Some(model.effective_zone_solver().as_str().to_string()),
        })
    })();
    let solve_elapsed = solve_started.elapsed().as_secs_f64();

    metrics::record_simulation(
        solve_elapsed,
        years,
        use_surrogates,
        solve_result.is_ok(),
        num_zones,
        solve_result.as_ref().ok().map(|o| o.total_energy),
        &format!(
            "{}+{}",
            selector.zone_solver.as_str(),
            selector.conduction_solver.as_str()
        ),
    );

    solve_result
}

/// Stable, non-cryptographic hash of a simulation schema used for audit
/// correlation (Issue #2546). Two requests with byte-identical canonical
/// JSON produce the same id; intentionally NOT a security primitive.
pub(crate) fn schema_audit_hash(schema: &SimulationSchemaV1) -> String {
    let mut hasher = DefaultHasher::new();
    if let Ok(canonical) = serde_json::to_string(schema) {
        canonical.hash(&mut hasher);
    }
    format!("0x{:016x}", hasher.finish())
}

#[tracing::instrument(skip_all, fields(request_id, num_zones, years))]
pub async fn simulate(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    ValidatedJson(req): ValidatedJson<SimulateRequest>,
) -> Result<Json<SimulateResponse>, ApiError> {
    let request_id = headers
        .get(X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    let client_id: Option<String> = headers
        .get("x-fluxion-client")
        .or_else(|| headers.get(axum::http::header::USER_AGENT))
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    let schema = req.schema.into_v1();
    let options = req.options;

    let selector = parse_selector_from_options(&options)?;

    let num_zones = schema.geometry.zones.len();
    let years = options.years;
    let use_surrogates = options.use_surrogates;
    let schema_hash = schema_audit_hash(&schema);

    tracing::Span::current().record("request_id", request_id.as_str());
    tracing::Span::current().record("num_zones", num_zones);
    tracing::Span::current().record("years", years);

    tracing::info!(
        target: "audit",
        event = "simulation_started",
        request_id = %request_id,
        schema_hash = %schema_hash,
        num_zones = num_zones,
        years = years,
        use_surrogates = use_surrogates,
        client_id = ?client_id,
    );

    let started = std::time::Instant::now();
    let schema_for_sim = schema.clone();
    let request_id_for_sim = request_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        run_simulation(
            &schema_for_sim,
            years,
            use_surrogates,
            selector,
            &request_id_for_sim,
        )
    })
    .await
    .map_err(|join_err| {
        ApiError::SimulationFailed(format!("simulation blocking task failed: {join_err}"), None)
    })?;
    tracing::info!(
        target: "audit",
        event = "simulation_completed",
        request_id = %request_id,
        duration_ms = started.elapsed().as_millis(),
        outcome = if result.is_ok() { "success" } else { "error" },
    );
    let output = result?;

    let schema_id = if let Some(id) = options.store_as.clone() {
        state.schemas.write().insert(id.clone(), schema);
        Some(id)
    } else {
        Some(state.store(schema).await)
    };

    Ok(Json(SimulateResponse { schema_id, output }))
}

/// SSE event payload for per-timestep zone temperatures (used by
/// `simulate_stream`).
#[derive(Debug, Clone, Serialize)]
pub struct TimestepEvent {
    pub timestep: usize,
    pub zone_temperatures: Vec<f64>,
}

/// SSE streaming handler for `POST /v1/simulate/stream`. Emits one SSE event
/// per timestep with the current zone temperatures.
pub async fn simulate_stream(
    State(state): State<AppState>,
    ValidatedJson(req): ValidatedJson<SimulateRequest>,
) -> Result<Response, ApiError> {
    let schema = req.schema.into_v1();
    let options = req.options;

    let heating = schema.controls.zone_control.heating_setpoint;
    let cooling = schema.controls.zone_control.cooling_setpoint;
    if heating >= cooling {
        return Err(ApiError::InvalidSchema(format!(
            "heating_setpoint ({heating}) must be < cooling_setpoint ({cooling})"
        )));
    }
    if schema.geometry.zones.is_empty() {
        return Err(ApiError::InvalidSchema(
            "geometry.zones must contain at least one zone".to_string(),
        ));
    }

    let years = options.years.clamp(1, MAX_YEARS);
    let steps = years as usize * 8760;
    let surrogates = SurrogateManager::new().map_err(|e| {
        ApiError::SimulationFailed(format!("failed to create SurrogateManager: {e}"), None)
    })?;
    let (tx, rx) = mpsc::channel::<Result<TimestepEvent, ApiError>>(100);

    let schema_for_stream = schema.clone();
    let empty_lighting =
        crate::sim::lighting::LightingSchedule::new(0.0, schema.geometry.total_floor_area);
    tokio::spawn(async move {
        let mut model = build_model_from_schema(&schema_for_stream);
        for zone_idx in 0..model.hvac.num_zones {
            model.setpoints.heating_setpoints.as_mut_slice()[zone_idx] = heating;
            model.setpoints.cooling_setpoints.as_mut_slice()[zone_idx] = cooling;
        }

        let dt_seconds = model.calculate_timestep_seconds();
        let _ = model.solve_timesteps_with_dt(
            steps,
            &surrogates,
            options.use_surrogates,
            Some(&empty_lighting),
            None,
            None,
            dt_seconds,
        );

        if let Some(hourly_temps) = model.get_hourly_temperatures() {
            for (timestep, zone_temps) in hourly_temps.iter().enumerate() {
                let event = TimestepEvent {
                    timestep,
                    zone_temperatures: zone_temps.clone(),
                };
                if tx.send(Ok(event)).await.is_err() {
                    break;
                }
            }
        }
    });

    let stream = stream! {
        let mut rx = rx;
        while let Some(item) = rx.recv().await {
            match item {
                Ok(event) => {
                    match serde_json::to_string(&event) {
                        Ok(json) => {
                            yield Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", json));
                        }
                        Err(e) => {
                            yield Ok::<_, std::convert::Infallible>(format!("data: {{\"error\": \"{}\"}}\n\n", ApiError::SerializationFailed(e.to_string())));
                        }
                    }
                }
                Err(e) => {
                    yield Ok::<_, std::convert::Infallible>(format!("data: {{\"error\": \"{}\"}}\n\n", e));
                }
            }
        }
    };

    let _ = state.store(schema).await;

    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(axum::body::Body::from_stream(stream))
        .unwrap();

    Ok(response)
}

// =====================================================================
// Quiet the unused-import warning when only some symbols are used.
// =====================================================================
#[doc(hidden)]
pub fn _force_use_simulate_imports() {
    let _ = Arc::new(0u8);
    let _ = MAX_CAMPAIGN_STEPS;
    let _ = MAX_BATCH_SIMULATIONS;
}
