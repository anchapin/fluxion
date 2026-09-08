// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Liveness (`/v1/healthz`) and readiness (`/v1/readyz`) endpoints plus the
//! shared `run_readiness_probes[_with]` core (Issue #2514). Decomposed
//! from the legacy `server.rs` so the health/readiness path is one focused
//! submodule (Issue #3457 / #3543 — module-size ratchet).

use axum::{http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};

use crate::api::server::state::AppState;

/// Body returned by `GET /v1/healthz`. Fields are deliberately minimal so
/// load balancers can parse the JSON without coupling to schema internals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

/// Liveness handler. Always returns `200 OK` with a static payload; we
/// deliberately do **not** ping downstream services here so a slow disk does
/// not flap the load balancer.
pub async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

// ── Readiness probes (Issue #2514) ──────────────────────────────────────
//
// `GET /v1/healthz` is deliberately liveness-only — it never pokes
// downstreams so a slow disk does not flap the load balancer. Kubernetes
// still needs a way to keep traffic out of a pod whose dependencies are
// not yet satisfied (missing ONNX model, unreadable weather file, broken
// state store). `/v1/readyz` is that probe: it runs three sub-checks and
// returns 200 only when all of them pass.
//
// The probe logic lives in a pure, synchronous function
// ([`run_readiness_probes_with`]) so the HTTP handler and the
// `fluxion-rest` startup self-check share one definition of "ready".

/// Outcome of a single readiness sub-probe.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessCheck {
    /// `"ok"` when the probe passed, `"fail"` otherwise.
    pub status: &'static str,
    /// Human-readable detail. On success a short note (e.g.
    /// `"mock (no model loaded)"`); on failure the error message.
    pub detail: String,
}

impl ReadinessCheck {
    /// `true` when `status == "ok"`.
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

impl From<Result<String, String>> for ReadinessCheck {
    fn from(res: Result<String, String>) -> Self {
        match res {
            Ok(detail) => ReadinessCheck {
                status: "ok",
                detail,
            },
            Err(detail) => ReadinessCheck {
                status: "fail",
                detail,
            },
        }
    }
}

/// Per-check breakdown returned by `GET /v1/readyz`.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessChecks {
    pub onnx: ReadinessCheck,
    pub weather: ReadinessCheck,
    pub appstate: ReadinessCheck,
}

/// Overall readiness report — the JSON body of `GET /v1/readyz`.
///
/// `status` is `"ok"` only when every check in [`ReadinessChecks`] is ok;
/// [`ReadinessReport::is_ready`] is the canonical accessor so callers do
/// not hard-code the literal.
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessReport {
    pub status: &'static str,
    pub checks: ReadinessChecks,
}

impl ReadinessReport {
    /// `true` when the service is ready to accept traffic.
    pub fn is_ready(&self) -> bool {
        self.status == "ok"
    }
}

/// ONNX surrogate probe.
///
/// When the `ort` feature is enabled, constructing a [`SurrogateManager`]
/// exercises the ONNX-runtime linkage — ABI / shared-library issues
/// surface here rather than on the first request. When an operator
/// explicitly sets `FLUXION_ONNX_MODEL`, the path is verified to exist on
/// disk first (a missing model file is the most common readiness failure
/// under k8s where a ConfigMap/PVC mount is misconfigured). When `ort` is
/// off the probe passes unconditionally: surrogate inference runs in
/// mock/analytical mode, so there is nothing to fail on.
pub(crate) fn probe_onnx(model_env: Option<&str>) -> Result<String, String> {
    #[cfg(feature = "ort")]
    {
        if let Some(path) = model_env {
            if !path.is_empty() && !std::path::Path::new(path).exists() {
                // Generic message — do NOT echo the user-supplied path
                // (Issue #2905: closes the path-oracle / error-leak window
                // now that the full validation pipeline lives one layer
                // down in `SurrogateManager::new_with_auto_load`).
                return Err("FLUXION_ONNX_MODEL file not found".to_string());
            }
        }
        match crate::ai::surrogate::SurrogateManager::new() {
            Ok(m) => {
                if m.model_loaded {
                    Ok("model loaded".to_string())
                } else {
                    Ok("mock (no model loaded)".to_string())
                }
            }
            Err(e) => Err(format!("SurrogateManager::new() failed: {e}")),
        }
    }
    #[cfg(not(feature = "ort"))]
    {
        // ONNX runtime is not compiled in — suppress the unused-param
        // warning so the non-ort build stays clippy-clean.
        let _ = model_env;
        Ok("skipped (ort feature off)".to_string())
    }
}

/// EPW / weather-file probe.
///
/// The REST API embeds weather inline in each schema, so no default file
/// is required for readiness. When `FLUXION_WEATHER_FILE` is set, however,
/// the path must be readable so a misconfigured mount does not get traffic
/// routed to a server that cannot load TMY data.
pub(crate) fn probe_weather(weather_file: Option<&str>) -> Result<String, String> {
    match weather_file.filter(|p| !p.is_empty()) {
        Some(path) => match std::fs::File::open(path) {
            Ok(_) => Ok(format!("readable: {path}")),
            Err(e) => Err(format!("FLUXION_WEATHER_FILE='{path}' not readable: {e}")),
        },
        None => Ok("no weather file configured".to_string()),
    }
}

/// AppState probe.
///
/// `AppState::default()` must construct (allocating the
/// [`InMemorySimulationStateStore`]). Construction is infallible today,
/// but the probe exists so a future state-store init that *can* fail
/// (e.g. a cloud store requiring credentials) has a deterministic fail
/// point at readiness time rather than on the first request.
pub(crate) fn probe_appstate() -> Result<String, String> {
    let _state = AppState::default();
    Ok("initialized".to_string())
}

/// Run all readiness probes against explicit inputs. Pure (no env read)
/// so it is deterministic under test; [`run_readiness_probes`] is the
/// env-reading wrapper used by the HTTP handler and startup self-check.
pub fn run_readiness_probes_with(
    onnx_model: Option<&str>,
    weather_file: Option<&str>,
) -> ReadinessReport {
    let onnx: ReadinessCheck = probe_onnx(onnx_model).into();
    let weather: ReadinessCheck = probe_weather(weather_file).into();
    let appstate: ReadinessCheck = probe_appstate().into();
    let ready = onnx.is_ok() && weather.is_ok() && appstate.is_ok();
    ReadinessReport {
        status: if ready { "ok" } else { "not ready" },
        checks: ReadinessChecks {
            onnx,
            weather,
            appstate,
        },
    }
}

/// Run all readiness probes, reading configuration from the environment:
///
/// - `FLUXION_ONNX_MODEL` — explicit ONNX model path (probe verifies it
///   exists when `--features ort` is on).
/// - `FLUXION_WEATHER_FILE` — EPW/TMY weather file path (probe verifies
///   it is readable when set).
///
/// This is the single source of truth shared by the `GET /v1/readyz`
/// handler and the `fluxion-rest` startup self-check, so the live endpoint
/// and the boot-time gate agree on "ready".
pub fn run_readiness_probes() -> ReadinessReport {
    let onnx_model = std::env::var("FLUXION_ONNX_MODEL").ok();
    let weather_file = std::env::var("FLUXION_WEATHER_FILE").ok();
    run_readiness_probes_with(onnx_model.as_deref(), weather_file.as_deref())
}

/// Readiness handler (Issue #2514).
///
/// Returns `200 OK` with a per-check breakdown when every probe passes,
/// or `503 Service Unavailable` with the same breakdown when any probe
/// fails. Unlike [`healthz`] (liveness), this endpoint *does* poke
/// downstream dependencies, so it must be wired to a k8s
/// `readinessProbe` (not `livenessProbe`) to avoid restart loops.
pub async fn readyz() -> Response {
    let report = run_readiness_probes();
    let status = if report.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(report)).into_response()
}

// Convenience: re-export the type used by router so we can keep the public
// API stable.
type Response = axum::response::Response;
