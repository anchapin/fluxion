// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `Router` builder, the route registry (`REST_ROUTES`), the
//! `method_router_for_path` resolver, the `SafeHeaderMakeSpan` allow-list
//! span builder, the timeout error handler, and the OpenAPI drift detector.
//!
//! Decomposed from the legacy `server.rs` so the router + middleware
//! wiring lives in one focused submodule (Issue #3457 / #3543 —
//! module-size ratchet).

use axum::{
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post, MethodRouter},
    Json, Router,
};
use tower::timeout::TimeoutLayer;
use tower::{BoxError, ServiceBuilder};
use tower_http::{
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::{DefaultOnResponse, MakeSpan, TraceLayer},
};
use tracing::Level;

use crate::api::metrics;
use crate::api::server::constants::REQUEST_TIMEOUT;
use crate::api::server::state::AppState;

use super::batch::{batch_simulate, get_simulation_status};
use super::campaigns::{get_campaign_status, submit_campaign};
use super::health::{healthz, readyz};
use super::import_format::import_format;
use super::schema_store::{get_schema, openapi_json, openapi_yaml};
use super::simulate::simulate;
use super::X_REQUEST_ID;

/// Error handler for the per-request timeout layer (Issue #2530).
pub async fn handle_timeout_error(err: BoxError) -> Response {
    if err
        .downcast_ref::<tower::timeout::error::Elapsed>()
        .is_some()
    {
        let body = Json(serde_json::json!({
            "error": {
                "kind": "request_timeout",
                "message": "request exceeded the 60-second server budget",
            }
        }));
        (StatusCode::REQUEST_TIMEOUT, body).into_response()
    } else {
        let body = Json(serde_json::json!({
            "error": {
                "kind": "internal_error",
                "message": format!("unhandled middleware error: {err}"),
            }
        }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

/// Request-header names that are safe to record on the `TraceLayer` span.
///
/// This is an **allow-list**, not a deny-list: a header is recorded only if
/// it appears here. Every credential-bearing header (`authorization`,
/// `cookie`, `x-api-key`, AWS Sig V4 `x-amz-*`, proxy-auth tokens, …) is
/// omitted by construction — there is no deny-list to keep in sync.
const SAFE_HEADER_ALLOWLIST: [&str; 3] = ["x-request-id", "content-type", "user-agent"];

/// A [`MakeSpan`] that records an explicit allow-list of safe request
/// headers onto the `tower_http` trace span.
#[derive(Clone)]
struct SafeHeaderMakeSpan {
    level: Level,
}

impl SafeHeaderMakeSpan {
    /// Create a new span builder that emits at `INFO` level.
    fn new() -> Self {
        Self { level: Level::INFO }
    }

    /// Read an allow-listed header off the request, returning `""` if it is
    /// absent or not valid UTF-8. The header name is matched case-insensitively.
    fn safe_header<'a>(headers: &'a axum::http::HeaderMap, name: &str) -> &'a str {
        debug_assert!(
            SAFE_HEADER_ALLOWLIST.contains(&name),
            "SafeHeaderMakeSpan::safe_header({name:?}) — not on SAFE_HEADER_ALLOWLIST; \
             refusing to record an un-vetted header (Issue #2504)"
        );
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
    }
}

impl<B> MakeSpan<B> for SafeHeaderMakeSpan {
    fn make_span(&mut self, request: &axum::http::Request<B>) -> tracing::Span {
        let headers = request.headers();
        let x_request_id = Self::safe_header(headers, "x-request-id");
        let content_type = Self::safe_header(headers, "content-type");
        let user_agent = Self::safe_header(headers, "user-agent");

        macro_rules! make_span {
            ($level:expr) => {
                tracing::span!(
                    $level,
                    "request",
                    method = %request.method(),
                    uri = %request.uri(),
                    version = ?request.version(),
                    x_request_id = %x_request_id,
                    content_type = %content_type,
                    user_agent = %user_agent,
                )
            };
        }
        match self.level {
            Level::ERROR => make_span!(Level::ERROR),
            Level::WARN => make_span!(Level::WARN),
            Level::INFO => make_span!(Level::INFO),
            Level::DEBUG => make_span!(Level::DEBUG),
            Level::TRACE => make_span!(Level::TRACE),
        }
    }
}

impl Default for SafeHeaderMakeSpan {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Issue #2812 — single source of truth for the `/v1/*` REST surface.
// =========================================================================

/// Access tier for a `/v1/*` route — drives where the builder mounts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteTier {
    Public,
    Protected,
}

/// HTTP method tag carried alongside each registry entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HttpMethod {
    Get,
    Post,
}

impl HttpMethod {
    /// Lowercase method name, matching the keys OpenAPI uses under each path.
    #[cfg(test)]
    const fn as_str(self) -> &'static str {
        match self {
            HttpMethod::Get => "get",
            HttpMethod::Post => "post",
        }
    }
}

/// The canonical list of `/v1/*` REST routes — **the single source of truth**
/// (Issue #2812). Each entry is `(HTTP method, path template, access tier)`.
#[rustfmt::skip]
const REST_ROUTES: &[(HttpMethod, &str, RouteTier)] = &[
    (HttpMethod::Get,  "/v1/healthz",                   RouteTier::Public),
    (HttpMethod::Get,  "/v1/readyz",                    RouteTier::Public),
    (HttpMethod::Get,  "/v1/metrics",                   RouteTier::Protected),
    (HttpMethod::Get,  "/v1/openapi.json",              RouteTier::Protected),
    (HttpMethod::Get,  "/v1/openapi.yaml",              RouteTier::Protected),
    (HttpMethod::Post, "/v1/simulate",                  RouteTier::Protected),
    (HttpMethod::Post, "/v1/simulate/stream",           RouteTier::Protected),
    (HttpMethod::Post, "/v1/batch",                     RouteTier::Protected),
    (HttpMethod::Get,  "/v1/simulation/{id}/status",    RouteTier::Protected),
    (HttpMethod::Get,  "/v1/schema/{id}",               RouteTier::Protected),
    (HttpMethod::Post, "/v1/import/{fmt}",              RouteTier::Protected),
    (HttpMethod::Post, "/v1/campaigns",                 RouteTier::Protected),
    (HttpMethod::Get,  "/v1/campaigns/{id}/status",     RouteTier::Protected),
];

/// Resolve a registry path to its [`MethodRouter`] handler pair.
fn method_router_for_path(path: &str) -> MethodRouter<AppState> {
    match path {
        "/v1/healthz" => get(healthz),
        "/v1/readyz" => get(readyz),
        "/v1/metrics" => get(crate::api::metrics::metrics_handler),
        "/v1/openapi.json" => get(openapi_json),
        "/v1/openapi.yaml" => get(openapi_yaml),
        "/v1/simulate" => post(simulate),
        "/v1/simulate/stream" => post(super::simulate::simulate_stream),
        "/v1/batch" => post(batch_simulate),
        "/v1/simulation/{id}/status" => get(get_simulation_status),
        "/v1/schema/{id}" => get(get_schema),
        "/v1/import/{fmt}" => post(import_format),
        "/v1/campaigns" => post(submit_campaign),
        "/v1/campaigns/{id}/status" => get(get_campaign_status),
        other => unreachable!(
            "REST_ROUTES references {other:?} but method_router_for_path has no handler — \
             registry/handler drift (Issue #2812). Add the handler arm or remove the entry."
        ),
    }
}

/// Symmetric-path and per-path method drift between a route registry (the
/// single source of truth) and an OpenAPI document. Produced by
/// [`openapi_router_drift`]; empty iff the two are in sync. (Issue #2812.)
#[cfg(test)]
#[derive(Debug, Default, PartialEq, Eq)]
struct OpenApiRouterDrift {
    only_in_router: Vec<String>,
    only_in_openapi: Vec<String>,
    method_mismatches: Vec<(String, Vec<String>, Vec<String>)>,
}

#[cfg(test)]
impl OpenApiRouterDrift {
    /// `true` iff there is no drift to report.
    fn is_clean(&self) -> bool {
        self.only_in_router.is_empty()
            && self.only_in_openapi.is_empty()
            && self.method_mismatches.is_empty()
    }
}

/// Pure drift detector between a route registry and an OpenAPI YAML document.
#[cfg(test)]
fn openapi_router_drift(registry: &[(HttpMethod, &str)], openapi_yaml: &str) -> OpenApiRouterDrift {
    use std::collections::{BTreeMap, BTreeSet};

    let mut router: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (method, path) in registry {
        router.entry(path).or_default().insert(method.as_str());
    }

    let parsed: serde_yaml::Value = match serde_yaml::from_str(openapi_yaml) {
        Ok(v) => v,
        Err(e) => panic!("OpenAPI YAML failed to parse: {e}"),
    };
    let paths = parsed
        .get("paths")
        .and_then(|v| v.as_mapping())
        .expect("OpenAPI document must have a top-level `paths:` mapping");
    let mut openapi: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (key, val) in paths.iter() {
        let path = key
            .as_str()
            .unwrap_or_else(|| panic!("OpenAPI path key {key:?} must be a string"))
            .to_string();
        let mut methods = BTreeSet::new();
        if let Some(ops) = val.as_mapping() {
            for (method_key, _) in ops.iter() {
                if let Some(s) = method_key.as_str() {
                    let lower = s.to_ascii_lowercase();
                    if matches!(
                        lower.as_str(),
                        "get" | "post" | "put" | "delete" | "patch" | "head" | "options"
                    ) {
                        methods.insert(lower);
                    }
                }
            }
        }
        openapi.insert(path, methods);
    }

    let router_paths: BTreeSet<&str> = router.keys().copied().collect();
    let openapi_paths: BTreeSet<&str> = openapi.keys().map(String::as_str).collect();

    let only_in_router: Vec<String> = router_paths
        .difference(&openapi_paths)
        .map(|s| (*s).to_string())
        .collect();
    let only_in_openapi: Vec<String> = openapi_paths
        .difference(&router_paths)
        .map(|s| (*s).to_string())
        .collect();

    let method_mismatches: Vec<(String, Vec<String>, Vec<String>)> = router_paths
        .intersection(&openapi_paths)
        .filter_map(|path| {
            let router_methods: BTreeSet<String> =
                router[path].iter().map(|s| (*s).to_string()).collect();
            let openapi_methods: &BTreeSet<String> = &openapi[*path];
            if &router_methods == openapi_methods {
                None
            } else {
                Some((
                    (*path).to_string(),
                    router_methods.into_iter().collect(),
                    openapi_methods.iter().cloned().collect(),
                ))
            }
        })
        .collect();

    OpenApiRouterDrift {
        only_in_router,
        only_in_openapi,
        method_mismatches,
    }
}

/// Construct the application's router. Exposed so integration tests can
/// mount it without going through the binary's env-var resolution path.
pub fn router(state: AppState) -> Router {
    let security_cfg = crate::api::security::RestSecurityConfig::from_env()
        .unwrap_or_else(|e| panic!("fluxion-rest security misconfiguration: {e}"));
    router_with_security(state, security_cfg)
}

/// Construct the application's router with an explicit security
/// configuration (Issue #2505).
pub fn router_with_security(
    state: AppState,
    cfg: crate::api::security::RestSecurityConfig,
) -> Router {
    // Touch the recorder so it is installed at server start-up rather than
    // on the first request (matters for `/v1/metrics` smoke checks).
    let _ = metrics::init_recorder();

    let middleware_stack = ServiceBuilder::new()
        .layer(axum::error_handling::HandleErrorLayer::new(
            handle_timeout_error,
        ))
        .layer(TimeoutLayer::new(REQUEST_TIMEOUT))
        .layer(SetRequestIdLayer::new(
            axum::http::HeaderName::from_static(X_REQUEST_ID),
            MakeRequestUuid,
        ))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(SafeHeaderMakeSpan::new())
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .layer(PropagateRequestIdLayer::new(
            axum::http::HeaderName::from_static(X_REQUEST_ID),
        ))
        .layer(middleware::from_fn(metrics::record))
        .layer(middleware::from_fn(metrics::track_in_flight))
        .into_inner();

    let mut protected: Router<AppState> = Router::new();
    let mut public: Router<AppState> = Router::new();
    for &(_, path, tier) in REST_ROUTES {
        let method_router = method_router_for_path(path);
        match tier {
            RouteTier::Protected => protected = protected.route(path, method_router),
            RouteTier::Public => public = public.route(path, method_router),
        }
    }
    let protected_routes = protected.layer(middleware::from_fn_with_state(
        cfg.auth_state(),
        crate::api::security::require_auth,
    ));

    public
        .merge(protected_routes)
        .with_state(state)
        .layer(axum::extract::DefaultBodyLimit::max(
            crate::api::security::MAX_REQUEST_BODY_BYTES,
        ))
        .layer(middleware::from_fn_with_state(
            cfg.rate_limiter(),
            crate::api::security::rate_limit_middleware,
        ))
        .layer(cfg.cors_layer())
        .layer(middleware_stack)
}
