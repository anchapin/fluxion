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

// Symbols deleted per issue #3874 — OpenAPI drift checker was disabled, see
// https://github.com/anchapin/fluxion/issues/3874. The drift detector
// (`openapi_router_drift`), its result type (`OpenApiRouterDrift`), the
// convenience predicate (`OpenApiRouterDrift::is_clean`), and the test-only
// string accessor (`HttpMethod::as_str`) were unreachable once the
// `scripts-tests.yml` `openapi_drift` job was retired. If option (a) of
// issue #3874 (re-enable the checker) is later chosen, restore them from
// the git history of `src/api/server/router.rs`.

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

// =========================================================================
// Issue #4201 — TraceLayer credential-redaction regression tests.
//
// The production deploy checklist (`docs/SECURITY.md`) names the regression
// test `tracelayer_does_not_log_credentials` as the guard for the OWASP
// A09 credential-leak control: `SafeHeaderMakeSpan` records only an
// explicit allow-list of safe headers onto the `tower_http` TraceLayer
// span, omitting credential headers by construction. These tests keep that
// control live: one asserts the allow-list is exactly the three vetted
// headers (any widening fails), the other routes a credential-carrying
// request through `SafeHeaderMakeSpan`'s span construction with a
// `tracing_subscriber` capture buffer and asserts neither the credential
// header names nor their secret values appear in the emitted span fields.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Capture buffer mirroring the `AuditCaptureBuf` pattern from
    /// `src/api/server/tests.rs` (Issue #3652): routes fmt-layer output
    /// into an in-memory buffer for assertion.
    #[derive(Clone)]
    struct SpanCaptureBuf(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for SpanCaptureBuf {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SpanCaptureBuf {
        type Writer = SpanCaptureBuf;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Build a request carrying one allow-listed header (positive control)
    /// plus the four credential headers the checklist names, each with a
    /// distinctive secret value that must never appear in span output.
    fn credential_carrying_request() -> axum::http::Request<()> {
        let mut request = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/healthz")
            .body(())
            .expect("request builder");
        let headers = request.headers_mut();
        headers.insert("user-agent", "fluxion-test-agent/1.0".parse().unwrap());
        headers.insert(
            "authorization",
            "Bearer sk-4201-secret-bearer-token".parse().unwrap(),
        );
        headers.insert(
            "cookie",
            "session=4201-secret-session-cookie".parse().unwrap(),
        );
        headers.insert("x-api-key", "4201-secret-api-key".parse().unwrap());
        headers.insert(
            "x-amz-security-token",
            "4201-secret-amz-session-token".parse().unwrap(),
        );
        request
    }

    /// The exact allow-list the OWASP A09 control is built on: assert the
    /// full contents so widening (new entry, rename, reorder) fails loudly
    /// instead of silently expanding what the TraceLayer records.
    #[test]
    fn safe_header_allowlist_is_exactly_the_vetted_three() {
        // Compare as a slice so widening the const fails this assertion at
        // runtime (array-vs-array `assert_eq!` would fail to compile
        // instead, which is a noisier signal in CI logs).
        assert_eq!(
            SAFE_HEADER_ALLOWLIST.as_slice(),
            ["x-request-id", "content-type", "user-agent"],
            "SAFE_HEADER_ALLOWLIST must stay exactly the three vetted headers \
             (Issues #2504 / #4201); do not widen it to any credential-bearing header"
        );
        assert_eq!(
            SAFE_HEADER_ALLOWLIST.len(),
            3,
            "allow-list growth must be a deliberate, reviewed change"
        );
    }

    #[test]
    fn tracelayer_does_not_log_credentials() {
        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(SpanCaptureBuf(buf.clone()))
            .with_max_level(tracing::Level::INFO)
            .with_target(false)
            .with_ansi(false)
            .without_time()
            .finish();
        let _dispatch_guard = tracing::dispatcher::set_default(&tracing::Dispatch::new(subscriber));

        let request = credential_carrying_request();
        let mut make_span = SafeHeaderMakeSpan::new();
        // `MakeSpan::make_span` is synchronous — no async runtime needed.
        let span = make_span.make_span(&request);
        // The default fmt layer prints span fields as context on event
        // lines (span creation alone emits nothing), so enter the span and
        // emit one event — mirroring how TraceLayer holds the span open for
        // the request — to get the fields into the capture buffer.
        let _entered = span.enter();
        tracing::info!("tracelayer credential-redaction probe");
        drop(_entered);
        drop(span);
        drop(_dispatch_guard);

        let captured = String::from_utf8(buf.lock().unwrap().clone()).expect("fmt output is UTF-8");

        // Positive control: the allow-listed header must be recorded, which
        // proves the span was actually emitted to the buffer (an empty
        // buffer would pass the negative assertions trivially).
        assert!(
            captured.contains("fluxion-test-agent/1.0"),
            "allow-listed user-agent value must be recorded on the span; captured: {captured}"
        );

        // Credential header names must not appear (fmt field names are
        // lowercase; compare case-insensitively to be thorough).
        let lowered = captured.to_lowercase();
        for name in [
            "authorization",
            "cookie",
            "x-api-key",
            "x-amz-security-token",
        ] {
            assert!(
                !lowered.contains(name),
                "credential header name {name:?} must not appear in span output; captured: {captured}"
            );
        }
        // Credential values must not appear.
        for secret in [
            "sk-4201-secret-bearer-token",
            "4201-secret-session-cookie",
            "4201-secret-api-key",
            "4201-secret-amz-session-token",
        ] {
            assert!(
                !captured.contains(secret),
                "credential value {secret:?} must not appear in span output; captured: {captured}"
            );
        }
    }
}
