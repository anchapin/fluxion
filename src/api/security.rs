// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! REST security controls for `fluxion-rest` (Issue #2505).
//!
//! The v1 REST surface previously mounted every `/v1/*` route behind zero
//! authentication, no CORS layer, no request body cap, and no rate limiter.
//! This module provides the four DoS / authz hardening primitives required
//! by the issue, plus the boot-guard decision function used by the binary:
//!
//! - **Auth middleware** ([`require_auth`]) — bearer-token check when
//!   `FLUXION_REST_AUTH=token`; proxy-verified mTLS header (or token
//!   fallback) when `FLUXION_REST_AUTH=tls`; no-op when `off` (default for
//!   local dev). `/v1/healthz` is mounted on a separate public sub-router so
//!   liveness probes succeed unauthenticated.
//! - **Per-IP token-bucket governor** ([`RateLimiter`] +
//!   [`rate_limit_middleware`]) — in-memory, dependency-light. Client IP is
//!   resolved from `X-Forwarded-For` / `X-Real-IP` and falls back to axum
//!   `ConnectInfo<SocketAddr>` (the binary wires the latter via
//!   `into_make_service_with_connect_info`).
//! - **CORS** ([`build_cors_layer`]) — explicit origin allow-list from
//!   `FLUXION_REST_CORS_ORIGINS`; defaults to localhost dev origins. Never
//!   `CorsLayer::permissive()`.
//! - **Boot guard** ([`is_insecure_bind_configuration`]) — pure decision
//!   function; the binary refuses to start when `0.0.0.0` + `auth=off`
//!   coincide in a release build unless `FLUXION_REST_ALLOW_INSECURE=1`.
//!
//! All primitives are pure-library code so they are unit-testable from
//! `cargo test -p fluxion --lib api` without spawning the binary.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use parking_lot::Mutex;
use tower_http::cors::CorsLayer;

/// Maximum accepted request body size (Issue #2505). Caps `/v1/import/*`
/// (which accepts arbitrary `Bytes`) and every other POST at **16 MiB** so a
/// single client cannot OOM the server or fill `/tmp` in one shot. This is a
/// hardening bump over the 8 MiB cap introduced by #2530 (which only needed
/// to accommodate a 1024-entry `/v1/batch`).
pub const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Default per-IP sustained request rate (requests / second) when the
/// operator has not set `FLUXION_REST_RATE_LIMIT_RPS`.
pub const DEFAULT_RATE_LIMIT_RPS: u32 = 100;

/// Default per-IP burst capacity when the operator has not set
/// `FLUXION_REST_RATE_LIMIT_BURST`. Intentionally generous (10× the
/// sustained rate) so legitimate bursts — including the ~300-request
/// `api_concurrent_throughput` integration test fired from `127.0.0.1` —
/// never trip the governor, while a sustained flood is still throttled to
/// [`DEFAULT_RATE_LIMIT_RPS`] once the bucket drains.
pub const DEFAULT_RATE_LIMIT_BURST: u32 = 1000;

/// Default header name a reverse proxy sets once it has validated the
/// client's mTLS certificate (used when `FLUXION_REST_AUTH=tls`).
pub const DEFAULT_VERIFIED_HEADER_NAME: &str = "x-verified-client";

/// Default value for the mTLS-verified header.
pub const DEFAULT_VERIFIED_HEADER_VALUE: &str = "1";

// =========================================================================
// Auth mode + security configuration
// =========================================================================

/// Authentication / authorization mode for the REST surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// No authentication (default for local dev / tests). Every route is
    /// reachable anonymously.
    Off,
    /// Require a bearer token matching `FLUXION_REST_AUTH_TOKEN`.
    Token,
    /// mTLS is terminated by a trusted reverse proxy which injects a
    /// verification header (see [`DEFAULT_VERIFIED_HEADER_NAME`]). A valid
    /// bearer token is also accepted as a fallback so the same deployment
    /// can serve both proxied-mTLS and direct-token clients.
    Tls,
}

impl AuthMode {
    /// Parse the mode from the `FLUXION_REST_AUTH` env value.
    ///
    /// **Fail-closed** (Issue #2689): an unrecognized *non-empty* value
    /// returns an `Err`, so a typo (e.g. `FLUXION_REST_AUTH=tken`,
    /// `=mtls`, `=bearer`) can never silently disable authentication. An
    /// empty / whitespace-only value is the legitimate default and
    /// resolves to [`AuthMode::Off`] — this lets `cargo run` and test
    /// harnesses boot without setting `FLUXION_REST_AUTH`, while still
    /// rejecting garbage like a stray unknown config-map key.
    ///
    /// Matching is ASCII-case-insensitive and trims surrounding whitespace,
    /// so `TLS`, ` off `, and `Token` are all accepted. A trailing newline
    /// around a *known* value (e.g. `"token\n"` from a Kubernetes
    /// ConfigMap) is trimmed and parses correctly.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "" | "off" => Ok(AuthMode::Off),
            "token" => Ok(AuthMode::Token),
            "tls" => Ok(AuthMode::Tls),
            _ => Err(format!(
                "unknown FLUXION_REST_AUTH value '{normalized}'; expected one of: off, token, tls"
            )),
        }
    }
}

/// Operator-tunable security configuration. Built from environment in
/// production ([`RestSecurityConfig::from_env`]) or constructed directly in
/// tests.
#[derive(Clone)]
pub struct RestSecurityConfig {
    /// Authentication mode.
    pub auth_mode: AuthMode,
    /// Expected bearer token for [`AuthMode::Token`] / [`AuthMode::Tls`].
    pub auth_token: Option<String>,
    /// CORS origin allow-list (parsed Origin strings, e.g.
    /// `https://app.fluxion.dev`).
    pub cors_origins: Vec<String>,
    /// Per-IP sustained request rate (requests / second).
    pub rate_limit_rps: u32,
    /// Per-IP burst capacity.
    pub rate_limit_burst: u32,
    /// Header name the trusted proxy sets after mTLS validation.
    pub verified_header_name: HeaderName,
    /// Expected value of [`Self::verified_header_name`].
    pub verified_header_value: String,
}

impl Default for RestSecurityConfig {
    fn default() -> Self {
        Self {
            auth_mode: AuthMode::Off,
            auth_token: None,
            cors_origins: default_dev_cors_origins(),
            rate_limit_rps: DEFAULT_RATE_LIMIT_RPS,
            rate_limit_burst: DEFAULT_RATE_LIMIT_BURST,
            verified_header_name: HeaderName::from_static(DEFAULT_VERIFIED_HEADER_NAME),
            verified_header_value: DEFAULT_VERIFIED_HEADER_VALUE.to_string(),
        }
    }
}

impl RestSecurityConfig {
    /// Resolve security configuration from the process environment.
    ///
    /// # Environment variables
    /// - `FLUXION_REST_AUTH` — `off` (default) | `token` | `tls`. Parsed
    ///   strictly by [`AuthMode::parse`]: an unrecognized non-empty value
    ///   (e.g. a typo) is propagated as an `Err` so the server refuses to
    ///   boot rather than silently disabling auth (Issue #2689). Unset /
    ///   empty is the legitimate `off` default.
    /// - `FLUXION_REST_AUTH_TOKEN` — bearer token (required for `token`)
    /// - `FLUXION_REST_CORS_ORIGINS` — comma-separated origin allow-list
    /// - `FLUXION_REST_RATE_LIMIT_RPS` — sustained req/s per IP
    /// - `FLUXION_REST_RATE_LIMIT_BURST` — burst capacity per IP
    /// - `FLUXION_REST_VERIFIED_HEADER_NAME` — mTLS proxy header name
    /// - `FLUXION_REST_VERIFIED_HEADER_VALUE` — mTLS proxy header value
    ///
    /// Returns `Err` iff `FLUXION_REST_AUTH` is set to an unrecognized
    /// non-empty value (fail-closed).
    pub fn from_env() -> Result<Self, String> {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("FLUXION_REST_AUTH") {
            cfg.auth_mode = AuthMode::parse(&v)?;
        }
        if let Ok(v) = std::env::var("FLUXION_REST_AUTH_TOKEN") {
            if !v.is_empty() {
                cfg.auth_token = Some(v);
            }
        }
        if let Ok(v) = std::env::var("FLUXION_REST_CORS_ORIGINS") {
            let parsed: Vec<String> = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !parsed.is_empty() {
                cfg.cors_origins = parsed;
            }
        }
        if let Ok(v) = std::env::var("FLUXION_REST_RATE_LIMIT_RPS") {
            if let Ok(n) = v.parse::<u32>() {
                if n > 0 {
                    cfg.rate_limit_rps = n;
                }
            }
        }
        if let Ok(v) = std::env::var("FLUXION_REST_RATE_LIMIT_BURST") {
            if let Ok(n) = v.parse::<u32>() {
                if n > 0 {
                    cfg.rate_limit_burst = n;
                }
            }
        }
        if let Ok(v) = std::env::var("FLUXION_REST_VERIFIED_HEADER_NAME") {
            if let Ok(name) = HeaderName::try_from(v.as_str()) {
                cfg.verified_header_name = name;
            }
        }
        if let Ok(v) = std::env::var("FLUXION_REST_VERIFIED_HEADER_VALUE") {
            cfg.verified_header_value = v;
        }

        Ok(cfg)
    }

    /// Build the shared [`AuthState`] handed to [`require_auth`].
    pub fn auth_state(&self) -> AuthState {
        AuthState(Arc::new(AuthStateInner {
            mode: self.auth_mode,
            token: self.auth_token.clone(),
            verified_header_name: self.verified_header_name.clone(),
            verified_header_value: self.verified_header_value.clone(),
        }))
    }

    /// Build a fresh [`RateLimiter`] sized to this configuration.
    pub fn rate_limiter(&self) -> RateLimiter {
        RateLimiter::new(self.rate_limit_rps, self.rate_limit_burst)
    }

    /// Build the tower-http [`CorsLayer`] for this configuration.
    pub fn cors_layer(&self) -> CorsLayer {
        build_cors_layer(&self.cors_origins)
    }
}

/// Localhost origins permitted by default when the operator has not set
/// `FLUXION_REST_CORS_ORIGINS`. Covers the common dev server ports (3000
/// for Next.js/CRA, 5173 for Vite, 8080 for the REST server itself).
pub fn default_dev_cors_origins() -> Vec<String> {
    vec![
        "http://localhost".to_string(),
        "http://localhost:3000".to_string(),
        "http://localhost:5173".to_string(),
        "http://localhost:8080".to_string(),
        "http://127.0.0.1".to_string(),
        "http://127.0.0.1:3000".to_string(),
        "http://127.0.0.1:5173".to_string(),
        "http://127.0.0.1:8080".to_string(),
    ]
}

// =========================================================================
// Auth middleware
// =========================================================================

/// Shared, cheaply-clonable state consumed by [`require_auth`].
#[derive(Clone)]
pub struct AuthState(Arc<AuthStateInner>);

struct AuthStateInner {
    mode: AuthMode,
    token: Option<String>,
    verified_header_name: HeaderName,
    verified_header_value: String,
}

/// axum middleware enforcing the configured authentication policy.
///
/// Mounted only on the *protected* sub-router (everything except
/// `/v1/healthz`), so liveness probes remain anonymous. Returns `401` for
/// a missing/wrong credential, and `500` when `token` mode is selected but
/// no token was configured (fail-closed against a misconfiguration that
/// would otherwise reject every request — or, worse, silently accept none).
pub async fn require_auth(
    State(auth): State<AuthState>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    match auth.0.mode {
        AuthMode::Off => Ok(next.run(req).await),
        AuthMode::Token => {
            let expected = match auth.0.token.as_deref() {
                Some(t) if !t.is_empty() => t,
                _ => {
                    tracing::error!(
                        "FLUXION_REST_AUTH=token but FLUXION_REST_AUTH_TOKEN is unset/empty; \
                         rejecting request (fail-closed)"
                    );
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };
            match bearer_token(&req) {
                Some(provided) if constant_time_eq(provided.as_bytes(), expected.as_bytes()) => {
                    Ok(next.run(req).await)
                }
                _ => Err(StatusCode::UNAUTHORIZED),
            }
        }
        AuthMode::Tls => {
            // mTLS is terminated upstream by a trusted reverse proxy. The
            // proxy validates the client certificate and, on success,
            // injects the agreed verification header. A valid bearer token
            // is also accepted so a deployment can mix both auth schemes.
            let proxy_verified = req
                .headers()
                .get(&auth.0.verified_header_name)
                .and_then(|v| v.to_str().ok())
                .map(|s| s == auth.0.verified_header_value)
                .unwrap_or(false);
            if proxy_verified {
                return Ok(next.run(req).await);
            }
            if let Some(expected) = auth.0.token.as_deref().filter(|t| !t.is_empty()) {
                if let Some(provided) = bearer_token(&req) {
                    if constant_time_eq(provided.as_bytes(), expected.as_bytes()) {
                        return Ok(next.run(req).await);
                    }
                }
            }
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// Extract a `Bearer <token>` value from the `Authorization` header, if
/// present and well-formed.
fn bearer_token(req: &Request) -> Option<String> {
    req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .filter(|h| h.starts_with("Bearer "))
        .map(|h| h["Bearer ".len()..].trim().to_string())
}

/// Constant-time byte-slice comparison to avoid timing oracles on the
/// bearer-token check. The length check is intentionally early (it leaks
/// the configured token length, which is acceptable and standard — see
/// e.g. `subtle::ConstantTimeEq`).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// =========================================================================
// Per-IP token-bucket rate limiter
// =========================================================================

/// In-memory, per-IP token-bucket governor. Cheaply clonable (inner state
/// is behind an [`Arc`]); all clones share the same bucket map, so the
/// limit is enforced process-globally per source IP.
#[derive(Clone)]
pub struct RateLimiter(Arc<RateLimiterInner>);

struct RateLimiterInner {
    buckets: Mutex<HashMap<IpAddr, Bucket>>,
    capacity: f64,
    refill_per_sec: f64,
}

struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    /// Construct a limiter with `rps` sustained refill rate and `burst`
    /// capacity. Values of `0` are clamped to `1` so a misconfiguration can
    /// never produce a zero-capacity bucket that would reject every
    /// request.
    pub fn new(rps: u32, burst: u32) -> Self {
        let capacity = burst.max(1) as f64;
        let refill_per_sec = rps.max(1) as f64;
        Self(Arc::new(RateLimiterInner {
            buckets: Mutex::new(HashMap::new()),
            capacity,
            refill_per_sec,
        }))
    }

    /// Attempt to consume one token for `ip`. Returns `true` if the
    /// request is allowed, `false` if the bucket is empty (caller should
    /// respond `429`). Lazily refills based on wall-clock elapsed time.
    pub fn try_acquire(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let mut map = self.0.buckets.lock();
        let bucket = map.entry(ip).or_insert_with(|| Bucket {
            tokens: self.0.capacity,
            last_refill: now,
        });
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            bucket.tokens = (bucket.tokens + elapsed * self.0.refill_per_sec).min(self.0.capacity);
            bucket.last_refill = now;
        }
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// axum middleware enforcing [`RateLimiter`] per source IP. Resolves the
/// client IP via [`client_ip`] (header-first, `ConnectInfo` fallback). When
/// no IP can be determined the request is allowed through (fail-open) — the
/// body limit and timeout still bound worst-case behaviour.
pub async fn rate_limit_middleware(
    State(limiter): State<RateLimiter>,
    req: Request,
    next: Next,
) -> Response {
    match client_ip(&req) {
        Some(ip) => {
            if limiter.try_acquire(ip) {
                next.run(req).await
            } else {
                rate_limited_response()
            }
        }
        None => next.run(req).await,
    }
}

/// Resolve the client IP for rate-limiting purposes. Order of precedence:
/// `X-Forwarded-For` (first hop) → `X-Real-IP` → axum `ConnectInfo`. The
/// header-first order matches deployments behind a trusted reverse proxy;
/// when no proxy is present, the binary wires `ConnectInfo` so the socket
/// peer address is used.
pub fn client_ip(req: &Request) -> Option<IpAddr> {
    if let Some(xff) = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(first) = xff.split(',').next() {
            if let Ok(ip) = first.trim().parse::<IpAddr>() {
                return Some(ip);
            }
        }
    }
    if let Some(xri) = req.headers().get("x-real-ip").and_then(|v| v.to_str().ok()) {
        if let Ok(ip) = xri.trim().parse::<IpAddr>() {
            return Some(ip);
        }
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip())
}

/// `429 Too Many Requests` response with a structured JSON envelope and a
/// `Retry-After: 1` hint.
fn rate_limited_response() -> Response {
    let body = Json(serde_json::json!({
        "error": {
            "kind": "rate_limited",
            "message": "per-IP request rate exceeded; retry later",
        }
    }));
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, HeaderValue::from_static("1"))],
        body,
    )
        .into_response()
}

// =========================================================================
// CORS
// =========================================================================

/// Build a tower-http [`CorsLayer`] allowing exactly the supplied origins.
///
/// An empty list produces a restrictive layer that adds no
/// `Access-Control-Allow-Origin` header, i.e. browsers deny all
/// cross-origin access (non-browser clients are unaffected by CORS).
/// [`CorsLayer::permissive`] is deliberately never used.
pub fn build_cors_layer(origins: &[String]) -> CorsLayer {
    let parsed: Vec<HeaderValue> = origins
        .iter()
        .filter_map(|o| o.trim().parse::<HeaderValue>().ok())
        .collect();

    let methods = [Method::GET, Method::POST, Method::OPTIONS, Method::HEAD];
    let headers = [
        header::CONTENT_TYPE,
        header::AUTHORIZATION,
        header::ACCEPT,
        HeaderName::from_static("x-request-id"),
    ];

    if parsed.is_empty() {
        // No origins configured → no cross-origin browser access. We do
        // NOT set `allow_origin`, so tower-http omits the
        // `Access-Control-Allow-Origin` header entirely and browsers deny
        // every cross-origin request (non-browser clients are unaffected
        // by CORS). `CorsLayer::permissive()` is deliberately never used.
        CorsLayer::new()
            .allow_methods(methods)
            .allow_headers(headers)
    } else {
        CorsLayer::new()
            .allow_origin(parsed)
            .allow_methods(methods)
            .allow_headers(headers)
            .expose_headers([HeaderName::from_static("x-request-id")])
            .max_age(std::time::Duration::from_secs(600))
    }
}

// =========================================================================
// Boot guard (release-only insecure-bind refusal)
// =========================================================================

/// Pure decision function used by the binary's boot guard (Issue #2505).
///
/// Returns `true` when the configuration binds all interfaces **and** runs
/// with authentication disabled — i.e. an anonymous client on the network
/// can reach every `/v1/*` route. The binary refuses to start in that case
/// (release builds only) unless `allow_insecure` is `true`
/// (`FLUXION_REST_ALLOW_INSECURE=1`).
///
/// `bind` may be a bare host (`0.0.0.0`), a `host:port` pair, or an IPv6
/// wildcard (`::` / `::0`); both wildcard families are flagged.
pub fn is_insecure_bind_configuration(
    bind: &str,
    auth_mode: AuthMode,
    allow_insecure: bool,
) -> bool {
    if allow_insecure {
        return false;
    }
    if auth_mode != AuthMode::Off {
        return false;
    }
    // Strip an optional scheme, then try to parse the remainder as a
    // `SocketAddr` (host:port / [v6]:port) first, falling back to a bare
    // `IpAddr`. `Ipv4Addr::UNSPECIFIED` (0.0.0.0) and
    // `Ipv6Addr::UNSPECIFIED` (::) both report `is_unspecified()` == true,
    // which is exactly the "bound to every interface" condition we refuse.
    use std::str::FromStr;
    let raw = bind
        .trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    let ip = std::net::SocketAddr::from_str(raw)
        .map(|sa| sa.ip())
        .or_else(|_| std::net::IpAddr::from_str(raw))
        .ok();
    matches!(ip, Some(ip) if ip.is_unspecified())
}

/// Convenience wrapper for the binary: reads the three inputs from the
/// environment and returns an error message when boot should be refused
/// (release builds). In debug builds the insecure-bind check is a no-op so
/// local `cargo run` keeps working with the defaults — but the
/// `FLUXION_REST_AUTH` value is still **validated in every build**
/// (Issue #2689): an unrecognized non-empty value refuses to boot rather
/// than silently resolving to `off`.
pub fn check_boot_guard_from_env() -> Result<(), String> {
    let auth_raw = std::env::var("FLUXION_REST_AUTH").unwrap_or_default();
    // Issue #2689 — fail-closed on an unrecognized FLUXION_REST_AUTH value
    // in *every* build. A typo that silently resolves to `off` would
    // disable authentication; erroring here is the fail-closed fix.
    let auth = AuthMode::parse(&auth_raw)?;

    // Release-only insecure-bind guard (Issue #2505). In debug builds the
    // default `0.0.0.0` + `off` combination must stay usable for
    // `cargo run`.
    #[cfg(not(debug_assertions))]
    {
        let bind = std::env::var("FLUXION_REST_BIND").unwrap_or_default();
        let allow_insecure = std::env::var("FLUXION_REST_ALLOW_INSECURE")
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        if is_insecure_bind_configuration(&bind, auth, allow_insecure) {
            return Err(format!(
                "fluxion-rest: refusing to boot — FLUXION_REST_BIND='{bind}' binds all interfaces \
                 while FLUXION_REST_AUTH=off. Set FLUXION_REST_AUTH=token (with \
                 FLUXION_REST_AUTH_TOKEN) or FLUXION_REST_AUTH=tls, bind to 127.0.0.1, or set \
                 FLUXION_REST_ALLOW_INSECURE=1 to explicitly opt in."
            ));
        }
    }
    // In debug builds `auth` is only computed to validate the env value;
    // touch it (and the remaining guard inputs) to avoid unused-variable
    // warnings. The env reads also keep the guard inputs "used".
    #[cfg(debug_assertions)]
    {
        let _ = auth;
        let _ = std::env::var("FLUXION_REST_BIND");
        let _ = std::env::var("FLUXION_REST_ALLOW_INSECURE");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AuthMode parsing (fail-closed, Issue #2689) ----
    #[test]
    fn auth_mode_parses_known_values() {
        assert_eq!(AuthMode::parse("token").unwrap(), AuthMode::Token);
        assert_eq!(AuthMode::parse("TLS").unwrap(), AuthMode::Tls);
        assert_eq!(AuthMode::parse("Tls").unwrap(), AuthMode::Tls);
        assert_eq!(AuthMode::parse(" off ").unwrap(), AuthMode::Off);
        assert_eq!(AuthMode::parse("OFF").unwrap(), AuthMode::Off);
        assert_eq!(AuthMode::parse("off").unwrap(), AuthMode::Off);
    }

    #[test]
    fn auth_mode_unset_or_empty_defaults_to_off() {
        // Unset / empty / whitespace-only is the legitimate default — NOT
        // an error — so `cargo run` and test harnesses boot without setting
        // FLUXION_REST_AUTH.
        assert_eq!(AuthMode::parse("").unwrap(), AuthMode::Off);
        assert_eq!(AuthMode::parse("   ").unwrap(), AuthMode::Off);
        assert_eq!(AuthMode::parse("\t\n").unwrap(), AuthMode::Off);
    }

    #[test]
    fn auth_mode_trims_whitespace_around_known_values() {
        // A stray trailing newline (e.g. from a Kubernetes ConfigMap value)
        // around a *known* value is trimmed away and parses correctly.
        assert_eq!(AuthMode::parse("token\n").unwrap(), AuthMode::Token);
        assert_eq!(AuthMode::parse("\n off \n").unwrap(), AuthMode::Off);
        assert_eq!(AuthMode::parse("  TLS  ").unwrap(), AuthMode::Tls);
    }

    #[test]
    fn auth_mode_unknown_value_fails_closed() {
        // Issue #2689: a typo or unrecognized alias must NOT silently coerce
        // to Off — that would disable authentication with no warning.
        let err = AuthMode::parse("tken").unwrap_err();
        assert!(
            err.contains("unknown FLUXION_REST_AUTH value"),
            "error should name the problem, got: {err}"
        );
        assert!(
            err.contains("'tken'"),
            "error should echo the offending value, got: {err}"
        );
        assert!(
            err.contains("off") && err.contains("token") && err.contains("tls"),
            "error should list the valid options, got: {err}"
        );

        // Other plausible typos / aliases called out in the issue.
        assert!(AuthMode::parse("bearer").is_err());
        assert!(AuthMode::parse("mtls").is_err());
        assert!(AuthMode::parse("tls-mode").is_err());
        assert!(AuthMode::parse("none").is_err());
        assert!(AuthMode::parse("disabled").is_err());
        // A trailing newline around an *unknown* value still errors (trim
        // leaves the unknown token intact).
        assert!(AuthMode::parse("mtls\n").is_err());
    }

    #[test]
    fn auth_mode_error_is_never_silently_off() {
        // Defense-in-depth: the Result variant itself must be Err — it must
        // never be Ok(Off) for an unknown input. This is the precise
        // fail-open regression from the bug report (`bogus -> Off`).
        let parsed = AuthMode::parse("bogus");
        assert!(parsed.is_err(), "unknown value must not parse to Ok");
        assert_ne!(parsed, Ok(AuthMode::Off));
    }

    // ---- constant_time_eq ----
    #[test]
    fn constant_time_eq_matches_and_differs() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"", b""));
    }

    // ---- RateLimiter token bucket ----
    #[test]
    fn rate_limiter_allows_up_to_burst_then_rejects() {
        let limiter = RateLimiter::new(0, 5);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..5 {
            assert!(limiter.try_acquire(ip), "within burst should be allowed");
        }
        assert!(!limiter.try_acquire(ip), "over burst should be rejected");
    }

    #[test]
    fn rate_limiter_refills_over_time() {
        // rps=10 → 1 token every 100 ms. Two consecutive synchronous calls
        // are always < 100 ms apart, so the bucket stays empty between
        // them; sleeping 150 ms then refills ≥1 token.
        let limiter = RateLimiter::new(10, 1);
        let ip: IpAddr = "10.0.0.2".parse().unwrap();
        assert!(limiter.try_acquire(ip)); // drains the single token
        assert!(
            !limiter.try_acquire(ip),
            "immediate second acquire must be rejected (no refill yet)"
        );
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(
            limiter.try_acquire(ip),
            "after 150 ms the bucket should have refilled a token"
        );
    }

    #[test]
    fn rate_limiter_isolates_ips() {
        let limiter = RateLimiter::new(0, 1);
        let a: IpAddr = "10.0.0.3".parse().unwrap();
        let b: IpAddr = "10.0.0.4".parse().unwrap();
        assert!(limiter.try_acquire(a));
        assert!(limiter.try_acquire(b), "different IP has its own bucket");
        assert!(!limiter.try_acquire(a));
    }

    // ---- Boot guard decision function ----
    #[test]
    fn boot_guard_flags_wildcard_with_auth_off() {
        assert!(is_insecure_bind_configuration(
            "0.0.0.0",
            AuthMode::Off,
            false
        ));
        assert!(is_insecure_bind_configuration("::", AuthMode::Off, false));
    }

    #[test]
    fn boot_guard_allows_loopback_or_auth() {
        assert!(!is_insecure_bind_configuration(
            "127.0.0.1",
            AuthMode::Off,
            false
        ));
        assert!(!is_insecure_bind_configuration(
            "0.0.0.0",
            AuthMode::Token,
            false
        ));
        assert!(!is_insecure_bind_configuration(
            "0.0.0.0:8080",
            AuthMode::Tls,
            false
        ));
    }

    #[test]
    fn boot_guard_respects_allow_insecure_override() {
        assert!(!is_insecure_bind_configuration(
            "0.0.0.0",
            AuthMode::Off,
            true
        ));
    }

    #[test]
    fn boot_guard_strips_scheme_and_port() {
        assert!(is_insecure_bind_configuration(
            "http://0.0.0.0:8080",
            AuthMode::Off,
            false
        ));
        assert!(is_insecure_bind_configuration(
            "0.0.0.0:8080",
            AuthMode::Off,
            false
        ));
    }
}
