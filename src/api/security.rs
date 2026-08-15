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
//!   resolved from the axum `ConnectInfo<SocketAddr>` socket peer address
//!   (the binary wires it via `into_make_service_with_connect_info`).
//!   `X-Forwarded-For` / `X-Real-IP` are **ignored by default** so a client
//!   cannot spoof a fresh rate-limit bucket per request (Issue #2688). They
//!   are honoured only when the operator explicitly configures a trusted
//!   proxy CIDR allow-list via `FLUXION_REST_TRUSTED_PROXIES`, and even then
//!   only when the connecting peer itself resolves to a trusted proxy. The
//!   per-IP bucket map is bounded by an LRU cap
//!   (`FLUXION_REST_RATE_LIMIT_MAX_ENTRIES`, default 100 000) so a
//!   spoofed-IP or many-IP flood cannot grow memory unboundedly.
//! - **CORS** ([`build_cors_layer`]) — explicit origin allow-list from
//!   `FLUXION_REST_CORS_ORIGINS`; defaults to localhost dev origins. Never
//!   `CorsLayer::permissive()`.
//! - **Boot guard** ([`is_insecure_bind_configuration`] +
//!   [`is_insecure_tls_configuration`]) — pure decision functions; the
//!   binary refuses to start when `0.0.0.0` + `auth=off` coincide in a
//!   release build unless `FLUXION_REST_ALLOW_INSECURE=1` (#2505), and
//!   refuses `auth=tls` without a `FLUXION_REST_TRUSTED_PROXIES`
//!   allow-list in *every* build (#2754) so the verified-client header
//!   can never silently degrade to no-op.
//!
//! All primitives are pure-library code so they are unit-testable from
//! `cargo test -p fluxion --lib api` without spawning the binary.

use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
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

/// Default cap on the number of distinct per-IP token buckets kept in
/// memory at once (Issue #2688). The map is LRU-evicted at this cap so a
/// spoofed-IP or many-source flood cannot grow memory unboundedly. Each
/// entry is ~tens of bytes, so 100 000 entries is well under 10 MiB.
pub const DEFAULT_RATE_LIMIT_MAX_ENTRIES: usize = 100_000;

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
    /// verification header (see [`DEFAULT_VERIFIED_HEADER_NAME`]). The
    /// header is honoured **only** when the connection's socket peer is in
    /// `FLUXION_REST_TRUSTED_PROXIES` (Issue #2754); without that gate the
    /// header would be trivially spoofable. A valid bearer token is also
    /// accepted as a fallback so the same deployment can serve both
    /// proxied-mTLS and direct-token clients.
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
    /// Maximum distinct per-IP buckets retained (LRU-evicted at the cap).
    /// Bounds memory against spoofed-IP / many-IP floods (Issue #2688).
    pub rate_limit_max_entries: usize,
    /// Trusted reverse-proxy CIDR allow-list. When **empty** (the default),
    /// `X-Forwarded-For` / `X-Real-IP` are ignored and the rate limiter
    /// keys on the socket peer address only — closing the spoofing hole.
    /// When non-empty, the headers are honoured **only** for connections
    /// whose peer address falls inside one of these CIDRs, and the
    /// rightmost non-trusted hop is taken as the client IP (matching nginx
    /// `realip_recursive on` / Express `proxy-addr` semantics — *not* the
    /// spoofable leftmost entry).
    pub trusted_proxies: Vec<TrustedProxyCidr>,
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
            rate_limit_max_entries: DEFAULT_RATE_LIMIT_MAX_ENTRIES,
            trusted_proxies: Vec::new(),
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
    /// - `FLUXION_REST_RATE_LIMIT_MAX_ENTRIES` — max distinct per-IP
    ///   buckets retained (LRU-evicted; default 100 000). Bounds memory
    ///   against spoofed-IP / many-IP floods (Issue #2688).
    /// - `FLUXION_REST_TRUSTED_PROXIES` — comma-separated CIDR/IP
    ///   allow-list of trusted reverse proxies (e.g.
    ///   `10.0.0.0/8,192.0.2.1`). When **unset/empty** (default),
    ///   `X-Forwarded-For` / `X-Real-IP` are ignored and the limiter keys
    ///   on the socket peer only. When set, the headers are honoured only
    ///   for peers inside the list, taking the rightmost non-trusted hop.
    ///   Malformed entries are skipped with a warning.
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
        if let Ok(v) = std::env::var("FLUXION_REST_RATE_LIMIT_MAX_ENTRIES") {
            if let Ok(n) = v.parse::<usize>() {
                if n > 0 {
                    cfg.rate_limit_max_entries = n;
                }
            }
        }
        cfg.trusted_proxies = parse_trusted_proxies_from_env();
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
            trusted_proxies: self.trusted_proxies.clone(),
        }))
    }

    /// Build a fresh [`RateLimiter`] sized to this configuration.
    pub fn rate_limiter(&self) -> RateLimiter {
        RateLimiter::new(
            self.rate_limit_rps,
            self.rate_limit_burst,
            self.rate_limit_max_entries,
            &self.trusted_proxies,
        )
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
    /// Trusted reverse-proxy CIDR allow-list (Issue #2754). The mTLS
    /// verification header is honoured **only** when the socket peer of
    /// the connection falls inside one of these CIDRs — mirroring the
    /// rate limiter's (#2688) `X-Forwarded-For` gate. When empty (the
    /// default) the header is *never* trusted, so a direct client cannot
    /// spoof `x-verified-client: 1` to bypass TLS auth.
    trusted_proxies: Vec<TrustedProxyCidr>,
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
            //
            // Issue #2754: the verification header is trusted ONLY when the
            // connection's socket peer is itself a configured trusted proxy
            // (`FLUXION_REST_TRUSTED_PROXIES`). Without this gate any direct
            // client could spoof `x-verified-client: 1` and bypass TLS auth
            // entirely — a complete authentication bypass. When the peer is
            // unknown or untrusted (including the default
            // `trusted_proxies = []`, and any request lacking
            // `ConnectInfo`), the header is ignored and the request falls
            // through to the bearer-token check. The `off` and `token`
            // modes are untouched by this gate.
            let peer_is_trusted_proxy = req
                .extensions()
                .get::<ConnectInfo<SocketAddr>>()
                .map(|ci| peer_is_trusted(ci.0.ip(), &auth.0.trusted_proxies))
                .unwrap_or(false);
            if peer_is_trusted_proxy {
                let proxy_verified = req
                    .headers()
                    .get(&auth.0.verified_header_name)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s == auth.0.verified_header_value)
                    .unwrap_or(false);
                if proxy_verified {
                    return Ok(next.run(req).await);
                }
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
// Trusted-proxy CIDR allow-list (Issue #2688)
// =========================================================================

/// A single trusted-proxy entry — either a bare IP (`192.0.2.1`) or a CIDR
/// block (`10.0.0.0/8`, `::1/128`). Used to gate when
/// `X-Forwarded-For` / `X-Real-IP` may be honoured (Issue #2688): the
/// headers are trusted only for connections whose socket peer falls inside
/// one of the configured CIDRs.
#[derive(Clone, Debug)]
pub struct TrustedProxyCidr {
    network: IpAddr,
    prefix_len: u8,
}

impl TrustedProxyCidr {
    /// Parse a single `ip` or `ip/prefix` entry. Returns `Err` on a
    /// malformed value so the caller can skip-and-warn rather than panic.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let (net_str, prefix_str) = match s.split_once('/') {
            Some((n, p)) => (n, Some(p)),
            None => (s, None),
        };
        let network: IpAddr = net_str
            .parse()
            .map_err(|e| format!("invalid IP '{net_str}': {e}"))?;
        let max_len: u8 = match network {
            IpAddr::V4(_) => 32,
            IpAddr::V6(_) => 128,
        };
        let prefix_len = match prefix_str {
            None => max_len,
            Some(p) => {
                let n: u8 = p
                    .parse()
                    .map_err(|e| format!("invalid prefix '/{p}': {e}"))?;
                if n > max_len {
                    return Err(format!(
                        "prefix /{n} out of range for {network} (max /{max_len})"
                    ));
                }
                n
            }
        };
        // Normalise the network address to the masked base so containment
        // checks and equality behave intuitively (e.g. `10.1.2.3/8` acts
        // like `10.0.0.0/8`).
        let network = apply_prefix(network, prefix_len);
        Ok(Self {
            network,
            prefix_len,
        })
    }

    /// `true` iff `ip` (same address family) falls inside this CIDR.
    pub fn contains(&self, ip: IpAddr) -> bool {
        match (self.network, ip) {
            (IpAddr::V4(net), IpAddr::V4(addr)) => {
                let m = v4_mask(self.prefix_len);
                (m & u32::from(net)) == (m & u32::from(addr))
            }
            (IpAddr::V6(net), IpAddr::V6(addr)) => {
                let m = v6_mask(self.prefix_len);
                (m & u128::from(net)) == (m & u128::from(addr))
            }
            _ => false, // address-family mismatch
        }
    }
}

/// Network mask for an IPv4 `/prefix` (0 → all-zero, 32 → all-one).
fn v4_mask(prefix: u8) -> u32 {
    match prefix {
        0 => 0,
        n if n >= 32 => u32::MAX,
        n => u32::MAX << (32 - n),
    }
}

/// Network mask for an IPv6 `/prefix` (0 → all-zero, 128 → all-one).
fn v6_mask(prefix: u8) -> u128 {
    match prefix {
        0 => 0,
        n if n >= 128 => u128::MAX,
        n => u128::MAX << (128 - n),
    }
}

/// Zero out the host bits of `ip` for the given prefix length.
fn apply_prefix(ip: IpAddr, prefix: u8) -> IpAddr {
    match ip {
        IpAddr::V4(v4) => IpAddr::V4(Ipv4Addr::from(v4_mask(prefix) & u32::from(v4))),
        IpAddr::V6(v6) => IpAddr::V6(Ipv6Addr::from(v6_mask(prefix) & u128::from(v6))),
    }
}

/// Parse `FLUXION_REST_TRUSTED_PROXIES` (comma-separated CIDR/IP list),
/// skipping malformed entries with a warning. Factored out so both
/// [`RestSecurityConfig::from_env`] and the boot guard
/// ([`check_boot_guard_from_env`]) share a single parsing path and cannot
/// drift apart.
fn parse_trusted_proxies_from_env() -> Vec<TrustedProxyCidr> {
    std::env::var("FLUXION_REST_TRUSTED_PROXIES")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .filter_map(|s| match TrustedProxyCidr::parse(s) {
                    Ok(cidr) => Some(cidr),
                    Err(e) => {
                        tracing::warn!(
                            entry = s, error = %e,
                            "FLUXION_REST_TRUSTED_PROXIES: skipping malformed entry"
                        );
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `true` iff `ip` falls inside any of the configured trusted-proxy CIDRs.
fn peer_is_trusted(ip: IpAddr, trusted: &[TrustedProxyCidr]) -> bool {
    trusted.iter().any(|c| c.contains(ip))
}

// =========================================================================
// Per-IP token-bucket rate limiter
// =========================================================================

/// In-memory, per-IP token-bucket governor. Cheaply clonable (inner state
/// is behind an [`Arc`]); all clones share the same bucket map, so the
/// limit is enforced process-globally per resolved client IP. The bucket
/// map is bounded by an LRU cap (Issue #2688) so memory cannot grow
/// unboundedly under a spoofed-IP or many-source flood.
///
/// Concurrency (Issue #2894): the inner state is split across three locks
/// so the hot path — refill+decrement on an existing bucket — runs under
/// a [`parking_lot::RwLock`] *read* lock. The mutable fields of each
/// [`Bucket`] are atomic so the read-locked path can safely update token
/// counts via compare-and-swap. The LRU ordering (`BTreeMap<u64, IpAddr>`)
/// and the `next_seq` counter live in their own structures; `next_seq` is
/// an [`AtomicU64`] (lock-free `fetch_add`) and the LRU map is guarded by
/// a [`parking_lot::Mutex`] so its critical section stays small. Only
/// LRU eviction (rare, only when over the cap) takes the *write* lock on
/// `buckets`.
#[derive(Clone)]
pub struct RateLimiter(Arc<RateLimiterInner>);

struct RateLimiterInner {
    /// Per-IP token buckets. Read-locked for the common
    /// refill+decrement path so concurrent requests for distinct IPs
    /// proceed in parallel; write-locked only for new-bucket insertion
    /// and LRU eviction (Issue #2894).
    buckets: parking_lot::RwLock<HashMap<IpAddr, Bucket>>,
    /// `seq → ip` ordering for LRU eviction; smallest `seq` = LRU.
    /// Touched on every call (position refresh) but isolated behind
    /// its own mutex so its small critical section doesn't block
    /// readers of `buckets`.
    lru: parking_lot::Mutex<BTreeMap<u64, IpAddr>>,
    /// Monotonic counter, lock-free `fetch_add`. Starts at 1 so `0`
    /// remains the "no LRU entry yet" sentinel inside
    /// [`Bucket::lru_seq`].
    next_seq: AtomicU64,
    capacity: f64,
    refill_per_sec: f64,
    max_entries: usize,
    trusted_proxies: Vec<TrustedProxyCidr>,
}

/// Per-IP token bucket with interior mutability so that refill+consume
/// can happen under a *read* lock on the parent [`HashMap`] via
/// compare-and-swap. The `lru_seq` is updated outside the bucket map's
/// lock (under the LRU mutex) so it is read with [`Ordering::Relaxed`].
struct Bucket {
    /// Token count encoded as [`f64::to_bits`] for atomic update.
    tokens: AtomicU64,
    /// [`Instant`] of the most recent refill, encoded as a count of
    /// arbitrary monotonic ticks relative to a per-process epoch
    /// ([`RateLimiter::EPOCH`]). Compared with [`Ordering::Relaxed`].
    last_refill_ticks: AtomicU64,
    /// Current key of this IP in the LRU map, or `0` if not yet
    /// registered. Updated under the LRU mutex and read here with
    /// [`Ordering::Relaxed`].
    lru_seq: AtomicU64,
}

impl RateLimiter {
    /// Construct a limiter with `rps` sustained refill rate, `burst`
    /// capacity, a hard cap on distinct buckets (`max_entries`), and a
    /// trusted-proxy allow-list governing `X-Forwarded-For` handling (see
    /// [`crate::api::security`] module docs). `rps`/`burst` of `0` are
    /// clamped to `1`; `max_entries` of `0` is clamped to `1` so the map
    /// can always hold at least the active client.
    pub fn new(
        rps: u32,
        burst: u32,
        max_entries: usize,
        trusted_proxies: &[TrustedProxyCidr],
    ) -> Self {
        let capacity = burst.max(1) as f64;
        let refill_per_sec = rps.max(1) as f64;
        Self(Arc::new(RateLimiterInner {
            buckets: parking_lot::RwLock::new(HashMap::new()),
            lru: parking_lot::Mutex::new(BTreeMap::new()),
            next_seq: AtomicU64::new(1),
            capacity,
            refill_per_sec,
            max_entries: max_entries.max(1),
            trusted_proxies: trusted_proxies.to_vec(),
        }))
    }

    /// Per-process monotonic epoch so that [`Instant`]s stored in atomic
    /// [`Bucket::last_refill_ticks`] are non-negative. Subtracting two
    /// values gives an elapsed duration in arbitrary monotonic units
    /// (the units do not matter — only the *difference* does, and both
    /// sides use the same epoch).
    fn epoch() -> Instant {
        // Lazily initialise the epoch once. `OnceLock` is already used
        // elsewhere in the crate; no new dependency introduced.
        use std::sync::OnceLock;
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        *EPOCH.get_or_init(Instant::now)
    }

    /// Convert an [`Instant`] to its monotonic tick count relative to
    /// [`RateLimiter::epoch`]. Saturating at `u64::MAX` so a clock
    /// anomaly cannot wrap to 0.
    fn ticks(now: Instant) -> u64 {
        now.saturating_duration_since(Self::epoch())
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64
    }

    /// Attempt to consume one token for `ip`. Returns `true` if the
    /// request is allowed, `false` if the bucket is empty (caller should
    /// respond `429`). Lazily refills based on wall-clock elapsed time and
    /// LRU-evicts the least-recently-touched bucket once the map exceeds
    /// `max_entries`, bounding memory (Issue #2688).
    ///
    /// Concurrency (Issue #2894): the hot path — refill+consume on an
    /// existing bucket — runs under a *read* lock on the bucket map
    /// using atomic CAS on the per-bucket token count. Multiple
    /// concurrent requests for distinct IPs proceed in parallel. Only
    /// new-bucket insertion and LRU eviction take the *write* lock.
    pub fn try_acquire(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let now_ticks = Self::ticks(now);
        let capacity = self.0.capacity;
        let capacity_bits = capacity.to_bits();
        let refill_per_sec = self.0.refill_per_sec;
        let max_entries = self.0.max_entries;

        // 1. Lock-free seq allocation. Monotonic across threads so the
        // LRU map's total order is preserved (Issue #2688).
        let seq = self.0.next_seq.fetch_add(1, Ordering::Relaxed);

        // 2. Hot path: existing bucket under a read lock. Token refill
        // and consume happen via compare-and-swap on the bucket's atomic
        // fields, so the read-locked path is safe to mutate them.
        let allowed = {
            let wait_start = Instant::now();
            let buckets = self.0.buckets.read();
            record_lock_wait(LOCK_KIND_READ, wait_start.elapsed());

            if let Some(bucket) = buckets.get(&ip) {
                refill_and_consume(bucket, now_ticks, capacity, refill_per_sec)
            } else {
                // Signal "new IP" to the caller via a sentinel — the
                // bucket does not exist under the read lock, so we
                // must take the write lock to insert it.
                drop(buckets);
                return self.try_acquire_insert(
                    ip,
                    seq,
                    now_ticks,
                    capacity,
                    capacity_bits,
                    refill_per_sec,
                    max_entries,
                );
            }
        };

        // 3. Update LRU position under the LRU mutex. Brief critical
        // section; readers of `buckets` are unaffected. Also update the
        // bucket's `lru_seq` atomically so the next call observes the
        // new key — this is essential for bit-identical LRU semantics
        // (Issue #2688): the bucket must always carry the current key
        // it sits under in the LRU map.
        // Read the bucket's prior `lru_seq` *before* acquiring the LRU mutex so
        // we don't hold the LRU mutex while waiting on the bucket map's
        // read lock — otherwise a thread inside `try_acquire_insert`
        // holding `buckets.write()` and waiting on the LRU mutex would
        // deadlock against us holding LRU and waiting on `buckets.read()`.
        let observed_old_seq: u64 = {
            let wait_start = Instant::now();
            let buckets = self.0.buckets.read();
            record_lock_wait(LOCK_KIND_READ, wait_start.elapsed());
            buckets
                .get(&ip)
                .map(|b| b.lru_seq.load(Ordering::Relaxed))
                .unwrap_or(0)
        };
        let wait_start = Instant::now();
        {
            let mut lru_guard = self.0.lru.lock();
            record_lock_wait(LOCK_KIND_LRU, wait_start.elapsed());
            if observed_old_seq != 0 {
                // Idempotent: if the bucket was evicted between our
                // observed read and the LRU lock acquisition, the seq is
                // no longer present (the eviction loop cleans up the LRU
                // entry before dropping the bucket). Removing a missing
                // entry is a no-op.
                lru_guard.remove(&observed_old_seq);
            }
            lru_guard.insert(seq, ip);
        }
        // Reflect the new seq in the bucket so the next LRU update can
        // remove `seq` (rather than a stale seq). `AtomicU64::store` is
        // safe to call under any lock — interior mutability.
        let wait_start = Instant::now();
        {
            let buckets = self.0.buckets.read();
            record_lock_wait(LOCK_KIND_READ, wait_start.elapsed());
            if let Some(bucket) = buckets.get(&ip) {
                bucket.lru_seq.store(seq, Ordering::Relaxed);
            }
        }

        allowed
    }

    /// New-IP insertion path (Issue #2894). Takes the write lock on
    /// `buckets`, inserts a fresh bucket if absent, performs the
    /// refill+consume under the write lock, registers the LRU position,
    /// and runs eviction if the map is over the cap. Returns the
    /// allow/deny decision.
    #[allow(clippy::too_many_arguments)]
    fn try_acquire_insert(
        &self,
        ip: IpAddr,
        seq: u64,
        now_ticks: u64,
        capacity: f64,
        capacity_bits: u64,
        refill_per_sec: f64,
        max_entries: usize,
    ) -> bool {
        let wait_start = Instant::now();
        let mut buckets = self.0.buckets.write();
        record_lock_wait(LOCK_KIND_WRITE, wait_start.elapsed());

        // Re-check: another thread may have inserted in between the
        // read-locked peek and the write lock acquisition.
        let bucket = buckets.entry(ip).or_insert_with(|| Bucket {
            tokens: AtomicU64::new(capacity_bits),
            last_refill_ticks: AtomicU64::new(now_ticks),
            lru_seq: AtomicU64::new(0),
        });
        let existing_lru_seq = bucket.lru_seq.load(Ordering::Relaxed);

        // Refill + consume under the write lock (interior mutability
        // is unnecessary here since we have write access).
        let elapsed_ticks =
            now_ticks.saturating_sub(bucket.last_refill_ticks.load(Ordering::Relaxed));
        let elapsed = elapsed_ticks as f64 / 1e9;
        let mut tokens = f64::from_bits(bucket.tokens.load(Ordering::Relaxed));
        if elapsed > 0.0 {
            tokens = (tokens + elapsed * refill_per_sec).min(capacity);
            bucket.tokens.store(tokens.to_bits(), Ordering::Relaxed);
            bucket.last_refill_ticks.store(now_ticks, Ordering::Relaxed);
        }
        let allowed = if tokens >= 1.0 {
            tokens -= 1.0;
            bucket.tokens.store(tokens.to_bits(), Ordering::Relaxed);
            bucket.last_refill_ticks.store(now_ticks, Ordering::Relaxed);
            true
        } else {
            false
        };

        // Register LRU position (under the LRU mutex, briefly).
        {
            let wait_start = Instant::now();
            let mut lru_guard = self.0.lru.lock();
            record_lock_wait(LOCK_KIND_LRU, wait_start.elapsed());
            if existing_lru_seq != 0 {
                lru_guard.remove(&existing_lru_seq);
            }
            lru_guard.insert(seq, ip);
            bucket.lru_seq.store(seq, Ordering::Relaxed);
        }

        // Bounded eviction (rare path; only when over cap). Drops the
        // LRU's oldest entry until we are at/under the cap. Preserves
        // the active client (`ip`) by skipping self-eviction (the cap
        // is clamped to >=1 so we can always hold the active client).
        //
        // Concurrency note (Issue #2894): between peeking the LRU's
        // oldest entry and removing it from `buckets`, another thread
        // may have already evicted the same IP. In that case
        // `buckets.remove(&oldest_ip)` is a no-op and `buckets.len()`
        // does not shrink, which would loop forever — so we explicitly
        // clean up the stale LRU entry whenever the bucket is absent
        // and continue.
        while buckets.len() > max_entries {
            // Peek the oldest LRU entry under the LRU mutex. Release
            // before any further work to keep the critical section
            // small.
            let oldest = {
                let wait_start = Instant::now();
                let lru_guard = self.0.lru.lock();
                record_lock_wait(LOCK_KIND_LRU, wait_start.elapsed());
                lru_guard.iter().next().map(|(&k, &v)| (k, v))
            };
            match oldest {
                Some((_oldest_seq, oldest_ip)) if oldest_ip == ip => break,
                Some((oldest_seq, oldest_ip)) => {
                    // Remove the stale LRU entry and the bucket. If
                    // `buckets.remove` is a no-op (the bucket was
                    // already evicted by another thread), the next
                    // loop iteration will pick the *next* LRU entry.
                    {
                        let wait_start = Instant::now();
                        let mut lru_guard = self.0.lru.lock();
                        record_lock_wait(LOCK_KIND_LRU, wait_start.elapsed());
                        // Idempotent: only remove if the seq still maps
                        // to this IP (a concurrent refresh may have
                        // reassigned it).
                        if lru_guard.get(&oldest_seq).copied() == Some(oldest_ip) {
                            lru_guard.remove(&oldest_seq);
                        }
                    }
                    buckets.remove(&oldest_ip);
                }
                None => break,
            }
        }

        allowed
    }

    /// Current number of distinct per-IP buckets retained. Bounded by the
    /// configured `max_entries`; exposed for observability and tests.
    pub fn num_entries(&self) -> usize {
        let wait_start = Instant::now();
        let buckets = self.0.buckets.read();
        record_lock_wait(LOCK_KIND_READ, wait_start.elapsed());
        buckets.len()
    }
}

/// Histogram name (Issue #2894) recording the wall-clock duration the
/// [`RateLimiter`] spent waiting to acquire its internal locks per
/// `try_acquire` call. Labelled by `kind`:
/// - `"read"` — read lock on the per-IP bucket map.
/// - `"write"` — write lock on the per-IP bucket map (rare; new-IP
///   insertion or LRU eviction).
/// - `"lru"` — mutex on the LRU ordering map (touched on every call).
pub const RATE_LIMIT_LOCK_WAIT_SECONDS: &str = "fluxion_rate_limit_lock_wait_seconds";

const LOCK_KIND_READ: &str = "read";
const LOCK_KIND_WRITE: &str = "write";
const LOCK_KIND_LRU: &str = "lru";

/// Record one observation of the lock-wait histogram. Cheap; the
/// [`metrics`] crate short-circuits when no recorder is installed (e.g.
/// in unit tests without `init_recorder`), so this is safe to call from
/// every [`RateLimiter::try_acquire`] without measurable overhead.
#[inline]
fn record_lock_wait(kind: &'static str, wait: std::time::Duration) {
    metrics::histogram!(RATE_LIMIT_LOCK_WAIT_SECONDS, "kind" => kind).record(wait.as_secs_f64());
}

/// Refill + consume one token from an *existing* bucket under a read
/// lock on the parent [`HashMap`]. Mutates the bucket's atomic fields
/// via compare-and-swap. Returns `true` if the request is allowed,
/// `false` if the bucket is empty.
///
/// [`HashMap`]: std::collections::HashMap
fn refill_and_consume(bucket: &Bucket, now_ticks: u64, capacity: f64, refill_per_sec: f64) -> bool {
    // Read current state.
    let old_tokens_bits = bucket.tokens.load(Ordering::Relaxed);
    let old_tokens = f64::from_bits(old_tokens_bits);
    let last_refill_ticks = bucket.last_refill_ticks.load(Ordering::Relaxed);
    let elapsed_ticks = now_ticks.saturating_sub(last_refill_ticks);
    let elapsed = elapsed_ticks as f64 / 1e9;

    // Compute the refilled value (still in local memory). The CAS
    // below operates against `old_tokens_bits` (the *previous* stored
    // value), NOT the refilled value — combining refill + consume into
    // a single atomic transition matches the original `bucket.tokens -= 1.0`
    // mutation semantics.
    let refilled = if elapsed > 0.0 {
        (old_tokens + elapsed * refill_per_sec).min(capacity)
    } else {
        old_tokens
    };

    if refilled < 1.0 {
        // Fast rejection: bucket exists but is empty after refill.
        // Best-effort update the refill bookkeeping so the next call
        // doesn't have to recompute the elapsed portion. CAS failure
        // is benign — another thread already updated `tokens` to a
        // value ≥ ours.
        let _ = bucket.tokens.compare_exchange(
            old_tokens_bits,
            refilled.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        let _ = bucket.last_refill_ticks.compare_exchange(
            last_refill_ticks,
            now_ticks,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        return false;
    }

    // Refill + consume in one CAS: replace the old (possibly stale)
    // token count with the post-refill, post-consume value. This
    // matches the semantics of the original `bucket.tokens -= 1.0`
    // mutation but is safe to perform under a *read* lock on the
    // parent map because the CAS is atomic.
    let after_consume = refilled - 1.0;
    let new_bits = after_consume.to_bits();
    match bucket.tokens.compare_exchange(
        old_tokens_bits,
        new_bits,
        Ordering::Relaxed,
        Ordering::Relaxed,
    ) {
        Ok(_) => {
            bucket.last_refill_ticks.store(now_ticks, Ordering::Relaxed);
            true
        }
        Err(actual_bits) => {
            // Another thread changed `tokens` between our load and CAS.
            // Retry once with the freshly observed value before giving
            // up. We compute the refill from the *new* `last_refill_ticks`
            // value because the previous one is now stale.
            let actual = f64::from_bits(actual_bits);
            let new_last_refill = bucket.last_refill_ticks.load(Ordering::Relaxed);
            let new_elapsed_ticks = now_ticks.saturating_sub(new_last_refill);
            let new_elapsed = new_elapsed_ticks as f64 / 1e9;
            let new_refilled = if new_elapsed > 0.0 {
                (actual + new_elapsed * refill_per_sec).min(capacity)
            } else {
                actual
            };
            if new_refilled < 1.0 {
                return false;
            }
            match bucket.tokens.compare_exchange(
                actual_bits,
                (new_refilled - 1.0).to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    bucket.last_refill_ticks.store(now_ticks, Ordering::Relaxed);
                    true
                }
                Err(_) => {
                    // Lost the race after one retry — caller can retry
                    // on the next request. Returning `false` is the
                    // safe side: rate-limiting prefers false-negatives
                    // over false-positives when overloaded.
                    false
                }
            }
        }
    }
}

/// axum middleware enforcing [`RateLimiter`] per resolved client IP.
///
/// The client IP is resolved via [`client_ip`] using the limiter's trusted
/// proxy allow-list. By default (no trusted proxies configured) only the
/// socket peer address is used and `X-Forwarded-For` / `X-Real-IP` are
/// ignored, so a client cannot spoof a fresh bucket per request (Issue
/// #2688). When no IP can be determined the request is allowed through
/// (fail-open) — the body limit and timeout still bound worst-case
/// behaviour.
pub async fn rate_limit_middleware(
    State(limiter): State<RateLimiter>,
    req: Request,
    next: Next,
) -> Response {
    match client_ip(&req, &limiter.0.trusted_proxies) {
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

/// Resolve the client IP for rate-limiting purposes (Issue #2688).
///
/// Resolution order:
/// 1. The axum `ConnectInfo<SocketAddr>` socket peer address is the
///    **primary** client identity. This is always trusted (it comes from
///    the kernel, not a header).
/// 2. `X-Forwarded-For` / `X-Real-IP` are consulted **only** when
///    `trusted_proxies` is non-empty **and** the socket peer itself falls
///    inside one of the configured CIDRs. In that case the **rightmost**
///    hop in `X-Forwarded-For` that is not itself a trusted proxy is
///    returned (nginx `realip_recursive on` / Express `proxy-addr`
///    semantics). The leftmost entry is deliberately *not* used — it is
///    trivially spoofable by the client.
/// 3. If the peer is unknown and no headers are trusted, returns `None`.
///
/// When `trusted_proxies` is empty (the default), headers are never read,
/// closing the spoofing hole regardless of what a client sends.
pub fn client_ip(req: &Request, trusted_proxies: &[TrustedProxyCidr]) -> Option<IpAddr> {
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip());

    // No trusted-proxy configuration → key strictly on the socket peer.
    // Headers are never consulted, so they cannot grant a fresh bucket.
    if trusted_proxies.is_empty() {
        return peer;
    }

    // Trusted proxies are configured: honour headers *only* when the peer
    // itself is a trusted proxy. A direct (non-proxy) connection still keys
    // on its real peer address — an attacker cannot bypass by pretending to
    // be behind a proxy.
    let peer = peer?;
    if !peer_is_trusted(peer, trusted_proxies) {
        return Some(peer);
    }

    // Peer is a trusted proxy → resolve the real client from the headers.
    if let Some(ip) = forwarded_client_ip(req, trusted_proxies) {
        return Some(ip);
    }

    // Header was absent/malformed: fall back to the trusted peer rather
    // than failing open with `None` (which would skip limiting entirely).
    Some(peer)
}

/// Extract the real client IP from `X-Forwarded-For` (rightmost
/// non-trusted hop) falling back to `X-Real-IP`. Both are only meaningful
/// once the peer has been verified as a trusted proxy.
fn forwarded_client_ip(req: &Request, trusted: &[TrustedProxyCidr]) -> Option<IpAddr> {
    if let Some(xff) = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        // Rightmost non-trusted hop. Walking right→left skips the chain of
        // trusted proxies that appended themselves; the first untrusted
        // address is the originating client. (The leftmost entry is
        // spoofable and intentionally not used.)
        let hops: Vec<Option<IpAddr>> = xff
            .split(',')
            .map(|s| s.trim().parse::<IpAddr>().ok())
            .collect();
        for hop in hops.into_iter().rev().flatten() {
            if !peer_is_trusted(hop, trusted) {
                return Some(hop);
            }
        }
    }
    // `X-Real-IP` is typically set by the immediate proxy to the client it
    // saw; since the peer is already trusted, honour it directly.
    if let Some(xri) = req.headers().get("x-real-ip").and_then(|v| v.to_str().ok()) {
        if let Ok(ip) = xri.trim().parse::<IpAddr>() {
            return Some(ip);
        }
    }
    None
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

/// Pure decision function for the TLS boot guard (Issue #2754).
///
/// Returns `true` when `FLUXION_REST_AUTH=tls` is selected but no trusted
/// reverse-proxy CIDR allow-list (`FLUXION_REST_TRUSTED_PROXIES`) has been
/// configured. In that state the proxy-verified mTLS header can never be
/// honoured — the [`require_auth`] gate refuses it for any non-trusted
/// peer, and the default empty list trusts *no* peer — so `tls` mode
/// silently degrades to token-only auth. That is the precise
/// misconfiguration that made the #2754 bypass possible: an operator who
/// selected `tls` expecting mTLS-header protection in fact had none.
///
/// Refusing to boot forces the operator to either configure the
/// trusted-proxy allow-list (enabling the header path) or switch to the
/// cheaper `token` mode, eliminating the silent degradation. Mirrors the
/// #2689 fail-closed posture (applies in **every** build, not only
/// release); `FLUXION_REST_ALLOW_INSECURE=1` is honoured as an explicit
/// opt-out for local dev / test, for parity with the sibling
/// [`is_insecure_bind_configuration`] guard.
pub fn is_insecure_tls_configuration(
    auth_mode: AuthMode,
    trusted_proxies: &[TrustedProxyCidr],
    allow_insecure: bool,
) -> bool {
    if allow_insecure {
        return false;
    }
    if auth_mode != AuthMode::Tls {
        return false;
    }
    trusted_proxies.is_empty()
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

    // `FLUXION_REST_ALLOW_INSECURE` is the shared opt-out for both boot
    // guards. Read once here so every build path uses the same value.
    let allow_insecure = std::env::var("FLUXION_REST_ALLOW_INSECURE")
        .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);

    // Issue #2754 — fail-closed when `tls` mode is selected with no
    // trusted-proxy allow-list: the verified-client header could never be
    // safely honoured (the auth gate refuses it for every non-trusted
    // peer, and the default empty list trusts none), so `tls` mode would
    // silently degrade to token-only auth — the exact misconfiguration
    // behind the bypass. Applies in every build (mirrors #2689); the
    // shared `allow_insecure` flag is the explicit opt-out.
    let trusted_proxies = parse_trusted_proxies_from_env();
    if is_insecure_tls_configuration(auth, &trusted_proxies, allow_insecure) {
        return Err(
            "fluxion-rest: refusing to boot — FLUXION_REST_AUTH=tls requires a non-empty \
             FLUXION_REST_TRUSTED_PROXIES allow-list so the proxy-verified mTLS header can be \
             honoured safely (Issue #2754). Set FLUXION_REST_TRUSTED_PROXIES to your reverse \
             proxy's CIDR (e.g. 10.0.0.0/8), switch to FLUXION_REST_AUTH=token, or set \
             FLUXION_REST_ALLOW_INSECURE=1 to explicitly opt out."
                .to_string(),
        );
    }

    // Release-only insecure-bind guard (Issue #2505). In debug builds the
    // default `0.0.0.0` + `off` combination must stay usable for
    // `cargo run`.
    #[cfg(not(debug_assertions))]
    {
        let bind = std::env::var("FLUXION_REST_BIND").unwrap_or_default();
        if is_insecure_bind_configuration(&bind, auth, allow_insecure) {
            return Err(format!(
                "fluxion-rest: refusing to boot — FLUXION_REST_BIND='{bind}' binds all interfaces \
                 while FLUXION_REST_AUTH=off. Set FLUXION_REST_AUTH=token (with \
                 FLUXION_REST_AUTH_TOKEN) or FLUXION_REST_AUTH=tls, bind to 127.0.0.1, or set \
                 FLUXION_REST_ALLOW_INSECURE=1 to explicitly opt in."
            ));
        }
    }
    // In debug the release-only insecure-bind guard above is skipped; the
    // env read stays "used". `auth`, `trusted_proxies`, and `allow_insecure`
    // are consumed by the #2754 TLS guard above in every build.
    #[cfg(debug_assertions)]
    {
        let _ = std::env::var("FLUXION_REST_BIND");
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
        let limiter = RateLimiter::new(0, 5, DEFAULT_RATE_LIMIT_MAX_ENTRIES, &[]);
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
        let limiter = RateLimiter::new(10, 1, DEFAULT_RATE_LIMIT_MAX_ENTRIES, &[]);
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
        let limiter = RateLimiter::new(0, 1, DEFAULT_RATE_LIMIT_MAX_ENTRIES, &[]);
        let a: IpAddr = "10.0.0.3".parse().unwrap();
        let b: IpAddr = "10.0.0.4".parse().unwrap();
        assert!(limiter.try_acquire(a));
        assert!(limiter.try_acquire(b), "different IP has its own bucket");
        assert!(!limiter.try_acquire(a));
    }

    // ---- Issue #2688: bounded LRU eviction ----

    #[test]
    fn rate_limiter_evicts_at_cap() {
        // Small cap; insert far more distinct IPs and assert the map stays
        // bounded at exactly the cap.
        let cap = 64usize;
        let limiter = RateLimiter::new(0, 1, cap, &[]);
        for i in 0..(cap * 4) as u32 {
            let o = i % 256;
            let ip: IpAddr = format!("10.{i}.{o}.1").parse().unwrap();
            assert!(
                limiter.try_acquire(ip),
                "first acquire of a fresh IP is allowed"
            );
            // Size must never exceed the cap by more than the single entry
            // we are mid-insert on (eviction runs at the end of try_acquire,
            // so after return it is always <= cap).
            assert!(
                limiter.num_entries() <= cap,
                "map grew to {} > cap {cap}",
                limiter.num_entries()
            );
        }
        assert_eq!(
            limiter.num_entries(),
            cap,
            "after inserting well past the cap the map should sit exactly at the cap"
        );
    }

    #[test]
    fn rate_limiter_lru_keeps_hot_ip() {
        // A frequently-renewed IP must survive eviction of older, cold IPs
        // even when the map is pinned at the cap.
        let cap = 8usize;
        let limiter = RateLimiter::new(0, 1, cap, &[]);
        let hot: IpAddr = "203.0.113.1".parse().unwrap();
        // Prime the hot bucket (burst=1 → now empty).
        assert!(limiter.try_acquire(hot));
        // Flood with `cap` distinct cold IPs, but touch `hot` each iteration
        // so it stays MRU and is never the eviction candidate. The renewing
        // touch ignores the (empty) token — LRU position refreshes regardless.
        for i in 0..cap as u32 {
            let _ = limiter.try_acquire(hot); // touch; bool intentionally unused
            let cold: IpAddr = format!("198.51.100.{i}").parse().unwrap();
            let _ = limiter.try_acquire(cold);
            assert!(limiter.num_entries() <= cap);
        }
        // Discrimination: if `hot` survived, its bucket is still empty
        // (burst drained, no time to refill) → the next acquire is rejected.
        // If `hot` had been evicted, this would mint a fresh full bucket and
        // be allowed — failing the assertion.
        assert!(
            !limiter.try_acquire(hot),
            "hot IP's bucket must have survived eviction"
        );
    }

    // ---- Issue #2894: concurrent throughput on the split-locks path ----

    /// Issue #2894 acceptance criterion: `RateLimiter::try_acquire` must
    /// handle at least 100 concurrent calls to *distinct* IPs without
    /// regression versus the previous single-mutex implementation. With
    /// the new read-locked hot path, distinct-IP calls should not
    /// serialise on each other; the assertion here is that they all
    /// succeed (each gets its own freshly-minted full bucket) and the
    /// internal map sits at exactly the number of distinct IPs.
    #[test]
    fn rate_limiter_concurrent_distinct_ips_all_allowed() {
        use std::sync::Arc;
        use std::thread;

        let limiter = Arc::new(RateLimiter::new(
            1000,
            5,
            DEFAULT_RATE_LIMIT_MAX_ENTRIES,
            &[],
        ));
        // 100 distinct IPs, fired in parallel. Burst capacity is 5
        // and refill rate is 1000/s, so each IP gets a full bucket
        // and its first acquire always succeeds.
        const N: usize = 100;
        let barrier = Arc::new(std::sync::Barrier::new(N));
        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let limiter = limiter.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                let ip: IpAddr = format!("10.0.{}.1", i).parse().unwrap();
                barrier.wait();
                limiter.try_acquire(ip)
            }));
        }
        let mut allowed = 0usize;
        for h in handles {
            if h.join().expect("thread join") {
                allowed += 1;
            }
        }
        assert_eq!(
            allowed, N,
            "every distinct-IP first acquire should be allowed under burst=5"
        );
        assert_eq!(
            limiter.num_entries(),
            N,
            "limiter should hold exactly one bucket per distinct IP"
        );
    }

    /// Issue #2894: 1000 concurrent requests to distinct IPs must show
    /// ≥20% throughput improvement over the previous single-mutex
    /// implementation. The acceptance criterion is relative: this test
    /// only verifies the absolute lower-bound that the new design
    /// targets (≥1000 acquires completed in ≤200 ms on a CI-class
    /// machine, which is roughly the throughput ceiling the old
    /// `parking_lot::Mutex` could sustain). It is intentionally
    /// generous so it stays stable on shared runners; the actual
    /// improvement ratio is measured in `tests/api_concurrent_throughput.rs`
    /// (extended with a 1000-client case).
    #[test]
    fn rate_limiter_concurrent_thousand_distinct_ips_under_budget() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Instant;

        let limiter = Arc::new(RateLimiter::new(
            100_000, // effectively infinite burst for this test
            10_000,
            DEFAULT_RATE_LIMIT_MAX_ENTRIES,
            &[],
        ));
        const N: usize = 1000;
        let barrier = Arc::new(std::sync::Barrier::new(N));
        let started = Instant::now();
        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let limiter = limiter.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                let ip: IpAddr = format!("172.16.{}.{}", i / 256, i % 256).parse().unwrap();
                barrier.wait();
                limiter.try_acquire(ip)
            }));
        }
        let mut allowed = 0usize;
        for h in handles {
            if h.join().expect("thread join") {
                allowed += 1;
            }
        }
        let elapsed = started.elapsed();
        assert_eq!(
            allowed, N,
            "every distinct-IP first acquire should be allowed under burst=10000"
        );
        // Generous bound for debug-mode CI runners. The acceptance
        // criterion is "≥20% improvement" — we measure that
        // relatively in `tests/api_concurrent_throughput.rs`. Here we
        // just verify the absolute throughput is at least one order
        // of magnitude better than per-call sequential (each try_acquire
        // is a few hundred ns under no contention; 1000 of them in
        // serial would still be sub-millisecond, but contended
        // acquisition against a single mutex would push it well past
        // this bound). The bound here is intentionally loose to absorb
        // debug-mode CI runner noise (1k thread spawn + barrier sync
        // alone can take 100-300ms on a shared 2-vCPU runner); the
        // tighter, relative throughput assertion lives in
        // `tests/api_concurrent_throughput.rs`.
        assert!(
            elapsed < std::time::Duration::from_millis(1500),
            "1000 concurrent distinct-IP acquires took {elapsed:?} — \
             lock-induced serialisation may have returned (Issue #2894)"
        );
    }

    /// Issue #2894: `RateLimiter::num_entries` is now read-locked, so
    /// concurrent readers must not block each other. Verify by
    /// hammering `num_entries` from many threads while a single
    /// writer thread inserts new buckets; the read side must keep
    /// making progress (no panics, no deadlock).
    #[test]
    fn rate_limiter_num_entries_concurrent_readers() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let limiter = Arc::new(RateLimiter::new(
            1000,
            100,
            DEFAULT_RATE_LIMIT_MAX_ENTRIES,
            &[],
        ));
        // Writer thread: churn through distinct IPs to grow the map.
        let writer = {
            let limiter = limiter.clone();
            thread::spawn(move || {
                for i in 0..500u32 {
                    // Build a 4-octet IP from the index so we never
                    // exceed 255 in any octet.
                    let o1 = (i >> 8) & 0xff;
                    let o2 = i & 0xff;
                    let ip: IpAddr = std::net::Ipv4Addr::new(10, 20, o1 as u8, o2 as u8).into();
                    let _ = limiter.try_acquire(ip);
                }
            })
        };
        // Reader threads: spin on num_entries. None must deadlock or
        // panic; all must terminate within a reasonable bound.
        let mut readers = Vec::new();
        for _ in 0..8 {
            let limiter = limiter.clone();
            readers.push(thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_millis(250);
                while Instant::now() < deadline {
                    let _ = limiter.num_entries();
                }
            }));
        }
        for r in readers {
            r.join().expect("reader thread join");
        }
        writer.join().expect("writer thread join");
    }

    // ---- Issue #2688: client_ip resolution (no blind XFF trust) ----

    /// Build a `Request` carrying an injected `ConnectInfo<SocketAddr>` peer
    /// (mirrors what `into_make_service_with_connect_info` wires in
    /// production) plus optional headers.
    fn req_with_peer(peer: SocketAddr, xff: Option<&str>, x_real_ip: Option<&str>) -> Request {
        let mut builder = Request::builder().method(Method::GET).uri("/v1/healthz");
        builder = builder.extension(ConnectInfo(peer));
        if let Some(v) = xff {
            builder = builder.header("x-forwarded-for", v);
        }
        if let Some(v) = x_real_ip {
            builder = builder.header("x-real-ip", v);
        }
        builder.body(axum::body::Body::empty()).unwrap()
    }

    #[test]
    fn client_ip_ignores_xff_by_default() {
        // Default (no trusted proxies): XFF/X-Real-IP must be IGNORED so a
        // client rotating a spoofed XFF cannot get a fresh bucket per
        // request. Only the socket peer is used.
        let peer: SocketAddr = "198.51.100.7:4000".parse().unwrap();
        let req = req_with_peer(peer, Some("1.1.1.1, 2.2.2.2"), Some("3.3.3.3"));
        assert_eq!(
            client_ip(&req, &[]),
            Some(peer.ip()),
            "default must resolve to the socket peer, ignoring spoofed headers"
        );
    }

    #[test]
    fn client_ip_rotating_xff_does_not_grant_fresh_bucket() {
        // Issue #2688 (1): the same peer sending a different spoofed XFF on
        // every request resolves to the SAME client IP, so it shares one
        // token bucket and is throttled like any single client.
        let limiter = RateLimiter::new(0, 2, DEFAULT_RATE_LIMIT_MAX_ENTRIES, &[]);
        let peer: SocketAddr = "198.51.100.7:4000".parse().unwrap();
        for i in 0..5u32 {
            let xff = format!("{i}.{i}.{i}.{i}");
            let req = req_with_peer(peer, Some(&xff), None);
            let ip = client_ip(&req, &[]).expect("peer resolves");
            assert_eq!(ip, peer.ip(), "spoofed XFF must not change the resolved IP");
            let _ = limiter.try_acquire(ip);
        }
        // Two acquires were allowed (burst=2); the remaining three drained
        // the same bucket → it is now empty.
        assert_eq!(limiter.num_entries(), 1, "all requests shared one bucket");
        assert!(
            !limiter.try_acquire(peer.ip()),
            "bucket must be empty after draining"
        );
    }

    #[test]
    fn client_ip_trusted_proxy_resolves_from_xff() {
        // Trusted-proxy mode: the peer is a trusted proxy, so the rightmost
        // non-trusted XFF hop is the real client.
        let trusted = vec![
            TrustedProxyCidr::parse("10.0.0.0/8").unwrap(),
            TrustedProxyCidr::parse("192.0.2.1").unwrap(),
        ];
        let proxy_peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        // XFF chain: spoofed-left, real-client, trusted-proxy. The
        // rightmost non-trusted hop is the real client (203.0.113.9), NOT
        // the spoofed leftmost entry.
        let req = req_with_peer(
            proxy_peer,
            Some("198.51.100.99, 203.0.113.9, 10.1.2.3"),
            None,
        );
        assert_eq!(
            client_ip(&req, &trusted),
            Some("203.0.113.9".parse::<IpAddr>().unwrap()),
            "trusted-proxy mode must take the rightmost non-trusted XFF hop"
        );
    }

    #[test]
    fn client_ip_trusted_proxy_falls_back_to_x_real_ip() {
        let trusted = vec![TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        let proxy_peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let req = req_with_peer(proxy_peer, None, Some("203.0.113.42"));
        assert_eq!(
            client_ip(&req, &trusted),
            Some("203.0.113.42".parse::<IpAddr>().unwrap()),
        );
    }

    #[test]
    fn client_ip_untrusted_peer_keys_on_peer_not_xff() {
        // Even with trusted proxies configured, a peer that is NOT in the
        // allow-list keys on its own address and its XFF is ignored — an
        // attacker cannot bypass by sending an XFF that names a trusted
        // proxy.
        let trusted = vec![TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        let direct_peer: SocketAddr = "203.0.113.50:6000".parse().unwrap();
        let req = req_with_peer(direct_peer, Some("10.1.2.3, 198.51.100.1"), None);
        assert_eq!(
            client_ip(&req, &trusted),
            Some(direct_peer.ip()),
            "untrusted peer must key on itself, ignoring XFF entirely"
        );
    }

    #[test]
    fn client_ip_no_connect_info_no_trust_returns_none() {
        // No peer available (e.g. test harness without ConnectInfo) and no
        // trusted proxies → None (middleware fails open; body/timeout bound).
        let req = Request::builder()
            .method(Method::GET)
            .uri("/v1/healthz")
            .header("x-forwarded-for", "1.2.3.4")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(client_ip(&req, &[]), None);
    }

    // ---- TrustedProxyCidr parsing / matching ----

    #[test]
    fn trusted_proxy_cidr_parse_and_contains() {
        let c = TrustedProxyCidr::parse("10.0.0.0/8").unwrap();
        assert!(c.contains("10.255.255.255".parse::<IpAddr>().unwrap()));
        assert!(!c.contains("11.0.0.0".parse::<IpAddr>().unwrap()));
        // Bare IP → /32.
        let c = TrustedProxyCidr::parse("192.0.2.1").unwrap();
        assert_eq!(c.prefix_len, 32);
        assert!(c.contains("192.0.2.1".parse::<IpAddr>().unwrap()));
        assert!(!c.contains("192.0.2.2".parse::<IpAddr>().unwrap()));
        // IPv6.
        let c = TrustedProxyCidr::parse("2001:db8::/32").unwrap();
        assert!(c.contains("2001:db8:ffff:ffff::1".parse::<IpAddr>().unwrap()));
        assert!(!c.contains("2001:db9::1".parse::<IpAddr>().unwrap()));
        // Match-all.
        let c = TrustedProxyCidr::parse("0.0.0.0/0").unwrap();
        assert!(c.contains("8.8.8.8".parse::<IpAddr>().unwrap()));
        // Malformed.
        assert!(TrustedProxyCidr::parse("not-an-ip").is_err());
        assert!(TrustedProxyCidr::parse("10.0.0.0/33").is_err());
        assert!(TrustedProxyCidr::parse("::1/129").is_err());
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

    // ---- Issue #2754: TLS boot-guard decision function ----

    #[test]
    fn tls_boot_guard_flags_tls_without_trusted_proxies() {
        // auth=tls + empty trusted list → insecure (the verified-client
        // header could never be safely honoured, so tls silently degrades
        // to token-only auth — the misconfiguration behind the bypass).
        assert!(is_insecure_tls_configuration(AuthMode::Tls, &[], false));
    }

    #[test]
    fn tls_boot_guard_passes_tls_with_trusted_proxies() {
        let trusted = [TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        assert!(!is_insecure_tls_configuration(
            AuthMode::Tls,
            &trusted,
            false
        ));
    }

    #[test]
    fn tls_boot_guard_ignores_non_tls_modes() {
        // off / token never trip the TLS guard regardless of the trusted
        // list — this guard must NOT weaken the other two modes.
        assert!(!is_insecure_tls_configuration(AuthMode::Off, &[], false));
        assert!(!is_insecure_tls_configuration(AuthMode::Token, &[], false));
    }

    #[test]
    fn tls_boot_guard_respects_allow_insecure_override() {
        // FLUXION_REST_ALLOW_INSECURE=1 is the explicit opt-out (parity
        // with the sibling is_insecure_bind_configuration guard).
        assert!(!is_insecure_tls_configuration(AuthMode::Tls, &[], true));
    }

    // ---- Issue #2754: require_auth TLS header trusted-proxy gate ----
    //
    // These exercise [`require_auth`] in isolation through a minimal
    // auth-only router (mirroring how it is mounted on the protected
    // sub-router), with `ConnectInfo<SocketAddr>` injected exactly as
    // `into_make_service_with_connect_info` does in production.

    /// Minimal router carrying only [`require_auth`] over a dummy 200-OK
    /// handler. Decouples the auth-middleware tests from the full API
    /// router while exercising the real middleware stack.
    fn tls_auth_router(cfg: &RestSecurityConfig) -> axum::Router {
        axum::Router::new()
            .route(
                "/v1/metrics",
                axum::routing::get(|| async { StatusCode::OK }),
            )
            .layer(axum::middleware::from_fn_with_state(
                cfg.auth_state(),
                require_auth,
            ))
    }

    /// Build a `GET /v1/metrics` request with optional `ConnectInfo`
    /// socket peer, optional `x-verified-client` header, and optional
    /// `Authorization: Bearer <token>`.
    fn tls_req(
        peer: Option<SocketAddr>,
        verified_header: Option<&str>,
        bearer: Option<&str>,
    ) -> Request {
        let mut b = Request::builder().method(Method::GET).uri("/v1/metrics");
        if let Some(p) = peer {
            b = b.extension(ConnectInfo(p));
        }
        if let Some(v) = verified_header {
            b = b.header("x-verified-client", v);
        }
        if let Some(tok) = bearer {
            b = b.header(header::AUTHORIZATION, format!("Bearer {tok}"));
        }
        b.body(axum::body::Body::empty()).unwrap()
    }

    /// Issue #2754 (negative): a direct client whose socket peer is NOT in
    /// the trusted-proxy list cannot bypass TLS auth by spoofing the
    /// verification header — it is rejected with 401.
    #[tokio::test]
    async fn tls_mode_rejects_spoofed_header_from_untrusted_peer() {
        let mut cfg = RestSecurityConfig::default();
        cfg.auth_mode = AuthMode::Tls;
        cfg.trusted_proxies = vec![TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        let app = tls_auth_router(&cfg);

        // Direct, untrusted peer carrying the default spoofed header.
        let peer: SocketAddr = "203.0.113.10:7000".parse().unwrap();
        let req = tls_req(Some(peer), Some("1"), None);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "spoofed header from a non-trusted peer must NOT bypass TLS auth"
        );
    }

    /// Issue #2754 (positive): a request whose socket peer IS a trusted
    /// proxy and which carries the agreed verification header is accepted.
    #[tokio::test]
    async fn tls_mode_accepts_header_from_trusted_proxy() {
        let mut cfg = RestSecurityConfig::default();
        cfg.auth_mode = AuthMode::Tls;
        cfg.trusted_proxies = vec![TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        let app = tls_auth_router(&cfg);

        let proxy_peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let req = tls_req(Some(proxy_peer), Some("1"), None);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "trusted proxy with the correct header must reach the handler"
        );
    }

    /// Issue #2754 (default): with `FLUXION_REST_TRUSTED_PROXIES` unset
    /// (the default empty list), the verification header is ALWAYS ignored
    /// — even when the peer address happens to look like a proxy — so a
    /// spoofed header never grants access.
    #[tokio::test]
    async fn tls_mode_ignores_header_when_no_trusted_proxies_configured() {
        let mut cfg = RestSecurityConfig::default();
        cfg.auth_mode = AuthMode::Tls;
        // trusted_proxies left at the default (empty).
        let app = tls_auth_router(&cfg);

        // A peer that WOULD be trusted if 10.0.0.0/8 were configured — but
        // it is not, so the header must be ignored.
        let peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let req = tls_req(Some(peer), Some("1"), None);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "with no trusted proxies configured the header must be ignored"
        );
    }

    /// Issue #2754: the bearer-token fallback is unaffected by the
    /// trusted-proxy gate — a direct (untrusted) client with a valid token
    /// is still accepted. Guards against over-tightening the fix and
    /// breaking token auth under `tls` mode.
    #[tokio::test]
    async fn tls_mode_token_fallback_works_for_untrusted_peer() {
        let mut cfg = RestSecurityConfig::default();
        cfg.auth_mode = AuthMode::Tls;
        cfg.auth_token = Some("s3cret".to_string());
        cfg.trusted_proxies = vec![TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        let app = tls_auth_router(&cfg);

        let direct_peer: SocketAddr = "203.0.113.20:7001".parse().unwrap();
        // Spoofed header present too — must be ignored; the token wins.
        let req = tls_req(Some(direct_peer), Some("1"), Some("s3cret"));
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "valid token must work regardless of trusted-proxy status"
        );
    }

    /// Issue #2754: a request with no `ConnectInfo` (e.g. a harness that
    /// forgot `into_make_service_with_connect_info`) cannot trust the
    /// header — fail-closed to 401.
    #[tokio::test]
    async fn tls_mode_no_connect_info_ignores_header() {
        let mut cfg = RestSecurityConfig::default();
        cfg.auth_mode = AuthMode::Tls;
        cfg.trusted_proxies = vec![TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        let app = tls_auth_router(&cfg);

        let req = tls_req(None, Some("1"), None);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "no ConnectInfo → peer unknown → header must be ignored"
        );
    }

    /// Issue #2754 defense-in-depth: a trusted proxy presenting the WRONG
    /// header value is still rejected.
    #[tokio::test]
    async fn tls_mode_trusted_proxy_wrong_header_value_rejected() {
        let mut cfg = RestSecurityConfig::default();
        cfg.auth_mode = AuthMode::Tls;
        cfg.trusted_proxies = vec![TrustedProxyCidr::parse("10.0.0.0/8").unwrap()];
        let app = tls_auth_router(&cfg);

        let proxy_peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let req = tls_req(Some(proxy_peer), Some("0"), None); // wrong value
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// Issue #2754 regression guard: the `off` and `token` modes are
    /// entirely unaffected by the trusted-proxy gate. `off` stays a no-op
    /// (the default); `token` keeps working with no trusted proxies.
    #[tokio::test]
    async fn off_and_token_modes_unaffected_by_trusted_proxy_gate() {
        // off (default) — never weakened.
        let cfg = RestSecurityConfig::default();
        let app = tls_auth_router(&cfg);
        let peer: SocketAddr = "203.0.113.30:7002".parse().unwrap();
        let req = tls_req(Some(peer), Some("1"), None);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "off mode must remain a no-op regardless of headers / proxies"
        );

        // token — works with no trusted proxies configured.
        let mut cfg = RestSecurityConfig::default();
        cfg.auth_mode = AuthMode::Token;
        cfg.auth_token = Some("s3cret".to_string());
        let app = tls_auth_router(&cfg);
        let req = tls_req(Some(peer), None, Some("s3cret"));
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "token mode must work without any trusted-proxy configuration"
        );
    }
}
