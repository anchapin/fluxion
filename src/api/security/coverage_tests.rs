//! Coverage tests for the parent module (extracted to keep the
//! ratcheted parent file under its Issue #2878/#3574 module-size
//! ceiling; child modules can see parent-private items).
//!
//! Coverage-expansion tests (PR #5): config defaults and builder methods,
//! `default_dev_cors_origins`, `build_cors_layer` allow-list behavior,
//! `TrustedProxyCidr` edge cases, extra `is_insecure_bind_configuration`
//! cases, and the env-driven `from_env` / `check_boot_guard_from_env` error
//! paths — none of which had dedicated tests.

use super::*;

// ---- Env isolation: save/restore every var we touch ----
const ENV_KEYS: &[&str] = &[
    "FLUXION_REST_AUTH",
    "FLUXION_REST_AUTH_TOKEN",
    "FLUXION_REST_CORS_ORIGINS",
    "FLUXION_REST_RATE_LIMIT_RPS",
    "FLUXION_REST_RATE_LIMIT_BURST",
    "FLUXION_REST_RATE_LIMIT_MAX_ENTRIES",
    "FLUXION_REST_TRUSTED_PROXIES",
    "FLUXION_REST_ALLOW_INSECURE",
    "FLUXION_REST_VERIFIED_HEADER_NAME",
    "FLUXION_REST_VERIFIED_HEADER_VALUE",
    "FLUXION_REST_BIND",
];

/// Serialises every env-mutating test: `cargo test` runs tests in
/// parallel threads of one process, so two env tests racing would
/// clobber each other's variables. The guard holds this lock from
/// construction until drop (when the saved environment is restored).
static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn new() -> Self {
        let lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let saved = ENV_KEYS
            .iter()
            .map(|k| (*k, std::env::var(k).ok()))
            .collect();
        for k in ENV_KEYS {
            std::env::remove_var(k);
        }
        Self { saved, _lock: lock }
    }

    fn set(&self, key: &str, val: &str) {
        std::env::set_var(key, val);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, v) in &self.saved {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

// ---- Config defaults and builders ----

#[test]
fn default_dev_cors_origins_lists_localhost_ports() {
    assert_eq!(
        default_dev_cors_origins(),
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
    );
}

#[test]
fn rest_security_config_default_values() {
    let cfg = RestSecurityConfig::default();
    assert_eq!(cfg.auth_mode, AuthMode::Off);
    assert!(cfg.auth_token.is_none());
    assert_eq!(cfg.cors_origins, default_dev_cors_origins());
    assert_eq!(cfg.rate_limit_rps, DEFAULT_RATE_LIMIT_RPS);
    assert_eq!(cfg.rate_limit_burst, DEFAULT_RATE_LIMIT_BURST);
    assert_eq!(cfg.rate_limit_max_entries, DEFAULT_RATE_LIMIT_MAX_ENTRIES);
    assert!(cfg.trusted_proxies.is_empty());
    assert_eq!(
        cfg.verified_header_name.as_str(),
        DEFAULT_VERIFIED_HEADER_NAME
    );
    assert_eq!(cfg.verified_header_value, DEFAULT_VERIFIED_HEADER_VALUE);
}

#[test]
fn config_builder_methods_produce_working_primitives() {
    let cfg = RestSecurityConfig::default();
    // auth_state() must not panic; the rate limiter it builds must be
    // fresh and functional; the CORS layer must build without origins
    // misconfiguration panics.
    let _auth = cfg.auth_state();
    let limiter = cfg.rate_limiter();
    assert_eq!(limiter.num_entries(), 0);
    let ip: IpAddr = "127.0.0.1".parse().unwrap();
    assert!(limiter.try_acquire(ip));
    let _cors = cfg.cors_layer();
}

// ---- build_cors_layer behavior ----

fn cors_router(origins: &[String]) -> axum::Router {
    axum::Router::new()
        .route("/", axum::routing::get(|| async { "ok" }))
        .layer(build_cors_layer(origins))
}

fn get_with_origin(origin: &str) -> Request {
    Request::builder()
        .uri("/")
        .header("origin", origin)
        .body(axum::body::Body::empty())
        .unwrap()
}

fn acao(resp: &axum::response::Response) -> Option<String> {
    resp.headers()
        .get("access-control-allow-origin")
        .and_then(|v| v.to_str().ok().map(str::to_string))
}

#[tokio::test]
async fn build_cors_layer_allows_listed_origin() {
    let app = cors_router(&default_dev_cors_origins());
    let resp = tower::ServiceExt::oneshot(app, get_with_origin("http://localhost:3000"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(acao(&resp).as_deref(), Some("http://localhost:3000"));
}

#[tokio::test]
async fn build_cors_layer_blocks_unlisted_origin() {
    let app = cors_router(&default_dev_cors_origins());
    let resp = tower::ServiceExt::oneshot(app, get_with_origin("https://evil.example"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        acao(&resp),
        None,
        "unlisted origins must not get an ACAO header"
    );
}

#[tokio::test]
async fn build_cors_layer_empty_origins_omits_acao_header() {
    // No origins configured → no cross-origin browser access.
    let app = cors_router(&[]);
    let resp = tower::ServiceExt::oneshot(app, get_with_origin("http://localhost:3000"))
        .await
        .unwrap();
    assert_eq!(acao(&resp), None);
}

// ---- TrustedProxyCidr edge cases ----

#[test]
fn trusted_proxy_cidr_trims_and_handles_ipv6_bare_ip() {
    // Leading/trailing whitespace is trimmed.
    let c = TrustedProxyCidr::parse("  10.0.0.0/8\n").unwrap();
    assert!(c.contains("10.9.9.9".parse::<IpAddr>().unwrap()));

    // Bare IPv6 address → /128.
    let c = TrustedProxyCidr::parse("::1").unwrap();
    assert_eq!(c.prefix_len, 128);
    assert!(c.contains("::1".parse::<IpAddr>().unwrap()));
    assert!(!c.contains("::2".parse::<IpAddr>().unwrap()));
}

#[test]
fn trusted_proxy_cidr_rejects_family_mismatch() {
    let v4 = TrustedProxyCidr::parse("10.0.0.0/8").unwrap();
    assert!(!v4.contains("::ffff:10.0.0.1".parse::<IpAddr>().unwrap()));
    let v6 = TrustedProxyCidr::parse("2001:db8::/32").unwrap();
    assert!(!v6.contains("10.0.0.1".parse::<IpAddr>().unwrap()));
}

// ---- is_insecure_bind_configuration extra cases ----

#[test]
fn insecure_bind_ipv6_wildcards_and_garbage() {
    assert!(is_insecure_bind_configuration("::", AuthMode::Off, false));
    assert!(is_insecure_bind_configuration(
        "[::]:8080",
        AuthMode::Off,
        false
    ));
    assert!(!is_insecure_bind_configuration(
        "[::1]:8080",
        AuthMode::Off,
        false
    ));
    // A non-parseable host is not treated as a wildcard bind.
    assert!(!is_insecure_bind_configuration(
        "not-a-host",
        AuthMode::Off,
        false
    ));
    // Any auth mode other than Off clears the flag, even on 0.0.0.0.
    assert!(!is_insecure_bind_configuration(
        "0.0.0.0",
        AuthMode::Token,
        false
    ));
    assert!(!is_insecure_bind_configuration(
        "0.0.0.0",
        AuthMode::Tls,
        false
    ));
}

// ---- from_env ----

#[test]
fn from_env_fail_closed_on_bad_auth() {
    let _guard = EnvGuard::new();
    std::env::set_var("FLUXION_REST_AUTH", "bogus");
    let err = RestSecurityConfig::from_env()
        .err()
        .expect("from_env must fail on a bad FLUXION_REST_AUTH value");
    assert!(
        err.contains("unknown FLUXION_REST_AUTH value"),
        "got: {err}"
    );
}

#[test]
fn from_env_parses_full_config() {
    let guard = EnvGuard::new();
    guard.set("FLUXION_REST_AUTH", "token");
    guard.set("FLUXION_REST_AUTH_TOKEN", "test-token-123");
    guard.set(
        "FLUXION_REST_CORS_ORIGINS",
        "https://app.example, https://api.example ,",
    );
    guard.set("FLUXION_REST_RATE_LIMIT_RPS", "50");
    guard.set("FLUXION_REST_RATE_LIMIT_BURST", "25");
    guard.set("FLUXION_REST_RATE_LIMIT_MAX_ENTRIES", "1000");
    guard.set(
        "FLUXION_REST_TRUSTED_PROXIES",
        "10.0.0.0/8, not-a-cidr, 192.0.2.1",
    );
    guard.set("FLUXION_REST_VERIFIED_HEADER_NAME", "x-custom-verified");
    guard.set("FLUXION_REST_VERIFIED_HEADER_VALUE", "yes");

    let cfg = RestSecurityConfig::from_env().unwrap();
    assert_eq!(cfg.auth_mode, AuthMode::Token);
    assert_eq!(cfg.auth_token.as_deref(), Some("test-token-123"));
    assert_eq!(
        cfg.cors_origins,
        vec!["https://app.example", "https://api.example"]
    );
    assert_eq!(cfg.rate_limit_rps, 50);
    assert_eq!(cfg.rate_limit_burst, 25);
    assert_eq!(cfg.rate_limit_max_entries, 1000);
    // The malformed entry is skipped with a warning, not fatal.
    assert_eq!(cfg.trusted_proxies.len(), 2);
    assert_eq!(cfg.verified_header_name.as_str(), "x-custom-verified");
    assert_eq!(cfg.verified_header_value, "yes");
}

#[test]
fn from_env_ignores_garbage_numeric_overrides() {
    let guard = EnvGuard::new();
    guard.set("FLUXION_REST_RATE_LIMIT_RPS", "not-a-number");
    guard.set("FLUXION_REST_RATE_LIMIT_BURST", "0");
    guard.set("FLUXION_REST_RATE_LIMIT_MAX_ENTRIES", "-5");

    let cfg = RestSecurityConfig::from_env().unwrap();
    assert_eq!(cfg.rate_limit_rps, DEFAULT_RATE_LIMIT_RPS);
    assert_eq!(cfg.rate_limit_burst, DEFAULT_RATE_LIMIT_BURST);
    assert_eq!(cfg.rate_limit_max_entries, DEFAULT_RATE_LIMIT_MAX_ENTRIES);
}

#[test]
fn from_env_unset_keeps_defaults() {
    let _guard = EnvGuard::new();
    let cfg = RestSecurityConfig::from_env().unwrap();
    assert_eq!(cfg.auth_mode, AuthMode::Off);
    assert_eq!(cfg.cors_origins, default_dev_cors_origins());
    assert_eq!(cfg.rate_limit_rps, DEFAULT_RATE_LIMIT_RPS);
}

// ---- check_boot_guard_from_env ----

#[test]
fn boot_guard_from_env_ok_with_defaults() {
    let _guard = EnvGuard::new();
    assert!(check_boot_guard_from_env().is_ok());
}

#[test]
fn boot_guard_from_env_rejects_bad_auth_value() {
    let guard = EnvGuard::new();
    guard.set("FLUXION_REST_AUTH", "bogus");
    let err = check_boot_guard_from_env().unwrap_err();
    assert!(
        err.contains("unknown FLUXION_REST_AUTH value"),
        "got: {err}"
    );
}

#[test]
fn boot_guard_from_env_rejects_tls_without_proxies() {
    let guard = EnvGuard::new();
    guard.set("FLUXION_REST_AUTH", "tls");
    let err = check_boot_guard_from_env().unwrap_err();
    assert!(err.contains("TRUSTED_PROXIES"), "got: {err}");
}

#[test]
fn boot_guard_from_env_accepts_tls_with_proxies() {
    let guard = EnvGuard::new();
    guard.set("FLUXION_REST_AUTH", "tls");
    guard.set("FLUXION_REST_TRUSTED_PROXIES", "10.0.0.0/8");
    assert!(check_boot_guard_from_env().is_ok());
}

#[test]
fn boot_guard_from_env_allow_insecure_opt_out() {
    let guard = EnvGuard::new();
    guard.set("FLUXION_REST_AUTH", "tls");
    guard.set("FLUXION_REST_ALLOW_INSECURE", "1");
    assert!(
        check_boot_guard_from_env().is_ok(),
        "ALLOW_INSECURE=1 must opt out of the TLS boot guard"
    );
}

// ---- RateLimiter clamping edge ----

#[test]
fn rate_limiter_clamps_zero_config_to_one() {
    // Documented: rps/burst of 0 are clamped to 1, so the limiter still
    // functions (single-token bucket) instead of rejecting everything.
    let limiter = RateLimiter::new(0, 0, 0, &[]);
    let ip: IpAddr = "10.9.8.7".parse().unwrap();
    assert!(limiter.try_acquire(ip), "first acquire must succeed");
    assert!(
        !limiter.try_acquire(ip),
        "burst is clamped to 1: an immediate second acquire must fail"
    );
}
