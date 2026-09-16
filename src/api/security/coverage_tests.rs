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
    // Issue #3742 — the bracketed bare wildcard `[::]` (no port) is now
    // resolved by the shared canonical resolver and correctly flagged;
    // pre-#3742 it slipped past the raw-string parser while the
    // listener's `{bind}:{port}` concatenation happily bound `[::]:port`
    // (all IPv6 interfaces).
    assert!(is_insecure_bind_configuration("[::]", AuthMode::Off, false));
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

// ---- Issue #3743: weak-token boot guard (min FLUXION_REST_AUTH_TOKEN length) ----

/// Issue #3743 acceptance: the pure decision function flags `token`/`tls`
/// with a *configured* token shorter than [`MIN_REST_AUTH_TOKEN_BYTES`]
/// (boundary: 15 bytes insecure, 16 bytes fine), leaves `off` mode
/// unaffected, treats an unset token as out of scope (the `require_auth`
/// middleware already fails closed per-request), and honours
/// `FLUXION_REST_ALLOW_INSECURE=1` as the explicit opt-out. The
/// release-only wiring in [`check_boot_guard_from_env`] is exercised by
/// the binary; unit tests target the decision function so they are
/// deterministic in debug builds (mirroring the #2505 bind-guard test
/// strategy).
#[test]
fn weak_token_boot_guard_boundary_and_mode_scoping() {
    let ok = "a".repeat(MIN_REST_AUTH_TOKEN_BYTES); // exactly 16 bytes
    let short = "a".repeat(MIN_REST_AUTH_TOKEN_BYTES - 1); // 15 bytes
    assert_eq!(ok.len(), 16);
    assert_eq!(short.len(), 15);

    // `token` mode: 15 bytes → insecure, 16 bytes → fine.
    assert!(is_weak_auth_token_configuration(
        AuthMode::Token,
        Some(&short),
        false
    ));
    assert!(!is_weak_auth_token_configuration(
        AuthMode::Token,
        Some(&ok),
        false
    ));

    // `tls` mode: the bearer token is the direct-client fallback — a
    // configured short token is refused there too.
    assert!(is_weak_auth_token_configuration(
        AuthMode::Tls,
        Some(&short),
        false
    ));
    assert!(!is_weak_auth_token_configuration(
        AuthMode::Tls,
        Some(&ok),
        false
    ));

    // `off` mode is unaffected: a short (or unset) token never trips the
    // guard when auth is disabled.
    assert!(!is_weak_auth_token_configuration(
        AuthMode::Off,
        Some(&short),
        false
    ));
    assert!(!is_weak_auth_token_configuration(
        AuthMode::Off,
        None,
        false
    ));

    // An unset token is out of scope for this guard: `token` mode fails
    // closed per-request in [`require_auth`], and `tls` mode legitimately
    // runs header-only. An empty *string* is zero bytes — flagged
    // defensively — although the env-reading paths normalize empty to
    // unset before the decision function ever sees it.
    assert!(!is_weak_auth_token_configuration(
        AuthMode::Token,
        None,
        false
    ));
    assert!(!is_weak_auth_token_configuration(
        AuthMode::Tls,
        None,
        false
    ));
    assert!(is_weak_auth_token_configuration(
        AuthMode::Token,
        Some(""),
        false
    ));

    // FLUXION_REST_ALLOW_INSECURE=1 is the explicit opt-out, mirroring the
    // sibling bind/TLS boot guards.
    assert!(!is_weak_auth_token_configuration(
        AuthMode::Token,
        Some(&short),
        true
    ));
    assert!(!is_weak_auth_token_configuration(
        AuthMode::Tls,
        Some(&short),
        true
    ));
}

// ---- Issue #3742: canonical bind resolver + unresolvable-bind guard ----

/// Issue #3742 acceptance: [`resolve_rest_bind_addr`] is the single
/// parsing path shared by the listener (`resolve_addr` in
/// `src/bin/fluxion_rest.rs`) and the boot guard. Pin the accepted forms
/// (bare v4/v6, bracketed bare v6, host:port with the embedded port
/// winning, scheme prefix, trimming), the rejected forms (hostnames such
/// as `localhost`, garbage, empty — every error names the variable so the
/// release abort is actionable), and the guard/listener agreement: the
/// #2505 wildcard verdict ([`is_insecure_bind_configuration`]) is always
/// computed from the address [`resolve_rest_bind_addr`] returns, and the
/// #3742 guard ([`is_unresolvable_bind_configuration`]) flags exactly the
/// values the resolver rejects — so the address the guard judged is the
/// address that gets bound. The release-only wiring in
/// [`check_boot_guard_from_env`] is exercised by the binary; unit tests
/// target the decision functions so they are deterministic in debug
/// builds (mirroring the #2505 / #3743 test strategy).
#[test]
fn rest_bind_resolver_matrix_and_guard_parity() {
    let port: u16 = 9090;

    // Accepted forms — each resolves to the exact socket address the
    // listener will bind.
    let assert_resolves = |raw: &str, expect: &str| {
        let sa = resolve_rest_bind_addr(raw, port)
            .unwrap_or_else(|e| panic!("'{raw}' must resolve: {e}"));
        assert_eq!(
            sa,
            expect.parse::<SocketAddr>().unwrap(),
            "raw='{raw}' must resolve to {expect}"
        );
    };
    assert_resolves("127.0.0.1", "127.0.0.1:9090");
    // Stray whitespace (e.g. a Kubernetes ConfigMap value) is trimmed.
    assert_resolves("  10.0.0.5\n", "10.0.0.5:9090");
    // Bare IPv6 — the legacy `{bind}:{port}` concatenation mangled these
    // into a parse failure (and then a silent 0.0.0.0 fallback).
    assert_resolves("::1", "[::1]:9090");
    assert_resolves("::", "[::]:9090");
    // Bracketed bare IPv6 was accepted by the legacy concatenation; keep.
    assert_resolves("[::1]", "[::1]:9090");
    assert_resolves("0.0.0.0", "0.0.0.0:9090");
    // A full socket address wins: the embedded port overrides the
    // FLUXION_REST_PORT-sourced `port` argument.
    assert_resolves("127.0.0.1:443", "127.0.0.1:443");
    assert_resolves("[::]:80", "[::]:80");
    // Scheme prefix is stripped (parity with the pre-#3742 guard).
    assert_resolves("https://127.0.0.1", "127.0.0.1:9090");

    // Rejected forms — the exact values whose legacy behaviour was the
    // silent 0.0.0.0 widening. Every error names FLUXION_REST_BIND so
    // the release-build abort points at the offending variable.
    for bad in [
        "localhost",
        "localhost:8080",
        "fluxion.internal",
        "",
        "   ",
        "0.0.0.0:not-a-port",
        "not-a-host",
    ] {
        let err = resolve_rest_bind_addr(bad, port)
            .err()
            .unwrap_or_else(|| panic!("'{bad}' must be rejected"));
        assert!(err.contains("FLUXION_REST_BIND"), "got: {err}");
    }

    // Guard/listener parity — the Issue #3742 pin. For every accepted
    // value the wildcard verdict matches the *resolved* address; for
    // every rejected value the unresolvable-bind guard flags it (and
    // only those). If either side ever switches to a different parser,
    // this loop is where the drift shows up.
    for raw in [
        "127.0.0.1",
        "::1",
        "[::1]:8080",
        "0.0.0.0:9000",
        "::",
        "[::]",
        "http://0.0.0.0",
        "localhost",
        "not-a-host",
        "",
    ] {
        match resolve_rest_bind_addr(raw, port) {
            Ok(sa) => {
                assert!(
                    !is_unresolvable_bind_configuration(raw, false),
                    "'{raw}' resolves; the #3742 guard must not flag it"
                );
                assert_eq!(
                    is_insecure_bind_configuration(raw, AuthMode::Off, false),
                    sa.ip().is_unspecified(),
                    "wildcard verdict must match the address that gets bound: '{raw}'"
                );
            }
            Err(_) => {
                assert!(
                    is_unresolvable_bind_configuration(raw, false),
                    "'{raw}' does not resolve; the #3742 guard must flag it"
                );
                // Preserved #2505 semantics: an unparseable value is not
                // *itself* a wildcard bind — the #3742 guard aborts on it
                // separately in release builds.
                assert!(
                    !is_insecure_bind_configuration(raw, AuthMode::Off, false),
                    "'{raw}' is unparseable, not wildcard: {raw:?}"
                );
            }
        }
    }

    // FLUXION_REST_ALLOW_INSECURE=1 is the explicit opt-out, mirroring
    // the sibling bind / TLS / weak-token boot guards.
    assert!(!is_unresolvable_bind_configuration("localhost", true));
    assert!(!is_unresolvable_bind_configuration("", true));

    // The exported defaults are exactly what the binary's
    // resolve_bind()/resolve_port() fall back to, so the guard judges
    // the same default address the listener binds when the variables
    // are unset.
    assert_eq!(DEFAULT_REST_BIND, "0.0.0.0");
    assert_eq!(DEFAULT_REST_PORT, 8080);
    assert!(resolve_rest_bind_addr(DEFAULT_REST_BIND, DEFAULT_REST_PORT).is_ok());
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
