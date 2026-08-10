// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Security regression tests for the email notifier header / endpoint
//! hardening in Issue #2554.
//!
//! These tests pin the three guarantees the fix relies on:
//!
//! 1. `EmailConfig::validate()` REJECTS any `api_auth_header` value that
//!    would let a caller forward a raw HTTP header to the upstream
//!    transactional-email provider (CRLF injection, oversized values, non-
//!    Bearer schemes, illegal characters, etc.). The transport then refuses
//!    to send — verified by checking the `MockEmailTransport` saw zero
//!    envelopes.
//!
//! 2. `EmailConfig::validate()` REJECTS `api_endpoint` URLs whose scheme is
//!    not `https://`, and (when `FLUXION_EMAIL_ENDPOINT_ALLOWLIST` is set)
//!    rejects hostnames outside the allow-list. This closes the SSRF vector
//!    in the original code, which accepted `http://`, `file://`,
//!    `gopher://`, and arbitrary internal IPs.
//!
//! 3. `EmailConfig::validate()` REJECTS `download_url_template` values whose
//!    scheme is not `https://`. This prevents the rendered email from
//!    pointing at a non-https URL (which the original code interpolated
//!    verbatim into the body and which could be used as an SSRF primitive
//!    in the user's mail client).
//!
//! The mock transport is the verification signal: a test that mutates an
//! `api_auth_header` and ends with `transport.len() == 1` is a regression
//! — it means the malicious value flowed through to the transport.
//!
//! All tests are pure (no network), so they don't need the
//! `FLUXION_EMAIL_API_AUTH` env var. Where the env var matters, we set it
//! inside the test process and rely on `serial_test` (or the test
//! ordering) to keep state predictable. Because the env-var-driven tests
//! only ASSERT the env var takes precedence, we explicitly clear the
//! relevant var at the start of each test that doesn't want it.

use fluxion::api::email_notification::{
    CampaignCompletion, EmailConfig, EmailNotifier, MockEmailTransport,
};

/// Process-wide mutex serializing tests that mutate process env vars
/// (so parallel test threads don't race on the same var). We don't pull in
/// `serial_test` just for this — one `Mutex` in the test file is enough.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn sample_completion() -> CampaignCompletion {
    CampaignCompletion::completed(
        "fluxion-abc123".to_string(),
        "2026-07-18T00:00:00+00:00".to_string(),
        "2026-07-18T01:00:00+00:00".to_string(),
        50,
        48,
        2,
        4.7,
        "https://fluxion.example.com/results/abc/".to_string(),
    )
}

fn base_config() -> EmailConfig {
    EmailConfig {
        from: "fluxion-noreply".to_string() + "@" + "fluxion.example",
        to: vec!["user".to_string() + "@" + "fluxion.example"],
        // download_url_template defaults to https://fluxion.example.com/...
        // api_endpoint left unset → notifier stays in preview mode and the
        // mock transport captures nothing.
        ..Default::default()
    }
}

fn config_with_endpoint(endpoint: &str) -> EmailConfig {
    let mut c = base_config();
    c.api_endpoint = Some(endpoint.to_string());
    c
}

// ---------------------------------------------------------------------------
// api_auth_header rejection (Issue #2554 — header injection / SSRF)
// ---------------------------------------------------------------------------

/// Bare minimum — every value that fails `^Bearer [A-Za-z0-9_\-\.]{8,256}$`
/// must be rejected. Group them in a table so the matrix is obvious.
#[test]
fn validate_rejects_malformed_api_auth_header() {
    let cases: &[(&str, &str)] = &[
        ("empty", ""),
        ("no_bearer_prefix", "SG.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("basic_auth_scheme", "Basic dXNlcjpwYXNz"),
        ("bearer_no_token", "Bearer "),
        ("bearer_token_too_short", "Bearer ab"),
        ("bearer_token_with_space", "Bearer abc def ghi jkl mno pqr"),
        ("bearer_token_with_colon", "Bearer abc:def"),
        ("bearer_token_with_slash", "Bearer abc/def/ghi"),
        ("bearer_token_with_at_sign", "Bearer attacker@victim.com"),
    ];

    for (name, value) in cases {
        let mut cfg = base_config();
        cfg.api_auth_header = Some(value.to_string());
        let err = cfg
            .validate()
            .expect_err(&format!("case `{name}` unexpectedly accepted `{value}`"));
        assert!(
            matches!(
                err,
                fluxion::api::email_notification::EmailError::InvalidConfig(_)
            ),
            "case `{name}` returned wrong variant: {err:?}"
        );
    }
}

/// The single most important case from the issue body: a header containing
/// `\r\nX-Injected: pwned` MUST be rejected. `reqwest::header::HeaderValue`
/// would also reject the raw CRLF later, but the security guarantee we
/// pin is "validate() refuses to send it" — no chance of a future
/// transport swap silently widening the attack surface.
#[test]
fn validate_rejects_crlf_injection_in_api_auth_header() {
    let malicious = "Bearer abcdefgh\r\nX-Injected: pwned";
    let mut cfg = base_config();
    cfg.api_auth_header = Some(malicious.to_string());
    assert!(
        cfg.validate().is_err(),
        "CRLF-injected header was accepted: {malicious:?}"
    );
}

/// Bare `\n` injection (some servers normalise CR but not LF).
#[test]
fn validate_rejects_lone_lf_in_api_auth_header() {
    let malicious = "Bearer abcdefgh\nX-Injected: pwned";
    let mut cfg = base_config();
    cfg.api_auth_header = Some(malicious.to_string());
    assert!(cfg.validate().is_err());
}

/// NUL byte injection — `reqwest::header::HeaderValue::from_str` rejects
/// NUL but our validator must catch it FIRST so we never even reach the
/// HTTP layer.
#[test]
fn validate_rejects_nul_byte_in_api_auth_header() {
    let malicious = "Bearer abcdefgh\0xyz";
    let mut cfg = base_config();
    cfg.api_auth_header = Some(malicious.to_string());
    assert!(cfg.validate().is_err());
}

/// Oversized token (regex `{8,256}` cap).
#[test]
fn validate_rejects_oversized_api_auth_header() {
    let oversized_token = "a".repeat(1024);
    let value = format!("Bearer {oversized_token}");
    let mut cfg = base_config();
    cfg.api_auth_header = Some(value);
    assert!(cfg.validate().is_err());
}

/// Even a syntactically VALID-looking but regex-non-matching `Authorization`
/// value must NOT be forwarded — the mock transport must see zero envelopes.
/// The valid regex is `^Bearer [A-Za-z0-9_\-\.]{8,256}$` (Issue #2554).
/// We use a `Basic …` scheme here to demonstrate that ANY non-Bearer scheme
/// is rejected, not just empty / oversized values.
#[test]
fn notifier_refuses_to_send_with_invalid_auth_header() {
    let transport = MockEmailTransport::new();
    let notifier = EmailNotifier::new(transport);
    let mut cfg = base_config();
    cfg.api_endpoint = Some("https://api.example.com/send".to_string());
    // Valid Basic-auth base64 — looks plausible, fails the regex.
    cfg.api_auth_header = Some("Basic dXNlcjpwYXNzd29yZA".to_string());

    let err = notifier
        .notify(&sample_completion(), &cfg)
        .expect_err("non-Bearer auth header must be rejected");

    assert!(
        matches!(
            err,
            fluxion::api::email_notification::EmailError::InvalidConfig(_)
        ),
        "expected InvalidConfig, got {err:?}"
    );
    assert_eq!(
        notifier.transport.len(),
        0,
        "mock transport must not have captured any envelope"
    );
}

// ---------------------------------------------------------------------------
// api_endpoint allow-list + scheme (Issue #2554 — SSRF)
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_http_api_endpoint() {
    let mut cfg = base_config();
    cfg.api_endpoint = Some("http://api.example.com/send".to_string());
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_file_scheme_api_endpoint() {
    let mut cfg = base_config();
    cfg.api_endpoint = Some("file:///etc/passwd".to_string());
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_gopher_scheme_api_endpoint() {
    let mut cfg = base_config();
    cfg.api_endpoint = Some("gopher://internal.corp/send".to_string());
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_localhost_api_endpoint() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut cfg = base_config();
    cfg.api_endpoint = Some("https://localhost:9000/internal".to_string());
    // localhost is rejected because the allow-list (when set) excludes it.
    // The default test runner has no allow-list → localhost is accepted.
    // We pin the *non-allow-listed* behaviour by setting the allow-list.
    std::env::set_var(
        fluxion::api::email_notification::EMAIL_ENDPOINT_ALLOWLIST_ENV,
        "api.example.com",
    );
    let result = cfg.validate();
    std::env::remove_var(fluxion::api::email_notification::EMAIL_ENDPOINT_ALLOWLIST_ENV);
    assert!(
        result.is_err(),
        "localhost must be rejected when allow-list is set and doesn't include it"
    );
}

#[test]
fn validate_accepts_allowlisted_endpoint() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(
        fluxion::api::email_notification::EMAIL_ENDPOINT_ALLOWLIST_ENV,
        "api.example.com,api.sendgrid.com",
    );
    let cfg = config_with_endpoint("https://api.example.com/send");
    let result = cfg.validate();
    std::env::remove_var(fluxion::api::email_notification::EMAIL_ENDPOINT_ALLOWLIST_ENV);
    assert!(
        result.is_ok(),
        "allow-listed endpoint should validate: {result:?}"
    );
}

#[test]
fn validate_rejects_off_allowlist_endpoint() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(
        fluxion::api::email_notification::EMAIL_ENDPOINT_ALLOWLIST_ENV,
        "api.example.com",
    );
    let cfg = config_with_endpoint("https://attacker.example.com/send");
    let result = cfg.validate();
    std::env::remove_var(fluxion::api::email_notification::EMAIL_ENDPOINT_ALLOWLIST_ENV);
    assert!(
        result.is_err(),
        "off-allow-list endpoint must be rejected: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// download_url_template scheme (Issue #2554 — SSRF in rendered email)
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_http_download_url_template() {
    let mut cfg = base_config();
    cfg.download_url_template = "http://fluxion.example.com/campaigns/{campaign_id}/".to_string();
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_ftp_download_url_template() {
    let mut cfg = base_config();
    cfg.download_url_template = "ftp://fluxion.example.com/{campaign_id}".to_string();
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_accepts_https_download_url_template() {
    let cfg = base_config();
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Server-side credential (FLUXION_EMAIL_API_AUTH) wins over api_auth_header
// ---------------------------------------------------------------------------

/// When `FLUXION_EMAIL_API_AUTH` is set, the transport MUST use it
/// regardless of what `api_auth_header` says. This is the "option (a)"
/// guarantee from the issue remediation: server-side credential only.
#[test]
fn env_credential_wins_over_user_supplied_auth_header() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use fluxion::api::email_notification::resolve_auth_header;

    std::env::set_var(
        fluxion::api::email_notification::EMAIL_API_AUTH_ENV,
        "Bearer SERVER_CONTROLLED_TOKEN_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    );
    let mut cfg = base_config();
    cfg.api_auth_header = Some("Bearer USER_SUPPLIED_bbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string());

    let resolved = resolve_auth_header(&cfg);
    std::env::remove_var(fluxion::api::email_notification::EMAIL_API_AUTH_ENV);

    assert_eq!(
        resolved.as_deref(),
        Some("Bearer SERVER_CONTROLLED_TOKEN_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        "env credential must take precedence over user-supplied header"
    );
}

/// If the env var is unset and the user supplied a valid Bearer, we use
/// the user's value (with the regex check enforced).
#[test]
fn user_supplied_auth_header_used_when_env_var_unset() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use fluxion::api::email_notification::resolve_auth_header;

    std::env::remove_var(fluxion::api::email_notification::EMAIL_API_AUTH_ENV);
    let mut cfg = base_config();
    cfg.api_auth_header = Some("Bearer USER_SUPPLIED_bbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string());

    assert_eq!(
        resolve_auth_header(&cfg).as_deref(),
        Some("Bearer USER_SUPPLIED_bbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
    );
}
