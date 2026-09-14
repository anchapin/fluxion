//! Regression test for Issue #2503.
//!
//! `AwsCredentials` previously derived `Debug`, which printed the raw
//! `secret_access_key` (and the STS `session_token`) verbatim on
//! `format!("{:?}", creds)`, `dbg!()`, or any `eprintln!("{:?}")` / log site.
//! That is a credential-leak footgun: a single stray debug print exposes the
//! long-lived AWS secret.
//!
//! This test asserts that the manual `Debug` impl redacts all secret material:
//! the secret access key value, the session token value, and the literal field
//! name `secret_access_key` must never appear alongside the secret value.

use fluxion::ai::s3_upload::AwsCredentials;

const SECRET_ACCESS_KEY: &str = "AKIAFAKESECRET1234567890";
const SESSION_TOKEN: &str = "FQoGZXIvYXdzEJ7//////FAKE_SESSION_TOKEN_VALUE";
const ACCESS_KEY_ID: &str = "AKIAIOSFODNN7EXAMPLE";

#[test]
fn debug_does_not_leak_secret_access_key() {
    let creds = AwsCredentials {
        access_key_id: ACCESS_KEY_ID.to_string(),
        secret_access_key: SECRET_ACCESS_KEY.to_string(),
        session_token: None,
    };

    let rendered = format!("{:?}", creds);

    // The secret value must never appear in the debug output.
    assert!(
        !rendered.contains(SECRET_ACCESS_KEY),
        "AwsCredentials Debug output leaked the secret access key: {rendered}"
    );
    // The redaction marker must be present.
    assert!(
        rendered.contains("****"),
        "AwsCredentials Debug output is missing the redaction marker: {rendered}"
    );
}

#[test]
fn debug_does_not_leak_session_token() {
    let creds = AwsCredentials {
        access_key_id: ACCESS_KEY_ID.to_string(),
        secret_access_key: SECRET_ACCESS_KEY.to_string(),
        session_token: Some(SESSION_TOKEN.to_string()),
    };

    let rendered = format!("{:?}", creds);

    assert!(
        !rendered.contains(SESSION_TOKEN),
        "AwsCredentials Debug output leaked the session token: {rendered}"
    );
    assert!(
        !rendered.contains(SECRET_ACCESS_KEY),
        "AwsCredentials Debug output leaked the secret access key: {rendered}"
    );
    // A present session token must render as a redaction, not as its value.
    assert!(
        rendered.contains("<redacted>"),
        "AwsCredentials Debug output did not redact the present session token: {rendered}"
    );
}

#[test]
fn debug_absent_session_token_renders_none() {
    let creds = AwsCredentials {
        access_key_id: ACCESS_KEY_ID.to_string(),
        secret_access_key: SECRET_ACCESS_KEY.to_string(),
        session_token: None,
    };

    let rendered = format!("{:?}", creds);

    assert!(
        rendered.contains("None"),
        "AwsCredentials Debug output did not render absent session token as None: {rendered}"
    );
}

#[test]
fn debug_pretty_and_alternate_also_redact() {
    // `:#?` (pretty-print) and other formatter flags go through the same manual
    // impl, so they must also be redacted.
    let creds = AwsCredentials {
        access_key_id: ACCESS_KEY_ID.to_string(),
        secret_access_key: SECRET_ACCESS_KEY.to_string(),
        session_token: Some(SESSION_TOKEN.to_string()),
    };

    let pretty = format!("{:#?}", creds);
    assert!(
        !pretty.contains(SECRET_ACCESS_KEY) && !pretty.contains(SESSION_TOKEN),
        "AwsCredentials pretty Debug leaked secrets: {pretty}"
    );
}

#[test]
fn access_key_id_is_truncated_not_full() {
    let creds = AwsCredentials {
        access_key_id: ACCESS_KEY_ID.to_string(),
        secret_access_key: SECRET_ACCESS_KEY.to_string(),
        session_token: None,
    };

    let rendered = format!("{:?}", creds);

    // The full access key id is less sensitive than the secret, but should
    // still be truncated to a non-reversible prefix rather than printed in full.
    assert!(
        !rendered.contains(ACCESS_KEY_ID),
        "AwsCredentials Debug output leaked the full access key id: {rendered}"
    );
    // The 4-char prefix is retained for log correlation.
    assert!(
        rendered.contains(&ACCESS_KEY_ID[..4]),
        "AwsCredentials Debug output dropped the access key id prefix: {rendered}"
    );
}
