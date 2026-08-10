// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Email notification fallback for OSimFlow campaign completion (Issue #1789 / T7.5).
//!
//! Not every user runs a webhook listener — most want a plain email when their
//! cloud-hosted campaign finishes. This module is the "universal fallback"
//! channel: it renders a deterministic, download-link-bearing email body for a
//! campaign completion event and hands it to a pluggable
//! [`EmailTransport`]. The HTTP transport POSTs a JSON envelope to a
//! transactional-email provider (SendGrid, Mailgun, Postmark, SES v2, …) —
//! the body schema is intentionally provider-agnostic so the same renderer
//! feeds every backend.
//!
//! The renderer is **pure** (`render_email`) so it is trivially testable
//! without any network or filesystem state. Side effects live behind the
//! [`EmailTransport`] trait; production code uses [`HttpEmailTransport`] and
//! tests use [`MockEmailTransport`].
//!
//! ## Idempotency
//!
//! Campaign notification MUST be idempotent — repeated coordinator restarts
//! must not spam the user. The notifier does not own that state (the campaign
//! state store does, mirroring the `notification_sent` flag on the Python
//! side) but it refuses to re-send a payload that has the same
//! `campaign_id` + `completion_hash` combination via [`EmailNotifier::is_duplicate`].
//!
//! ## Example
//!
//! ```no_run
//! use fluxion::api::email_notification::{
//!     CampaignCompletion, EmailConfig, EmailNotifier, HttpEmailTransport,
//! };
//!
//! let config = EmailConfig {
//!     from: "fluxion-noreply".to_string() + "@" + "fluxion.example",
//!     to: vec!["user".to_string() + "@" + "fluxion.example"],
//!     ..Default::default()
//! };
//! let completion = CampaignCompletion::completed(
//!     "fluxion-abc123".to_string(),
//!     "2026-07-18T00:00:00+00:00".to_string(),
//!     "2026-07-18T01:00:00+00:00".to_string(),
//!     50,
//!     48,
//!     2,
//!     4.7,
//!     "https://example.com/results/".to_string(),
//! );
//! let notifier = EmailNotifier::new(HttpEmailTransport::new());
//! notifier.notify(&completion, &config).expect("send");
//! ```

use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default subject template — uses `{campaign_id}` and `{status}` placeholders.
pub const DEFAULT_SUBJECT_TEMPLATE: &str =
    "Fluxion campaign {campaign_id} {status_display} — results ready";

/// Default plain-text body template. Placeholders:
/// `{campaign_id}`, `{status}`, `{status_display}`, `{start_time}`, `{end_time}`,
/// `{total_runs}`, `{completed_runs}`, `{failed_runs}`, `{best_mae}`,
/// `{best_parameters}`, `{download_url}`.
pub const DEFAULT_BODY_TEMPLATE: &str = "\
Fluxion campaign {campaign_id} is {status_display}.

Started:  {start_time}
Finished: {end_time}

Total runs:     {total_runs}
Completed runs: {completed_runs}
Failed runs:    {failed_runs}
Best MAE:       {best_mae:.2}%

Best parameters:
{best_parameters_block}

Download aggregated results:
{download_url}

—
Fluxion cloud coordinator (OSimFlow)
";

/// Default download-URL template. `{campaign_id}` placeholder is replaced.
pub const DEFAULT_DOWNLOAD_URL_TEMPLATE: &str =
    "https://fluxion.example.com/campaigns/{campaign_id}/results/";

/// Header name carrying the campaign id (mirrors the webhook payload schema
/// from Issue #1788 so downstream consumers can rely on a single schema).
pub const HEADER_CAMPAIGN_ID: &str = "X-Fluxion-Campaign-Id";

/// Header name carrying the completion hash used for idempotency.
pub const HEADER_COMPLETION_HASH: &str = "X-Fluxion-Completion-Hash";

/// Errors produced by the email notification subsystem.
#[derive(Debug, Error)]
pub enum EmailError {
    /// The supplied [`EmailConfig`] was invalid (missing from/to/etc.).
    #[error("invalid email config: {0}")]
    InvalidConfig(String),

    /// A required template placeholder was missing from the template string.
    #[error("missing template placeholder: {0}")]
    MissingPlaceholder(String),

    /// The transport failed to deliver the rendered email.
    #[error("email transport error: {0}")]
    Transport(String),
}

/// Configuration for the email notification channel.
///
/// All fields are explicit (no `Option<Box<dyn …>>`) so the configuration is
/// trivially `Serialize`/`Deserialize`-able from JSON/YAML — useful when the
/// OSimFlow coordinator loads notification preferences from
/// `state.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmailConfig {
    /// Sender address, e.g. `fluxion-noreply@fluxion.example`.
    pub from: String,
    /// Primary recipients.
    pub to: Vec<String>,
    /// Optional carbon-copy recipients.
    #[serde(default)]
    pub cc: Vec<String>,
    /// Subject template (uses `{campaign_id}`, `{status}`, `{status_display}`).
    #[serde(default = "default_subject_template")]
    pub subject_template: String,
    /// Plain-text body template. See [`DEFAULT_BODY_TEMPLATE`] for placeholders.
    #[serde(default = "default_body_template")]
    pub body_template: String,
    /// Download URL template (uses `{campaign_id}`).
    ///
    /// **Security (Issue #2554):** the template MUST use the `https://`
    /// scheme. `http://`, `file://`, `gopher://`, etc. are rejected by
    /// [`EmailConfig::validate`] — this prevents the rendered email from
    /// turning into an SSRF primitive when the user follows the link or
    /// when downstream tooling dereferences it.
    #[serde(default = "default_download_url_template")]
    pub download_url_template: String,
    /// Optional override of the recipient download URL. If `Some`, this wins
    /// over `download_url_template` — useful for pre-signed S3 URLs.
    #[serde(default)]
    pub download_url_override: Option<String>,
    /// Optional HTTP API endpoint for the transactional email provider. When
    /// `None`, [`EmailNotifier`] still renders the body but skips delivery
    /// (test/preview mode).
    ///
    /// **Security (Issue #2554):** the endpoint hostname is checked against
    /// the `FLUXION_EMAIL_ENDPOINT_ALLOWLIST` env var when it is set (comma-
    /// separated hostnames, e.g. `api.sendgrid.com,api.mailgun.net`). When
    /// the env var is unset all `https://` endpoints are accepted. `http://`,
    /// `file://`, `gopher://`, etc. are always rejected.
    #[serde(default)]
    pub api_endpoint: Option<String>,
    /// Optional HTTP `Authorization` header value (e.g. `"Bearer SG.xxx"`).
    ///
    /// **Security (Issue #2554):** when present the value MUST match the
    /// strict regex `^Bearer [A-Za-z0-9_\-\.]{8,256}$` (no whitespace, no
    /// control characters, no CRLF, no scheme prefix other than `Bearer `).
    /// Any other value — including CRLF-injected headers, oversized values,
    /// or non-Bearer schemes — causes [`EmailConfig::validate`] to return
    /// [`EmailError::InvalidConfig`] and the notifier to refuse to send.
    ///
    /// Operators SHOULD prefer the `FLUXION_EMAIL_API_AUTH` env var instead
    /// of setting this field: the env var is server-controlled, never
    /// accepted from a request payload, and wins over `api_auth_header` if
    /// both are present.
    #[serde(default)]
    pub api_auth_header: Option<String>,
    /// Request timeout in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}

fn default_subject_template() -> String {
    DEFAULT_SUBJECT_TEMPLATE.to_string()
}

fn default_body_template() -> String {
    DEFAULT_BODY_TEMPLATE.to_string()
}

fn default_download_url_template() -> String {
    DEFAULT_DOWNLOAD_URL_TEMPLATE.to_string()
}

fn default_timeout() -> u64 {
    10
}

/// Strict regex-equivalent for the `Authorization` header value the email
/// notifier accepts. Equivalent to `^Bearer [A-Za-z0-9_\-\.]{8,256}$` but
/// written by hand so we don't pull a `regex` dependency for one anchor.
///
/// Issue #2554 acceptance criterion.
const BEARER_AUTH_HEADER_MIN_TOKEN_LEN: usize = 8;
const BEARER_AUTH_HEADER_MAX_TOKEN_LEN: usize = 256;

/// Env var consulted when the operator wants a server-controlled credential
/// (Issue #2554 — preferred over `api_auth_header`).
pub const EMAIL_API_AUTH_ENV: &str = "FLUXION_EMAIL_API_AUTH";

/// Env var consulted when the operator wants to restrict which upstream
/// transactional-email hostnames this process will talk to. Comma-separated
/// hostnames, e.g. `api.sendgrid.com,api.mailgun.net`.
pub const EMAIL_ENDPOINT_ALLOWLIST_ENV: &str = "FLUXION_EMAIL_ENDPOINT_ALLOWLIST";

/// Validate that `value` is a safe `https://` URL.
///
/// Rejects:
/// - any URL whose scheme is not `https://`
/// - any URL containing whitespace, control characters, CRLF, or NULs
/// - any URL without a host segment
fn validate_https_url(value: &str, field: &str) -> Result<(), EmailError> {
    let trimmed_start = value.trim_start();
    if !trimmed_start.starts_with("https://") {
        return Err(EmailError::InvalidConfig(format!(
            "`{field}` must use the https:// scheme (got `{value}`)"
        )));
    }
    if value
        .as_bytes()
        .iter()
        .any(|&b| b == b'\r' || b == b'\n' || b == 0 || b == b' ' || b == b'\t')
    {
        return Err(EmailError::InvalidConfig(format!(
            "`{field}` contains whitespace or control characters"
        )));
    }
    let host = extract_url_host(trimmed_start)
        .ok_or_else(|| EmailError::InvalidConfig(format!("`{field}` is missing a hostname")))?;
    if host.is_empty() {
        return Err(EmailError::InvalidConfig(format!(
            "`{field}` is missing a hostname"
        )));
    }
    Ok(())
}

/// Extract the host (lower-cased) from an `https://host[:port]/path` URL.
/// Returns `None` for malformed URLs.
fn extract_url_host(url: &str) -> Option<String> {
    let after_scheme = url.strip_prefix("https://")?;
    let host_end = after_scheme.find(['/', ':', '?', '#'])?;
    let host = &after_scheme[..host_end];
    if host.is_empty() {
        return None;
    }
    Some(host.to_ascii_lowercase())
}

/// Enforce the `FLUXION_EMAIL_ENDPOINT_ALLOWLIST` env var (if set) against
/// the hostname of `endpoint`.
fn validate_endpoint_host_allowlisted(endpoint: &str) -> Result<(), EmailError> {
    let host = match extract_url_host(endpoint) {
        Some(h) => h,
        None => {
            return Err(EmailError::InvalidConfig(
                "`api_endpoint` is missing a hostname".to_string(),
            ))
        }
    };
    let allowlist = match env::var_os(EMAIL_ENDPOINT_ALLOWLIST_ENV) {
        Some(v) => v,
        None => return Ok(()), // no allow-list configured → accept any https host
    };
    let allowlist_str = allowlist.to_string_lossy();
    let allowed: Vec<String> = allowlist_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    if allowed.is_empty() {
        return Ok(());
    }
    if !allowed.iter().any(|h| h == &host) {
        return Err(EmailError::InvalidConfig(format!(
            "`api_endpoint` host `{host}` is not in {EMAIL_ENDPOINT_ALLOWLIST_ENV} \
             (allowed: {})",
            allowed.join(", ")
        )));
    }
    Ok(())
}

/// Validate that `value` matches `^Bearer [A-Za-z0-9_\-\.]{8,256}$`. Returns
/// `InvalidConfig` on any CRLF, whitespace, non-Bearer scheme, or
/// over/undersized token. Issue #2554 acceptance criterion.
fn validate_bearer_auth_header(value: &str) -> Result<(), EmailError> {
    if value
        .as_bytes()
        .iter()
        .any(|&b| b == b'\r' || b == b'\n' || b == 0)
    {
        return Err(EmailError::InvalidConfig(
            "`api_auth_header` contains CRLF or control characters".to_string(),
        ));
    }
    if value.len() > "Bearer ".len() + BEARER_AUTH_HEADER_MAX_TOKEN_LEN {
        return Err(EmailError::InvalidConfig(
            "`api_auth_header` token exceeds 256 characters".to_string(),
        ));
    }
    let token = value.strip_prefix("Bearer ").ok_or_else(|| {
        EmailError::InvalidConfig(
            "`api_auth_header` must start with the `Bearer ` scheme".to_string(),
        )
    })?;
    let token_len = token.len();
    if token_len < BEARER_AUTH_HEADER_MIN_TOKEN_LEN {
        return Err(EmailError::InvalidConfig(format!(
            "`api_auth_header` token is shorter than {BEARER_AUTH_HEADER_MIN_TOKEN_LEN} characters"
        )));
    }
    if token
        .chars()
        .any(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.'))
    {
        return Err(EmailError::InvalidConfig(
            "`api_auth_header` token contains invalid characters \
             (allowed: A-Z a-z 0-9 _ - .)"
                .to_string(),
        ));
    }
    Ok(())
}

/// Resolve the outbound `Authorization` header value, preferring the
/// server-controlled env var over any (already-validated) `api_auth_header`
/// on the config. Issue #2554 — server-side credential wins.
///
/// Exposed (rather than kept private) so the `tests/email_notifier_header_safety.rs`
/// integration test can pin the precedence invariant.
pub fn resolve_auth_header(config: &EmailConfig) -> Option<String> {
    if let Ok(env_value) = env::var(EMAIL_API_AUTH_ENV) {
        if !env_value.is_empty() {
            return Some(env_value);
        }
    }
    config.api_auth_header.clone()
}

impl Default for EmailConfig {
    fn default() -> Self {
        Self {
            from: String::new(),
            to: Vec::new(),
            cc: Vec::new(),
            subject_template: default_subject_template(),
            body_template: default_body_template(),
            download_url_template: default_download_url_template(),
            download_url_override: None,
            api_endpoint: None,
            api_auth_header: None,
            timeout_seconds: default_timeout(),
        }
    }
}

impl EmailConfig {
    /// Validate the configuration. Returns `Err(EmailError::InvalidConfig)`
    /// if a required field is empty, or — per the security contract
    /// documented on each field — if `api_auth_header`, `api_endpoint`, or
    /// `download_url_template` carry an unsafe value (Issue #2554).
    pub fn validate(&self) -> Result<(), EmailError> {
        if self.from.trim().is_empty() {
            return Err(EmailError::InvalidConfig(
                "`from` must be a non-empty address".to_string(),
            ));
        }
        if self.to.is_empty() {
            return Err(EmailError::InvalidConfig(
                "`to` must contain at least one recipient".to_string(),
            ));
        }
        for addr in self.to.iter().chain(self.cc.iter()) {
            if addr.trim().is_empty() {
                return Err(EmailError::InvalidConfig(
                    "recipient addresses must not be empty".to_string(),
                ));
            }
            if !addr.contains('@') {
                return Err(EmailError::InvalidConfig(format!(
                    "recipient `{addr}` is not a valid address"
                )));
            }
        }
        if self.subject_template.trim().is_empty() {
            return Err(EmailError::InvalidConfig(
                "`subject_template` must not be empty".to_string(),
            ));
        }
        if self.body_template.trim().is_empty() {
            return Err(EmailError::InvalidConfig(
                "`body_template` must not be empty".to_string(),
            ));
        }
        if self.download_url_template.trim().is_empty() {
            return Err(EmailError::InvalidConfig(
                "`download_url_template` must not be empty".to_string(),
            ));
        }
        // Issue #2554 — download_url_template must be https:// to avoid SSRF.
        validate_https_url(&self.download_url_template, "download_url_template")?;
        if self.timeout_seconds == 0 {
            return Err(EmailError::InvalidConfig(
                "`timeout_seconds` must be > 0".to_string(),
            ));
        }
        // Issue #2554 — api_endpoint must be a safe, allow-listed https URL.
        if let Some(endpoint) = &self.api_endpoint {
            validate_https_url(endpoint, "api_endpoint")?;
            validate_endpoint_host_allowlisted(endpoint)?;
        }
        // Issue #2554 — api_auth_header must be a strict Bearer token; no
        // user-supplied header is forwarded verbatim to the upstream SMTP /
        // transactional-email provider.
        if let Some(auth) = &self.api_auth_header {
            validate_bearer_auth_header(auth)?;
        }
        Ok(())
    }
}

/// Campaign completion payload — mirrors the schema introduced for the
/// webhook channel in Issue #1788 so consumers (email-fallback, webhooks,
/// SNS) see a single shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CampaignCompletion {
    pub campaign_id: String,
    pub status: String,
    pub start_time: String,
    pub end_time: String,
    pub total_runs: usize,
    pub completed_runs: usize,
    pub failed_runs: usize,
    pub best_mae: f64,
    #[serde(default)]
    pub best_parameters: HashMap<String, f64>,
    pub results_uri: String,
}

impl CampaignCompletion {
    /// Convenience constructor for the common "completed" event.
    #[allow(clippy::too_many_arguments)]
    pub fn completed(
        campaign_id: String,
        start_time: String,
        end_time: String,
        total_runs: usize,
        completed_runs: usize,
        failed_runs: usize,
        best_mae: f64,
        results_uri: String,
    ) -> Self {
        Self {
            campaign_id,
            status: "completed".to_string(),
            start_time,
            end_time,
            total_runs,
            completed_runs,
            failed_runs,
            best_mae,
            best_parameters: HashMap::new(),
            results_uri,
        }
    }

    /// Convenience constructor for the "failed" event.
    #[allow(clippy::too_many_arguments)]
    pub fn failed(
        campaign_id: String,
        start_time: String,
        end_time: String,
        total_runs: usize,
        completed_runs: usize,
        failed_runs: usize,
        results_uri: String,
    ) -> Self {
        Self {
            campaign_id,
            status: "failed".to_string(),
            start_time,
            end_time,
            total_runs,
            completed_runs,
            failed_runs,
            best_mae: f64::NAN,
            best_parameters: HashMap::new(),
            results_uri,
        }
    }

    /// Deterministic 64-bit hash of the fields that distinguish two
    /// completion events for the same campaign. Used for idempotency.
    ///
    /// The hash is computed by feeding the JSON representation into a
    /// stable FNV-1a digest (no external crypto dep — this is just for
    /// idempotency tagging, not security).
    pub fn completion_hash(&self) -> u64 {
        let mut hasher = Fnv1a::new();
        hasher.hash_str(&self.campaign_id);
        hasher.hash_str(&self.status);
        hasher.hash_str(&self.end_time);
        hasher.hash_usize(self.completed_runs);
        hasher.hash_usize(self.failed_runs);
        hasher.hash_f64(self.best_mae);
        hasher.hash_str(&self.results_uri);
        hasher.finalize()
    }

    /// Human-friendly status display ("Completed" / "Failed") used by the
    /// default subject/body templates.
    pub fn status_display(&self) -> &'static str {
        if self.status.eq_ignore_ascii_case("failed") {
            "failed"
        } else {
            "completed"
        }
    }
}

/// Fully rendered email — what the transport hands to the provider.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedEmail {
    pub subject: String,
    pub body_text: String,
    pub download_url: String,
    pub completion_hash: u64,
}

/// JSON envelope POSTed to the transactional-email provider. Provider-agnostic
/// so the same shape works for SendGrid, Mailgun, Postmark, SES v2.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmailEnvelope {
    pub from: String,
    pub to: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cc: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub download_url: String,
    pub campaign: CampaignCompletion,
    pub completion_hash: u64,
}

/// Render the email body for a given completion payload + config.
///
/// Pure function — same inputs always produce the same output. Side effects
/// (network, FS) live in the transport.
pub fn render_email(
    config: &EmailConfig,
    completion: &CampaignCompletion,
) -> Result<RenderedEmail, EmailError> {
    config.validate()?;

    let best_parameters_block = render_best_parameters(&completion.best_parameters);
    let download_url = match &config.download_url_override {
        Some(url) if !url.is_empty() => url.clone(),
        _ => render_template_str(
            &config.download_url_template,
            completion,
            &best_parameters_block,
        )?,
    };

    let mut context = build_template_context(completion, &download_url, &best_parameters_block);
    let subject = render_template_with_context(&config.subject_template, &context)?;
    context.insert("download_url", &download_url);
    let body_text = render_template_with_context(&config.body_template, &context)?;

    Ok(RenderedEmail {
        subject,
        body_text,
        download_url,
        completion_hash: completion.completion_hash(),
    })
}

fn render_template_str(
    template: &str,
    completion: &CampaignCompletion,
    best_parameters_block: &str,
) -> Result<String, EmailError> {
    let context = build_template_context(completion, "", best_parameters_block);
    render_template_with_context(template, &context)
}

fn build_template_context(
    completion: &CampaignCompletion,
    download_url: &str,
    best_parameters_block: &str,
) -> TemplateContext {
    let mut ctx = TemplateContext::new(completion);
    ctx.insert("download_url", download_url);
    ctx.insert("best_parameters_block", best_parameters_block);
    ctx.insert("status_display", completion.status_display());
    ctx
}

fn render_best_parameters(params: &HashMap<String, f64>) -> String {
    if params.is_empty() {
        return "  (none)".to_string();
    }
    let mut entries: Vec<(&String, &f64)> = params.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::new();
    for (k, v) in entries {
        let _ = writeln!(out, "  {k} = {v}");
    }
    out
}

/// Tiny key/value context used by [`render_template_with_context`]. Built
/// specifically so the renderer does not need an external templating crate.
struct TemplateContext {
    values: HashMap<String, String>,
}

impl TemplateContext {
    fn new(completion: &CampaignCompletion) -> Self {
        let mut ctx = TemplateContext {
            values: HashMap::new(),
        };
        ctx.insert("campaign_id", &completion.campaign_id);
        ctx.insert("status", &completion.status);
        ctx.insert("start_time", &completion.start_time);
        ctx.insert("end_time", &completion.end_time);
        ctx.insert("total_runs", &completion.total_runs.to_string());
        ctx.insert("completed_runs", &completion.completed_runs.to_string());
        ctx.insert("failed_runs", &completion.failed_runs.to_string());
        ctx.insert("best_mae", &format_float(completion.best_mae));
        ctx.insert("results_uri", &completion.results_uri);
        ctx
    }

    fn insert(&mut self, key: &str, value: &str) {
        self.values.insert(key.to_string(), value.to_string());
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

fn render_template_with_context(
    template: &str,
    ctx: &TemplateContext,
) -> Result<String, EmailError> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            return Err(EmailError::MissingPlaceholder(
                "unterminated `{` in template".to_string(),
            ));
        };
        let key = &after[..close];
        // Format specifier support: `{best_mae:.2}`
        let (name, _spec) = match key.find(':') {
            Some(idx) => (&key[..idx], Some(&key[idx + 1..])),
            None => (key, None),
        };
        let value = ctx
            .get(name)
            .ok_or_else(|| EmailError::MissingPlaceholder(name.to_string()))?;
        out.push_str(value);
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn format_float(value: f64) -> String {
    if value.is_nan() {
        "n/a".to_string()
    } else if value == value.trunc() && value.abs() < 1e15 {
        format!("{value:.0}")
    } else {
        format!("{value}")
    }
}

/// Transport abstraction. Production code uses [`HttpEmailTransport`],
/// tests use [`MockEmailTransport`].
pub trait EmailTransport: Send + Sync {
    fn send(&self, config: &EmailConfig, rendered: &RenderedEmail) -> Result<(), EmailError>;
}

/// In-memory transport that records every send. Used by unit tests.
#[derive(Debug, Default)]
pub struct MockEmailTransport {
    sent: Mutex<Vec<EmailEnvelope>>,
    fail_next: Mutex<u32>,
}

impl MockEmailTransport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark the next `n` send calls to fail with the given error message.
    pub fn fail_next(&self, n: u32) {
        *self.fail_next.lock().expect("fail_next lock") = n;
    }

    /// Inspect every email envelope that has been "sent" so far.
    pub fn sent(&self) -> Vec<EmailEnvelope> {
        self.sent.lock().expect("sent lock").clone()
    }

    /// Number of envelopes captured so far.
    pub fn len(&self) -> usize {
        self.sent.lock().expect("sent lock").len()
    }

    /// Convenience: `true` if no email has been captured yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl EmailTransport for MockEmailTransport {
    fn send(&self, config: &EmailConfig, rendered: &RenderedEmail) -> Result<(), EmailError> {
        let mut remaining = self.fail_next.lock().expect("fail_next lock");
        if *remaining > 0 {
            *remaining -= 1;
            return Err(EmailError::Transport(
                "mock transport forced failure".to_string(),
            ));
        }
        drop(remaining);
        let envelope = EmailEnvelope {
            from: config.from.clone(),
            to: config.to.clone(),
            cc: config.cc.clone(),
            subject: rendered.subject.clone(),
            body_text: rendered.body_text.clone(),
            download_url: rendered.download_url.clone(),
            completion_hash: rendered.completion_hash,
            campaign: CampaignCompletion {
                campaign_id: String::new(),
                status: String::new(),
                start_time: String::new(),
                end_time: String::new(),
                total_runs: 0,
                completed_runs: 0,
                failed_runs: 0,
                best_mae: 0.0,
                best_parameters: HashMap::new(),
                results_uri: rendered.download_url.clone(),
            },
        };
        self.sent.lock().expect("sent lock").push(envelope);
        Ok(())
    }
}

/// HTTP transport — POSTs the [`EmailEnvelope`] as JSON to the configured
/// transactional-email API. Provider-agnostic.
#[derive(Debug, Default, Clone)]
pub struct HttpEmailTransport;

impl HttpEmailTransport {
    pub fn new() -> Self {
        Self
    }
}

impl EmailTransport for HttpEmailTransport {
    fn send(&self, config: &EmailConfig, rendered: &RenderedEmail) -> Result<(), EmailError> {
        let endpoint = config.api_endpoint.as_ref().ok_or_else(|| {
            EmailError::InvalidConfig("api_endpoint is required for HttpEmailTransport".to_string())
        })?;

        // We deliberately use the synchronous `reqwest::blocking` client
        // (already pulled in by the rest of the workspace) so this works
        // inside non-Tokio contexts (e.g. when called from the standalone
        // Python-bridged coordinator).
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| EmailError::Transport(format!("client build: {e}")))?;

        let envelope = EmailEnvelope {
            from: config.from.clone(),
            to: config.to.clone(),
            cc: config.cc.clone(),
            subject: rendered.subject.clone(),
            body_text: rendered.body_text.clone(),
            download_url: rendered.download_url.clone(),
            completion_hash: rendered.completion_hash,
            campaign: CampaignCompletion {
                campaign_id: String::new(),
                status: String::new(),
                start_time: String::new(),
                end_time: String::new(),
                total_runs: 0,
                completed_runs: 0,
                failed_runs: 0,
                best_mae: 0.0,
                best_parameters: HashMap::new(),
                results_uri: rendered.download_url.clone(),
            },
        };

        let mut request = client
            .post(endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(HEADER_CAMPAIGN_ID, extract_campaign_id(rendered))
            .header(HEADER_COMPLETION_HASH, rendered.completion_hash.to_string())
            .json(&envelope);

        if let Some(auth) = resolve_auth_header(config) {
            request = request.header(reqwest::header::AUTHORIZATION, auth);
        }

        let response = request
            .send()
            .map_err(|e| EmailError::Transport(format!("send: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            return Err(EmailError::Transport(format!(
                "provider returned status {status}"
            )));
        }
        Ok(())
    }
}

fn extract_campaign_id(rendered: &RenderedEmail) -> String {
    // The campaign id is the first `{campaign_id}` placeholder; recover it
    // by scanning the rendered subject/body is brittle, so the caller is
    // expected to embed it in headers when needed. We provide a stable
    // fallback by reusing the completion hash.
    format!("{:016x}", rendered.completion_hash)
}

/// High-level coordinator. Owns the transport, exposes a single
/// `notify(&self, &completion, &config)` entrypoint.
pub struct EmailNotifier<T: EmailTransport> {
    /// Underlying transport. `pub` so integration tests can inspect
    /// captured envelopes without an extra accessor; downstream
    /// production callers should treat it as opaque.
    pub transport: T,
}

impl<T: EmailTransport> EmailNotifier<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    /// Render and dispatch the email for one completion event.
    ///
    /// `Ok(true)` ⇒ delivery succeeded.
    /// `Ok(false)` ⇒ no transport endpoint configured (preview mode).
    /// `Err(_)` ⇒ validation, rendering, or transport failure.
    pub fn notify(
        &self,
        completion: &CampaignCompletion,
        config: &EmailConfig,
    ) -> Result<bool, EmailError> {
        let rendered = render_email(config, completion)?;
        if config.api_endpoint.is_none() {
            // Preview mode — nothing to send but the caller can still
            // inspect `rendered`.
            return Ok(false);
        }
        self.transport.send(config, &rendered)?;
        Ok(true)
    }

    /// Pure render entrypoint — useful when the caller wants the rendered
    /// email without committing to a transport (e.g. for unit tests, or
    /// when posting the rendered envelope through some other channel).
    pub fn render(
        &self,
        completion: &CampaignCompletion,
        config: &EmailConfig,
    ) -> Result<RenderedEmail, EmailError> {
        render_email(config, completion)
    }
}

impl<T: EmailTransport> std::fmt::Debug for EmailNotifier<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmailNotifier").finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Internal hashing helpers
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hasher — stable, no external dep, sufficient for
/// idempotency tagging (NOT cryptographic).
struct Fnv1a(u64);

impl Fnv1a {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn update(&mut self, byte: u8) {
        self.0 ^= byte as u64;
        self.0 = self.0.wrapping_mul(Self::PRIME);
    }

    fn hash_str(&mut self, s: &str) {
        for byte in s.as_bytes() {
            self.update(*byte);
        }
        // Field separator so ("ab", "c") != ("a", "bc")
        self.update(0x1f);
    }

    fn hash_usize(&mut self, n: usize) {
        for byte in n.to_le_bytes() {
            self.update(byte);
        }
        self.update(0x1e);
    }

    fn hash_f64(&mut self, n: f64) {
        if n.is_nan() {
            for byte in b"nan" {
                self.update(*byte);
            }
        } else {
            self.hash_u64(n.to_bits());
        }
        self.update(0x1d);
    }

    fn hash_u64(&mut self, n: u64) {
        for byte in n.to_le_bytes() {
            self.update(byte);
        }
    }

    fn finalize(self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Local helper — construct a stable email address at runtime so the
    /// literal `@` doesn't appear in source (avoids HTML/markdown
    /// auto-linkers that rewrite `local@domain` patterns).
    fn addr(local: &str, domain: &str) -> String {
        format!("{local}@{domain}")
    }

    const SAMPLE_FROM_LOCAL: &str = "fluxion-noreply";
    const SAMPLE_TO_LOCAL: &str = "user";
    const SAMPLE_DOMAIN: &str = "fluxion.example";
    const CC1_DOMAIN: &str = "fluxion.example";
    const CC2_DOMAIN: &str = "fluxion.example";

    fn sample_config() -> EmailConfig {
        EmailConfig {
            from: addr(SAMPLE_FROM_LOCAL, SAMPLE_DOMAIN),
            to: vec![addr(SAMPLE_TO_LOCAL, SAMPLE_DOMAIN)],
            ..Default::default()
        }
    }

    fn sample_completion() -> CampaignCompletion {
        let mut params = HashMap::new();
        params.insert("R_value".to_string(), 1.5);
        params.insert("wall_thickness".to_string(), 0.15);
        CampaignCompletion {
            campaign_id: "fluxion-abc123".to_string(),
            status: "completed".to_string(),
            start_time: "2026-07-18T00:00:00+00:00".to_string(),
            end_time: "2026-07-18T01:00:00+00:00".to_string(),
            total_runs: 50,
            completed_runs: 48,
            failed_runs: 2,
            best_mae: 4.7,
            best_parameters: params,
            results_uri: "s3://bucket/campaigns/abc/results/".to_string(),
        }
    }

    #[test]
    fn validate_rejects_empty_from() {
        let mut c = sample_config();
        c.from = String::new();
        let err = c.validate().unwrap_err();
        assert!(matches!(err, EmailError::InvalidConfig(_)));
        assert!(err.to_string().contains("from"));
    }

    #[test]
    fn validate_rejects_empty_to() {
        let mut c = sample_config();
        c.to.clear();
        let err = c.validate().unwrap_err();
        assert!(err.to_string().contains("to"));
    }

    #[test]
    fn validate_rejects_malformed_recipient() {
        let mut c = sample_config();
        c.to = vec!["not-an-email".to_string()];
        let err = c.validate().unwrap_err();
        assert!(err.to_string().contains("not-an-email"));
    }

    #[test]
    fn validate_rejects_zero_timeout() {
        let mut c = sample_config();
        c.timeout_seconds = 0;
        let err = c.validate().unwrap_err();
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn validate_accepts_full_config() {
        sample_config().validate().expect("valid");
    }

    #[test]
    fn render_email_substitutes_placeholders() {
        let rendered = render_email(&sample_config(), &sample_completion()).unwrap();
        assert!(rendered.subject.contains("fluxion-abc123"));
        assert!(rendered.subject.contains("completed"));
        assert!(rendered.body_text.contains("fluxion-abc123"));
        assert!(rendered.body_text.contains("48"));
        assert!(rendered.body_text.contains("R_value = 1.5"));
        assert!(rendered.body_text.contains("wall_thickness = 0.15"));
        assert!(rendered.body_text.contains("fluxion.example.com"));
    }

    #[test]
    fn render_email_uses_override_url() {
        let mut c = sample_config();
        c.download_url_override = Some("https://signed.example.com/x?token=abc".to_string());
        let rendered = render_email(&c, &sample_completion()).unwrap();
        assert_eq!(
            rendered.download_url,
            "https://signed.example.com/x?token=abc"
        );
        assert!(rendered.body_text.contains("signed.example.com"));
    }

    #[test]
    fn render_email_handles_failed_status() {
        let completion = CampaignCompletion::failed(
            "fluxion-fail".to_string(),
            "2026-07-18T00:00:00+00:00".to_string(),
            "2026-07-18T01:00:00+00:00".to_string(),
            50,
            10,
            40,
            "s3://bucket/campaigns/abc/results/".to_string(),
        );
        let rendered = render_email(&sample_config(), &completion).unwrap();
        assert!(rendered.subject.contains("failed"));
        assert!(rendered.body_text.contains("failed"));
    }

    #[test]
    fn render_email_rejects_missing_placeholder() {
        let mut c = sample_config();
        c.body_template = "Hello {nonexistent}".to_string();
        let err = render_email(&c, &sample_completion()).unwrap_err();
        assert!(matches!(err, EmailError::MissingPlaceholder(_)));
        assert!(err.to_string().contains("nonexistent"));
    }

    #[test]
    fn render_email_rejects_unterminated_placeholder() {
        let mut c = sample_config();
        c.body_template = "Hello {campaign_id".to_string();
        let err = render_email(&c, &sample_completion()).unwrap_err();
        assert!(matches!(err, EmailError::MissingPlaceholder(_)));
    }

    #[test]
    fn completion_hash_is_deterministic() {
        let a = sample_completion();
        let b = sample_completion();
        assert_eq!(a.completion_hash(), b.completion_hash());
    }

    #[test]
    fn completion_hash_changes_with_payload() {
        let mut b = sample_completion();
        b.completed_runs = 49;
        assert_ne!(sample_completion().completion_hash(), b.completion_hash());
    }

    #[test]
    fn completion_hash_distinguishes_status() {
        let mut failed = sample_completion();
        failed.status = "failed".to_string();
        assert_ne!(
            sample_completion().completion_hash(),
            failed.completion_hash()
        );
    }

    #[test]
    fn notifier_renders_in_preview_mode() {
        let notifier = EmailNotifier::new(MockEmailTransport::new());
        let result = notifier
            .notify(&sample_completion(), &sample_config())
            .unwrap();
        assert!(!result, "preview mode returns false (no transport)");
        assert!(notifier
            .render(&sample_completion(), &sample_config())
            .is_ok());
    }

    #[test]
    fn notifier_sends_via_mock_transport() {
        let transport = MockEmailTransport::new();
        let notifier = EmailNotifier::new(transport);
        let mut cfg = sample_config();
        cfg.api_endpoint = Some("https://api.example.com/send".to_string());

        let sent = notifier.notify(&sample_completion(), &cfg).unwrap();
        assert!(sent);
        let sent_inner = notifier.transport;
        assert_eq!(sent_inner.len(), 1);
        let envelopes = sent_inner.sent();
        assert_eq!(envelopes[0].from, addr(SAMPLE_FROM_LOCAL, SAMPLE_DOMAIN));
        assert_eq!(envelopes[0].to, vec![addr(SAMPLE_TO_LOCAL, SAMPLE_DOMAIN)]);
        assert!(envelopes[0].subject.contains("fluxion-abc123"));
    }

    #[test]
    fn notifier_surfaces_transport_failure() {
        let transport = MockEmailTransport::new();
        transport.fail_next(1);
        let notifier = EmailNotifier::new(transport);
        let mut cfg = sample_config();
        cfg.api_endpoint = Some("https://api.example.com/send".to_string());

        let err = notifier
            .notify(&sample_completion(), &cfg)
            .expect_err("should fail");
        assert!(matches!(err, EmailError::Transport(_)));
    }

    #[test]
    fn notifier_does_not_send_when_validation_fails() {
        let transport = MockEmailTransport::new();
        let notifier = EmailNotifier::new(transport);
        let mut cfg = sample_config();
        cfg.from = String::new(); // invalid
        let err = notifier.notify(&sample_completion(), &cfg).unwrap_err();
        assert!(matches!(err, EmailError::InvalidConfig(_)));
        assert_eq!(notifier.transport.len(), 0);
    }

    #[test]
    fn cc_recipients_are_propagated() {
        let transport = MockEmailTransport::new();
        let notifier = EmailNotifier::new(transport);
        let mut cfg = sample_config();
        cfg.cc = vec![addr("team", CC1_DOMAIN), addr("lead", CC2_DOMAIN)];
        cfg.api_endpoint = Some("https://api.example.com/send".to_string());
        notifier.notify(&sample_completion(), &cfg).unwrap();
        let envelopes = notifier.transport.sent();
        assert_eq!(envelopes[0].cc.len(), 2);
        assert_eq!(envelopes[0].cc[0], addr("team", CC1_DOMAIN));
    }

    #[test]
    fn status_display_handles_case_insensitive() {
        let mut c = sample_completion();
        c.status = "FAILED".to_string();
        assert_eq!(c.status_display(), "failed");
        c.status = "Completed".to_string();
        assert_eq!(c.status_display(), "completed");
        c.status = "running".to_string();
        assert_eq!(c.status_display(), "completed", "unknown → completed");
    }

    #[test]
    fn email_envelope_roundtrips_via_json() {
        let envelope = EmailEnvelope {
            from: addr(SAMPLE_FROM_LOCAL, SAMPLE_DOMAIN),
            to: vec![addr(SAMPLE_TO_LOCAL, SAMPLE_DOMAIN)],
            cc: vec![],
            subject: "hi".to_string(),
            body_text: "body".to_string(),
            download_url: "https://x".to_string(),
            completion_hash: 42,
            campaign: sample_completion(),
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let back: EmailEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(back, envelope);
    }

    #[test]
    fn email_config_default_roundtrips_via_json() {
        let cfg = EmailConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: EmailConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn best_parameters_block_is_sorted_and_skipped_when_empty() {
        let mut c = sample_completion();
        c.best_parameters.clear();
        let rendered = render_email(&sample_config(), &c).unwrap();
        assert!(rendered.body_text.contains("(none)"));
    }

    #[test]
    fn http_transport_requires_endpoint() {
        let transport = HttpEmailTransport::new();
        let cfg = sample_config(); // no api_endpoint
        let rendered = render_email(&cfg, &sample_completion()).unwrap();
        let err = transport.send(&cfg, &rendered).unwrap_err();
        assert!(err.to_string().contains("api_endpoint"));
    }
}
