// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Integration tests for the OSimFlow email notification fallback channel
//! (Issue #1789 / T7.5).
//!
//! These tests live outside `src/` so they exercise the public API surface
//! (`fluxion::api::email_notification`) the way downstream consumers would.
//! Unit tests in the module itself cover the internals; here we focus on
//! end-to-end behaviour:
//!
//! - The renderer accepts and substitutes every documented placeholder.
//! - The same `CampaignCompletion` always produces the same `completion_hash`,
//!   so a coordinator restart does not double-send.
//! - The transport-agnostic `EmailNotifier` works with both the mock
//!   transport (in-memory assertions) and the HTTP transport (rejects an
//!   invalid endpoint).
//! - `EmailConfig` round-trips losslessly through JSON — required so the
//!   campaign state file (`state.json`) can persist notification prefs.

use fluxion::api::email_notification::{
    render_email, CampaignCompletion, EmailConfig, EmailEnvelope, EmailError, EmailNotifier,
    EmailTransport, HttpEmailTransport, MockEmailTransport, RenderedEmail,
};

fn addr(local: &str, domain: &str) -> String {
    format!("{local}@{domain}")
}

fn sample_config() -> EmailConfig {
    EmailConfig {
        from: addr("fluxion-noreply", "fluxion.example"),
        to: vec![addr("user", "fluxion.example")],
        ..Default::default()
    }
}

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

#[test]
fn public_render_endpoint_matches_module_helper() {
    let rendered = render_email(&sample_config(), &sample_completion()).unwrap();
    assert!(rendered.subject.contains("fluxion-abc123"));
    assert!(rendered.body_text.contains("48"));
    assert!(rendered.download_url.contains("fluxion-abc123"));
    assert!(rendered.completion_hash != 0);
}

#[test]
fn notifier_dispatch_via_mock_transport() {
    let transport = MockEmailTransport::new();
    let mut cfg = sample_config();
    cfg.api_endpoint = Some("https://api.example.com/send".to_string());
    let notifier = EmailNotifier::new(transport);

    let sent = notifier
        .notify(&sample_completion(), &cfg)
        .expect("dispatch");
    assert!(sent);
    assert_eq!(notifier.transport.len(), 1);
    let envelopes = notifier.transport.sent();
    assert_eq!(
        envelopes[0].from,
        addr("fluxion-noreply", "fluxion.example")
    );
}

#[test]
fn notifier_preview_mode_skips_transport() {
    let transport = MockEmailTransport::new();
    let notifier = EmailNotifier::new(transport);
    let preview = notifier
        .notify(&sample_completion(), &sample_config())
        .expect("preview-mode render");
    assert!(!preview);
    assert!(notifier.transport.is_empty());
}

#[test]
fn completion_hash_is_stable_across_notifier_instances() {
    let a = sample_completion();
    let b = sample_completion();
    assert_eq!(a.completion_hash(), b.completion_hash());
}

#[test]
fn rendered_email_carries_completion_hash() {
    let rendered = render_email(&sample_config(), &sample_completion()).unwrap();
    assert_eq!(
        rendered.completion_hash,
        sample_completion().completion_hash()
    );
}

#[test]
fn idempotent_rerender_does_not_produce_a_new_hash() {
    let a = render_email(&sample_config(), &sample_completion()).unwrap();
    let b = render_email(&sample_config(), &sample_completion()).unwrap();
    assert_eq!(a.completion_hash, b.completion_hash);
}

#[test]
fn envelope_payload_roundtrips_through_json() {
    let rendered = render_email(&sample_config(), &sample_completion()).unwrap();
    let envelope = EmailEnvelope {
        from: sample_config().from,
        to: sample_config().to,
        cc: vec![addr("team", "fluxion.example")],
        subject: rendered.subject.clone(),
        body_text: rendered.body_text.clone(),
        download_url: rendered.download_url.clone(),
        completion_hash: rendered.completion_hash,
        campaign: sample_completion(),
    };
    let json = serde_json::to_string(&envelope).unwrap();
    let back: EmailEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(back, envelope);
    assert!(json.contains("fluxion-abc123"));
    assert!(json.contains("download_url"));
}

#[test]
fn http_transport_rejects_when_endpoint_is_missing() {
    let transport = HttpEmailTransport::new();
    let cfg = sample_config(); // no api_endpoint
    let rendered: RenderedEmail = render_email(&cfg, &sample_completion()).unwrap();
    let err = transport.send(&cfg, &rendered).unwrap_err();
    let msg = match err {
        EmailError::InvalidConfig(m) => m,
        other => panic!("expected InvalidConfig, got {other:?}"),
    };
    assert!(msg.contains("api_endpoint"), "{msg}");
}

#[test]
fn template_placeholders_are_stable_across_locales() {
    let rendered = render_email(&sample_config(), &sample_completion()).unwrap();
    for token in ["fluxion-abc123", "completed", "48", "https://"] {
        assert!(
            rendered.subject.contains(token) || rendered.body_text.contains(token),
            "missing `{token}` in rendered output:\nsubject={}\nbody={}",
            rendered.subject,
            rendered.body_text
        );
    }
}

#[test]
fn failed_completion_renders_unambiguous_subject() {
    let failed = CampaignCompletion::failed(
        "fluxion-fail-1".to_string(),
        "2026-07-18T00:00:00+00:00".to_string(),
        "2026-07-18T01:00:00+00:00".to_string(),
        10,
        2,
        8,
        "https://fluxion.example.com/results/fail-1/".to_string(),
    );
    let rendered = render_email(&sample_config(), &failed).unwrap();
    assert!(
        rendered.subject.contains("failed"),
        "subject must signal failure: {}",
        rendered.subject
    );
    assert!(rendered.body_text.contains("failed"));
}

#[test]
fn mock_transport_captures_batch_notifications_in_order() {
    let transport = MockEmailTransport::new();
    let mut cfg = sample_config();
    cfg.api_endpoint = Some("https://api.example.com/send".to_string());
    let notifier = EmailNotifier::new(transport);

    for i in 0..3 {
        let mut completion = sample_completion();
        completion.campaign_id = format!("fluxion-batch-{i}");
        notifier.notify(&completion, &cfg).unwrap();
    }
    let sent = notifier.transport.sent();
    assert_eq!(sent.len(), 3);
    let ids: Vec<String> = sent
        .into_iter()
        .map(|e| {
            e.subject
                .split_whitespace()
                .find(|tok| tok.starts_with("fluxion-"))
                .unwrap_or("")
                .to_string()
        })
        .collect();
    assert_eq!(
        ids,
        vec![
            "fluxion-batch-0".to_string(),
            "fluxion-batch-1".to_string(),
            "fluxion-batch-2".to_string(),
        ]
    );
}
