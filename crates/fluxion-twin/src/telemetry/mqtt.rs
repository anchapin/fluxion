//! MQTT Telemetry Consumer.
//!
//! Subscribes to an MQTT broker topic, parses JSON payloads into typed telemetry
//! messages, and forwards them through a bounded [`tokio::sync::mpsc`] channel.
//!
//! rumqttc handles automatic reconnection on transient disconnects — the event
//! loop keeps polling and the broker connection is restored transparently.
//!
//! # Transport security (default: TLS)
//!
//! The consumer defaults to **MQTT-over-TLS** (`mqtts://`, port 8883) using
//! rustls with the platform trust store. Server certificates are **validated**
//! by default. Plaintext (`mqtt://` / `tcp://`) broker URLs are rejected unless
//! `FLUXION_MQTT_ALLOW_INSECURE=true` is set, and certificate validation can be
//! disabled with `FLUXION_MQTT_INSECURE=1` for local development only. See
//! [`MqttTelemetryConsumer::connect`].
//!
//! # Example
//!
//! ```ignore
//! let (consumer, mut rx) = MqttTelemetryConsumer::connect(
//!     "mqtts://broker.local:8883",
//!     "fluxion/sensors/#",
//! )
//! .await?;
//!
//! tokio::spawn(async move { let _ = consumer.start().await; });
//!
//! while let Some(msg) = rx.recv().await {
//!     println!("sensor={} temp={:?}", msg.sensor_id, msg.temperature_c);
//! }
//! ```

// rumqttc's `ConnectionError` is inherently large (~144 B). Boxing every
// Result path would add noise without real benefit, so we suppress the lint
// at the module level.
#![allow(clippy::result_large_err)]

use rumqttc::{
    AsyncClient, Event, EventLoop, MqttOptions, Packet, QoS, TlsConfiguration, Transport,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;

/// Default bounded channel capacity.
///
/// A bounded channel prevents unbounded memory growth if the consumer drains
/// messages slower than the MQTT broker delivers them.
const CHANNEL_CAPACITY: usize = 1024;

/// Default MQTT-over-TLS port (secure) when not specified in the broker URL.
const DEFAULT_MQTTS_PORT: u16 = 8883;

/// Default plaintext MQTT port — only used when plaintext transport is
/// explicitly permitted via [`ENV_ALLOW_PLAINTEXT`].
const DEFAULT_MQTT_PORT: u16 = 1883;

/// Env var: when truthy, permits plaintext (`mqtt://` / `tcp://`) broker URLs.
///
/// Plaintext means telemetry payloads travel unencrypted; intended only for
/// local development.
const ENV_ALLOW_PLAINTEXT: &str = "FLUXION_MQTT_ALLOW_INSECURE";

/// Env var: when truthy, skips TLS server-certificate validation (e.g. for
/// self-signed brokers). **Disables all certificate checking** — local dev only.
const ENV_INSECURE_CERTS: &str = "FLUXION_MQTT_INSECURE";

/// Errors produced by the MQTT telemetry consumer.
#[derive(Error, Debug)]
pub enum MqttTelemetryError {
    /// Broker URL or topic failed validation (empty, malformed, insecure
    /// transport requested without an explicit opt-in, etc.).
    #[error("invalid broker configuration: {0}")]
    InvalidConfig(String),

    /// Underlying rumqttc connection error (transient — auto-reconnect handled).
    #[error("MQTT connection error: {0}")]
    Connection(#[from] rumqttc::ConnectionError),

    /// rumqttc client request error (subscribe, publish, etc.).
    #[error("MQTT client error: {0}")]
    Client(#[from] rumqttc::ClientError),

    /// JSON payload could not be deserialized into [`MqttTelemetryMessage`].
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Telemetry message parsed from an MQTT JSON payload.
///
/// The `sensor_id` and `timestamp` fields are required. Measurement fields are
/// optional ([`Option`]) so that heterogeneous sensor types can share the same
/// topic — a temperature-only sensor populates [`Self::temperature_c`] and leaves
/// the others `None`.
///
/// # Wire Format
///
/// ```json
/// {
///     "sensor_id": "zone-1-temp",
///     "timestamp": 1700000000,
///     "temperature_c": 22.5,
///     "humidity_pct": 45.0,
///     "power_w": 150.0
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MqttTelemetryMessage {
    /// Unique sensor identifier (e.g. MAC address, serial number, zone name).
    pub sensor_id: String,

    /// Unix epoch timestamp in seconds.
    pub timestamp: i64,

    /// Air temperature in degrees Celsius.
    #[serde(default)]
    pub temperature_c: Option<f64>,

    /// Relative humidity as a percentage (0–100).
    #[serde(default)]
    pub humidity_pct: Option<f64>,

    /// Instantaneous power draw in watts.
    #[serde(default)]
    pub power_w: Option<f64>,
}

impl MqttTelemetryMessage {
    /// Parse a JSON payload byte slice into a telemetry message.
    ///
    /// Returns [`MqttTelemetryError::Json`] if the payload is not valid JSON or
    /// is missing required fields (`sensor_id`, `timestamp`). Never panics.
    pub fn parse(payload: &[u8]) -> Result<Self, MqttTelemetryError> {
        Ok(serde_json::from_slice(payload)?)
    }
}

impl TryFrom<&[u8]> for MqttTelemetryMessage {
    type Error = MqttTelemetryError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Self::parse(bytes)
    }
}

impl TryFrom<Vec<u8>> for MqttTelemetryMessage {
    type Error = MqttTelemetryError;

    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::parse(&bytes)
    }
}

/// MQTT telemetry consumer.
///
/// Connects to a broker, subscribes to a topic with QoS [`AtLeastOnce`][QoS], and
/// forwards parsed [`MqttTelemetryMessage`] values through a bounded
/// [`mpsc::Receiver`]. rumqttc handles automatic reconnection on disconnect.
pub struct MqttTelemetryConsumer {
    #[allow(dead_code)]
    client: AsyncClient,
    eventloop: EventLoop,
    tx: mpsc::Sender<MqttTelemetryMessage>,
    topic: String,
}

impl MqttTelemetryConsumer {
    /// Connect to `broker` and subscribe to `topic` with QoS AtLeastOnce.
    ///
    /// Returns the consumer and the receiving end of a bounded channel
    /// (capacity 1024). Call [`Self::start`] to begin consuming messages.
    ///
    /// # Transport Security (default: TLS)
    ///
    /// The default transport is **MQTT-over-TLS** (`mqtts://`, port 8883) using
    /// rustls with the platform trust store. Server certificates are validated.
    ///
    /// | Input | Transport | Host | Port |
    /// |-------|-----------|------|------|
    /// | `mqtts://broker.local` | TLS (validated) | `broker.local` | `8883` |
    /// | `mqtts://broker.local:8883` | TLS (validated) | `broker.local` | `8883` |
    /// | `broker.local` | TLS (validated) | `broker.local` | `8883` (default) |
    /// | `broker.local:1883` | TLS (validated) | `broker.local` | `1883` |
    /// | `mqtt://broker.local:1883` | **plaintext** | `broker.local` | `1883` |
    /// | `tcp://10.0.0.5:1883` | **plaintext** | `10.0.0.5` | `1883` |
    ///
    /// Plaintext URLs (`mqtt://` / `tcp://`) are **rejected** unless the
    /// `FLUXION_MQTT_ALLOW_INSECURE` environment variable is set to a truthy
    /// value (`1`/`true`/`yes`/`on`); telemetry would otherwise travel
    /// unencrypted.
    ///
    /// Certificate validation can be disabled (e.g. for a self-signed local
    /// broker) by setting `FLUXION_MQTT_INSECURE=1`. This is **dangerous** and
    /// logged as a warning — never use it in production.
    ///
    /// # Errors
    ///
    /// Returns [`MqttTelemetryError::InvalidConfig`] if the broker URL or topic
    /// is empty/malformed, or if a plaintext URL is supplied without
    /// `FLUXION_MQTT_ALLOW_INSECURE=true`.
    pub async fn connect(
        broker: &str,
        topic: &str,
    ) -> Result<(Self, mpsc::Receiver<MqttTelemetryMessage>), MqttTelemetryError> {
        if broker.trim().is_empty() {
            return Err(MqttTelemetryError::InvalidConfig(
                "broker URL must not be empty".to_string(),
            ));
        }
        if topic.trim().is_empty() {
            return Err(MqttTelemetryError::InvalidConfig(
                "topic must not be empty".to_string(),
            ));
        }

        let (scheme, host, port) = parse_broker_url(broker)?;

        let allow_plaintext = env_flag(ENV_ALLOW_PLAINTEXT);
        let insecure_certs = env_flag(ENV_INSECURE_CERTS);
        let transport = resolve_transport(scheme, allow_plaintext, insecure_certs)?;

        let mut mqttoptions = MqttOptions::new("fluxion-twin-consumer", host, port);
        mqttoptions.set_keep_alive(Duration::from_secs(5));

        match transport {
            ResolvedTransport::Tls { verify_certs: true } => {
                // rustls with the platform trust store; certs are validated.
                mqttoptions.set_transport(Transport::tls_with_default_config());
            }
            ResolvedTransport::Tls {
                verify_certs: false,
            } => {
                log::warn!(
                    "FLUXION_MQTT_INSECURE is set: MQTT server certificates will NOT be \
                     validated. This disables TLS trust verification and must only be \
                     used for local development against self-signed brokers."
                );
                mqttoptions.set_transport(Transport::tls_with_config(TlsConfiguration::from(
                    insecure_tls_config(),
                )));
            }
            ResolvedTransport::Plaintext => {
                log::warn!(
                    "FLUXION_MQTT_ALLOW_INSECURE is set: connecting to MQTT broker '{broker}' \
                     over plaintext TCP. Telemetry payloads will be unencrypted."
                );
                // rumqttc's default transport is already `Transport::Tcp`.
            }
        }

        let (client, eventloop) = AsyncClient::new(mqttoptions, CHANNEL_CAPACITY);
        client.subscribe(topic, QoS::AtLeastOnce).await?;

        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);

        Ok((
            Self {
                client,
                eventloop,
                tx,
                topic: topic.to_string(),
            },
            rx,
        ))
    }

    /// Run the event loop: poll for `Publish` packets, parse JSON payloads into
    /// [`MqttTelemetryMessage`] values, and forward them through the bounded
    /// channel.
    ///
    /// # Lifecycle
    ///
    /// - **Normal operation** — polls indefinitely, parsing and forwarding
    ///   messages.
    /// - **Transient disconnect** — rumqttc auto-reconnects; the loop continues
    ///   after a brief backoff sleep.
    /// - **Channel receiver dropped** — `tx.send` fails; the loop exits cleanly
    ///   with `Ok(())`.
    /// - **Malformed payload** — logs a warning and continues (never panics).
    ///
    /// # Errors
    ///
    /// Only returns `Err` if a fatal, unrecoverable error occurs. In practice the
    /// loop tends to run indefinitely until the consumer is dropped or the channel
    /// receiver is dropped.
    pub async fn start(mut self) -> Result<(), MqttTelemetryError> {
        loop {
            match self.eventloop.poll().await {
                Ok(Event::Incoming(Packet::Publish(publish))) => {
                    if topic_matches(&self.topic, &publish.topic) {
                        match MqttTelemetryMessage::parse(publish.payload.as_ref()) {
                            Ok(msg) => {
                                if self.tx.send(msg).await.is_err() {
                                    // Receiver dropped — consumer no longer needed.
                                    break;
                                }
                            }
                            Err(e) => {
                                log::warn!(
                                    "Failed to parse MQTT payload on topic '{}': {}",
                                    publish.topic,
                                    e
                                );
                                // Continue — don't crash on a single bad message.
                            }
                        }
                    }
                }
                // Other incoming/outgoing events (SubAck, ConnAck, etc.) — ignore.
                Ok(_) => {}
                Err(e) => {
                    // rumqttc auto-reconnects on the next poll(). Brief backoff
                    // to avoid tight-looping on persistent failures (e.g. DNS).
                    log::warn!("MQTT connection error (auto-reconnecting): {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Scheme recognised in the broker URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BrokerScheme {
    /// `mqtts://` — MQTT-over-TLS (secure). Also the default for bare hosts.
    Tls,
    /// `mqtt://` or `tcp://` — plaintext. Rejected unless explicitly permitted.
    Plaintext,
}

/// Transport chosen after combining the URL scheme with the security policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedTransport {
    /// TLS over rustls. `verify_certs == false` means certificate validation is
    /// skipped (gated behind `FLUXION_MQTT_INSECURE=1`).
    Tls { verify_certs: bool },
    /// Plaintext TCP (only chosen when `FLUXION_MQTT_ALLOW_INSECURE=true`).
    Plaintext,
}

/// Decide the transport from the URL scheme and the two policy flags.
///
/// Pure (no I/O) so it can be unit-tested directly.
fn resolve_transport(
    scheme: BrokerScheme,
    allow_plaintext: bool,
    insecure_certs: bool,
) -> Result<ResolvedTransport, MqttTelemetryError> {
    match scheme {
        BrokerScheme::Plaintext if !allow_plaintext => Err(MqttTelemetryError::InvalidConfig(
            "plaintext broker URL ('mqtt://'/'tcp://') rejected: telemetry would be \
             unencrypted. To permit plaintext for local development, set the \
             FLUXION_MQTT_ALLOW_INSECURE environment variable to a truthy value."
                .to_string(),
        )),
        BrokerScheme::Plaintext => Ok(ResolvedTransport::Plaintext),
        BrokerScheme::Tls => Ok(ResolvedTransport::Tls {
            verify_certs: !insecure_certs,
        }),
    }
}

/// Read a boolean environment flag.
///
/// Truthy values (case-insensitive): `1`, `true`, `yes`, `on`. Anything else,
/// or an unset variable, is `false`.
fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

/// A [`rustls::client::danger::ServerCertVerifier`] that accepts **any** server
/// certificate without validation.
///
/// This intentionally disables all TLS certificate checks and MUST only be used
/// for local development against brokers with self-signed certificates. It is
/// gated behind `FLUXION_MQTT_INSECURE=1` and a warning is logged when active.
#[derive(Debug)]
struct NoCertificateVerification;

impl rustls::client::danger::ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        // Broad list covering the schemes the aws-lc-rs / ring providers verify.
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::ED448,
        ]
    }
}

/// Build a rustls [`ClientConfig`] that performs **no** certificate validation.
fn insecure_tls_config() -> rustls::ClientConfig {
    rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoCertificateVerification))
        .with_no_client_auth()
}

/// Parse a broker URL into `(scheme, host, port)`.
///
/// Recognised schemes:
/// - `mqtts://` → [`BrokerScheme::Tls`], default port [`DEFAULT_MQTTS_PORT`].
/// - `mqtt://` / `tcp://` → [`BrokerScheme::Plaintext`], default port
///   [`DEFAULT_MQTT_PORT`].
/// - bare host (no scheme) → [`BrokerScheme::Tls`], default port
///   [`DEFAULT_MQTTS_PORT`].
///
/// Returns [`MqttTelemetryError::InvalidConfig`] for empty hosts or invalid ports.
fn parse_broker_url(url: &str) -> Result<(BrokerScheme, String, u16), MqttTelemetryError> {
    let (scheme, stripped) = if let Some(rest) = url.strip_prefix("mqtts://") {
        (BrokerScheme::Tls, rest)
    } else if let Some(rest) = url.strip_prefix("mqtt://") {
        (BrokerScheme::Plaintext, rest)
    } else if let Some(rest) = url.strip_prefix("tcp://") {
        (BrokerScheme::Plaintext, rest)
    } else {
        // Bare host — default to the secure transport.
        (BrokerScheme::Tls, url)
    };

    let (host, port_part) = match stripped.rsplit_once(':') {
        // Don't treat bare IPv6 address `[::1]` as host:port.
        Some((h, p)) if !h.is_empty() && !p.is_empty() && !h.ends_with(']') => (h, Some(p)),
        _ => (stripped, None),
    };

    let host = host.trim();
    if host.is_empty() {
        return Err(MqttTelemetryError::InvalidConfig(format!(
            "broker URL '{url}' has no host"
        )));
    }

    let default_port = match scheme {
        BrokerScheme::Tls => DEFAULT_MQTTS_PORT,
        BrokerScheme::Plaintext => DEFAULT_MQTT_PORT,
    };

    let port = match port_part {
        Some(raw) => raw.trim().parse::<u16>().map_err(|_| {
            MqttTelemetryError::InvalidConfig(format!("invalid port '{raw}' in broker URL"))
        })?,
        None => default_port,
    };

    Ok((scheme, host.to_string(), port))
}

/// Check if an MQTT topic matches a subscription filter.
///
/// Supports MQTT wildcard characters:
/// - `+` — matches exactly one topic level
/// - `#` — matches all remaining levels (must be the last filter level)
fn topic_matches(filter: &str, topic: &str) -> bool {
    let filter_parts: Vec<&str> = filter.split('/').collect();
    let topic_parts: Vec<&str> = topic.split('/').collect();

    for (i, fp) in filter_parts.iter().enumerate() {
        if *fp == "#" {
            // Multi-level wildcard — matches everything from here down.
            return true;
        }

        match topic_parts.get(i) {
            None => return false,
            Some(tp) if *fp == "+" || fp == tp => continue,
            Some(_) => return false,
        }
    }

    // Exact match requires same number of levels.
    filter_parts.len() == topic_parts.len()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- JSON payload parsing ----

    #[test]
    fn test_parse_temperature_only_payload() {
        let json = br#"{"sensor_id":"temp-1","timestamp":1700000000,"temperature_c":22.5}"#;
        let msg = MqttTelemetryMessage::parse(json).unwrap();

        assert_eq!(msg.sensor_id, "temp-1");
        assert_eq!(msg.timestamp, 1700000000);
        assert!((msg.temperature_c.unwrap() - 22.5).abs() < 1e-9);
        assert_eq!(msg.humidity_pct, None);
        assert_eq!(msg.power_w, None);
    }

    #[test]
    fn test_parse_humidity_only_payload() {
        let json = br#"{"sensor_id":"rh-1","timestamp":1700000000,"humidity_pct":45.0}"#;
        let msg = MqttTelemetryMessage::parse(json).unwrap();

        assert_eq!(msg.sensor_id, "rh-1");
        assert_eq!(msg.humidity_pct, Some(45.0));
        assert_eq!(msg.temperature_c, None);
        assert_eq!(msg.power_w, None);
    }

    #[test]
    fn test_parse_power_only_payload() {
        let json = br#"{"sensor_id":"pwr-1","timestamp":1700000000,"power_w":150.0}"#;
        let msg = MqttTelemetryMessage::parse(json).unwrap();

        assert_eq!(msg.sensor_id, "pwr-1");
        assert_eq!(msg.power_w, Some(150.0));
        assert_eq!(msg.temperature_c, None);
        assert_eq!(msg.humidity_pct, None);
    }

    #[test]
    fn test_parse_all_fields_payload() {
        let json = br#"{"sensor_id":"multi-1","timestamp":1700000000,"temperature_c":22.5,"humidity_pct":45.0,"power_w":150.0}"#;
        let msg = MqttTelemetryMessage::parse(json).unwrap();

        assert_eq!(msg.sensor_id, "multi-1");
        assert_eq!(msg.timestamp, 1700000000);
        assert_eq!(msg.temperature_c, Some(22.5));
        assert_eq!(msg.humidity_pct, Some(45.0));
        assert_eq!(msg.power_w, Some(150.0));
    }

    #[test]
    fn test_parse_negative_temperature() {
        let json = br#"{"sensor_id":"outdoor","timestamp":1700000000,"temperature_c":-15.3}"#;
        let msg = MqttTelemetryMessage::parse(json).unwrap();

        assert!((msg.temperature_c.unwrap() - (-15.3)).abs() < 1e-9);
    }

    #[test]
    fn test_deserialize_roundtrip() {
        let original = MqttTelemetryMessage {
            sensor_id: "rt-1".to_string(),
            timestamp: 1234567890,
            temperature_c: Some(21.0),
            humidity_pct: None,
            power_w: Some(99.9),
        };

        let json = serde_json::to_vec(&original).unwrap();
        let parsed: MqttTelemetryMessage = serde_json::from_slice(&json).unwrap();

        assert_eq!(original, parsed);
    }

    // ---- Malformed / missing-field handling ----

    #[test]
    fn test_malformed_payload_garbage_returns_error() {
        let payload = b"this is not json at all";
        let result = MqttTelemetryMessage::parse(payload);
        assert!(matches!(result, Err(MqttTelemetryError::Json(_))));
    }

    #[test]
    fn test_malformed_payload_truncated_returns_error() {
        let payload = br#"{"sensor_id":"abc","timestamp""#;
        let result = MqttTelemetryMessage::parse(payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_sensor_id_returns_error() {
        let json = br#"{"timestamp":1700000000,"temperature_c":22.5}"#;
        let result = MqttTelemetryMessage::parse(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_timestamp_returns_error() {
        let json = br#"{"sensor_id":"abc","temperature_c":22.5}"#;
        let result = MqttTelemetryMessage::parse(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_payload_returns_error() {
        let payload = b"";
        let result = MqttTelemetryMessage::parse(payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_null_payload_returns_error() {
        let payload = b"null";
        let result = MqttTelemetryMessage::parse(payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_type_sensor_id_returns_error() {
        let json = br#"{"sensor_id":123,"timestamp":1700000000}"#;
        let result = MqttTelemetryMessage::parse(json);
        assert!(result.is_err());
    }

    // ---- try_from convenience ----

    #[test]
    fn test_try_from_byte_slice() {
        let bytes: &[u8] = br#"{"sensor_id":"tf","timestamp":1}"#;
        let msg = MqttTelemetryMessage::try_from(bytes).unwrap();
        assert_eq!(msg.sensor_id, "tf");
    }

    #[test]
    fn test_try_from_vec() {
        let bytes: Vec<u8> = br#"{"sensor_id":"tfv","timestamp":1}"#.to_vec();
        let msg = MqttTelemetryMessage::try_from(bytes).unwrap();
        assert_eq!(msg.sensor_id, "tfv");
    }

    // ---- Connection config validation ----

    #[tokio::test]
    async fn test_empty_broker_url_returns_error() {
        let result = MqttTelemetryConsumer::connect("", "sensors/#").await;
        assert!(matches!(
            result,
            Err(MqttTelemetryError::InvalidConfig(ref msg))
                if msg.contains("empty")
        ));
    }

    #[tokio::test]
    async fn test_whitespace_broker_url_returns_error() {
        let result = MqttTelemetryConsumer::connect("   ", "sensors/#").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_empty_topic_returns_error() {
        let result = MqttTelemetryConsumer::connect("mqtts://localhost:8883", "").await;
        assert!(matches!(
            result,
            Err(MqttTelemetryError::InvalidConfig(ref msg))
                if msg.contains("topic")
        ));
    }

    #[tokio::test]
    async fn test_plaintext_broker_rejected_by_default() {
        // Plaintext must be rejected unless explicitly opted in. We do NOT set
        // the env var here, so the default policy (require TLS) applies.
        let result = MqttTelemetryConsumer::connect("mqtt://localhost:1883", "sensors/#").await;
        assert!(matches!(
            result,
            Err(MqttTelemetryError::InvalidConfig(ref msg))
                if msg.contains("FLUXION_MQTT_ALLOW_INSECURE")
        ));
    }

    // ---- Transport policy resolution (pure function) ----

    #[test]
    fn test_resolve_transport_plaintext_rejected_without_flag() {
        let err = resolve_transport(BrokerScheme::Plaintext, false, false).unwrap_err();
        assert!(matches!(err, MqttTelemetryError::InvalidConfig(_)));
    }

    #[test]
    fn test_resolve_transport_plaintext_allowed_with_flag() {
        let t = resolve_transport(BrokerScheme::Plaintext, true, false).unwrap();
        assert_eq!(t, ResolvedTransport::Plaintext);
    }

    #[test]
    fn test_resolve_transport_tls_validates_by_default() {
        let t = resolve_transport(BrokerScheme::Tls, false, false).unwrap();
        assert_eq!(t, ResolvedTransport::Tls { verify_certs: true });
    }

    #[test]
    fn test_resolve_transport_tls_skips_validation_when_insecure() {
        let t = resolve_transport(BrokerScheme::Tls, false, true).unwrap();
        assert_eq!(
            t,
            ResolvedTransport::Tls {
                verify_certs: false
            }
        );
    }

    #[test]
    fn test_resolve_transport_tls_still_validates_when_plaintext_allowed() {
        // Allowing plaintext must NOT silently weaken TLS connections.
        let t = resolve_transport(BrokerScheme::Tls, true, false).unwrap();
        assert_eq!(t, ResolvedTransport::Tls { verify_certs: true });
    }

    // ---- env_flag parsing ----

    #[test]
    fn test_env_flag_truthy_values() {
        assert!(check_env_flag("1"));
        assert!(check_env_flag("true"));
        assert!(check_env_flag("TRUE"));
        assert!(check_env_flag("Yes"));
        assert!(check_env_flag("on"));
    }

    #[test]
    fn test_env_flag_falsy_values() {
        assert!(!check_env_flag("0"));
        assert!(!check_env_flag("false"));
        assert!(!check_env_flag(""));
        assert!(!check_env_flag("anything"));
    }

    /// Evaluate [`env_flag`] against `value` using a process-unique variable
    /// name.
    ///
    /// `std::env::set_var` is process-global, so two parallel tests mutating the
    /// *same* name would race. By minting a fresh, unique name per call (via an
    /// atomic counter) and never restoring a prior value, every call is fully
    /// independent — no shared state, no race.
    fn check_env_flag(value: &str) -> bool {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("FLUXION_MQTT_TEST_ENV_FLAG_{id}");
        // SAFETY: each call uses a globally unique name that no other code reads
        // or writes, so there is no concurrent access to the same variable.
        unsafe {
            std::env::set_var(&name, value);
        }
        let result = env_flag(&name);
        unsafe {
            std::env::remove_var(&name);
        }
        result
    }

    // ---- Broker URL parsing ----

    #[test]
    fn test_parse_broker_url_with_mqtts_scheme() {
        let (scheme, host, port) = parse_broker_url("mqtts://broker.local:8883").unwrap();
        assert_eq!(scheme, BrokerScheme::Tls);
        assert_eq!(host, "broker.local");
        assert_eq!(port, 8883);
    }

    #[test]
    fn test_parse_broker_url_mqtts_default_port() {
        let (scheme, host, port) = parse_broker_url("mqtts://broker.local").unwrap();
        assert_eq!(scheme, BrokerScheme::Tls);
        assert_eq!(host, "broker.local");
        assert_eq!(port, DEFAULT_MQTTS_PORT);
    }

    #[test]
    fn test_parse_broker_url_with_mqtt_scheme() {
        let (scheme, host, port) = parse_broker_url("mqtt://broker.local:8883").unwrap();
        assert_eq!(scheme, BrokerScheme::Plaintext);
        assert_eq!(host, "broker.local");
        assert_eq!(port, 8883);
    }

    #[test]
    fn test_parse_broker_url_with_tcp_scheme() {
        let (scheme, host, port) = parse_broker_url("tcp://10.0.0.5:1883").unwrap();
        assert_eq!(scheme, BrokerScheme::Plaintext);
        assert_eq!(host, "10.0.0.5");
        assert_eq!(port, 1883);
    }

    #[test]
    fn test_parse_broker_url_bare_host_defaults_to_tls() {
        let (scheme, host, port) = parse_broker_url("broker.local").unwrap();
        assert_eq!(scheme, BrokerScheme::Tls);
        assert_eq!(host, "broker.local");
        assert_eq!(port, DEFAULT_MQTTS_PORT);
    }

    #[test]
    fn test_parse_broker_url_bare_host_with_port() {
        let (scheme, host, port) = parse_broker_url("broker.local:9999").unwrap();
        assert_eq!(scheme, BrokerScheme::Tls);
        assert_eq!(host, "broker.local");
        assert_eq!(port, 9999);
    }

    #[test]
    fn test_parse_broker_url_ip_address() {
        let (scheme, host, port) = parse_broker_url("192.168.1.100:1883").unwrap();
        assert_eq!(scheme, BrokerScheme::Tls);
        assert_eq!(host, "192.168.1.100");
        assert_eq!(port, 1883);
    }

    #[test]
    fn test_parse_broker_url_empty_host_error() {
        assert!(parse_broker_url("mqtts://").is_err());
        assert!(parse_broker_url("mqtt://").is_err());
        assert!(parse_broker_url("").is_err());
    }

    #[test]
    fn test_parse_broker_url_invalid_port_error() {
        assert!(parse_broker_url("mqtts://host:notaport").is_err());
        assert!(parse_broker_url("mqtts://host:99999").is_err());
    }

    // ---- Topic filter matching ----

    #[test]
    fn test_topic_exact_match() {
        assert!(topic_matches("sensors/temp", "sensors/temp"));
        assert!(!topic_matches("sensors/temp", "sensors/humidity"));
    }

    #[test]
    fn test_topic_single_level_wildcard() {
        assert!(topic_matches("sensors/+/temp", "sensors/zone1/temp"));
        assert!(topic_matches("sensors/+/temp", "sensors/zone2/temp"));
        assert!(!topic_matches("sensors/+/temp", "sensors/zone1/zone2/temp"));
    }

    #[test]
    fn test_topic_multi_level_wildcard() {
        assert!(topic_matches("sensors/#", "sensors/zone1/temp"));
        assert!(topic_matches("sensors/#", "sensors/zone1/zone2/power"));
        assert!(topic_matches("sensors/#", "sensors"));
    }

    #[test]
    fn test_topic_no_match_different_levels() {
        assert!(!topic_matches("a/b/c", "a/b"));
        assert!(!topic_matches("a/b", "a/b/c"));
    }

    // ---- Integration test (requires a real broker) ----

    #[tokio::test]
    #[ignore = "requires a running TLS MQTT broker at mqtts://localhost:8883"]
    async fn test_mqtt_consumer_integration() {
        // Defaults to validated TLS on port 8883. For a self-signed local broker
        // set FLUXION_MQTT_INSECURE=1 before running: `cargo test -- --ignored`.
        let broker = "mqtts://localhost:8883";
        let topic = "fluxion/test/zone1";

        let consumer = MqttTelemetryConsumer::connect(broker, topic).await;
        if consumer.is_err() {
            eprintln!("Skipping integration test — no broker at {broker}");
            return;
        }
        let (consumer, mut rx) = consumer.unwrap();

        let handle = tokio::spawn(async move {
            let _ = consumer.start().await;
        });

        // Give the consumer a moment, then abort.
        tokio::time::sleep(Duration::from_secs(2)).await;
        handle.abort();

        // Drain anything received (likely nothing without a publisher).
        let _ = rx.try_recv();
    }
}
