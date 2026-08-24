//! MQTT Telemetry Consumer.
//!
//! Subscribes to an MQTT broker topic, parses JSON payloads into typed telemetry
//! messages, and forwards them through a bounded [`tokio::sync::mpsc`] channel.
//!
//! rumqttc handles automatic reconnection on transient disconnects — the event
//! loop keeps polling and the broker connection is restored transparently.
//!
//! # Transport security (always TLS)
//!
//! The consumer requires **MQTT-over-TLS** (`mqtts://`, port 8883) using
//! rustls with the platform trust store. Server certificates are **always
//! validated** — there is no runtime bypass for certificate verification
//! (Issue #3162). Plaintext (`mqtt://` / `tcp://`) broker URLs are
//! **always rejected**.
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

use rumqttc::{AsyncClient, Event, EventLoop, MqttOptions, Packet, QoS, Transport};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::Span;

/// Default bounded channel capacity.
///
/// A bounded channel prevents unbounded memory growth if the consumer drains
/// messages slower than the MQTT broker delivers them.
const CHANNEL_CAPACITY: usize = 1024;

/// Default MQTT-over-TLS port (secure) when not specified in the broker URL.
const DEFAULT_MQTTS_PORT: u16 = 8883;

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
    /// # Transport Security (always TLS — Issue #3162)
    ///
    /// TLS with validated certificates is **always required** — there is no runtime
    /// bypass for certificate verification. Plaintext (`mqtt://` / `tcp://`) broker
    /// URLs are **always rejected**. This is the MQTT analogue of the `fluxion-rest`
    /// TLS requirement.
    ///
    /// | Input | Transport | Host | Port |
    /// |-------|-----------|------|------|
    /// | `mqtts://broker.local` | TLS (validated) | `broker.local` | `8883` |
    /// | `mqtts://broker.local:8883` | TLS (validated) | `broker.local` | `8883` |
    /// | `broker.local` | TLS (validated) | `broker.local` | `8883` (default) |
    /// | `broker.local:1883` | TLS (validated) | `broker.local` | `1883` |
    /// | `mqtt://broker.local:1883` | **rejected** | — | — |
    /// | `tcp://10.0.0.5:1883` | **rejected** | — | — |
    ///
    /// # Errors
    ///
    /// Returns [`MqttTelemetryError::InvalidConfig`] if the broker URL or topic
    /// is empty/malformed, or if a plaintext URL is supplied.
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
        resolve_transport(scheme)?;

        let mut mqttoptions = MqttOptions::new("fluxion-twin-consumer", host, port);
        mqttoptions.set_keep_alive(Duration::from_secs(5));
        mqttoptions.set_transport(Transport::tls_with_default_config());

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

    /// Poll the event loop and return the next parsed telemetry message that
    /// matches the subscription topic.
    ///
    /// Carries a `tracing` span with `topic` and `payload_bytes` fields
    /// (populated from the matched `Publish`), and increments the
    /// [`fluxion_twin_mqtt_messages_total`] counter with `outcome="received"`
    /// or `outcome="error"` (parse failure). Transient rumqttc disconnects are
    /// logged and auto-recovered internally — they do **not** count as a
    /// message outcome. Returns `None` only if the event loop yields a fatal,
    /// unrecoverable condition; in practice callers loop until `None` or until
    /// they no longer need messages.
    ///
    /// This is the primary observability seam for the MQTT path (Issue #2519):
    /// every delivered message flows through here, so per-message spans and
    /// counters are uniform regardless of whether the consumer is driven by
    /// [`Self::start`] or polled directly.
    ///
    /// [`fluxion_twin_mqtt_messages_total`]: crate::telemetry
    #[tracing::instrument(skip(self), fields(topic, payload_bytes))]
    pub async fn next(&mut self) -> Option<MqttTelemetryMessage> {
        loop {
            match self.eventloop.poll().await {
                Ok(Event::Incoming(Packet::Publish(publish))) => {
                    if !topic_matches(&self.topic, &publish.topic) {
                        continue;
                    }

                    let payload_bytes = publish.payload.len();
                    // Record the declared span fields now that we have a
                    // concrete matched message. `Span::current()` resolves to
                    // the span created by `#[instrument]` above.
                    Span::current()
                        .record("topic", publish.topic.as_str())
                        .record("payload_bytes", payload_bytes);

                    match parse_and_count(publish.payload.as_ref()) {
                        Ok(msg) => return Some(msg),
                        Err(e) => {
                            // Counter already emitted inside `parse_and_count`.
                            tracing::warn!(
                                topic = publish.topic.as_str(),
                                payload_bytes,
                                error = %e,
                                "Failed to parse MQTT payload"
                            );
                            // Continue — don't crash the loop on a single bad
                            // message. The span is still open and already has
                            // `topic`/`payload_bytes` recorded; the next matched
                            // message will overwrite them before this call
                            // returns.
                        }
                    }
                }
                // Other incoming/outgoing events (SubAck, ConnAck, etc.) — not
                // a telemetry message, so neither span fields nor counters
                // apply.
                Ok(_) => {}
                Err(e) => {
                    // rumqttc auto-reconnects on the next poll(). Brief backoff
                    // to avoid tight-looping on persistent failures (e.g. DNS).
                    // This is transport-level churn, not a message outcome, so
                    // the messages_total counter is NOT incremented.
                    tracing::warn!(
                        error = %e,
                        "MQTT connection error (auto-reconnecting)"
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    /// Run the event loop: poll for `Publish` packets, parse JSON payloads into
    /// [`MqttTelemetryMessage`] values, and forward them through the bounded
    /// channel.
    ///
    /// Thin driver around [`Self::next`] — all per-message observability
    /// (spans + `fluxion_twin_mqtt_messages_total`) lives in `next()` so it is
    /// shared with direct callers.
    ///
    /// # Lifecycle
    ///
    /// - **Normal operation** — polls indefinitely, parsing and forwarding
    ///   messages.
    /// - **Transient disconnect** — rumqttc auto-reconnects; the loop continues
    ///   after a brief backoff sleep.
    /// - **Channel receiver dropped** — `tx.send` fails; the loop exits cleanly
    ///   with `Ok(())`.
    /// - **Malformed payload** — counted as `outcome="error"` and continues.
    ///
    /// # Errors
    ///
    /// Only returns `Err` if a fatal, unrecoverable error occurs. In practice the
    /// loop tends to run indefinitely until the consumer is dropped or the channel
    /// receiver is dropped.
    pub async fn start(mut self) -> Result<(), MqttTelemetryError> {
        while let Some(msg) = self.next().await {
            if self.tx.send(msg).await.is_err() {
                // Receiver dropped — consumer no longer needed.
                break;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse an MQTT JSON payload and record the
/// [`fluxion_twin_mqtt_messages_total`] counter with the appropriate
/// `outcome` label (`received` on success, `error` on parse failure).
///
/// Factored out of [`MqttTelemetryConsumer::next`] so the metric emission is
/// unit-testable without a live broker (Issue #2519). The topic-aware span
/// recording and the diagnostic `warn!` stay in `next()`; this helper only
/// owns parse + counter.
///
/// Returns the parsed message on success, or the underlying
/// [`MqttTelemetryError`] on failure (the counter has already been
/// incremented — callers should not count again).
///
/// [`fluxion_twin_mqtt_messages_total`]: crate::telemetry
fn parse_and_count(payload: &[u8]) -> Result<MqttTelemetryMessage, MqttTelemetryError> {
    match MqttTelemetryMessage::parse(payload) {
        Ok(msg) => {
            metrics::counter!(
                "fluxion_twin_mqtt_messages_total",
                "outcome" => "received"
            )
            .increment(1);
            Ok(msg)
        }
        Err(e) => {
            metrics::counter!(
                "fluxion_twin_mqtt_messages_total",
                "outcome" => "error"
            )
            .increment(1);
            Err(e)
        }
    }
}

/// Scheme recognised in the broker URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BrokerScheme {
    /// `mqtts://` — MQTT-over-TLS (secure). Also the default for bare hosts.
    Tls,
    /// `mqtt://` or `tcp://` — plaintext. Rejected unless explicitly permitted.
    Plaintext,
}

/// Decide the transport from the URL scheme.
///
/// TLS is always validated; there is no runtime bypass for certificate
/// verification (Issue #3162). Plaintext is always rejected.
///
/// Pure (no I/O) so it can be unit-tested directly.
fn resolve_transport(scheme: BrokerScheme) -> Result<(), MqttTelemetryError> {
    match scheme {
        BrokerScheme::Plaintext => Err(MqttTelemetryError::InvalidConfig(
            "plaintext broker URL ('mqtt://'/'tcp://') rejected: \
             telemetry would be unencrypted. TLS (mqtts://, port 8883) is required."
                .to_string(),
        )),
        BrokerScheme::Tls => Ok(()),
    }
}

/// Parse a broker URL into `(scheme, host, port)`.
///
/// Recognised schemes:
/// - `mqtts://` → [`BrokerScheme::Tls`], default port [`DEFAULT_MQTTS_PORT`].
/// - `mqtt://` / `tcp://` → [`BrokerScheme::Plaintext`], default port `1883`.
///   Plaintext URLs are always rejected — TLS is required (Issue #3162).
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
        BrokerScheme::Plaintext => 1883,
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

    // ---- Observability (Issue #2519): MQTT message counter ----

    /// A well-formed payload must emit
    /// `fluxion_twin_mqtt_messages_total{outcome="received"}` exactly once.
    /// Uses a thread-local `DebuggingRecorder` (Issue #2498) so it is isolated
    /// from the process-global Prometheus recorder.
    #[test]
    fn mqtt_received_counter_increments_on_valid_payload() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let payload = br#"{"sensor_id":"m-1","timestamp":1700000000,"temperature_c":22.5}"#;
        metrics::with_local_recorder(&recorder, || {
            let msg = parse_and_count(payload).unwrap();
            assert_eq!(msg.sensor_id, "m-1");
        });

        let map = snapshotter.snapshot().into_hashmap();
        let found = map.keys().any(|ck| {
            ck.key().name() == "fluxion_twin_mqtt_messages_total"
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "outcome" && l.value() == "received")
        });
        assert!(
            found,
            "expected fluxion_twin_mqtt_messages_total{{outcome=\"received\"}} to be emitted"
        );
    }

    /// A malformed payload must emit
    /// `fluxion_twin_mqtt_messages_total{outcome="error"}` and return `Err`.
    #[test]
    fn mqtt_error_counter_increments_on_bad_payload() {
        use metrics_util::debugging::DebuggingRecorder;

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            assert!(parse_and_count(b"not json").is_err());
        });

        let map = snapshotter.snapshot().into_hashmap();
        let found = map.keys().any(|ck| {
            ck.key().name() == "fluxion_twin_mqtt_messages_total"
                && ck
                    .key()
                    .labels()
                    .any(|l| l.key() == "outcome" && l.value() == "error")
        });
        assert!(
            found,
            "expected fluxion_twin_mqtt_messages_total{{outcome=\"error\"}} to be emitted"
        );
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
        let result = MqttTelemetryConsumer::connect("mqtt://localhost:1883", "sensors/#").await;
        assert!(matches!(
            result,
            Err(MqttTelemetryError::InvalidConfig(ref msg))
                if msg.contains("plaintext")
        ));
    }

    // ---- Transport policy resolution (pure function) ----

    #[test]
    fn test_resolve_transport_plaintext_rejected() {
        let err = resolve_transport(BrokerScheme::Plaintext).unwrap_err();
        assert!(matches!(err, MqttTelemetryError::InvalidConfig(ref msg)
            if msg.contains("plaintext")));
    }

    #[test]
    fn test_resolve_transport_tls_accepted() {
        assert!(resolve_transport(BrokerScheme::Tls).is_ok());
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
        // Requires a validated TLS broker at mqtts://localhost:8883.
        // There is no bypass for certificate verification (Issue #3162).
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
