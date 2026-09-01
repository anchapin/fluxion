//! Kafka Telemetry Consumer for enterprise-scale telemetry ingestion.
//!
//! Provides a Kafka consumer implementation for large deployments using Kafka
//! for scalability. Consumer group support enables horizontal scaling.
//!
//! # Example
//! ```ignore
//! use fluxion::twin::KafkaTelemetryConsumer;
//!
//! let consumer = KafkaTelemetryConsumer::new(
//!     "localhost:9092",
//!     "fluxion-telemetry-group",
//!     "building-telemetry",
//! ).expect("Failed to create Kafka consumer");
//!
//! loop {
//!     if let Some(msg) = consumer.poll(Duration::from_millis(100)) {
//!         // Process telemetry message
//!     }
//! }
//! ```

use crossbeam::channel::Sender;
use serde::Deserialize;
use std::marker::PhantomData;
use std::time::Duration;

#[cfg(feature = "kafka")]
use rdkafka::config::ClientConfig;

#[cfg(feature = "kafka")]
use rdkafka::consumer::{BaseConsumer, Consumer, DefaultConsumerContext};

#[cfg(feature = "kafka")]
use rdkafka::error::KafkaError;

#[cfg(feature = "kafka")]
use rdkafka::message::Message;

#[cfg(feature = "kafka")]
use rdkafka::Offset;

pub const DEFAULT_CHANNEL_CAPACITY: usize = 1024;

#[cfg(feature = "kafka")]
type KafkaConsumerType = BaseConsumer<DefaultConsumerContext>;

#[cfg(not(feature = "kafka"))]
#[allow(dead_code)]
type KafkaConsumerType = PhantomData<()>;

#[cfg(feature = "kafka")]
type KafkaConsumerContextType = DefaultConsumerContext;

#[cfg(not(feature = "kafka"))]
#[allow(dead_code)]
type KafkaConsumerContextType = PhantomData<()>;

#[derive(Clone, Debug, Deserialize)]
pub struct TelemetryMsg {
    pub zone_id: String,
    pub timestamp: i64,
    pub temperature: f64,
    pub humidity: Option<f64>,
    pub occupancy: Option<u32>,
    pub hvac_mode: Option<String>,
    pub setpoint_heating: Option<f64>,
    pub setpoint_cooling: Option<f64>,
    pub energy_heating_kwh: Option<f64>,
    pub energy_cooling_kwh: Option<f64>,
    pub solar_gain_w_m2: Option<f64>,
    pub internal_gains_w: Option<f64>,
}

impl Default for TelemetryMsg {
    fn default() -> Self {
        Self {
            zone_id: String::new(),
            timestamp: 0,
            temperature: 0.0,
            humidity: None,
            occupancy: None,
            hvac_mode: None,
            setpoint_heating: None,
            setpoint_cooling: None,
            energy_heating_kwh: None,
            energy_cooling_kwh: None,
            solar_gain_w_m2: None,
            internal_gains_w: None,
        }
    }
}

#[derive(Debug)]
pub enum KafkaConsumerError {
    #[cfg(feature = "kafka")]
    Kafka(KafkaError),
    FeatureNotEnabled(String),
    /// Configuration was rejected because it would put the Kafka consumer on
    /// an insecure transport (plaintext) in a release build without the
    /// `FLUXION_KAFKA_ALLOW_INSECURE=1` opt-in. Issue #2910.
    InsecureConfig(String),
}

impl std::fmt::Display for KafkaConsumerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "kafka")]
            KafkaConsumerError::Kafka(e) => write!(f, "Kafka error: {}", e),
            KafkaConsumerError::FeatureNotEnabled(msg) => write!(f, "{}", msg),
            KafkaConsumerError::InsecureConfig(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for KafkaConsumerError {}

#[cfg(feature = "kafka")]
impl From<KafkaError> for KafkaConsumerError {
    fn from(e: KafkaError) -> Self {
        KafkaConsumerError::Kafka(e)
    }
}

// =========================================================================
// Kafka TLS / plaintext configuration (Issue #2910)
//
// The Kafka telemetry consumer previously built its `ClientConfig` with
// `bootstrap.servers`, `group.id`, and offset flags, but never set
// `security.protocol` — defaulting to `PLAINTEXT`, so every Kafka message
// the digital twin ingested traveled unencrypted and unauthenticated. A
// network observer could read sensor telemetry; an attacker on the wire path
// could inject forged payloads into the UKF state estimator.
//
// This block hardens the configuration to mirror the MQTT parity closed in
// #2531 / #2703 (`FLUXION_MQTT_ALLOW_INSECURE` + release boot guard) and
// the REST surface guard in #2505 / #2703:
// - default `security.protocol=ssl` (TLS, broker cert verification),
// - plaintext only when the operator sets `FLUXION_KAFKA_ALLOW_INSECURE=1`,
// - release builds refuse to start with plaintext WITHOUT that opt-in
//   (debug builds emit a warning and proceed, so local dev keeps working).
// =========================================================================

#[cfg(feature = "kafka")]
const ENV_KAFKA_ALLOW_INSECURE: &str = "FLUXION_KAFKA_ALLOW_INSECURE";

/// Resolved Kafka transport — what we will hand to `security.protocol`.
///
/// Kept tiny intentionally: the only knob the public API exposes today is the
/// `FLUXION_KAFKA_ALLOW_INSECURE` opt-in. If `sasl_ssl` ever becomes a
/// first-class option it should be added here and threaded through
/// `build_client_config`.
#[cfg(feature = "kafka")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KafkaSecurity {
    /// TLS, broker cert verification enabled (default).
    Ssl,
    /// Plaintext — telemetry travels unencrypted. Only chosen when the
    /// operator has explicitly opted in via `FLUXION_KAFKA_ALLOW_INSECURE=1`.
    Plaintext,
}

/// Pure decision function used by the boot guard (Issue #2910).
///
/// Returns `true` when the resolved Kafka transport is insecure — i.e. we are
/// about to hand `security.protocol=PLAINTEXT` to librdkafka — and the
/// operator has NOT explicitly opted in via `FLUXION_KAFKA_ALLOW_INSECURE=1`.
///
/// Pure (no I/O) so it can be unit-tested directly, mirroring the MQTT
/// `is_insecure_mqtt_configuration` and the REST `is_insecure_bind_configuration`.
#[cfg(feature = "kafka")]
fn is_insecure_kafka_configuration(security: KafkaSecurity, allow_insecure: bool) -> bool {
    if allow_insecure {
        return false;
    }
    matches!(security, KafkaSecurity::Plaintext)
}

/// Convenience wrapper that turns the [`is_insecure_kafka_configuration`]
/// decision into a `Result` carrying a clear, operator-facing refusal message.
///
/// The binary refuses to start in release builds (see
/// [`KafkaTelemetryConsumer::new`]) when this returns `Err`. In debug builds
/// the guard is skipped so local `cargo run` / `cargo test` keeps working
/// against a plaintext broker — exactly mirroring the MQTT guard's
/// `#[cfg(not(debug_assertions))]` enforcement.
#[cfg(feature = "kafka")]
fn check_kafka_boot_guard(security: KafkaSecurity, allow_insecure: bool) -> Result<(), String> {
    if !is_insecure_kafka_configuration(security, allow_insecure) {
        return Ok(());
    }
    // Unreachable when the decision function is false — but defensively
    // pattern-match so adding a new variant that IS allowed here is a
    // compile error.
    let reason = match security {
        KafkaSecurity::Plaintext => "plaintext broker (security.protocol=PLAINTEXT)",
        KafkaSecurity::Ssl => return Ok(()),
    };
    Err(format!(
        "fluxion-twin: refusing to boot in release build — Kafka transport is insecure ({reason}). \
         Use a TLS-enabled broker (security.protocol=ssl or sasl_ssl) with valid certificates, \
         or set FLUXION_KAFKA_ALLOW_INSECURE=1 to explicitly opt in to insecure Kafka \
         transport. (Release boot guard, parity with FLUXION_MQTT_ALLOW_INSECURE — \
         Issue #2910.)"
    ))
}

/// Read a boolean environment flag.
///
/// Truthy values (case-insensitive): `1`, `true`, `yes`, `on`. Anything else,
/// or an unset variable, is `false`. Mirrors the `env_flag` helper in
/// `crates/fluxion-twin/src/telemetry/mqtt.rs`.
#[cfg(feature = "kafka")]
fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

/// Build the rdkafka [`ClientConfig`] for the Kafka telemetry consumer.
///
/// Defaults to `security.protocol=ssl` (TLS, broker cert verification
/// enabled). When the operator has explicitly opted in via
/// `FLUXION_KAFKA_ALLOW_INSECURE=1`, the resolved [`KafkaSecurity`] is
/// `Plaintext` and we set `security.protocol=plaintext` instead.
///
/// Extracted from `new()` so the inline unit test in this module can assert
/// on the rendered configuration without needing a live broker.
#[cfg(feature = "kafka")]
fn build_client_config(brokers: &str, group_id: &str, security: KafkaSecurity) -> ClientConfig {
    let mut config = ClientConfig::new();
    config
        .set("bootstrap.servers", brokers)
        .set("group.id", group_id)
        .set("enable.partition.eof", "false")
        .set("enable.auto.commit", "true")
        .set("auto.offset.reset", "earliest");
    match security {
        KafkaSecurity::Ssl => {
            config.set("security.protocol", "ssl");
        }
        KafkaSecurity::Plaintext => {
            config.set("security.protocol", "plaintext");
        }
    }
    config
}

pub struct KafkaTelemetryConsumer {
    #[cfg(feature = "kafka")]
    consumer: KafkaConsumerType,
    #[cfg(not(feature = "kafka"))]
    _phantom: PhantomData<()>,
    #[allow(dead_code)]
    tx: Sender<TelemetryMsg>,
    #[cfg(feature = "kafka")]
    _context: PhantomData<KafkaConsumerContextType>,
    // Subscribed topic. Retained so the lag gauge (Issue #2519) can resolve
    // partition watermarks without re-deriving it on every poll. Under the
    // `kafka` feature it's read by `try_record_lag`'s span/instrumentation;
    // without the feature the consumer is never constructed, hence the
    // `dead_code` allowance.
    #[allow(dead_code)]
    topic: String,
    /// Retained consumer group id (rdkafka 0.39 removed
    /// `ConsumerContext::group_id`, #3308).
    #[allow(dead_code)]
    group_id: String,
}

#[cfg(feature = "kafka")]
impl KafkaTelemetryConsumer {
    pub fn new(brokers: &str, group_id: &str, topic: &str) -> Result<Self, KafkaConsumerError> {
        let (tx, _rx) = crossbeam::channel::bounded(DEFAULT_CHANNEL_CAPACITY);

        let allow_insecure = env_flag(ENV_KAFKA_ALLOW_INSECURE);
        let security = if allow_insecure {
            KafkaSecurity::Plaintext
        } else {
            KafkaSecurity::Ssl
        };

        // Release-only boot guard (Issue #2910) — mirrors the MQTT guard in
        // `crates/fluxion-twin/src/telemetry/mqtt.rs`. The decision is computed
        // in every build (so the helper stays unit-tested) but only ACTED on
        // in release builds. Debug builds keep working against plaintext
        // brokers for local dev.
        let boot_guard = check_kafka_boot_guard(security, allow_insecure);
        #[cfg(not(debug_assertions))]
        if let Err(msg) = boot_guard {
            return Err(KafkaConsumerError::InsecureConfig(msg));
        }
        #[cfg(debug_assertions)]
        let _ = boot_guard;

        let config = build_client_config(brokers, group_id, security);

        if matches!(security, KafkaSecurity::Plaintext) {
            tracing::warn!(
                brokers = %brokers,
                "FLUXION_KAFKA_ALLOW_INSECURE is set: connecting to Kafka broker over \
                 plaintext. Telemetry payloads will be unencrypted."
            );
        }

        let consumer: BaseConsumer<DefaultConsumerContext> = config.create()?;

        consumer.subscribe(&[topic])?;

        Ok(Self {
            consumer,
            tx,
            _context: PhantomData,
            topic: topic.to_string(),
            group_id: group_id.to_string(),
        })
    }

    /// Poll the broker for the next telemetry record (up to `timeout`).
    ///
    /// Each poll drives the `fluxion_twin_kafka_messages_total{outcome}`
    /// counter (`received` on a successfully parsed message, `error` on a
    /// parse failure) and, when offset metadata is available, sets the
    /// `fluxion_twin_kafka_lag` gauge to the per-partition watermark lag
    /// (high watermark − consumed offset). Lag computation is best-effort —
    /// any failure (no assignment, metadata timeout, unknown offsets) is
    /// swallowed and the gauge is simply not updated for that poll.
    ///
    /// (Issue #2519 — structured observability for the Kafka telemetry path.)
    #[tracing::instrument(skip(self), fields(topic = %self.topic))]
    pub fn poll(&self, timeout: Duration) -> Option<TelemetryMsg> {
        let parsed = match self.consumer.poll(timeout) {
            Some(Ok(message)) => message
                .payload()
                .and_then(|p| serde_json::from_slice(p).ok()),
            // Transient / no-message polls are not a Kafka message outcome, so
            // they don't increment the counter (mirrors the MQTT path, where
            // empty polls and reconnects are transport-level churn).
            _ => return None,
        };

        let outcome = if parsed.is_some() {
            "received"
        } else {
            "error"
        };
        metrics::counter!("fluxion_twin_kafka_messages_total", "outcome" => outcome).increment(1);

        // Best-effort lag update. Done after the counter so a metadata hiccup
        // never suppresses the message-outcome signal.
        let _ = self.try_record_lag();

        parsed
    }

    /// Best-effort total consumer-lag computation across all assigned
    /// partitions of the subscribed topic.
    ///
    /// Returns `Some(total_lag)` if at least one partition yielded a numeric
    /// `(position, high_watermark)` pair; returns `None` if the consumer has
    /// no assignment yet, metadata timed out, or every partition reported a
    /// non-numeric offset (e.g. `Beginning`/`End`/`Invalid`). Never panics
    /// and never blocks longer than the per-partition watermark timeout.
    ///
    /// Lag = `high_watermark − consumed_offset`, summed across partitions.
    /// Clamped to `>= 0` so a stale low-watermark race never produces a
    /// negative gauge.
    ///
    /// (Issue #2519 — `fluxion_twin_kafka_lag`.)
    fn try_record_lag(&self) -> Option<i64> {
        let position = self.consumer.position().ok()?;
        let mut total_lag: i64 = 0;
        let mut any = false;
        for elem in position.elements() {
            let topic = elem.topic();
            let partition = elem.partition();
            let consumed = match elem.offset() {
                Offset::Offset(n) => n,
                // Non-numeric offset — can't compute lag for this partition.
                _ => continue,
            };
            // Short timeout: lag is best-effort and must not stall the poll
            // loop. 50 ms per partition is well within rdkafka's metadata
            // refresh budget for a healthy cluster.
            let (_low, high) = self
                .consumer
                .fetch_watermarks(topic, partition, Duration::from_millis(50))
                .ok()?;
            any = true;
            let lag = (high - consumed).max(0);
            total_lag += lag;
        }
        if any {
            metrics::gauge!("fluxion_twin_kafka_lag").set(total_lag as f64);
        }
        if any {
            Some(total_lag)
        } else {
            None
        }
    }

    pub fn sender(&self) -> &Sender<TelemetryMsg> {
        &self.tx
    }

    pub fn consumer_group_id(&self) -> String {
        self.group_id.clone()
    }
}

#[cfg(not(feature = "kafka"))]
impl KafkaTelemetryConsumer {
    pub fn new(_brokers: &str, _group_id: &str, _topic: &str) -> Result<Self, KafkaConsumerError> {
        Err(KafkaConsumerError::FeatureNotEnabled(
            "kafka feature not enabled".to_string(),
        ))
    }

    pub fn poll(&self, _timeout: Duration) -> Option<TelemetryMsg> {
        None
    }

    pub fn sender(&self) -> &Sender<TelemetryMsg> {
        panic!("kafka feature not enabled")
    }

    pub fn consumer_group_id(&self) -> String {
        panic!("kafka feature not enabled")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_msg_default() {
        let msg = TelemetryMsg::default();
        assert_eq!(msg.zone_id, String::new());
        assert_eq!(msg.timestamp, 0);
        assert_eq!(msg.temperature, 0.0);
        assert!(msg.humidity.is_none());
        assert!(msg.occupancy.is_none());
    }

    #[test]
    fn test_telemetry_msg_deserialize() {
        let json = r#"{
            "zone_id": "zone-1",
            "timestamp": 1700000000,
            "temperature": 22.5,
            "humidity": 0.45,
            "occupancy": 10,
            "hvac_mode": "heating",
            "setpoint_heating": 21.0,
            "setpoint_cooling": 26.0,
            "energy_heating_kwh": 15.5,
            "energy_cooling_kwh": 0.0,
            "solar_gain_w_m2": 120.0,
            "internal_gains_w": 500.0
        }"#;

        let msg: TelemetryMsg = serde_json::from_str(json).unwrap();
        assert_eq!(msg.zone_id, "zone-1");
        assert_eq!(msg.timestamp, 1700000000);
        assert!((msg.temperature - 22.5).abs() < f64::EPSILON);
        assert_eq!(msg.humidity, Some(0.45));
        assert_eq!(msg.occupancy, Some(10));
        assert_eq!(msg.hvac_mode, Some("heating".to_string()));
        assert_eq!(msg.setpoint_heating, Some(21.0));
        assert_eq!(msg.setpoint_cooling, Some(26.0));
        assert_eq!(msg.energy_heating_kwh, Some(15.5));
        assert_eq!(msg.solar_gain_w_m2, Some(120.0));
        assert_eq!(msg.internal_gains_w, Some(500.0));
    }

    #[test]
    fn test_telemetry_msg_partial_deserialize() {
        let json = r#"{
            "zone_id": "zone-2",
            "timestamp": 1700000001,
            "temperature": 20.0
        }"#;

        let msg: TelemetryMsg = serde_json::from_str(json).unwrap();
        assert_eq!(msg.zone_id, "zone-2");
        assert_eq!(msg.timestamp, 1700000001);
        assert!((msg.temperature - 20.0).abs() < f64::EPSILON);
        assert!(msg.humidity.is_none());
        assert!(msg.occupancy.is_none());
    }

    #[cfg(not(feature = "kafka"))]
    #[test]
    fn test_consumer_feature_not_available() {
        let result = KafkaTelemetryConsumer::new("localhost:9092", "group", "topic");
        assert!(result.is_err());
        match result {
            Err(KafkaConsumerError::FeatureNotEnabled(msg)) => {
                assert_eq!(msg, "kafka feature not enabled");
            }
            Err(KafkaConsumerError::InsecureConfig(_)) => {
                panic!("Expected feature-not-enabled error, got InsecureConfig");
            }
            #[cfg(feature = "kafka")]
            Err(KafkaConsumerError::Kafka(_)) => {
                panic!("Expected feature-not-enabled error, got Kafka error");
            }
            Ok(_) => {
                panic!("Expected error, got Ok");
            }
        }
    }

    #[test]
    fn test_kafka_consumer_error_display() {
        let err = KafkaConsumerError::FeatureNotEnabled("test".to_string());
        assert_eq!(format!("{}", err), "test");
    }

    #[test]
    fn test_kafka_consumer_error_display_insecure_config() {
        let err = KafkaConsumerError::InsecureConfig("plaintext refused".to_string());
        assert_eq!(format!("{}", err), "plaintext refused");
    }

    // ============================================================
    // Issue #2910 — Kafka TLS / plaintext boot guard
    // ============================================================
    //
    // The defaults are the only thing the public API exposes today, so we
    // exercise the helpers directly. The release-only enforcement is in
    // `check_kafka_boot_guard`; the resulting `Err` is what
    // `KafkaTelemetryConsumer::new` would bubble up as
    // `KafkaConsumerError::InsecureConfig` in release builds.

    #[cfg(feature = "kafka")]
    fn unique_env_name() -> String {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("FLUXION_KAFKA_TEST_ENV_{id}")
    }

    #[cfg(feature = "kafka")]
    fn check_kafka_env_flag(value: &str) -> bool {
        let name = unique_env_name();
        // SAFETY: each call uses a globally unique name that no other code
        // reads or writes, so there is no concurrent access to the same
        // variable.
        unsafe {
            std::env::set_var(&name, value);
        }
        let result = env_flag(&name);
        unsafe {
            std::env::remove_var(&name);
        }
        result
    }

    #[cfg(feature = "kafka")]
    #[test]
    fn build_client_config_defaults_to_ssl() {
        // Default (no opt-in) MUST request TLS — Issue #2910 acceptance #1.
        let config = build_client_config("localhost:9092", "test-group", KafkaSecurity::Ssl);
        let protocol = config
            .get("security.protocol")
            .expect("default config must set security.protocol");
        assert_eq!(
            protocol, "ssl",
            "default KafkaTelemetryConsumer must request TLS, not plaintext"
        );
        // Sanity-check the other mandatory fields are still wired up.
        assert_eq!(config.get("bootstrap.servers"), Some("localhost:9092"));
        assert_eq!(config.get("group.id"), Some("test-group"));
    }

    #[cfg(feature = "kafka")]
    #[test]
    fn build_client_config_explicit_plaintext() {
        // When the operator opts in to plaintext, the config must reflect it
        // so the warning + boot guard are honest.
        let config = build_client_config("localhost:9092", "test-group", KafkaSecurity::Plaintext);
        assert_eq!(config.get("security.protocol"), Some("plaintext"));
    }

    #[cfg(feature = "kafka")]
    #[test]
    fn is_insecure_kafka_configuration_decision_matrix() {
        // Opt-in neutralises the flag — operator escapes the guard.
        assert!(!is_insecure_kafka_configuration(
            KafkaSecurity::Plaintext,
            true
        ));
        assert!(!is_insecure_kafka_configuration(KafkaSecurity::Ssl, true));
        // TLS is always safe.
        assert!(!is_insecure_kafka_configuration(KafkaSecurity::Ssl, false));
        // Plaintext without opt-in is the failure mode the guard catches.
        assert!(is_insecure_kafka_configuration(
            KafkaSecurity::Plaintext,
            false
        ));
    }

    #[cfg(feature = "kafka")]
    #[test]
    fn check_kafka_boot_guard_message_names_opt_in_env_var() {
        // The refusal message must tell the operator exactly how to opt in —
        // mirrors the MQTT guard's actionable error string.
        let err = check_kafka_boot_guard(KafkaSecurity::Plaintext, false).unwrap_err();
        assert!(
            err.contains("FLUXION_KAFKA_ALLOW_INSECURE"),
            "expected refusal message to name FLUXION_KAFKA_ALLOW_INSECURE, got: {err}"
        );
        assert!(
            err.contains("release"),
            "expected 'release' in message: {err}"
        );
    }

    #[cfg(feature = "kafka")]
    #[test]
    fn check_kafka_boot_guard_permits_tls() {
        assert!(check_kafka_boot_guard(KafkaSecurity::Ssl, false).is_ok());
        assert!(check_kafka_boot_guard(KafkaSecurity::Ssl, true).is_ok());
    }

    #[cfg(feature = "kafka")]
    #[test]
    fn check_kafka_boot_guard_permits_plaintext_with_opt_in() {
        assert!(check_kafka_boot_guard(KafkaSecurity::Plaintext, true).is_ok());
    }

    // -- Release-only enforcement (the guard REFUSES insecure configs) --

    #[cfg(all(feature = "kafka", not(debug_assertions)))]
    #[test]
    fn boot_guard_release_refuses_plaintext_without_opt_in() {
        // Acceptance #4 — missing TLS config in release is rejected.
        let err = check_kafka_boot_guard(KafkaSecurity::Plaintext, false).unwrap_err();
        assert!(
            err.contains("FLUXION_KAFKA_ALLOW_INSECURE"),
            "release refusal must mention the opt-in env var: {err}"
        );
    }

    #[cfg(all(feature = "kafka", not(debug_assertions)))]
    #[test]
    fn boot_guard_release_silent_on_default_tls() {
        // No error on the default configuration — the guard only fires when
        // the resolved transport is plaintext without opt-in.
        assert!(check_kafka_boot_guard(KafkaSecurity::Ssl, false).is_ok());
    }

    #[cfg(feature = "kafka")]
    #[test]
    fn env_flag_truthy_and_falsy_values() {
        // Mirrors `crates/fluxion-twin/src/telemetry/mqtt.rs` test env_flag.
        assert!(check_kafka_env_flag("1"));
        assert!(check_kafka_env_flag("true"));
        assert!(check_kafka_env_flag("TRUE"));
        assert!(check_kafka_env_flag("Yes"));
        assert!(check_kafka_env_flag("on"));
        assert!(!check_kafka_env_flag("0"));
        assert!(!check_kafka_env_flag("false"));
        assert!(!check_kafka_env_flag(""));
        assert!(!check_kafka_env_flag("anything"));
    }

    #[cfg(feature = "kafka")]
    #[test]
    fn env_flag_unset_env_is_false() {
        // A completely fresh name (never set) must read as false.
        let name = unique_env_name();
        unsafe {
            std::env::remove_var(&name);
        }
        assert!(!env_flag(&name));
    }
}
