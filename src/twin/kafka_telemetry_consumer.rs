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
use std::time::Duration;

#[cfg(feature = "kafka")]
use rdkafka::config::ClientConfig;

#[cfg(feature = "kafka")]
use rdkafka::consumer::{BaseConsumer, Consumer, DefaultConsumerContext};

#[cfg(feature = "kafka")]
use rdkafka::error::KafkaError;

#[cfg(feature = "kafka")]
use rdkafka::message::Message;

#[cfg(not(feature = "kafka"))]
use std::marker::PhantomData;

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
}

impl std::fmt::Display for KafkaConsumerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "kafka")]
            KafkaConsumerError::Kafka(e) => write!(f, "Kafka error: {}", e),
            KafkaConsumerError::FeatureNotEnabled(msg) => write!(f, "{}", msg),
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

pub struct KafkaTelemetryConsumer {
    #[cfg(feature = "kafka")]
    consumer: KafkaConsumerType,
    #[cfg(not(feature = "kafka"))]
    _phantom: PhantomData<()>,
    #[allow(dead_code)]
    tx: Sender<TelemetryMsg>,
    #[cfg(feature = "kafka")]
    _context: PhantomData<KafkaConsumerContextType>,
}

#[cfg(feature = "kafka")]
impl KafkaTelemetryConsumer {
    pub fn new(brokers: &str, group_id: &str, topic: &str) -> Result<Self, KafkaConsumerError> {
        let (tx, _rx) = crossbeam::channel::bounded(DEFAULT_CHANNEL_CAPACITY);

        let consumer: BaseConsumer<DefaultConsumerContext> = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .set("group.id", group_id)
            .set("enable.partition.eof", "false")
            .set("enable.auto.commit", "true")
            .set("auto.offset.reset", "earliest")
            .create()
            .map_err(KafkaError::ConsumerCreation)?;

        consumer.subscribe(&[topic])?;

        Ok(Self {
            consumer,
            tx,
            _context: PhantomData,
        })
    }

    pub fn poll(&self, timeout: Duration) -> Option<TelemetryMsg> {
        match self.consumer.poll(timeout) {
            Some(Ok(message)) => {
                if let Some(payload) = message.payload() {
                    serde_json::from_slice(payload).ok()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn sender(&self) -> &Sender<TelemetryMsg> {
        &self.tx
    }

    pub fn consumer_group_id(&self) -> String {
        self.consumer.context().group_id().to_string()
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
}
