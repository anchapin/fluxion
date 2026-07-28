//! Telemetry subsystem with backpressure and deduplication.
//!
//! # Module Overview
//!
//! - [`consumer::TelemetryConsumer`] — Bounded channel consumer with out-of-order deduplication
//! - [`message::TelemetryMsg`] — Telemetry message with sensor ID, sequence, timestamp, payload
//! - [`mqtt::MqttTelemetryConsumer`] — MQTT-based telemetry consumer
//! - [`error::TelemetryError`] — Error types for backpressure and channel operations

pub mod consumer;
pub mod error;
pub mod message;
pub mod mqtt;

pub use consumer::{Sender, TelemetryConsumer};
pub use error::TelemetryError;
pub use message::TelemetryMsg;
pub use mqtt::{MqttTelemetryConsumer, MqttTelemetryError, MqttTelemetryMessage};
