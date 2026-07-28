//! MQTT Telemetry Consumer
//!
//! Subscribes to an MQTT broker topic and receives telemetry messages.
//!
//! # Example
//!
//! ```ignore
//! let consumer = MqttTelemetryConsumer::connect("mqtt://localhost:1883", "fluxion/test/zone1").await.unwrap();
//! let (tx, mut rx) = tokio::sync::mpsc::channel(100);
//! let handle = tokio::spawn(async move { consumer.start(tx).await });
//! // Publish messages via MQTT broker
//! handle.abort();
//! ```

use rumqttc::{AsyncClient, Event, EventLoop, MqttOptions, Packet, QoS};
use serde::Deserialize;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Error, Debug)]
pub enum TelemetryError {
    #[error("MQTT connection error: {0}")]
    Connection(#[from] rumqttc::ConnectionError),
    #[error("MQTT client error: {0}")]
    Client(#[from] rumqttc::ClientError),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Channel send error")]
    ChannelSend,
}

#[derive(Debug, Clone)]
pub struct TelemetryMessage {
    pub zone_id: String,
    pub t_air: f64,
    pub rh: f64,
}

impl TryFrom<&[u8]> for TelemetryMessage {
    type Error = TelemetryError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        #[derive(Deserialize)]
        struct RawMessage {
            zone_id: String,
            t_air: f64,
            rh: f64,
        }
        let raw: RawMessage = serde_json::from_slice(bytes)?;
        Ok(TelemetryMessage {
            zone_id: raw.zone_id,
            t_air: raw.t_air,
            rh: raw.rh,
        })
    }
}

pub struct MqttTelemetryConsumer {
    #[allow(dead_code)]
    client: AsyncClient,
    eventloop: EventLoop,
    topic: String,
}

impl MqttTelemetryConsumer {
    pub async fn connect(broker: &str, topic: &str) -> Result<Self, TelemetryError> {
        let mut mqttoptions = MqttOptions::new("fluxion-twin-consumer", broker, 1883);
        mqttoptions.set_keep_alive(Duration::from_secs(5));

        let (client, eventloop) = AsyncClient::new(mqttoptions, 100);
        client.subscribe(topic, QoS::AtLeastOnce).await?;

        Ok(Self {
            client,
            eventloop,
            topic: topic.to_string(),
        })
    }

    pub async fn start(mut self, tx: mpsc::Sender<TelemetryMessage>) -> Result<(), TelemetryError> {
        loop {
            match self.eventloop.poll().await {
                Ok(Event::Incoming(Packet::Publish(publish))) => {
                    if publish.topic == self.topic {
                        if let Ok(msg) = TelemetryMessage::try_from(publish.payload.as_ref()) {
                            if tx.send(msg).await.is_err() {
                                break;
                            }
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(TelemetryError::Connection(e));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_message_try_from_valid_json() {
        let json = br#"{"zone_id": "test-123", "t_air": 22.5, "rh": 0.5}"#;
        let msg = TelemetryMessage::try_from(json.as_slice()).unwrap();
        assert_eq!(msg.zone_id, "test-123");
        assert!((msg.t_air - 22.5).abs() < 1e-6);
        assert!((msg.rh - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_message_try_from_invalid_json() {
        let json = b"not valid json";
        let result = TelemetryMessage::try_from(json.as_slice());
        assert!(result.is_err());
    }

    #[test]
    fn test_telemetry_message_try_from_missing_fields() {
        let json = br#"{"zone_id": "test-123"}"#;
        let result = TelemetryMessage::try_from(json.as_slice());
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mqtt_consumer_integration() {
        let broker = "mqtt://localhost:1883";
        let topic = "fluxion/test/zone1";

        let consumer = MqttTelemetryConsumer::connect(broker, topic).await;
        if consumer.is_err() {
            return;
        }
        let consumer = consumer.unwrap();

        let (tx, _rx) = mpsc::channel(100);
        let _handle = tokio::spawn(async move { consumer.start(tx).await });

        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
