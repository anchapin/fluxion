//! Telemetry message definition for backpressure-aware telemetry.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryMsg {
    pub sensor_id: Uuid,
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    pub payload: Vec<f64>,
}

impl TelemetryMsg {
    pub fn new(sensor_id: Uuid, sequence: u64, payload: Vec<f64>) -> Self {
        Self {
            sensor_id,
            sequence,
            timestamp: Utc::now(),
            payload,
        }
    }
}
