//! Error types for telemetry backpressure and deduplication.

use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TelemetryError {
    #[error("channel receive error: {0}")]
    Recv(String),

    #[error("channel send error")]
    Send,

    #[error("backpressure: channel at capacity, slow consumer")]
    Backpressure,

    #[error("out-of-order buffer full for sensor {0}")]
    BufferFull(Uuid),
}
