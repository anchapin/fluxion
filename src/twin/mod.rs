// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! LiveTwin WebSocket subsystem (Issue #2063).
//!
//! Provides connection pool management and heartbeat for real-time
//! WebSocket-based twin synchronization.
//!
//! # Modules
//!
//! - [`connection_pool`] - Thread-safe connection pool with heartbeat

pub mod connection_pool;

pub use connection_pool::{
    ConnectionId, ConnectionPool, ConnectionState, PoolError, DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MAX_CONNECTIONS, DEFAULT_STALE_TIMEOUT,
};

//! LiveTwin WebSocket state broadcaster (Issue #2025).
//!
//! Streams physical state vectors to visualization engines via WebSockets
//! with MessagePack binary serialization for ~2x bandwidth savings vs JSON.
//!
//! # Memory Leak Testing
//! Issue #2064 requires a memory leak test that broadcasts 1000 sequential
//! states at 60 FPS without memory growth exceeding 1MB RSS.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneState {
    pub zone_id: Uuid,
    pub t_air: f64,
    pub t_mass: f64,
    pub rh: f64,
    pub energy_consumed: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTwinPayload {
    pub timestamp: DateTime<Utc>,
    pub simulation_id: Uuid,
    pub zone_states: Vec<ZoneState>,
}

pub struct LiveTwinBroadcaster {
    connections: Arc<RwLock<HashMap<Uuid, broadcast::Sender<LiveTwinPayload>>>>,
    simulation_id: Uuid,
}

impl LiveTwinBroadcaster {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            simulation_id: Uuid::new_v4(),
        }
    }

    pub fn with_simulation_id(simulation_id: Uuid) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            simulation_id,
        }
    }

    pub fn simulation_id(&self) -> Uuid {
        self.simulation_id
    }

    pub fn broadcast(&self, payload: &LiveTwinPayload) -> Result<usize, BroadcastError> {
        let senders = self.connections.read();
        let mut count = 0;
        for sender in senders.values() {
            if sender.send(payload.clone()).is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    pub fn subscribe(&self) -> (Uuid, broadcast::Receiver<LiveTwinPayload>) {
        let (tx, rx) = broadcast::channel(1024);
        let id = Uuid::new_v4();
        self.connections.write().insert(id, tx);
        (id, rx)
    }

    pub fn unsubscribe(&self, id: Uuid) {
        self.connections.write().remove(&id);
    }

    pub fn connection_count(&self) -> usize {
        self.connections.read().len()
    }
}

impl Default for LiveTwinBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BroadcastError {
    #[error("broadcast channel error")]
    ChannelError(#[from] broadcast::error::SendError<LiveTwinPayload>),
}

pub fn create_test_payload(index: usize) -> LiveTwinPayload {
    LiveTwinPayload {
        timestamp: Utc::now(),
        simulation_id: Uuid::new_v4(),
        zone_states: vec![ZoneState {
            zone_id: Uuid::new_v4(),
            t_air: 20.0 + (index as f64 * 0.1),
            t_mass: 22.0 + (index as f64 * 0.1),
            rh: 50.0,
            energy_consumed: index as f64 * 100.0,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcaster_creation() {
        let broadcaster = LiveTwinBroadcaster::new();
        assert_eq!(broadcaster.connection_count(), 0);
    }

    #[test]
    fn test_broadcast_single_subscriber() {
        let broadcaster = LiveTwinBroadcaster::new();
        let (_id, mut rx) = broadcaster.subscribe();

        let payload = create_test_payload(0);
        let count = broadcaster.broadcast(&payload).unwrap();
        assert_eq!(count, 1);

        let received = rx.blocking_recv().unwrap();
        assert_eq!(received.timestamp, payload.timestamp);
    }

    #[test]
    fn test_broadcast_multiple_subscribers() {
        let broadcaster = LiveTwinBroadcaster::new();
        let (_id1, mut rx1) = broadcaster.subscribe();
        let (_id2, mut rx2) = broadcaster.subscribe();

        let payload = create_test_payload(0);
        broadcaster.broadcast(&payload).unwrap();

        let received1 = rx1.blocking_recv().unwrap();
        let received2 = rx2.blocking_recv().unwrap();
        assert_eq!(received1.timestamp, payload.timestamp);
        assert_eq!(received2.timestamp, payload.timestamp);
    }

    #[test]
    fn test_unsubscribe() {
        let broadcaster = LiveTwinBroadcaster::new();
        let (id, mut rx) = broadcaster.subscribe();
        assert_eq!(broadcaster.connection_count(), 1);

        broadcaster.unsubscribe(id);
        assert_eq!(broadcaster.connection_count(), 0);

        let payload = create_test_payload(0);
        broadcaster.broadcast(&payload).unwrap();
        assert!(rx.blocking_recv().is_err());
    }

    #[test]
    fn test_payload_serialization() {
        let payload = create_test_payload(42);
        let encoded = rmp_serde::to_vec(&payload).unwrap();
        let decoded: LiveTwinPayload = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded.zone_states[0].t_air, payload.zone_states[0].t_air);
    }
}
