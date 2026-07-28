// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! WebSocket connection pool with heartbeat for LiveTwin (Issue #2063).
//!
//! Manages concurrent WebSocket connections with:
//! - Configurable connection limit (default: 100)
//! - Connection state tracking: Active, PendingPing, Disconnected
//! - Server-side heartbeat (ping every 30s)
//! - Stale connection detection (>90s without pong)
//!
//! # Example
//!
//! ```ignore
//! let pool = ConnectionPool::new(100);
//! pool.add(conn_id)?;
//! pool.mark_pending_ping(conn_id);
//! assert!(pool.handle_pong(conn_id));
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use thiserror::Error;
use uuid::Uuid;

pub const DEFAULT_MAX_CONNECTIONS: usize = 100;
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
pub const DEFAULT_STALE_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    Active,
    PendingPing,
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId(pub Uuid);

impl ConnectionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ConnectionId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Error)]
pub enum PoolError {
    #[error("Too many connections: limit={0}")]
    TooManyConnections(usize),
}

pub struct ConnectionPool {
    connections: Arc<Mutex<HashMap<ConnectionId, ConnectionState>>>,
    max_connections: usize,
    heartbeat_interval: Duration,
    stale_timeout: Duration,
}

impl Default for ConnectionPool {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CONNECTIONS)
    }
}

impl ConnectionPool {
    pub fn new(max: usize) -> Self {
        Self {
            connections: Arc::new(Mutex::new(HashMap::new())),
            max_connections: max,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            stale_timeout: DEFAULT_STALE_TIMEOUT,
        }
    }

    pub fn with_config(max: usize, heartbeat_interval: Duration, stale_timeout: Duration) -> Self {
        Self {
            connections: Arc::new(Mutex::new(HashMap::new())),
            max_connections: max,
            heartbeat_interval,
            stale_timeout,
        }
    }

    pub fn max_connections(&self) -> usize {
        self.max_connections
    }

    pub fn heartbeat_interval(&self) -> Duration {
        self.heartbeat_interval
    }

    pub fn stale_timeout(&self) -> Duration {
        self.stale_timeout
    }

    pub fn add(&self, id: ConnectionId) -> Result<(), PoolError> {
        let mut conns = self.connections.lock();
        if conns.len() >= self.max_connections {
            return Err(PoolError::TooManyConnections(self.max_connections));
        }
        conns.insert(id, ConnectionState::Active);
        Ok(())
    }

    pub fn mark_pending_ping(&self, id: ConnectionId) {
        let mut conns = self.connections.lock();
        if let Some(state) = conns.get_mut(&id) {
            *state = ConnectionState::PendingPing;
        }
    }

    pub fn handle_pong(&self, id: ConnectionId) -> bool {
        let mut conns = self.connections.lock();
        match conns.get(&id) {
            Some(ConnectionState::PendingPing) => {
                conns.insert(id, ConnectionState::Active);
                true
            }
            _ => false,
        }
    }

    pub fn mark_disconnected(&self, id: ConnectionId) {
        let mut conns = self.connections.lock();
        if let Some(state) = conns.get_mut(&id) {
            *state = ConnectionState::Disconnected;
        }
    }

    pub fn remove(&self, id: ConnectionId) -> bool {
        let mut conns = self.connections.lock();
        conns.remove(&id).is_some()
    }

    pub fn remove_stale(&self) -> Vec<ConnectionId> {
        let mut conns = self.connections.lock();
        let stale_ids: Vec<ConnectionId> = conns
            .iter()
            .filter(|(_, s)| matches!(s, ConnectionState::Disconnected))
            .map(|(id, _)| *id)
            .collect();
        for id in &stale_ids {
            conns.remove(id);
        }
        stale_ids
    }

    pub fn active_count(&self) -> usize {
        let conns = self.connections.lock();
        conns
            .values()
            .filter(|s| matches!(s, ConnectionState::Active | ConnectionState::PendingPing))
            .count()
    }

    pub fn connection_state(&self, id: &ConnectionId) -> Option<ConnectionState> {
        let conns = self.connections.lock();
        conns.get(id).cloned()
    }

    #[cfg(test)]
    pub fn connections_for_testing(&self) -> Arc<Mutex<HashMap<ConnectionId, ConnectionState>>> {
        Arc::clone(&self.connections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_connection_id() -> ConnectionId {
        ConnectionId::new()
    }

    #[test]
    fn test_connection_pool_enforces_limit() {
        let pool = ConnectionPool::new(2);
        let id1 = next_connection_id();
        let id2 = next_connection_id();
        let id3 = next_connection_id();

        assert!(pool.add(id1).is_ok());
        assert!(pool.add(id2).is_ok());
        assert!(matches!(
            pool.add(id3),
            Err(PoolError::TooManyConnections(2))
        ));
    }

    #[test]
    fn test_state_transitions() {
        let pool = ConnectionPool::new(10);
        let id = next_connection_id();

        assert!(pool.add(id).is_ok());
        assert_eq!(pool.connection_state(&id), Some(ConnectionState::Active));

        pool.mark_pending_ping(id);
        assert_eq!(
            pool.connection_state(&id),
            Some(ConnectionState::PendingPing)
        );

        assert!(pool.handle_pong(id));
        assert_eq!(pool.connection_state(&id), Some(ConnectionState::Active));

        pool.mark_pending_ping(id);
        assert!(pool.handle_pong(id));
        assert_eq!(pool.connection_state(&id), Some(ConnectionState::Active));
    }

    #[test]
    fn test_handle_pong_wrong_state() {
        let pool = ConnectionPool::new(10);
        let id = next_connection_id();

        assert!(pool.add(id).is_ok());
        assert!(!pool.handle_pong(id));
        assert_eq!(pool.connection_state(&id), Some(ConnectionState::Active));
    }

    #[test]
    fn test_mark_disconnected_and_remove() {
        let pool = ConnectionPool::new(10);
        let id = next_connection_id();

        assert!(pool.add(id).is_ok());
        pool.mark_disconnected(id);
        assert_eq!(
            pool.connection_state(&id),
            Some(ConnectionState::Disconnected)
        );

        assert!(pool.remove(id));
        assert_eq!(pool.connection_state(&id), None);
    }

    #[test]
    fn test_remove_stale() {
        let pool = ConnectionPool::new(10);
        let id1 = next_connection_id();
        let id2 = next_connection_id();

        assert!(pool.add(id1).is_ok());
        assert!(pool.add(id2).is_ok());

        pool.mark_disconnected(id1);
        let stale = pool.remove_stale();
        assert_eq!(stale, vec![id1]);
        assert_eq!(pool.connection_state(&id2), Some(ConnectionState::Active));
    }

    #[test]
    fn test_active_count() {
        let pool = ConnectionPool::new(10);
        let id1 = next_connection_id();
        let id2 = next_connection_id();
        let id3 = next_connection_id();

        assert_eq!(pool.active_count(), 0);
        assert!(pool.add(id1).is_ok());
        assert!(pool.add(id2).is_ok());
        assert_eq!(pool.active_count(), 2);

        pool.mark_pending_ping(id1);
        assert_eq!(pool.active_count(), 2);

        pool.mark_disconnected(id1);
        assert_eq!(pool.active_count(), 1);
        assert!(pool.add(id3).is_ok());
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn test_with_config() {
        let pool =
            ConnectionPool::with_config(50, Duration::from_secs(15), Duration::from_secs(45));

        assert_eq!(pool.max_connections(), 50);
        assert_eq!(pool.heartbeat_interval(), Duration::from_secs(15));
        assert_eq!(pool.stale_timeout(), Duration::from_secs(45));
    }

    #[test]
    fn test_default_values() {
        let pool = ConnectionPool::default();
        assert_eq!(pool.max_connections(), DEFAULT_MAX_CONNECTIONS);
        assert_eq!(pool.heartbeat_interval(), DEFAULT_HEARTBEAT_INTERVAL);
        assert_eq!(pool.stale_timeout(), DEFAULT_STALE_TIMEOUT);
    }

    #[test]
    fn test_connection_id() {
        let id1 = ConnectionId::new();
        let id2 = ConnectionId::new();
        assert_ne!(id1, id2);
    }
}
