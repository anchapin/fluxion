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
