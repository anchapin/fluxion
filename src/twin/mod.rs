// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! LiveTwin WebSocket subsystem (Issues #2063, #2062).
//!
//! Provides connection pool management, heartbeat, and MessagePack binary
//! serialization for real-time WebSocket-based twin synchronization.
//!
//! # Modules
//!
//! - [`connection_pool`] - Thread-safe connection pool with heartbeat
//! - [`live_twin_broadcaster`] - WebSocket server with MessagePack serialization

pub mod connection_pool;
pub mod live_twin_broadcaster;

pub use connection_pool::{
    ConnectionId, ConnectionPool, ConnectionState, PoolError, DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MAX_CONNECTIONS, DEFAULT_STALE_TIMEOUT,
};
pub use live_twin_broadcaster::{
    BroadcastError, LiveTwinBroadcaster, LiveTwinPayload, ZoneState, DEFAULT_LIVE_TWIN_PORT,
    LIVE_TWIN_PATH,
};
