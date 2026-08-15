//! Telemetry broadcaster — fan-out for telemetry payloads (Issue #2064).
//!
//! A lightweight wrapper over [`tokio::sync::broadcast`] that lets multiple
//! downstream consumers (REST API, MQTT re-publisher, recorder, …) subscribe
//! to a stream of telemetry messages from a single source without coupling to
//! the filter's internal state.
//!
//! Mirrors the WebSocket broadcaster pattern in the main crate's
//! [`LiveTwinBroadcaster`](../../../../src/twin/live_twin_broadcaster.rs) but
//! operates fully in-process via `tokio::sync::broadcast`. Each subscriber
//! receives every payload; if a subscriber lags behind, older payloads are
//! dropped (lossy semantics) — exactly like a WebSocket frame broadcast.
//!
//! # Example
//!
//! ```no_run
//! use fluxion_twin::telemetry::TwinBroadcaster;
//!
//! # async fn run() {
//! let broadcaster: TwinBroadcaster<String> = TwinBroadcaster::new();
//! let mut rx = broadcaster.subscribe();
//!
//! broadcaster.send("frame-1".to_string()).unwrap();
//! let frame = rx.recv().await.unwrap();
//! assert_eq!(&*frame, "frame-1");
//! # }
//! ```

use std::sync::Arc;

use thiserror::Error;
use tokio::sync::broadcast;

/// Default broadcast channel capacity — mirrors the main crate's
/// `LiveTwinBroadcaster` (broadcast::channel(100) for payload + 100 for
/// per-connection backpressure in `live_twin_broadcaster.rs`).
pub const DEFAULT_BROADCAST_CAPACITY: usize = 1024;

/// Errors emitted by [`TwinBroadcaster::send`].
#[derive(Debug, Error)]
pub enum BroadcastError {
    /// No active receivers are currently subscribed. The payload was dropped.
    #[error("no active receivers — broadcast dropped the payload")]
    NoActiveReceivers,
}

/// Lock-free fan-out broadcaster for telemetry / state-correction payloads.
///
/// Holds a single [`broadcast::Sender<Arc<T>>`]; each [`Self::subscribe`] call
/// hands out a fresh [`broadcast::Receiver`]. Payloads are wrapped in
/// [`Arc<T>`] so every receiver can read them concurrently without copying the
/// underlying data.
///
/// Cloning is cheap (the inner `Sender` is itself an `Arc`), so a single
/// broadcaster can be shared across threads via `Arc<TwinBroadcaster<T>>`.
pub struct TwinBroadcaster<T>
where
    T: Clone + Send + Sync + 'static,
{
    tx: broadcast::Sender<Arc<T>>,
}

impl<T> TwinBroadcaster<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a new broadcaster with [`DEFAULT_BROADCAST_CAPACITY`].
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BROADCAST_CAPACITY)
    }

    /// Create a new broadcaster with the given ring-buffer capacity.
    ///
    /// `capacity` is the maximum number of un-consumed payloads each subscriber
    /// may lag behind by. When a subscriber falls further behind, the oldest
    /// payloads are dropped (lossy).
    pub fn with_capacity(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Subscribe to the broadcaster.
    ///
    /// Each call returns an independent receiver; receivers see payloads from
    /// the moment they subscribe (not from broadcaster creation).
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<T>> {
        self.tx.subscribe()
    }

    /// Broadcast `payload` to every active subscriber.
    ///
    /// Returns the number of subscribers that received the payload. Returns
    /// [`BroadcastError::NoActiveReceivers`] (and drops the payload) if no
    /// receiver is currently subscribed — this matches the lossy semantics of
    /// a WebSocket frame broadcast to zero connections.
    pub fn send(&self, payload: T) -> Result<usize, BroadcastError> {
        let arc = Arc::new(payload);
        self.tx
            .send(arc)
            .map_err(|_| BroadcastError::NoActiveReceivers)
    }

    /// Number of currently-active receivers.
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl<T> Default for TwinBroadcaster<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for TwinBroadcaster<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        // `broadcast::Sender` is internally Arc-backed, so cloning the sender
        // (and therefore the broadcaster) is cheap and threadsafe.
        Self {
            tx: self.tx.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn send_returns_receiver_count() {
        let broadcaster: TwinBroadcaster<u32> = TwinBroadcaster::new();
        let _r1 = broadcaster.subscribe();
        let _r2 = broadcaster.subscribe();
        assert_eq!(broadcaster.receiver_count(), 2);

        let count = broadcaster.send(42).unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn send_to_zero_receivers_returns_error() {
        let broadcaster: TwinBroadcaster<u32> = TwinBroadcaster::new();
        assert_eq!(broadcaster.receiver_count(), 0);
        assert!(matches!(
            broadcaster.send(1),
            Err(BroadcastError::NoActiveReceivers)
        ));
    }

    #[tokio::test]
    async fn receivers_get_payload() {
        let broadcaster: TwinBroadcaster<String> = TwinBroadcaster::new();
        let mut rx = broadcaster.subscribe();
        broadcaster.send("hello".to_string()).unwrap();
        let got = rx.recv().await.unwrap();
        assert_eq!(&*got, "hello");
    }

    #[test]
    fn clone_shares_sender() {
        let broadcaster: TwinBroadcaster<u32> = TwinBroadcaster::new();
        let clone = broadcaster.clone();
        let _rx = clone.subscribe();
        assert_eq!(broadcaster.receiver_count(), 1);
    }
}
