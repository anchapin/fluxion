// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Memory leak test for LiveTwin WebSocket broadcaster (Issue #2064).
//!
//! Tests that broadcasting 1000 sequential states at 60 FPS does not leak memory.
//! Memory growth must stay under 1MB RSS.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fluxion::twin::{create_test_payload, LiveTwinBroadcaster};
use memory_stats::memory_stats;
use tokio::time::sleep;

#[tokio::test]
async fn test_no_memory_leak_1000_states() {
    let broadcaster = LiveTwinBroadcaster::new();

    let (_id, mut rx) = broadcaster.subscribe();

    let received = Arc::new(AtomicUsize::new(0));
    let received_clone = received.clone();

    let handle = tokio::spawn(async move {
        let mut count = 0;
        while count < 1000 {
            if rx.recv().await.is_ok() {
                count += 1;
            }
        }
        received_clone.store(count, Ordering::SeqCst);
    });

    let mem_before = memory_stats().map(|m| m.physical_mem).unwrap_or(0);

    for i in 0..1000 {
        let payload = create_test_payload(i);
        broadcaster.broadcast(&payload).unwrap();
        sleep(Duration::from_millis(16)).await;
    }

    handle.await.unwrap();

    sleep(Duration::from_secs(2)).await;

    let mem_after = memory_stats().map(|m| m.physical_mem).unwrap_or(0);

    let count = received.load(Ordering::SeqCst);
    assert_eq!(count, 1000, "Client should receive all 1000 messages");

    let memory_growth = mem_after as i64 - mem_before as i64;
    assert!(
        memory_growth.abs() < 1024 * 1024,
        "Memory leak detected: {} bytes growth",
        memory_growth
    );
}
