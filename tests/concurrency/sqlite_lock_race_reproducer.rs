//! SQLite "database is locked" race condition reproducer for distributed array jobs (Issue #1790)
//!
//! ## Background
//!
//! OSimFlow's distributed array jobs run multiple concurrent simulation workers that
//! each write results to a shared SQLite database (e.g., EnergyPlus output databases).
//! Under high concurrency, workers hit `SQLITE_BUSY` / "database is locked" errors because:
//!
//! 1. **Writer serialization** — SQLite allows only one writer at a time, using a
//!    [database-level lock](https://www.sqlite.org/ltsv/locklrm.html). When one
//!    writer holds the lock, all other writers receive `SQLITE_BUSY`.
//!
//! 2. **Default busy timeout = 0** — The default `busy_timeout` is 0, meaning
//!    SQLite returns `SQLITE_BUSY` immediately instead of retrying. Workers that
//!    don't set a busy timeout will fail on the first contention.
//!
//! 3. **Long-running transactions** — A writer holding a write transaction (e.g.,
//!    during a bulk INSERT) blocks all other writers for the duration.
//!
//! ## Root Cause Analysis
//!
//! The race occurs in this sequence:
//!
//! ```text
//! Worker A                 Worker B                 Worker C
//! ──────────────────────────────────────────────────────────────────────
//! BEGIN EXCLUSIVE (acquired) BEGIN EXCLUSIVE ──► SQLITE_BUSY (busy_timeout=0)
//! INSERT...                → "database is locked"
//! COMMIT
//! ```
//!
//! The root cause is **writer concurrency / lock timeout** — SQLite's serialized
//! writer model cannot accommodate concurrent write transactions without either:
//! - Increasing `busy_timeout` to allow retries, OR
//! - Moving to a distributed cache (T8.2) that handles concurrent writers
//!
//! ## Acceptance Criteria (Issue #1790)
//!
//! - [x] Isolated test mock that reproduces the lock contention under concurrent workers
//! - [x] Documented root-cause analysis (writer concurrency / lock timeout)

use rusqlite::{params, Connection, Result as SqlResult};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

/// Number of concurrent workers in the reproducer test.
const NUM_WORKERS: usize = 16;

/// Number of rows each worker attempts to INSERT per transaction.
const ROWS_PER_WORKER: usize = 50;

/// Number of transactions each worker runs.
const TXS_PER_WORKER: usize = 20;

/// Creates a temporary SQLite database with a results table.
fn setup_database(path: &str) -> SqlResult<Connection> {
    let conn = Connection::open(path)?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS simulation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            energy_kwh REAL NOT NULL,
            timestamp TEXT NOT NULL
        )",
        [],
    )?;
    Ok(conn)
}

/// Demonstrates the SQLite "database is locked" race condition under concurrent writers.
///
/// This test reproduces Issue #1790 where distributed array jobs hit `database is locked`
/// errors when multiple concurrent workers try to write to the same SQLite database.
#[test]
fn test_sqlite_database_locked_race_reproducer() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let db_path = temp_dir.path().join("test_concurrent_writes.db");
    let db_path_str = db_path.to_str().unwrap();

    {
        let conn = setup_database(db_path_str).expect("failed to create database");
        // busy_timeout=0 means SQLite returns SQLITE_BUSY immediately instead of retrying
        conn.busy_timeout(std::time::Duration::ZERO)
            .expect("failed to set busy_timeout");
        drop(conn);
    }

    let total_success = Arc::new(AtomicUsize::new(0));
    let total_busy_errors = Arc::new(AtomicUsize::new(0));
    let total_begin_failures = Arc::new(AtomicUsize::new(0));
    let start_barrier = Arc::new(Barrier::new(NUM_WORKERS));

    let mut handles = vec![];
    for worker_id in 0..NUM_WORKERS {
        let db_path_str = db_path_str.to_string();
        let barrier = start_barrier.clone();
        let success = total_success.clone();
        let busy = total_busy_errors.clone();
        let begin_fail = total_begin_failures.clone();

        let handle = thread::spawn(move || {
            let conn = Connection::open(&db_path_str).expect("failed to open database");
            conn.busy_timeout(std::time::Duration::ZERO)
                .expect("failed to set busy_timeout");

            barrier.wait();

            for tx_id in 0..TXS_PER_WORKER {
                // BEGIN EXCLUSIVE fails immediately if database is locked (busy_timeout=0)
                let begin_result = conn.execute_batch("BEGIN EXCLUSIVE");
                if begin_result.is_err() {
                    begin_fail.fetch_add(1, Ordering::Relaxed);
                    busy.fetch_add(ROWS_PER_WORKER, Ordering::Relaxed);
                    continue;
                }

                let mut tx_success = 0;
                let mut tx_busy = 0;

                for run_id in 0..ROWS_PER_WORKER {
                    let insert_result = conn.execute(
                        "INSERT INTO simulation_results (worker_id, run_id, energy_kwh, timestamp)
                         VALUES (?1, ?2, ?3, datetime('now'))",
                        params![
                            worker_id as i32,
                            (tx_id * ROWS_PER_WORKER + run_id) as i32,
                            (worker_id as f64 * 100.0 + run_id as f64)
                        ],
                    );

                    match insert_result {
                        Ok(_) => tx_success += 1,
                        Err(rusqlite::Error::SqliteFailure(e, _))
                            if e.code == rusqlite::ErrorCode::DatabaseBusy =>
                        {
                            tx_busy += 1;
                        }
                        Err(_) => {}
                    }
                }

                let commit_result = conn.execute_batch("COMMIT");
                if commit_result.is_err() {
                    let remaining = ROWS_PER_WORKER - tx_success - tx_busy;
                    busy.fetch_add(remaining, Ordering::Relaxed);
                }

                success.fetch_add(tx_success, Ordering::Relaxed);
                busy.fetch_add(tx_busy, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("worker thread should not panic");
    }

    let total_attempts = NUM_WORKERS * TXS_PER_WORKER * ROWS_PER_WORKER;
    let total_successes = total_success.load(Ordering::Relaxed);
    let total_busy = total_busy_errors.load(Ordering::Relaxed);
    let total_begin = total_begin_failures.load(Ordering::Relaxed);

    println!(
        "SQLite lock contention results (busy_timeout=0):
        - Total attempts: {}
        - Successful writes: {}
        - SQLITE_BUSY errors: {}
        - BEGIN EXCLUSIVE failures: {}",
        total_attempts, total_successes, total_busy, total_begin
    );

    // The test reproduces the race condition - BEGIN EXCLUSIVE should fail when
    // another transaction holds the lock, even with busy_timeout=0
    assert!(
        total_busy > 0 || total_begin > 0,
        "Expected at least one SQLITE_BUSY or BEGIN failure to reproduce the race condition.\n\
         Got busy={}, begin_failures={}",
        total_busy,
        total_begin
    );

    let conn = Connection::open(db_path_str).expect("failed to reopen database");
    let row_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM simulation_results", [], |row| {
            row.get(0)
        })
        .expect("failed to query row count");

    assert_eq!(
        row_count, total_successes as i32,
        "Row count in database should match successful writes"
    );
}

/// Demonstrates that with busy_timeout > 0, workers can retry and eventually succeed,
/// but high contention can still exceed the retry window.
#[test]
fn test_sqlite_busy_timeout_retry() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let db_path = temp_dir.path().join("test_busy_timeout.db");
    let db_path_str = db_path.to_str().unwrap();

    let conn = setup_database(db_path_str).expect("failed to create database");
    // Set a 100ms busy timeout - workers will retry for up to 100ms
    conn.busy_timeout(std::time::Duration::from_millis(100))
        .expect("failed to set busy_timeout");
    drop(conn);

    let start_barrier = Arc::new(Barrier::new(NUM_WORKERS));
    let total_success = Arc::new(AtomicUsize::new(0));
    let total_busy_errors = Arc::new(AtomicUsize::new(0));
    let total_begin_failures = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    for worker_id in 0..NUM_WORKERS {
        let db_path_str = db_path_str.to_string();
        let barrier = start_barrier.clone();
        let success = total_success.clone();
        let busy = total_busy_errors.clone();
        let begin_fail = total_begin_failures.clone();

        let handle = thread::spawn(move || {
            let conn = Connection::open(&db_path_str).expect("failed to open database");
            conn.busy_timeout(std::time::Duration::from_millis(100))
                .expect("failed to set busy_timeout");

            barrier.wait();

            for tx_id in 0..TXS_PER_WORKER {
                // BEGIN IMMEDIATE waits for busy_timeout if lock is held
                let result = conn.execute_batch("BEGIN IMMEDIATE");
                if result.is_err() {
                    begin_fail.fetch_add(1, Ordering::Relaxed);
                    busy.fetch_add(ROWS_PER_WORKER, Ordering::Relaxed);
                    continue;
                }

                let mut tx_success = 0;
                let mut tx_busy = 0;

                for run_id in 0..ROWS_PER_WORKER {
                    let insert_result = conn.execute(
                        "INSERT INTO simulation_results (worker_id, run_id, energy_kwh, timestamp)
                         VALUES (?1, ?2, ?3, datetime('now'))",
                        params![
                            worker_id as i32,
                            (tx_id * ROWS_PER_WORKER + run_id) as i32,
                            (worker_id as f64 * 100.0 + run_id as f64)
                        ],
                    );

                    match insert_result {
                        Ok(_) => tx_success += 1,
                        Err(rusqlite::Error::SqliteFailure(e, _))
                            if e.code == rusqlite::ErrorCode::DatabaseBusy =>
                        {
                            tx_busy += 1;
                        }
                        Err(_) => {}
                    }
                }

                let commit_result = conn.execute_batch("COMMIT");
                if commit_result.is_err() {
                    let remaining = ROWS_PER_WORKER - tx_success - tx_busy;
                    busy.fetch_add(remaining, Ordering::Relaxed);
                }

                success.fetch_add(tx_success, Ordering::Relaxed);
                busy.fetch_add(tx_busy, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("worker thread should not panic");
    }

    let total_successes = total_success.load(Ordering::Relaxed);
    let total_busy = total_busy_errors.load(Ordering::Relaxed);
    let total_begin = total_begin_failures.load(Ordering::Relaxed);

    println!(
        "With busy_timeout=100ms:
        - Successful writes: {}
        - SQLITE_BUSY errors: {}
        - BEGIN IMMEDIATE failures: {}",
        total_successes, total_busy, total_begin
    );

    // With busy_timeout, some BEGIN IMMEDIATE calls will wait and retry
    // High contention can still cause failures when timeout is exceeded
}
