// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `fluxion-rest` — Issue #1342 binary entrypoint.
//!
//! Boots the axum HTTP server described in [`fluxion::api::server`].
//! Defaults to `0.0.0.0:8080`; both bind address and port can be overridden
//! by environment variables:
//!
//! - `FLUXION_REST_BIND`  — default `0.0.0.0`
//! - `FLUXION_REST_PORT`  — default `8080`
//! - `RUST_LOG`           — `tracing-subscriber` filter (forwarded into the
//!                          layer set up below).
//!
//! Graceful shutdown is wired to `SIGINT` (Ctrl-C) via `tokio::signal`.

use std::net::SocketAddr;
use std::str::FromStr;

use fluxion::api::server::{router, AppState};
use tokio::net::TcpListener;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

const DEFAULT_BIND: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8080";

fn resolve_bind() -> String {
    std::env::var("FLUXION_REST_BIND").unwrap_or_else(|_| DEFAULT_BIND.to_string())
}

fn resolve_port() -> u16 {
    let raw = std::env::var("FLUXION_REST_PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());
    match u16::from_str(&raw) {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "fluxion-rest: FLUXION_REST_PORT='{raw}' is not a valid u16; falling back to {DEFAULT_PORT}"
            );
            DEFAULT_PORT
                .parse::<u16>()
                .expect("DEFAULT_PORT must parse")
        }
    }
}

fn resolve_addr() -> SocketAddr {
    let bind = resolve_bind();
    let port = resolve_port();
    // Build a SocketAddr from `(bind, port)`. If the user gave a bare IPv4
    // address without a port, the format!() below yields `addr:port` which
    // FromStr accepts.
    let s = format!("{bind}:{port}");
    SocketAddr::from_str(&s).unwrap_or_else(|e| {
        eprintln!(
            "fluxion-rest: invalid bind '{bind}:{port}' ({e}); falling back to 0.0.0.0:{port}"
        );
        SocketAddr::from_str(&format!("0.0.0.0:{port}")).expect("fallback addr must parse")
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging; respect RUST_LOG if the operator set it, otherwise
    // default to `info`. We use *both* `env_logger` (for any `log::info!`
    // lines emitted by dependencies that bypass `tracing`) and a
    // `tracing_subscriber::fmt` layer so `TraceLayer`-emitted per-request
    // log lines (Issue #1447) make it to stdout.
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "fluxion=info,fluxion_rest=info,info");
    }
    let _ = env_logger::try_init();

    // Build a `tracing-subscriber` registry that respects `RUST_LOG` and
    // writes structured human-readable lines. `try_init` makes the call
    // idempotent so running tests under `--nocapture` does not panic when
    // a second binary in the same process tries to install a subscriber.
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("fluxion=info,fluxion_rest=info,info"));
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_level(true)
        .with_thread_ids(false)
        .compact();
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .try_init();

    let addr = resolve_addr();
    let listener = TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    tracing_or_log_info(&format!(
        "fluxion-rest listening on {bound} (FLUXION_REST_BIND={} FLUXION_REST_PORT={})",
        resolve_bind(),
        bound.port()
    ));

    let app = router(AppState::default());

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Cheap log helper — `tracing` is wired through the workspace but the
/// binary imports `env_logger` directly; this lets us pick either one
/// depending on which subscribers the operator initialized.
fn tracing_or_log_info(msg: &str) {
    tracing::info!("{msg}");
    log::info!("{msg}");
}

/// Wait for a SIGINT (Ctrl-C) and resolve. Returns immediately after the
/// first signal so axum can finish in-flight requests.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    log::info!("fluxion-rest: shutdown signal received");
    tracing::info!("fluxion-rest: shutdown signal received");
}
