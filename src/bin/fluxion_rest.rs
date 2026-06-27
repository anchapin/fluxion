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
//!
//! Graceful shutdown is wired to `SIGINT` (Ctrl-C) via `tokio::signal`.

use std::net::SocketAddr;
use std::str::FromStr;

use fluxion::api::server::{router, AppState};
use tokio::net::TcpListener;

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
    // default to `info`. We use env_logger because it is already in our
    // dependency tree.
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "fluxion=info,fluxion_rest=info,info");
    }
    let _ = env_logger::try_init();

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
}