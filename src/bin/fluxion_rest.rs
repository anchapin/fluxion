// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `fluxion-rest` — Issue #1342 binary entrypoint.
//!
//! Boots the axum HTTP server described in [`fluxion::api::server`].
//! Defaults to `0.0.0.0:8080`; both bind address and port can be overridden
//! by environment variables:
//!
//! - `FLUXION_REST_BIND` — default `0.0.0.0`
//! - `FLUXION_REST_PORT` — default `8080`
//! - `RUST_LOG` — `tracing-subscriber` filter (forwarded into the layer set
//!   up below).
//! - `FLUXION_AUDIT_LOG` — optional path; when set, `/v1/simulate` audit
//!   events (`target = "audit"`) are tee'd to this file in addition to the
//!   default stdout log (Issue #2546).
//!
//! # Security (Issue #2505)
//!
//! Every `/v1/*` route is wrapped by auth + CORS + a 16 MiB body limit +
//! a per-IP token-bucket governor, configured from the environment via
//! [`fluxion::api::security::RestSecurityConfig`]:
//!
//! - `FLUXION_REST_AUTH` — `off` (default) | `token` | `tls`
//! - `FLUXION_REST_AUTH_TOKEN` — bearer token (required for `token`)
//! - `FLUXION_REST_CORS_ORIGINS` — comma-separated origin allow-list
//! - `FLUXION_REST_RATE_LIMIT_RPS` / `FLUXION_REST_RATE_LIMIT_BURST`
//! - `FLUXION_REST_ALLOW_INSECURE=1` — opt out of the release-build boot
//!   guard that refuses `0.0.0.0` + `auth=off`.
//!
//! Graceful shutdown is wired to `SIGINT` (Ctrl-C) via `tokio::signal`.

use std::net::SocketAddr;
use std::str::FromStr;

use fluxion::api::security::{check_boot_guard_from_env, RestSecurityConfig};
use fluxion::api::server::{router_with_security, run_readiness_probes, AppState};
use tokio::net::TcpListener;
use tracing::Level;
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

    // Issue #2546 — opt-in dedicated audit log. When `FLUXION_AUDIT_LOG`
    // points at a writable path, install a second subscriber layer that
    // funnels only `target = "audit"` events (the `simulation_started` /
    // `simulation_completed` records emitted by `/v1/simulate`) into that
    // file as plain lines. When unset or unopenable, audit events still
    // flow to the default `fmt_layer` above via the bare `info` filter, so
    // nothing is silently dropped.
    let audit_path = std::env::var("FLUXION_AUDIT_LOG").ok();
    let audit_file = audit_path.as_ref().and_then(|p| {
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
        {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!(
                    "fluxion-rest: FLUXION_AUDIT_LOG='{p}' open failed ({e}); audit events stay on stdout only"
                );
                None
            }
        }
    });

    match audit_file {
        Some(file) => {
            // `Mutex<File>` implements `for<'a> MakeWriter<'a>` (returning
            // a `MutexGuardWriter`), which is what `with_writer` requires.
            // The layer owns the writer; no `Box::leak` / `&'static` needed.
            let audit_filter =
                tracing_subscriber::filter::Targets::new().with_target("audit", Level::INFO);
            let audit_layer = fmt::layer()
                .with_writer(std::sync::Mutex::new(file))
                .with_target(false)
                .with_ansi(false)
                .with_filter(audit_filter);
            let _ = tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .with(audit_layer)
                .try_init();
            tracing::info!(
                "fluxion-rest: FLUXION_AUDIT_LOG writing audit events to {}",
                audit_path.as_deref().unwrap_or("?")
            );
        }
        None => {
            let _ = tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .try_init();
        }
    }

    let addr = resolve_addr();

    // Issue #2505 — release-build boot guard: refuse to start when binding
    // all interfaces with authentication disabled (an anonymous network
    // client could reach every `/v1/*` route). `check_boot_guard_from_env`
    // is a no-op in debug builds so `cargo run` keeps working locally.
    if let Err(msg) = check_boot_guard_from_env() {
        eprintln!("{msg}");
        std::process::exit(1);
    }

    // Issue #2505 — resolve the full security configuration from the
    // environment once and hand it to the router builder.
    let security_cfg = RestSecurityConfig::from_env();
    tracing::info!(
        "fluxion-rest security: auth={:?} cors_origins={} rate_limit_rps={} rate_limit_burst={}",
        security_cfg.auth_mode,
        if security_cfg.cors_origins.is_empty() {
            "<dev-defaults>".to_string()
        } else {
            security_cfg.cors_origins.join(",")
        },
        security_cfg.rate_limit_rps,
        security_cfg.rate_limit_burst,
    );

    let listener = TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    tracing_or_log_info(&format!(
        "fluxion-rest listening on {bound} (FLUXION_REST_BIND={} FLUXION_REST_PORT={})",
        resolve_bind(),
        bound.port()
    ));

    let app = router_with_security(AppState::default(), security_cfg);

    // Issue #2514 — startup self-check. Run the same readiness probes the
    // `/v1/readyz` endpoint exposes *after* the router is constructed but
    // *before* `axum::serve` accepts traffic. A misconfigured pod (missing
    // ONNX model / unreadable weather file / broken state store) exits
    // non-zero here instead of serving 503 to every request. The listener
    // is already bound so the port is reserved, but no request is handled
    // until the probe passes.
    let report = run_readiness_probes();
    if !report.is_ready() {
        let c = &report.checks;
        eprintln!("fluxion-rest: readiness self-check failed — not accepting traffic:");
        eprintln!("  onnx:     [{}] {}", c.onnx.status, c.onnx.detail);
        eprintln!("  weather:  [{}] {}", c.weather.status, c.weather.detail);
        eprintln!("  appstate: [{}] {}", c.appstate.status, c.appstate.detail);
        std::process::exit(1);
    }
    tracing_or_log_info(&format!(
        "fluxion-rest: readiness self-check passed (onnx: {}, weather: {}, appstate: {})",
        report.checks.onnx.detail, report.checks.weather.detail, report.checks.appstate.detail,
    ));

    // Issue #2505 — `into_make_service_with_connect_info` injects the
    // accepted socket's peer address into each request's extensions as
    // `ConnectInfo<SocketAddr>`, which the per-IP rate limiter reads as a
    // fallback when no `X-Forwarded-For` / `X-Real-IP` header is present.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
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
