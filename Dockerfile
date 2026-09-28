# Fluxion Docker Image
#
# Multi-stage build that produces a self-contained `fluxion-rest`
# binary (the only Fluxion deployment surface tracked under issue
# #1411). The runtime image exposes port 8080 and healthchecks
# `/v1/healthz`, matching the defaults in `src/bin/fluxion_rest.rs`
# and `docs/REST_API.md`.
#
# Build:  docker build -t fluxion-rest .
# Run:    docker run --rm -p 8080:8080 \
#            -e FLUXION_REST_AUTH=token \
#            -e FLUXION_REST_AUTH_TOKEN=change-me \
#            fluxion-rest
# Smoke:  curl -s http://localhost:8080/v1/healthz
#
# Notes:
#   * `FLUXION_REST_AUTH` is REQUIRED. The image defaults to
#     `FLUXION_REST_BIND=0.0.0.0` (below) with auth off, and a release
#     build REFUSES to boot that combination — it exits non-zero with
#     "refusing to boot" rather than serving an unauthenticated endpoint
#     on every interface (Issue #2526 / #4139 follow-up). The run
#     command above is therefore not optional boilerplate: without an
#     auth flag the container exits immediately.
#   * `FLUXION_REST_AUTH=token` is the cheapest option that satisfies
#     the guard. `/v1/healthz` and `/v1/readyz` are `RouteTier::Public`
#     (src/api/server/router.rs), so the smoke check above needs no token.
#     `FLUXION_REST_AUTH=tls` is the production choice;
#     `FLUXION_REST_ALLOW_INSECURE=1` is an explicit escape hatch for
#     throwaway local containers and should not be used in CI.
#   * Bind address / port are overridable at runtime:
#       docker run -e FLUXION_REST_BIND=127.0.0.1 -e FLUXION_REST_PORT=8080 \
#              -p 8080:8080 -e FLUXION_REST_AUTH=token \
#              -e FLUXION_REST_AUTH_TOKEN=change-me fluxion-rest
#     Note that binding 127.0.0.1 *inside* the container is incompatible
#     with `-p` port publishing, which forwards to the container's
#     external interface.
#   * The old `fluxion-api` image (port 8000, `python -m api.main`,
#     healthcheck on `/health`) no longer exists. Any reference to it
#     in the wild is a stale doc that should be redirected to the
#     Rust binary.

# ============================================
# Stage 1: Build the `fluxion-rest` binary
# ============================================
# Pinned base image — fail-closed supply-chain control (Issue #3580, Goal #5).
#   * Tag:    rust:1.98.0-bookworm
#   * Digest: sha256:82150a52ec202c1b14d7817e14516c392bb7f5cfebd88f1ed531cb37ebd39922
#   * Pinned: 2026-09-28
#   * Refresh: re-run `scripts/pin_docker_base_images.sh` (quarterly cadence).
#   * NOTE: the tag must satisfy the workspace MSRV (`rust-version` in
#     Cargo.toml, 1.98.0 as of #3321). It lagged for 5 weeks after the
#     2026-09-03 MSRV bump, so cargo could not compile the dependency
#     graph at all (Issue #4138). scripts/check_docker_base_image_msrv.py
#     now gates the tag against `rust-version`.
FROM rust:1.98.0-bookworm@sha256:82150a52ec202c1b14d7817e14516c392bb7f5cfebd88f1ed531cb37ebd39922 AS builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the manifests first so Docker can cache the dependency
# layer when only the source changes.
COPY Cargo.toml Cargo.lock ./
COPY fluxion-core/ ./fluxion-core/
COPY fluxion-city/ ./fluxion-city/
COPY fluxion-fluid/ ./fluxion-fluid/
COPY fluxion-grid/ ./fluxion-grid/
COPY fluxion-behavior/ ./fluxion-behavior/
COPY fluxion-mcp/ ./fluxion-mcp/
COPY fluxion-wasm/ ./fluxion-wasm/
COPY crates/fluxion-toon/ ./crates/fluxion-toon/
COPY crates/fluxion-twin/ ./crates/fluxion-twin/
COPY crates/fluxion-evaluator/ ./crates/fluxion-evaluator/
COPY fluxion-cfd/ ./fluxion-cfd/
COPY fluxion-tauri/src-tauri/ ./fluxion-tauri/src-tauri/
COPY src/ ./src/
# `Cargo.toml` references a few bench harnesses; copy them so the
# manifest parses even when we are only building the `fluxion-rest`
# binary. The runtime image never executes these.
COPY benches/ ./benches/
# Issue #4135 — the root manifest declares one `[[example]]` target, and
# cargo fails to *parse* a manifest whose declared target file is missing
# from the context. `.dockerignore` excludes the rest of `examples/`
# (it is a separate, non-member package), so only this file is carried.
COPY examples/grid_coupling_demo.rs ./examples/grid_coupling_demo.rs

# Build the REST binary. We deliberately skip the python-bindings
# and napi features so we do not pull in PyO3 / NAPI headers and
# linker deps — the runtime stage is a plain Debian image.
RUN cargo build --release --bin fluxion-rest --no-default-features

# ============================================
# Stage 2: Production runtime
# ============================================
# Pinned base image — fail-closed supply-chain control (Issue #3580, Goal #5).
#   * Tag:    debian:bookworm-slim
#   * Digest: sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171
#   * Pinned: 2026-09-09
#   * Refresh: re-run `scripts/pin_docker_base_images.sh` (quarterly cadence).
FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 fluxion

WORKDIR /home/fluxion

# Copy the built binary from the builder stage
COPY --from=builder /build/target/release/fluxion-rest /usr/local/bin/fluxion-rest

# Create data directory
RUN mkdir -p /home/fluxion/data && chown -R fluxion:fluxion /home/fluxion

# Switch to non-root user
USER fluxion

# Environment variables (override with `-e KEY=VALUE` at run time)
ENV FLUXION_REST_BIND=0.0.0.0 \
    FLUXION_REST_PORT=8080 \
    RUST_LOG=info

# Expose REST port — must match FLUXION_REST_PORT above
EXPOSE 8080

# Health check — must hit `/v1/healthz` (returns 200 + JSON),
# not the legacy `/health` (which the binary does not serve).
# Uses shell form so curl can resolve the localhost loopback.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/v1/healthz || exit 1

# Default command — runs the REST server
CMD ["/usr/local/bin/fluxion-rest"]
