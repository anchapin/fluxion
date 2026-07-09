# Fluxion Docker Image
#
# Multi-stage build that produces a self-contained `fluxion-rest`
# binary (the only Fluxion deployment surface tracked under issue
# #1411). The runtime image exposes port 8080 and healthchecks
# `/v1/healthz`, matching the defaults in `src/bin/fluxion_rest.rs`
# and `docs/REST_API.md`.
#
# Build:  docker build -t fluxion-rest .
# Run:    docker run --rm -p 8080:8080 fluxion-rest
# Smoke:  curl -s http://localhost:8080/v1/healthz
#
# Notes:
#   * Bind address / port are overridable at runtime:
#       docker run -e FLUXION_REST_BIND=0.0.0.0 -e FLUXION_REST_PORT=8080 \
#              -p 8080:8080 fluxion-rest
#   * The old `fluxion-api` image (port 8000, `python -m api.main`,
#     healthcheck on `/health`) no longer exists. Any reference to it
#     in the wild is a stale doc that should be redirected to the
#     Rust binary.

# ============================================
# Stage 1: Build the `fluxion-rest` binary
# ============================================
FROM rust:1.87-bookworm AS builder

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
COPY src/ ./src/
# `Cargo.toml` references a few bench harnesses; copy them so the
# manifest parses even when we are only building the `fluxion-rest`
# binary. The runtime image never executes these.
COPY benches/ ./benches/

# Build the REST binary. We deliberately skip the python-bindings
# and napi features so we do not pull in PyO3 / NAPI headers and
# linker deps — the runtime stage is a plain Debian image.
RUN cargo build --release --bin fluxion-rest --no-default-features

# ============================================
# Stage 2: Production runtime
# ============================================
FROM debian:bookworm-slim AS runtime

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
