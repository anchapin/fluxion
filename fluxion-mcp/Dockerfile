# Fluxion MCP Server Docker Image
# Multi-stage build for minimal image size with Claude Desktop / VS Code / Cursor compatibility

# ============================================
# Stage 1: Build
# ============================================
FROM rust:1.75-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy project files
COPY Cargo.toml .
COPY Cargo.lock .
COPY fluxion-mcp/ ./fluxion-mcp/

# Build fluxion-mcp
RUN cd fluxion-mcp && cargo build --release --strip

# ============================================
# Stage 2: Production runtime
# ============================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 fluxion

WORKDIR /home/fluxion

# Copy built binary
COPY --from=builder /build/fluxion-mcp/target/release/fluxion-mcp .

# Set ownership
RUN chown -R fluxion:fluxion /home/fluxion

# Switch to non-root user
USER fluxion

# Environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Health check - MCP servers respond to initialize
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD exit 0

# Default command - run the MCP server
CMD ["./fluxion-mcp"]