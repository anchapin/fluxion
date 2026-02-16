# Fluxion Docker Image
# Multi-stage build for minimal image size

# ============================================
# Stage 1: Build Rust dependencies
# ============================================
FROM rust:1.75-slim AS builder-rust

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy Cargo files
COPY Cargo.toml ./
COPY Cargo.lock ./

# Create dummy source to build dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Build dependencies only (cache layer)
RUN cargo build --release
RUN rm -rf src target/release/deps/main*

# ============================================
# Stage 2: Build Python bindings
# ============================================
FROM python:3.11-slim AS builder-python

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy Python project files
COPY pyproject.toml .
COPY requirements-dev.txt .

# Install Python build dependencies
RUN pip install --no-cache-dir maturin pytest

# Copy Rust source
COPY --from=builder-rust /build ./

# Build Python bindings
RUN maturin build --release --strip

# ============================================
# Stage 3: Production runtime
# ============================================
FROM python:3.11-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 fluxion

WORKDIR /home/fluxion

# Copy built wheel from builder
COPY --from=builder-python /build/target/wheels/*.whl .

# Install the wheel
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy API server files
COPY --from=builder-python /build/api ./api

# Create data and model directories
RUN mkdir -p /home/fluxion/data /home/fluxion/models

# Set ownership
RUN chown -R fluxion:fluxion /home/fluxion

# Switch to non-root user
USER fluxion

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV RUST_LOG=info

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["python", "-m", "api.main"]
