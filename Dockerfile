# Fluxion Docker Image
# Multi-stage build for minimal image size

# ============================================
# Stage 1: Build Python bindings
# ============================================
FROM python:3.11-slim AS builder

# Install system and Rust build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy project files
COPY Cargo.toml .
COPY Cargo.lock .
COPY pyproject.toml .
COPY requirements-dev.txt .
COPY src/ ./src/
COPY benches/ ./benches/
COPY README.md .
COPY api/ ./api/

# Install Python build dependencies
RUN pip install --no-cache-dir maturin pytest

# Build Python bindings
RUN maturin build --release --strip

# ============================================
# Stage 2: Production runtime
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
COPY --from=builder /build/target/wheels/*.whl .

# Install the wheel
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy API server files
COPY --from=builder /build/api ./api

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
