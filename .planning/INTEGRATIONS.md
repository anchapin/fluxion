# External Integrations

**Analysis Date:** 2026-03-08

## APIs & External Services

**Machine Learning Inference:**
- ONNX Runtime 2.0.0-rc.10 - Neural surrogate model inference
  - SDK/Client: `ort` crate (Rust)
  - Auth: None (local inference)
  - Supported backends: CPU, CUDA, CoreML, DirectML, OpenVINO
  - Session pool pattern for concurrent inference (`src/ai/surrogate.rs`)

**REST API (FastAPI):**
- Self-hosted REST API server (`api/main.py`)
  - No external API integrations detected
  - Exposes endpoints for population evaluation, LLM queries, distributed inference
  - CORS middleware enabled

**Local LLM Support:**
- Module: `api/llm.py`
  - Purpose: Local LLM inference for function calling
  - Implementation: Local inference pool (no external API)
  - No cloud LLM providers integrated

## Data Storage

**Databases:**
- None detected (no database files, SQL, or database drivers found)

**File Storage:**
- Local filesystem only
  - EPW weather files: External EnergyPlus Weather format (`src/weather/epw.rs`)
  - ONNX models: Stored in `models/` directory (gitignored)
  - Python wheels: Built to `target/wheels/`

**Caching:**
- GitHub Actions cache (`.github/workflows/ci.yml`)
  - Cargo registry cache
  - Cargo git cache
  - Target directory cache
  - Docker layer cache (GHA cache type)

## Authentication & Identity

**Auth Provider:**
- Custom (None for local execution)
  - No external authentication providers detected
  - API server has no auth middleware (development mode)
  - Recommendations: Add authentication for production deployment

**Security:**
- GitHub Container Registry (ghcr.io)
  - Auth: GitHub token (`secrets.GITHUB_TOKEN`)
  - Used for Docker image push

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, Rollbar, or similar integrations)

**Logs:**
- Python: Standard logging module (`api/main.py`)
  - Configured to INFO level
- Rust: env_logger via `RUST_LOG` environment variable
  - Set to "info" in Dockerfile
- Monitoring module: `api/monitoring.py` for BAS integration
  - Real-time monitoring capabilities
  - No external log aggregation

## CI/CD & Deployment

**Hosting:**
- GitHub - Source code and issue tracking
- GitHub Container Registry (ghcr.io) - Docker images
- PyPI - Python package distribution

**CI Pipeline:**
- GitHub Actions
  - `ci.yml` - Main CI (Rust tests, Python examples)
  - `rust-tests.yml` - Rust-specific tests
  - `python-bindings.yml` - Python bindings validation
  - `ashrae_140_validation.yml` - ASHRAE 140 validation
  - `docker.yml` - Docker build, test, multi-platform build, security scan (Trivy)
  - `pypi-release.yml` - PyPI/Test PyPI publishing
  - `docs.yml` - Documentation build
  - `code-coverage.yml` - Code coverage reports
  - `security.yml` - Security audits
  - `data_gen_test.yml` - Data generation testing

**Security Scanning:**
- Trivy - Container vulnerability scanning (`.github/workflows/docker.yml`)
- cargo audit - Rust dependency security auditing (pre-commit hook)
- GitHub Security tab - SARIF upload for Trivy results

**Deployment:**
- Docker multi-stage build (`Dockerfile`)
  - Stage 1: Build Rust and Python bindings
  - Stage 2: Minimal runtime (python:3.11-slim)
- Multi-platform support: linux/amd64, linux/arm64
- Health check: HTTP endpoint `http://localhost:8000/health`

## Environment Configuration

**Required env vars:**
- `RUST_LOG` - Rust logging level (default: "info")
- `PYTHONUNBUFFERED` - Python output buffering (default: "1")
- No other environment variables detected in code

**Secrets location:**
- GitHub Actions secrets: `secrets.GITHUB_TOKEN` (automatically provided)
- No `.env` files present (gitignored in `.gitignore`)
- No external secret managers detected (no Vault, AWS Secrets Manager, etc.)

## Webhooks & Callbacks

**Incoming:**
- GitHub Actions triggers (repository webhooks)
  - Pull requests: CI pipeline
  - Push to main/develop: Docker build, PyPI release
  - Tags: PyPI release, Docker release
  - Releases: PyPI release

**Outgoing:**
- None detected (no outgoing webhooks configured)
- No external API calls to third-party services

**FMI 3.0 Co-simulation (Planned):**
- Mentioned in documentation as Phase 3
- Not yet implemented

---

*Integration audit: 2026-03-08*
