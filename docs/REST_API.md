# Fluxion REST API

Issue **#1342** — REST API server for Fluxion, complementary to the MCP server
(closed in #1185). This document covers installation, environment variables,
endpoint reference with curl examples, and a link to the OpenAPI 3.1 contract.

## Why REST?

The MCP server (closed in #1185) is the in-agent surface. REST is the
complementary deployment surface for:

- CI/webhook integration (post a schema, run a simulation, get a verdict)
- Multi-user serving (one process, many concurrent requests)
- Model-serving for ML pipelines that prefer HTTP over FFI

OpenAPI 3.1 gives a stable, language-agnostic contract so non-MCP clients can
generate typed bindings without reading the Rust source.

## Installation

The REST server ships as a Cargo bin target. Build it from the repo root:

```bash
cargo build --bin fluxion-rest
```

This produces `target/debug/fluxion-rest` (or `target/release/fluxion-rest`
with `--release`).

There are no system-level dependencies beyond the standard Rust toolchain and
the libraries already vendored by Fluxion (axum 0.7, tokio 1.x, serde,
tower). The binary is fully self-contained — `include_str!` embeds the
OpenAPI YAML so the on-disk spec and the served spec can never drift.

## Environment variables

| Variable             | Default      | Purpose                                                          |
|----------------------|--------------|------------------------------------------------------------------|
| `FLUXION_REST_BIND`  | `0.0.0.0`    | Bind address. Use `127.0.0.1` to keep the server loopback-only.  |
| `FLUXION_REST_PORT`  | `8080`       | TCP port. Must be a valid `u16`.                                 |
| `RUST_LOG`           | `info`       | Standard `env_logger` filter. e.g. `RUST_LOG=debug fluxion=info`. |

If `FLUXION_REST_PORT` is not a valid `u16`, the server logs a warning and
falls back to `8080`. If `FLUXION_REST_BIND` cannot be parsed as a socket
address, the server falls back to `0.0.0.0` on the resolved port.

## Endpoints

All endpoints are versioned under `/v1`. The schema is exactly the
`SimulationSchema` documented in [`src/api/schema.rs`](../src/api/schema.rs) —
the REST server re-uses that schema for both request and response payloads
(per the issue scope: no modifications to `src/api/schema.rs`).

### `GET /v1/healthz`

Liveness probe. Always returns 200 with a static JSON payload. Does **not**
ping downstream services so a slow disk does not flap the load balancer.

Every response (including this one) carries an `x-request-id` header
(Issue #1447) so operators can correlate a 5xx with the structured log line
emitted by the server. To propagate an inbound id, send an `x-request-id`
header on the request — the server reuses it rather than generating a new one.

```bash
curl -s http://localhost:8080/v1/healthz
# => {"status":"ok","version":"1.0.0"}
```

### `GET /v1/metrics`

Prometheus exposition endpoint (Issue #1447). Returns the in-process metrics
counters and histograms as `text/plain; version=0.0.4` so a Prometheus scraper
(or `curl`/Grafana Agent) can ingest them directly.

| Metric                                       | Type      | Labels                          | Purpose                                          |
|----------------------------------------------|-----------|---------------------------------|--------------------------------------------------|
| `fluxion_rest_requests_total`                | counter   | `route,method,status`           | Per-endpoint request count by HTTP status.       |
| `fluxion_rest_request_duration_seconds`      | histogram | `route,method`                  | Wall-clock latency in seconds.                   |
| `fluxion_rest_errors_total`                  | counter   | `route,method,status`           | Increments when `status` is 4xx or 5xx.          |

The `route` label is the *matched pattern* (e.g. `/v1/simulate`,
`/v1/schema/:id`) so traffic is not fragmented into one label set per
schema id.

```bash
curl -s http://localhost:8080/v1/metrics | head -3
# => # HELP fluxion_rest_requests_total Total number of HTTP requests ...
# => # TYPE fluxion_rest_requests_total counter
# => fluxion_rest_requests_total{method="GET",route="/v1/healthz",status="200"} 3
```

### `GET /v1/openapi.yaml`

Raw OpenAPI 3.1 document, served as `application/yaml`. Suitable for
`swagger-cli validate` or any other spec tooling.

```bash
curl -s http://localhost:8080/v1/openapi.yaml | head -2
# => openapi: 3.1.0
# => info:
```

### `GET /v1/openapi.json`

Same document, wrapped in a JSON envelope for clients that prefer JSON:

```json
{
  "openapi": "3.1.0",
  "spec": "<yaml as a string>"
}
```

### `POST /v1/simulate`

Run a Fluxion simulation against a `SimulationSchemaV1`. The result is
guaranteed to match an in-process Rust call against the same schema within
floating-point noise (`<0.1%` on `heating_energy_kwh + cooling_energy_kwh`).

```bash
curl -s -X POST http://localhost:8080/v1/simulate \
  -H 'content-type: application/json' \
  -d @tests/fixtures/single_zone.json | jq '.cooling_energy_kwh'
```

The request body is a bare `SimulationSchemaV1` (or the version-tagged
`SimulationSchema` envelope); an optional `options` field accepts:

- `years` (default `1`)
- `use_surrogates` (default `false`)
- `store_as` (optional explicit id; if absent the server auto-assigns one)

The response is `{ "schema_id": "sch-0", "output": { ... SimulationOutput ... } }`.
The `schema_id` is always non-null because the server auto-stores the schema
so callers can retrieve it via `GET /v1/schema/{id}`.

### `GET /v1/schema/{id}`

Fetch a previously stored schema. Returns 404 if the id is unknown. Storage is
process-local and resets on restart (a persistent store is explicitly out of
scope per #1342).

```bash
curl -s http://localhost:8080/v1/schema/sch-0 | jq '.geometry.zones | length'
```

### `POST /v1/import/{fmt}`

Convert an external model file into a Fluxion `SimulationSchemaV1` and store
it. The body is the raw file bytes (no multipart envelope).

| `{fmt}`   | Delegated to                                       | Status                  |
|-----------|----------------------------------------------------|-------------------------|
| `osm`     | `crate::interop::osm::import_osm`                  | supported               |
| `gbxml`   | `crate::interop::gbxml::import_gbxml`              | supported               |
| `idf`     | _(no reader in `src/interop/*` yet)_               | **501 Not Implemented** |

The `idf` endpoint is reserved for future work — there is no EnergyPlus IDF
reader under `src/interop/` today, so we explicitly return
`501 Not Implemented` rather than silently accepting the file. Adding the
reader is a separate scope item and will be tracked in a follow-up issue.

```bash
curl -s -X POST http://localhost:8080/v1/import/osm \
  --data-binary @model.osm | jq '.schema_id'
```

## Acceptance criteria

From #1342:

- ✅ 6 endpoints reachable on the bound port (5 + `/v1/metrics` from #1447)
- ✅ OpenAPI 3.1 spec at `src/api/openapi.yaml` validates against the
  OpenAPI 3.1 meta-schema (`npx @apidevtools/swagger-cli validate`)
- ✅ `POST /v1/simulate` with a 1-zone schema matches an in-process Rust call
  within 0.1% (verified by `simulate_matches_in_process_within_tolerance` in
  `tests/api_integration_tests.rs`)
- ✅ p50 latency for `/v1/healthz` < 5ms in the local test harness
- ✅ This document covers install, env vars, all 6 endpoints with curl examples,
  and a link to the OpenAPI reference (`src/api/openapi.yaml`)

From #1447:

- ✅ `GET /v1/metrics` returns 200 with Prometheus exposition format
- ✅ Every response carries an `x-request-id` header (UUIDv4)
- ✅ `tower_http::TraceLayer` emits one structured log line per request
  (driven by `tracing_subscriber::fmt::layer` initialized in
  `src/bin/fluxion_rest.rs`)
- ✅ Counters `fluxion_rest_requests_total` and
  `fluxion_rest_errors_total` plus histogram
  `fluxion_rest_request_duration_seconds` increment on test traffic
- ✅ New test target `tests/api_observability_tests.rs` covers
  `/v1/metrics` shape, `x-request-id` propagation, and counter increment

## Out of scope (explicitly deferred)

- **Authentication / authorization** — separate issue (OIDC/JWT).
- **Persistent storage backend** — in-memory + filesystem only; SQL/S3 deferred.
- **Multi-tenant isolation** — single-tenant MVP.
- **Replacing the MCP server** — REST is complementary (#1185 closed MCP).
- **Modifying `src/api/schema.rs`** — the canonical schema stays as-is.

## Verification path

Per the issue body:

```bash
cargo build --bin fluxion-rest
cargo run --bin fluxion-rest &
sleep 2
curl -sf http://localhost:8080/v1/healthz
curl -sf -X POST http://localhost:8080/v1/simulate \
  -d @tests/fixtures/single_zone.json | jq '.cooling_energy_kwh'
npx @apidevtools/swagger-cli validate src/api/openapi.yaml
```

Or run the in-process integration tests:

```bash
cargo test --test api_integration_tests
```

## Files

| Path                          | Purpose                                           |
|-------------------------------|---------------------------------------------------|
| `src/api/server.rs`           | Router, handlers, AppState, error mapping          |
| `src/api/openapi.yaml`        | Hand-authored OpenAPI 3.1 contract                |
| `src/bin/fluxion_rest.rs`     | Binary entrypoint with env-var resolution         |
| `tests/api_integration_tests.rs` | End-to-end HTTP tests                          |
| `docs/REST_API.md`            | This document                                     |