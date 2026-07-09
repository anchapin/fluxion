#!/usr/bin/env bash
# Run the Fluxion REST server in a sibling terminal and verify every
# endpoint documented in `docs/REST_API.md`. Issue #1411.
#
# Usage:
#   cargo run --bin fluxion-rest &      # in one terminal
#   bash examples/run_rest.sh           # in another
#
# The script is idempotent: every call has a fixed expected status and
# is independent of the previous one. Exits non-zero on the first
# unexpected status.

set -euo pipefail

BASE_URL="${FLUXION_REST_URL:-http://localhost:8080}"
FIXTURE="${FLUXION_FIXTURE:-tests/fixtures/single_zone.json}"

if [[ ! -f "${FIXTURE}" ]]; then
    echo "fixture not found: ${FIXTURE}" >&2
    echo "Run this script from the repository root." >&2
    exit 2
fi

# Tiny helper: print "[label] -> <status>" and exit 1 on mismatch.
expect() {
    local label="$1" want="$2" got
    shift 2
    got=$(curl -s -o /dev/null -w "%{http_code}" "$@")
    printf "%-40s -> %s (expected %s)\n" "${label}" "${got}" "${want}"
    if [[ "${got}" != "${want}" ]]; then
        echo "FAIL: ${label} returned ${got}, expected ${want}" >&2
        exit 1
    fi
}

echo "Fluxion REST smoke (Issue #1411) against ${BASE_URL}"
echo "fixture: ${FIXTURE}"
echo

# 1. Liveness
expect "GET  /v1/healthz"      200 "${BASE_URL}/v1/healthz"

# 2. OpenAPI YAML
expect "GET  /v1/openapi.yaml" 200 "${BASE_URL}/v1/openapi.yaml"

# 3. OpenAPI JSON envelope
expect "GET  /v1/openapi.json" 200 "${BASE_URL}/v1/openapi.json"

# 4. Simulate
echo
echo "POST /v1/simulate (body from ${FIXTURE})"
SIM_RESP=$(curl -s -X POST \
    -H 'content-type: application/json' \
    -d "@${FIXTURE}" \
    "${BASE_URL}/v1/simulate")
echo "${SIM_RESP}" | python3 -c "
import json, sys
body = json.load(sys.stdin)
out = body.get('output', {})
sid = body.get('schema_id')
assert sid, 'missing schema_id'
for k in ('eui', 'total_energy', 'heating_energy', 'cooling_energy'):
    assert k in out, f'missing {k} in output'
print('  -> ok; schema_id=%s, eui=%.3f' % (sid, out['eui']))
print(sid)  # last line -> captured below
" > /tmp/fluxion_sim.out
cat /tmp/fluxion_sim.out | sed 's/^/  /'
SCHEMA_ID=$(tail -n 1 /tmp/fluxion_sim.out)

# 5. Retrieve the schema we just stored
expect "GET  /v1/schema/${SCHEMA_ID}" 200 "${BASE_URL}/v1/schema/${SCHEMA_ID}"

# 6. Import endpoints
expect "POST /v1/import/osm"   200 \
    -X POST --data-binary "OSM" \
    -H 'content-type: application/octet-stream' \
    "${BASE_URL}/v1/import/osm"
expect "POST /v1/import/gbxml" 200 \
    -X POST --data-binary "<gbxml/>" \
    -H 'content-type: application/octet-stream' \
    "${BASE_URL}/v1/import/gbxml"
expect "POST /v1/import/idf"   501 \
    -X POST --data-binary "Version,9.0;" \
    -H 'content-type: application/octet-stream' \
    "${BASE_URL}/v1/import/idf"
expect "POST /v1/import/banana" 400 \
    -X POST --data-binary "x" \
    -H 'content-type: application/octet-stream' \
    "${BASE_URL}/v1/import/banana"

# 7. Schema not found
expect "GET  /v1/schema/sch-does-not-exist" 404 \
    "${BASE_URL}/v1/schema/sch-does-not-exist"

echo
echo "All REST smoke checks passed."
