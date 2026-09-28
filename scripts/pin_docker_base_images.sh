#!/usr/bin/env bash
#
# Resolve and pin the Docker base images that `Dockerfile` pulls by
# mutable tag (Issue #3580, Goal #5). Refreshes the `@sha256:DIGEST`
# pin on each `FROM` line and rewrites the matching
# `RUST_BUILDER_DIGEST` / `DEBIAN_RUNTIME_DIGEST` env entries in
# `.github/workflows/docker.yml` so the CI digest assertion stays in
# sync with the Dockerfile. Re-run on a quarterly cadence or
# whenever a CVE in the base image lands upstream.
#
# Prerequisites:
#   * `docker` on $PATH (script uses `docker pull` + `docker inspect`).
#   * Run from the repository root.
#
# Usage::
#
#     ./scripts/pin_docker_base_images.sh                  # resolve and rewrite
#     ./scripts/pin_docker_base_images.sh --check          # verify current pin matches upstream (CI mode)
#
# `--check` exits non-zero if any pinned digest has drifted from
# what the registry currently serves, **or** if the `Dockerfile`
# `FROM` pins have diverged from the `docker.yml` env pins, so this
# script can be wired into a weekly cron job or a "Dependency
# Review" workflow to surface stale pins early. It does NOT rewrite
# the files in `--check` mode.
#
# Why pull-by-tag-then-inspect, instead of `docker pull --quiet
# <image>@sha256:<old>` then compare? `docker pull <image>@sha256`
# fetches the OLD digest regardless of what the floating tag now
# points at, so it cannot detect upstream rotation by itself. The
# pattern below pulls the CURRENT floating tag, asks the local
# daemon for `RepoDigests[0]` (which is the canonical
# registry/repository@sha256:DIGEST form), and compares to the pin.
#
# Side effects: the script invokes `docker pull` for each base
# image, so the local daemon cache is updated as a side effect.
#
# Structure: everything below `normalize_digest` is a pure,
# network-free function. The docker-touching `main` is guarded by
# `BASH_SOURCE` so `scripts/ci/test_pin_docker_base_images.py` can
# source this file and exercise `apply_pins` against `tmp_path`
# fixtures without a daemon, a network, or a `docker` stub on
# `$PATH`. Sourcing must never trigger a build or a registry call.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/Dockerfile"
WORKFLOW="${REPO_ROOT}/.github/workflows/docker.yml"
CARGO_TOML="${REPO_ROOT}/Cargo.toml"

# Issue #4138: derive the builder tag from the workspace MSRV
# (`rust-version` in Cargo.toml) instead of hardcoding a version. The
# hardcoded tag lagged 5 weeks behind the 1.89 -> 1.98.0 bump (#3321),
# leaving the image unable to compile the dependency graph at all.
# Deriving it makes a future MSRV bump automatically self-consistent on
# the next pin refresh.
MSRV="$(grep -oE '^rust-version = "[0-9]+\.[0-9]+\.[0-9]+"' "${CARGO_TOML}" \
        | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true)"
if [[ -z "${MSRV}" ]]; then
    echo "::error::could not read rust-version from ${CARGO_TOML}" >&2
    exit 2
fi
RUST_BUILDER_TAG="rust:${MSRV}-bookworm"
DEBIAN_TAG="debian:bookworm-slim"

# --------------------------------------------------------------------------
# Pure helpers (no docker, no network, no repo access)
# --------------------------------------------------------------------------

# Canonicalise a digest to the `sha256:<64 lowercase hex>` form.
#
# `docker inspect` yields `rust@sha256:<hex>`, `RepoDigests[0]` yields
# the same, and the pinned files spell it either as `<hex>` or as
# `sha256:<hex>` depending on which code path wrote it. Accepting all
# three and emitting one canonical form is what stops the pre-2026-09
# corruption in which every write site prepended `sha256:` to a value
# that already carried it, writing `FROM rust:1.98.0-bookworm@sha256:sha256:…`
# -- an unparseable image reference that `docker build` rejects.
# Anything that is not exactly 64 hex characters is rejected here rather
# than being written into a Dockerfile.
normalize_digest() {
    local raw="${1:-}" hex
    hex="${raw##*@}"          # strip any `repo@` prefix
    hex="${hex#sha256:}"      # strip the algorithm prefix if present
    if [[ ! "${hex}" =~ ^[a-f0-9]{64}$ ]]; then
        echo "::error::malformed digest '${raw}' (expected sha256: followed by 64 lowercase hex chars)" >&2
        return 1
    fi
    printf 'sha256:%s' "${hex}"
}

# Rewrite the Dockerfile pin comment blocks and the `FROM` lines.
#
# Idempotent by construction: the block that documents a pin is emitted
# as exactly one Tag / one Digest / one Pinned line, and the Digest and
# Pinned lines that immediately follow the Tag line are swallowed before
# the fresh pair is written. The previous implementation *appended* after
# the Tag line, so every run accumulated another Digest/Pinned pair and
# the file grew without bound.
#
# The `FROM` line is rebuilt from `tag` + `digest` + its `AS <stage>`
# suffix rather than substituted in place, which makes the rewrite
# self-healing: a previously corrupted or unpinned `FROM` is repaired
# rather than skipped.
#
# Usage: apply_pins <dockerfile> <tag> <digest> <date>
apply_pins() {
    local target="$1" tag="$2" digest="$3" date="$4"
    local tmp
    tmp="$(mktemp)"
    awk -v tag="${tag}" -v digest="${digest}" -v date="${date}" '
        BEGIN { skip = 0; prefix = "#   * Tag:" }
        {
            line = $0
            if (skip == 1) {
                # Swallow the stale Digest/Pinned pair emitted by a
                # previous run of this script.
                if (line ~ /^#[[:space:]]+\*[[:space:]]+(Digest|Pinned):/) { next }
                skip = 0
            }
            if (substr(line, 1, length(prefix)) == prefix) {
                rest = substr(line, length(prefix) + 1)
                gsub(/^[ \t]+/, "", rest)
                gsub(/[ \t]+$/, "", rest)
                if (rest == tag) {
                    print line
                    print "#   * Digest: " digest
                    print "#   * Pinned: " date
                    skip = 1
                    next
                }
            }
            if (index(line, "FROM " tag "@") == 1) {
                stage = ""
                if (match(line, / AS /)) { stage = substr(line, RSTART) }
                print "FROM " tag "@" digest stage
                next
            }
            if (index(line, "FROM " tag " ") == 1) {
                stage = ""
                if (match(line, / AS /)) { stage = substr(line, RSTART) }
                print "FROM " tag "@" digest stage
                next
            }
            print line
        }
    ' "${target}" > "${tmp}"
    cat "${tmp}" > "${target}"
    rm -f "${tmp}"
}

# Rewrite the `docker.yml` env pins that the CI assertion consumes.
#
# Usage: apply_workflow_pins <workflow> <rust_digest> <debian_digest>
apply_workflow_pins() {
    local target="$1" rust_digest="$2" debian_digest="$3"
    local tmp
    tmp="$(mktemp)"
    awk -v rust_digest="${rust_digest}" -v debian_digest="${debian_digest}" '
        /^  RUST_BUILDER_DIGEST:/   { print "  RUST_BUILDER_DIGEST: " rust_digest;   next }
        /^  DEBIAN_RUNTIME_DIGEST:/ { print "  DEBIAN_RUNTIME_DIGEST: " debian_digest; next }
        { print }
    ' "${target}" > "${tmp}"
    cat "${tmp}" > "${target}"
    rm -f "${tmp}"
}

# Read the digest currently pinned on a `FROM <tag>@sha256:...` line.
dockerfile_digest() {
    local dockerfile="$1" tag="$2" ref
    ref="$(grep -m1 -E "^FROM ${tag}@" "${dockerfile}" || true)"
    if [[ -z "${ref}" ]]; then
        echo ""
        return 0
    fi
    # Reduce the whole `FROM` line to the bare image reference before
    # splitting on `@`, otherwise the ` AS <stage>` suffix rides along
    # into the digest and the comparison can never succeed.
    ref="${ref#FROM }"
    ref="${ref%% *}"
    printf '%s' "${ref#*@}"
}

# --------------------------------------------------------------------------
# docker-touching driver
# --------------------------------------------------------------------------

main() {
    local check_only=0
    if [[ "${1:-}" == "--check" ]]; then
        check_only=1
    fi

    local base_images=("${RUST_BUILDER_TAG}" "${DEBIAN_TAG}")
    declare -A NEW_DIGESTS=()

    for image in "${base_images[@]}"; do
        echo "Resolving ${image} ..." >&2
        docker pull --quiet "${image}" >/dev/null
        local digest
        digest="$(docker inspect --format='{{index .RepoDigests 0}}' "${image}" || true)"
        if [[ -z "${digest}" ]]; then
            echo "::error::Could not resolve RepoDigests for ${image}" >&2
            exit 1
        fi
        NEW_DIGESTS["${image}"]="$(normalize_digest "${digest}")"
        echo "  -> ${digest}" >&2
    done

    if [[ "${check_only}" -eq 1 ]]; then
        local drift=0 image pinned_env pinned_from
        for image in "${base_images[@]}"; do
            if [[ "${image}" == "${RUST_BUILDER_TAG}" ]]; then
                pinned_env="$(grep -m1 -oE 'RUST_BUILDER_DIGEST: sha256:[a-f0-9]+' "${WORKFLOW}" \
                             | sed -E 's/^RUST_BUILDER_DIGEST: //' || true)"
            else
                pinned_env="$(grep -m1 -oE 'DEBIAN_RUNTIME_DIGEST: sha256:[a-f0-9]+' "${WORKFLOW}" \
                             | sed -E 's/^DEBIAN_RUNTIME_DIGEST: //' || true)"
            fi
            if [[ "${pinned_env}" != "${NEW_DIGESTS[${image}]}" ]]; then
                echo "::error::Drift detected for ${image}: pinned=${pinned_env:-<none>}  upstream=${NEW_DIGESTS[${image}]}" >&2
                drift=1
                continue
            fi
            # Upstream parity is necessary but not sufficient: the CI
            # assertion reads the `docker.yml` env pins, so a Dockerfile
            # that drifted away from them would keep CI green while the
            # image build pulls an unaudited base layer. Check both.
            pinned_from="$(dockerfile_digest "${DOCKERFILE}" "${image}")"
            if [[ "$(normalize_digest "${pinned_from}" 2>/dev/null || true)" != "${NEW_DIGESTS[${image}]}" ]]; then
                echo "::error::Dockerfile pin for ${image} disagrees with docker.yml: FROM=${pinned_from:-<unpinned>}  expected=${NEW_DIGESTS[${image}]}" >&2
                drift=1
                continue
            fi
            echo "OK: ${image} pin matches upstream and the Dockerfile (${NEW_DIGESTS[${image}]})" >&2
        done
        exit "${drift}"
    fi

    local today
    today="$(date -u +%Y-%m-%d)"
    apply_pins "${DOCKERFILE}" "${RUST_BUILDER_TAG}" "${NEW_DIGESTS[${RUST_BUILDER_TAG}]}" "${today}"
    apply_pins "${DOCKERFILE}" "${DEBIAN_TAG}" "${NEW_DIGESTS[${DEBIAN_TAG}]}" "${today}"
    apply_workflow_pins "${WORKFLOW}" \
        "${NEW_DIGESTS[${RUST_BUILDER_TAG}]}" "${NEW_DIGESTS[${DEBIAN_TAG}]}"

    echo >&2
    echo "Updated digests:" >&2
    for image in "${base_images[@]}"; do
        echo "  ${image} -> ${NEW_DIGESTS[${image}]}" >&2
    done
    echo >&2
    echo "Next steps:" >&2
    echo "  1. Inspect the diff: git diff Dockerfile .github/workflows/docker.yml" >&2
    echo "  2. Re-verify: ./scripts/pin_docker_base_images.sh --check" >&2
    echo "  3. Smoke-test the new pins: docker build -f Dockerfile ." >&2
    echo "  4. Commit with: git commit -am 'chore(security): refresh Dockerfile + docker.yml base image digests'" >&2
}

# Only drive the rewrite when executed directly. Sourcing this file
# (as the pytest suite does) must not touch the daemon or the registry.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
