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
# what the registry currently serves, so this script can be wired
# into a weekly cron job or a "Dependency Review" workflow to
# surface stale pins early. It does NOT rewrite the files in
# `--check` mode.
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/Dockerfile"
WORKFLOW="${REPO_ROOT}/.github/workflows/docker.yml"

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
fi

# Extract the current tag for each `FROM <image>:<tag> ... AS <stage>`
# line. The pin comment block above each `FROM` documents the
# expected digest; the script replaces it in-place after re-resolving.
#
# sed anchors on the unique comments so we only rewrite the pin lines
# (not stray mentions of "rust" or "debian" elsewhere in the file).
BASE_IMAGES=(
    "rust:1.87-bookworm"
    "debian:bookworm-slim"
)

declare -A NEW_DIGESTS
declare -A OLD_DIGESTS

for image in "${BASE_IMAGES[@]}"; do
    echo "Resolving ${image} ..." >&2
    docker pull --quiet "${image}" >/dev/null
    digest="$(docker inspect --format='{{index .RepoDigests 0}}' "${image}" || true)"
    if [[ -z "${digest}" ]]; then
        echo "::error::Could not resolve RepoDigests for ${image}" >&2
        exit 1
    fi
    NEW_DIGESTS["${image}"]="${digest##*@}"
    echo "  -> ${digest}" >&2
done

# Capture the current pinned digests so `--check` mode can diff.
if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    OLD_DIGESTS["rust:1.87-bookworm"]="$(grep -oE 'RUST_BUILDER_DIGEST: sha256:[a-f0-9]+' "${WORKFLOW}" | head -1 | sed -E 's/^RUST_BUILDER_DIGEST: //')"
    OLD_DIGESTS["debian:bookworm-slim"]="$(grep -oE 'DEBIAN_RUNTIME_DIGEST: sha256:[a-f0-9]+' "${WORKFLOW}" | head -1 | sed -E 's/^DEBIAN_RUNTIME_DIGEST: //')"
    drift=0
    for image in "${BASE_IMAGES[@]}"; do
        if [[ "${OLD_DIGESTS[${image}]}" != "${NEW_DIGESTS[${image}]}" ]]; then
            echo "::error::Drift detected for ${image}: pinned=${OLD_DIGESTS[${image}]}  upstream=${NEW_DIGESTS[${image}]}" >&2
            drift=1
        else
            echo "OK: ${image} pin matches upstream (${NEW_DIGESTS[${image}]})" >&2
        fi
    done
    exit "${drift}"
fi

# Rewrite the Dockerfile pin comments + `FROM` lines.
for image in "${BASE_IMAGES[@]}"; do
    new="${NEW_DIGESTS[${image}]}"
    case "${image}" in
        rust:1.87-bookworm)
            sed -i \
                -e "s|^#   \* Tag:    rust:1.87-bookworm$|#   * Tag:    rust:1.87-bookworm\n#   * Digest: sha256:${new}\n#   * Pinned: $(date -u +%Y-%m-%d)|" \
                -e "s|FROM rust:1.87-bookworm@sha256:[a-f0-9]\{64\}|FROM rust:1.87-bookworm@sha256:${new}|" \
                "${DOCKERFILE}"
            ;;
        debian:bookworm-slim)
            sed -i \
                -e "s|^#   \* Tag:    debian:bookworm-slim$|#   * Tag:    debian:bookworm-slim\n#   * Digest: sha256:${new}\n#   * Pinned: $(date -u +%Y-%m-%d)|" \
                -e "s|FROM debian:bookworm-slim@sha256:[a-f0-9]\{64\}|FROM debian:bookworm-slim@sha256:${new}|" \
                "${DOCKERFILE}"
            ;;
    esac
done

# Rewrite the workflow env entries.
sed -i \
    -e "s|^  RUST_BUILDER_DIGEST: sha256:[a-f0-9]\{64\}$|  RUST_BUILDER_DIGEST: sha256:${NEW_DIGESTS[rust:1.87-bookworm]}|" \
    -e "s|^  DEBIAN_RUNTIME_DIGEST: sha256:[a-f0-9]\{64\}$|  DEBIAN_RUNTIME_DIGEST: sha256:${NEW_DIGESTS[debian:bookworm-slim]}|" \
    "${WORKFLOW}"

echo
echo "Updated digests:" >&2
for image in "${BASE_IMAGES[@]}"; do
    echo "  ${image} -> sha256:${NEW_DIGESTS[${image}]}" >&2
done
echo
echo "Next steps:" >&2
echo "  1. Inspect the diff: git diff Dockerfile .github/workflows/docker.yml" >&2
echo "  2. Smoke-test the new pins: docker build -f Dockerfile ." >&2
echo "  3. Commit with: git commit -am 'chore(security): refresh Dockerfile + docker.yml base image digests'" >&2