#!/usr/bin/env bash
# Hook: cargo-check-features — match CI's feature graph locally (#2888)
#
# Problem
# -------
# The doublify/pre-commit-rust `cargo-check` hook at .pre-commit-config.yaml:31-33
# runs `cargo check` against default features only. CI, however, exercises a
# wider feature graph:
#
#   - cargo test --features wiring-tracing,multi-zone     (rust-tests.yml L76)
#   - cargo build --features cuda --features ort          (rust-tests.yml L696)
#   - cargo build --features cuda --features ort          (cuda smoke)
#   - cargo deny check with --all-features               (CI supply-chain gate)
#
# The result: locally-green PRs surface feature-gated compile errors ~15min
# into CI. Each false-positive wastes a CI round-trip and a reviewer cycle.
#
# Fix
# ---
# This hook replaces the doublify cargo-check with a diff-driven feature
# matrix that mirrors CI. Two modes:
#
#   (default) pre-commit invocation
#     - parses `git diff --cached --name-only` for staged files
#     - maps file patterns to feature flags:
#         src/python/**.rs                      -> python-bindings
#         src/sim/(multi_zone|multi_node|
#                  thermal_model)*.rs            -> multi-zone
#         src/ai/(surrogate|onnx|equipment_
#                 surrogate|modular_surrogate|
#                 neural_field)*.rs              -> ort
#         Cargo.toml feature-table edits         -> all features (escalation)
#     - runs `cargo check --workspace --features <F>` for each match
#     - always also runs `cargo check --workspace` (default features) so the
#       hook contract is strictly stronger than the old doublify hook
#
#   --all  full CI match
#     - runs `cargo check --workspace --all-features`
#     - matches the cargo-deny matrix exactly (slow but exhaustive)
#
#   --selftest
#     - exercises the diff-parser + feature-decider on synthetic fixtures
#     - does NOT invoke cargo (fast, no compiler needed)
#
# See issue #2888 for context.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root. pre-commit invokes hooks from the repo root, but a manual
# `bash .githooks/cargo-check-features.sh` may be run from anywhere.
# ---------------------------------------------------------------------------
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

log() { printf '[cargo-check-features] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# run_cargo_check <feature-flags...>
#
# Cargo check is workspace-wide; we forward the requested feature set
# verbatim. `--workspace` is added so sibling crates (fluxion-core,
# fluxion-behavior, fluxion-twin, ...) are checked too — see issue #2983,
# which closed the gap where bare `cargo check` only compiled the root.
# ---------------------------------------------------------------------------
run_cargo_check() {
    log "running: cargo check --workspace $*"
    cargo check --workspace "$@"
}

# ---------------------------------------------------------------------------
# detect_features_from_diff
#
# Returns a space-separated list of feature flags whose gated code paths
# are touched by the *staged* diff (vs HEAD). If no staged files match
# any known feature gate, returns the empty string.
#
# Cargo.toml feature-table edits escalate to `__all__` (handled by the
# caller), because the blast radius of changing [features] sections is
# large enough that targeted re-runs are likely to miss something.
# ---------------------------------------------------------------------------
detect_features_from_diff() {
    local files
    files=$(git diff --cached --name-only --diff-filter=ACMR -- '*.rs' '*.toml' 2>/dev/null || true)
    if [ -z "$files" ]; then
        printf ''
        return 0
    fi

    local feats=""
    # python-bindings: src/python/** — PyO3 bindings gated by `python-bindings`.
    if printf '%s\n' "$files" | grep -qE '^src/python/.*\.rs$'; then
        feats="$feats python-bindings"
    fi
    # multi-zone: src/sim/multi_zone_network.rs and friends. The `multi-zone`
    # feature gates the multi_zone_throughput benchmark (see Cargo.toml L478)
    # and is exercised by CI's `--features wiring-tracing,multi-zone` matrix.
    if printf '%s\n' "$files" | grep -qE '^src/sim/(multi_zone_network|multi_node|thermal_model)' ; then
        feats="$feats multi-zone"
    fi
    # ort: ONNX-runtime-backed surrogate under src/ai/. The `ort` feature pulls
    # in the `ort` crate (see Cargo.toml L102) and is required by the
    # surrogate_backend_parity test and the CUDA smoke job.
    if printf '%s\n' "$files" | grep -qE '^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)\.rs$|^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)_'; then
        feats="$feats ort"
    fi
    # Cargo.toml feature-table edits: small surface area, large blast radius.
    if printf '%s\n' "$files" | grep -qE '^Cargo\.toml$' \
       && git diff --cached -- Cargo.toml 2>/dev/null \
            | grep -qE '^\+[^+].*(python-bindings|pyo3|\bort\b|cuda|multi-zone|fluxion-cfd|fluxion-city|fluid|gauge-solver|kafka|dwave|tracing-subscriber-json)'; then
        feats="__all__"
    fi
    printf '%s' "$feats"
}

# ---------------------------------------------------------------------------
# selftest — synthetic diff scenarios, no cargo invocation.
# ---------------------------------------------------------------------------
selftest() {
    # Inline the regex tests directly. No temp dir is needed because we
    # exercise the parser with hard-coded fixture strings rather than files.
    local pass=0 fail=0

    check_re() {
        local label="$1" pattern="$2" input="$3"
        if printf '%s\n' "$input" | grep -qE "$pattern"; then
            printf '  ok    %s\n' "$label"; pass=$((pass+1))
        else
            printf '  FAIL  %s (pattern %s did not match %s)\n' "$label" "$pattern" "$input"; fail=$((fail+1))
        fi
    }
    no_match_re() {
        local label="$1" pattern="$2" input="$3"
        if printf '%s\n' "$input" | grep -qE "$pattern"; then
            printf '  FAIL  %s (pattern %s unexpectedly matched %s)\n' "$label" "$pattern" "$input"; fail=$((fail+1))
        else
            printf '  ok    %s\n' "$label"; pass=$((pass+1))
        fi
    }

    check_re "python-bindings: src/python/* matches" \
        '^src/python/.*\.rs$' \
        'src/python/bindings.rs'

    check_re "python-bindings: src/python/sub/foo.rs matches" \
        '^src/python/.*\.rs$' \
        'src/python/sub/foo.rs'

    no_match_re "python-bindings: src/lib.rs does NOT match" \
        '^src/python/.*\.rs$' \
        'src/lib.rs'

    check_re "multi-zone: multi_zone_network.rs matches" \
        '^src/sim/(multi_zone_network|multi_node|thermal_model)' \
        'src/sim/multi_zone_network.rs'

    check_re "multi-zone: multi_node_hvac_runner.rs matches" \
        '^src/sim/(multi_zone_network|multi_node|thermal_model)' \
        'src/sim/multi_node_hvac_runner.rs'

    check_re "multi-zone: thermal_model_core.rs matches" \
        '^src/sim/(multi_zone_network|multi_node|thermal_model)' \
        'src/sim/thermal_model_core.rs'

    no_match_re "multi-zone: src/sim/solar.rs does NOT match" \
        '^src/sim/(multi_zone_network|multi_node|thermal_model)' \
        'src/sim/solar.rs'

    check_re "ort: surrogate.rs matches" \
        '^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)\.rs$|^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)_' \
        'src/ai/surrogate.rs'

    check_re "ort: equipment_surrogate.rs matches" \
        '^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)\.rs$|^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)_' \
        'src/ai/equipment_surrogate.rs'

    check_re "ort: onnx_helpers.rs (underscore variant) matches" \
        '^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)\.rs$|^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)_' \
        'src/ai/onnx_helpers.rs'

    no_match_re "ort: src/ai/ukf.rs does NOT match" \
        '^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)\.rs$|^src/ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field)_' \
        'src/ai/ukf.rs'

    no_match_re "no-match: src/foo.rs does NOT match any" \
        '^src/(python|sim/(multi_zone_network|multi_node|thermal_model)|ai/(surrogate|onnx|equipment_surrogate|modular_surrogate|neural_field))' \
        'src/foo.rs'

    if [ "$fail" -eq 0 ]; then
        printf 'selftest: all %d pattern checks PASS\n' "$pass"
        return 0
    else
        printf 'selftest: %d FAIL / %d pass\n' "$fail" "$pass" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main dispatch.
# ---------------------------------------------------------------------------
case "${1:-diff}" in
    --all)
        log "--all: cargo check --workspace --all-features (slow, CI-equivalent)"
        run_cargo_check --all-features
        ;;

    --selftest)
        log "selftest mode (no cargo invocation)"
        selftest
        ;;

    diff|"")
        feats=$(detect_features_from_diff)
        if [ -z "$feats" ]; then
            log "no feature-gated files in staged diff → running default-features check only"
            run_cargo_check
            exit $?
        fi
        if [ "$feats" = "__all__" ]; then
            log "Cargo.toml feature table changed → escalating to --all-features"
            run_cargo_check --all-features
            exit $?
        fi

        log "diff-gated features: $feats"
        rc=0
        for feat in $feats; do
            if ! run_cargo_check --features "$feat"; then
                rc=1
            fi
        done
        # Always also run the default-features check. This preserves the
        # original doublify/pre-commit-rust hook contract.
        if ! run_cargo_check; then
            rc=1
        fi
        exit $rc
        ;;

    -h|--help|help)
        cat <<EOF
Usage: $0 [MODE]

Modes:
  diff      (default) parse staged diff, run cargo check for each matched
            feature, plus default-features check.
  --all     run cargo check --workspace --all-features (CI-equivalent, slow).
  --selftest exercise the diff-parser/pattern matrix without invoking cargo.
EOF
        ;;

    *)
        printf 'unknown mode: %s\n' "$1" >&2
        printf 'run with --help for usage\n' >&2
        exit 2
        ;;
esac
