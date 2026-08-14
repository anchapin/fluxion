#!/usr/bin/env bash
# scripts/check_mojo_toolchain.sh
#
# Advisory detect-gate for the Mojo SDK and Modular MAX CLI.
# Closes GitHub issue #2979 — informs the wave-orchestrator and human
# reviewers whether `mojo` / `max` are present on the host without
# blocking CI. The Rust path remains the source of truth per
# ARCHITECTURE.md and RULES.md; this script never exits non-zero.
#
# Usage:
#   bash scripts/check_mojo_toolchain.sh            # default (advisory)
#   bash scripts/check_mojo_toolchain.sh --strict    # exit 1 if missing
#                                                  # (for local pre-commit
#                                                  #   / wave pre-flight)
#
# Exit codes:
#   0 — advisory mode: toolchain may or may not be installed.
#   1 — --strict mode: `mojo` and/or `max` not on PATH.
#   2 — usage / internal error.
#
# Verifies (for each binary: `mojo`, `max`, optional `modular`):
#   1. `command -v <bin>` resolves to an executable on PATH.
#   2. `<bin> --version` exits 0 and prints a non-empty version string.
#
# Output lines are tagged `PASS` / `WARN` / `FAIL` / `INFO` so log parsers
# can grep on the prefix. Always exit 0 in the default (advisory) mode.
#
# See `docs/agents/mojo-setup.md` for install instructions.

set -u

strict=0
if [[ "${1:-}" == "--strict" ]]; then
    strict=1
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '3,30p' "$0"
    exit 0
elif [[ -n "${1:-}" ]]; then
    echo "usage: $0 [--strict|--help]" >&2
    exit 2
fi

# Colours (only when stderr is a TTY and NO_COLOR is unset).
if [[ -t 2 && -z "${NO_COLOR:-}" ]]; then
    c_pass=$'\033[32m'   # green
    c_warn=$'\033[33m'   # yellow
    c_fail=$'\033[31m'   # red
    c_info=$'\033[36m'   # cyan
    c_off=$'\033[0m'
else
    c_pass=""; c_warn=""; c_fail=""; c_info=""; c_off=""
fi

pass_count=0
warn_count=0
fail_count=0
# Track which required binaries (mojo, max) were missing on PATH. Used by
# the strict exit policy below. `modular` is optional and never contributes
# to the strict failure count.
missing_bins=()

emit_pass() { echo "${c_pass}PASS${c_off}  $*"; pass_count=$((pass_count + 1)); }
emit_warn() { echo "${c_warn}WARN${c_off}  $*"; warn_count=$((warn_count + 1)); }
emit_fail() { echo "${c_fail}FAIL${c_off}  $*"; fail_count=$((fail_count + 1)); }
emit_info() { echo "${c_info}INFO${c_off}  $*"; }

check_binary() {
    local bin="$1"
    local resolved
    if ! resolved="$(command -v "$bin" 2>/dev/null)"; then
        emit_warn "$bin not found on PATH (${bin} SDK not installed)"
        missing_bins+=("$bin")
        return 1
    fi
    emit_pass "$bin found: $resolved"

    # `<bin> --version` must exit 0 and print a non-empty version string.
    local version_output
    local version_rc=0
    # 10-second timeout via `timeout` (coreutils; widely available).
    if command -v timeout >/dev/null 2>&1; then
        version_output="$(timeout 10s "$resolved" --version 2>&1)" || version_rc=$?
    else
        version_output="$("$resolved" --version 2>&1)" || version_rc=$?
    fi

    # Trim leading whitespace; pick first non-empty line.
    local first_line
    first_line="$(printf '%s\n' "$version_output" | sed -n 's/^[[:space:]]*//;/./p' | head -n 1 || true)"

    if [[ $version_rc -ne 0 ]]; then
        emit_warn "$bin found but '$bin --version' exited $version_rc"
        missing_bins+=("$bin")
        return 1
    fi
    if [[ -z "$first_line" ]]; then
        emit_warn "$bin found but '$bin --version' produced no output"
        missing_bins+=("$bin")
        return 1
    fi
    emit_pass "$bin version: $first_line"
    return 0
}

echo "=== Mojo Toolchain Detect Gate (issue #2979) ==="
echo "Mode: $([[ $strict -eq 1 ]] && echo 'strict (exit 1 if missing)' || echo 'advisory (always exit 0)')"
echo "Host: $(uname -srm 2>/dev/null || uname -a)"
echo

# Mojo language compiler (the primary gate signal).
check_binary mojo || true
echo

# Modular MAX framework CLI (`max serve`, `max generate`, etc.).
check_binary max || true
echo

# Optional: legacy `modular` CLI — present if Path C in mojo-setup.md was
# used. Not all install paths install it (pixi/uv install `mojo` directly
# without the `modular` wrapper), so a missing `modular` is WARN not FAIL.
if command -v modular >/dev/null 2>&1; then
    emit_pass "modular CLI found: $(command -v modular)"
    modular_version="$(modular --version 2>&1 | head -n 1 || true)"
    if [[ -n "$modular_version" ]]; then
        emit_pass "modular version: $modular_version"
    else
        emit_warn "modular CLI found but 'modular --version' produced no output"
    fi
else
    emit_warn "modular CLI not on PATH (legacy installer only; pixi/uv installs skip it)"
fi
echo

# Summary
total=$((pass_count + warn_count + fail_count))
echo "=== Summary ==="
echo "  PASS: $pass_count"
echo "  WARN: $warn_count"
echo "  FAIL: $fail_count"
echo "  Total checks: $total"

if [[ $fail_count -eq 0 && $warn_count -eq 0 ]]; then
    emit_info "Mojo toolchain fully installed."
elif [[ $fail_count -eq 0 ]]; then
    emit_info "Mojo toolchain partially installed (or absent — non-blocking)."
    emit_info "Install guide: docs/agents/mojo-setup.md"
else
    emit_info "Mojo toolchain check reported $fail_count hard failures."
    emit_info "See docs/agents/mojo-setup.md for install paths."
fi

# Exit policy:
#   - advisory (default): always 0 — informational only.
#   - --strict: 1 if any *required* binary (mojo or max) was missing on PATH
#     OR any check reported a hard FAIL. WARN-only findings (e.g. missing
#     `modular` legacy CLI) do not block.
if [[ $strict -eq 1 ]]; then
    required_missing=0
    for bin in "${missing_bins[@]}"; do
        case "$bin" in
            mojo|max) required_missing=$((required_missing + 1)) ;;
        esac
    done
    if [[ $required_missing -gt 0 || $fail_count -gt 0 ]]; then
        exit 1
    fi
fi
exit 0
