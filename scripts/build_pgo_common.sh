#!/usr/bin/env bash
# build_pgo_common.sh — Sourceable helpers for scripts/build_pgo.sh
#
# Provides:
#   - Color-aware log helpers (pg::info, pg::warn, pg::error, pg::step)
#   - Tool detection (pg::require_tool, pg::find_llvm_profdata)
#   - Path utilities (pg::realpath, pg::ensure_dir)
#   - Timing utilities (pg::now_ns, pg::duration_human)
#   - Size formatting (pg::format_bytes)
#
# Source this file from other shell scripts:
#
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     # shellcheck source=build_pgo_common.sh
#     source "$SCRIPT_DIR/build_pgo_common.sh"
#
# All helpers live under the `pg::` namespace to avoid colliding with
# ad-hoc helpers in the calling script. Helpers are intentionally
# framework-free (no dependency on cargo, llvm, or jq) so the test
# suite can exercise them in isolation.

set -o pipefail

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Resolve a path to an absolute, normalised form without requiring `realpath`.
# Echoes the result on stdout. Falls back to `readlink -f` when available.
pg::_abs_path() {
    local target="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath "$target" 2>/dev/null || echo "$target"
    elif command -v readlink >/dev/null 2>&1 && readlink -f "$target" >/dev/null 2>&1; then
        readlink -f "$target"
    else
        # Last-resort fallback: prepend $PWD if relative.
        case "$target" in
            /*) echo "$target" ;;
            *) echo "$PWD/$target" ;;
        esac
    fi
}

# True when stdout is a TTY. Used to suppress colour codes in CI logs.
pg::_is_tty() {
    [[ -t 1 ]]
}

pg::_color_red()    { pg::_is_tty && printf '\033[31m' || true; }
pg::_color_green()  { pg::_is_tty && printf '\033[32m' || true; }
pg::_color_yellow() { pg::_is_tty && printf '\033[33m' || true; }
pg::_color_blue()   { pg::_is_tty && printf '\033[34m' || true; }
pg::_color_bold()   { pg::_is_tty && printf '\033[1m'  || true; }
pg::_color_reset()  { pg::_is_tty && printf '\033[0m'  || true; }

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Emit a labelled log line on stderr.
#
# Usage: pg::info "Compiling crate"
pg::info() {
    pg::_color_blue   >&2
    printf '[INFO] ' >&2
    pg::_color_reset  >&2
    printf '%s\n' "$*" >&2
}

pg::warn() {
    pg::_color_yellow >&2
    printf '[WARN] ' >&2
    pg::_color_reset  >&2
    printf '%s\n' "$*" >&2
}

pg::error() {
    pg::_color_red    >&2
    printf '[ERROR] ' >&2
    pg::_color_reset  >&2
    printf '%s\n' "$*" >&2
}

# Highlight a numbered pipeline step.
#
# Usage: pg::step 1 "Building instrumented binary"
pg::step() {
    local idx="$1"; shift
    pg::_color_bold   >&2
    pg::_color_green  >&2
    printf '\n[STEP %s] ' "$idx" >&2
    pg::_color_reset  >&2
    pg::_color_bold   >&2
    printf '%s\n' "$*" >&2
    pg::_color_reset  >&2
}

# ---------------------------------------------------------------------------
# Tool detection
# ---------------------------------------------------------------------------

# Verify that a tool is on PATH; abort with a clear error if not.
#
# Usage: pg::require_tool cargo
#        pg::require_tool llvm-profdata 1.0
pg::require_tool() {
    local tool="$1"
    local min_version="${2:-}"
    local path

    if ! path="$(command -v "$tool" 2>/dev/null)"; then
        pg::error "Required tool '$tool' not found on PATH"
        case "$tool" in
            llvm-profdata)
                pg::error "Install via: rustup component add llvm-tools-preview"
                ;;
            cargo-pgo)
                pg::error "Install via: cargo install cargo-pgo"
                ;;
            cargo)
                pg::error "Install Rust via https://rustup.rs"
                ;;
        esac
        return 127
    fi

    if [[ -n "$min_version" ]]; then
        local actual
        actual="$("$tool" --version 2>/dev/null | head -n1 | grep -Eo '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1)"
        if [[ -z "$actual" ]]; then
            pg::warn "Could not parse version for $tool; required >= $min_version"
        else
            # Compare lexicographically; works for 1.10 vs 1.9 etc.
            if [[ "$actual" != "$min_version"* ]] \
                && ! printf '%s\n%s\n' "$min_version" "$actual" | sort -V -C; then
                pg::error "$tool version $actual < required $min_version"
                return 1
            fi
        fi
    fi

    printf '%s' "$path"
}

# Locate llvm-profdata. Prefers `llvm-profdata` on PATH, falling back to
# the rustup-managed copy that ships with `llvm-tools-preview`.
#
# Echoes the absolute path on stdout. Returns non-zero when no binary is
# available so the caller can surface a friendly error.
pg::find_llvm_profdata() {
    if command -v llvm-profdata >/dev/null 2>&1; then
        command -v llvm-profdata
        return 0
    fi

    # rustup-managed toolchain bins
    local rustup_bin
    rustup_bin="$(rustup which llvm-profdata 2>/dev/null || true)"
    if [[ -n "$rustup_bin" && -x "$rustup_bin" ]]; then
        printf '%s\n' "$rustup_bin"
        return 0
    fi

    # Direct fallback: walk known rustup toolchain locations.
    local candidate
    for candidate in \
        "$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata" \
        "$HOME/.rustup/toolchains/stable-aarch64-unknown-linux-gnu/lib/rustlib/aarch64-unknown-linux-gnu/bin/llvm-profdata" \
        "$HOME/.rustup/toolchains/stable-x86_64-apple-darwin/lib/rustlib/x86_64-apple-darwin/bin/llvm-profdata"
    do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

# Recursively create a directory if missing. Echoes the absolute path.
#
# Usage: pg::ensure_dir "$PGO_DIR/raw"
pg::ensure_dir() {
    local dir="$1"
    if [[ -z "$dir" ]]; then
        pg::error "pg::ensure_dir: empty path"
        return 1
    fi
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir" || {
            pg::error "Failed to create directory: $dir"
            return 1
        }
    fi
    pg::_abs_path "$dir"
}

# Compute the size of a file or directory in bytes using portable stat.
# Echoes the size on stdout. Returns 0 even when the path is missing
# (echoes 0 in that case) so callers can use it without guards.
pg::path_size_bytes() {
    local target="$1"
    if [[ ! -e "$target" ]]; then
        printf '0\n'
        return 0
    fi

    if [[ -d "$target" ]]; then
        # Sum apparent sizes via `du -sb` when available; otherwise fall
        # back to a find-driven portable loop.
        if command -v du >/dev/null 2>&1 && du -sb "$target" >/dev/null 2>&1; then
            du -sb "$target" | awk '{print $1}'
            return 0
        fi
        find "$target" -type f -printf '%s\n' 2>/dev/null | awk 'BEGIN{s=0} {s+=$1} END{print s+0}'
        return 0
    fi

    wc -c < "$target" | tr -d '[:space:]'
    printf '\n'
}

# Format a byte count as a human-readable string (KiB/MiB/GiB).
# Usage: pg::format_bytes 1234567
pg::format_bytes() {
    local bytes="${1:-0}"
    if ! [[ "$bytes" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$bytes"
        return 0
    fi

    if   (( bytes >= 1073741824 )); then printf '%.2f GiB\n' "$(awk -v b="$bytes" 'BEGIN{print b/1073741824}')"
    elif (( bytes >= 1048576 ));    then printf '%.2f MiB\n' "$(awk -v b="$bytes" 'BEGIN{print b/1048576}')"
    elif (( bytes >= 1024 ));       then printf '%.2f KiB\n' "$(awk -v b="$bytes" 'BEGIN{print b/1024}')"
    else                                printf '%d B\n'    "$bytes"
    fi
}

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

# Monotonic nanosecond timestamp (falls back to seconds on platforms
# without `date +%s%N`).
pg::now_ns() {
    local stamp
    stamp="$(date +%s%N 2>/dev/null || true)"
    if [[ "$stamp" == *N* || -z "$stamp" ]]; then
        # `date` does not support %N — return seconds as ns.
        printf '%s000000000\n' "$(date +%s)"
    else
        printf '%s\n' "$stamp"
    fi
}

# Convert a nanosecond duration to a human-readable string like
# "1m 23s" or "4.2s". Accepts integer nanoseconds on stdin or as $1.
pg::duration_human() {
    local ns="${1:-}"
    if [[ -z "$ns" ]]; then
        read -r ns
    fi
    if ! [[ "$ns" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$ns"
        return 0
    fi

    awk -v ns="$ns" 'BEGIN {
        s = ns / 1000000000
        if (s >= 3600) printf "%dh %dm %ds\n", int(s/3600), int((s%3600)/60), int(s%60)
        else if (s >= 60) printf "%dm %ds\n", int(s/60), int(s%60)
        else if (s >= 1) printf "%.2fs\n", s
        else printf "%dms\n", ns/1000000
    }'
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

# Print a small usage banner. Used by build_pgo.sh and tests.
pg::usage() {
    cat <<'USAGE'
Fluxion PGO build helper
Usage: build_pgo.sh [options]

Options:
  --pgo-dir DIR          Where to store profile data (default: target/pgo)
  --target TRIPLE        Cargo --target triple (default: host triple)
  --features LIST        Cargo --features list (default: empty)
  --profile NAME         Cargo profile name (default: release)
  --train-workload CMD   Override the training workload command.
                         Default: cargo test --release --test ashrae_140_validation
  --skip-generate        Skip the profile-generation build step
  --skip-train           Skip the training workload step (use existing .profraw)
  --skip-use             Skip the profile-use build step (generate-only)
  --clean                Wipe the PGO directory before starting
  --quiet                Suppress non-error output
  -h, --help             Show this help and exit

Environment:
  RUSTFLAGS              Extra flags passed to every cargo invocation.
  CARGO                  Cargo binary to invoke (default: cargo)
  LLVM_PROFDATA          llvm-profdata binary (auto-detected by default)
USAGE
}

# Tiny test-only entry point. When run as `bash build_pgo_common.sh`
# (without being sourced) print a summary of exported helpers and exit
# with status 0. The companion `scripts/ci/test_build_pgo.sh` exercises
# each helper.
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
    echo "build_pgo_common.sh — sourced helpers for the Fluxion PGO pipeline."
    echo "Helpers available under the pg:: namespace:"
    echo "  logging : info warn error step"
    echo "  tools   : require_tool find_llvm_profdata"
    echo "  fs      : ensure_dir path_size_bytes format_bytes"
    echo "  time    : now_ns duration_human"
    echo "  parse   : usage"
    exit 0
fi