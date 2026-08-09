#!/usr/bin/env bash
# test_build_pgo.sh — Bash test suite for scripts/build_pgo_common.sh
#
# Exercises every helper in the pg:: namespace without requiring cargo,
# llvm-profdata, or network access. Designed to run in CI under the
# pre-commit hooks and as part of the python-tests workflow.
#
# Usage:
#     bash scripts/ci/test_build_pgo.sh
#     bash scripts/ci/test_build_pgo.sh --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PGO_COMMON="$PROJECT_ROOT/scripts/build_pgo_common.sh"

VERBOSE=false
if [[ "${1:-}" == "--verbose" ]]; then VERBOSE=true; fi

# Verbose log helper. We deliberately check the variable explicitly instead
# of writing `$VERBOSE && echo …` — bash otherwise treats the literal
# string "false" as a command name and runs `false`, which trips set -e.
pg::test_log() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "  ok  $*"
    fi
}

PASS=0
FAIL=0
FAILURES=()

# ---------------------------------------------------------------------------
# Tiny assertion helpers
# ---------------------------------------------------------------------------

assert_eq() {
    local actual="$1" expected="$2" label="$3"
    if [[ "$actual" == "$expected" ]]; then
        PASS=$((PASS + 1))
        pg::test_log "$label"
    else
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: expected '$expected', got '$actual'")
        echo "  FAIL  $label"
        echo "        expected: $expected"
        echo "        actual  : $actual"
    fi
}

assert_contains() {
    local haystack="$1" needle="$2" label="$3"
    if [[ "$haystack" == *"$needle"* ]]; then
        PASS=$((PASS + 1))
        pg::test_log "$label"
    else
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: '$haystack' does not contain '$needle'")
        echo "  FAIL  $label"
        echo "        needle  : $needle"
        echo "        haystack: $haystack"
    fi
}

assert_true() {
    local value="$1" label="$2"
    if [[ "$value" == "0" || "$value" == "true" ]]; then
        PASS=$((PASS + 1))
        $VERBOSE && echo "  ok  $label"
    else
        FAIL=$((FAIL + 1))
        FAILURES+=("$label: expected truthy, got '$value'")
        echo "  FAIL  $label"
    fi
}

# ---------------------------------------------------------------------------
# Source the library under test
# ---------------------------------------------------------------------------

if [[ ! -f "$PGO_COMMON" ]]; then
    echo "FATAL: $PGO_COMMON not found" >&2
    exit 2
fi

# shellcheck source=/dev/null
source "$PGO_COMMON"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

echo "==> pg::format_bytes"

assert_eq "$(pg::format_bytes 0)"          "0 B"     "format_bytes 0 -> 0 B"
assert_eq "$(pg::format_bytes 1023)"       "1023 B"  "format_bytes 1023 -> 1023 B"
assert_eq "$(pg::format_bytes 1024)"       "1.00 KiB" "format_bytes 1024 -> 1.00 KiB"
assert_eq "$(pg::format_bytes 1048576)"    "1.00 MiB" "format_bytes 1048576 -> 1.00 MiB"
assert_eq "$(pg::format_bytes 1073741824)" "1.00 GiB" "format_bytes 1073741824 -> 1.00 GiB"
assert_eq "$(pg::format_bytes 15728640)"   "15.00 MiB" "format_bytes 15MiB"

echo "==> pg::duration_human"

assert_eq "$(pg::duration_human 500000000)"   "500ms"       "duration_human 500ms"
assert_eq "$(pg::duration_human 1500000000)"  "1.50s"       "duration_human 1.5s"
assert_eq "$(pg::duration_human 90000000000)" "1m 30s"      "duration_human 90s"
assert_eq "$(pg::duration_human 3723000000000)" "1h 2m 3s"  "duration_human 1h2m3s"
assert_eq "$(pg::duration_human 250000)"      "0ms"         "duration_human 250us"
assert_eq "$(pg::duration_human 999000000)"   "999ms"       "duration_human 999ms (sub-second boundary)"

echo "==> pg::now_ns"

ns="$(pg::now_ns)"
if [[ "$ns" =~ ^[0-9]+$ ]] && [[ ${#ns} -ge 10 ]]; then
    PASS=$((PASS + 1))
    pg::test_log "now_ns is numeric >= 10 digits"
else
    FAIL=$((FAIL + 1))
    FAILURES+=("now_ns returned '$ns' (expected >=10 digit integer)")
    echo "  FAIL  now_ns"
fi

echo "==> pg::ensure_dir"

tmpdir="$(mktemp -d -t pgo_test_XXXXXX)"
trap 'rm -rf "$tmpdir"' EXIT

made="$(pg::ensure_dir "$tmpdir/sub/deeper" 2>/dev/null)"
assert_contains "$made" "sub/deeper" "ensure_dir returns abs path"
if [[ -d "$tmpdir/sub/deeper" ]]; then
    PASS=$((PASS + 1))
    pg::test_log "ensure_dir actually creates dirs"
else
    FAIL=$((FAIL + 1))
    FAILURES+=("ensure_dir did not create directory")
    echo "  FAIL  ensure_dir creates dir"
fi

# Re-running on existing dir must not fail.
existing="$(pg::ensure_dir "$tmpdir" 2>/dev/null)"
assert_contains "$existing" "$(basename "$tmpdir")" "ensure_dir idempotent"

echo "==> pg::path_size_bytes"

# Use a small known file we just created.
echo "hello world" > "$tmpdir/sample.txt"
sz="$(pg::path_size_bytes "$tmpdir/sample.txt")"
assert_eq "$sz" "12" "path_size_bytes counts file bytes"

# Missing path returns 0 (caller-friendly default).
missing="$(pg::path_size_bytes "$tmpdir/does-not-exist")"
assert_eq "$missing" "0" "path_size_bytes of missing path is 0"

# Directory size is the sum of children.
mkdir "$tmpdir/with_files"
echo "abcd" > "$tmpdir/with_files/a.txt"
echo "efghijkl" > "$tmpdir/with_files/b.txt"
dirsz="$(pg::path_size_bytes "$tmpdir/with_files")"
if [[ "$dirsz" -ge 12 ]]; then
    PASS=$((PASS + 1))
    pg::test_log "path_size_bytes sums dir contents (>=12)"
else
    FAIL=$((FAIL + 1))
    FAILURES+=("path_size_bytes(dir) = $dirsz, expected >=12")
    echo "  FAIL  path_size_bytes sums dir"
fi

echo "==> pg::find_llvm_profdata"

profdata_path="$(pg::find_llvm_profdata 2>/dev/null || echo MISSING)"
if [[ "$profdata_path" == "MISSING" ]]; then
    # Not fatal — many CI sandboxes lack llvm-tools-preview. Skip with note.
    echo "  SKIP  llvm-profdata not installed (expected on minimal runners)"
else
    if [[ -x "$profdata_path" ]]; then
        PASS=$((PASS + 1))
        pg::test_log "find_llvm_profdata returns executable ($profdata_path)"
    else
        FAIL=$((FAIL + 1))
        FAILURES+=("find_llvm_profdata returned non-executable '$profdata_path'")
        echo "  FAIL  find_llvm_profdata"
    fi
fi

echo "==> pg::require_tool"

# Always available
cargo_path="$(pg::require_tool cargo 2>/dev/null || echo MISSING)"
if [[ "$cargo_path" != "MISSING" ]]; then
    PASS=$((PASS + 1))
    pg::test_log "require_tool cargo finds cargo"
else
    FAIL=$((FAIL + 1))
    FAILURES+=("require_tool cargo failed")
    echo "  FAIL  require_tool cargo"
fi

# Missing tool should fail with non-zero status; we capture stderr to keep
# the test output tidy.
set +e
pg::require_tool definitely-not-a-real-tool-xyz 2>/dev/null
rc=$?
set -e
assert_eq "$rc" "127" "require_tool returns 127 for missing tool"

echo "==> pg::usage"

usage_output="$(pg::usage 2>/dev/null)"
assert_contains "$usage_output" "Fluxion PGO build helper" "usage contains title"
assert_contains "$usage_output" "--pgo-dir"                "usage documents --pgo-dir"
assert_contains "$usage_output" "--train-workload"         "usage documents --train-workload"
assert_contains "$usage_output" "--skip-use"               "usage documents --skip-use"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "===================================================="
echo "  test_build_pgo.sh — $PASS passed, $FAIL failed"
echo "===================================================="

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "Failures:"
    for f in "${FAILURES[@]}"; do echo "  - $f"; done
    exit 1
fi

exit 0