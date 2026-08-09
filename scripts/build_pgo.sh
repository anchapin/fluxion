#!/usr/bin/env bash
# build_pgo.sh — Profile-Guided Optimization (PGO) build pipeline
#
# Implements the acceptance criterion from issue #2563:
#   1. Build the crate with `-Cprofile-generate` (instrumented binary).
#   2. Run the ASHRAE 140 validation suite as the training workload so the
#      hot paths (5R1C/9R4C solvers, conduction transfer functions, weather
#      interpolation) get representative edge/branch counts.
#   3. Merge the raw `.profraw` files into a single `.profdata` using
#      `llvm-profdata`.
#   4. Rebuild with `-Cprofile-use=<profdata>` to produce the optimized
#      binary.
#
# Usage:
#     ./scripts/build_pgo.sh                    # default settings
#     ./scripts/build_pgo.sh --pgo-dir pgo      # custom PGO dir
#     ./scripts/build_pgo.sh --skip-generate    # reuse existing .profraw
#     ./scripts/build_pgo.sh --clean            # wipe PGO dir first
#
# Environment:
#   RUSTFLAGS    — extra flags passed through to every cargo invocation.
#                 The script sets `-Cprofile-generate=...` /
#                 `-Cprofile-use=...` itself; existing RUSTFLAGS are
#                 preserved.
#   CARGO        — cargo binary (default: cargo).
#   LLVM_PROFDATA— explicit path to llvm-profdata; auto-detected otherwise.
#
# Exit codes:
#   0   success — profile-use binary built
#   1   user/configuration error (missing tool, bad flag)
#   2   build failure (cargo invocation failed)
#   3   profile merge failure (llvm-profdata error)
#
# See CONTRIBUTING.md "PGO build pipeline" for a fuller walk-through.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script location and source the helper library
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=build_pgo_common.sh
source "$SCRIPT_DIR/build_pgo_common.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PGO_DIR="$PROJECT_ROOT/target/pgo"
CARGO_TARGET=""
CARGO_FEATURES=""
CARGO_PROFILE="release"
TRAIN_WORKLOAD=""
SKIP_GENERATE=false
SKIP_TRAIN=false
SKIP_USE=false
CLEAN=false
QUIET=false

PROFILE_GENERATE_FLAGS=("-Cprofile-generate")
PROFILE_USE_FLAGS=("-Cprofile-use")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pgo-dir)
            PGO_DIR="$2"; shift 2 ;;
        --target)
            CARGO_TARGET="$2"; shift 2 ;;
        --features)
            CARGO_FEATURES="$2"; shift 2 ;;
        --profile)
            CARGO_PROFILE="$2"; shift 2 ;;
        --train-workload)
            TRAIN_WORKLOAD="$2"; shift 2 ;;
        --skip-generate)
            SKIP_GENERATE=true; shift ;;
        --skip-train)
            SKIP_TRAIN=true; shift ;;
        --skip-use)
            SKIP_USE=true; shift ;;
        --clean)
            CLEAN=true; shift ;;
        --quiet)
            QUIET=true; shift ;;
        -h|--help)
            pg::usage
            exit 0 ;;
        *)
            pg::error "Unknown argument: $1"
            pg::usage >&2
            exit 1 ;;
    esac
done

if $QUIET; then
    # Silence pg::info / pg::warn by redirecting them via a no-op fd.
    pg::info()    { :; }
    pg::warn()    { :; }
    pg::step()    { printf '\n[STEP %s] %s\n' "$1" "${*:2}" >&2; }
fi

# ---------------------------------------------------------------------------
# Tool checks
# ---------------------------------------------------------------------------
pg::require_tool cargo >/dev/null

PROFDATA_BIN="${LLVM_PROFDATA:-}"
if [[ -z "$PROFDATA_BIN" ]]; then
    if ! PROFDATA_BIN="$(pg::find_llvm_profdata)"; then
        pg::error "llvm-profdata not found on PATH and LLVM_PROFDATA is unset."
        pg::error "Install via: rustup component add llvm-tools-preview"
        exit 1
    fi
fi
pg::info "Using llvm-profdata: $PROFDATA_BIN"

# ---------------------------------------------------------------------------
# Layout under PGO_DIR
#   raw/       — .profraw files dropped by the instrumented binary
#   merged/    — merged .profdata consumed by -Cprofile-use
#   logs/      — captured stdout/stderr from each step
# ---------------------------------------------------------------------------
RAW_DIR="$PGO_DIR/raw"
MERGED_DIR="$PGO_DIR/merged"
LOG_DIR="$PGO_DIR/logs"
PROFDATA_FILE="$MERGED_DIR/profdata"

if $CLEAN; then
    pg::warn "Cleaning $PGO_DIR"
    rm -rf "$PGO_DIR"
fi

pg::ensure_dir "$RAW_DIR" >/dev/null
pg::ensure_dir "$MERGED_DIR" >/dev/null
pg::ensure_dir "$LOG_DIR" >/dev/null

ABS_PGO_DIR="$(pg::_abs_path "$PGO_DIR")"
ABS_RAW_DIR="$(pg::_abs_path "$RAW_DIR")"
ABS_PROFDATA="$(pg::_abs_path "$PROFDATA_FILE")"

# ---------------------------------------------------------------------------
# Cargo argument builder
# ---------------------------------------------------------------------------
# Build the trailing portion of a `cargo build/test …` invocation from
# the parsed options. Echoes a single string on stdout.
#
# Usage: cargo_args_for_build
cargo_args_for_build() {
    local parts=("--profile" "$CARGO_PROFILE")
    if [[ -n "$CARGO_TARGET" ]]; then
        parts+=("--target" "$CARGO_TARGET")
    fi
    if [[ -n "$CARGO_FEATURES" ]]; then
        parts+=("--features" "$CARGO_FEATURES")
    fi
    printf '%s ' "${parts[@]}"
}

# Compose RUSTFLAGS, preserving any caller-supplied flags.
# Usage: rustflags_for_generate ; rustflags_for_use
rustflags_for_generate() {
    local extra="${RUSTFLAGS:-}"
    printf '%s -Cprofile-generate=%s' "$extra" "$ABS_RAW_DIR"
}

rustflags_for_use() {
    local extra="${RUSTFLAGS:-}"
    printf '%s -Cprofile-use=%s' "$extra" "$ABS_PROFDATA"
}

# ---------------------------------------------------------------------------
# Step 1: profile generation build
# ---------------------------------------------------------------------------
step_generate() {
    if $SKIP_GENERATE; then
        pg::warn "Skipping profile-generation build (--skip-generate)"
        return 0
    fi

    pg::step 1 "Building instrumented binary (-Cprofile-generate)"
    local start_ns end_ns
    local cargo_args log_file
    cargo_args="$(cargo_args_for_build)"
    log_file="$LOG_DIR/01-generate.log"

    start_ns="$(pg::now_ns)"
    # shellcheck disable=SC2086  # word-splitting is intentional for cargo args
    if ! RUSTFLAGS="$(rustflags_for_generate)" \
         "${CARGO:-cargo}" build $cargo_args 2>&1 | tee "$log_file"; then
        pg::error "Profile-generation build failed; see $log_file"
        exit 2
    fi
    end_ns="$(pg::now_ns)"

    pg::info "Profile-generation build completed in $(pg::duration_human $((end_ns - start_ns)))"
}

# ---------------------------------------------------------------------------
# Step 2: training workload
# ---------------------------------------------------------------------------
step_train() {
    if $SKIP_TRAIN; then
        pg::warn "Skipping training workload (--skip-train)"
        return 0
    fi

    pg::step 2 "Running training workload (ASHRAE 140 validation)"
    local start_ns end_ns
    local log_file="$LOG_DIR/02-train.log"
    local cmd=()

    if [[ -n "$TRAIN_WORKLOAD" ]]; then
        # Split the user-supplied command into an argv array. We don't
        # attempt to honour quotes/escapes; the documented contract is a
        # plain space-separated command line.
        # shellcheck disable=SC2206
        cmd=($TRAIN_WORKLOAD)
    else
        cmd=("${CARGO:-cargo}" "test" "--profile" "$CARGO_PROFILE"
             "--test" "ashrae_140_validation"
             "--" "--nocapture")
    fi

    start_ns="$(pg::now_ns)"
    if ! "${cmd[@]}" 2>&1 | tee "$log_file"; then
        pg::error "Training workload failed; see $log_file"
        exit 2
    fi
    end_ns="$(pg::now_ns)"

    pg::info "Training workload completed in $(pg::duration_human $((end_ns - start_ns)))"
}

# ---------------------------------------------------------------------------
# Step 3: merge .profraw into .profdata
# ---------------------------------------------------------------------------
step_merge() {
    pg::step 3 "Merging profile data with llvm-profdata"
    local log_file="$LOG_DIR/03-merge.log"

    # Discover any .profraw files dropped into the raw/ directory. If
    # none are found, abort — running the use build against an empty
    # profile silently produces an unoptimised binary.
    local -a raw_files=()
    shopt -s nullglob
    raw_files=("$ABS_RAW_DIR"/*.profraw)
    shopt -u nullglob

    if [[ ${#raw_files[@]} -eq 0 ]]; then
        pg::error "No .profraw files found under $ABS_RAW_DIR"
        pg::error "Did the training workload actually run the instrumented binary?"
        exit 3
    fi

    pg::info "Found ${#raw_files[@]} .profraw file(s); merging into profdata"

    local start_ns end_ns
    start_ns="$(pg::now_ns)"
    if ! "$PROFDATA_BIN" merge -sparse \
            -output "$ABS_PROFDATA" \
            "${raw_files[@]}" 2>&1 | tee "$log_file"; then
        pg::error "llvm-profdata merge failed; see $log_file"
        exit 3
    fi
    end_ns="$(pg::now_ns)"

    pg::info "Merge completed in $(pg::duration_human $((end_ns - start_ns)))"
}

# ---------------------------------------------------------------------------
# Step 4: profile-use build
# ---------------------------------------------------------------------------
step_use() {
    if $SKIP_USE; then
        pg::warn "Skipping profile-use build (--skip-use)"
        return 0
    fi

    if [[ ! -f "$ABS_PROFDATA" ]]; then
        pg::error "Profile data not found at $ABS_PROFDATA; cannot run profile-use build"
        exit 1
    fi

    pg::step 4 "Building optimized binary (-Cprofile-use)"
    local start_ns end_ns
    local cargo_args log_file
    cargo_args="$(cargo_args_for_build)"
    log_file="$LOG_DIR/04-use.log"

    start_ns="$(pg::now_ns)"
    # shellcheck disable=SC2086  # word-splitting is intentional for cargo args
    if ! RUSTFLAGS="$(rustflags_for_use)" \
         "${CARGO:-cargo}" build $cargo_args 2>&1 | tee "$log_file"; then
        pg::error "Profile-use build failed; see $log_file"
        exit 2
    fi
    end_ns="$(pg::now_ns)"

    pg::info "Profile-use build completed in $(pg::duration_human $((end_ns - start_ns)))"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
    local profdata_bytes=0
    if [[ -f "$ABS_PROFDATA" ]]; then
        profdata_bytes="$(pg::path_size_bytes "$ABS_PROFDATA")"
    fi

    local raw_bytes=0
    raw_bytes="$(pg::path_size_bytes "$ABS_RAW_DIR")"

    cat <<SUMMARY

==========================================================
PGO build pipeline summary
==========================================================
PGO directory        : $ABS_PGO_DIR
Cargo profile        : $CARGO_PROFILE
Cargo target         : ${CARGO_TARGET:-<host>}
Cargo features       : ${CARGO_FEATURES:-<none>}
llvm-profdata        : $PROFDATA_BIN
----------------------------------------------------------
Raw profile data     : $(pg::format_bytes "$raw_bytes")
Merged profile data  : $(pg::format_bytes "$profdata_bytes")  ($ABS_PROFDATA)
Build logs           : $LOG_DIR/
----------------------------------------------------------
To re-run only the use step against an existing profile:

    RUSTFLAGS="-Cprofile-use=$ABS_PROFDATA" \\
        ${CARGO:-cargo} build --profile $CARGO_PROFILE

To inspect the merged profile:

    $PROFDATA_BIN show "$ABS_PROFDATA" | head

==========================================================
SUMMARY
}

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
pg::info "Fluxion PGO pipeline starting"
pg::info "PGO dir : $ABS_PGO_DIR"
pg::info "Profile : $CARGO_PROFILE"

step_generate
step_train
step_merge
step_use
print_summary

pg::info "PGO pipeline finished successfully"