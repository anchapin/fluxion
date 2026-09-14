#!/bin/sh
# Low-memory linker selection for Linux (Issue #3765).
#
# rustc invokes this wrapper in place of the system linker on Linux targets.
# Preference order:
#   1. mold — millisecond links, ~70% less RAM than GNU ld on Rust debug builds
#   2. lld  — broadly available, still far leaner than GNU ld
#   3. system default (`cc` driver) — guaranteed fallback so developers and
#      CI images without mold/lld keep building exactly as before.
#
# `-fuse-ld=<name>` makes the cc driver look up `ld.<name>` on PATH, so the
# wrapper checks for `ld.mold` / `ld.lld` (not the bare `mold` binary) before
# selecting them. The final `exec` falls back to the driver's default linker,
# which means the wrapper degrades to today's behavior when neither
# memory-efficient linker is installed — builds never hard-fail.
#
# This script must stay POSIX sh; it runs on every Linux link.
set -eu

driver=""
for candidate in cc gcc clang; do
    if command -v "$candidate" >/dev/null 2>&1; then
        driver="$candidate"
        break
    fi
done
if [ -z "$driver" ]; then
    echo ".cargo/linker-wrapper.sh: no C compiler driver (cc/gcc/clang) found in PATH" >&2
    exit 127
fi

if command -v ld.mold >/dev/null 2>&1; then
    exec "$driver" -fuse-ld=mold "$@"
fi
if command -v ld.lld >/dev/null 2>&1; then
    exec "$driver" -fuse-ld=lld "$@"
fi
exec "$driver" "$@"
