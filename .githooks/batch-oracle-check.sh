#!/bin/bash
# Hook: Validate BatchOracle / rayon parallelism pattern
# Purpose: Ensure evaluate_population uses single-level rayon parallelism only,
#          and detect nested par_iter/into_par_iter anywhere in scope.
#          Nested parallelism causes rayon thread-pool exhaustion (#1065).
# Scope (#2524): lib.rs + src/sim/ + src/ai/ + src/validation/
#          (previously lib.rs only). par_chunks / par_bridge are safe and excluded.
#
# This is critical for the 100x speedup goal (10k+ configs/sec).
#
# Modes:
#   (default)          pre-commit: args are staged file paths
#   --scan FILE...     run nested-par_iter scan on FILEs, exit 1 on violation
#   --selftest         run internal fixtures + the real tree, report PASS/FAIL

set -e

# ---------------------------------------------------------------------------
# Nested par_iter/into_par_iter detector (#2524).
#
# A par_iter statement is "active" from its first par_iter/into_par_iter token
# until its terminating ';' (at the same/shallower brace depth) or until brace
# depth drops below the par_iter's depth. A second par_iter/into_par_iter found
# at a STRICTLY GREATER brace depth while a statement is active == nested
# parallelism (it executes inside the outer call's worker closure).
#
# This correctly accepts legitimate patterns that a naive "count > 1" rule would
# false-positive on: sequential par_iter calls in separate statements, a par_iter
# in an `if` branch after an earlier par_iter statement terminated, and a
# `.par_iter_mut().zip(other.par_iter_mut())` single parallel traversal.
#
# par_chunks / par_bridge do not contain the `par_iter` token, so they are
# naturally excluded.
# ---------------------------------------------------------------------------
scan_nested() {
    python3 - "$@" <<'PYEOF'
import re, sys

PAR = r'(?<![A-Za-z0-9_])(?:into_)?par_iter'

def clean(src):
    # Blank out comments and string/char-literal contents while preserving
    # length and newlines, so braces inside literals/text don't corrupt depth.
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':          # line comment
            while i < n and src[i] != '\n':
                out.append(' '); i += 1
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '*':          # block comment
            out += [' ', ' ']; i += 2
            while i < n and not (src[i] == '*' and i + 1 < n and src[i + 1] == '/'):
                out.append('\n' if src[i] == '\n' else ' '); i += 1
            if i < n:
                out += [' ', ' ']; i += 2
            continue
        if c == 'r' and i + 1 < n and src[i + 1] == '"':          # raw string r"..."
            out += [' ', ' ']; i += 2
            while i < n and src[i] != '"':
                out.append('\n' if src[i] == '\n' else ' '); i += 1
            if i < n:
                out.append(' '); i += 1
            continue
        if c == '"':                                              # string literal
            out.append(' '); i += 1
            while i < n and src[i] != '"':
                if src[i] == '\\' and i + 1 < n:
                    out += [' ', ' ']; i += 2; continue
                out.append('\n' if src[i] == '\n' else ' '); i += 1
            if i < n:
                out.append(' '); i += 1
            continue
        if c == "'":                                              # char literal (skip lifetimes)
            j = i + 1
            while j < n and src[j] != "'" and j - i <= 4:
                if src[j] == '\\' and j + 1 < n:
                    j += 2; continue
                j += 1
            if j < n and src[j] == "'" and j - i >= 2:
                out.append(' ' * (j - i + 1)); i = j + 1; continue
            out.append(c); i += 1; continue
        out.append(c); i += 1
    return ''.join(out)

def find_nested(src):
    text = clean(src)
    events = []
    for m in re.finditer(r'[{};]|' + PAR, text):
        events.append((m.start(), m.group()))
    events.sort()
    depth = 0
    active = False
    active_depth = 0
    bad = []
    for pos, tok in events:
        line = text.count('\n', 0, pos) + 1
        if tok == '{':
            depth += 1
        elif tok == '}':
            depth -= 1
            if active and depth < active_depth:
                active = False
        elif tok == ';':
            if active and depth <= active_depth:
                active = False
        else:                                                     # par_iter / into_par_iter
            if active and depth > active_depth:
                bad.append(line)
            elif (not active) or depth <= active_depth:
                active = True
                active_depth = depth
    return bad

def main(argv):
    failed = False
    for path in argv:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                src = fh.read()
        except OSError as e:
            print("batch-oracle-check: cannot read %s: %s" % (path, e),
                  file=sys.stderr)
            failed = True
            continue
        for ln in find_nested(src):
            print("PERF REGRESSION: %s:%d: nested par_iter/into_par_iter "
                  "inside an active par_iter statement "
                  "(rayon thread-pool exhaustion, #1065/#2524)" % (path, ln),
                  file=sys.stderr)
            failed = True
    if failed:
        print("  -> Fix: use single-level population-wide parallelism only; "
              "replace the inner par_iter with a sequential loop, par_chunks, "
              "or hoist the work out of the closure.", file=sys.stderr)
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
PYEOF
}

# ---------------------------------------------------------------------------
# --scan: run the nested detector on explicit file args.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--scan" ]; then
    shift
    scan_nested "$@"
    exit $?
fi

# ---------------------------------------------------------------------------
# --selftest: synthetic fixtures + the real in-scope tree must behave correctly.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--selftest" ]; then
    set +e
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT

    cat > "$tmp/safe.rs" <<'RUST'
pub fn good(v: Vec<Vec<f64>>) -> Vec<f64> {
    let a: Vec<f64> = v.par_iter().map(|r| r[0]).collect();
    let mut e = vec![0.0; v.len()];
    v.par_iter_mut().zip(e.par_iter_mut()).for_each(|(m, x)| { *x = 1.0; });
    if !a.is_empty() {
        let b: Vec<f64> = v.par_iter().map(|r| r[1]).collect();
        return b;
    }
    a
}
RUST
    cat > "$tmp/nested.rs" <<'RUST'
pub fn bad(pop: Vec<Vec<f64>>) -> Vec<f64> {
    pop.par_iter().map(|cfg| {
        cfg.par_iter().copied().sum::<f64>()
    }).collect()
}
RUST
    cat > "$tmp/nested2.rs" <<'RUST'
pub fn bad2(c: Vec<Vec<f64>>) {
    c.into_par_iter().for_each(|row| {
        let _: f64 = (0..8).into_par_iter().map(|x| x as f64).sum();
    });
}
RUST
    cat > "$tmp/chunks.rs" <<'RUST'
pub fn ok_chunks(v: Vec<Vec<f64>>) {
    v.par_chunks(100).for_each(|chunk| { let _ = chunk.len(); });
}
RUST

    fail=0
    scan_nested "$tmp/safe.rs" >/dev/null 2>&1; [ $? -eq 0 ] || { echo "selftest FAIL: safe fixture flagged as violation"; fail=1; }
    scan_nested "$tmp/nested.rs" >/dev/null 2>&1; [ $? -eq 1 ] || { echo "selftest FAIL: nested.rs not caught"; fail=1; }
    scan_nested "$tmp/nested2.rs" >/dev/null 2>&1; [ $? -eq 1 ] || { echo "selftest FAIL: nested2.rs not caught"; fail=1; }
    scan_nested "$tmp/chunks.rs" >/dev/null 2>&1; [ $? -eq 0 ] || { echo "selftest FAIL: par_chunks falsely flagged"; fail=1; }

    real=$(find src/lib.rs src/batch_oracle.rs src/sim src/ai src/validation -name '*.rs' 2>/dev/null)
    if [ -n "$real" ]; then
        scan_nested $real >/dev/null 2>&1 || { echo "selftest FAIL: real in-scope tree has false positives"; fail=1; }
    fi

    if [ "$fail" -eq 0 ]; then
        echo "selftest: ALL PASS (safe=ok, nested=caught, par_chunks=ok, real tree=clean)"
        exit 0
    else
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Default: pre-commit flow over staged files.
# ---------------------------------------------------------------------------
# Scope (#2524): lib.rs + batch_oracle.rs + sim/ + ai/ + validation/.
# batch_oracle.rs was added in #2493 (evaluate_population extracted from lib.rs);
# without it the hook is blind to nested-par_iter regressions in the very file
# that owns the population-level parallelism contract (#1065).
SCOPE_RE='lib\.rs$|batch_oracle\.rs$|sim/.*\.rs$|ai/.*\.rs$|validation/.*\.rs$'

in_scope=()
for file in "$@"; do
    if [[ ! "$file" =~ $SCOPE_RE ]]; then
        continue
    fi
    in_scope+=("$file")

    # [existing] evaluate_population-specific checks. These only fire on a file
    # whose signature is exactly `pub fn evaluate_population(` (lib.rs). Generic
    # variants like `evaluate_population<F>(` in sim/ are intentionally not
    # matched here; the general nested scan below covers them.
    if grep -q "par_iter" "$file"; then
        eval_pop_start=$(grep -n "pub fn evaluate_population(" "$file" | cut -d: -f1 | head -n1)
        if [ -n "$eval_pop_start" ]; then
            eval_pop_end=$(tail -n +"$eval_pop_start" "$file" | grep -n -m1 "^}" | head -n1 | cut -d: -f1)
            if [ -n "$eval_pop_end" ]; then
                eval_pop_end=$((eval_pop_start + eval_pop_end - 1))
                eval_pop_body=$(sed -n "${eval_pop_start},${eval_pop_end}p" "$file")
                par_iter_count=$(echo "$eval_pop_body" | grep -c "par_iter" || echo 0)
                if [ "$par_iter_count" -gt 1 ]; then
                    echo "PERF REGRESSION: $file"
                    echo "   Nested par_iter detected in evaluate_population"
                    echo "   -> Fix: use single-level population-wide parallelism only"
                    exit 1
                fi
            fi
        fi
    fi

    if grep -q "pub fn evaluate_population(" "$file"; then
        if ! grep -A 300 "fn evaluate_population" "$file" | grep -q "par_iter"; then
            echo "PERFORMANCE WARNING: $file"
            echo "   evaluate_population missing rayon parallelism"
            echo "   -> Fix: add .par_iter() for population-level parallelism"
            exit 1
        fi
    fi
done

# Second pass (#2524): nested par_iter/into_par_iter scan across all in-scope files.
if [ ${#in_scope[@]} -gt 0 ]; then
    scan_nested "${in_scope[@]}"
fi

exit 0
