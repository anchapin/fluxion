#!/usr/bin/env python3
"""
CI guard: no script under `scripts/` may write into a production source
tree (Issue #4163).

Background: `scripts/grid_search_h_si.py` and `scripts/sweep_h_ms_coeff.py`
were operator calibration helpers that grid-searched a physics constant
(`H_SI`, `h_ms_coeff`) until an ASHRAE 140 case landed inside its reference
band, then wrote the tuned value back into `src/`. Two problems, both
fatal:

  1. Tuning a constant to make an ASHRAE 140 case pass violates the
     repository's physics-integrity rule (`RULES.md`). Constants are not
     free parameters; the fix path is the underlying physics.
  2. They were actively destructive. `grid_search_h_si.py` called
     `update_h_si(3.45)` *before* capturing the originals, so
     `restore_h_si` wrote the 3.45 sweep seed back and the script still
     reported having restored the original. `sweep_h_ms_coeff.py` opened a
     hardcoded absolute path into `src/sim/` for writing. Both are now
     deleted.

The deletion alone is not the durable fix — the same helper can be
committed again. This gate rejects the class: a Python file under
`scripts/` that writes to `src/`, `fluxion-core/src/`, or `fluxion-fluid/src/`
cannot land. It is a *static* check (AST, not execution) so it cannot be
defeated by a dynamic path, and it deliberately errs toward reporting: a
finding names the file and line so the author can restructure.

Why AST rather than grep: `grid_search_h_si.py` wrote via
`Path.write_text` while `sweep_h_ms_coeff.py` wrote via `open(..., "w")`
with a string literal path. A text-level "does this mention src/sim"
heuristic also flags legitimate *read* paths (several scripts read source
files to verify them), so the check resolves call targets and
distinguishes read modes from write modes.

Read-only access is explicitly allowed and is not reported. A script that
opens a source file for reading, or that writes only under `tmp/`, is
compliant.

Scope: `scripts/**/*.py`. Shell and Rust tooling is out of scope for this
gate; it exists to close the specific hole the deleted scripts opened.

Wired into the `Scripts Test Suite` job of
`.github/workflows/scripts-tests.yml` ("Forbid scripts writing into src/",
Issue #4163). The matcher's invariants are additionally covered by
`scripts/ci/test_check_no_src_writes.py` against hermetic `tmp_path`
fixtures.

Exit codes:
    0 — no script under `scripts/` writes into a production source tree.
    1 — one or more scripts can write into `src/`.
    2 — script error (e.g. `scripts/` missing).

Usage:
    python3 scripts/check_no_src_writes.py
    python3 scripts/check_no_src_writes.py --scripts-dir scripts
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Path prefixes a script must never write into. These are the production
# physics trees; `scripts/` and `tmp/` are the legitimate write targets.
FORBIDDEN_PREFIXES = (
    "src/",
    "fluxion-core/src/",
    "fluxion-fluid/src/",
)

# `open(path, mode)` and `Path.open(mode)` write unless the mode is
# explicitly read-only. An empty/absent mode defaults to "r".
_WRITE_MODE_CHARS = set("wax+")


def _is_write_mode(mode: str | None) -> bool:
    """True when an `open()` mode string permits writing."""
    if not mode:
        return False
    return any(ch in _WRITE_MODE_CHARS for ch in mode)


def _string_value(node: ast.AST) -> str | None:
    """Return the value of a plain string literal node, else None.

    Only statically-literal paths are resolved. A `f"src/{name}.rs"` or a
    variable is deliberately not resolved: this gate reports the
    literal case, and a dynamic writer is caught by review (and, if it
    ever lands, the same gate fires on its literal call sites).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _path_candidates(
    node: ast.AST, bindings: dict[str, str] | None = None
) -> list[str]:
    """Best-effort static path strings for a call's target expression."""
    out: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, ast.JoinedStr):
        # f-string: keep the literal segments so `f"src/{x}.rs"` still
        # yields a checkable prefix.
        parts = [
            v.value
            for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
        if parts:
            out.append("".join(parts))
    elif isinstance(node, ast.Call):
        # Path("src") / "x" -> recurse into the receiver for the literal.
        for arg in node.args:
            out.extend(_path_candidates(arg, bindings))
        func = node.func
        if isinstance(func, ast.Attribute):
            out.extend(_path_candidates(func.value, bindings))
    elif isinstance(node, ast.Name):
        # A local variable bound to a literal path -- resolve it so
        # `p = Path("src/..."); p.write_text(...)` is caught.
        if bindings and node.id in bindings:
            return [bindings[node.id]]
        return []
    return out


def _normalize(path_str: str) -> str:
    """Normalize a candidate path for prefix comparison.

    An *absolute* path pointing into this checkout (e.g.
    ``/home/alex/Projects/fluxion/src/sim/x.rs``) is rewritten to its
    repo-relative tail so it compares equal to the ``src/`` prefix. That
    rewrite is applied only to absolute paths -- doing it for relative
    paths would make ``docs/src/a.md`` look like a production write.
    """
    p = path_str.replace("\\", "/").lstrip()
    while p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        for prefix in FORBIDDEN_PREFIXES:
            idx = p.find("/" + prefix)
            if idx != -1:
                return p[idx + 1 :]
    return p


def _forbidden(candidate: str) -> str | None:
    """Return the matched forbidden prefix for `candidate`, else None."""
    norm = _normalize(candidate)
    for prefix in FORBIDDEN_PREFIXES:
        if norm == prefix.rstrip("/") or norm.startswith(prefix):
            return prefix
    return None


def _mode_kwarg(call: ast.Call) -> str | None:
    """Extract the `mode=` keyword from a call, as a static string."""
    for kw in call.keywords:
        if kw.arg == "mode":
            return _string_value(kw.value)
    return None


def _literal_path_assignments(tree: ast.AST) -> dict[str, str]:
    """Map variable name -> literal path for simple constant assignments.

    The deleted `grid_search_h_si.py` bound its target as
    ``p = Path("src/sim/...")`` and then called ``p.write_text(...)``, so
    resolving only the call's inline expression would miss the real
    defect. This is a deliberate one-step dataflow approximation: only
    assignments whose right-hand side is a literal path are recorded, and
    a later reassignment overwrites the entry. It does not model scoping
    or aliasing -- a genuinely dynamic writer stays a review concern, as
    documented on ``_path_candidates``.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        candidates = _path_candidates(value, bindings)
        if not candidates:
            continue
        # `Assign` carries a list of targets; `AnnAssign` a single one.
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for tgt in targets:
            if isinstance(tgt, ast.Name) and isinstance(tgt.id, str):
                bindings[tgt.id] = candidates[0]
    return bindings


def check_script(path: Path, rel: str | None = None) -> list[str]:
    """Return write-into-src findings for one Python file."""
    rel = rel or path.name
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{rel}: unparseable Python ({exc})"]

    bindings = _literal_path_assignments(tree)
    findings: list[str] = []

    def record(lineno: int, detail: str, candidates: list[str]) -> None:
        shown = ", ".join(repr(c) for c in candidates) or "<dynamic path>"
        findings.append(
            f"{rel}:{lineno}: {detail} into a production source tree "
            f"({shown}). Calibration helpers must not write physics "
            "constants back into `src/`; tuning a constant to satisfy an "
            "ASHRAE 140 band violates RULES.md. Read-only access is fine."
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # Path.write_text(...) / Path.write_bytes(...)
        if (
            isinstance(func, ast.Attribute)
            and func.attr in {"write_text", "write_bytes"}
        ):
            targets = _path_candidates(func.value, bindings)
            hits = [t for t in targets if _forbidden(t)]
            if hits:
                record(
                    node.lineno, f"`{func.attr}()` call", hits
                )
            continue

        # builtins.open(path, mode) with a writing mode
        if isinstance(func, ast.Name) and func.id == "open":
            if not node.args:
                continue
            mode = _string_value(node.args[1]) if len(node.args) > 1 else _mode_kwarg(node)
            if not _is_write_mode(mode):
                continue
            targets = _path_candidates(node.args[0], bindings)
            hits = [t for t in targets if _forbidden(t)]
            if hits:
                record(
                    node.lineno,
                    f"`open(..., {mode!r})` call",
                    hits,
                )
            continue

        # Path.open(mode) with a writing mode: receiver is the path
        if isinstance(func, ast.Attribute) and func.attr == "open":
            mode = _mode_kwarg(node)
            if not _is_write_mode(mode):
                continue
            targets = _path_candidates(func.value, bindings)
            hits = [t for t in targets if _forbidden(t)]
            if hits:
                record(
                    node.lineno,
                    f"`Path.open({mode!r})` call",
                    hits,
                )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=SCRIPTS_DIR,
        help="Directory of operator scripts to scan (default: scripts/).",
    )
    args = parser.parse_args()

    if not args.scripts_dir.is_dir():
        print(f"ERROR: {args.scripts_dir} not found", file=sys.stderr)
        return 2

    files = sorted(args.scripts_dir.rglob("*.py"))
    findings: list[str] = []
    for path in files:
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = path.name
        findings.extend(check_script(path, rel=rel))

    if findings:
        print("Script writes into a production source tree:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s) across {len(files)} script(s). "
            "Do not commit calibration helpers that write physics constants "
            "back into `src/`. Read-only source access is allowed; write "
            "scratch output under `tmp/`.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: none of {len(files)} script(s) under {args.scripts_dir.name}/ "
        "write into `src/`."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
