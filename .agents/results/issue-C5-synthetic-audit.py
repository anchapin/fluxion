#!/usr/bin/env python3
"""Walk the AST of ``tools/train_surrogate.py`` and print every callsite of
``generate_synthetic_thermal_data`` along with its disposition.

Companion to ``.agents/results/issue-C5-synthetic-audit.md`` (Issue #1338).
Invoked from the verification path::

    python .agents/results/issue-C5-synthetic-audit.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TARGET = REPO_ROOT / "tools" / "train_surrogate.py"
TAG = "synthetic-only benchmark path — NOT for production models"


def disposition_for(node: ast.Call, src_lines: list[str]) -> str:
    """Infer the callsite disposition from surrounding context."""
    window_above = "\n".join(src_lines[max(0, node.lineno - 12): node.lineno])
    if TAG in window_above:
        return "benchmark-only (annotated)"
    return "UNKNOWN — missing annotation"


def main() -> int:
    src = TARGET.read_text()
    tree = ast.parse(src, filename=str(TARGET))
    lines = src.splitlines()

    print(f"AST scan: {TARGET.relative_to(REPO_ROOT)}")
    print("=" * 72)

    # 1. Function definition.
    defs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "generate_synthetic_thermal_data"
    ]
    for n in defs:
        win = "\n".join(lines[max(0, n.lineno - 1): n.lineno + 30])
        has_tag = TAG in win
        print(f"  def  line {n.lineno:>4}  generate_synthetic_thermal_data")
        print(f"        docstring carries tag? {'YES' if has_tag else 'NO'}")

    # 2. Call sites.
    calls: list[ast.Call] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name == "generate_synthetic_thermal_data":
                calls.append(n)

    print(f"\nFound {len(calls)} call site(s):")
    if not calls:
        print("  (none)")
        return 0
    rc = 0
    for c in calls:
        d = disposition_for(c, lines)
        print(f"  call line {c.lineno:>4}  generate_synthetic_thermal_data(...)  -> {d}")
        if d == "UNKNOWN — missing annotation":
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())