#!/usr/bin/env python3
"""
Guard against destructive tuning scripts (Issue #4163).

Some historical tuning scripts (``grid_search_h_si.py``,
``sweep_h_ms_coeff.py``) rewrote calibrated physics constants in place
under ``src/`` / ``fluxion-core/src/``.  Those scripts are deleted; this
checker fails CI if any Python script under ``scripts/`` writes to a path
under ``src/sim/`` or ``fluxion-core/src/``.

The scan is AST-based: it flags ``Path(...).write_text(...)``,
``open(..., "w")``, ``os.replace``/``shutil.move`` and similar write calls
whose target path literal (or a string built from literals) points under
``src/sim/`` or ``fluxion-core/src/``.  Read-only analysis scripts that
merely mention those directories in comments or string constants are not
flagged.

Exposed as ``scan_scripts(repo_root) -> list[Violation]`` so the pytest
harness in ``scripts/ci/test_check_no_destructive_scripts.py`` can drive
it directly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# Path prefixes that scripts must never write to.
PROTECTED_PREFIXES = ("src/sim/", "fluxion-core/src/")


@dataclass(frozen=True)
class Violation:
    script: str
    lineno: int
    detail: str


def _is_protected(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/")
    return normalized.startswith(PROTECTED_PREFIXES) or any(
        f"/{prefix}" in normalized or normalized == prefix.rstrip("/")
        for prefix in PROTECTED_PREFIXES
    )


def _literal_string(node: ast.AST) -> str | None:
    """Best-effort extraction of a string value from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    return None


class _WriteVisitor(ast.NodeVisitor):
    def __init__(self, script_name: str) -> None:
        self.script_name = script_name
        self.violations: list[Violation] = []

    def _flag(self, node: ast.AST, path_text: str, detail: str) -> None:
        if _is_protected(path_text):
            self.violations.append(
                Violation(
                    script=self.script_name,
                    lineno=getattr(node, "lineno", 0),
                    detail=f"{detail}: {path_text!r}",
                )
            )

    def _check_path_arg(self, node: ast.AST, arg: ast.AST, detail: str) -> None:
        text = _literal_string(arg)
        if text is not None:
            self._flag(node, text, detail)
            return
        # Path("src/sim/...") / "file.rs"  -> BinOp; check both sides.
        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Div):
            for side in (arg.left, arg.right):
                inner = _literal_string(side)
                if inner is not None:
                    self._flag(node, inner, detail)
                elif isinstance(side, ast.Call):
                    for sub in side.args:
                        self._check_path_arg(node, sub, detail)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # Path("src/sim/x").write_text(...) / .write_bytes(...)
        if isinstance(func, ast.Attribute) and func.attr in (
            "write_text",
            "write_bytes",
        ):
            receiver = func.value
            if isinstance(receiver, ast.Call):
                for arg in receiver.args:
                    self._check_path_arg(node, arg, f".{func.attr}() target")
            elif isinstance(receiver, ast.Name):
                # Variable receiver: only flaggable if the name itself looks
                # like a protected path constant (conservative).
                if _is_protected(receiver.id):
                    self._flag(node, receiver.id, f".{func.attr}() target")
        # open("src/sim/x", "w") / open(..., mode="w")
        elif isinstance(func, ast.Name) and func.id == "open" and node.args:
            mode = _literal_string(node.args[1]) if len(node.args) > 1 else None
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode = _literal_string(kw.value)
            if mode is not None and any(
                m in mode for m in ("w", "a", "+")
            ):
                self._check_path_arg(node, node.args[0], "open() target")
        # os.replace / os.rename / shutil.move / Path.rename targeting protected
        elif isinstance(func, ast.Attribute) and func.attr in (
            "replace",
            "rename",
            "move",
        ):
            for arg in node.args:
                self._check_path_arg(node, arg, f".{func.attr}() target")
        self.generic_visit(node)


def scan_scripts(repo_root: Path) -> list[Violation]:
    """Scan ``scripts/**/*.py`` for writes under protected source prefixes."""
    violations: list[Violation] = []
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return violations
    for script in sorted(scripts_dir.rglob("*.py")):
        try:
            tree = ast.parse(
                script.read_text(encoding="utf-8"), filename=str(script)
            )
        except (SyntaxError, UnicodeDecodeError):
            continue
        visitor = _WriteVisitor(script.name)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    violations = scan_scripts(repo_root)
    if violations:
        print("Destructive script writes detected (Issue #4163):")
        for v in violations:
            print(f"  {v.script}:{v.lineno}: {v.detail}")
        print(
            "\nScripts must not write under src/sim/ or fluxion-core/src/. "
            "Move generated outputs outside the source tree."
        )
        return 1
    print("No destructive script writes detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
