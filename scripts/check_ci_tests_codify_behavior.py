#!/usr/bin/env python3
"""
check_ci_tests_codify_behavior.py — audit ``scripts/ci/`` tests for the
"codifies broken behavior" anti-pattern (Issue #3120).

During the 2026-08-18 wave-orchestration that closed issue #3105, the test
``scripts/ci/test_check_known_issues_stale.py::test_main_returns_zero_when_marker_absent``
codified the broken ``exit 0 + WARN + silently skipped`` behavior of
``scripts/check_known_issues_stale.py`` -- the script's silent-skip on
regex mismatch was the bug. Any future hardening (PR #3112) required
also updating the test, which an automated agent would silently revert
rather than correctly update.

The anti-pattern signature is::

    def test_<...>_returns_zero_when_<something>():
        '''<ad-hoc justification>'''
        ...
        assert rc == 0
        assert "WARN" in out  # OR "skip" / "not found" / "No baseline" / etc.

A test that matches the signature is acceptable **iff** its docstring
explicitly references the policy decision that justifies the silent-skip
(e.g. ``Issue #1723`` for the file-missing skip in
``check_known_issues_stale.py``). The reference must be discoverable
from the docstring alone -- ``scripts/ci/test_<...>.py:func_name``
without the reader having to grep the script under test.

This walker emits one finding per test that matches the signature
WITHOUT a policy reference. Each finding surfaces:

* the file + function name (path:line for click-through),
* the soft-marker assertion(s) detected,
* a remediation hint (either add an issue reference to the docstring,
  or harden the script + rename the test).

Usage::

    python3 scripts/check_ci_tests_codify_behavior.py            # default: walk repo
    python3 scripts/check_ci_tests_codify_behavior.py <ci_root>  # explicit root

Exit codes:
    0 — No tests match the anti-pattern without a policy reference, OR all
        matches have explicit policy references in their docstrings.
    1 — One or more tests match the anti-pattern without a policy reference.
        Each match is printed with its file, function name, and the
        soft-marker assertion that triggered the finding.
    2 — Script error (e.g. ``scripts/ci/`` missing).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CI_ROOT = REPO_ROOT / "scripts" / "ci"

# Soft-failure markers that, combined with ``rc == 0``, suggest the
# silent-skip-on-failure anti-pattern. Lower-cased before comparison.
SOFT_MARKERS: tuple[str, ...] = (
    "warn",
    "skip",
    "not found",
    "graceful",
    "no baseline",
    "skipped",
    "informational",
)

# Regexes that, when matched in a test docstring, indicate the policy
# decision is referenced explicitly. Treated case-insensitively.
POLICY_REF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"issue\s*#\s*\d+", re.IGNORECASE),
    re.compile(r"per\s+#\d{3,}", re.IGNORECASE),
    re.compile(r"per\s+issue\s+#\d+", re.IGNORECASE),
    re.compile(r"see\s+issue\s+#\d+", re.IGNORECASE),
    re.compile(r"see\s+#\d{3,}", re.IGNORECASE),
    re.compile(r"mirrors\s+(?:the\s+)?(?:#\d{3,}|issue\s+#\d+)", re.IGNORECASE),
    re.compile(r"opt[- ]in", re.IGNORECASE),
)


def _has_rc_zero_assert(node: ast.AST) -> bool:
    """Return True iff ``node`` contains an ``assert rc == 0`` (or
    equivalent like ``checker.main() == 0`` / ``.returncode == 0``).

    Walks only ``Assert`` nodes so f-string error messages containing
    ``rc == 0`` (e.g. ``assert rc == 0, f"...{rc}..."``) count, which
    is what the audit wants -- a test whose exit-code assertion is the
    positive case for the silent-skip contract.
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.Assert):
            continue
        src = ast.unparse(child.test)
        if src == "rc == 0":
            return True
        if "returncode" in src and "== 0" in src:
            return True
        # Pattern: ``checker.main() == 0`` / ``gate.main() == 0``
        if re.search(r"\.main\(\)\s*==\s*0", src):
            return True
    return False


def _soft_marker_asserts(node: ast.AST) -> list[tuple[str, str]]:
    """Return ``[(marker, assert_src), ...]`` for every Assert inside
    ``node`` whose source contains a soft-failure marker substring
    inside a substring check against an output capture (``out`` /
    ``captured``).
    """
    findings: list[tuple[str, str]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Assert):
            continue
        src = ast.unparse(child)
        low = src.lower()
        if not any(m in low for m in SOFT_MARKERS):
            continue
        # Require the assertion to actually inspect stdout/stderr so we
        # don't flag f-string error messages that mention "WARN".
        if not (
            " in out" in low
            or " in captured.out" in low
            or " in captured.err" in low
            or " in captured" in low
        ):
            continue
        # Pull the first matching marker (deterministic ordering).
        marker = next(m for m in SOFT_MARKERS if m in low)
        findings.append((marker, src.strip()))
    return findings


def _has_policy_ref(docstring: str) -> bool:
    """Return True iff ``docstring`` contains at least one policy-ref
    regex match.
    """
    if not docstring:
        return False
    return any(p.search(docstring) for p in POLICY_REF_PATTERNS)


class Finding(TypedDict):
    """One anti-pattern finding emitted by ``audit_file``.

    The keys are stable across the audit's lifetime so callers (the
    pytest tests in ``scripts/ci/test_check_ci_tests_codify_behavior.py``
    and the future pre-commit hook) can rely on them.
    """

    path: Path
    name: str
    lineno: int
    markers: list[tuple[str, str]]
    doc_head: str


def audit_file(path: Path) -> list[Finding]:
    """Walk ``path`` (a single test_*.py file) and return one finding
    dict per test that matches the anti-pattern signature WITHOUT a
    policy reference.

    Finding dict shape::

        {
            "path": Path,
            "name": str,         # function name
            "lineno": int,       # 1-indexed line of def
            "markers": [(marker, assert_src), ...],
            "doc_head": str,     # first docstring line, for context
        }
    """
    findings: list[Finding] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        if not _has_rc_zero_assert(node):
            continue
        markers = _soft_marker_asserts(node)
        if not markers:
            continue
        docstring = ast.get_docstring(node) or ""
        if _has_policy_ref(docstring):
            continue
        doc_head = docstring.splitlines()[0] if docstring else "(no docstring)"
        findings.append(
            {
                "path": path,
                "name": node.name,
                "lineno": node.lineno,
                "markers": markers,
                "doc_head": doc_head,
            }
        )
    return findings


def walk_ci_root(ci_root: Path) -> list[Finding]:
    """Walk ``ci_root`` for ``test_*.py`` files and return all
    anti-pattern findings.
    """
    if not ci_root.is_dir():
        raise FileNotFoundError(f"CI test root not found: {ci_root}")
    findings: list[Finding] = []
    for path in sorted(ci_root.glob("test_*.py")):
        findings.extend(audit_file(path))
    return findings


def main() -> int:
    print("=== Fluxion scripts/ci/ Codify-Broken-Behavior Audit ===")
    ci_root = DEFAULT_CI_ROOT
    if len(sys.argv) > 1:
        ci_root = Path(sys.argv[1]).resolve()
    print(f"Repo: {REPO_ROOT}")
    try:
        scan_label = ci_root.relative_to(REPO_ROOT)
    except ValueError:
        # External scan root (used by tests); show absolute path.
        scan_label = ci_root
    print(f"Scanning: {scan_label}")
    print()

    findings = walk_ci_root(ci_root)
    if not findings:
        print("OK: No tests match the codify-broken-behavior anti-pattern")
        print("     without an explicit policy reference in the docstring.")
        print()
        print("Reviewed signatures:")
        print("  - rc == 0 / .returncode == 0 / *.main() == 0 exit-code assertions")
        print(f"  - soft markers in stdout/stderr: {', '.join(SOFT_MARKERS)}")
        print(
            f"  - policy-ref regexes in docstring: {[p.pattern for p in POLICY_REF_PATTERNS]}"
        )
        return 0

    print(
        f"FAIL: {len(findings)} test(s) match the anti-pattern WITHOUT a policy reference:"
    )
    print()
    for f in findings:
        try:
            relpath = f["path"].relative_to(REPO_ROOT)
        except ValueError:
            # External scan root (used by tests); fall back to absolute path.
            relpath = f["path"]
        print(f"  {relpath}:{f['lineno']}  {f['name']}")
        print(f"    doc[0]: {f['doc_head']}")
        for marker, asrc in f["markers"]:  # type: ignore[arg-type]
            print(f"    soft-marker: {marker!r}  →  {asrc}")
        print()
    print("Remediation:")
    print("  1. If the silent-skip is genuinely the right policy (e.g. file")
    print("     missing is a legitimate skip per Issue #1723):")
    print("       - rewrite the test docstring to reference the policy")
    print("         decision explicitly (e.g. ``Issue #1723``,")
    print("         ``per issue #1723``, ``see #3105``, ``mirrors the``")
    print("         pattern in another test, ``opt-in``).")
    print("  2. If the silent-skip is the bug (the more common case per")
    print("       the #3105 finding):")
    print("       - harden the script under test to exit 1 on the")
    print("         violation,")
    print("       - rename the test to ``test_main_returns_one_when_<x>``,")
    print("       - flip the asserts (``rc == 1``, ``'ERROR' in out``, etc.).")
    print()
    print("For the audit summary (per-test classification):")
    print("  docs/ci/ci-test-codify-behavior-audit.md")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, OSError, ValueError) as e:
        # Catch only the well-defined failure modes. SyntaxError /
        # NameError / etc. propagate so an actual bug in the walker
        # surfaces a real traceback instead of being swallowed.
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
