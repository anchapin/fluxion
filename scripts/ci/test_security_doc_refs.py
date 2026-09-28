"""``docs/SECURITY.md`` backticked ``fn <name>`` reference check — Issue #4201.

``docs/SECURITY.md``'s production deploy checklist names regression tests as
the guards for its hardening controls (e.g. ``tracelayer_does_not_log_credentials``
for the TraceLayer credential-redaction control, Issue #2504). Issue #4201 found
the named test in no ``.rs`` file while the doc claimed coverage — and no
``scripts/check_*.py`` gate referenced the doc's backticked identifiers, so the
stale claim could not be detected automatically.

This case closes that gap: it scans ``docs/SECURITY.md`` for backticked
``fn <name>`` identifiers (plus backticked identifiers named as tests, e.g.
"Regression test `some_test`") and asserts each resolves to a real Rust
``fn <name>`` definition somewhere under the repo. It is a direct repo-relative
scan — read-only over docs and sources — so no ``load_script`` gate-mocking is
needed for the live-repo assertion; the synthetic-tree cases use the standard
``tmp_path`` fixture to prove the scanner fails closed on a dangling reference.

Collection note (Issue #4201 follow-up): this file is picked up automatically
by the ``python3 -m pytest scripts/`` step in
``.github/workflows/scripts-tests.yml`` (``scripts/pytest.ini::testpaths``),
so no workflow edit was made or is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_RELPATH = Path("docs/SECURITY.md")

# Backticked ``fn <name>`` identifiers, and backticked identifiers introduced
# as a test name ("test `foo`", "Regression test `foo`") — the phrasing the
# SECURITY.md checklist uses for its regression-test references.
_REF_PATTERNS = (
    re.compile(r"`fn\s+([A-Za-z_][A-Za-z0-9_]*)`"),
    re.compile(r"(?:[Rr]egression\s+)?[Tt]est\s+`([A-Za-z_][A-Za-z0-9_]*)`"),
)

# Directories never worth scanning for Rust definitions.
_SKIP_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv"}


def find_doc_test_refs(doc_path: Path) -> list[str]:
    """Return the sorted unique test identifiers referenced in a doc file."""
    text = doc_path.read_text(encoding="utf-8")
    refs: set[str] = set()
    for pattern in _REF_PATTERNS:
        refs.update(pattern.findall(text))
    return sorted(refs)


def find_rust_fn_defs(root: Path, name: str) -> list[Path]:
    """Return ``.rs`` files under ``root`` defining ``fn <name>``."""
    fn_pat = re.compile(r"\bfn\s+" + re.escape(name) + r"\b")
    hits: list[Path] = []
    for rs in root.rglob("*.rs"):
        if any(part in _SKIP_DIRS for part in rs.parts):
            continue
        try:
            if fn_pat.search(rs.read_text(encoding="utf-8", errors="replace")):
                hits.append(rs)
        except OSError:
            continue
    return hits


def unresolved_doc_test_refs(repo_root: Path, doc_relpath: Path = DOC_RELPATH) -> list[str]:
    """Return referenced test names with no ``fn <name>`` definition in the repo."""
    doc_path = repo_root / doc_relpath
    return [
        name
        for name in find_doc_test_refs(doc_path)
        if not find_rust_fn_defs(repo_root, name)
    ]


# ---------------------------------------------------------------------------
# Live-repo regression: the doc must not name a test that does not exist.
# ---------------------------------------------------------------------------


def test_live_security_doc_refs_all_resolve(repo_root):
    """Every backticked test reference in ``docs/SECURITY.md`` must resolve.

    This is the Issue #4201 regression: the checklist named
    ``tracelayer_does_not_log_credentials`` while no such ``fn`` existed.
    """
    assert unresolved_doc_test_refs(repo_root) == [], (
        "docs/SECURITY.md references test(s) with no `fn` definition: "
        + ", ".join(unresolved_doc_test_refs(repo_root))
    )


# ---------------------------------------------------------------------------
# Hermetic cases: prove the scanner fails closed and passes open.
# ---------------------------------------------------------------------------


def _write_tree(root: Path, doc_text: str, rs_text: str) -> Path:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "SECURITY.md").write_text(doc_text, encoding="utf-8")
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "lib.rs").write_text(rs_text, encoding="utf-8")
    return root


def test_scanner_catches_dangling_fn_reference(tmp_path):
    root = _write_tree(
        tmp_path,
        "Regression control guarded by `fn does_not_exist_anywhere`.\n",
        "#[test]\nfn some_other_test() {}\n",
    )
    assert unresolved_doc_test_refs(root) == ["does_not_exist_anywhere"]


def test_scanner_catches_dangling_regression_test_reference(tmp_path):
    root = _write_tree(
        tmp_path,
        "Regression test `missing_test_xyz` asserts the control holds.\n",
        "#[test]\nfn some_other_test() {}\n",
    )
    assert unresolved_doc_test_refs(root) == ["missing_test_xyz"]


def test_scanner_resolves_real_fn_reference(tmp_path):
    root = _write_tree(
        tmp_path,
        "Regression test `my_real_test` guards the control.\n",
        "#[test]\nfn my_real_test() {}\n",
    )
    assert unresolved_doc_test_refs(root) == []
    assert find_rust_fn_defs(root, "my_real_test")
