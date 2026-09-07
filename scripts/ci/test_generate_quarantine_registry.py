"""Tests for ``scripts/generate_quarantine_registry.py`` -- Issue #3211, #3393.

Regression guard for the quarantine registry audit. The script reads
every ``#[ignore]`` attribute under ``tests/**/*.rs`` and cross-
references it against the human-curated registry at
``tests/QUARANTINE.md``. The hermetic ``tmp_path`` fixture lets the
tests plant synthetic ``#[ignore]`` attributes and a synthetic
``QUARANTINE.md`` to exercise the orphan / ghost detection paths
without depending on the real repo's tests/ tree.

The tests pin three invariants:

1. **Orphan detection**: an ``#[ignore]`` in a synthetic test file
   that has no matching registry row is reported as ``orphans``.
2. **Ghost detection**: a registry row whose function-name substring
   appears in NO scanned ``#[ignore]`` is reported as ``ghosts``.
3. **Wildcard matching**: a registry row with ``test_dhat_*`` matches
   every actual ``test_dhat_xxx`` function under the same file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "generate_quarantine_registry"


@pytest.fixture
def audit_script(load_script, monkeypatch):
    """Freshly-loaded copy of ``scripts/generate_quarantine_registry.py``.

    The script's ``REPO_ROOT`` constant is computed at import time from
    the script's location; for hermetic ``tmp_path`` tests we redirect
    REPO_ROOT to ``tmp_path`` AFTER loading so every helper that walks
    ``REPO_ROOT / "tests" / "QUARANTINE.md"`` operates against the
    synthetic fixture.
    """
    mod = load_script(SCRIPT_NAME)
    monkeypatch.setattr(mod, "REPO_ROOT", None)  # placeholder
    return mod


@pytest.fixture
def audit_at(audit_script, tmp_path, monkeypatch):
    """Pin ``audit_script.REPO_ROOT`` at ``tmp_path`` for the test's lifetime.

    Tests that need to scan a synthetic ``tests/`` tree call this
    fixture to redirect the module-level path constants before driving
    ``scan_ignores`` / ``scan_registry`` / ``audit``.
    """
    monkeypatch.setattr(audit_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_script, "TESTS_DIR", tmp_path / "tests")
    monkeypatch.setattr(
        audit_script, "QUARANTINE_MD", tmp_path / "tests" / "QUARANTINE.md"
    )
    return audit_script


def _write_synthetic_tests(tmp_path: Path) -> Path:
    """Create a minimal ``tests/`` directory with one synthetic test file
    containing three ``#[ignore]`` attributes.

    Returns the synthetic ``tests/`` root.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "synthetic.rs").write_text(
        "//! Synthetic test file for the quarantine audit.\n"
        "\n"
        "#[test]\n"
        '#[ignore = "LIMIT-99: structural gap"]\n'
        "fn test_synthetic_quarantined_a() {\n"
        "    assert!(true);\n"
        "}\n"
        "\n"
        "#[test]\n"
        "#[ignore]\n"
        "fn test_synthetic_quarantined_b() {\n"
        "    assert!(true);\n"
        "}\n"
        "\n"
        "#[test]\n"
        '#[ignore = "diagnostic-only; run with --ignored"]\n'
        "fn test_synthetic_diagnostic_c() {\n"
        "    assert!(true);\n"
        "}\n",
        encoding="utf-8",
    )
    return tests_dir


def _write_synthetic_registry(
    quarantine_md: Path,
    rows: list[tuple[str, str]],
) -> Path:
    """Create a synthetic ``QUARANTINE.md`` with the given table rows.

    Each row is a ``(file, function)`` tuple that the script will
    extract as a registered entry.
    """
    lines = [
        "# Test Quarantine Registry",
        "",
        "Synthetic registry for the audit tests.",
        "",
        "| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |",
        "|-----------|-----------|----------------|-------------------|--------|",
    ]
    for file_cell, fn_cell in rows:
        lines.append(
            f"| `{file_cell}` | `{fn_cell}` | #9999 | Test unblocked | `pending` |"
        )
    quarantine_md.parent.mkdir(parents=True, exist_ok=True)
    quarantine_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return quarantine_md


# ---------------------------------------------------------------------------
# scan_ignores
# ---------------------------------------------------------------------------


def test_scan_ignores_finds_three_attributes(audit_at, tmp_path):
    """Three ``#[ignore]`` attrs in the synthetic file are all detected."""
    tests_dir = _write_synthetic_tests(tmp_path)
    ignores = audit_at.scan_ignores(tests_dir)
    assert len(ignores) == 3
    fn_names = sorted(i["function"] for i in ignores)
    assert fn_names == [
        "test_synthetic_diagnostic_c",
        "test_synthetic_quarantined_a",
        "test_synthetic_quarantined_b",
    ]


def test_scan_ignores_extracts_reason_string(audit_at, tmp_path):
    """``#[ignore = "reason"]`` carries the reason; bare ``#[ignore]`` is empty."""
    tests_dir = _write_synthetic_tests(tmp_path)
    ignores = audit_at.scan_ignores(tests_dir)
    by_fn = {i["function"]: i for i in ignores}
    assert by_fn["test_synthetic_quarantined_a"]["reason"] == "LIMIT-99: structural gap"
    assert by_fn["test_synthetic_quarantined_b"]["reason"] == ""
    assert by_fn["test_synthetic_diagnostic_c"]["reason"] == (
        "diagnostic-only; run with --ignored"
    )


def test_scan_ignores_skips_doc_comment_mentions(audit_at, tmp_path):
    """A ``#[ignore]`` mentioned inside a doc-comment is NOT counted."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "doc_only.rs").write_text(
        "//! Module docs that mention `#[ignore]` to describe the convention.\n"
        "/// Helper that runs with `#[ignore]` set.\n"
        "fn helper() {}\n",
        encoding="utf-8",
    )
    assert audit_at.scan_ignores(tests_dir) == []


# ---------------------------------------------------------------------------
# scan_registry
# ---------------------------------------------------------------------------


def test_scan_registry_extracts_table_rows(audit_script, tmp_path):
    """Every row in the synthetic QUARANTINE.md is parsed."""
    qmd = tmp_path / "QUARANTINE.md"
    _write_synthetic_registry(
        qmd,
        [
            ("tests/foo.rs", "test_foo"),
            ("tests/bar.rs", "test_bar_*"),
        ],
    )
    rows = audit_script.scan_registry(qmd)
    assert len(rows) == 2
    assert rows[0] == {"file": "tests/foo.rs", "function": "test_foo"}
    assert rows[1] == {"file": "tests/bar.rs", "function": "test_bar_*"}


def test_scan_registry_returns_empty_when_missing(audit_script, tmp_path):
    """A missing QUARANTINE.md returns [] (informational mode)."""
    qmd = tmp_path / "no_such_file.md"
    assert audit_script.scan_registry(qmd) == []


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def test_audit_identifies_orphan(audit_at, tmp_path):
    """An ``#[ignore]`` in code with no matching registry row is an orphan."""
    tests_dir = _write_synthetic_tests(tmp_path)
    qmd = tmp_path / "QUARANTINE.md"
    _write_synthetic_registry(
        qmd,
        # Only register A; B and C are orphans.
        [("tests/synthetic.rs", "test_synthetic_quarantined_a")],
    )
    ignores = audit_at.scan_ignores(tests_dir)
    registry = audit_at.scan_registry(qmd)
    orphans, ghosts = audit_at.audit(ignores, registry)
    orphan_fns = sorted(o["function"] for o in orphans)
    assert orphan_fns == [
        "test_synthetic_diagnostic_c",
        "test_synthetic_quarantined_b",
    ]
    assert ghosts == []


def test_audit_identifies_ghost(audit_at, tmp_path):
    """A registry row whose function appears in NO code is a ghost."""
    tests_dir = _write_synthetic_tests(tmp_path)
    qmd = tmp_path / "QUARANTINE.md"
    _write_synthetic_registry(
        qmd,
        # Register a function that doesn't exist in the synthetic file.
        [("tests/synthetic.rs", "test_nonexistent")],
    )
    ignores = audit_at.scan_ignores(tests_dir)
    registry = audit_at.scan_registry(qmd)
    orphans, ghosts = audit_at.audit(ignores, registry)
    # All 3 actual ignores are orphans (no registry match).
    assert len(orphans) == 3
    # The single registry row is a ghost.
    assert len(ghosts) == 1
    assert ghosts[0]["function"] == "test_nonexistent"


def test_audit_wildcard_matches_every_function(audit_at, tmp_path):
    """A registry row with ``test_dhat_*`` matches every actual
    ``test_dhat_xxx`` under the same file -- so they are NOT orphans."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "dhat_file.rs").write_text(
        "#[test]\n#[ignore]\nfn test_dhat_one() {}\n"
        "#[test]\n#[ignore]\nfn test_dhat_two() {}\n"
        "#[test]\n#[ignore]\nfn test_other() {}\n",
        encoding="utf-8",
    )
    qmd = tmp_path / "QUARANTINE.md"
    _write_synthetic_registry(
        qmd,
        [
            ("tests/dhat_file.rs", "test_dhat_*"),
            ("tests/dhat_file.rs", "test_other"),
        ],
    )
    ignores = audit_at.scan_ignores(tests_dir)
    registry = audit_at.scan_registry(qmd)
    orphans, ghosts = audit_at.audit(ignores, registry)
    orphan_fns = sorted(o["function"] for o in orphans)
    # test_other is registered explicitly, test_dhat_one/two match the
    # wildcard; no orphans.
    assert orphan_fns == []
    assert ghosts == []


def test_audit_in_sync_returns_no_orphans_no_ghosts(audit_at, tmp_path):
    """Identical code + registry -> no orphans, no ghosts."""
    tests_dir = _write_synthetic_tests(tmp_path)
    qmd = tmp_path / "QUARANTINE.md"
    _write_synthetic_registry(
        qmd,
        [
            ("tests/synthetic.rs", "test_synthetic_quarantined_a"),
            ("tests/synthetic.rs", "test_synthetic_quarantined_b"),
            ("tests/synthetic.rs", "test_synthetic_diagnostic_c"),
        ],
    )
    ignores = audit_at.scan_ignores(tests_dir)
    registry = audit_at.scan_registry(qmd)
    orphans, ghosts = audit_at.audit(ignores, registry)
    assert orphans == []
    assert ghosts == []


# ---------------------------------------------------------------------------
# classify_ignore
# ---------------------------------------------------------------------------


def test_classify_ignore_buckets_by_reason(audit_script):
    """The classifier uses reason keywords to bucket each ignore."""
    samples = [
        ({"file": "tests/foo.rs", "reason": "LIMIT-05: structural"}, "structural"),
        ({"file": "tests/dhat_x.rs", "reason": "Memory profiling"}, "performance"),
        (
            {"file": "tests/diagnostics/diag_x.rs", "reason": "#2536"},
            "diagnostic",
        ),
        ({"file": "tests/gpu_x.rs", "reason": "GPU only"}, "hardware"),
        ({"file": "tests/cal_x.rs", "reason": "awaiting calibration"}, "calibration"),
        ({"file": "tests/ci_x.rs", "reason": "#1577 ci broken"}, "ci-broken"),
        ({"file": "tests/m_x.rs", "reason": "manual regener"}, "manual-baseline"),
        ({"file": "tests/x.rs", "reason": ""}, "other"),
    ]
    for sample, expected_cat in samples:
        cat = audit_script.classify_ignore(sample)
        assert cat == expected_cat, f"expected {expected_cat}, got {cat} for {sample}"
