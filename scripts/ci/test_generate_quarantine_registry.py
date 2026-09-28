"""Tests for ``scripts/generate_quarantine_registry.py`` -- Issue #3211, #3393, #3443,
#4178, #4179.

Regression guard for the quarantine registry audit. The script reads
every ``#[ignore]`` attribute under the default scan roots (``tests/``,
``src/``, and every workspace member's ``src/``/``tests/``; overridable
with repeatable ``--scan-root``) and cross-references them against the
human-curated registry at ``tests/QUARANTINE.md``. The hermetic
``tmp_path`` fixture lets the tests plant synthetic ``#[ignore]``
attributes and a synthetic ``QUARANTINE.md`` to exercise the orphan /
ghost detection paths without depending on the real repo's tree.

The tests pin these invariants:

1. **Orphan detection**: an ``#[ignore]`` in a synthetic test file
   that has no matching registry row is reported as ``orphans``.
2. **Ghost detection**: a registry row whose function-name substring
   appears in NO scanned ``#[ignore]`` is reported as ``ghosts``.
3. **Wildcard matching**: a registry row with ``test_dhat_*`` matches
   every actual ``test_dhat_xxx`` function under the same file.
4. **cfg_attr(...ignore...) support**: a ``#[cfg_attr(feature =
   "gauge-solver", ignore = "...")]`` attribute is detected as a
   conditional ignore so the LIMIT-22 cohort doesn't show up as
   ghost rows. (Issue #3443.)
5. **Widened scan roots (Issue #4178)**: ``_default_scan_roots``
   covers ``src/`` and every workspace member's ``src/``/``tests/``
   parsed from the root ``Cargo.toml``; ``--scan-root`` overrides the
   defaults.
6. **Row-schema validation (Issue #4179)**: every registry row must
   carry non-empty Category / Blocking Issue / Owner / Un-Ignore
   Criteria / Status cells, and Category must be one of the frozen
   values.
7. **Key-membership ratchet (Issue #4179)**: ``--strict`` fails when
   ANY orphan/ghost key is absent from the freeze snapshot,
   regardless of the integer baseline -- the old "orphan within
   baseline exits zero" behaviour is gone; a key present in the
   freeze snapshot is not "new" and still passes.
"""

from __future__ import annotations

import json
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
    ``scan_ignores`` / ``scan_registry`` / ``audit``. ``main()``'s scan
    roots are pinned to the synthetic ``tests/`` dir so the tests stay
    hermetic (there is no ``Cargo.toml`` under ``tmp_path``).
    """
    monkeypatch.setattr(audit_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_script, "TESTS_DIR", tmp_path / "tests")
    monkeypatch.setattr(
        audit_script, "QUARANTINE_MD", tmp_path / "tests" / "QUARANTINE.md"
    )
    monkeypatch.setattr(
        audit_script, "_default_scan_roots", lambda: [tmp_path / "tests"]
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
    category: str = "structural",
    owner: str = "unassigned",
) -> Path:
    """Create a synthetic ``QUARANTINE.md`` with the given table rows.

    Each row is a ``(file, function)`` tuple that the script will
    extract as a registered entry, written in the canonical 7-column
    schema (Issue #4179).
    """
    lines = [
        "# Test Quarantine Registry",
        "",
        "Synthetic registry for the audit tests.",
        "",
        (
            "| Test File | Test Name | Category | Blocking Issue | Owner | "
            "Un-Ignore Criteria | Status |"
        ),
        (
            "|-----------|-----------|----------|----------------|-------|"
            "-------------------|--------|"
        ),
    ]
    for file_cell, fn_cell in rows:
        lines.append(
            f"| `{file_cell}` | `{fn_cell}` | `{category}` | #9999 | "
            f"`{owner}` | Test unblocked | `pending` |"
        )
    quarantine_md.parent.mkdir(parents=True, exist_ok=True)
    quarantine_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return quarantine_md


def _write_synthetic_registry_cells(
    quarantine_md: Path,
    rows: list[tuple[str, str, str, str, str, str, str]],
) -> Path:
    """Create a synthetic ``QUARANTINE.md`` with explicit 7-cell rows.

    Each row is a ``(file, function, category, issue, owner, criteria,
    status)`` tuple; empty strings are written as empty cells so the
    schema pass can flag them (Issue #4179).
    """
    lines = [
        "# Test Quarantine Registry",
        "",
        "Synthetic registry for the audit tests.",
        "",
        (
            "| Test File | Test Name | Category | Blocking Issue | Owner | "
            "Un-Ignore Criteria | Status |"
        ),
        (
            "|-----------|-----------|----------|----------------|-------|"
            "-------------------|--------|"
        ),
    ]
    for cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
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
# scan_registry (Issue #4179: full 7-column schema)
# ---------------------------------------------------------------------------


def test_scan_registry_extracts_all_schema_columns(audit_script, tmp_path):
    """Every schema column is parsed from the synthetic QUARANTINE.md."""
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
    first = rows[0]
    assert first["file"] == "tests/foo.rs"
    assert first["function"] == "test_foo"
    assert first["category"] == "structural"
    assert first["issue"] == "#9999"
    assert first["owner"] == "unassigned"
    assert first["unignore_criteria"] == "Test unblocked"
    assert first["status"] == "pending"
    assert first["line"] == 7
    assert rows[1]["function"] == "test_bar_*"


def test_scan_registry_returns_empty_when_missing(audit_script, tmp_path):
    """A missing QUARANTINE.md returns [] (informational mode)."""
    qmd = tmp_path / "no_such_file.md"
    assert audit_script.scan_registry(qmd) == []


def test_scan_registry_ignores_summary_table(audit_script, tmp_path):
    """A second table WITHOUT a 'Test File' header (the ## Summary
    table) is not parsed as registry rows."""
    qmd = tmp_path / "QUARANTINE.md"
    _write_synthetic_registry(qmd, [("tests/foo.rs", "test_foo")])
    with qmd.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n## Summary\n\n"
            "| Category | Count | Status |\n"
            "|----------|-------|--------|\n"
            "| Diagnostic tests | 1 | `pending` |\n"
        )
    rows = audit_script.scan_registry(qmd)
    assert len(rows) == 1
    assert rows[0]["function"] == "test_foo"


# ---------------------------------------------------------------------------
# validate_registry_schema (Issue #4179)
# ---------------------------------------------------------------------------


def _clean_row(**overrides):
    row = {
        "file": "tests/a.rs",
        "function": "test_a",
        "category": "structural",
        "issue": "#1",
        "owner": "unassigned",
        "unignore_criteria": "criteria",
        "status": "pending",
        "line": 7,
    }
    row.update(overrides)
    return row


def test_validate_registry_schema_clean_row(audit_script):
    """A fully-populated row with a valid Category passes."""
    assert audit_script.validate_registry_schema([_clean_row()]) == []


def test_validate_registry_schema_empty_owner(audit_script):
    """An empty Owner cell is a violation."""
    violations = audit_script.validate_registry_schema([_clean_row(owner="")])
    assert len(violations) == 1
    assert "empty Owner" in violations[0]
    assert "tests/a.rs" in violations[0]


def test_validate_registry_schema_unknown_category(audit_script):
    """A Category outside the frozen set is a violation."""
    violations = audit_script.validate_registry_schema(
        [_clean_row(category="pending-data")]
    )
    assert len(violations) == 1
    assert "unknown Category" in violations[0]
    assert "pending-data" in violations[0]


def test_validate_registry_schema_reports_every_gap(audit_script):
    """Multiple empty cells each produce their own violation."""
    violations = audit_script.validate_registry_schema(
        [_clean_row(owner="", issue="", status="")]
    )
    labels = " | ".join(violations)
    assert "empty Owner" in labels
    assert "empty Blocking Issue" in labels
    assert "empty Status" in labels


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


# ---------------------------------------------------------------------------
# scan_ignores cfg_attr(...) (Issue #3443 LIMIT-22 cohort)
# ---------------------------------------------------------------------------


def test_scan_ignores_detects_cfg_attr_ignore(audit_at, tmp_path):
    """``#[cfg_attr(feature = "gauge-solver", ignore = "...")]`` is detected
    as a conditional ``#[ignore]`` and surfaced with ``conditional=True`` so
    the LIMIT-22 gauge-build-only cohort doesn't appear as a ghost row.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "gauge_only.rs").write_text(
        '#[cfg_attr(feature = "gauge-solver", ignore = "LIMIT-22: gauge-build-only")]\n'
        "#[test]\n"
        "fn test_case_950_gauge_mass_node() {\n"
        "    assert!(true);\n"
        "}\n",
        encoding="utf-8",
    )
    ignores = audit_at.scan_ignores(tests_dir)
    assert len(ignores) == 1
    entry = ignores[0]
    assert entry["file"].endswith("gauge_only.rs")
    assert entry["function"] == "test_case_950_gauge_mass_node"
    assert entry["reason"] == "LIMIT-22: gauge-build-only"
    assert entry["conditional"] is True


def test_scan_ignores_does_not_double_count_cfg_attr_with_unconditional(
    audit_at,
    tmp_path,
):
    """A test that has BOTH ``#[ignore]`` and ``#[cfg_attr(..., ignore)]``
    is reported once (the unconditional wins; the conditional is a no-op
    duplicate in source).
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "both.rs").write_text(
        "#[ignore]\n"
        '#[cfg_attr(feature = "gauge-solver", ignore = "LIMIT-22 duplicate")]\n'
        "#[test]\n"
        "fn test_both_attrs() {\n"
        "    assert!(true);\n"
        "}\n",
        encoding="utf-8",
    )
    ignores = audit_at.scan_ignores(tests_dir)
    assert len(ignores) == 1
    assert ignores[0]["conditional"] is False
    assert ignores[0]["reason"] == ""


# ---------------------------------------------------------------------------
# scan roots (Issue #4178)
# ---------------------------------------------------------------------------


def test_default_scan_roots_covers_workspace_members(
    audit_script, tmp_path, monkeypatch
):
    """``_default_scan_roots`` parses ``[workspace] members`` from the
    root ``Cargo.toml`` and returns ``tests/``, ``src/`` plus every
    member's existing ``src/`` and ``tests/`` dirs."""
    monkeypatch.setattr(audit_script, "REPO_ROOT", tmp_path)
    (tmp_path / "Cargo.toml").write_text(
        '[workspace]\nresolver = "2"\nmembers = ["crate-a", "crates/crate-b"]\n',
        encoding="utf-8",
    )
    (tmp_path / "crate-a" / "src").mkdir(parents=True)
    (tmp_path / "crates" / "crate-b" / "tests").mkdir(parents=True)
    # crates/crate-b/src does NOT exist: it must not appear in the roots.
    roots = audit_script._default_scan_roots()
    rel = sorted(str(r.relative_to(tmp_path)) for r in roots)
    assert rel == ["crate-a/src", "crates/crate-b/tests", "src", "tests"]


def test_default_scan_roots_without_cargo_toml(audit_script, tmp_path, monkeypatch):
    """Without a root ``Cargo.toml`` (hermetic fixture) the defaults are
    just ``tests/`` and ``src/``."""
    monkeypatch.setattr(audit_script, "REPO_ROOT", tmp_path)
    roots = audit_script._default_scan_roots()
    rel = sorted(str(r.relative_to(tmp_path)) for r in roots)
    assert rel == ["src", "tests"]


def test_main_scan_root_override_replaces_defaults(
    audit_at, tmp_path, monkeypatch, capsys
):
    """``--scan-root custom`` scans ONLY the custom root: an ``#[ignore]``
    under ``tests/`` is invisible to the audit (Issue #4178)."""
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "c.rs").write_text(
        "#[test]\n#[ignore]\nfn test_custom() {}\n", encoding="utf-8"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "t.rs").write_text(
        "#[test]\n#[ignore]\nfn test_in_tests() {}\n", encoding="utf-8"
    )
    _write_synthetic_registry(
        tests_dir / "QUARANTINE.md",
        [("custom/c.rs", "test_custom")],
    )
    rc = _run_main_with_args(audit_at, monkeypatch, ["--scan-root", "custom", "--json"])
    captured = capsys.readouterr().out
    assert rc == 0  # informational mode always exits 0
    data = json.loads(captured)
    assert data["scan_roots"] == ["custom"]
    # Only the custom-root ignore was scanned; the tests/ ignore is
    # invisible, so there are no orphans either way.
    assert data["total_ignores"] == 1
    assert data["orphans"] == []


# ---------------------------------------------------------------------------
# main() ratchet (Issue #3443 key-membership, hardened by Issue #4179)
# ---------------------------------------------------------------------------


def _write_clean_synthetic_repo(tmp_path: Path) -> None:
    """Plant a synthetic repo with one in-sync test + registry row."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "sync.rs").write_text(
        '#[test]\n#[ignore = "synthetic"]\nfn test_sync() {}\n',
        encoding="utf-8",
    )
    _write_synthetic_registry(
        tests_dir / "QUARANTINE.md",
        [("tests/sync.rs", "test_sync")],
    )


def _run_main_with_args(audit_script, monkeypatch, args: list[str]) -> int:
    """Drive ``audit_script.main(argv)`` with the given argv and a clean
    synthetic repo pinned at ``tmp_path``. Returns the process exit code.
    """
    import sys as _sys

    _sys.argv = ["generate_quarantine_registry.py"] + args
    return audit_script.main()


def test_main_strict_clean_tree_exits_zero(audit_at, tmp_path, monkeypatch):
    """A clean synthetic tree (no orphans, no ghosts, schema-clean)
    exits 0 under ``--strict``. Mirrors the post-#4179 expected state
    on the real repo: empty freeze snapshots, zero baselines."""
    _write_clean_synthetic_repo(tmp_path)
    monkeypatch.setattr(audit_at, "BASELINE_ORPHANED_IGNORES", 0)
    monkeypatch.setattr(audit_at, "BASELINE_GHOST_ROWS", 0)
    assert _run_main_with_args(audit_at, monkeypatch, ["--strict"]) == 0


def test_main_strict_new_orphan_exits_one(audit_at, tmp_path, monkeypatch, capsys):
    """A synthetic tree with one new orphan exits 1 under ``--strict``
    when the freeze snapshot does NOT contain the orphan. Mirrors the
    Issue #3443/#4179 key-membership ratchet."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "orphan.rs").write_text(
        '#[test]\n#[ignore = "new orphan"]\nfn test_orphan() {}\n',
        encoding="utf-8",
    )
    _write_synthetic_registry(tests_dir / "QUARANTINE.md", [])
    monkeypatch.setattr(audit_at, "BASELINE_ORPHANED_IGNORES", 0)
    monkeypatch.setattr(audit_at, "_BASELINE_ORPHANED_IGNORES_SET", frozenset())
    monkeypatch.setattr(audit_at, "BASELINE_GHOST_ROWS", 0)
    rc = _run_main_with_args(audit_at, monkeypatch, ["--strict"])
    captured = capsys.readouterr().out
    assert rc == 1
    assert "NEW ORPHAN" in captured
    assert "tests/orphan.rs" in captured


def test_main_strict_orphan_with_baseline_headroom_still_exits_one(
    audit_at, tmp_path, monkeypatch, capsys
):
    """Issue #4179: the integer baseline no longer buys headroom. With
    ``BASELINE_ORPHANED_IGNORES = 5`` but an EMPTY freeze snapshot, one
    new orphan still exits 1 -- the freeze SET is authoritative."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "orphan.rs").write_text(
        '#[test]\n#[ignore = "new orphan"]\nfn test_orphan() {}\n',
        encoding="utf-8",
    )
    _write_synthetic_registry(tests_dir / "QUARANTINE.md", [])
    monkeypatch.setattr(audit_at, "BASELINE_ORPHANED_IGNORES", 5)
    monkeypatch.setattr(audit_at, "_BASELINE_ORPHANED_IGNORES_SET", frozenset())
    monkeypatch.setattr(audit_at, "BASELINE_GHOST_ROWS", 0)
    rc = _run_main_with_args(audit_at, monkeypatch, ["--strict"])
    captured = capsys.readouterr().out
    assert rc == 1
    assert "NEW ORPHAN" in captured


def test_main_strict_new_ghost_exits_one(audit_at, tmp_path, monkeypatch, capsys):
    """A synthetic tree with one new ghost exits 1 under ``--strict``
    when the freeze snapshot does NOT contain the ghost. Mirrors the
    Issue #3443/#4179 ghost key-membership ratchet."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "live.rs").write_text(
        "#[test]\nfn test_live_unignored() {}\n",
        encoding="utf-8",
    )
    _write_synthetic_registry(
        tests_dir / "QUARANTINE.md",
        [("tests/live.rs", "test_live_unignored")],
    )
    monkeypatch.setattr(audit_at, "BASELINE_ORPHANED_IGNORES", 0)
    monkeypatch.setattr(audit_at, "_BASELINE_ORPHANED_IGNORES_SET", frozenset())
    monkeypatch.setattr(audit_at, "BASELINE_GHOST_ROWS", 0)
    monkeypatch.setattr(audit_at, "_BASELINE_GHOST_ROWS_SET", frozenset())
    rc = _run_main_with_args(audit_at, monkeypatch, ["--strict"])
    captured = capsys.readouterr().out
    assert rc == 1
    assert "NEW GHOST" in captured
    assert "tests/live.rs" in captured


def test_main_strict_readded_freeze_key_passes(audit_at, tmp_path, monkeypatch):
    """A key present in the freeze snapshot is not 'new': re-adding a
    baseline entry (e.g. after a revert) stays green. This replaces the
    old Issue #3443 'orphan within baseline exits zero' test, whose
    count-based semantics Issue #4179 removed."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "orphan.rs").write_text(
        '#[test]\n#[ignore = "tracked orphan"]\nfn test_tracked_orphan() {}\n',
        encoding="utf-8",
    )
    _write_synthetic_registry(tests_dir / "QUARANTINE.md", [])
    monkeypatch.setattr(audit_at, "BASELINE_ORPHANED_IGNORES", 0)
    monkeypatch.setattr(
        audit_at,
        "_BASELINE_ORPHANED_IGNORES_SET",
        frozenset({("tests/orphan.rs", "test_tracked_orphan")}),
    )
    monkeypatch.setattr(audit_at, "BASELINE_GHOST_ROWS", 0)
    rc = _run_main_with_args(audit_at, monkeypatch, ["--strict"])
    assert rc == 0


def test_main_strict_empty_owner_exits_one(audit_at, tmp_path, monkeypatch, capsys):
    """A registry row with an empty Owner cell exits 1 under
    ``--strict`` (Issue #4179 row-schema gate)."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "sync.rs").write_text(
        '#[test]\n#[ignore = "synthetic"]\nfn test_sync() {}\n',
        encoding="utf-8",
    )
    _write_synthetic_registry_cells(
        tests_dir / "QUARANTINE.md",
        [
            (
                "`tests/sync.rs`",
                "`test_sync`",
                "`structural`",
                "#9999",
                "",  # empty Owner
                "Test unblocked",
                "`pending`",
            ),
        ],
    )
    monkeypatch.setattr(audit_at, "BASELINE_ORPHANED_IGNORES", 0)
    monkeypatch.setattr(audit_at, "_BASELINE_ORPHANED_IGNORES_SET", frozenset())
    monkeypatch.setattr(audit_at, "BASELINE_GHOST_ROWS", 0)
    monkeypatch.setattr(audit_at, "_BASELINE_GHOST_ROWS_SET", frozenset())
    rc = _run_main_with_args(audit_at, monkeypatch, ["--strict"])
    captured = capsys.readouterr().out
    assert rc == 1
    assert "REGISTRY SCHEMA VIOLATIONS" in captured
    assert "empty Owner" in captured


def test_main_strict_unknown_category_exits_one(
    audit_at, tmp_path, monkeypatch, capsys
):
    """A registry row with a Category outside the frozen set exits 1
    under ``--strict`` (Issue #4179 row-schema gate)."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "sync.rs").write_text(
        '#[test]\n#[ignore = "synthetic"]\nfn test_sync() {}\n',
        encoding="utf-8",
    )
    _write_synthetic_registry_cells(
        tests_dir / "QUARANTINE.md",
        [
            (
                "`tests/sync.rs`",
                "`test_sync`",
                "`bogus`",  # not in _VALID_CATEGORIES
                "#9999",
                "`unassigned`",
                "Test unblocked",
                "`pending`",
            ),
        ],
    )
    monkeypatch.setattr(audit_at, "BASELINE_ORPHANED_IGNORES", 0)
    monkeypatch.setattr(audit_at, "_BASELINE_ORPHANED_IGNORES_SET", frozenset())
    monkeypatch.setattr(audit_at, "BASELINE_GHOST_ROWS", 0)
    monkeypatch.setattr(audit_at, "_BASELINE_GHOST_ROWS_SET", frozenset())
    rc = _run_main_with_args(audit_at, monkeypatch, ["--strict"])
    captured = capsys.readouterr().out
    assert rc == 1
    assert "REGISTRY SCHEMA VIOLATIONS" in captured
    assert "unknown Category" in captured
