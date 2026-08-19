"""
Tests for ``scripts/check_ci_tests_codify_behavior.py`` -- Issue #3120.

Verifies the audit walker:

* flags ``test_*_returns_zero_when_*`` tests that assert both
  ``rc == 0`` and a soft-failure marker in stdout/stderr WITHOUT an
  explicit policy reference in the docstring,
* leaves policy-referenced silent-skip tests alone (e.g. the
  ``Issue #1723`` file-missing skip in
  ``test_check_known_issues_stale.py``),
* leaves clean-state ``rc == 0`` tests alone (no soft marker in output).

Each test plants a synthetic ``tmp_path`` ``test_*.py`` file with the
exact anti-pattern shape so the walker is exercised in isolation -- no
edit to real repo files.

Mirrors the ``load_script`` fixture pattern from ``conftest.py``: the
checker module is freshly loaded so per-test ``main()`` invocations see
the synthetic CI root via the ``sys.argv[1]`` override.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_ci_tests_codify_behavior"


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the codify-behavior checker."""
    return load_script(SCRIPT_NAME)


def _write_test_file(tmp_path: Path, body: str, name: str = "test_fixture.py") -> Path:
    """Write ``body`` to ``tmp_path / name`` and return the path."""
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _run_main(checker, ci_root: Path, capsys: pytest.CaptureFixture[str]) -> int:
    """Invoke ``checker.main()`` against ``ci_root`` and return its exit code.

    Mirrors the ``sys.argv`` scrub pattern used in
    ``test_check_root_hygiene.py`` and ``test_check_audit_ignores_fresh.py``.
    """
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, str(ci_root)]
    try:
        rc = checker.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved
    return rc


# ---------------------------------------------------------------------------
# audit_file -- per-file detection
# ---------------------------------------------------------------------------


def test_audit_file_flags_anti_pattern_without_policy_ref(checker, tmp_path):
    """A test that asserts ``rc == 0`` and a soft marker in stdout
    without referencing an issue in its docstring MUST be flagged.
    """
    body = '''\
"""Fixture: anti-pattern example (no policy reference)."""


def test_main_returns_zero_when_something_skipped(capsys):
    """Ad-hoc justification: WARN is fine."""
    rc = 0
    print("WARN: skipping due to missing config")
    out = capsys.readouterr().out
    assert rc == 0
    assert "WARN" in out
'''
    p = _write_test_file(tmp_path, body)
    findings = checker.audit_file(p)
    assert len(findings) == 1, f"expected 1 finding, got {len(findings)}: {findings}"
    f = findings[0]
    assert f["name"] == "test_main_returns_zero_when_something_skipped"
    assert f["path"] == p
    assert f["lineno"] > 0
    markers = f["markers"]
    assert any(m == "warn" for m, _ in markers), markers
    assert f["doc_head"] == "Ad-hoc justification: WARN is fine."


def test_audit_file_passes_policy_referenced_skip(checker, tmp_path):
    """A test that asserts ``rc == 0`` and a soft marker in stdout
    WITH ``Issue #1723`` referenced in the docstring MUST be ignored
    (mirrors ``test_main_returns_zero_when_file_missing`` in
    ``test_check_known_issues_stale.py``).
    """
    body = '''\
"""Fixture: legitimate file-missing skip per Issue #1723."""


def test_main_returns_zero_when_file_missing(capsys):
    """Missing KNOWN_ISSUES.md -> exit 0 with skip notice.

    Issue #1723 explicitly says: "If the file doesn't exist, skip the
    check (not a failure)".
    """
    print("file not found — skipping")
    out = capsys.readouterr().out
    assert "not found" in out.lower()
'''
    p = _write_test_file(tmp_path, body)
    findings = checker.audit_file(p)
    assert findings == [], f"expected no findings, got {findings}"


def test_audit_file_passes_clean_state_test(checker, tmp_path):
    """A test that asserts ``rc == 0`` with NO soft marker in stdout
    MUST be ignored (it's a clean-state test, not the anti-pattern).
    """
    body = '''\
"""Fixture: clean-state test (no soft marker)."""


def test_main_returns_zero_when_clean(capsys):
    """Clean state -> exit 0 with OK banner."""
    print("OK: clean state")
    out = capsys.readouterr().out
    assert "OK" in out
    assert "clean" in out.lower()
'''
    p = _write_test_file(tmp_path, body)
    findings = checker.audit_file(p)
    assert findings == [], f"expected no findings, got {findings}"


def test_audit_file_passes_when_policy_ref_uses_per_issue_form(checker, tmp_path):
    """``per issue #NNNN`` (without the ``Issue`` prefix) is also a
    valid policy reference.
    """
    body = '''\
"""Fixture: 'per issue' form is also a valid policy reference."""


def test_main_returns_zero_when_marker_absent(capsys):
    """File exists but no marker -> per issue #3105 the gate must BLOCK.

    (Negative case: this test would normally be renamed to
    test_main_returns_one_when_marker_absent. We keep rc==0 here
    only to exercise the policy-ref detection logic.)
    """
    print("ERROR: marker absent")
    out = capsys.readouterr().out
    assert "ERROR" in out
'''
    p = _write_test_file(tmp_path, body)
    findings = checker.audit_file(p)
    assert findings == [], f"expected no findings, got {findings}"


def test_audit_file_flags_baseline_missing_without_policy(checker, tmp_path):
    """``No baseline`` soft marker WITHOUT a policy ref MUST be flagged
    (mirrors the pre-#3120 ``test_main_returns_zero_when_no_baseline_on_pr``
    that triggered the audit).
    """
    body = '''\
"""Fixture: no-baseline silent skip without policy reference."""


def test_main_returns_zero_when_no_baseline_on_pr(capsys):
    """--check on a PR with no baseline -> exit 0 (graceful skip)."""
    rc = 0
    print("No baseline found")
    out = capsys.readouterr().out
    assert rc == 0
    assert "No baseline" in out or "baseline" in out.lower()
'''
    p = _write_test_file(tmp_path, body)
    findings = checker.audit_file(p)
    assert len(findings) == 1, f"expected 1 finding, got {findings}"
    f = findings[0]
    assert f["name"] == "test_main_returns_zero_when_no_baseline_on_pr"
    markers = f["markers"]
    assert any(m == "no baseline" for m, _ in markers), markers


def test_audit_file_handles_syntax_error_gracefully(checker, tmp_path):
    """A file that fails to parse must not crash the walker."""
    p = _write_test_file(tmp_path, "def test_foo(:\n    pass\n", name="test_broken.py")
    findings = checker.audit_file(p)
    assert findings == []


def test_audit_file_handles_no_test_functions(checker, tmp_path):
    """A file with no test_* functions returns an empty findings list."""
    p = _write_test_file(tmp_path, "x = 1\ny = 2\n")
    findings = checker.audit_file(p)
    assert findings == []


# ---------------------------------------------------------------------------
# walk_ci_root -- aggregation across multiple files
# ---------------------------------------------------------------------------


def test_walk_ci_root_finds_anti_pattern(checker, tmp_path):
    """``walk_ci_root`` aggregates findings across multiple files in a
    synthetic CI root.
    """
    clean_body = '''\
"""Clean fixture."""


def test_main_returns_zero_when_clean(capsys):
    """Clean state."""
    rc = 0
    print("OK")
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
'''
    bad_body = '''\
"""Bad fixture (no policy ref)."""


def test_main_returns_zero_when_skip_notice(capsys):
    """Skip notice without policy."""
    rc = 0
    print("WARN: skipped")
    out = capsys.readouterr().out
    assert rc == 0
    assert "skip" in out.lower()
'''
    _write_test_file(tmp_path, clean_body, name="test_clean.py")
    p2 = _write_test_file(tmp_path, bad_body, name="test_bad.py")
    findings = checker.walk_ci_root(tmp_path)
    assert len(findings) == 1
    assert findings[0]["path"] == p2
    assert findings[0]["name"] == "test_main_returns_zero_when_skip_notice"


def test_walk_ci_root_missing_dir_raises(checker):
    """``walk_ci_root`` raises ``FileNotFoundError`` for a missing root
    so ``main()`` can translate that into exit code 2.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            checker.walk_ci_root(missing)


# ---------------------------------------------------------------------------
# main() -- end-to-end
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_real_repo(checker, capsys):
    """End-to-end: walking the real ``scripts/ci/`` MUST exit 0
    (per the Issue #3120 audit, all matches have explicit policy refs).
    """
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME]
    try:
        rc = checker.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved
    out = capsys.readouterr().out
    assert rc == 0, f"expected exit 0 on real repo, got {rc}\noutput:\n{out}"
    assert "OK" in out or "No tests match" in out


def test_main_returns_zero_when_explicit_root_is_clean(checker, tmp_path, capsys):
    """End-to-end: a synthetic clean CI root MUST exit 0."""
    body = '''\
"""Clean fixture."""


def test_main_returns_zero_when_clean(capsys):
    """Clean state."""
    print("OK")
    out = capsys.readouterr().out
    assert "OK" in out
'''
    _write_test_file(tmp_path, body, name="test_clean.py")
    rc = _run_main(checker, tmp_path, capsys)
    out = capsys.readouterr().out
    assert rc == 0, f"expected exit 0, got {rc}\noutput:\n{out}"


def test_main_returns_one_when_root_has_unreferenced_skip(checker, tmp_path, capsys):
    """End-to-end: a synthetic root with an unreferenced silent-skip
    MUST exit 1 and surface the file:line finding.
    """
    body = '''\
"""Bad fixture."""


def test_main_returns_zero_when_something_skipped(capsys):
    """Skip without policy reference."""
    rc = 0
    print("WARN: skipping")
    out = capsys.readouterr().out
    assert rc == 0
    assert "WARN" in out
'''
    p = _write_test_file(tmp_path, body, name="test_bad.py")
    rc = _run_main(checker, tmp_path, capsys)
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert str(p) in out or p.name in out
    assert "test_main_returns_zero_when_something_skipped" in out


# ---------------------------------------------------------------------------
# Module-level constants pin
# ---------------------------------------------------------------------------


def test_soft_markers_constant_is_complete(checker):
    """The SOFT_MARKERS tuple covers the markers referenced in the
    Issue #3120 audit body (WARN / skip / not found / graceful / no
    baseline / Skipped / INFORMATIONAL -- the latter only appears as
    the WASM opt-in pattern).
    """
    expected = {
        "warn",
        "skip",
        "not found",
        "graceful",
        "no baseline",
        "skipped",
        "informational",
    }
    actual = set(checker.SOFT_MARKERS)
    missing = expected - actual
    assert not missing, f"SOFT_MARKERS is missing: {missing}"
    assert len(actual) == len(expected), (
        f"SOFT_MARKERS has unexpected extras: {actual - expected}"
    )


def test_policy_ref_patterns_accept_real_references(checker):
    """Each Issue #3120 audit docstring's policy reference MUST match
    at least one regex in POLICY_REF_PATTERNS.
    """
    cases = {
        "issue_1723_explicit": "Issue #1723 explicitly says: skip the check",
        "per_issue_no_capital": "per issue #1723 the gate must block",
        "per_hash_form": "per #1723 the gate must block",
        "see_issue_form": "see issue #1723 for rationale",
        "see_hash_form": "see #3105 for the fix",
        "mirrors_form": "mirrors the Issue #1723 file-missing pattern",
        "mirrors_hash_form": "mirrors #1723 file-missing pattern",
        "opt_in_form": "documented opt-in case",
        "opt-in_hyphen": "documented opt-in case",
    }
    for label, doc in cases.items():
        assert any(p.search(doc) for p in checker.POLICY_REF_PATTERNS), (
            f"{label}: no policy ref pattern matched for {doc!r}"
        )


def test_policy_ref_patterns_reject_unrelated_text(checker):
    """Each regex MUST NOT match arbitrary unrelated text -- otherwise
    the audit would never flag anything.
    """
    negatives = [
        "This test exits 0 when the script returns PASS.",
        "No assertions are made about the exit code.",
        "Clean state: the ignore list documents an unsatisfied REMOVE condition.",
        "Fixture for check_ci_tests_codify_behavior.py anti-pattern detection.",
    ]
    for doc in negatives:
        # Only the issue- and hash-numbered patterns must not match;
        # ``opt[- ]in`` may legitimately match unrelated text and the
        # audit only flags tests that ALSO have a soft marker in stdout,
        # which prevents over-matching in practice.
        issue_match = any(
            p.search(doc)
            for p in checker.POLICY_REF_PATTERNS
            if "issue" in p.pattern or "\\d" in p.pattern
        )
        assert not issue_match, f"unexpected issue-ref match for {doc!r}"


def test_default_ci_root_points_at_repo_scripts_ci(checker, repo_root):
    """``DEFAULT_CI_ROOT`` MUST point at ``<repo>/scripts/ci`` so the
    walker covers the same files ``scripts/ci/conftest.py`` exercises.
    """
    assert checker.DEFAULT_CI_ROOT == repo_root / "scripts" / "ci"
    assert checker.DEFAULT_CI_ROOT.is_dir(), (
        f"DEFAULT_CI_ROOT does not exist: {checker.DEFAULT_CI_ROOT}"
    )
