"""Tests for ``scripts/check_workflow_dup_keys.py`` -- Issue #4068.

The script rejects duplicate YAML mapping keys in
``.github/workflows/*.yml``. PR #4018's phase-gating mechanically
inserted ``needs: precheck`` / ``if: needs.precheck...`` after each
job's ``name:`` line, creating duplicate ``needs:`` / ``if:`` keys in
four jobs across ``fast_math_check.yml``, ``ashrae_validation.yml``,
and ``performance_dashboard.yml``. GitHub's parser rejects the whole
file at startup ("This run likely failed because of a workflow file
issue", zero jobs, zero logs) while stock PyYAML silently keeps the
last duplicate -- which is why every local gate passed.

The tests below drive ``check_workflow()`` against hermetic
``tmp_path`` fixture workflows (planted duplicates at job level and
nested level, plus clean controls) so a detector regression surfaces
here instead of on a real PR. Mirrors the ``load_script`` +
``tmp_path`` mock-repo pattern from ``test_check_workflow_pin.py``
(Issue #3475).

Coverage:

* ``check_workflow()`` -- duplicate ``needs:`` / ``if:`` at job level
  (the exact #4068 shape), nested duplicates, triple definitions,
  clean controls (sibling mappings reusing a key, ``on:`` boolean key),
* ``check_all()`` -- multi-file aggregation via redirected
  ``WORKFLOWS_DIR``,
* ``main()`` exit-code contract: 0 clean / 1 drift.

No test depends on the network or on the repo's real workflows; the
end-to-end against the real tree is the direct
``check_workflow_dup_keys.py`` invocation in ``scripts-tests.yml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_workflow_dup_keys"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_workflow_dup_keys.py``."""
    return load_script(SCRIPT_NAME)


def _write_workflow(tmp_path: Path, name: str, text: str) -> Path:
    """Write a synthetic workflow into the mock repo's workflows dir."""
    target = tmp_path / ".github" / "workflows" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _findings_for(checker, tmp_path, text, name: str = "ci.yml") -> list[str]:
    """``check_workflow()`` findings for a single synthetic workflow."""
    return checker.check_workflow(_write_workflow(tmp_path, name, text))


# ---------------------------------------------------------------------------
# check_workflow() -- planted duplicates
# ---------------------------------------------------------------------------

DUP_NEEDS = """\
name: Demo
on: push
jobs:
  compare:
    name: Compare
    needs: precheck
    if: needs.precheck.outputs.should_run == 'true'
    needs: [probe-default, probe-fastmath]
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"""

DUP_IF = """\
jobs:
  bench:
    name: Bench
    needs: precheck
    if: needs.precheck.outputs.should_run == 'true'
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - run: echo hi
"""

DUP_NESTED = """\
jobs:
  b:
    runs-on: ubuntu-latest
    steps:
      - name: s
        with:
          token: abc
          token: def
"""

TRIPLE_DUP = """\
jobs:
  b:
    runs-on: ubuntu-latest
    runs-on: windows-latest
    runs-on: macos-latest
    steps:
      - run: echo hi
"""

CLEAN = """\
name: Demo
on: push
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
  b:
    needs: [a]
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
"""


def test_dup_needs_job_level(checker, tmp_path):
    findings = _findings_for(checker, tmp_path, DUP_NEEDS)
    assert len(findings) == 1
    assert "duplicate key 'needs'" in findings[0]
    assert "line 6" in findings[0] and "line 8" in findings[0]


def test_dup_if_job_level(checker, tmp_path):
    findings = _findings_for(checker, tmp_path, DUP_IF)
    assert len(findings) == 1
    assert "duplicate key 'if'" in findings[0]


def test_dup_nested_step_level(checker, tmp_path):
    findings = _findings_for(checker, tmp_path, DUP_NESTED)
    assert len(findings) == 1
    assert "duplicate key 'token'" in findings[0]


def test_triple_definition_reports_both_redfines(checker, tmp_path):
    findings = _findings_for(checker, tmp_path, TRIPLE_DUP)
    assert len(findings) == 2
    assert all("duplicate key 'runs-on'" in f for f in findings)


def test_clean_workflow_no_findings(checker, tmp_path):
    assert _findings_for(checker, tmp_path, CLEAN) == []


def test_on_boolean_key_no_crash(checker, tmp_path):
    # YAML 1.1 parses ``on:`` as boolean True -- the detector must not
    # crash on non-string keys.
    assert _findings_for(checker, tmp_path, CLEAN) == []


def test_malformed_yaml_reported(checker, tmp_path):
    findings = _findings_for(checker, tmp_path, "jobs:\n  a: [unclosed\n")
    assert len(findings) == 1
    assert "YAML parse error" in findings[0]


# ---------------------------------------------------------------------------
# check_all() / main() -- exit-code contract
# ---------------------------------------------------------------------------


def test_check_all_aggregates_across_files(checker, tmp_path, monkeypatch):
    workflows = tmp_path / ".github" / "workflows"
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", workflows)
    _write_workflow(tmp_path, "a.yml", DUP_NEEDS)
    _write_workflow(tmp_path, "b.yml", CLEAN)
    _write_workflow(tmp_path, "c.yml", DUP_IF)
    findings = checker.check_all(checker.WORKFLOWS_DIR)
    assert len(findings) == 2
    assert any("a.yml" in f for f in findings)
    assert any("c.yml" in f for f in findings)


def test_main_zero_on_clean_tree(checker, tmp_path, monkeypatch):
    workflows = tmp_path / ".github" / "workflows"
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", workflows)
    _write_workflow(tmp_path, "a.yml", CLEAN)
    assert checker.main([]) == 0


def test_main_one_on_drift(checker, tmp_path, monkeypatch, capsys):
    workflows = tmp_path / ".github" / "workflows"
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", workflows)
    _write_workflow(tmp_path, "a.yml", DUP_NEEDS)
    assert checker.main([]) == 1
    assert "duplicate key 'needs'" in capsys.readouterr().out
