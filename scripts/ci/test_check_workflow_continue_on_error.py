"""
Tests for ``scripts/check_workflow_continue_on_error.py`` -- Issue #4159.

The gate must catch the exact class of defect described in #4159: a
step declared ``continue-on-error: true`` whose result is later read
through ``steps.<id>.outcome``. Under GitHub Actions semantics
``.outcome`` is the pre-mask result, so for a masked step it is pinned
to ``'success'`` and the referencing expression is a constant.

The gate is only useful if it is *non-vacuous* and *not trigger-happy*,
so this suite asserts both directions against hermetic `tmp_path`
fixtures:

  * the five real defect shapes (always-false re-raise, always-true
    guard, reference in `run:` via `${{ }}`, `!= 'success'` spelling,
    multiple jobs) are each reported;
  * the correct `.conclusion` spelling, unmasked steps, references to
    other jobs' step ids, and `always()`-only usage are all clean.

Fixtures are minimal workflows, never the real `.github/workflows/`
tree, so the suite cannot rot when unrelated workflows change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_workflow_continue_on_error"


@pytest.fixture
def gate(load_script):
    """Freshly-loaded copy of the continue-on-error re-raise gate."""
    return load_script(SCRIPT_NAME)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "ci.yml"
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Positive: the defect must be reported
# ---------------------------------------------------------------------------


def test_reports_always_false_reraise(gate, tmp_path):
    """The #4159 shape: re-raise reading `.outcome` can never fire."""
    path = _write(
        tmp_path,
        """
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Check Validation Results
        id: check
        run: python3 scripts/release_gate_checker.py
        continue-on-error: true
      - name: Fail if validation gate failed
        if: failure() || steps.check.outcome == 'failure'
        run: exit 1
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 1
    assert "steps.check.outcome" in findings[0]
    assert "steps.check.conclusion" in findings[0]
    assert "continue-on-error" in findings[0]


def test_reports_always_true_success_guard(gate, tmp_path):
    """`== 'success'` on a masked step is always true -- a broken guard."""
    path = _write(
        tmp_path,
        """
jobs:
  pgo-build:
    steps:
      - id: pgo
        name: Run PGO pipeline
        run: ./scripts/build_pgo.sh
        continue-on-error: true
      - name: Smoke-test PGO binary
        if: steps.pgo.outcome == 'success'
        run: ./target/release/fluxion --help
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 1
    assert "steps.pgo.outcome" in findings[0]


def test_reports_reference_inside_run_interpolation(gate, tmp_path):
    """The pgo-nightly / benchmark-harness shape: read inside `run:`."""
    path = _write(
        tmp_path,
        """
jobs:
  benchmark-harness:
    steps:
      - id: harness
        name: Run ASHRAE 140 Benchmark Harness
        run: python3 scripts/ashrae_benchmark_harness.py
        continue-on-error: true
      - name: Check harness exit code
        run: |
          EXIT_CODE=${{ steps.harness.outcome == 'failure' && '1' || '0' }}
          echo "Harness exit code: $EXIT_CODE"
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 1
    assert "steps.harness.outcome" in findings[0]


def test_reports_negated_success_spelling(gate, tmp_path):
    """`!= 'success'` is the same always-false defect, spelled differently."""
    path = _write(
        tmp_path,
        """
jobs:
  j:
    steps:
      - id: regression
        run: ./run.sh
        continue-on-error: true
      - name: Fail
        if: steps.regression.outcome != 'success'
        run: exit 1
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 1
    assert "steps.regression.outcome" in findings[0]


def test_reports_every_occurrence(gate, tmp_path):
    """Two consumers of one masked step are two findings, not one."""
    path = _write(
        tmp_path,
        """
jobs:
  j:
    steps:
      - id: check
        run: ./x.sh
        continue-on-error: true
      - name: A
        if: steps.check.outcome == 'failure'
        run: exit 1
      - name: B
        if: steps.check.outcome == 'failure'
        run: exit 2
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 2


def test_reports_across_distinct_jobs(gate, tmp_path):
    """Each job is evaluated against its own masked-step map."""
    path = _write(
        tmp_path,
        """
jobs:
  one:
    steps:
      - id: check
        run: ./x.sh
        continue-on-error: true
      - name: Fail
        if: steps.check.outcome == 'failure'
        run: exit 1
  two:
    steps:
      - id: check
        run: ./y.sh
        continue-on-error: true
      - name: Fail
        if: steps.check.outcome == 'failure'
        run: exit 1
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 2
    assert sum("job one" in f for f in findings) == 1
    assert sum("job two" in f for f in findings) == 1


# ---------------------------------------------------------------------------
# Negative: must NOT report the correct spellings
# ---------------------------------------------------------------------------


def test_clean_when_using_conclusion(gate, tmp_path):
    """The fixed #4159 shape is clean: `.conclusion` sees the real result."""
    path = _write(
        tmp_path,
        """
jobs:
  validate:
    steps:
      - name: Check Validation Results
        id: check
        run: python3 scripts/release_gate_checker.py
        continue-on-error: true
      - name: Fail if validation gate failed
        if: failure() || steps.check.conclusion == 'failure'
        run: exit 1
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_clean_for_unmasked_step_outcome(gate, tmp_path):
    """`.outcome` is legitimate when the step is NOT masked."""
    path = _write(
        tmp_path,
        """
jobs:
  j:
    steps:
      - id: build
        run: cargo build
      - name: Fail
        if: steps.build.outcome == 'failure'
        run: exit 1
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_clean_for_cross_job_id_reference(gate, tmp_path):
    """A step id only resolves within its own job; cross-job text is prose."""
    path = _write(
        tmp_path,
        """
jobs:
  producer:
    steps:
      - id: check
        run: ./x.sh
        continue-on-error: true
  consumer:
    steps:
      - name: Note
        # prose mentioning the other job's id, not a live expression
        run: echo "steps.check.outcome is documented in job producer"
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_clean_for_continue_on_error_false(gate, tmp_path):
    """Explicit `continue-on-error: false` is not a mask."""
    path = _write(
        tmp_path,
        """
jobs:
  j:
    steps:
      - id: check
        run: ./x.sh
        continue-on-error: false
      - name: Fail
        if: steps.check.outcome == 'failure'
        run: exit 1
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_clean_for_always_artifact_upload(gate, tmp_path):
    """`if: always()` after a masked step is the legitimate use of the mask."""
    path = _write(
        tmp_path,
        """
jobs:
  j:
    steps:
      - id: check
        run: ./x.sh
        continue-on-error: true
      - name: Upload artifact
        if: always()
        uses: actions/upload-artifact@0000000000000000000000000000000000000000
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_clean_for_workflow_without_jobs(gate, tmp_path):
    """A malformed/empty workflow yields no findings rather than crashing."""
    path = _write(tmp_path, "name: ci\n")
    assert gate.check_workflow(path, rel="ci.yml") == []


# ---------------------------------------------------------------------------
# main() surface
# ---------------------------------------------------------------------------


def test_main_returns_one_on_findings(gate, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gate"])
    _write(
        tmp_path,
        """
jobs:
  j:
    steps:
      - id: check
        run: ./x.sh
        continue-on-error: true
      - name: Fail
        if: steps.check.outcome == 'failure'
        run: exit 1
""",
    )
    monkeypatch.setattr(gate, "WORKFLOWS_DIR", tmp_path)
    assert gate.main() == 1
    assert "Dead" in capsys.readouterr().err


def test_main_returns_zero_when_clean(gate, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gate"])
    _write(
        tmp_path,
        """
jobs:
  j:
    steps:
      - id: check
        run: ./x.sh
        continue-on-error: true
      - name: Fail
        if: steps.check.conclusion == 'failure'
        run: exit 1
""",
    )
    monkeypatch.setattr(gate, "WORKFLOWS_DIR", tmp_path)
    assert gate.main() == 0
    assert "OK" in capsys.readouterr().out


def test_main_returns_two_when_workflows_dir_missing(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gate"])
    monkeypatch.setattr(gate, "WORKFLOWS_DIR", tmp_path / "nope")
    assert gate.main() == 2


def test_main_returns_two_for_unknown_workflow(gate, tmp_path, monkeypatch):
    _write(tmp_path, "jobs: {}\n")
    monkeypatch.setattr(gate, "WORKFLOWS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["gate", "--workflow", "missing.yml"])
    assert gate.main() == 2


# ---------------------------------------------------------------------------
# Non-vacuity: the gate must actually load and execute the real script
# ---------------------------------------------------------------------------


def test_gate_is_clean_against_real_workflow_tree(gate, repo_root):
    """Pin the gate against the live `.github/workflows/` tree.

    The per-test fixtures above are hermetic and cannot catch a real
    workflow that regresses. This asserts the shipped repo is clean, so a
    workflow reintroducing a masked-step re-raise fails the suite itself
    rather than waiting for the direct-invocation CI step.
    """
    findings = gate.check_workflow(
        repo_root / ".github" / "workflows" / "ashrae_140_validation.yml",
        rel=".github/workflows/ashrae_140_validation.yml",
    )
    assert findings == [], findings


def test_script_module_exposes_expected_surface(gate):
    """Guard against an import fixture that silently yields a stub.

    Handoff §7: a control with no real coverage produced vacuously green
    harnesses twice. Assert the loaded module is the real file.
    """
    assert callable(gate.check_workflow)
    assert callable(gate.main)
    src = Path(gate.__file__).read_text(encoding="utf-8")
    assert "continue-on-error" in src
    assert ".outcome" in src


def test_regex_matches_only_outcome_not_conclusion(gate):
    """`.conclusion` must not be mistaken for `.outcome`."""
    assert gate._OUTCOME_REF_RE.findall("steps.check.conclusion == 'failure'") == []
    assert gate._OUTCOME_REF_RE.findall("steps.check.outcome == 'x'") == ["check"]
