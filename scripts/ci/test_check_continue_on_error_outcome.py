"""Tests for ``scripts/check_continue_on_error_outcome.py`` -- Issue #4159.

The script rejects the ``continue-on-error: true`` + downstream
``steps.<id>.outcome`` anti-pattern: on a masked step ``.outcome`` is
always ``'success'`` (it reflects the continue-on-error handling), so a
terminal "fail if X failed" re-raise testing ``.outcome`` can never fire.
The correct form is ``steps.<id>.conclusion``.

The tests below drive the matcher against hermetic ``tmp_path`` mock
repos so a detector regression surfaces here instead of on a real PR.
Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_workflow_pin.py`` (Issue #3475).

Coverage:

* ``_masked_step_ids()`` -- masked vs unmasked step classification,
* ``check_workflow()`` -- per-file findings for each planted shape
  (re-raise on ``.outcome``, on ``.conclusion``, unmasked ``.outcome``,
  trailing-comment ``continue-on-error``, ``.outcome`` inside a
  ``run:`` body),
* ``main()`` exit-code contract: 0 clean / 1 drift / 2 script error.

No test depends on the network or on the repo's real workflows; the
end-to-end against the real tree is the direct
``check_continue_on_error_outcome.py`` invocation in
``scripts-tests.yml`` (Issue #4159 acceptance criterion #b).
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_continue_on_error_outcome"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_continue_on_error_outcome.py``."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, monkeypatch, tmp_path: Path) -> Path:
    """Point the checker's module constants at a ``tmp_path`` mock repo
    and return its ``.github/workflows`` directory."""
    workflows = tmp_path / ".github" / "workflows"
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", workflows)
    return workflows


def _write_workflow(workflows: Path, name: str, text: str) -> Path:
    """Write a synthetic workflow into the mock repo's workflows dir."""
    target = workflows / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


_MASKED_OUTCOME = (
    "name: Gate\n"
    "on: push\n"
    "jobs:\n"
    "  gate:\n"
    "    steps:\n"
    "      - name: Check\n"
    "        id: check\n"
    "        continue-on-error: true\n"
    "        run: ./check.sh\n"
    "      - name: Fail if gate failed\n"
    "        if: steps.check.outcome == 'failure'\n"
    "        run: exit 1\n"
)

_MASKED_CONCLUSION = _MASKED_OUTCOME.replace(
    "steps.check.outcome == 'failure'",
    "steps.check.conclusion == 'failure'",
)

_UNMASKED_OUTCOME = _MASKED_OUTCOME.replace(
    "        continue-on-error: true\n", ""
)

_MASKED_OUTCOME_IN_RUN = (
    "name: Harness\n"
    "on: push\n"
    "jobs:\n"
    "  gate:\n"
    "    steps:\n"
    "      - name: Run harness\n"
    "        id: harness\n"
    "        run: ./harness.sh\n"
    "        continue-on-error: true  # capture exit code\n"
    "      - name: Check exit code\n"
    "        run: |\n"
    "          EXIT_CODE=${{ steps.harness.outcome == 'failure' && '1' || '0' }}\n"
    "          echo \"exit=$EXIT_CODE\"\n"
)


# ---------------------------------------------------------------------------
# _masked_step_ids() -- classification
# ---------------------------------------------------------------------------


def test_masked_step_detected(checker):
    ids = checker._masked_step_ids(_MASKED_OUTCOME.splitlines())
    assert ids == {"check"}


def test_unmasked_step_not_detected(checker):
    ids = checker._masked_step_ids(_UNMASKED_OUTCOME.splitlines())
    assert ids == set()


def test_trailing_comment_still_masked(checker):
    """``continue-on-error: true  # comment`` counts as masked."""
    ids = checker._masked_step_ids(_MASKED_OUTCOME_IN_RUN.splitlines())
    assert ids == {"harness"}


# ---------------------------------------------------------------------------
# check_workflow() -- per-file findings
# ---------------------------------------------------------------------------


def test_masked_outcome_reraise_is_flagged(checker, monkeypatch, tmp_path):
    """The Issue #4159 bug shape -- re-raise on a masked ``.outcome``."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    path = _write_workflow(workflows, "gate.yml", _MASKED_OUTCOME)
    findings = checker.check_workflow(path)
    assert len(findings) == 1
    assert "steps.check.outcome" in findings[0]
    assert "steps.check.conclusion" in findings[0]


def test_conclusion_reraise_passes(checker, monkeypatch, tmp_path):
    """The fixed shape -- ``.conclusion`` -- is clean."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    path = _write_workflow(workflows, "gate.yml", _MASKED_CONCLUSION)
    assert checker.check_workflow(path) == []


def test_unmasked_outcome_passes(checker, monkeypatch, tmp_path):
    """``.outcome`` on a step WITHOUT ``continue-on-error`` is reliable
    (e.g. mutation-testing.yml's ``steps.mutants.outcome != 'skipped'``)
    and must not be flagged."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    path = _write_workflow(workflows, "gate.yml", _UNMASKED_OUTCOME)
    assert checker.check_workflow(path) == []


def test_outcome_inside_run_body_is_flagged(checker, monkeypatch, tmp_path):
    """The detector is not limited to ``if:`` lines -- the
    ashrae_benchmark_harness.yml shape (``${{ steps.harness.outcome }}``
    inside a ``run:`` body) is the same bug."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    path = _write_workflow(workflows, "harness.yml", _MASKED_OUTCOME_IN_RUN)
    findings = checker.check_workflow(path)
    assert len(findings) == 1
    assert "steps.harness.outcome" in findings[0]


def test_unrelated_outcome_reference_passes(checker, monkeypatch, tmp_path):
    """A ``.outcome`` reference to a step id that is not masked (or not
    declared at all) is not this gate's business."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    text = _MASKED_OUTCOME.replace(
        "if: steps.check.outcome == 'failure'",
        "if: steps.other.outcome == 'failure'",
    )
    path = _write_workflow(workflows, "gate.yml", text)
    assert checker.check_workflow(path) == []


# ---------------------------------------------------------------------------
# main() -- exit-code contract
# ---------------------------------------------------------------------------


def test_main_returns_0_on_clean_tree(checker, monkeypatch, tmp_path):
    workflows = _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(workflows, "gate.yml", _MASKED_CONCLUSION)
    assert checker.main([]) == 0


def test_main_returns_1_on_anti_pattern(checker, monkeypatch, tmp_path, capsys):
    workflows = _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(workflows, "gate.yml", _MASKED_OUTCOME)
    assert checker.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_returns_2_when_workflows_dir_missing(
    checker, monkeypatch, tmp_path
):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        checker, "WORKFLOWS_DIR", tmp_path / ".github" / "workflows"
    )
    assert checker.main([]) == 2
