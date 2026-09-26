"""Tests for ``scripts/ci_wait.py`` — Issue #4067.

Regression guard for the ci-wait false positive: a PR head with zero
registered checks (GitHub dropped the ``pull_request`` synchronize
dispatches) must be reported as FAILURE, never as SUCCESS. The tests drive
the pure ``evaluate_rollup`` function with simulated
``gh pr view --json statusCheckRollup`` payloads — no network, no
subprocess.
"""

from __future__ import annotations

import pytest

SCRIPT_NAME = "ci_wait"
HEAD = "a3280e5b1c2d"


@pytest.fixture
def ci_wait(load_script):
    """Freshly-loaded copy of the ci-wait script."""
    return load_script(SCRIPT_NAME)


def _check_run(name, status="COMPLETED", conclusion="SUCCESS"):
    return {
        "__typename": "CheckRun",
        "name": name,
        "status": status,
        "conclusion": conclusion,
    }


def _status_context(context, state="SUCCESS"):
    return {"__typename": "StatusContext", "context": context, "state": state}


# ---------------------------------------------------------------------------
# The #4067 false positive: empty / skipped-only rollups must fail
# ---------------------------------------------------------------------------


def test_empty_rollup_is_failure_not_success(ci_wait):
    verdict = ci_wait.evaluate_rollup([], "CLEAN", HEAD)
    assert verdict.outcome == "empty"
    assert verdict.meaningful == 0


def test_skipped_only_rollup_is_failure(ci_wait):
    # Reproduces the observed #4065 case: the rollup showed only
    # "Sourcery review: skipped" while zero real checks were registered.
    entries = [_check_run("Sourcery review", conclusion="SKIPPED")]
    verdict = ci_wait.evaluate_rollup(entries, "CLEAN", HEAD)
    assert verdict.outcome == "empty"
    assert verdict.meaningful == 0
    assert "force-with-lease" in verdict.message


def test_none_rollup_is_failure(ci_wait):
    # gh can return null statusCheckRollup; main() coerces to [], but the
    # pure function must also handle the degenerate input.
    verdict = ci_wait.evaluate_rollup([], "UNKNOWN", HEAD)
    assert verdict.outcome == "empty"


def test_below_min_checks_is_failure(ci_wait):
    entries = [_check_run("CI Gates")]
    verdict = ci_wait.evaluate_rollup(entries, "CLEAN", HEAD, min_checks=3)
    assert verdict.outcome == "empty"
    assert verdict.meaningful == 1


# ---------------------------------------------------------------------------
# Normal verdicts
# ---------------------------------------------------------------------------


def test_all_success_is_success(ci_wait):
    entries = [
        _check_run("CI Gates"),
        _check_run("Docs Hygiene Gate"),
        _status_context("coverage"),
    ]
    verdict = ci_wait.evaluate_rollup(entries, "CLEAN", HEAD)
    assert verdict.outcome == "success"
    assert verdict.meaningful == 3


def test_skipped_checks_do_not_block_success(ci_wait):
    entries = [
        _check_run("CI Gates"),
        _check_run("Sourcery review", conclusion="SKIPPED"),
    ]
    verdict = ci_wait.evaluate_rollup(entries, "CLEAN", HEAD)
    assert verdict.outcome == "success"
    assert verdict.meaningful == 1


def test_failed_check_is_failure(ci_wait):
    entries = [
        _check_run("CI Gates"),
        _check_run("ASHRAE 140 Validation (GH)", conclusion="FAILURE"),
    ]
    verdict = ci_wait.evaluate_rollup(entries, "CLEAN", HEAD)
    assert verdict.outcome == "failure"
    assert verdict.failed == ["ASHRAE 140 Validation (GH)"]


def test_failed_status_context_is_failure(ci_wait):
    entries = [_status_context("codecov/patch", state="FAILURE")]
    verdict = ci_wait.evaluate_rollup(entries, "CLEAN", HEAD)
    assert verdict.outcome == "failure"


def test_pending_check_is_pending(ci_wait):
    entries = [
        _check_run("CI Gates"),
        _check_run("Heavy Suite", status="IN_PROGRESS", conclusion=None),
    ]
    verdict = ci_wait.evaluate_rollup(entries, "CLEAN", HEAD)
    assert verdict.outcome == "pending"
    assert verdict.pending == ["Heavy Suite"]


def test_failure_takes_precedence_over_pending(ci_wait):
    entries = [
        _check_run("Broken", conclusion="FAILURE"),
        _check_run("Slow", status="IN_PROGRESS", conclusion=None),
    ]
    verdict = ci_wait.evaluate_rollup(entries, "CLEAN", HEAD)
    assert verdict.outcome == "failure"


def test_unknown_merge_state_warns_but_does_not_fail(ci_wait):
    entries = [_check_run("CI Gates")]
    verdict = ci_wait.evaluate_rollup(entries, "UNKNOWN", HEAD)
    assert verdict.outcome == "success"
    assert any("UNKNOWN" in w for w in verdict.warnings)


# ---------------------------------------------------------------------------
# CLI exit-code contract
# ---------------------------------------------------------------------------


def test_main_empty_rollup_exits_1(ci_wait, monkeypatch, capsys):
    monkeypatch.setattr(
        ci_wait,
        "fetch_pr_data",
        lambda pr: {
            "number": 4065,
            "headRefOid": HEAD,
            "mergeStateStatus": "UNKNOWN",
            "statusCheckRollup": [_check_run("Sourcery review", conclusion="SKIPPED")],
        },
    )
    rc = ci_wait.main(["4065", "--once"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAILURE (no checks registered)" in out


def test_main_success_exits_0(ci_wait, monkeypatch, capsys):
    monkeypatch.setattr(
        ci_wait,
        "fetch_pr_data",
        lambda pr: {
            "number": 4065,
            "headRefOid": HEAD,
            "mergeStateStatus": "CLEAN",
            "statusCheckRollup": [_check_run("CI Gates")],
        },
    )
    rc = ci_wait.main(["4065", "--once"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUCCESS: All checks PASSED" in out
