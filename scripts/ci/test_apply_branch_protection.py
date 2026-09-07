"""Tests for ``scripts/apply_branch_protection.py`` -- Issue #3386.

Regression guard for the apply-side companion of
``scripts/check_required_checks_sync.py``. Mirrors the ``load_script`` +
``tmp_path`` mock-repo pattern from ``test_check_required_checks_sync.py``
and ``test_check_known_issues_stale.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``REPO_ROOT`` constant at a synthetic
  ``tmp_path`` fixture that contains a minimal but realistic
  ``release_gates.yaml``,
* drive ``build_put_payload``, ``compute_diff``, ``diff_has_changes``
  through clean and planted-drift scenarios.

The script talks to ``gh api`` only inside the write path; the tests
exercise the pure-function surface (``build_put_payload``,
``compute_diff``, ``diff_has_changes``) which is the same surface the
operator inspects in ``--dry-run --json`` mode.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_NAME = "apply_branch_protection"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def applier(load_script):
    """Freshly-loaded copy of ``scripts/apply_branch_protection.py``.

    The script does ``from scripts.check_required_checks_sync import ...``
    at module-import time, so we register that dependency in
    ``sys.modules`` BEFORE loading the applier. The dep already lives in
    ``sys.modules`` once ``check_required_checks_sync`` has been loaded
    elsewhere in the test session, but pinning it here makes the test
    self-contained.
    """
    import sys

    dep_name = "check_required_checks_sync"
    if dep_name not in sys.modules:
        dep_path = Path(__file__).resolve().parents[1] / f"{dep_name}.py"
        dep_spec = importlib.util.spec_from_file_location(dep_name, dep_path)
        dep_mod = importlib.util.module_from_spec(dep_spec)
        sys.modules[dep_name] = dep_mod
        assert dep_spec.loader is not None
        dep_spec.loader.exec_module(dep_mod)

    return load_script(SCRIPT_NAME)


def _write_release_gates(tmp_path: Path, contexts: list[str]) -> Path:
    """Write a synthetic ``release_gates.yaml`` containing a single
    ``ci.required_checks`` list. Returns the path.

    The YAML is built by hand rather than via ``yaml.dump`` so the test
    surfaces any parser-regression in ``check_required_checks_sync.load_release_gates``
    (which uses ``yaml.safe_load``). Round-tripping through
    ``yaml.dump`` would hide a parser regression by re-serializing the
    list element.
    """
    lines = ["ci:", "  required_checks:"]
    for c in contexts:
        lines.append(f'    - "{c}"')
    lines.append("  workflow_index: []")
    target = tmp_path / "release_gates.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _redirect(applier, tmp_path: Path, monkeypatch) -> None:
    """Deprecated: see ``test_load_release_gates_returns_yaml_required_checks``.

    Kept as a stub so existing imports remain stable; new tests should
    inline the redirect pattern. Will be removed once the inline pattern
    is exercised in >=3 tests.
    """
    import sys

    monkeypatch.setattr(applier, "REPO_ROOT", tmp_path)
    dep_mod = sys.modules["scripts.check_required_checks_sync"]
    monkeypatch.setattr(dep_mod, "RELEASE_GATES_YAML", tmp_path / "release_gates.yaml")


# ---------------------------------------------------------------------------
# build_put_payload
# ---------------------------------------------------------------------------


def test_build_put_payload_includes_all_required_contexts(applier):
    """``build_put_payload`` must surface every required_check verbatim.

    GitHub's branch-protection API matches contexts by exact string, so
    the payload must round-trip the YAML list with no normalization
    (whitespace, sorting, dedup) — that's why ``build_put_payload`` does
    ``list(required_checks)`` rather than ``set(...)``.
    """
    required = ["A (GH)", "B", "C (Issue #1234)"]
    payload = applier.build_put_payload(required)
    assert payload["required_status_checks"]["contexts"] == required
    assert payload["required_status_checks"]["strict"] is True
    assert payload["enforce_admins"] is True
    assert (
        payload["required_pull_request_reviews"]["required_approving_review_count"] == 1
    )


def test_build_put_payload_preserves_duplicates(applier):
    """If the YAML somehow contains duplicate contexts, the payload must
    preserve them — GitHub's API rejects duplicates with a 422, so the
    script intentionally surfaces them so the operator can fix the YAML
    upstream rather than silently de-duplicating.
    """
    payload = applier.build_put_payload(["A", "A", "B"])
    assert payload["required_status_checks"]["contexts"] == ["A", "A", "B"]


def test_build_put_payload_strict_can_be_disabled(applier):
    """``strict=false`` must round-trip; the script's default is True."""
    payload = applier.build_put_payload(["A"], strict=False)
    assert payload["required_status_checks"]["strict"] is False


# ---------------------------------------------------------------------------
# compute_diff
# ---------------------------------------------------------------------------


def _live_payload(contexts: list[str], *, enforce_admins: bool = True) -> dict:
    return {
        "required_status_checks": {
            "strict": True,
            "contexts": list(contexts),
        },
        "enforce_admins": {
            "enabled": enforce_admins,
        },
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
        },
    }


def test_compute_diff_identifies_missing_contexts(applier):
    """Live is missing one context the YAML requires -> diff has it in add."""
    live = _live_payload(["A", "B"])
    payload = applier.build_put_payload(["A", "B", "C"])
    diff = applier.compute_diff(live, payload)
    assert diff["contexts"]["add"] == ["C"]
    assert diff["contexts"]["remove"] == []
    assert sorted(diff["contexts"]["already_present"]) == ["A", "B"]
    assert applier.diff_has_changes(diff) is True


def test_compute_diff_identifies_extra_contexts(applier):
    """Live has a context the YAML does not require -> diff has it in remove."""
    live = _live_payload(["A", "B", "STALE"])
    payload = applier.build_put_payload(["A", "B"])
    diff = applier.compute_diff(live, payload)
    assert diff["contexts"]["add"] == []
    assert diff["contexts"]["remove"] == ["STALE"]
    assert applier.diff_has_changes(diff) is True


def test_compute_diff_identifies_enforce_admins_flip(applier):
    """The cron-detected ``enforce_admins=false`` flip must surface."""
    live = _live_payload(["A"], enforce_admins=False)
    payload = applier.build_put_payload(["A"])
    diff = applier.compute_diff(live, payload)
    assert diff["enforce_admins"]["from"] is False
    assert diff["enforce_admins"]["to"] is True
    assert diff["enforce_admins"]["would_change"] is True
    assert applier.diff_has_changes(diff) is True


def test_compute_diff_in_sync_returns_no_changes(applier):
    """Identical live + payload -> ``diff_has_changes`` returns False."""
    live = _live_payload(["A", "B"])
    payload = applier.build_put_payload(["A", "B"])
    diff = applier.compute_diff(live, payload)
    assert applier.diff_has_changes(diff) is False
    assert diff["contexts"]["add"] == []
    assert diff["contexts"]["remove"] == []
    assert diff["enforce_admins"]["would_change"] is False
    assert diff["strict"]["would_change"] is False
    assert diff["required_approving_review_count"]["would_change"] is False


def test_compute_diff_treats_empty_live_as_full_add(applier):
    """First-time apply: live protection may have empty contexts.

    The cron drift case from Issue #3386's acceptance criterion starts
    with the live ``contexts`` array missing every entry. Every YAML
    entry should land in ``add``.
    """
    live = {
        "required_status_checks": {"strict": True, "contexts": []},
        "enforce_admins": {"enabled": False},
        "required_pull_request_reviews": {
            "required_approving_review_count": 0,
        },
    }
    payload = applier.build_put_payload(["A", "B"])
    diff = applier.compute_diff(live, payload)
    assert sorted(diff["contexts"]["add"]) == ["A", "B"]
    assert diff["enforce_admins"]["would_change"] is True
    assert diff["required_approving_review_count"]["would_change"] is True
    assert applier.diff_has_changes(diff) is True


def test_compute_diff_handles_missing_live_keys(applier):
    """A live payload missing ``enforce_admins`` / ``reviews`` keys must
    not raise — the GitHub API returns those nested dicts but a partial
    response (e.g. from a custom API proxy) may not. The diff should
    surface ``would_change=True`` for the missing fields.
    """
    live = {"required_status_checks": {"strict": False, "contexts": []}}
    payload = applier.build_put_payload(["A"])
    diff = applier.compute_diff(live, payload)
    assert diff["enforce_admins"]["from"] is False
    assert diff["enforce_admins"]["to"] is True
    assert diff["enforce_admins"]["would_change"] is True
    assert diff["strict"]["from"] is False
    assert diff["strict"]["would_change"] is True
    assert diff["required_approving_review_count"]["from"] == 0
    assert diff["required_approving_review_count"]["would_change"] is True


# ---------------------------------------------------------------------------
# YAML integration (via redirect)
# ---------------------------------------------------------------------------


def test_load_release_gates_returns_yaml_required_checks(
    applier, tmp_path, monkeypatch
):
    """End-to-end: a synthetic ``release_gates.yaml`` must feed the payload.

    This is the path ``main()`` exercises — confirms the import plumbing
    into ``scripts.check_required_checks_sync`` is wired correctly.
    Without this, a refactor that breaks the import would silently fall
    through to an empty context list and the cron-detected drift would
    re-occur.
    """
    import sys

    _write_release_gates(
        tmp_path,
        ["Workspace Check (GH)", "Architecture Drift Detection"],
    )
    monkeypatch.setattr(applier, "REPO_ROOT", tmp_path)
    # The applier does ``from scripts.check_required_checks_sync import
    # load_release_gates`` — that registers the dep as
    # ``scripts.check_required_checks_sync`` in sys.modules (NOT the bare
    # ``check_required_checks_sync``), which is why the redirect must use
    # the dotted name. The dep module's ``RELEASE_GATES_YAML`` constant
    # is the one ``load_release_gates`` actually reads (line 161 of the
    # dep module: ``path = RELEASE_GATES_YAML``).
    dep_mod = sys.modules["scripts.check_required_checks_sync"]
    monkeypatch.setattr(dep_mod, "RELEASE_GATES_YAML", tmp_path / "release_gates.yaml")
    gates = applier.load_release_gates()
    checks = applier.get_required_checks(gates)
    payload = applier.build_put_payload(checks)
    assert payload["required_status_checks"]["contexts"] == [
        "Workspace Check (GH)",
        "Architecture Drift Detection",
    ]
