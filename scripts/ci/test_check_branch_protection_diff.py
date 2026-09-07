"""Tests for ``scripts/check_branch_protection_diff.py`` -- Issue #3426.

Regression guard for the diagnostic half of the branch-protection
reconcile (the applier half is covered by
``test_apply_branch_protection.py``). Mirrors the ``load_script`` +
``tmp_path`` mock-repo pattern from that file:

* load the script as a fresh module via the shared ``load_script``
  fixture,
* drive the offline-testable surface: ``load_canonical_required_checks``
  (YAML parsing + error exits), ``compute_diff`` (canonical-vs-live set
  diff), ``build_desired_put_payload`` (PUT payload shape), and
  ``fetch_live_protection``'s ``GH_TOKEN``-stripping subprocess boundary,
* pin the ``main()`` exit-code contract: 0 no drift / 1 drift / 2 script
  error, with ``gh`` mocked at the ``fetch_live_protection`` seam.

The GitHub-API boundary is isolated behind ``fetch_live_protection``'s
subprocess call, so no test here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT_NAME = "check_branch_protection_diff"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_branch_protection_diff.py``."""
    return load_script(SCRIPT_NAME)


def _write_release_gates(tmp_path: Path, contexts: list[str]) -> Path:
    """Write a synthetic ``release_gates.yaml`` with one
    ``ci.required_checks`` list and return its path.

    Hand-built YAML (not ``yaml.dump``) so a parser regression in the
    script's ``yaml.safe_load`` path surfaces instead of being hidden by
    re-serialization — same rationale as
    ``test_apply_branch_protection.py::_write_release_gates``.
    """
    lines = ["ci:", "  required_checks:"]
    for c in contexts:
        lines.append(f'    - "{c}"')
    target = tmp_path / "release_gates.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _live_payload(contexts: list[str], enforce_admins: bool) -> dict:
    """Minimal live protection payload shaped like GitHub's
    ``GET /repos/{owner}/{repo}/branches/develop/protection`` response.
    """
    return {
        "required_status_checks": {"contexts": list(contexts)},
        "enforce_admins": {"enabled": enforce_admins},
    }


# ---------------------------------------------------------------------------
# load_canonical_required_checks
# ---------------------------------------------------------------------------


def test_load_canonical_required_checks_parses_yaml(checker, tmp_path):
    """``ci.required_checks`` round-trips verbatim from the YAML fixture."""
    expected = ["Rust Tests (GH)", "Docs Hygiene Gate", "MSRV Check"]
    gates = _write_release_gates(tmp_path, expected)
    assert checker.load_canonical_required_checks(gates) == expected


def test_load_canonical_missing_file_exits_2(checker, tmp_path):
    """A missing release_gates.yaml is a script error (exit 2)."""
    with pytest.raises(SystemExit) as excinfo:
        checker.load_canonical_required_checks(tmp_path / "nope.yaml")
    assert excinfo.value.code == 2


def test_load_canonical_missing_ci_key_exits_2(checker, tmp_path):
    """A release_gates.yaml without ``ci.required_checks`` exits 2."""
    gates = tmp_path / "release_gates.yaml"
    gates.write_text("ci:\n  workflow_index: []\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        checker.load_canonical_required_checks(gates)
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# compute_diff
# ---------------------------------------------------------------------------


def test_compute_diff_reports_missing_and_extra(checker):
    """Set-symmetric diff: canonical-minus-live and live-minus-canonical."""
    canonical = ["A (GH)", "B (GH)", "C"]
    live = ["B (GH)", "C", "D (GH)"]
    diff = checker.compute_diff(canonical, live)
    assert diff["missing_from_live"] == ["A (GH)"]
    assert diff["extra_in_live"] == ["D (GH)"]
    assert diff["in_both"] == ["B (GH)", "C"]


def test_compute_diff_identical_lists_is_empty(checker):
    diff = checker.compute_diff(["A", "B"], ["B", "A"])
    assert diff["missing_from_live"] == []
    assert diff["extra_in_live"] == []
    assert diff["in_both"] == ["A", "B"]


# ---------------------------------------------------------------------------
# build_desired_put_payload
# ---------------------------------------------------------------------------


def test_build_desired_put_payload_shape(checker):
    """The payload surfaces both PUTs verbatim and never executes them.

    GitHub matches contexts by exact string, so ``contexts`` must be the
    canonical list un-normalized; ``strict`` must be True; the
    ``enforce_admins`` toggle lives on the protection endpoint, not the
    required_status_checks endpoint.
    """
    canonical = ["X (GH)", "Y"]
    payload = checker.build_desired_put_payload(canonical, enforce_admins=True)
    assert payload["_note"].startswith("Diagnostic output")
    assert "#3386" in payload["_note"]
    assert (
        payload["required_status_checks_put_payload"]["url"]
        == "/repos/<owner>/<repo>/branches/develop/protection/required_status_checks"
    )
    body = payload["required_status_checks_put_payload"]["body"]
    assert body["contexts"] == canonical
    assert body["strict"] is True
    assert payload["protection_put_payload"]["url"] == (
        "/repos/<owner>/<repo>/branches/develop/protection"
    )
    assert payload["protection_put_payload"]["body"] == {
        "enforce_admins": {"enabled": True}
    }


# ---------------------------------------------------------------------------
# fetch_live_protection (subprocess boundary)
# ---------------------------------------------------------------------------


def test_fetch_live_protection_strips_gh_token(checker, monkeypatch):
    """The subprocess env must drop a stale ``GH_TOKEN``.

    The project shell exports an invalid ``GH_TOKEN`` that overrides gh's
    default-account token — stripping it is a load-bearing guardrail, not
    a nicety (see the script docstring).
    """
    captured: dict = {}

    def fake_check_output(cmd, text, stderr, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return json.dumps(_live_payload(["A"], True))

    monkeypatch.setattr(checker.subprocess, "check_output", fake_check_output)
    monkeypatch.setenv("GH_TOKEN", "stale-token-value")

    live = checker.fetch_live_protection("owner/repo")

    assert live["enforce_admins"]["enabled"] is True
    assert captured["cmd"] == [
        "gh",
        "api",
        "/repos/owner/repo/branches/develop/protection",
    ]
    assert "GH_TOKEN" not in captured["env"]


def test_fetch_live_protection_propagates_gh_failure(checker, monkeypatch):
    """A ``gh api`` failure surfaces as ``CalledProcessError`` for
    ``main()`` to translate into exit code 2."""
    import subprocess

    def fake_check_output(cmd, text, stderr, env):
        raise subprocess.CalledProcessError(1, cmd, stderr="HTTP 401")

    monkeypatch.setattr(checker.subprocess, "check_output", fake_check_output)
    with pytest.raises(subprocess.CalledProcessError):
        checker.fetch_live_protection("owner/repo")


# ---------------------------------------------------------------------------
# main() exit-code contract: 0 no drift / 1 drift / 2 script error
# ---------------------------------------------------------------------------


def test_main_exit_0_when_live_matches_canonical(
    checker, tmp_path, monkeypatch, capsys
):
    canonical = ["A (GH)", "B (GH)"]
    gates = _write_release_gates(tmp_path, canonical)
    live = _live_payload(canonical, enforce_admins=True)
    monkeypatch.setattr(checker, "fetch_live_protection", lambda repo: live)

    code = checker.main(
        ["--release-gates", str(gates), "--repo", "owner/repo", "--json"]
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["diff"]["missing_from_live"] == []
    assert report["diff"]["extra_in_live"] == []
    assert report["enforce_admins"] is True


def test_main_exit_1_on_missing_checks_and_enforce_admins_false(
    checker, tmp_path, monkeypatch, capsys
):
    """The #3383 scenario: 6 canonical checks missing from live AND
    ``enforce_admins.enabled=false`` — both drift classes in one run."""
    canonical = ["A (GH)", "B (GH)", "C (GH)"]
    gates = _write_release_gates(tmp_path, canonical)
    live = _live_payload(["B (GH)"], enforce_admins=False)
    monkeypatch.setattr(checker, "fetch_live_protection", lambda repo: live)

    code = checker.main(
        ["--release-gates", str(gates), "--repo", "owner/repo", "--json"]
    )

    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["diff"]["missing_from_live"] == ["A (GH)", "C (GH)"]
    assert report["enforce_admins"] is False
    # The desired PUT reconciles to enforce_admins=true regardless of the
    # current (drifted) live value.
    desired = report["desired_put"]["protection_put_payload"]["body"]
    assert desired == {"enforce_admins": {"enabled": True}}
    assert report["desired_put"]["required_status_checks_put_payload"][
        "body"
    ]["contexts"] == canonical


def test_main_exit_1_on_extra_in_live(
    checker, tmp_path, monkeypatch, capsys
):
    """Live enforcing a check the canonical list retired is still drift."""
    canonical = ["A (GH)"]
    gates = _write_release_gates(tmp_path, canonical)
    live = _live_payload(["A (GH)", "RETIRED (GH)"], enforce_admins=True)
    monkeypatch.setattr(checker, "fetch_live_protection", lambda repo: live)

    code = checker.main(
        ["--release-gates", str(gates), "--repo", "owner/repo", "--json"]
    )

    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["diff"]["extra_in_live"] == ["RETIRED (GH)"]


def test_main_exit_2_on_gh_failure(checker, tmp_path, monkeypatch, capsys):
    """``gh api`` auth/network failure exits 2 (script error), not 1."""
    import subprocess

    gates = _write_release_gates(tmp_path, ["A (GH)"])

    def fake_fetch(repo):
        raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="HTTP 401")

    monkeypatch.setattr(checker, "fetch_live_protection", fake_fetch)

    code = checker.main(
        ["--release-gates", str(gates), "--repo", "owner/repo", "--json"]
    )

    assert code == 2
    assert "gh api failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# resolve_default_repo (Issue #3429 resolution order)
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_resolve_default_repo_prefers_github_repository(checker, monkeypatch):
    """``$GITHUB_REPOSITORY`` wins — always correct on Actions runners."""
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *a, **k: pytest.fail(
            "git must not be consulted when GITHUB_REPOSITORY is set"
        ),
    )
    assert (
        checker.resolve_default_repo({"GITHUB_REPOSITORY": "fork/fluxion"})
        == "fork/fluxion"
    )


def test_resolve_default_repo_parses_origin_remote(checker, monkeypatch):
    """https and ssh origin URLs normalize to ``owner/repo``."""
    urls = [
        "https://github.com/mirror/fluxion.git",
        "git@github.com:mirror/fluxion.git",
        "https://github.com/anchapin/fluxion",
    ]
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        out = _FakeCompleted(urls[calls["n"]])
        calls["n"] += 1
        return out

    monkeypatch.setattr(checker.subprocess, "run", fake_run)
    assert checker.resolve_default_repo({}) == "mirror/fluxion"
    assert checker.resolve_default_repo({}) == "mirror/fluxion"
    assert checker.resolve_default_repo({}) == "anchapin/fluxion"


def test_resolve_default_repo_falls_back_to_constant(checker, monkeypatch):
    """No env var + git failure (or unparseable URL) → DEFAULT_REPO."""
    import subprocess

    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(checker.subprocess, "run", failing_run)
    assert checker.resolve_default_repo({}) == "anchapin/fluxion"
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *a, **k: _FakeCompleted("/local/path/repo"),
    )
    assert checker.resolve_default_repo({}) == "anchapin/fluxion"
