"""Tests for ``scripts/check_pip_pinning.py`` (Issue #3812).

Covers the job-iteration parser, the credential-bearing classifier,
the pip-install arg classifier (incl. pinned-spec, constraints-file,
and self-upgrade whitelists), and the end-to-end ``scan_workflow``
and ``main()`` contracts. Hermetic ``tmp_path`` fixtures only — no
network access.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_pip_pinning"


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def test_iter_jobs_two_top_level_jobs(checker, tmp_path):
    """`_iter_jobs` splits a workflow body on the top-level
    `jobs:` keys (2-space indent) and yields each job's body as a
    separate string."""
    text = (
        "name: t\n"
        "on: [push]\n"
        "jobs:\n"
        "  first:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo hi\n"
        "  second:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo bye\n"
    )
    jobs = checker._iter_jobs(text)
    assert [k for k, _ in jobs] == ["first", "second"]
    assert "echo hi" in jobs[0][1]
    assert "echo bye" in jobs[1][1]


def test_credential_bearing_secrets(checker):
    """A job with a `secrets.*` env reference is credential-bearing."""
    job_text = (
        "permissions:\n"
        "  contents: read\n"
        "env:\n"
        "  GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n"
        "steps:\n"
        "  - run: pip install pyyaml\n"
    )
    assert checker._is_credential_bearing_job(job_text)


def test_credential_bearing_id_token_write(checker):
    """A job with `id-token: write` is credential-bearing even without
    a secrets env."""
    job_text = (
        "permissions:\n"
        "  id-token: write\n"
        "steps:\n"
        "  - run: pip install boto3\n"
    )
    assert checker._is_credential_bearing_job(job_text)


def test_credential_bearing_contents_write(checker):
    """A job with `contents: write` is credential-bearing (low-bar
    proxy for branch-mutation credentials)."""
    job_text = (
        "permissions:\n"
        "  contents: write\n"
        "steps:\n"
        "  - run: pip install build\n"
    )
    assert checker._is_credential_bearing_job(job_text)


def test_credential_bearing_no_creds(checker):
    """A job with neither secrets nor write permissions is NOT
    credential-bearing."""
    job_text = (
        "permissions:\n"
        "  contents: read\n"
        "steps:\n"
        "  - run: pip install pyyaml\n"
    )
    assert not checker._is_credential_bearing_job(job_text)


def test_classifier_pinned_exact(checker, tmp_path):
    ok, reason = checker._classify_pip_install_args(
        "boto3==1.35.36", Path("dummy.yml")
    )
    assert ok
    assert "pinned" in reason


def test_classifier_pinned_floor(checker, tmp_path):
    ok, reason = checker._classify_pip_install_args(
        "pyyaml>=6.0", Path("dummy.yml")
    )
    assert ok


def test_classifier_pinned_quoted(checker, tmp_path):
    ok, reason = checker._classify_pip_install_args(
        "'twine==5.1.1'", Path("dummy.yml")
    )
    assert ok


def test_classifier_self_upgrade_pip(checker, tmp_path):
    ok, reason = checker._classify_pip_install_args("pip", Path("dummy.yml"))
    assert ok
    assert "self-upgrade" in reason


def test_classifier_self_upgrade_with_flag(checker, tmp_path):
    """`pip install --upgrade pip` is the self-upgrade pattern."""
    ok, reason = checker._classify_pip_install_args(
        "--upgrade pip", Path("dummy.yml")
    )
    assert ok
    assert "self-upgrade" in reason


def test_classifier_unpinned_bare(checker, tmp_path):
    ok, reason = checker._classify_pip_install_args("pyyaml", Path("dummy.yml"))
    assert not ok
    assert "unpinned" in reason


def test_classifier_constraints_file(checker, tmp_path):
    """`-r <file>` requires the file to exist with at least one
    pinned specifier."""
    req = tmp_path / "req.txt"
    req.write_text("pyyaml>=6.0\n", encoding="utf-8")
    ok, reason = checker._classify_pip_install_args(
        f"-r {req}", Path("dummy.yml")
    )
    assert ok
    assert "constraints file" in reason


def test_classifier_constraints_file_without_pin(checker, tmp_path):
    """`-r <file>` with NO pinned specifier in the file fails — a
    constraints file with bare names is no safer than a bare install."""
    req = tmp_path / "req.txt"
    req.write_text("pyyaml\nboto3\n", encoding="utf-8")
    ok, reason = checker._classify_pip_install_args(
        f"-r {req}", Path("dummy.yml")
    )
    assert not ok
    assert "no version-pinned" in reason


def test_classifier_constraints_file_missing(checker, tmp_path):
    ok, reason = checker._classify_pip_install_args(
        "-r /nonexistent/req.txt", Path("dummy.yml")
    )
    assert not ok
    assert "does not exist" in reason


def test_scan_workflow_credential_job_unpinned(checker, tmp_path, monkeypatch):
    """End-to-end: a credential-bearing job with a bare `pip install`
    produces a failure line referencing the workflow, job id, and
    bare package."""
    wf = tmp_path / ".github" / "workflows" / "ci.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text(
        "name: t\n"
        "jobs:\n"
        "  cj:\n"
        "    permissions:\n"
        "      contents: write\n"
        "    steps:\n"
        "      - run: pip install pyyaml\n"
    )
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", wf.parent)
    failures = checker.scan_workflow(wf)
    assert any("cj" in line and "pyyaml" in line for line in failures)


def test_scan_workflow_safe_job_ignored(checker, tmp_path, monkeypatch):
    """A bare `pip install` in a NON-credential-bearing job does NOT
    produce a failure."""
    wf = tmp_path / ".github" / "workflows" / "ci.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text(
        "name: t\n"
        "jobs:\n"
        "  sj:\n"
        "    permissions:\n"
        "      contents: read\n"
        "    steps:\n"
        "      - run: pip install pyyaml\n"
    )
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", wf.parent)
    failures = checker.scan_workflow(wf)
    assert failures == []


def test_main_returns_zero_when_clean(checker, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    wf = tmp_path / ".github" / "workflows" / "ci.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text(
        "name: t\n"
        "jobs:\n"
        "  cj:\n"
        "    permissions:\n"
        "      contents: read\n"
        "    steps:\n"
        "      - run: pip install pyyaml\n"
    )
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", wf.parent)
    assert checker.main([]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_main_returns_one_when_dirty(checker, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    wf = tmp_path / ".github" / "workflows" / "ci.yml"
    wf.parent.mkdir(parents=True)
    wf.write_text(
        "name: t\n"
        "jobs:\n"
        "  cj:\n"
        "    permissions:\n"
        "      contents: write\n"
        "    steps:\n"
        "      - run: pip install pyyaml\n"
    )
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", wf.parent)
    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "pyyaml" in out