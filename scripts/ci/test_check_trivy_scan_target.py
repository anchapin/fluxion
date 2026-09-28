"""Tests for ``scripts/check_trivy_scan_target.py`` -- Issue #4186.

The script rejects ``aquasecurity/trivy-action`` steps that scan the repo
filesystem (``scan-type: 'fs'``) without a documented
``trivy-fs-justified:`` comment, keeping the Issue #4186 regression from
recurring: Trivy must scan the BUILT IMAGE (``input:`` tarball from the
``build-and-test`` job in ``.github/workflows/docker.yml``) with a
fail-closed ``exit-code: '1'``.

The tests below drive the matcher against hermetic ``tmp_path`` mock
repos so a scanner regression surfaces here instead of on a real PR.
Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_workflow_pin.py`` (Issue #3475).

Coverage:

* ``check_workflow()`` -- per-file findings for each planted shape
  (bare fs scan, quoted fs scan, image scan, justified fs scan, no
  trivy step, justification on the wrong step),
* ``main()`` exit-code contract: 0 clean / 1 drift / 2 script error.

No test depends on the network or on the repo's real workflows; the
end-to-end against the real tree is the direct
``check_trivy_scan_target.py`` invocation in ``scripts-tests.yml``
(Issue #4186 acceptance criterion #4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_trivy_scan_target"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_trivy_scan_target.py``."""
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


_IMAGE_SCAN = (
    "name: Docker\n"
    "on: push\n"
    "jobs:\n"
    "  security:\n"
    "    steps:\n"
    "      - name: Run Trivy vulnerability scanner\n"
    "        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25\n"
    "        with:\n"
    "          input: /tmp/trivy-scan/fluxion-image.tar\n"
    "          scan-type: 'image'\n"
    "          format: 'sarif'\n"
    "          output: 'trivy-results.sarif'\n"
    "          exit-code: '1'\n"
)

_FS_SCAN = (
    "name: Docker\n"
    "on: push\n"
    "jobs:\n"
    "  security:\n"
    "    steps:\n"
    "      - name: Run Trivy vulnerability scanner\n"
    "        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25\n"
    "        with:\n"
    "          scan-type: 'fs'\n"
    "          scan-ref: '.'\n"
    "          format: 'sarif'\n"
    "          output: 'trivy-results.sarif'\n"
)

_JUSTIFIED_FS_SCAN = (
    "name: Docs\n"
    "on: push\n"
    "jobs:\n"
    "  audit:\n"
    "    steps:\n"
    "      # trivy-fs-justified: lockfile audit for the docs site;\n"
    "      # the runtime layer is covered by the image scan in docker.yml.\n"
    "      - name: Run Trivy vulnerability scanner\n"
    "        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25\n"
    "        with:\n"
    "          scan-type: 'fs'\n"
    "          scan-ref: 'docs/'\n"
)


# ---------------------------------------------------------------------------
# check_workflow() -- per-file findings
# ---------------------------------------------------------------------------


def test_image_scan_is_clean(checker, monkeypatch, tmp_path):
    """The Issue #4186 target shape -- image tarball input -- passes."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    path = _write_workflow(workflows, "docker.yml", _IMAGE_SCAN)
    assert checker.check_workflow(path) == []


def test_bare_fs_scan_is_flagged(checker, monkeypatch, tmp_path):
    """Unjustified ``scan-type: 'fs'`` is a finding."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    path = _write_workflow(workflows, "docker.yml", _FS_SCAN)
    findings = checker.check_workflow(path)
    assert len(findings) == 1
    assert "scan-type 'fs'" in findings[0]
    assert "docker.yml:7" in findings[0]


def test_quoted_fs_scan_is_flagged(checker, monkeypatch, tmp_path):
    """``scan-type: \"fs\"`` (double-quoted) is the same violation."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    text = _FS_SCAN.replace("scan-type: 'fs'", 'scan-type: "fs"')
    path = _write_workflow(workflows, "docker.yml", text)
    assert len(checker.check_workflow(path)) == 1


def test_justified_fs_scan_passes(checker, monkeypatch, tmp_path):
    """A ``trivy-fs-justified:`` comment above the step whitelists it."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    path = _write_workflow(workflows, "docs.yml", _JUSTIFIED_FS_SCAN)
    assert checker.check_workflow(path) == []


def test_justification_on_unrelated_step_does_not_whitelist(
    checker, monkeypatch, tmp_path
):
    """A justification comment attached to a DIFFERENT step must not
    whitelist the trivy step -- the marker has to sit above the trivy
    step itself."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    text = (
        "name: Docker\n"
        "on: push\n"
        "jobs:\n"
        "  security:\n"
        "    steps:\n"
        "      # trivy-fs-justified: this comment is about the checkout step.\n"
        "      - name: Checkout\n"
        "        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n"
        "      - name: Run Trivy vulnerability scanner\n"
        "        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25\n"
        "        with:\n"
        "          scan-type: 'fs'\n"
    )
    path = _write_workflow(workflows, "docker.yml", text)
    assert len(checker.check_workflow(path)) == 1


def test_workflow_without_trivy_is_clean(checker, monkeypatch, tmp_path):
    """Workflows with no trivy-action step produce no findings."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    path = _write_workflow(
        workflows, "ci.yml", "name: CI\non: push\njobs:\n  t:\n    steps: []\n"
    )
    assert checker.check_workflow(path) == []


def test_commented_out_trivy_uses_is_ignored(checker, monkeypatch, tmp_path):
    """A commented-out trivy ``uses:`` line is not an active step."""
    workflows = _redirect(checker, monkeypatch, tmp_path)
    text = _FS_SCAN.replace(
        "        uses: aquasecurity/trivy-action@",
        "        # uses: aquasecurity/trivy-action@",
    )
    path = _write_workflow(workflows, "docker.yml", text)
    assert checker.check_workflow(path) == []


# ---------------------------------------------------------------------------
# main() -- exit-code contract
# ---------------------------------------------------------------------------


def test_main_returns_0_on_clean_tree(checker, monkeypatch, tmp_path):
    workflows = _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(workflows, "docker.yml", _IMAGE_SCAN)
    assert checker.main([]) == 0


def test_main_returns_1_on_fs_drift(checker, monkeypatch, tmp_path, capsys):
    workflows = _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(workflows, "docker.yml", _FS_SCAN)
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
