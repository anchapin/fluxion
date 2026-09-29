"""
Tests for ``scripts/check_trivy_scan_target.py`` -- Issue #4186.

``docs/SECURITY.md`` §7 defers the unpinned ``apt-get install`` layers to
Trivy as the compensating control. The step it pointed at ran
``scan-type: 'fs'`` against the repository tree -- a scan that structurally
cannot see the image's OS packages -- and set no ``exit-code``, so it also
could not fail on any finding. Both halves of the compensating control were
absent.

The gate must catch both misconfigurations, and it must not become a
straightjacket: a source-tree scan is legitimate for a different concern
(SBOM / license policy), so a documented justification comment is the
single escape hatch for either invariant.

Fixtures are synthetic workflows under ``tmp_path``; the suite never scans
the real workflow tree except for one final canary.
"""
from __future__ import annotations

import sys

import pytest

SCRIPT_NAME = "check_trivy_scan_target"


@pytest.fixture
def gate(load_script):
    """Freshly-loaded copy of the Trivy scan-target gate."""
    return load_script(SCRIPT_NAME)


def _wf(tmp_path, body: str, name: str = "ci.yml"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Positive: the #4186 defect shapes must be reported
# ---------------------------------------------------------------------------


def test_reports_fs_source_scan(gate, tmp_path):
    """The #4186 shape: `fs` scan claims to cover the built image."""
    path = _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    # The pre-fix step also lacked `exit-code`, so both defects surface.
    assert len(findings) == 2
    assert any("scan-type: 'fs'" in f for f in findings)
    assert any("SECURITY.md" in f for f in findings)


def test_reports_missing_exit_code(gate, tmp_path):
    """No `exit-code` means TRIVY_EXIT_CODE is never exported -> advisory."""
    path = _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          scan-type: 'image'
          image-ref: 'ghcr.io/x/y:1'
          format: 'sarif'
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 1
    assert "exit-code" in findings[0]
    assert "cannot fail" in findings[0]


def test_reports_both_defects_together(gate, tmp_path):
    """The pre-fix step had both; both must surface."""
    path = _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Trivy
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          scan-type: 'fs'
          scan-ref: '.'
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 2


def test_reports_non_trivy_steps_ignored(gate, tmp_path):
    """Only `trivy-action` steps are in scope."""
    path = _wf(
        tmp_path,
        """
jobs:
  build:
    steps:
      - name: Build
        uses: docker/build-push-action@c3c9e263c25d99ce0380d002d59b67737d91b0dc
        with:
          scan-type: 'fs'
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_reports_step_without_with_block(gate, tmp_path):
    """A bare `uses:` still defaults to `image` but lacks `exit-code`."""
    path = _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Trivy
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 1
    assert "exit-code" in findings[0]


# ---------------------------------------------------------------------------
# Negative: the fixed shape and documented exemptions must stay clean
# ---------------------------------------------------------------------------


def test_clean_image_tarball_scan_with_exit_code(gate, tmp_path):
    """The post-#4186 shape: image tarball + explicit exit-code threshold."""
    path = _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Download image tarball
        uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c
        with:
          name: docker-image-tarball
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          scan-type: 'image'
          input: 'image.tar'
          exit-code: '1'
          severity: 'CRITICAL'
          ignore-unfixed: true
          format: 'sarif'
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_clean_default_scan_type_is_image(gate, tmp_path):
    """The action's own default is `image`; omitting it is not a defect."""
    path = _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Trivy
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          image-ref: 'ghcr.io/x/y:1'
          exit-code: '1'
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_clean_justified_source_scan(gate, tmp_path):
    """A documented source scan (SBOM/license policy) is legitimate."""
    path = _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Trivy source scan for SBOM/license policy
        # Intentionally a source-tree scan: this step inventories the
        # repository's own dependency manifests for license policy, which
        # is a different concern from image CVEs.
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          scan-type: 'fs'
          scan-ref: '.'
          exit-code: '1'
""",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_justification_does_not_launder_a_stale_comment(gate, tmp_path):
    """A comment that documents no justification must not suppress a finding."""
    path = _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Trivy
        # Pinned to a commit SHA for supply-chain security.
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          scan-type: 'fs'
          scan-ref: '.'
          exit-code: '1'
""",
    )
    findings = gate.check_workflow(path, rel="ci.yml")
    assert len(findings) == 1
    assert "scan-type: 'fs'" in findings[0]


def test_clean_workflow_without_trivy(gate, tmp_path):
    """A workflow with no Trivy step is trivially clean."""
    path = _wf(
        tmp_path,
        "jobs:\n  build:\n    steps:\n      - name: Build\n        run: make\n",
    )
    assert gate.check_workflow(path, rel="ci.yml") == []


def test_clean_workflow_without_jobs(gate, tmp_path):
    path = _wf(tmp_path, "name: ci\n")
    assert gate.check_workflow(path, rel="ci.yml") == []


# ---------------------------------------------------------------------------
# main() surface
# ---------------------------------------------------------------------------


def test_main_returns_one_on_findings(gate, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gate"])
    _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Trivy
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          scan-type: 'fs'
          scan-ref: '.'
          exit-code: '1'
""",
    )
    monkeypatch.setattr(
        sys, "argv", ["gate", "--workflows-dir", str(tmp_path)]
    )
    assert gate.main() == 1
    assert "Trivy step misconfiguration" in capsys.readouterr().err


def test_main_returns_zero_when_clean(gate, tmp_path, monkeypatch, capsys):
    _wf(
        tmp_path,
        """
jobs:
  security:
    steps:
      - name: Trivy
        uses: aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25
        with:
          scan-type: 'image'
          image-ref: 'x:1'
          exit-code: '1'
""",
    )
    monkeypatch.setattr(
        sys, "argv", ["gate", "--workflows-dir", str(tmp_path)]
    )
    assert gate.main() == 0
    assert "OK" in capsys.readouterr().out


def test_main_returns_two_when_dir_missing(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["gate", "--workflows-dir", str(tmp_path / "nope")]
    )
    assert gate.main() == 2


# ---------------------------------------------------------------------------
# Non-vacuity / scope guards
# ---------------------------------------------------------------------------


def test_script_module_exposes_expected_surface(gate):
    """Guard against an import fixture that silently yields a stub."""
    assert callable(gate.check_workflow)
    assert callable(gate.main)
    with open(gate.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert "TRIVY_ACTION" in src
    assert "exit-code" in src


def test_real_workflow_tree_is_clean(gate, repo_root):
    """Pin the gate against the live workflow tree after the #4186 fix."""
    findings = []
    for path in sorted((repo_root / ".github" / "workflows").glob("*.yml")):
        findings.extend(
            gate.check_workflow(
                path, rel=path.relative_to(repo_root).as_posix()
            )
        )
    assert findings == [], findings
