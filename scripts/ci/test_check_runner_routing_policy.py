"""
Tests for ``scripts/check_runner_routing_policy.py`` — Issues #3531/#3718.

Regression guard for the ``vars.FLUXION_LINUX_RUNNER`` trust-boundary
policy (docs/SECURITY.md §7): on ``pull_request`` events the runner
expression MUST resolve to ``ubuntu-latest`` regardless of the repo
variable. Issue #3718 closed the matrix-indirect bypass where a
``strategy.matrix`` entry embeds the unguarded variable and the job
consumes it via ``runs-on: ${{ matrix.os }}`` — a shape the original
``runs-on:``-line scanner never matched.

Mirrors the test pattern in ``scripts/ci/test_check_physics_sim_cycle.py``:
import the script as a module, monkey-patch the repo-rooted path
constants to point at a ``tmp_path`` fixture tree, then drive both clean
and planted-violation scenarios.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_runner_routing_policy.py"
)


def _load_checker():
    """Load scripts/check_runner_routing_policy.py as a fresh module.

    The script uses module-level ``REPO_ROOT`` / ``WORKFLOWS_DIR``
    constants rooted at the real repo, so we reload it fresh for each
    test that wants to monkey-patch those paths.
    """
    spec = importlib.util.spec_from_file_location(
        "check_runner_routing_policy", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checker():
    """Return a freshly-loaded copy of the runner-routing policy gate.

    Use ``monkeypatch.setattr(checker, "WORKFLOWS_DIR", tmp_path / ...)
    (and ``REPO_ROOT``) to redirect the scan at a synthetic fixture tree.
    """
    return _load_checker()


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(text), encoding="utf-8")
    return p


def _plant_workflow(tmp_path: Path, text: str, name: str = "pr.yml") -> Path:
    """Plant a workflow under ``tmp_path/.github/workflows/`` and point
    the freshly-loaded checker's path constants at the synthetic tree."""
    return _write(tmp_path / ".github" / "workflows" / name, text)


# ---------------------------------------------------------------------------
# Real-repo pins (regression: the shipped workflows must stay clean)
# ---------------------------------------------------------------------------


def test_real_repo_wasm_bindings_matrix_is_clean(checker, repo_root):
    """The shipped wasm-bindings.yml (Issue #3718 fix) must scan clean."""
    path = repo_root / ".github" / "workflows" / "wasm-bindings.yml"
    assert checker.scan_workflow(path) == []


def test_real_repo_node_bindings_matrix_is_clean(checker, repo_root):
    """The shipped node-bindings.yml (same matrix shape) must scan clean."""
    path = repo_root / ".github" / "workflows" / "node-bindings.yml"
    assert checker.scan_workflow(path) == []


def test_gate_passes_on_real_repo(checker, repo_root):
    """End-to-end: the real checkout must pass the whole-workflow scan."""
    assert repo_root == checker.REPO_ROOT
    assert checker.main() == 0


# ---------------------------------------------------------------------------
# Matrix-indirect scan (Issue #3718)
# ---------------------------------------------------------------------------

# The pre-fix wasm-bindings.yml shape: PR-triggered, job-level
# ``runs-on: ${{ matrix.os }}``, matrix entry embedding the unguarded
# runner variable.
WASM_UNGUARDED_MATRIX = """\
    name: WASM Bindings

    on:
      push:
        branches: [main, develop]
      pull_request:
        paths:
          - 'fluxion-wasm/**'

    permissions:
      contents: read

    jobs:
      build-test:
        name: WASM Bindings (${{ matrix.os }})
        runs-on: ${{ matrix.os }}
        strategy:
          fail-fast: false
          matrix:
            os:
              - ${{ vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}
              - windows-latest
        steps:
          - uses: actions/checkout@v4
"""

WASM_GUARDED_MATRIX = WASM_UNGUARDED_MATRIX.replace(
    "- ${{ vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}",
    "- ${{ github.event_name == 'push' && github.ref == 'refs/heads/main'"
    " && vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}",
)

WASM_GUARDED_MATRIX_FOLDED = """\
    name: WASM Bindings

    on:
      push:
        branches: [main, develop]
      pull_request:
        paths:
          - 'fluxion-wasm/**'

    jobs:
      build-test:
        runs-on: ${{ matrix.os }}
        strategy:
          matrix:
            os:
              - >-
                ${{
                  github.event_name == 'push'
                  && github.ref == 'refs/heads/main'
                  && vars.FLUXION_LINUX_RUNNER
                  || 'ubuntu-latest'
                }}
              - windows-latest
        steps:
          - uses: actions/checkout@v4
"""


def test_flags_unguarded_matrix_entry_wasm_shape(checker, tmp_path, monkeypatch):
    """The pre-fix wasm-bindings.yml shape must be flagged (#3718)."""
    path = _plant_workflow(tmp_path, WASM_UNGUARDED_MATRIX, "wasm-bindings.yml")
    findings = checker.scan_workflow(path)
    assert len(findings) == 1
    job_id, line_no, lines = findings[0]
    assert job_id == "build-test"
    assert line_no > 0
    joined = " ".join(lines)
    assert "vars.FLUXION_LINUX_RUNNER" in joined
    assert "matrix:" in joined


def test_accepts_single_line_guarded_matrix_entry(checker, tmp_path, monkeypatch):
    """The canonical single-line guarded form (acceptance criterion 1)."""
    path = _plant_workflow(tmp_path, WASM_GUARDED_MATRIX, "wasm-bindings.yml")
    assert checker.scan_workflow(path) == []


def test_accepts_folded_guarded_matrix_entry(checker, tmp_path, monkeypatch):
    """The folded multi-line guarded form must also pass."""
    path = _plant_workflow(tmp_path, WASM_GUARDED_MATRIX_FOLDED, "wasm-bindings.yml")
    assert checker.scan_workflow(path) == []


def test_main_only_job_matrix_exempt(checker, tmp_path, monkeypatch):
    """A main-only job (``if:`` filters out pull_request) may use the
    unguarded matrix entry — it never executes on PR-controlled code."""
    text = WASM_UNGUARDED_MATRIX.replace(
        "      build-test:\n        name:",
        "      build-test:\n        if: github.event_name == 'push'"
        " && github.ref == 'refs/heads/main'\n        name:",
    )
    path = _plant_workflow(tmp_path, text, "main-only.yml")
    assert checker.scan_workflow(path) == []


def test_no_pull_request_trigger_matrix_exempt(checker, tmp_path, monkeypatch):
    """Without a ``pull_request`` trigger the matrix scan is exempt."""
    text = WASM_UNGUARDED_MATRIX.replace(
        "      pull_request:\n        paths:\n          - 'fluxion-wasm/**'\n", ""
    )
    path = _plant_workflow(tmp_path, text, "push-only.yml")
    assert checker.scan_workflow(path) == []


def test_main_exits_nonzero_on_matrix_violation(checker, tmp_path, monkeypatch):
    """``main()`` must fail the gate on the planted wasm shape."""
    _plant_workflow(tmp_path, WASM_UNGUARDED_MATRIX, "wasm-bindings.yml")
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", tmp_path / ".github" / "workflows")
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    assert checker.main() == 1


# ---------------------------------------------------------------------------
# Direct runs-on scan (Issue #3531 regression guard)
# ---------------------------------------------------------------------------


def test_direct_unguarded_runs_on_still_flagged(checker, tmp_path, monkeypatch):
    """The #3531 direct form must remain flagged by the extended gate."""
    text = """\
        name: Direct

        on:
          pull_request:

        jobs:
          build:
            runs-on: ${{ vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}
            steps:
              - uses: actions/checkout@v4
    """
    path = _plant_workflow(tmp_path, text, "direct.yml")
    findings = checker.scan_workflow(path)
    assert len(findings) == 1
    assert findings[0][0] == "build"
    assert any("vars.FLUXION_LINUX_RUNNER" in line for line in findings[0][2])
