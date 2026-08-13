"""Shared pytest fixtures for the ``scripts/`` CI-gate test harness.

Issue #2817: the branch-protection-critical gate scripts under ``scripts/``
previously had *no* automated test coverage — they were only verified by
running them directly in CI. This harness (built on the
``importlib.util`` load-by-path pattern pioneered by
``test_check_physics_sim_cycle.py``) lets every gate contribute parametric
clean-tree / planted-violation cases driven from a hermetic ``tmp_path``
mock repo.

The fixtures here are intentionally minimal and reusable so a new gate
test only has to: load a fresh copy of the script, redirect its
module-level path constants (``REPO_ROOT`` / ``DOCS_ROOT`` / ...) at a
synthetic tree, and assert on the structured result.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The real repository root.

    Used by tests that pin a gate against a known-clean checkout (a
    regression in the scanner would flip the real repo from clean to
    failing), distinct from the per-test ``tmp_path`` mock roots.
    """
    return REPO_ROOT


@pytest.fixture
def load_script():
    """Return a factory that loads ``scripts/<name>.py`` as a fresh module.

    Each call returns a brand-new module object so per-test
    ``monkeypatch.setattr(module, "REPO_ROOT", tmp_path)`` is isolated.
    Mirrors the per-file ``_load_checker`` helpers in the existing
    ``test_check_physics_sim_cycle.py`` / ``test_check_cycle_downward_trend.py``.
    """

    def _load(name: str):
        path = SCRIPTS_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    return _load


@pytest.fixture
def write_file():
    """Return a helper that writes ``text`` to ``path`` (``mkdir -p`` parents)."""

    def _write(path: Path, text: str = "") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return _write
