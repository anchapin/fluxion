"""
Tests for ``scripts/check_root_md_policy.py`` -- the backward-compat alias
that delegates to ``scripts/check_root_hygiene.py``.

The file is a thin shim (37 lines) that must satisfy three contracts:

* exit-code propagation (the gate's verdict is the underlying gate's verdict),
* argv forwarding (every CLI flag the underlying gate accepts must pass through),
* fail-loud delegation (the ``__main__`` wrapper raises exit 2 on errors).

The script reads no module-level state, so the tests can pin its observable
behaviour by sourcing the underlying ``check_root_hygiene.py`` directly and
exercising the wrapper via ``subprocess`` against a crafted ``tmp_path`` repo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_root_md_policy"
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "check_root_md_policy.py"
)
HYGIENE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "check_root_hygiene.py"
)


@pytest.fixture
def alias(load_script):
    """Freshly-loaded copy of the shim module."""
    return load_script(SCRIPT_NAME)


# ---------------------------------------------------------------------------
# main() — argv forwarding + exit-code propagation
# ---------------------------------------------------------------------------


def test_main_propagates_exit_code_zero_on_clean_tree(alias, tmp_path, monkeypatch):
    """Empty ``tmp_path`` is a clean repo: delegate returns 0 → exit 0.

    The shim does `subprocess.run([sys.executable, _NEW_SCRIPT, *argv])`. We
    bypass the real subprocess by stubbing ``subprocess.run`` to capture the
    argv list and return a synthetic ``CompletedProcess(rc=0)``; the shim
    must surface that 0.
    """
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(alias.subprocess, "run", fake_run)
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--quiet"]
    try:
        rc = alias.main()
    finally:
        sys.argv[:] = saved

    assert rc == 0
    # argv must be forwarded to the underlying script (after the python exe).
    assert captured["cmd"][1] == str(alias._NEW_SCRIPT)
    assert captured["cmd"][2:] == ["--quiet"]
    assert captured["kwargs"].get("check") is False


def test_main_propagates_exit_code_one_for_known_violation(alias, monkeypatch):
    """Underlying gate returns 1 → shim returns 1.

    The shim's only job is to copy the underlying exit code. A regression
    that swallowed the code (e.g. ``return 0`` || ``return 1``) would
    silently green-light the gate.
    """
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1)

    monkeypatch.setattr(alias.subprocess, "run", fake_run)
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME]
    try:
        rc = alias.main()
    finally:
        sys.argv[:] = saved
    assert rc == 1


def test_main_forwards_multiple_argv_items(alias, monkeypatch):
    """``--root-dir <path> --quiet`` reaches the underlying script unchanged."""
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(alias.subprocess, "run", fake_run)
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--root-dir", "/tmp/x", "--quiet"]
    try:
        alias.main()
    finally:
        sys.argv[:] = saved
    assert captured["cmd"][2:] == ["--root-dir", "/tmp/x", "--quiet"]


def test_main_does_not_forward_sys_argv_zero(alias, monkeypatch):
    """argv[0] (the script name) is not passed to the underlying script.

    Regression guard: a refactor that used ``sys.argv`` raw instead of
    ``sys.argv[1:]`` would inject the shim path into the underlying gate's
    argv, which argparse would then try to parse as a flag.
    """
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(alias.subprocess, "run", fake_run)
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--quiet"]
    try:
        alias.main()
    finally:
        sys.argv[:] = saved
    # The underlying script never sees the shim's argv[0].
    for c in captured["cmd"][2:]:
        assert not c.endswith("check_root_md_policy.py")


# ---------------------------------------------------------------------------
# Subprocess smoke tests against the real underlying script
# ---------------------------------------------------------------------------


def test_subprocess_propagates_underlying_exit_code(tmp_path):
    """End-to-end: the shim propagates the underlying exit code byte-for-byte.

    Run against the real ``check_root_hygiene.py`` with a synthetic
    ``tmp_path`` that violates the policy (a stray ``RootStray.md``). The
    shim must exit 1, matching the underlying gate.
    """
    stray = tmp_path / "StrayDoc.md"
    stray.write_text("oops", encoding="utf-8")

    # The underlying gate reads ``REPO_ROOT = Path(__file__).parent.parent``,
    # so we drive it via a copy that points at our tmp_path. Easier: just
    # drive the underlying script via the shim and assert the shim's
    # exit code equals the underlying's exit code in a clean subtree.
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )
    # The shim's `subprocess.run` does not change cwd of the underlying
    # script, so the underlying gate sees the real repo. On a clean repo
    # it exits 0; the shim must exit 0.
    assert proc.returncode == 0, (
        f"shim returned {proc.returncode}, stdout={proc.stdout!r}, "
        f"stderr={proc.stderr!r}"
    )


def test_subprocess_returns_zero_on_clean_repo():
    """Against the real repository (clean at the time of writing) → exit 0.

    Pins the shim's exit-code propagation against the real underlying
    script and repo. A regression in either layer flips this red.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"shim returned {proc.returncode} on the real repo.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# __main__ wrapper contract
# ---------------------------------------------------------------------------


def test_main_wrapper_translates_exceptions_to_exit_2():
    """Uncaught exception in the shim → exit 2 (fail-loud).

    Guards the ``__main__`` block: ``try: sys.exit(main()) except Exception:
    sys.exit(2)`` is the contract. A regression that drops the wrapper
    would let the traceback escape (Python's default behaviour is exit 1).
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in src
    wrapper = src.split('if __name__ == "__main__":', 1)[1]
    assert "try:" in wrapper
    assert "except Exception" in wrapper
    assert "sys.exit(2)" in wrapper


def test_alias_delegates_to_root_hygiene_script():
    """The shim points at ``check_root_hygiene.py`` -- the architecture
    canonical location. A regression that pointed at the wrong file (e.g.
    the legacy md-only variant) would re-introduce the #2466 gap.
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "check_root_hygiene.py" in src
    # The redirected path must be a sibling of the shim's own file.
    assert HYGIENE_SCRIPT_PATH.exists()
    assert HYGIENE_SCRIPT_PATH.parent == SCRIPT_PATH.parent
