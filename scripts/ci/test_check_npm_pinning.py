"""Tests for ``scripts/check_npm_pinning.py`` -- Issue #4202.

The gate keeps the deterministic npm install honest: ``npm/package.json``
dependency specs must be exact semver (no ``^``/``~``/ranges),
``npm/package-lock.json`` must exist and mirror the manifest, and
``node-bindings.yml`` must install with ``npm ci`` (never bare
``npm install``).

The tests below drive the checkers against hermetic ``tmp_path`` fixtures
(the ``load_script`` + ``monkeypatch`` pattern from
``test_check_workflow_pin.py``). The end-to-end against the real tree is the
direct ``check_npm_pinning.py`` invocation in ``scripts-tests.yml``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT_NAME = "check_npm_pinning"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_npm_pinning.py``."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, monkeypatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    """Point the gate's module constants at a ``tmp_path`` mock repo and
    return its ``(package.json, package-lock.json, node-bindings.yml)``."""
    npm_dir = tmp_path / "npm"
    npm_dir.mkdir(parents=True, exist_ok=True)
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    package_json = npm_dir / "package.json"
    package_lock = npm_dir / "package-lock.json"
    workflow = wf_dir / "node-bindings.yml"
    monkeypatch.setattr(checker, "NPM_DIR", npm_dir)
    monkeypatch.setattr(checker, "PACKAGE_JSON", package_json)
    monkeypatch.setattr(checker, "PACKAGE_LOCK", package_lock)
    monkeypatch.setattr(checker, "NODE_BINDINGS_WORKFLOW", workflow)
    return package_json, package_lock, workflow


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _compliant_repo(checker, monkeypatch, tmp_path: Path) -> None:
    """A mock repo that satisfies all three checks."""
    package_json, package_lock, workflow = _redirect(
        checker, monkeypatch, tmp_path
    )
    _write_json(package_json, {"devDependencies": {"@napi-rs/cli": "3.10.5"}})
    _write_json(package_lock, {"packages": {"": {"devDependencies": {
        "@napi-rs/cli": "3.10.5"}}}})
    workflow.write_text(
        "      - name: Install npm dependencies\n"
        "        working-directory: npm\n"
        "        run: npm ci\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# check_manifest_exact_pins()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["3.10.5", "3.0.0-alpha.0", "1.2.3+build.1"])
def test_exact_specs_pass(checker, spec):
    assert checker.check_manifest_exact_pins(
        {"devDependencies": {"@napi-rs/cli": spec}}) == []


@pytest.mark.parametrize("spec", [
    "^3.0.0-alpha.0", "~3.10.5", ">=3.0.0", "^3 || ^4", "3.10.x", "*",
    "latest", "file:../sibling", "git+https://example.com/x.git",
])
def test_range_specs_fail(checker, spec):
    findings = checker.check_manifest_exact_pins(
        {"dependencies": {"some-pkg": spec}})
    assert len(findings) == 1
    assert "some-pkg" in findings[0]
    assert "exact" in findings[0]


def test_all_dep_sections_checked(checker):
    manifest = {
        "dependencies": {"a": "^1.0.0"},
        "devDependencies": {"b": "2.0.0"},
        "optionalDependencies": {"c": "~3.0.0"},
        "peerDependencies": {"d": ">=4.0.0"},
    }
    findings = checker.check_manifest_exact_pins(manifest)
    assert len(findings) == 3  # b is exact; a, c, d fail


# ---------------------------------------------------------------------------
# check_lockfile_in_sync()
# ---------------------------------------------------------------------------


def test_in_sync_lockfile_passes(checker):
    manifest = {"devDependencies": {"@napi-rs/cli": "3.10.5"}}
    lock = {"packages": {"": {"devDependencies": {"@napi-rs/cli": "3.10.5"}}}}
    assert checker.check_lockfile_in_sync(manifest, lock) == []


def test_drifted_lockfile_fails(checker):
    manifest = {"devDependencies": {"@napi-rs/cli": "3.10.5"}}
    lock = {"packages": {"": {"devDependencies": {"@napi-rs/cli": "3.10.4"}}}}
    findings = checker.check_lockfile_in_sync(manifest, lock)
    assert len(findings) == 1
    assert "3.10.4" in findings[0] and "3.10.5" in findings[0]


def test_lockfile_missing_root_entry_fails(checker):
    findings = checker.check_lockfile_in_sync({"dependencies": {"a": "1.0.0"}},
                                              {"packages": {}})
    assert len(findings) == 1
    assert 'packages[""]' in findings[0]


# ---------------------------------------------------------------------------
# check_workflow_uses_npm_ci()
# ---------------------------------------------------------------------------


def test_npm_ci_step_passes(checker):
    text = ("      - name: Install npm dependencies\n"
            "        working-directory: npm\n"
            "        run: npm ci\n")
    assert checker.check_workflow_uses_npm_ci(text) == []


def test_bare_npm_install_fails_with_line(checker):
    text = ("      - name: Install npm dependencies\n"
            "        working-directory: npm\n"
            "        run: npm install\n")
    findings = checker.check_workflow_uses_npm_ci(text)
    assert len(findings) == 1
    assert "node-bindings.yml:3:" in findings[0]
    assert "npm ci" in findings[0]


def test_commented_npm_install_skipped(checker):
    text = "#        run: npm install\n        run: npm ci\n"
    assert checker.check_workflow_uses_npm_ci(text) == []


# ---------------------------------------------------------------------------
# main() exit-code contract
# ---------------------------------------------------------------------------


def test_main_exit_0_on_compliant_repo(checker, monkeypatch, tmp_path,
                                       capsys):
    _compliant_repo(checker, monkeypatch, tmp_path)
    assert checker.main([]) == 0
    assert "OK:" in capsys.readouterr().out


def test_main_exit_1_on_caret_spec(checker, monkeypatch, tmp_path, capsys):
    _compliant_repo(checker, monkeypatch, tmp_path)
    package_json = tmp_path / "npm" / "package.json"
    _write_json(package_json,
                {"devDependencies": {"@napi-rs/cli": "^3.0.0-alpha.0"}})
    assert checker.main([]) == 1
    assert "npm pinning drift" in capsys.readouterr().err


def test_main_exit_1_on_missing_lockfile(checker, monkeypatch, tmp_path,
                                         capsys):
    _compliant_repo(checker, monkeypatch, tmp_path)
    (tmp_path / "npm" / "package-lock.json").unlink()
    assert checker.main([]) == 1
    assert "package-lock.json is missing" in capsys.readouterr().err


def test_main_exit_1_on_bare_npm_install(checker, monkeypatch, tmp_path,
                                         capsys):
    _compliant_repo(checker, monkeypatch, tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "node-bindings.yml"
    workflow.write_text("        run: npm install\n", encoding="utf-8")
    assert checker.main([]) == 1
    assert "bare" in capsys.readouterr().err


def test_main_exit_2_when_package_json_missing(checker, monkeypatch,
                                               tmp_path, capsys):
    _redirect(checker, monkeypatch, tmp_path)  # dirs exist, no files
    assert checker.main([]) == 2
    assert "ERROR" in capsys.readouterr().err
