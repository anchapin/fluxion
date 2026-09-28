"""
Tests for ``scripts/check_no_destructive_scripts.py`` — Issue #4163.

The guard fails CI if any Python script under ``scripts/`` writes to a
path under ``src/sim/`` or ``fluxion-core/src/`` (the destructive tuning
scripts ``grid_search_h_si.py`` / ``sweep_h_ms_coeff.py`` did exactly
this and are deleted).
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_no_destructive_scripts"


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def _write_script(scripts_dir: Path, name: str, body: str) -> Path:
    p = scripts_dir / name
    p.write_text(body, encoding="utf-8")
    return p


def _mock_repo(tmp_path: Path) -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    return scripts_dir


# ---------------------------------------------------------------------------
# Clean cases
# ---------------------------------------------------------------------------


def test_clean_tree_has_no_violations(checker, tmp_path):
    scripts_dir = _mock_repo(tmp_path)
    _write_script(
        scripts_dir,
        "analyze.py",
        'from pathlib import Path\n'
        'p = Path("src/sim/thermal_model.rs")\n'
        'print(p.read_text())\n',
    )
    assert checker.scan_scripts(tmp_path) == []


def test_comment_mentioning_protected_path_is_not_a_violation(checker, tmp_path):
    scripts_dir = _mock_repo(tmp_path)
    _write_script(
        scripts_dir,
        "docs.py",
        '# See src/sim/thermal_model.rs for the 5R1C network.\n'
        'NOTE = "fluxion-core/src/construction.rs holds the film coefficients"\n',
    )
    assert checker.scan_scripts(tmp_path) == []


def test_write_outside_protected_prefix_is_not_a_violation(checker, tmp_path):
    scripts_dir = _mock_repo(tmp_path)
    _write_script(
        scripts_dir,
        "gen.py",
        'from pathlib import Path\n'
        'Path("tests/reference_data/out.json").write_text("{}")\n',
    )
    assert checker.scan_scripts(tmp_path) == []


# ---------------------------------------------------------------------------
# Planted violations
# ---------------------------------------------------------------------------


def test_write_text_to_src_sim_is_a_violation(checker, tmp_path):
    scripts_dir = _mock_repo(tmp_path)
    _write_script(
        scripts_dir,
        "tune.py",
        'from pathlib import Path\n'
        'Path("src/sim/thermal_model_solvers.rs").write_text("tuned")\n',
    )
    violations = checker.scan_scripts(tmp_path)
    assert len(violations) == 1
    assert violations[0].script == "tune.py"
    assert "src/sim/thermal_model_solvers.rs" in violations[0].detail


def test_write_text_to_fluxion_core_src_is_a_violation(checker, tmp_path):
    scripts_dir = _mock_repo(tmp_path)
    _write_script(
        scripts_dir,
        "tune2.py",
        'from pathlib import Path\n'
        'Path("fluxion-core/src/construction.rs").write_text("tuned")\n',
    )
    violations = checker.scan_scripts(tmp_path)
    assert len(violations) == 1
    assert "fluxion-core/src/construction.rs" in violations[0].detail


def test_open_write_mode_to_protected_path_is_a_violation(checker, tmp_path):
    scripts_dir = _mock_repo(tmp_path)
    _write_script(
        scripts_dir,
        "tune3.py",
        'with open("src/sim/foo.rs", "w") as f:\n'
        '    f.write("x")\n',
    )
    violations = checker.scan_scripts(tmp_path)
    assert len(violations) == 1


def test_open_read_mode_is_not_a_violation(checker, tmp_path):
    scripts_dir = _mock_repo(tmp_path)
    _write_script(
        scripts_dir,
        "read.py",
        'with open("src/sim/foo.rs") as f:\n'
        '    data = f.read()\n',
    )
    assert checker.scan_scripts(tmp_path) == []


# ---------------------------------------------------------------------------
# Live repo
# ---------------------------------------------------------------------------


def test_live_scripts_tree_has_no_violations(checker):
    repo_root = Path(__file__).resolve().parent.parent.parent
    violations = checker.scan_scripts(repo_root)
    assert violations == [], (
        "Destructive script writes detected: "
        + "; ".join(f"{v.script}:{v.lineno}" for v in violations)
    )
