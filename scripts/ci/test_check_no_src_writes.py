"""
Tests for ``scripts/check_no_src_writes.py`` -- Issue #4163.

``scripts/grid_search_h_si.py`` and ``scripts/sweep_h_ms_coeff.py`` were
operator calibration helpers that grid-searched a physics constant until an
ASHRAE 140 case landed inside its reference band, then wrote the tuned
value back into ``src/``. Both are deleted; this gate closes the class so
the next such helper cannot land.

The gate is AST-based and must tell *write* from *read*, because many
legitimate scripts read source files to verify them and a
"mentions-src" text heuristic would flag all of them. So the suite asserts
three groups:

  * the two real defect shapes (``Path.write_text`` on a ``src/`` path, and
    ``open(path, "w")`` with a hardcoded absolute path) are reported;
  * read-only access, ``tmp/`` writes, and unrelated files are clean;
  * the matchinger's non-obvious cases (f-string paths, ``Path.open`` mode
    keyword, explicit ``continue-on-error``-style read modes) behave.

Fixtures are synthetic files under ``tmp_path``; the suite never scans the
real ``scripts/`` tree except for one final canary.
"""
from __future__ import annotations

import sys

import pytest

SCRIPT_NAME = "check_no_src_writes"


@pytest.fixture
def gate(load_script):
    """Freshly-loaded copy of the no-src-writes gate."""
    return load_script(SCRIPT_NAME)


def _script(tmp_path, body: str, name: str = "helper.py"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Positive: writing into src/ must be reported
# ---------------------------------------------------------------------------


def test_reports_path_write_text_on_rel_src(gate, tmp_path):
    """The `grid_search_h_si.py` shape: rewrite a constant under `src/`."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "def apply(value):\n"
        "    p = Path('src/sim/thermal_model_solvers.rs')\n"
        "    p.write_text(p.read_text().replace('3.45', value))\n",
    )
    findings = gate.check_script(path, rel="helper.py")
    assert len(findings) == 1
    assert "write_text" in findings[0]
    assert "src/sim/thermal_model_solvers.rs" in findings[0]
    assert "RULES.md" in findings[0]


def test_reports_open_write_mode_on_abs_src_path(gate, tmp_path):
    """The `sweep_h_ms_coeff.py` shape: `open(<abs>/src/..., "w")`."""
    path = _script(
        tmp_path,
        "def patch():\n"
        "    with open('/home/alex/Projects/fluxion/src/sim/thermal_model_core.rs', 'w') as f:\n"
        "        f.write('const H_MS: f64 = 2.0;')\n",
    )
    findings = gate.check_script(path, rel="helper.py")
    assert len(findings) == 1
    assert "open(" in findings[0]
    assert "src/sim/thermal_model_core.rs" in findings[0]


def test_reports_fluxion_core_src(gate, tmp_path):
    """`fluxion-core/src/` is a production tree too."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "Path('fluxion-core/src/construction.rs').write_text('x')\n",
    )
    findings = gate.check_script(path, rel="helper.py")
    assert len(findings) == 1
    assert "fluxion-core/src/" in findings[0]


def test_reports_fstring_path(gate, tmp_path):
    """An f-string path still yields a checkable literal prefix."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "def write_all(names):\n"
        "    for n in names:\n"
        "        Path(f'src/sim/{n}.rs').write_text('')\n",
    )
    findings = gate.check_script(path, rel="helper.py")
    assert len(findings) == 1
    assert "src/sim/" in findings[0]


def test_reports_append_mode(gate, tmp_path):
    """`"a"` is a write mode and must be reported."""
    path = _script(
        tmp_path,
        "open('src/physics/mod.rs', 'a').write('# note\\n')\n",
    )
    findings = gate.check_script(path, rel="helper.py")
    assert len(findings) == 1


def test_reports_write_bytes(gate, tmp_path):
    """`write_bytes` is a write, not a read."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "Path('src/lib.rs').write_bytes(b'x')\n",
    )
    findings = gate.check_script(path, rel="helper.py")
    assert len(findings) == 1
    assert "write_bytes" in findings[0]


def test_reports_path_open_mode_keyword(gate, tmp_path):
    """`Path(...).open(mode='w')` with the mode as a keyword is a write."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "Path('src/sim/a.rs').open(mode='w').write('x')\n",
    )
    findings = gate.check_script(path, rel="helper.py")
    assert len(findings) == 1
    assert "Path.open" in findings[0]


def test_reports_every_offending_line(gate, tmp_path):
    """Two writes are two findings, so the author sees the whole picture."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "Path('src/a.rs').write_text('x')\n"
        "Path('src/b.rs').write_text('y')\n",
    )
    assert len(gate.check_script(path, rel="helper.py")) == 2


# ---------------------------------------------------------------------------
# Negative: legitimate scripts must stay clean
# ---------------------------------------------------------------------------


def test_clean_read_only_open_of_src(gate, tmp_path):
    """Many gates READ source files to verify them. That is allowed."""
    path = _script(
        tmp_path,
        "def verify():\n"
        "    with open('src/sim/thermal_model_solvers.rs') as f:\n"
        "        return 'const H_SI' in f.read()\n",
    )
    assert gate.check_script(path, rel="helper.py") == []


def test_clean_read_mode_r(gate, tmp_path):
    """Explicit read mode is clean."""
    path = _script(
        tmp_path,
        "open('src/lib.rs', 'r').read()\n",
    )
    assert gate.check_script(path, rel="helper.py") == []


def test_clean_read_text(gate, tmp_path):
    """`read_text` is a read, not a write."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "Path('src/sim/a.rs').read_text()\n",
    )
    assert gate.check_script(path, rel="helper.py") == []


def test_clean_write_to_tmp(gate, tmp_path):
    """Scratch output under `tmp/` is the legitimate write target."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "Path('tmp/sweep.json').write_text('{}')\n",
    )
    assert gate.check_script(path, rel="helper.py") == []


def test_clean_write_to_other_repo_file(gate, tmp_path):
    """Writing a sibling script or docs file is fine."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "Path('scripts/other.py').write_text('# hi')\n"
        "Path('docs/x.md').write_text('# hi')\n",
    )
    assert gate.check_script(path, rel="helper.py") == []


def test_clean_dynamic_path_not_flagged(gate, tmp_path):
    """A non-literal path is not resolvable statically; not reported here.

    Documented behavior: the gate resolves literal paths only. A dynamic
    writer is a review concern, and its literal call sites would fire the
    gate if any were added.
    """
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "def w(target):\n"
        "    Path(target).write_text('x')\n",
    )
    assert gate.check_script(path, rel="helper.py") == []


def test_clean_unrelated_script(gate, tmp_path):
    """A script that never touches the source tree is trivially clean."""
    path = _script(tmp_path, "import json\nprint(json.dumps({'a': 1}))\n")
    assert gate.check_script(path, rel="helper.py") == []


def test_clean_similar_prefix_that_is_not_src(gate, tmp_path):
    """`srcs/` or `docs/src/` must not false-positive on the `src/` prefix."""
    path = _script(
        tmp_path,
        "from pathlib import Path\n"
        "Path('srcs/data.txt').write_text('x')\n"
        "Path('docs/src/a.md').write_text('x')\n",
    )
    assert gate.check_script(path, rel="helper.py") == []


# ---------------------------------------------------------------------------
# main() surface
# ---------------------------------------------------------------------------


def test_main_returns_one_on_findings(gate, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gate", "--scripts-dir", str(tmp_path)])
    _script(
        tmp_path,
        "from pathlib import Path\nPath('src/a.rs').write_text('x')\n",
    )
    assert gate.main() == 1
    assert "production source tree" in capsys.readouterr().err


def test_main_returns_zero_when_clean(gate, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gate", "--scripts-dir", str(tmp_path)])
    _script(tmp_path, "print('hi')\n")
    assert gate.main() == 0
    assert "OK" in capsys.readouterr().out


def test_main_returns_two_when_dir_missing(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gate"])
    monkeypatch.setattr(
        sys, "argv", ["gate", "--scripts-dir", str(tmp_path / "nope")]
    )
    assert gate.main() == 2


def test_main_honors_scripts_dir_flag(gate, tmp_path, monkeypatch):
    """The scan root is the flag, not the repo default."""
    monkeypatch.setattr(
        sys, "argv", ["gate", "--scripts-dir", str(tmp_path)]
    )
    _script(tmp_path, "print('hi')\n")
    assert gate.main() == 0


# ---------------------------------------------------------------------------
# Non-vacuity / scope guards
# ---------------------------------------------------------------------------


def test_script_module_exposes_expected_surface(gate):
    """Guard against an import fixture that silently yields a stub.

    Handoff §7: a control with no real coverage produced vacuously green
    harnesses twice. Assert the loaded module is the real file.
    """
    assert callable(gate.check_script)
    assert callable(gate.main)
    with open(gate.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert "RULES.md" in src
    assert "FORBIDDEN_PREFIXES" in src


def test_write_mode_detection(gate):
    """`_is_write_mode` must treat only read modes as non-writing."""
    assert gate._is_write_mode("r") is False
    assert gate._is_write_mode("rb") is False
    assert gate._is_write_mode(None) is False
    assert gate._is_write_mode("w") is True
    assert gate._is_write_mode("a") is True
    assert gate._is_write_mode("r+") is True
    assert gate._is_write_mode("w+") is True


def test_deleted_scripts_are_absent(gate, repo_root):
    """The two destructive helpers must stay deleted (Issue #4163)."""
    for name in ("grid_search_h_si.py", "sweep_h_ms_coeff.py"):
        assert not (repo_root / "scripts" / name).exists(), name


def test_real_scripts_tree_is_clean(gate, repo_root):
    """Pin the gate against the live `scripts/` tree.

    The synthetic fixtures prove the matcher works; this proves the
    shipped tree actually passes, so a future regression fails the suite
    rather than waiting for the direct-invocation CI step.
    """
    findings = []
    for path in sorted((repo_root / "scripts").rglob("*.py")):
        findings.extend(
            gate.check_script(path, rel=path.relative_to(repo_root).as_posix())
        )
    assert findings == [], findings
