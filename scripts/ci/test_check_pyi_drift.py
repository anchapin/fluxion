"""
Tests for ``scripts/check_pyi_drift.py`` -- Issue #2509.

The drift gate ensures that every ``#[pyclass]`` registered in the
PyO3 ``fluxion`` module has a corresponding ``class`` entry in
``fluxion.pyi`` (and the same for ``#[pyfunction]`` and exception types).
A regression that adds a new Rust class but forgets the ``.pyi`` stub
fails the build until the stub is added.

The script's load-bearing pieces are:

* ``_build_pyclass_name_map`` -- parse ``#[pyclass(name = "...")]`` attrs.
* ``registered_classes`` -- list of names registered in the ``fluxion`` /
  ``multi_zone`` modules.
* ``registered_functions`` -- top-level ``#[pyfunction]`` names.
* ``registered_exceptions`` -- exception types added via ``m.add("...", ...)``.
* ``pyi_classes`` / ``pyi_functions`` -- what the ``.pyi`` declares.
* ``main()`` -- exit 0 on clean, 1 on drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_pyi_drift"


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the drift gate."""
    return load_script(SCRIPT_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _redirect(checker, tmp_path: Path, monkeypatch, *, lib_rs: str,
              bindings_rs: str = "", pyi: str = "") -> None:
    """Point the script's four path constants at synthetic files.

    The script declares ``LIB_RS``, ``BINDINGS_RS``, ``MOD_RS``, ``PYI`` at
    module scope, so we substitute them with tmp_path fixtures.
    """
    lib_path = tmp_path / "src" / "lib.rs"
    bindings_path = tmp_path / "src" / "python" / "bindings.rs"
    pyi_path = tmp_path / "fluxion.pyi"

    _write(lib_path, lib_rs)
    _write(bindings_path, bindings_rs)
    _write(pyi_path, pyi)

    monkeypatch.setattr(checker, "LIB_RS", lib_path)
    monkeypatch.setattr(checker, "BINDINGS_RS", bindings_path)
    monkeypatch.setattr(checker, "PYI", pyi_path)
    # MOD_RS is referenced by ``_read`` but the script only reads it when
    # ``registered_classes`` is called for the multi_zone module. We point
    # it at a non-existent path so the test surfaces if that path is read.
    monkeypatch.setattr(checker, "MOD_RS", tmp_path / "src" / "python" / "mod.rs")


# ---------------------------------------------------------------------------
# _build_pyclass_name_map
# ---------------------------------------------------------------------------


def test_pyclass_name_map_extracts_explicit_name(checker, tmp_path, monkeypatch):
    """``#[pyclass(name = "X")]` on a struct ``Y`` → ``Y -> X``."""
    src = '''\
#[pyclass(name = "Foo")]
pub struct Foo;
'''
    _write(tmp_path / "src" / "api" / "parameters.rs", "")
    _write(tmp_path / "src" / "python" / "hvac_bindings.rs", "")
    _write(tmp_path / "src" / "python" / "model_bindings.rs", "")
    _write(tmp_path / "src" / "python" / "multi_node_bindings.rs", "")
    _write(tmp_path / "src" / "python" / "osm_bindings.rs", "")
    monkeypatch.setattr(checker, "PYCLASS_SOURCES", [
        tmp_path / "src" / "lib.rs",
    ])
    _write(tmp_path / "src" / "lib.rs", src)
    mapping = checker._build_pyclass_name_map()
    assert mapping.get("Foo") == "Foo"


def test_pyclass_name_map_falls_back_to_strip_py_prefix(checker, tmp_path, monkeypatch):
    """``#[pyclass]`` on ``PyBar`` → ``PyBar -> Bar`` (default)."""
    src = '''\
#[pyclass]
pub struct PyBar;
'''
    monkeypatch.setattr(checker, "PYCLASS_SOURCES", [
        tmp_path / "src" / "lib.rs",
    ])
    _write(tmp_path / "src" / "lib.rs", src)
    mapping = checker._build_pyclass_name_map()
    assert mapping.get("PyBar") == "Bar"


# ---------------------------------------------------------------------------
# registered_classes — happy path
# ---------------------------------------------------------------------------


def test_registered_classes_returns_class_names(checker, tmp_path, monkeypatch):
    """``add_class::<Foo>`` in ``#[pymodule] fn fluxion`` → ``['Foo']``."""
    lib = '''\
#[pymodule]
pub fn fluxion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Foo>()?;
    m.add_class::<Bar>()?;
    Ok(())
}
'''
    bindings = '''\
#[pymodule]
pub fn multi_zone(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Baz>()?;
    Ok(())
}
'''
    _redirect(checker, tmp_path, monkeypatch, lib_rs=lib, bindings_rs=bindings)
    classes = checker.registered_classes()
    assert "Foo" in classes
    assert "Bar" in classes
    assert "Baz" in classes


# ---------------------------------------------------------------------------
# registered_functions
# ---------------------------------------------------------------------------


def test_registered_functions_uses_wrap_pyfunction(checker, tmp_path, monkeypatch):
    """``wrap_pyfunction!(foo, m)`` → ``['foo']``."""
    lib = '''\
#[pymodule]
pub fn fluxion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(compute_load, m)?)?;
    Ok(())
}
'''
    bindings = '''\
#[pymodule]
pub fn multi_zone(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_multi_zone_model, m)?)?;
    Ok(())
}
'''
    _redirect(checker, tmp_path, monkeypatch, lib_rs=lib, bindings_rs=bindings)
    funcs = checker.registered_functions()
    assert "run_simulation" in funcs
    assert "compute_load" in funcs
    assert "create_multi_zone_model" in funcs


def test_registered_functions_strips_module_prefix(checker, tmp_path, monkeypatch):
    """``wrap_pyfunction!(crate::mod::fn)`` → ``['fn']`` (last path segment)."""
    lib = '''\
#[pymodule]
pub fn fluxion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sim::orchestrator::step, m)?)?;
    Ok(())
}
'''
    _redirect(checker, tmp_path, monkeypatch, lib_rs=lib, bindings_rs="")
    funcs = checker.registered_functions()
    assert "step" in funcs


# ---------------------------------------------------------------------------
# registered_exceptions
# ---------------------------------------------------------------------------


def test_registered_exceptions_extracts_m_add(checker, tmp_path, monkeypatch):
    """``m.add("FluxionError", ...)` → ``['FluxionError']``."""
    lib = '''\
#[pymodule]
pub fn fluxion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FluxionError", _py.get_type_bound::<FluxionError>())?;
    m.add("ValidationError", _py.get_type_bound::<ValidationError>())?;
    Ok(())
}
'''
    _redirect(checker, tmp_path, monkeypatch, lib_rs=lib, bindings_rs="")
    excs = checker.registered_exceptions()
    assert "FluxionError" in excs
    assert "ValidationError" in excs


# ---------------------------------------------------------------------------
# pyi_classes / pyi_functions
# ---------------------------------------------------------------------------


def test_pyi_classes_extracts_top_level_class_names(checker, tmp_path, monkeypatch):
    """``class Foo:`` declarations are captured, indented ones are not."""
    pyi = '''\
class Foo:
    pass

class Bar:
    """docstring."""

    class Nested:
        pass
'''
    _redirect(checker, tmp_path, monkeypatch, lib_rs="", pyi=pyi)
    classes = checker.pyi_classes()
    assert "Foo" in classes
    assert "Bar" in classes
    assert "Nested" not in classes  # indented class is not a top-level def


def test_pyi_functions_extracts_top_level_def_names(checker, tmp_path, monkeypatch):
    """``def foo():`` declarations are captured."""
    pyi = '''\
def run_simulation():
    pass

def compute_load():
    pass
'''
    _redirect(checker, tmp_path, monkeypatch, lib_rs="", pyi=pyi)
    funcs = checker.pyi_functions()
    assert "run_simulation" in funcs
    assert "compute_load" in funcs


# ---------------------------------------------------------------------------
# main() — drift detection
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_pyi_complete(checker, tmp_path, monkeypatch, capsys):
    """Every registered class/function/exception has a ``.pyi`` entry → exit 0."""
    lib = '''\
#[pymodule]
pub fn fluxion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Foo>()?;
    m.add_function(wrap_pyfunction!(run_sim, m)?)?;
    m.add("FluxionError", _py.get_type_bound::<FluxionError>())?;
    Ok(())
}
'''
    bindings = '''\
#[pymodule]
pub fn multi_zone(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Baz>()?;
    Ok(())
}
'''
    pyi = '''\
class Foo:
    pass

class Baz:
    pass

class FluxionError(Exception):
    pass

def run_sim():
    pass
'''
    _redirect(checker, tmp_path, monkeypatch,
              lib_rs=lib, bindings_rs=bindings, pyi=pyi)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected exit 0, got {rc}\noutput:\n{out}"
    assert "OK" in out
    assert "in sync" in out


def test_main_returns_one_when_class_missing(checker, tmp_path, monkeypatch, capsys):
    """A registered class with no ``.pyi`` stub → exit 1."""
    lib = '''\
#[pymodule]
pub fn fluxion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Foo>()?;
    m.add_class::<Bar>()?;
    Ok(())
}
'''
    bindings = ""
    pyi = '''\
class Foo:
    pass
'''
    _redirect(checker, tmp_path, monkeypatch,
              lib_rs=lib, bindings_rs=bindings, pyi=pyi)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert "DRIFT" in out
    assert "Bar" in out
    assert "Missing class" in out


def test_main_returns_one_when_function_missing(checker, tmp_path, monkeypatch, capsys):
    """A registered function with no ``.pyi`` stub → exit 1."""
    lib = '''\
#[pymodule]
pub fn fluxion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sim, m)?)?;
    m.add_function(wrap_pyfunction!(compute_load, m)?)?;
    Ok(())
}
'''
    pyi = '''\
def run_sim():
    pass
'''
    _redirect(checker, tmp_path, monkeypatch, lib_rs=lib, pyi=pyi)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert "DRIFT" in out
    assert "compute_load" in out
    assert "Missing function" in out


def test_main_returns_one_when_exception_missing(checker, tmp_path, monkeypatch, capsys):
    """A registered exception with no ``.pyi`` class → exit 1."""
    lib = '''\
#[pymodule]
pub fn fluxion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FluxionError", _py.get_type_bound::<FluxionError>())?;
    Ok(())
}
'''
    pyi = ""
    _redirect(checker, tmp_path, monkeypatch, lib_rs=lib, pyi=pyi)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert "DRIFT" in out
    assert "FluxionError" in out
    assert "Missing exception" in out


def test_main_returns_two_when_lib_rs_missing(checker, tmp_path, monkeypatch, capsys):
    """Missing ``src/lib.rs`` → exit 2 (script-error)."""
    monkeypatch.setattr(checker, "LIB_RS", tmp_path / "no" / "lib.rs")
    monkeypatch.setattr(checker, "BINDINGS_RS", tmp_path / "no" / "bindings.rs")
    monkeypatch.setattr(checker, "PYI", tmp_path / "no" / "fluxion.pyi")
    try:
        rc = checker.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    assert rc == 2


# ---------------------------------------------------------------------------
# PYCLASS_SOURCES pin
# ---------------------------------------------------------------------------


def test_pyclass_sources_includes_expected_files(checker):
    """The script scans all known binding source files."""
    sources = {p.name for p in checker.PYCLASS_SOURCES}
    assert "lib.rs" in sources
    assert "bindings.rs" in sources
    assert "hvac_bindings.rs" in sources
    assert "model_bindings.rs" in sources
    assert "multi_node_bindings.rs" in sources
    assert "osm_bindings.rs" in sources
    assert "parameters.rs" in sources
