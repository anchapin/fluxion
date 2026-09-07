"""
Regression tests for ``scripts/coverage_critical_paths.py`` (Issue #3395).

The pre-#3395 ``CRITICAL_PATHS`` mapping declared two stale references that
silently dropped coverage data:

- ``src/sim/thermal_model_data.rs`` (single file) was replaced by a
  ``src/sim/thermal_model_data/`` directory by PR #3034 (Issue #2878). The
  glob-matcher would skip the directory form, so coverage for every file
  under the new module was silently lost from the ``conduction_zone``
  bucket.
- ``src/sim/thermal_model_hvac.rs`` was removed by PR #3020 (Issue #2896)
  and replaced with ``src/sim/thermal_model_solvers.rs``. The
  ``hvac_zone`` glob matched nothing.

These tests pin the contract that every glob in ``CRITICAL_PATHS`` must
resolve to at least one existing file under the repo root, so a future
rename that leaves a stale glob behind is caught at CI time rather than
silently dropping coverage.

The test uses ``fnmatch`` directly (same matcher the production script
uses) so a glob that resolves only under a different matcher is rejected.
"""

from __future__ import annotations

import fnmatch
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "coverage_critical_paths.py"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_critical_paths() -> dict[str, list[str]]:
    """Load ``CRITICAL_PATHS`` from the script as a fresh module.

    Registers the module in ``sys.modules`` BEFORE ``exec_module`` so the
    ``@dataclass`` decorator (which inspects ``sys.modules[cls.__module__]``
    to resolve string annotations introduced by
    ``from __future__ import annotations``) can resolve forward
    references. Without this, ``Optional[...]``-annotated dataclass
    fields raise ``AttributeError: 'NoneType' object has no attribute
    '__dict__'`` on class creation.
    """
    import sys as _sys

    name = "coverage_critical_paths"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    _sys.modules.setdefault(name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CRITICAL_PATHS


def _glob_resolves_to_files(repo_root: Path, glob: str) -> list[Path]:
    """Replicate the production glob-matcher and return the matched files.

    Mirrors ``scripts/coverage_critical_paths.py::_matches_any``: globs
    ending in ``/**`` match the prefix and any nested path; other globs
    use ``fnmatch.fnmatch`` directly. We enumerate every file under the
    repo root (via ``rglob``) and ask whether ``fnmatch`` would match
    that file's repo-relative path. This stays consistent with the
    production matcher even though the test enumerates a wider file set
    than the production bucketing pass.
    """
    matches: list[Path] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_root).as_posix()
        if glob.endswith("/**"):
            prefix = glob[:-3]
            if rel == prefix or rel.startswith(prefix + "/"):
                matches.append(p)
        elif fnmatch.fnmatch(rel, glob):
            matches.append(p)
    return matches


def test_critical_paths_glob_resolves_to_at_least_one_file():
    """Every glob in ``CRITICAL_PATHS`` must resolve to at least one file.

    Pre-#3395 the ``conduction_zone`` and ``hvac_zone`` paths contained
    stale globs (``src/sim/thermal_model_data.rs`` and
    ``src/sim/thermal_model_hvac.rs``) that no longer existed on disk,
    silently dropping their coverage from the ratchet gate.
    """
    critical_paths = _load_critical_paths()
    assert "conduction_zone" in critical_paths
    assert "hvac_zone" in critical_paths

    unresolved: list[tuple[str, str]] = []
    for path_name, globs in critical_paths.items():
        for glob in globs:
            matches = _glob_resolves_to_files(REPO_ROOT, glob)
            if not matches:
                unresolved.append((path_name, glob))

    assert unresolved == [], (
        "The following CRITICAL_PATHS globs do not resolve to any file "
        "under the repo root (the production glob-matcher would silently "
        "drop coverage for these paths):\n"
        + "\n".join(f"  {p}: {g}" for p, g in unresolved)
    )


def test_critical_paths_known_stale_globs_are_gone():
    """The pre-#3395 stale globs must NOT reappear in any future edit.

    - ``src/sim/thermal_model_data.rs`` (single file) was replaced by the
      directory form ``src/sim/thermal_model_data/**`` in PR #3034 (Issue
      #2878); including the bare-file form here would silently match
      nothing if the directory is renamed.
    - ``src/sim/thermal_model_hvac.rs`` was removed in PR #3020 (Issue
      #2896); the canonical HVAC replacement is
      ``src/sim/thermal_model_solvers.rs``.
    """
    critical_paths = _load_critical_paths()
    for path_name, globs in critical_paths.items():
        for glob in globs:
            assert "thermal_model_data.rs" not in glob, (
                f"Stale glob {glob!r} in {path_name!r}: PR #3034 "
                "(Issue #2878) replaced the file with a directory form"
            )
            assert "thermal_model_hvac.rs" not in glob, (
                f"Stale glob {glob!r} in {path_name!r}: PR #3020 "
                "(Issue #2896) removed this file; canonical replacement "
                "is src/sim/thermal_model_solvers.rs"
            )


def test_fnmatch_glob_matches_existing_file(tmp_path):
    """Sanity check that ``fnmatch`` resolves a literal file glob."""
    fake_repo = tmp_path
    (fake_repo / "src").mkdir()
    (fake_repo / "src" / "foo.rs").write_text("// present")
    matches = _glob_resolves_to_files(fake_repo, "src/foo.rs")
    assert len(matches) == 1
    assert matches[0].name == "foo.rs"


def test_fnmatch_double_star_recursive_walk(tmp_path):
    """Sanity check that ``**`` patterns walk recursively."""
    fake_repo = tmp_path
    (fake_repo / "src" / "nested").mkdir(parents=True)
    (fake_repo / "src" / "nested" / "deep.rs").write_text("// deep")
    matches = _glob_resolves_to_files(fake_repo, "src/**")
    # Returns all files under the directory tree.
    assert any(m.name == "deep.rs" for m in matches)


def test_unresolved_glob_yields_no_match(tmp_path):
    """A glob pointing at a non-existent path returns zero matches.

    Documents the silent-drop pathology: this is exactly what the production
    script did for ``src/sim/thermal_model_data.rs`` after PR #3034
    removed it. The regression test above prevents that from happening
    again; this test pins the underlying matcher semantics.
    """
    fake_repo = tmp_path
    (fake_repo / "src").mkdir()
    matches = _glob_resolves_to_files(fake_repo, "src/thermal_model_data.rs")
    assert matches == []