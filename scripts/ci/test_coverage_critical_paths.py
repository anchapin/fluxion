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
import os
from functools import cache
from pathlib import Path

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


# Directories that never contain sources matched by ``CRITICAL_PATHS``
# globs but can hold 100k+ generated entries on a developer checkout
# (measured: .git = 60k files, target/ = 137k files). Walking them once
# per glob made this file take >300s; pruning them keeps the whole file
# under a second while remaining filesystem-truthful for real sources.
_PRUNED_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
        "__pycache__",
    }
)


@cache
def _enumerate_repo_files(repo_root: Path) -> tuple[str, ...]:
    """Enumerate repo-relative file paths under ``repo_root`` exactly once.

    ENUMERATE-ONCE RULE: this module must never walk the repository
    per-glob. ``CRITICAL_PATHS`` contains 17 globs and pre-#3425 each
    glob triggered a full ``rglob("*")`` walk (208k+ entries, 12s each
    on this repo), which made the Scripts Tests CI job flaky-by-timeout.
    All glob tests share this cached enumeration; if you need the file
    list, call this function — do not reintroduce ``rglob``/``os.walk``
    loops elsewhere in this module.

    Prunes VCS/build/dependency directories (see ``_PRUNED_DIR_NAMES``).
    Like the pre-#3425 ``rglob("*")`` walk this reports untracked files
    (filesystem truth), so a freshly added-but-uncommitted source file
    still resolves.
    """
    rel_files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _PRUNED_DIR_NAMES]
        for name in filenames:
            rel_files.append(
                Path(dirpath, name).relative_to(repo_root).as_posix()
            )
    return tuple(rel_files)


def _glob_resolves_to_files(repo_root: Path, glob: str) -> list[Path]:
    """Replicate the production glob-matcher and return the matched files.

    Mirrors ``scripts/coverage_critical_paths.py::_matches_any``: globs
    ending in ``/**`` match the prefix and any nested path; other globs
    use ``fnmatch.fnmatch`` directly. The candidate file set comes from
    the shared cached enumeration (see ``_enumerate_repo_files``), which
    stays consistent with the production matcher even though the test
    enumerates a wider file set than the production bucketing pass.
    """
    matches: list[Path] = []
    for rel in _enumerate_repo_files(repo_root):
        if glob.endswith("/**"):
            prefix = glob[:-3]
            if rel == prefix or rel.startswith(prefix + "/"):
                matches.append(repo_root / rel)
        elif fnmatch.fnmatch(rel, glob):
            matches.append(repo_root / rel)
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


# ---------------------------------------------------------------------------
# parse_lcov / bucket_coverage / evaluate_gate — the production pipeline
# (Issue #3427: the executable logic had zero tests; only the CRITICAL_PATHS
# data table was covered above)
# ---------------------------------------------------------------------------


def _load_cov_module():
    """Load the full ``coverage_critical_paths`` module (not just the
    CRITICAL_PATHS table) so the parse/bucket/gate pipeline is testable.

    Same registration dance as ``_load_critical_paths`` above: register in
    ``sys.modules`` BEFORE ``exec_module`` so ``@dataclass`` can resolve
    string annotations from ``from __future__ import annotations``.
    """
    import sys as _sys

    name = "coverage_critical_paths"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    _sys.modules.setdefault(name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_lcov(tmp_path: Path, records: list[dict]) -> Path:
    """Write a synthetic lcov.info from record dicts.

    Each dict maps LCOV keys (SF/LF/LH/BRF/BRH) to values, e.g.
    ``{"SF": "src/a.rs", "LF": 100, "LH": 80}``. Emitted verbatim in the
    order given, each followed by ``end_of_record``.
    """
    lines: list[str] = []
    for rec in records:
        for key, value in rec.items():
            lines.append(f"{key}:{value}")
        lines.append("end_of_record")
    target = tmp_path / "lcov.info"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def test_parse_lcov_extracts_expected_numbers(tmp_path):
    """LF/LH/BRF/BRH round-trip; absolute SF: paths normalize repo-relative.

    The ``/src/`` marker-strip branch of ``_repo_relative`` is exercised
    by the absolute-path record — llvm-cov emits such prefixes in CI.
    """
    mod = _load_cov_module()
    lcov = _write_lcov(
        tmp_path,
        [
            {
                "SF": "src/sim/ventilation.rs",
                "LF": 100,
                "LH": 80,
                "BRF": 10,
                "BRH": 5,
            },
            {
                "SF": "/ci/checkout/fluxion-core/src/weather/solar_position.rs",
                "LF": 50,
                "LH": 25,
                "BRF": 4,
                "BRH": 4,
            },
        ],
    )
    files = mod.parse_lcov(lcov)
    assert len(files) == 2
    assert files[0].path == "src/sim/ventilation.rs"
    assert (files[0].lines_found, files[0].lines_hit) == (100, 80)
    assert (files[0].branches_found, files[0].branches_hit) == (10, 5)
    assert files[1].path == "fluxion-core/src/weather/solar_position.rs"
    assert files[1].line_pct == 50.0


def test_parse_lcov_skips_zero_instrumented_files(tmp_path):
    """A record with LF:0 adds noise without signal and is skipped."""
    mod = _load_cov_module()
    lcov = _write_lcov(
        tmp_path,
        [
            {"SF": "src/empty.rs", "LF": 0, "LH": 0},
            {"SF": "src/full.rs", "LF": 10, "LH": 10},
        ],
    )
    files = mod.parse_lcov(lcov)
    assert [f.path for f in files] == ["src/full.rs"]


def test_parse_lcov_missing_file_raises(tmp_path):
    mod = _load_cov_module()
    import pytest

    with pytest.raises(FileNotFoundError):
        mod.parse_lcov(tmp_path / "nope.info")


def test_bucket_coverage_assigns_file_to_multiple_paths():
    """A file matching two paths contributes to both buckets (+ overall).

    ``fluxion-core/src/weather/**`` is deliberately on both the solar and
    ventilation paths, and ``thermal_model_solvers.rs`` on both the
    conduction and HVAC paths — the CRITICAL_PATHS comment documents this
    as the intended data-flow shape (ARCHITECTURE.md).
    """
    mod = _load_cov_module()
    weather = mod.FileCoverage(
        path="fluxion-core/src/weather/site.rs", lines_found=10, lines_hit=10
    )
    solvers = mod.FileCoverage(
        path="src/sim/thermal_model_solvers.rs", lines_found=100, lines_hit=50
    )
    unrelated = mod.FileCoverage(
        path="docs/README.md", lines_found=5, lines_hit=0
    )
    reports = mod.bucket_coverage([weather, solvers, unrelated])

    weather_paths = {
        name
        for name, rep in reports.items()
        if name != "overall" and weather in rep.files
    }
    assert weather_paths == {"weather_solar", "weather_ventilation"}

    solvers_paths = {
        name
        for name, rep in reports.items()
        if name != "overall" and solvers in rep.files
    }
    assert solvers_paths == {"conduction_zone", "hvac_zone"}

    # Unrelated files only land in overall.
    for name, rep in reports.items():
        if name != "overall":
            assert unrelated not in rep.files
    assert len(reports["overall"].files) == 3
    # Aggregation: hvac_zone sees only the solvers file.
    assert reports["hvac_zone"].lines_found == 100
    assert reports["hvac_zone"].line_pct == 50.0


def test_evaluate_gate_passes_at_floor_and_trips_beyond_tolerance(capsys):
    """Ratchet: 80% baseline × (1 − 1%) = 79.2% floor.

    At/above the floor passes; below it fails. Both dimensions are
    enforced independently (#2533).
    """
    mod = _load_cov_module()
    at_floor = mod.FileCoverage(
        path="fluxion-core/src/weather/site.rs",
        lines_found=100,
        lines_hit=80,  # exactly the baseline
        branches_found=10,
        branches_hit=8,
    )
    reports = mod.bucket_coverage([at_floor])
    baseline = {
        "paths": {"weather_solar": {"line": 80.0, "branch": 80.0}}
    }
    assert mod.evaluate_gate(reports, baseline, tolerance=0.01) == []

    regressed = mod.FileCoverage(
        path="fluxion-core/src/weather/site.rs",
        lines_found=100,
        lines_hit=78,  # below 79.2 floor
        branches_found=10,
        branches_hit=8,
    )
    reports_bad = mod.bucket_coverage([regressed])
    failures = mod.evaluate_gate(reports_bad, baseline, tolerance=0.01)
    assert len(failures) == 1
    assert "weather_solar: line coverage" in failures[0]
    assert "fell below" in failures[0]


def test_evaluate_gate_branch_baseline_with_zero_instrumented_branches():
    """Set branch baseline + current run instrumented 0 branches fails loud
    (CI forgot ``--branch-coverage``) rather than silently passing."""
    mod = _load_cov_module()
    no_branches = mod.FileCoverage(
        path="fluxion-core/src/weather/site.rs",
        lines_found=100,
        lines_hit=100,  # line dimension healthy
    )
    reports = mod.bucket_coverage([no_branches])
    baseline = {"paths": {"weather_solar": {"line": 80.0, "branch": 50.0}}}
    failures = mod.evaluate_gate(reports, baseline, tolerance=0.01)
    assert any("instrumented 0 branches" in f for f in failures)


def test_evaluate_gate_unenforced_baseline_emits_notice_not_failure(capsys):
    """Baseline 0.0/absent per dimension = unenforced: notice, no failure."""
    mod = _load_cov_module()
    covered = mod.FileCoverage(
        path="fluxion-core/src/weather/site.rs", lines_found=10, lines_hit=1
    )
    reports = mod.bucket_coverage([covered])
    assert mod.evaluate_gate(reports, baseline={}, tolerance=0.01) == []
    assert "unenforced" in capsys.readouterr().out


def test_main_gate_exit_codes(tmp_path, monkeypatch, capsys):
    """End-to-end through ``main()``: exit 0 at baseline, exit 1 on
    regression, with ``--json`` emitting the computed metrics."""
    import json as _json
    import sys as _sys

    mod = _load_cov_module()
    lcov = _write_lcov(
        tmp_path,
        [
            {
                "SF": "fluxion-core/src/weather/site.rs",
                "LF": 100,
                "LH": 80,
            }
        ],
    )
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(
        _json.dumps({"paths": {"weather_solar": {"line": 80.0, "branch": 0.0}}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "coverage_critical_paths.py",
            "--lcov",
            str(lcov),
            "--baseline",
            str(baseline_file),
            "--gate",
        ],
    )
    assert mod.main() == 0

    regressed = _write_lcov(
        tmp_path / "sub",
        [{"SF": "fluxion-core/src/weather/site.rs", "LF": 100, "LH": 50}],
    )
    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "coverage_critical_paths.py",
            "--lcov",
            str(regressed),
            "--baseline",
            str(baseline_file),
            "--gate",
        ],
    )
    assert mod.main() == 1
    capsys.readouterr()  # clear the buffer so only the --json run remains

    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "coverage_critical_paths.py",
            "--lcov",
            str(lcov),
            "--json",
        ],
    )
    assert mod.main() == 0
    out = capsys.readouterr().out
    payload = _json.loads(out[: out.index("## Code Coverage")])
    assert payload["weather_solar"]["line_pct"] == 80.0
    assert payload["weather_solar"]["files"] == 1