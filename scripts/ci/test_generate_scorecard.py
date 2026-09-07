"""
Tests for ``scripts/generate_scorecard.py`` -- Issues #2496 and #3436.

The scorecard is the release-time dashboard read from committed sources
(``docs/ASHRAE140_RESULTS.md``, ``release_gates.yaml``, ``README.md``, and
-- since #3436 -- the tracked snapshot
``validation/performance_history.latest.json``) and emitted as
``SCORECARD.md``. CI runs ``--check`` to byte-compare the committed
scorecard against the regenerated copy; any drift fails the build.

The script's load-bearing pieces are pure:

* ``parse_ashrae(text)`` -- extract headline metrics from the ASHRAE doc.
* ``parse_series(text)`` -- per-series pass/warn/fail counts.
* ``parse_gates(text)`` -- gate budgets from the YAML.
* ``parse_readme_throughput(text)`` -- README throughput claim.
* ``render(v, series, g, b)`` -- compile the markdown body.
* ``apply_performance_history(v)`` -- tracked-snapshot throughput override.
* ``main()`` -- CLI with ``--check`` mode and ``--perf-history`` opt-in.

The CI byte-comparison contract is the load-bearing acceptance criterion
(issue #2496: "CI can fail when any metric regresses"), so we exercise
both the parse functions and the ``--check`` CLI. Issue #3436 adds the
determinism fence: generation must be invariant to the presence of the
untracked ``target/performance_history.jsonl`` build artifact.
"""

from __future__ import annotations

import json
import sys

import pytest

SCRIPT_NAME = "generate_scorecard"

# The real latest perf-history entry (seeded into the tracked snapshot by
# #3436): throughput 13.83..., run 2026-09-07.
TRACKED_SNAPSHOT_TEXT = (
    json.dumps(
        {
            "timestamp": "2026-09-07T00:01:56.881908452+00:00",
            "mae": 49.8242105592446,
            "max_deviation": 470.1099213055421,
            "pass_rate": 14.130434782608695,
            "validation_time_seconds": 1.518408016,
            "throughput": 13.830274721099734,
            "git_sha": None,
        },
        indent=2,
    )
    + "\n"
)


@pytest.fixture
def gen(load_script):
    """Freshly-loaded copy of the scorecard generator."""
    return load_script(SCRIPT_NAME)


def _run_main(gen, *argv):
    """Invoke ``gen.main()`` with a synthetic argv; return the exit code."""
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, *argv]
    try:
        rc = gen.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved
    return rc


def _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text=None):
    """Plant a hermetic tmp repo (docs/yaml/readme [+ tracked snapshot]) and
    redirect every module-level path constant at it."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ASHRAE140_RESULTS.md").write_text(
        "*Generated: 2026-01-15 12:34 UTC*\n\n"
        "## Summary\n\n"
        "| Metric | Value |\n|--------|-------|\n"
        "| Pass Rate | 14.1% |\n"
        "| Throughput | 35.36 cases/sec |\n",
        encoding="utf-8",
    )
    (tmp_path / "release_gates.yaml").write_text(
        "validation:\n  min_pass_rate: 60.0\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "We support ~900 configs/sec throughput in release mode.\n",
        encoding="utf-8",
    )
    snapshot = tmp_path / "validation" / "performance_history.latest.json"
    if snapshot_text is not None:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(snapshot_text, encoding="utf-8")
    monkeypatch.setattr(gen, "REPO", tmp_path)
    monkeypatch.setattr(gen, "ASHRAE_DOC", tmp_path / "docs" / "ASHRAE140_RESULTS.md")
    monkeypatch.setattr(gen, "GATES_YAML", tmp_path / "release_gates.yaml")
    monkeypatch.setattr(gen, "README_MD", tmp_path / "README.md")
    monkeypatch.setattr(gen, "SCORECARD", tmp_path / "SCORECARD.md")
    monkeypatch.setattr(gen, "PERF_SNAPSHOT", snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# _num — leading numeric extractor
# ---------------------------------------------------------------------------


def test_num_extracts_leading_float(gen):
    """``'55.09%'`` → 55.09."""
    assert gen._num("55.09%") == 55.09


def test_num_handles_thousands_separators(gen):
    """``'1,234.5'`` → 1234.5."""
    assert gen._num("1,234.5") == 1234.5


def test_num_returns_none_when_no_digit(gen):
    """``'N/A'`` → None."""
    assert gen._num("N/A") is None


def test_num_handles_negative(gen):
    """``'-3.5%'`` → -3.5 (no special logic)."""
    assert gen._num("-3.5%") == -3.5


# ---------------------------------------------------------------------------
# parse_ashrae — metric extractor
# ---------------------------------------------------------------------------


SAMPLE_ASHRAE_DOC = """\
# ASHRAE 140 Results

*Generated: 2026-01-15 12:34 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Passed | 9 |
| Failed | 50 |
| Warnings | 5 |
| Pass Rate | 14.1% |
| Mean Absolute Error | 52.41% |
| Max Deviation | 476.39% |
| Throughput | 12.34 cases/sec |

## Detailed Results

### Case 600

| Heating | Cooling | Status |
|---------|---------|--------|
| 5.0 | 4.0 | PASS |
| 4.5 | 3.5 | WARN |
| 6.0 | 5.0 | FAIL |

### Case 900

| Heating | Cooling | Status |
|---------|---------|--------|
| 1.5 | 2.5 | PASS |
"""


def test_parse_ashrae_extracts_headline_metrics(gen):
    """Total, passed, failed, warnings, pass_rate, mae, max_deviation populated."""
    v = gen.parse_ashrae(SAMPLE_ASHRAE_DOC)
    assert v.total == 64
    assert v.passed == 9
    assert v.failed == 50
    assert v.warnings == 5
    assert v.pass_rate == 14.1
    assert v.mae == 52.41
    assert v.max_deviation == 476.39
    assert v.throughput_cases_per_sec == 12.34


def test_parse_ashrae_extracts_generated_timestamp(gen):
    """Leading ``*Generated: ...*`` line is captured verbatim."""
    v = gen.parse_ashrae(SAMPLE_ASHRAE_DOC)
    assert v.generated_utc == "2026-01-15 12:34 UTC"


def test_parse_ashrae_handles_missing_summary(gen):
    """No Summary section → zeroes (no exception)."""
    v = gen.parse_ashrae("# ASHRAE 140\n\n*Generated: 2026-01-15 12:34 UTC*\n")
    assert v.total == 0
    assert v.pass_rate == 0.0


def test_parse_ashrae_ignores_lines_outside_tables(gen):
    """Body prose is ignored — only `|`-prefixed rows parse."""
    doc = """\
# T

*Generated: 2026-01-15 12:34 UTC*

Some intro prose that mentions a 99% pass rate (must NOT be picked up).

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 12 |
| Pass Rate | 25.0% |
"""
    v = gen.parse_ashrae(doc)
    assert v.total == 12
    assert v.pass_rate == 25.0


# ---------------------------------------------------------------------------
# parse_series — per-series case-level counts
# ---------------------------------------------------------------------------


def test_parse_series_extracts_per_series_counts(gen):
    """Case 600 → 1 PASS, 1 WARN, 1 FAIL; Case 900 → 1 PASS."""
    rows = gen.parse_series(SAMPLE_ASHRAE_DOC)
    by_name = {r.name: r for r in rows}
    assert "Case 600" in by_name
    assert by_name["Case 600"].cases == 3
    assert by_name["Case 600"].passed == 1
    assert by_name["Case 600"].warn == 1
    assert by_name["Case 600"].failed == 1
    assert "Case 900" in by_name
    assert by_name["Case 900"].passed == 1
    assert by_name["Case 900"].cases == 1


def test_parse_series_excludes_sections_outside_detailed_results(gen):
    """A ``## Systematic Issues`` section must not be parsed as series."""
    doc = """\
# T

*Generated: 2026-01-15 12:34 UTC*

## Detailed Results

### Case 100

| S | Status |
|---|--------|
| x | PASS |

## Systematic Issues

### Leak in envelope

| S | Status |
|---|--------|
| x | FAIL |
"""
    rows = gen.parse_series(doc)
    names = {r.name for r in rows}
    assert "Case 100" in names
    assert "Leak in envelope" not in names


def test_parse_series_pass_rate_property(gen):
    """``SeriesRow.pass_rate`` is computed on access."""
    row = gen.SeriesRow(name="x", cases=10, passed=7, warn=2, failed=1)
    assert row.pass_rate == 70.0


def test_parse_series_zero_cases_returns_zero_pass_rate(gen):
    """``SeriesRow.cases == 0`` → pass_rate 0 (no division-by-zero)."""
    row = gen.SeriesRow(name="empty")
    assert row.pass_rate == 0.0


# ---------------------------------------------------------------------------
# parse_gates — YAML gate budgets
# ---------------------------------------------------------------------------


SAMPLE_GATES_YAML = """\
# Example
validation:
  min_pass_rate: 60.0
  max_mae: 50.0

benchmark:
  throughput:
    min_configs_per_sec: 150
  latency:
    max_ms_per_config: 10.0
  absolute_min_throughput: 100

ci:
  required_checks:
    - "Tests (#2983)"
    - "Rustfmt"
  known_failures:
    - "600"
    - "900"

# ~157 configs/sec is the CI runner figure
"""


def test_parse_gates_extracts_thresholds(gen):
    """All five gates are read from the YAML."""
    g = gen.parse_gates(SAMPLE_GATES_YAML)
    assert g.min_pass_rate == 60.0
    assert g.max_mae == 50.0
    assert g.min_throughput == 150.0
    assert g.max_latency_ms == 10.0
    assert g.absolute_min_throughput == 100.0


def test_parse_gates_extracts_required_checks(gen):
    """``required_checks`` list is parsed."""
    g = gen.parse_gates(SAMPLE_GATES_YAML)
    assert "Tests (#2983)" in g.required_checks
    assert "Rustfmt" in g.required_checks


def test_parse_gates_extracts_known_failures(gen):
    """``known_failures`` list is parsed (digits only)."""
    g = gen.parse_gates(SAMPLE_GATES_YAML)
    assert "600" in g.known_failures
    assert "900" in g.known_failures


def test_parse_gates_extracts_ci_throughput_comment(gen):
    """``~157 configs/sec`` comment is parsed as 157.0."""
    g = gen.parse_gates(SAMPLE_GATES_YAML)
    assert g.ci_throughput_comment == 157.0


def test_parse_gates_uses_defaults_when_missing(gen):
    """Empty YAML → dataclass defaults."""
    g = gen.parse_gates("")
    assert g.min_pass_rate == 60.0
    assert g.max_mae == 50.0
    assert g.ci_throughput_comment == 0.0


# ---------------------------------------------------------------------------
# parse_readme_throughput
# ---------------------------------------------------------------------------


def test_parse_readme_throughput_extracts_value(gen):
    """``~900 configs/sec throughput in release mode`` → 900.0."""
    text = "We support ~900 configs/sec throughput in release mode benchmarks."
    b = gen.parse_readme_throughput(text)
    assert b.readme_release_throughput == 900.0


def test_parse_readme_throughput_handles_no_match(gen):
    """No pattern → 0.0 (no exception)."""
    assert gen.parse_readme_throughput("just some text").readme_release_throughput == 0.0


# ---------------------------------------------------------------------------
# render — the smoke test
# ---------------------------------------------------------------------------


def test_render_emits_headline_table(gen):
    """``render`` produces a markdown body containing the Headline table."""
    v = gen.Validation(pass_rate=70.0, mae=30.0, throughput_cases_per_sec=100.0)
    g = gen.Gates(min_pass_rate=60.0, max_mae=50.0, min_throughput=150.0)
    b = gen.Benchmark(readme_release_throughput=200.0)
    out = gen.render(v, [], g, b)
    assert "# Fluxion Release Scorecard" in out
    assert "## Headline" in out
    assert "70.0%" in out  # pass rate
    assert "✅" in out  # status marker


def test_render_marks_low_pass_rate_as_fail(gen):
    """Pass rate below min → Fail banner."""
    v = gen.Validation(pass_rate=10.0, mae=30.0)
    g = gen.Gates(min_pass_rate=60.0, max_mae=50.0)
    out = gen.render(v, [], g, gen.Benchmark())
    assert "❌" in out
    assert "Below" in out or "10.0%" in out


def test_render_emits_ci_regenerate_block(gen):
    """The ``## Regenerate`` section is always present."""
    v = gen.Validation(pass_rate=70.0, mae=30.0)
    g = gen.Gates(min_pass_rate=60.0, max_mae=50.0)
    out = gen.render(v, [], g, gen.Benchmark())
    assert "## Regenerate" in out
    assert "scripts/generate_scorecard.py" in out


# ---------------------------------------------------------------------------
# main() — CLI byte-comparison (the load-bearing acceptance criterion)
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_committed_matches_regenerated(
    gen, tmp_path, monkeypatch, capsys
):
    """``--check`` against a current scorecard → exit 0."""
    # Plant a fake doc + yaml + readme + scorecard so all loaders succeed.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ASHRAE140_RESULTS.md").write_text(
        "*Generated: 2026-01-15 12:34 UTC*\n\n"
        "## Summary\n\n"
        "| Metric | Value |\n|--------|-------|\n"
        "| Pass Rate | 14.1% |\n",
        encoding="utf-8",
    )
    (tmp_path / "release_gates.yaml").write_text(
        "validation:\n  min_pass_rate: 60.0\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "We support ~900 configs/sec throughput in release mode.\n",
        encoding="utf-8",
    )
    # Generate the byte-stable scorecard once and write to the committed path.
    monkeypatch.setattr(gen, "REPO", tmp_path)
    monkeypatch.setattr(gen, "ASHRAE_DOC", tmp_path / "docs" / "ASHRAE140_RESULTS.md")
    monkeypatch.setattr(gen, "GATES_YAML", tmp_path / "release_gates.yaml")
    monkeypatch.setattr(gen, "README_MD", tmp_path / "README.md")
    scorecard_path = tmp_path / "SCORECARD.md"
    monkeypatch.setattr(gen, "SCORECARD", scorecard_path)
    monkeypatch.setattr(
        gen,
        "PERF_SNAPSHOT",
        tmp_path / "validation" / "performance_history.latest.json",
    )

    # First call writes the SCORECARD.md.
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME]
    try:
        rc1 = gen.main()
    except SystemExit as e:
        rc1 = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved
    assert rc1 == 0
    assert scorecard_path.exists()

    # Second call (--check) compares against the just-written SCORECARD.md.
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--check"]
    try:
        rc2 = gen.main()
    except SystemExit as e:
        rc2 = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    out = capsys.readouterr().out
    assert rc2 == 0, f"expected exit 0, got {rc2}\noutput:\n{out}"
    assert "up to date" in out or "no drift" in out


def test_main_returns_one_when_scorecard_drifts(gen, tmp_path, monkeypatch, capsys):
    """Plan A: regenerate to a fresh SCORECARD and the gate detects drift.

    Issue #2496 acceptance criterion: "CI can fail when any metric regresses".
    A regression that flips the validation report but not the scorecard
    must trigger --check → exit 1.
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ASHRAE140_RESULTS.md").write_text(
        "*Generated: 2026-01-15 12:34 UTC*\n\n"
        "## Summary\n\n"
        "| Metric | Value |\n|--------|-------|\n"
        "| Pass Rate | 14.1% |\n",
        encoding="utf-8",
    )
    (tmp_path / "release_gates.yaml").write_text(
        "validation:\n  min_pass_rate: 60.0\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "We support ~900 configs/sec throughput in release mode.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gen, "REPO", tmp_path)
    monkeypatch.setattr(gen, "ASHRAE_DOC", tmp_path / "docs" / "ASHRAE140_RESULTS.md")
    monkeypatch.setattr(gen, "GATES_YAML", tmp_path / "release_gates.yaml")
    monkeypatch.setattr(gen, "README_MD", tmp_path / "README.md")
    scorecard_path = tmp_path / "SCORECARD.md"
    monkeypatch.setattr(gen, "SCORECARD", scorecard_path)
    monkeypatch.setattr(
        gen,
        "PERF_SNAPSHOT",
        tmp_path / "validation" / "performance_history.latest.json",
    )

    # Step 1: write the committed scorecard.
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME]
    try:
        gen.main()
    except SystemExit:
        pass
    finally:
        sys.argv[:] = saved

    # Step 2: plant a stale committed scorecard (different bytes).
    scorecard_path.write_text("# STALE COMMITTED COPY\n", encoding="utf-8")

    # Step 3: --check must detect drift.
    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--check"]
    try:
        rc = gen.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{combined}"
    assert "drift" in combined.lower() or "stale" in combined.lower()


def test_main_returns_one_when_scorecard_missing(gen, tmp_path, monkeypatch, capsys):
    """``--check`` without a committed SCORECARD.md → exit 1 (fail-loud)."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ASHRAE140_RESULTS.md").write_text(
        "*Generated: 2026-01-15 12:34 UTC*\n\n", encoding="utf-8"
    )
    (tmp_path / "release_gates.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr(gen, "REPO", tmp_path)
    monkeypatch.setattr(gen, "ASHRAE_DOC", tmp_path / "docs" / "ASHRAE140_RESULTS.md")
    monkeypatch.setattr(gen, "GATES_YAML", tmp_path / "release_gates.yaml")
    monkeypatch.setattr(gen, "README_MD", tmp_path / "README.md")
    monkeypatch.setattr(gen, "SCORECARD", tmp_path / "SCORECARD.md")
    monkeypatch.setattr(
        gen,
        "PERF_SNAPSHOT",
        tmp_path / "validation" / "performance_history.latest.json",
    )

    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--check"]
    try:
        rc = gen.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    assert rc == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "missing" in combined.lower() or "SCORECARD" in combined


# ---------------------------------------------------------------------------
# apply_performance_history — tracked-snapshot throughput source (#3436)
# ---------------------------------------------------------------------------


def test_tracked_snapshot_overrides_throughput_and_names_source(
    gen, tmp_path, monkeypatch
):
    """A valid tracked snapshot overrides the docs figure and the attribution
    names the tracked file (never ``target/performance_history.jsonl``)."""
    _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text=TRACKED_SNAPSHOT_TEXT)
    v = gen.Validation(throughput_cases_per_sec=35.36)
    src = gen.apply_performance_history(v)
    assert v.throughput_cases_per_sec == pytest.approx(13.830274721099734)
    assert src == "`validation/performance_history.latest.json` (latest run 2026-09-07)"


def test_missing_snapshot_falls_back_to_docs(gen, tmp_path, monkeypatch):
    """No tracked snapshot → docs figure is kept, docs attribution returned."""
    _plant_tmp_repo(gen, tmp_path, monkeypatch)  # snapshot deliberately absent
    v = gen.Validation(throughput_cases_per_sec=35.36)
    src = gen.apply_performance_history(v)
    assert v.throughput_cases_per_sec == 35.36
    assert src == "`docs/ASHRAE140_RESULTS.md`"


def test_corrupt_snapshot_falls_back_to_docs(gen, tmp_path, monkeypatch):
    """Corrupt (non-JSON) snapshot → deterministic docs fallback."""
    _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text="not json {{{\n")
    v = gen.Validation(throughput_cases_per_sec=35.36)
    src = gen.apply_performance_history(v)
    assert v.throughput_cases_per_sec == 35.36
    assert src == "`docs/ASHRAE140_RESULTS.md`"


def test_zero_throughput_snapshot_falls_back_to_docs(gen, tmp_path, monkeypatch):
    """A snapshot entry with non-positive throughput is not applied."""
    bad = json.dumps({"timestamp": "2026-09-07T00:00:00+00:00", "throughput": 0.0})
    _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text=bad)
    v = gen.Validation(throughput_cases_per_sec=35.36)
    src = gen.apply_performance_history(v)
    assert v.throughput_cases_per_sec == 35.36
    assert src == "`docs/ASHRAE140_RESULTS.md`"


def test_untracked_target_history_is_never_read(gen, tmp_path, monkeypatch):
    """Unit-level determinism fence (#3436): a decoy untracked
    ``target/performance_history.jsonl`` must be ignored -- with a valid
    tracked snapshot AND with none (old buggy behavior picked it up)."""
    decoy = tmp_path / "target" / "performance_history.jsonl"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text(
        json.dumps({"timestamp": "2026-09-07T00:00:00+00:00", "throughput": 99.99})
        + "\n",
        encoding="utf-8",
    )

    # With a tracked snapshot: figure comes from the snapshot, not the decoy.
    _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text=TRACKED_SNAPSHOT_TEXT)
    v = gen.Validation(throughput_cases_per_sec=35.36)
    src = gen.apply_performance_history(v)
    assert v.throughput_cases_per_sec == pytest.approx(13.830274721099734)
    assert "target/" not in src

    # Without a tracked snapshot: docs fallback, decoy still ignored.
    monkeypatch.setattr(gen, "PERF_SNAPSHOT", tmp_path / "validation" / "absent.json")
    v2 = gen.Validation(throughput_cases_per_sec=35.36)
    src2 = gen.apply_performance_history(v2)
    assert v2.throughput_cases_per_sec == 35.36
    assert src2 == "`docs/ASHRAE140_RESULTS.md`"


# ---------------------------------------------------------------------------
# main() determinism — the #3436 core acceptance criterion
# ---------------------------------------------------------------------------


def test_generation_invariant_to_untracked_perf_history(
    gen, tmp_path, monkeypatch, capsys
):
    """End-to-end fence (#3436): a decoy untracked
    ``target/performance_history.jsonl`` present vs absent must produce a
    byte-identical SCORECARD.md."""
    _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text=TRACKED_SNAPSHOT_TEXT)

    out_fresh = tmp_path / "out_fresh.md"
    assert _run_main(gen, "-o", str(out_fresh)) == 0
    capsys.readouterr()

    # Plant the decoy build artifact with a DIFFERENT throughput value.
    decoy = tmp_path / "target" / "performance_history.jsonl"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text(
        json.dumps({"timestamp": "2026-09-07T23:59:59+00:00", "throughput": 99.99})
        + "\n",
        encoding="utf-8",
    )

    out_decoy = tmp_path / "out_decoy.md"
    assert _run_main(gen, "-o", str(out_decoy)) == 0

    assert out_fresh.read_bytes() == out_decoy.read_bytes()
    body = out_fresh.read_text(encoding="utf-8")
    # The run used the tracked snapshot figure, not the docs 35.36 fallback.
    assert "13.83 cases/sec" in body
    assert (
        "`validation/performance_history.latest.json` (latest run 2026-09-07)" in body
    )
    assert "99.99" not in body
    assert "35.36" not in body


def test_perf_history_flag_overrides_from_jsonl(gen, tmp_path, monkeypatch):
    """``--perf-history <jsonl>`` applies the LAST non-empty entry and names
    the operator-supplied path in the attribution."""
    _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text=TRACKED_SNAPSHOT_TEXT)
    hist = tmp_path / "adhoc_history.jsonl"
    hist.write_text(
        json.dumps({"timestamp": "2026-09-05T00:00:00+00:00", "throughput": 1.5})
        + "\n"
        + json.dumps({"timestamp": "2026-09-06T12:00:00+00:00", "throughput": 7.5})
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "out_flag.md"
    assert _run_main(gen, "-o", str(out), "--perf-history", str(hist)) == 0
    body = out.read_text(encoding="utf-8")
    assert "7.50 cases/sec" in body
    assert f"`{hist}` (latest run 2026-09-06)" in body
    assert "13.83 cases/sec" not in body  # snapshot figure not used


def test_perf_history_env_var_overrides_from_jsonl(gen, tmp_path, monkeypatch):
    """``$FLUXION_PERF_HISTORY`` is the env-var equivalent of the flag."""
    _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text=TRACKED_SNAPSHOT_TEXT)
    hist = tmp_path / "adhoc_history.jsonl"
    hist.write_text(
        json.dumps({"timestamp": "2026-09-06T12:00:00+00:00", "throughput": 7.5})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLUXION_PERF_HISTORY", str(hist))

    out = tmp_path / "out_env.md"
    assert _run_main(gen, "-o", str(out)) == 0
    body = out.read_text(encoding="utf-8")
    assert "7.50 cases/sec" in body
    assert f"`{hist}` (latest run 2026-09-06)" in body


def test_perf_history_bad_path_fails_loud(gen, tmp_path, monkeypatch, capsys):
    """A bad explicit ``--perf-history`` path exits 2 (fail-loud) instead of
    silently regressing to the docs fallback."""
    _plant_tmp_repo(gen, tmp_path, monkeypatch, snapshot_text=TRACKED_SNAPSHOT_TEXT)
    rc = _run_main(
        gen,
        "-o",
        str(tmp_path / "out.md"),
        "--perf-history",
        str(tmp_path / "does_not_exist.jsonl"),
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "perf-history" in (captured.err + captured.out).lower()
    assert not (tmp_path / "out.md").exists()


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------


def test_module_paths_point_at_repo(gen, repo_root):
    """The module-level paths are anchored at the repo root, not the script dir."""
    assert gen.REPO == repo_root
    assert gen.ASHRAE_DOC == repo_root / "docs" / "ASHRAE140_RESULTS.md"
    assert gen.GATES_YAML == repo_root / "release_gates.yaml"
    assert gen.README_MD == repo_root / "README.md"
    assert gen.SCORECARD == repo_root / "SCORECARD.md"
    # #3436: the perf-history source is the TRACKED snapshot, never the
    # untracked target/ build artifact.
    assert (
        gen.PERF_SNAPSHOT
        == repo_root / "validation" / "performance_history.latest.json"
    )
