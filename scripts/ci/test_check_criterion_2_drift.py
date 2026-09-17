"""Tests for ``scripts/check_criterion_2_drift.py`` (Issue #3744).

The cohort-drift detector parses a ``cargo test`` log, extracts the
live failing-test set, and diffs it against the canonical §LIMIT-21
cohort from ``scripts/check_beta_soak_gate.criterion_2_failures()``.
These tests pin:

* the log parser against representative cargo output (progress lines,
  ``failures:`` blocks, ``test result:`` summaries, multi-line
  panics, panic-context noise),
* the canonical-cohort loader (subprocess-free via a JSON fixture),
* the diff computation and the v1-schema artifact payload,
* the GitHub-comment rendering in both the drift and no-drift paths,
* the CLI exit-code contract (0 = clean, 1 = drift, 2 = error).

The tests are hermetic: they never spawn
``check_beta_soak_gate.py``, they read the real script via the
``load_script`` fixture from ``scripts/ci/conftest.py`` for class-shape
checks, and they supply pre-computed canonical JSON for the loader
path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "check_criterion_2_drift"

CANONICAL_PAYLOAD = {
    "schema_version": "1",
    "tracker_doc": "docs/agents/beta-soak-criterion-2-tracker.md",
    "limit_ref": "§LIMIT-21",
    "limit_owner_issue": "#3297",
    "failures": [
        {
            "name": "test_physics_thermal_model_eplus_case_600_reference_csv",
            "file": "tests/zone_balance_eplus_isolation.rs",
            "fn_line": 187,
            "panic_site": "tests/zone_balance_eplus_isolation.rs:298:5",
            "symptom": "Wild step-to-step oscillation",
            "case": "ASHRAE 140 Case 600 (low-mass conditioned)",
            "known_issues": "§LIMIT-21",
        },
        {
            "name": "test_free_floating_case_900ff_isolation",
            "file": "tests/zone_balance_eplus_isolation.rs",
            "fn_line": 430,
            "panic_site": "tests/zone_balance_eplus_isolation.rs:445:5",
            "symptom": "Numerical divergence to non-finite",
            "case": "ASHRAE 140 Case 900FF (high-mass free-floating)",
            "known_issues": "§LIMIT-21",
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_criterion_2_drift.py``."""
    return load_script(SCRIPT_NAME)


@pytest.fixture
def write_canonical(tmp_path):
    """Write the canonical Criterion 2 failures JSON to ``tmp_path`` and
    return the path. Used to hermetically seed
    ``load_canonical_cohort`` without spawning the beta-soak subprocess.
    """

    def _inner(payload: dict | None = None) -> Path:
        path = tmp_path / "canonical.json"
        path.write_text(
            json.dumps(payload if payload is not None else CANONICAL_PAYLOAD),
            encoding="utf-8",
        )
        return path

    return _inner


# ---------------------------------------------------------------------------
# Parser — happy paths
# ---------------------------------------------------------------------------


def test_parser_extracts_progress_line_failures(checker):
    """Per-test progress lines (`test ... FAILED`) are the canonical
    cargo surface."""
    text = (
        "running 21 tests\n"
        "test test_physics_thermal_model_eplus_case_600_reference_csv ... FAILED\n"
        "test test_free_floating_case_900ff_isolation ... FAILED\n"
        "test test_5r1c_network_steady_state_four_walls ... ok\n"
        "test result: FAILED. 19 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.0s\n"
    )
    parsed = checker.parse_criterion_2_log(text)
    assert parsed.failures == {
        "test_physics_thermal_model_eplus_case_600_reference_csv",
        "test_free_floating_case_900ff_isolation",
    }
    assert parsed.test_results == [
        {"verdict": "FAILED", "passed": 19, "failed": 2, "ignored": 0}
    ]


def test_parser_extracts_module_prefixed_progress_lines(checker):
    """The consolidated runner emits ``<module>::<fn>`` names; the
    parser strips the prefix and keeps the bare snake_case identifier."""
    text = (
        "test zone_balance_eplus_isolation::test_physics_thermal_model_eplus_case_600_reference_csv ... FAILED\n"
        "test zone_balance_eplus_isolation::test_free_floating_case_900ff_isolation ... FAILED\n"
    )
    parsed = checker.parse_criterion_2_log(text)
    assert parsed.failures == {
        "test_physics_thermal_model_eplus_case_600_reference_csv",
        "test_free_floating_case_900ff_isolation",
    }


def test_parser_extracts_failures_block_lines(checker):
    """The ``failures:`` block (verbose cargo output) lists each failing
    test on its own indented line."""
    text = (
        "failures:\n"
        "\n"
        "    zone_balance_eplus_isolation::test_physics_thermal_model_eplus_case_600_reference_csv\n"
        "    zone_balance_eplus_isolation::test_free_floating_case_900ff_isolation\n"
        "\n"
        "test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.5s\n"
    )
    parsed = checker.parse_criterion_2_log(text)
    assert parsed.failures == {
        "test_physics_thermal_model_eplus_case_600_reference_csv",
        "test_free_floating_case_900ff_isolation",
    }


def test_parser_dedupes_progress_and_failures_block(checker):
    """A test named in the progress lines and the failures block is
    counted once."""
    text = (
        "test zone_balance_eplus_isolation::test_xxx_yyy ... FAILED\n"
        "failures:\n"
        "\n"
        "    zone_balance_eplus_isolation::test_xxx_yyy\n"
        "\n"
        "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    parsed = checker.parse_criterion_2_log(text)
    assert parsed.failures == {"test_xxx_yyy"}


def test_parser_ignores_passing_tests(checker):
    """Passing tests never land in ``failures``."""
    text = (
        "test test_a ... ok\n"
        "test test_b ... ok\n"
        "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    parsed = checker.parse_criterion_2_log(text)
    assert parsed.failures == set()
    assert parsed.test_results == [
        {"verdict": "ok", "passed": 2, "failed": 0, "ignored": 0}
    ]


def test_parser_handles_panic_message_noise(checker):
    """Panic messages contain `FAILED` substrings; the parser must not
    pick them up as test names."""
    text = (
        "running 21 tests\n"
        "test test_xxx_yyy ... FAILED\n"
        "\n"
        "---- test_xxx_yyy stdout ----\n"
        "thread 'main' panicked at 'wild oscillation: max step jump = 34.007 °C',\n"
        "tests/zone_balance_eplus_isolation.rs:298:5\n"
        "test result: FAILED. 20 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    parsed = checker.parse_criterion_2_log(text)
    assert parsed.failures == {"test_xxx_yyy"}


def test_parser_surfaces_parse_warnings_via_public_api(checker):
    """The public parser's progress-line regex enforces snake_case on
    the captured name (Rust test fns are always snake_case). Lines
    that don't match the per-test progress regex are silently ignored
    rather than reported as warnings, because the only reliable signal
    of "this is a per-test progress line" is the regex match itself.

    This test pins that contract: an unparseable progress line is
    simply absent from the failure set, no warning is emitted, and
    the parser does not raise. (The parse-warnings path remains a
    defensive surface for the failures-block renderer when the regex
    matches but the trailing segment fails snake_case, which is a
    state the regex itself excludes; see ``_normalise_test_name``
    for the unit-level check.)
    """
    text = (
        "test 123_not_valid ... FAILED\n"
        "test test_xxx ... FAILED\n"
        "test result: FAILED. 0 passed; 1 failed; 0 ignored\n"
    )
    parsed = checker.parse_criterion_2_log(text)
    assert parsed.failures == {"test_xxx"}
    # No warning is emitted (the digit-prefixed line never matches
    # the regex). The artefact embeds an empty ``parse_warnings``
    # list so downstream consumers see a stable JSON shape.
    assert parsed.parse_warnings == []


# ---------------------------------------------------------------------------
# Parser — `_normalise_test_name` unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("test_physics_thermal_model_eplus_case_600_reference_csv",
         "test_physics_thermal_model_eplus_case_600_reference_csv"),
        ("zone_balance_eplus_isolation::test_xxx",
         "test_xxx"),
        ("tests::all_tests::zone_balance_eplus_isolation::test_yyy",
         "test_yyy"),
        ("", ""),
        ("123notvalid", ""),
        ("::test_xxx", "test_xxx"),
        ("test_x_y_z", "test_x_y_z"),
    ],
)
def test_normalise_test_name(checker, raw, expected):
    """The normaliser strips module prefixes and rejects non-identifier
    tails."""
    out = checker._normalise_test_name(raw)
    if expected == "":
        assert out is None
    else:
        assert out == expected


# ---------------------------------------------------------------------------
# Parser — file IO
# ---------------------------------------------------------------------------


def test_parse_criterion_2_log_file_reads_disk(checker, tmp_path):
    """The file-level reader surfaces the same parser contract."""
    log = tmp_path / "criterion-2.log"
    log.write_text(
        "test test_xxx_yyy ... FAILED\n"
        "test result: FAILED. 0 passed; 1 failed; 0 ignored\n",
        encoding="utf-8",
    )
    parsed = checker.parse_criterion_2_log_file(log)
    assert parsed.failures == {"test_xxx_yyy"}


def test_parse_criterion_2_log_file_missing_raises(checker, tmp_path):
    """Missing log file raises ``FileNotFoundError`` (caught by main)."""
    with pytest.raises(FileNotFoundError):
        checker.parse_criterion_2_log_file(tmp_path / "missing.log")


# ---------------------------------------------------------------------------
# Canonical cohort loader
# ---------------------------------------------------------------------------


def test_load_canonical_cohort_reads_json(checker, write_canonical):
    """The loader reads a pre-computed canonical JSON without spawning
    the beta-soak subprocess."""
    path = write_canonical()
    names, payload = checker.load_canonical_cohort(canonical_json=path)
    assert names == {
        "test_physics_thermal_model_eplus_case_600_reference_csv",
        "test_free_floating_case_900ff_isolation",
    }
    assert payload["limit_ref"] == "§LIMIT-21"


def test_load_canonical_cohort_skips_failures_without_name(checker, tmp_path):
    """Failures missing the ``name`` field are skipped silently
    (defensive: the canonical schema requires it but a downstream
    caller should not crash)."""
    payload = dict(CANONICAL_PAYLOAD)
    payload["failures"] = [
        CANONICAL_PAYLOAD["failures"][0],
        {"file": "x.rs"},  # no name
    ]
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    names, _ = checker.load_canonical_cohort(canonical_json=path)
    assert names == {
        "test_physics_thermal_model_eplus_case_600_reference_csv"
    }


def test_load_canonical_cohort_rejects_malformed_json(checker, tmp_path):
    """Malformed canonical JSON raises (caught by main, exit 2)."""
    bad = tmp_path / "canonical.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        checker.load_canonical_cohort(canonical_json=bad)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def test_compute_drift_no_change(checker):
    """Live ≡ canonical → no drift."""
    canonical = {
        "test_physics_thermal_model_eplus_case_600_reference_csv",
        "test_free_floating_case_900ff_isolation",
    }
    drift = checker.compute_drift(canonical, canonical)
    assert drift.drift_detected is False
    assert drift.added == set()
    assert drift.removed == set()
    assert drift.unchanged == canonical


def test_compute_drift_regression_only(checker):
    """Live has a new failure not in canonical → regression."""
    canonical = {"test_canonical_a", "test_canonical_b"}
    live = {"test_canonical_a", "test_canonical_b", "test_new_regression_in_isolation"}
    drift = checker.compute_drift(live, canonical)
    assert drift.drift_detected is True
    assert drift.added == {"test_new_regression_in_isolation"}
    assert drift.removed == set()
    assert drift.unchanged == {"test_canonical_a", "test_canonical_b"}


def test_compute_drift_canonical_shrunk(checker):
    """Live is missing a canonical failure → cohort shrank (could be
    recovery OR stale tracker doc)."""
    canonical = {"test_canonical_a", "test_canonical_b"}
    live = {"test_canonical_a"}
    drift = checker.compute_drift(live, canonical)
    assert drift.drift_detected is True
    assert drift.added == set()
    assert drift.removed == {"test_canonical_b"}


def test_compute_drift_mixed(checker):
    """Both added and removed → mixed drift."""
    canonical = {"test_canonical_a", "test_canonical_b"}
    live = {"test_canonical_a", "test_new_regression_in_isolation"}
    drift = checker.compute_drift(live, canonical)
    assert drift.drift_detected is True
    assert drift.added == {"test_new_regression_in_isolation"}
    assert drift.removed == {"test_canonical_b"}


def test_compute_drift_live_empty(checker):
    """Live is empty (Criterion 2 went green) is a HUGE drift — every
    canonical failure is `removed`. This is the §LIMIT-21 closing
    signal, but the detector flags it because the canonical tracker
    doc may be stale (test renamed, refactored, etc.) and we want the
    human operator to confirm either way."""
    canonical = {"test_canonical_a", "test_canonical_b"}
    live: set[str] = set()
    drift = checker.compute_drift(live, canonical)
    assert drift.drift_detected is True
    assert drift.added == set()
    assert drift.removed == canonical


# ---------------------------------------------------------------------------
# Artifact + comment rendering
# ---------------------------------------------------------------------------


def test_render_drift_artifact_v1_schema(checker):
    """The artifact payload is the v1 schema with all required fields."""
    drift = checker.compute_drift(
        {
            "test_physics_thermal_model_eplus_case_600_reference_csv",
            "test_free_floating_case_900ff_isolation",
            "test_new_failure_in_isolation",
        },
        {
            "test_physics_thermal_model_eplus_case_600_reference_csv",
            "test_free_floating_case_900ff_isolation",
        },
    )
    parsed = checker.parse_criterion_2_log(
        "test test_physics_thermal_model_eplus_case_600_reference_csv ... FAILED\n"
        "test test_free_floating_case_900ff_isolation ... FAILED\n"
        "test test_new_failure_in_isolation ... FAILED\n"
        "test result: FAILED. 0 passed; 3 failed; 0 ignored\n"
    )
    artifact = checker.render_drift_artifact(
        drift, CANONICAL_PAYLOAD, parsed, run_id="12345",
        source_log="criterion-2.log",
    )
    assert artifact["schema_version"] == "1"
    assert artifact["drift_detected"] is True
    assert artifact["live_failing_count"] == 3
    assert artifact["canonical_count"] == 2
    assert artifact["added"] == ["test_new_failure_in_isolation"]
    assert artifact["removed"] == []
    assert set(artifact["unchanged"]) == {
        "test_physics_thermal_model_eplus_case_600_reference_csv",
        "test_free_floating_case_900ff_isolation",
    }
    assert artifact["tracker_doc"] == "docs/agents/beta-soak-criterion-2-tracker.md"
    assert artifact["limit_ref"] == "§LIMIT-21"
    assert artifact["limit_owner_issue"] == "#3297"
    assert artifact["run_id"] == "12345"
    assert artifact["source_log"] == "criterion-2.log"
    assert "generated_at" in artifact
    # test_results is non-empty (the summary line above).
    assert artifact["test_results"]


def test_render_drift_comment_clean(checker):
    """No-drift path is a short ✅ status line."""
    drift = checker.compute_drift(
        {
            "test_physics_thermal_model_eplus_case_600_reference_csv",
            "test_free_floating_case_900ff_isolation",
        },
        {
            "test_physics_thermal_model_eplus_case_600_reference_csv",
            "test_free_floating_case_900ff_isolation",
        },
    )
    artifact = {"schema_version": "1"}
    comment = checker.render_drift_comment(drift, artifact)
    assert "✅" in comment
    assert "Criterion 2 cohort drift check" in comment
    assert "No drift" in comment
    assert "<!-- criterion-2-cohort-drift-marker -->" in comment


def test_render_drift_comment_regression(checker):
    """Regression path calls out the new failure with a 🚨 marker."""
    drift = checker.compute_drift(
        {
            "test_canonical_a",
            "test_canonical_b",
            "test_new_regression_in_isolation",
        },
        {"test_canonical_a", "test_canonical_b"},
    )
    artifact = {"schema_version": "1"}
    comment = checker.render_drift_comment(drift, artifact)
    assert "🚨" in comment
    assert "Regression suspected" in comment
    assert "`test_new_regression_in_isolation`" in comment


def test_render_drift_comment_canonical_shrunk(checker):
    """Canonical-shrunk path uses a 📉 marker and notes the
    dual-significance (closing OR stale tracker)."""
    drift = checker.compute_drift(
        {"test_canonical_a"}, {"test_canonical_a", "test_canonical_b"},
    )
    artifact = {"schema_version": "1"}
    comment = checker.render_drift_comment(drift, artifact)
    assert "📉" in comment
    assert "Canonical cohort shrank" in comment
    assert "`test_canonical_b`" in comment
    assert "stale" in comment or "narrowing" in comment


def test_render_drift_comment_mixed(checker):
    """Mixed-drift path uses a ⚠️ marker and lists both directions."""
    drift = checker.compute_drift(
        {"test_canonical_a", "test_new_regression_in_isolation"},
        {"test_canonical_a", "test_canonical_b"},
    )
    artifact = {"schema_version": "1"}
    comment = checker.render_drift_comment(drift, artifact)
    assert "⚠️" in comment
    assert "Mixed drift" in comment
    assert "`test_new_regression_in_isolation`" in comment
    assert "`test_canonical_b`" in comment


def test_render_drift_comment_includes_run_url(checker):
    """When ``run_url`` is provided, the comment links to the run log."""
    drift = checker.compute_drift(
        {"test_canonical_a"}, {"test_canonical_a", "test_canonical_b"},
    )
    artifact = {"schema_version": "1"}
    comment = checker.render_drift_comment(
        drift, artifact, run_url="https://github.com/foo/bar/actions/runs/1"
    )
    assert "https://github.com/foo/bar/actions/runs/1" in comment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_main(checker, monkeypatch, argv):
    """Invoke ``checker.main(argv)`` with captured stdout/stderr."""
    from io import StringIO

    out, err = StringIO(), StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = checker.main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture
def seed_clean_run(checker, write_canonical, tmp_path):
    """Seed a Criterion 2 log + canonical JSON where live ≡ canonical."""
    canonical_path = write_canonical()
    log = tmp_path / "clean.log"
    log.write_text(
        "test zone_balance_eplus_isolation::test_physics_thermal_model_eplus_case_600_reference_csv ... FAILED\n"
        "test zone_balance_eplus_isolation::test_free_floating_case_900ff_isolation ... FAILED\n"
        "test result: FAILED. 19 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out\n",
        encoding="utf-8",
    )
    return log, canonical_path


def test_cli_clean_exits_zero(checker, monkeypatch, seed_clean_run, tmp_path):
    """No drift → exit 0, no drift verdict in stdout.

    Pass ``--artifact`` so the v1-schema file lands in ``tmp_path``
    instead of the real repo root — otherwise the next pytest module
    (root-hygiene) picks the leftover up as a transient-artifact
    violation and the real-repo gate flips red.
    """
    log, canonical = seed_clean_run
    artifact_path = tmp_path / "out.json"
    rc, out, _err = _run_main(
        checker,
        monkeypatch,
        [
            "--log",
            str(log),
            "--canonical",
            str(canonical),
            "--artifact",
            str(artifact_path),
        ],
    )
    assert rc == 0
    assert "clean" in out
    assert "DRIFT" not in out


def test_cli_clean_emits_artifact(checker, monkeypatch, seed_clean_run, tmp_path):
    """Default mode writes the v1-schema artifact."""
    log, canonical = seed_clean_run
    artifact_path = tmp_path / "out.json"
    rc, _out, _err = _run_main(
        checker,
        monkeypatch,
        [
            "--log",
            str(log),
            "--canonical",
            str(canonical),
            "--artifact",
            str(artifact_path),
        ],
    )
    assert rc == 0
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["drift_detected"] is False


def test_cli_clean_json_to_stdout(checker, monkeypatch, seed_clean_run):
    """``--json`` emits the v1-schema artifact to stdout (no file write)."""
    log, canonical = seed_clean_run
    rc, out, _err = _run_main(
        checker,
        monkeypatch,
        ["--log", str(log), "--canonical", str(canonical), "--json"],
    )
    assert rc == 0
    payload = json.loads(out)
    assert payload["schema_version"] == "1"
    assert payload["drift_detected"] is False


def test_cli_drift_exits_one(checker, monkeypatch, write_canonical, tmp_path):
    """Drift detected → exit 1.

    Pass ``--artifact`` so the v1-schema file lands in ``tmp_path``
    instead of the real repo root — otherwise the next pytest module
    (root-hygiene) picks the leftover up as a transient-artifact
    violation and the real-repo gate flips red.
    """
    canonical = write_canonical()
    log = tmp_path / "drift.log"
    log.write_text(
        "test zone_balance_eplus_isolation::test_physics_thermal_model_eplus_case_600_reference_csv ... FAILED\n"
        "test zone_balance_eplus_isolation::test_free_floating_case_900ff_isolation ... FAILED\n"
        "test zone_balance_eplus_isolation::test_new_regression_in_isolation ... FAILED\n"
        "test result: FAILED. 18 passed; 3 failed; 0 ignored\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "out.json"
    rc, out, _err = _run_main(
        checker,
        monkeypatch,
        [
            "--log",
            str(log),
            "--canonical",
            str(canonical),
            "--artifact",
            str(artifact_path),
        ],
    )
    assert rc == 1
    assert "DRIFT" in out


def test_cli_drift_comment_regression(checker, monkeypatch, write_canonical, tmp_path):
    """``--comment`` emits a 🚨 regression comment body suitable for #3286."""
    canonical = write_canonical()
    log = tmp_path / "drift.log"
    log.write_text(
        "test zone_balance_eplus_isolation::test_physics_thermal_model_eplus_case_600_reference_csv ... FAILED\n"
        "test zone_balance_eplus_isolation::test_free_floating_case_900ff_isolation ... FAILED\n"
        "test zone_balance_eplus_isolation::test_new_regression_in_isolation ... FAILED\n"
        "test result: FAILED. 18 passed; 3 failed; 0 ignored\n",
        encoding="utf-8",
    )
    rc, out, _err = _run_main(
        checker,
        monkeypatch,
        [
            "--log",
            str(log),
            "--canonical",
            str(canonical),
            "--comment",
        ],
    )
    assert rc == 1
    assert "🚨" in out
    assert "`test_new_regression_in_isolation`" in out
    assert "<!-- criterion-2-cohort-drift-marker -->" in out


def test_cli_missing_log_exits_two(checker, monkeypatch, tmp_path):
    """Missing log file → script error, exit 2."""
    rc, _out, err = _run_main(
        checker,
        monkeypatch,
        ["--log", str(tmp_path / "missing.log"), "--canonical",
         str(tmp_path / "canonical.json")],
    )
    assert rc == 2
    assert "not found" in err


def test_cli_missing_canonical_via_json_exits_two(
    checker, monkeypatch, tmp_path
):
    """Malformed canonical JSON → script error, exit 2."""
    log = tmp_path / "log.txt"
    log.write_text("test result: ok. 1 passed; 0 failed; 0 ignored\n",
                  encoding="utf-8")
    canonical = tmp_path / "canonical.json"
    canonical.write_text("{not json", encoding="utf-8")
    rc, _out, err = _run_main(
        checker,
        monkeypatch,
        ["--log", str(log), "--canonical", str(canonical)],
    )
    assert rc == 2
    assert "canonical" in err.lower() or "json" in err.lower()