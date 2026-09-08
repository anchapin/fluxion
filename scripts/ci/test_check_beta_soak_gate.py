"""Tests for ``scripts/check_beta_soak_gate.py`` -- Issue #3286.

The β-phase 30-day soak window gate is the production-path gate that
allows removal of the ``gauge-solver`` cargo feature (Phase A8,
Issue #3291, PR #3482). The gate's contract is encoded in this script:

* ``beta-soak-state.json`` schema validation (the workflow's inline
  state-construction predates the validator but is accepted by the
  back-compat path),
* ``gate_open`` semantics (``streak >= target``),
* CLI exit-code contract: ``--gate enforce`` exits 1 when the gate is
  closed, ``--gate report`` always exits 0, ``--write-template`` exits
  0 with a fresh template written.

These tests pin the schema and exit-code contract against hermetic
``tmp_path`` fixtures and the ``load_script`` pattern from
``scripts/ci/conftest.py``. A regression in any of the four invariants
above surfaces here instead of as a silent false-green on a real PR.
"""

from __future__ import annotations

import json
import sys

import pytest

SCRIPT_NAME = "check_beta_soak_gate"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_beta_soak_gate.py``."""
    return load_script(SCRIPT_NAME)


def _write_state(tmp_path, payload: dict) -> str:
    """Write a state file under ``tmp_path`` and return its path string."""
    target = tmp_path / "beta-soak-state.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return str(target)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_schema_version_is_v1(checker):
    """The schema version constant is the canonical v1."""
    assert checker.SCHEMA_VERSION == "1"


def test_default_target_is_30(checker):
    """The gate target is fixed at 30 nightly runs by Issue #3286."""
    assert checker.DEFAULT_TARGET == 30


# ---------------------------------------------------------------------------
# Schema validation — happy paths
# ---------------------------------------------------------------------------


def _valid_full_payload(streak: int = 0, target: int = 30) -> dict:
    return {
        "schema_version": "1",
        "streak": streak,
        "target": target,
        "unblocked": streak >= target,
        "generated_at": "2026-09-08T06:00:00+00:00",
        "remaining": max(0, target - streak),
        "runs_considered": 0,
        "last_run_id": "12345678",
    }


def test_validate_state_accepts_full_v1_payload(checker):
    state = checker.validate_state(_valid_full_payload(streak=15))
    assert state.streak == 15
    assert state.target == 30
    assert state.unblocked is False
    assert state.remaining == 15
    assert state.last_run_id == "12345678"
    assert state.generated_at == "2026-09-08T06:00:00+00:00"


def test_validate_state_accepts_minimum_required_fields(checker):
    """The four required fields alone (no optionals) is a valid v1."""
    state = checker.validate_state(
        {
            "streak": 0,
            "target": 30,
            "unblocked": False,
            "generated_at": "2026-09-08T06:00:00+00:00",
        }
    )
    assert state.streak == 0
    assert state.remaining is None
    assert state.last_run_id is None


def test_validate_state_accepts_pre_3286_workflow_payload(checker):
    """Back-compat: the nightly workflow's inline construction omits
    ``schema_version`` and writes the v1 field set directly. This must
    validate as v1."""
    payload = {
        "streak": 5,
        "target": 30,
        "unblocked": False,
        "runs_considered": 12,
        "last_run_id": "12345678",
        "generated_at": "2026-09-08T06:00:00+00:00",
    }
    state = checker.validate_state(payload)
    assert state.streak == 5
    assert state.runs_considered == 12


def test_validate_state_accepts_streak_at_target(checker):
    state = checker.validate_state(_valid_full_payload(streak=30))
    assert state.unblocked is True
    assert state.remaining == 0


def test_validate_state_accepts_streak_above_target(checker):
    state = checker.validate_state(_valid_full_payload(streak=45))
    assert state.unblocked is True
    assert state.remaining == 0


# ---------------------------------------------------------------------------
# Schema validation — rejection paths
# ---------------------------------------------------------------------------


def test_validate_state_rejects_non_object(checker):
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state([1, 2, 3])
    assert "top-level must be an object" in str(exc.value)


def test_validate_state_rejects_missing_required_fields(checker):
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state({"streak": 0, "target": 30, "unblocked": False})
    msg = str(exc.value)
    assert "generated_at" in msg
    assert "missing required" in msg


def test_validate_state_rejects_unknown_field(checker):
    payload = _valid_full_payload()
    payload["bogus_field"] = "x"
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state(payload)
    assert "unknown fields" in str(exc.value)
    assert "bogus_field" in str(exc.value)


def test_validate_state_rejects_inconsistent_unblocked(checker):
    """``unblocked`` must agree with ``streak >= target``."""
    payload = _valid_full_payload(streak=10)
    payload["unblocked"] = True
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state(payload)
    assert "unblocked" in str(exc.value)
    assert "inconsistent" in str(exc.value)


def test_validate_state_rejects_inconsistent_remaining(checker):
    """``remaining`` must agree with ``max(0, target - streak)``."""
    payload = _valid_full_payload(streak=10)
    payload["remaining"] = 5
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state(payload)
    assert "remaining" in str(exc.value)
    assert "inconsistent" in str(exc.value)


def test_validate_state_rejects_non_iso_timestamp(checker):
    payload = _valid_full_payload()
    payload["generated_at"] = "yesterday"
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state(payload)
    assert "ISO-8601" in str(exc.value)


def test_validate_state_rejects_zero_target(checker):
    payload = _valid_full_payload(target=30)
    payload["target"] = 0
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state(payload)
    assert "target" in str(exc.value)


def test_validate_state_rejects_negative_streak(checker):
    payload = _valid_full_payload()
    payload["streak"] = -1
    payload["unblocked"] = False
    payload["remaining"] = 31
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state(payload)
    assert "streak" in str(exc.value)


def test_validate_state_rejects_unsupported_schema_version(checker):
    payload = _valid_full_payload()
    payload["schema_version"] = "2"
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state(payload)
    assert "schema_version" in str(exc.value)


def test_validate_state_rejects_bool_as_streak(checker):
    """``streak`` must be an integer; a bool is rejected even though
    ``isinstance(True, int)`` is True in Python."""
    payload = _valid_full_payload()
    payload["streak"] = True
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.validate_state(payload)
    assert "streak" in str(exc.value)


# ---------------------------------------------------------------------------
# load_state — file-level behaviour
# ---------------------------------------------------------------------------


def test_load_state_reads_state_file(checker, tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(_valid_full_payload(streak=15)), encoding="utf-8")
    state = checker.load_state(path)
    assert state.streak == 15


def test_load_state_missing_file_raises_helpful_error(checker, tmp_path):
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.load_state(tmp_path / "missing.json")
    msg = str(exc.value)
    assert "missing" in msg
    assert "--write-template" in msg


def test_load_state_malformed_json_raises_helpful_error(checker, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(checker.ValidationFailure) as exc:
        checker.load_state(path)
    assert "invalid JSON" in str(exc.value)


# ---------------------------------------------------------------------------
# write_template
# ---------------------------------------------------------------------------


def test_write_template_creates_valid_state_file(checker, tmp_path):
    path = tmp_path / "out.json"
    state = checker.write_template(path, target=30)
    assert state.streak == 0
    assert state.target == 30
    assert state.unblocked is False
    assert state.remaining == 30
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["streak"] == 0


def test_write_template_creates_parent_directories(checker, tmp_path):
    path = tmp_path / "deep" / "nested" / "state.json"
    checker.write_template(path, target=30)
    assert path.exists()


def test_write_template_honours_target(checker, tmp_path):
    path = tmp_path / "state.json"
    state = checker.write_template(path, target=10)
    assert state.target == 10
    assert state.remaining == 10


# ---------------------------------------------------------------------------
# gate_open / gate_status
# ---------------------------------------------------------------------------


def test_gate_open_false_when_streak_below_target(checker):
    state = checker.validate_state(_valid_full_payload(streak=10))
    assert checker.gate_open(state) is False


def test_gate_open_true_when_streak_at_target(checker):
    state = checker.validate_state(_valid_full_payload(streak=30))
    assert checker.gate_open(state) is True


def test_gate_status_includes_gate_open_field(checker):
    state = checker.validate_state(_valid_full_payload(streak=20))
    status = checker.gate_status(state)
    assert status["gate_open"] is False
    assert status["streak"] == 20
    assert status["target"] == 30
    assert status["remaining"] == 10


def test_gate_status_remaining_clamped_to_zero(checker):
    state = checker.validate_state(_valid_full_payload(streak=45))
    status = checker.gate_status(state)
    assert status["remaining"] == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_main(checker, monkeypatch, argv):
    """Invoke ``checker.main(argv)`` after capturing stdout/stderr."""
    from io import StringIO

    out, err = StringIO(), StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = checker.main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_cli_report_mode_exits_zero_on_valid_state(checker, monkeypatch, tmp_path):
    path = _write_state(tmp_path, _valid_full_payload(streak=5))
    rc, out, err = _run_main(checker, monkeypatch, ["--state", path])
    assert rc == 0
    assert "5/30" in out
    assert "CLOSED" in out


def test_cli_enforce_mode_exits_nonzero_when_gate_closed(
    checker, monkeypatch, tmp_path
):
    path = _write_state(tmp_path, _valid_full_payload(streak=10))
    rc, _out, err = _run_main(
        checker, monkeypatch, ["--state", path, "--gate", "enforce"]
    )
    assert rc == 1


def test_cli_enforce_mode_exits_zero_when_gate_open(checker, monkeypatch, tmp_path):
    path = _write_state(tmp_path, _valid_full_payload(streak=30))
    rc, out, _err = _run_main(
        checker, monkeypatch, ["--state", path, "--gate", "enforce"]
    )
    assert rc == 0
    assert "OPEN" in out


def test_cli_missing_state_exits_two_in_report_mode(checker, monkeypatch, tmp_path):
    """Missing state file is a script error (exit 2) in report mode."""
    rc, _out, err = _run_main(
        checker, monkeypatch, ["--state", str(tmp_path / "missing.json")]
    )
    assert rc == 2
    assert "missing" in err


def test_cli_enforce_mode_on_missing_state_exits_one(checker, monkeypatch, tmp_path):
    """Missing state file in enforce mode = gate closed (exit 1) so
    that PR-time gating fails fast."""
    rc, _out, err = _run_main(
        checker,
        monkeypatch,
        ["--state", str(tmp_path / "missing.json"), "--gate", "enforce"],
    )
    assert rc == 1
    assert "missing" in err


def test_cli_json_emits_machine_readable_output(checker, monkeypatch, tmp_path):
    path = _write_state(tmp_path, _valid_full_payload(streak=20))
    rc, out, _err = _run_main(checker, monkeypatch, ["--state", path, "--json"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["valid"] is True
    assert payload["streak"] == 20
    assert payload["gate_open"] is False


def test_cli_json_on_invalid_state_returns_error_payload(
    checker, monkeypatch, tmp_path
):
    path = tmp_path / "bad.json"
    path.write_text('{"streak": "not a number"}', encoding="utf-8")
    rc, out, _err = _run_main(checker, monkeypatch, ["--state", str(path), "--json"])
    assert rc == 2
    payload = json.loads(out)
    assert payload["valid"] is False
    assert "streak" in payload["error"]


def test_cli_write_template_writes_and_exits_zero(checker, monkeypatch, tmp_path):
    out_path = tmp_path / "fresh.json"
    rc, out, _err = _run_main(checker, monkeypatch, ["--write-template", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["streak"] == 0
    assert payload["target"] == 30
    assert "Wrote" in out


def test_cli_target_override_used_for_fresh(checker, monkeypatch, tmp_path):
    out_path = tmp_path / "fresh.json"
    rc, _out, _err = _run_main(
        checker, monkeypatch, ["--write-template", str(out_path), "--target", "5"]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["target"] == 5


def test_cli_rejects_target_below_one(checker, monkeypatch):
    rc, _out, err = _run_main(checker, monkeypatch, ["--target", "0"])
    assert rc == 2
    assert "--target" in err


def test_cli_validate_state_reports_drift(checker, monkeypatch, tmp_path):
    """Drift in the state file surfaces with a precise error message,
    not a silent green."""
    payload = _valid_full_payload()
    payload["unblocked"] = True  # inconsistent with streak=0
    payload["new_field"] = "x"  # unknown field
    path = _write_state(tmp_path, payload)
    rc, _out, err = _run_main(checker, monkeypatch, ["--state", path])
    assert rc == 2
    assert "unbounded" not in err
    assert "unblocked" in err
    assert "unknown" in err


# ---------------------------------------------------------------------------
# criterion_2_failures — Issue #3359 (β-soak Criterion 2 tracker)
# ---------------------------------------------------------------------------


def test_criterion_2_failures_returns_v1_payload(checker):
    """The canonical Criterion 2 failures summary mirrors
    ``docs/agents/beta-soak-criterion-2-tracker.md``."""
    summary = checker.criterion_2_failures()
    assert summary["schema_version"] == "1"
    assert summary["tracker_doc"] == "docs/agents/beta-soak-criterion-2-tracker.md"
    assert summary["limit_ref"] == "§LIMIT-21"
    assert summary["limit_owner_issue"] == "#3297"


def test_criterion_2_failures_lists_both_nightly_failures(checker):
    """Both nightly Criterion 2 failures are listed verbatim."""
    summary = checker.criterion_2_failures()
    names = [f["name"] for f in summary["failures"]]
    assert (
        "test_physics_thermal_model_eplus_case_600_reference_csv" in names
    ), "Case 600 oscillation test must be in the canonical list"
    assert "test_free_floating_case_900ff_isolation" in names, (
        "Case 900FF divergence test must be in the canonical list"
    )
    assert len(summary["failures"]) == 2, (
        "Only the 2 nightly Criterion 2 failures are tracked here "
        "(the wider §LIMIT-21 cohort is enumerated in §LIMIT-21 itself)"
    )


def test_criterion_2_failures_includes_panic_sites(checker):
    """The panic sites match the verbatim sources in Issue #3359."""
    summary = checker.criterion_2_failures()
    by_name = {f["name"]: f for f in summary["failures"]}
    case_600 = by_name["test_physics_thermal_model_eplus_case_600_reference_csv"]
    assert case_600["panic_site"] == "tests/zone_balance_eplus_isolation.rs:298:5"
    assert case_600["file"] == "tests/zone_balance_eplus_isolation.rs"
    assert "34.007" in case_600["symptom"]
    case_900ff = by_name["test_free_floating_case_900ff_isolation"]
    assert case_900ff["panic_site"] == "tests/zone_balance_eplus_isolation.rs:445:5"
    assert "is_finite" in case_900ff["symptom"]


def test_criterion_2_failures_cross_references_include_parent_issues(checker):
    """Cross-references link the upstream tracking chain."""
    summary = checker.criterion_2_failures()
    issues = {ref["issue"] for ref in summary["cross_references"]}
    for required in {
        "3359",  # this tracker issue
        "3354",  # parent diagnostic
        "3286",  # gate contract
        "3285",  # escape hatch
        "3291",  # Phase A8 umbrella
        "3297",  # §LIMIT-21 cohort owner
        "1465",  # Phase 3 gauge validation
        "1462",  # Phase 1b gauge shadow mode
    }:
        assert required in issues, f"cross-reference #{required} missing"


def test_criterion_2_failures_escape_hatch_pointer(checker):
    """The escape-hatch block points at the operational bypass mechanism."""
    summary = checker.criterion_2_failures()
    hatch = summary["escape_hatch"]
    assert hatch["issue"] == "3285"
    assert hatch["doc"] == "docs/agents/beta-soak-escape-hatch.md"
    assert hatch["env_var"] == "BETA_SOAK_ESCAPE_AUTHORIZED_BY"
    assert hatch["cli_flag"] == "--escape"


def test_criterion_2_failures_recovery_steps_list_closure_path(checker):
    """Recovery steps enumerate the §LIMIT-21 closure path explicitly."""
    summary = checker.criterion_2_failures()
    steps = summary["recovery_steps"]
    assert any("1465" in s and "1462" in s for s in steps), (
        "Recovery steps must reference #1465 / #1462"
    )
    assert any("§LIMIT-21" in s for s in steps), (
        "Recovery steps must cite §LIMIT-21"
    )
    assert any("--gate enforce" in s for s in steps), (
        "Recovery steps must name the PR-time gate flag"
    )


def test_criterion_2_failures_definition_of_done_enumerates_close_paths(checker):
    """Definition-of-done lists every path that legitimately closes #3359."""
    summary = checker.criterion_2_failures()
    dod = summary["definition_of_done"]
    assert any("§LIMIT-21" in s for s in dod)
    assert any("ADR-0014" in s or "supersede" in s.lower() for s in dod)
    assert any("human" in s.lower() for s in dod)


def test_cli_criterion_2_failures_emits_json(checker, monkeypatch):
    """``--criterion-2-failures`` exits 0 with the canonical JSON on stdout."""
    rc, out, _err = _run_main(
        checker, monkeypatch, ["--criterion-2-failures"]
    )
    assert rc == 0
    payload = json.loads(out)
    assert payload["schema_version"] == "1"
    assert payload["tracker_doc"] == "docs/agents/beta-soak-criterion-2-tracker.md"


def test_cli_criterion_2_failures_independent_of_state(checker, monkeypatch, tmp_path):
    """``--criterion-2-failures`` works even when no state file exists."""
    missing = tmp_path / "missing.json"
    rc, out, _err = _run_main(
        checker, monkeypatch, ["--state", str(missing), "--criterion-2-failures"]
    )
    assert rc == 0, "criterion-2 summary must work independent of state validity"
    payload = json.loads(out)
    assert len(payload["failures"]) == 2


def test_cli_criterion_2_failures_short_circuits_other_flags(
    checker, monkeypatch, tmp_path
):
    """``--criterion-2-failures`` short-circuits before state validation, so
    a malformed state + ``--gate enforce`` does NOT exit 1."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    rc, _out, _err = _run_main(
        checker,
        monkeypatch,
        ["--state", str(bad), "--gate", "enforce", "--criterion-2-failures"],
    )
    assert rc == 0
