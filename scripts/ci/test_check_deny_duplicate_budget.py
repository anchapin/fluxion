"""
Tests for ``scripts/check_deny_duplicate_budget.py`` -- Issue #2994.

Regression guard for the deny.toml [bans] duplicate-version budget
gate. Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_audit_ignores_fresh.py`` (#3075) and
``test_check_required_checks_sync.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level ``BASELINE_FILE`` / ``DENY_TOML`` /
  ``CARGO_LOCK`` constants at a synthetic ``tmp_path`` tree, then
* drive ``main()`` through in-budget / over-budget / missing-config /
  baseline-toml-drift scenarios by stubbing ``run_cargo_deny`` with a
  synthetic JSONL payload.

Issue #2994 acceptance criteria are realised as five scenarios:

1. **Real baseline is parseable** -- the shipped baseline JSON at
   ``tests/reference_data/deny_budget_baseline.json`` round-trips
   through ``load_baseline`` and exposes the schema keys the script
   expects (``total_duplicates``, ``duplicates_baseline``,
   ``clusters``).
2. **Synthetic in-budget count** -- ``main()`` returns ``0`` when the
   mocked ``run_cargo_deny`` produces a JSONL stream with fewer
   ``code=duplicate`` diagnostics than the baseline.
3. **Synthetic over-budget count** -- ``main()`` returns ``1`` when the
   mocked stream exceeds the baseline. Regression test for the budget
   direction (a future regression where the script silently passes on
   growth would trip this).
4. **Missing baseline file** -- ``main()`` returns ``2`` when the
   baseline JSON is absent (script-error path; not a quiet PASS).
5. **deny.toml inline baseline drift** -- when the inline
   ``# duplicates_baseline: N`` comment in the synthetic deny.toml
   disagrees with the JSON baseline, ``main()`` returns ``2`` and
   points at the drift. Mirrors the issue's "single source of truth"
   requirement.

The harness keeps the existing fixtures from ``scripts/ci/conftest.py``
(``load_script``) and adds local helpers for synthesizing JSONL cargo
deny output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import pytest

SCRIPT_NAME = "check_deny_duplicate_budget"


def _scrub_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset ``sys.argv`` so the script's argparse doesn't see pytest's CLI.

    The script's ``main()`` calls ``argparse.ArgumentParser().parse_args()``
    with no explicit args, which defaults to ``sys.argv[1:]``. Under pytest
    that's ``["-v", "--no-cov", ...]`` and argparse rejects the unknown
    flags with exit 2, masking whatever the test was trying to verify.
    """
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME])


def _write(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(text), encoding="utf-8")
    return p


def _make_duplicate_diagnostic(crate: str, n_versions: int) -> dict:
    """Build a single cargo-deny JSONL line for a ``code=duplicate`` diagnostic.

    Mirrors the JSON shape ``cargo deny -f json check bans`` emits on
    stdout (one JSON object per line). The ``labels`` / ``graphs``
    blocks are intentionally minimal -- the script only parses
    ``type``, ``fields.code``, ``fields.message`` and (for ``--json-out``
    consumers) ``labels[].line``.
    """
    return {
        "type": "diagnostic",
        "fields": {
            "code": "duplicate",
            "message": f"found {n_versions} duplicate entries for crate '{crate}'",
            "graphs": [
                {"Krate": {"name": crate, "version": f"0.{i}.0"}}
                for i in range(n_versions)
            ],
            "labels": [
                {"column": 1, "line": 100 + n_versions, "message": "lock entries"}
            ],
        },
    }


def _make_summary(warnings: int) -> dict:
    return {
        "type": "summary",
        "fields": {"bans": {"errors": 0, "helps": 0, "notes": 0, "warnings": warnings}},
    }


def _make_jsonl(crates: list[tuple[str, int]]) -> str:
    """Return the JSONL payload that ``run_cargo_deny`` would produce for ``crates``.

    ``crates`` is a list of ``(crate_name, n_versions)`` pairs; each pair
    becomes one ``code=duplicate`` diagnostic and the trailing summary
    line carries the grand total.
    """
    payload = [_make_duplicate_diagnostic(c, n) for c, n in crates]
    payload.append(_make_summary(sum(n for _, n in crates)))
    return "\n".join(json.dumps(line) for line in payload) + "\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the duplicate-version budget script."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    """Point the script's ``BASELINE_FILE`` / ``DENY_TOML`` / ``CARGO_LOCK`` /
    ``REPO_ROOT`` at synthetic files in ``tmp_path`` and return their resolved
    paths.

    All four constants are computed at import time from the script's
    location, so each test that wants a synthetic fixture must redirect
    them before calling ``main()``.
    """
    baseline = tmp_path / "tests" / "reference_data" / "deny_budget_baseline.json"
    deny_toml = tmp_path / "deny.toml"
    cargo_lock = tmp_path / "Cargo.lock"
    monkeypatch.setattr(checker, "BASELINE_FILE", baseline)
    monkeypatch.setattr(checker, "DENY_TOML", deny_toml)
    monkeypatch.setattr(checker, "CARGO_LOCK", cargo_lock)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    return baseline, deny_toml, cargo_lock


def _baseline_payload(
    duplicates_baseline: int = 45,
    wildcards_baseline: int = 1,
    total_duplicates: int = 45,
    schema_version: int = 1,
    captured_at: str = "2026-08-17",
) -> dict:
    """Build a minimal-but-valid baseline JSON for synthetic tests.

    Mirrors the schema_version=1 contract of
    ``tests/reference_data/deny_budget_baseline.json``. Tests can override
    individual fields to drive the budget arithmetic.
    """
    return {
        "_doc": "synthetic baseline for tests",
        "schema_version": schema_version,
        "captured_at": captured_at,
        "total_duplicates": total_duplicates,
        "duplicates_baseline": duplicates_baseline,
        "wildcards_baseline": wildcards_baseline,
        "milestones": [],
        "clusters": [
            {
                "name": "test-cluster",
                "crates": ["example"],
                "crate_versions": {"example": ["1.0.0", "2.0.0"]},
                "diagnostic_count": 1,
                "source": "synthetic",
                "removal_strategy": "n/a",
                "issue_ref": "#2994-test",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Real-baseline fixture (regression-locks the shipped artifact)
# ---------------------------------------------------------------------------


def test_real_baseline_json_is_parseable_and_well_formed(checker):
    """Issue #2994 acceptance: the shipped baseline JSON must round-trip.

    This is the regression lock against accidental schema drift on
    ``tests/reference_data/deny_budget_baseline.json``. A schema
    breakage that takes out ``load_baseline`` would otherwise only be
    caught when CI runs the script for real -- and only if the right
    branch is exercised.

    The test pins three things:

    * ``json.loads`` accepts the file without raising;
    * the schema-version-1 required keys are present and well-typed;
    * the cluster count is consistent with ``total_duplicates``
      (each cluster contributes its ``diagnostic_count``).
    """
    real = (
        Path(checker.REPO_ROOT)
        / "tests"
        / "reference_data"
        / "deny_budget_baseline.json"
    )
    assert real.exists(), f"baseline file missing at {real}"
    data = checker.load_baseline(real)

    assert isinstance(data, dict)
    assert data.get("schema_version") == 1
    assert isinstance(data.get("total_duplicates"), int)
    assert isinstance(data.get("duplicates_baseline"), int)
    assert isinstance(data.get("clusters"), list)
    assert data["clusters"], "clusters must be non-empty"

    diag_total = sum(int(c.get("diagnostic_count", 0)) for c in data["clusters"])
    assert diag_total == data["total_duplicates"], (
        f"sum of cluster diagnostic_count ({diag_total}) "
        f"disagrees with total_duplicates ({data['total_duplicates']})"
    )

    # The real baseline captures the live 45-duplicate state per issue
    # #2933's gate baseline. Pin this so a future reduction PR can only
    # LOWER this number (RULES.md: no parameter tuning to hide growth).
    assert data["total_duplicates"] == 45
    assert data["duplicates_baseline"] == 45


# ---------------------------------------------------------------------------
# Synthetic-tree main() scenarios
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_live_count_within_budget(
    checker, tmp_path, monkeypatch, capsys
):
    """In-budget scenario: the synthetic JSONL has 10 duplicates; the
    synthetic baseline allows 45 -> ``main()`` returns ``0``.

    Mirrors the production happy-path: live count equals baseline on the
    develop branch and the gate is currently green.
    """
    baseline, deny_toml, _ = _redirect(checker, tmp_path, monkeypatch)
    _write(
        baseline,
        json.dumps(_baseline_payload(duplicates_baseline=45)),
    )
    # deny.toml inline comment matches the JSON baseline so the cross-
    # check in main() doesn't fire its drift-detector.
    _write(
        deny_toml,
        '[bans]\nmultiple-versions = "warn"\n# duplicates_baseline: 45\n',
    )

    # Build a 10-crate synthetic JSONL stream.
    crates = [(f"crate_{i:02d}", 2) for i in range(10)]
    monkeypatch.setattr(checker, "run_cargo_deny", lambda *a, **k: _make_jsonl(crates))

    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected PASS, got rc={rc}\noutput:\n{out}"
    assert "PASS" in out
    assert "10" in out  # live count


def test_main_returns_one_when_live_count_exceeds_budget(
    checker, tmp_path, monkeypatch, capsys
):
    """Over-budget scenario: synthetic JSONL has 50 duplicates; baseline
    allows 45 -> ``main()`` returns ``1``.

    Regression-locks the budget direction: a future bug that silently
    passes when the count grows (e.g. the comparison flipped to ``<``,
    or the baseline was hard-coded to ``0``) would trip this test. Issue
    #2933 specifically calls out that the gate must catch REGRESSIONS,
    not just the absolute-count ceiling.
    """
    baseline, deny_toml, _ = _redirect(checker, tmp_path, monkeypatch)
    _write(
        baseline,
        json.dumps(_baseline_payload(duplicates_baseline=45)),
    )
    _write(
        deny_toml,
        '[bans]\nmultiple-versions = "warn"\n# duplicates_baseline: 45\n',
    )

    # 50 crates * 2 versions each = 50 duplicate diagnostics (over budget).
    crates = [(f"crate_{i:02d}", 2) for i in range(50)]
    monkeypatch.setattr(checker, "run_cargo_deny", lambda *a, **k: _make_jsonl(crates))

    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, f"expected FAIL (over budget), got rc={rc}\noutput:\n{out}"
    assert "FAIL" in out
    assert "exceeds baseline 45" in out
    assert "delta +5" in out


def test_main_returns_two_when_baseline_file_missing(
    checker, tmp_path, monkeypatch, capsys
):
    """Missing-baseline scenario: the baseline JSON is absent ->
    ``main()`` returns ``2`` (script-error path).

    Pin the FAIL-CLOSED contract: the script does NOT silently fall back
    to PASS when the source of truth is missing. Issue #2994 explicitly
    tracks the baseline JSON as the machine-readable cluster inventory;
    if it goes missing, the gate must surface this loudly.
    """
    baseline, deny_toml, _ = _redirect(checker, tmp_path, monkeypatch)
    # Intentionally do NOT write the baseline file.
    _write(
        deny_toml,
        '[bans]\nmultiple-versions = "warn"\n# duplicates_baseline: 45\n',
    )
    _scrub_argv(monkeypatch)
    rc = checker.main()
    out = capsys.readouterr().out + capsys.readouterr().err
    assert rc == 2, f"expected script-error exit, got rc={rc}\noutput:\n{out}"
    assert "baseline" in out.lower()


def test_main_returns_two_when_deny_toml_inline_baseline_drifts(
    checker, tmp_path, monkeypatch, capsys
):
    """deny.toml inline baseline drift: the JSON baseline says 45 but the
    ``# duplicates_baseline: N`` comment in ``deny.toml`` says 30 ->
    ``main()`` returns ``2``.

    Issue #2933's CI step parses the deny.toml comment (because
    cargo-deny 0.20.2 rejects unknown keys in ``[bans]``). The JSON
    baseline is the script's source of truth. When the two disagree,
    the script must surface the drift rather than silently pick one --
    otherwise a future PR can lower one without lowering the other and
    the CI step would gate on the wrong number.
    """
    baseline, deny_toml, _ = _redirect(checker, tmp_path, monkeypatch)
    _write(
        baseline,
        json.dumps(_baseline_payload(duplicates_baseline=45)),
    )
    # deny.toml declares 30 but JSON says 45 -> drift.
    _write(
        deny_toml,
        '[bans]\nmultiple-versions = "warn"\n# duplicates_baseline: 30\n',
    )

    # Don't even need to stub run_cargo_deny -- drift is detected before
    # the cargo deny invocation.
    _scrub_argv(monkeypatch)
    rc = checker.main()
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 2, f"expected drift exit 2, got rc={rc}\noutput:\n{out}"
    # The drift message prints to stderr; capture both streams.
    assert "disagrees" in out, f"drift error not surfaced in output:\n{out}"
    assert "deny.toml" in out and "duplicates_baseline" in out


# ---------------------------------------------------------------------------
# Output structure (issue #2994 acceptance criterion: machine-readable JSON)
# ---------------------------------------------------------------------------


def test_main_json_out_emits_expected_structured_payload(
    checker, tmp_path, monkeypatch, capsys
):
    """``--json-out`` mode must emit a parseable JSON object with the
    structured fields the issue's "machine-readable cluster inventory"
    acceptance criterion requires.

    Pins the schema so a downstream consumer (CI dashboard, future
    tracking script) can rely on ``payload['baseline_file']``,
    ``payload['live_count']``, ``payload['over_budget']``,
    ``payload['crates']``, and the other documented fields.
    """
    baseline, deny_toml, _ = _redirect(checker, tmp_path, monkeypatch)
    _write(
        baseline,
        json.dumps(_baseline_payload(duplicates_baseline=45)),
    )
    _write(
        deny_toml,
        '[bans]\nmultiple-versions = "warn"\n# duplicates_baseline: 45\n',
    )

    crates = [("alpha", 2), ("beta", 2), ("gamma", 3)]
    monkeypatch.setattr(checker, "run_cargo_deny", lambda *a, **k: _make_jsonl(crates))

    _scrub_argv(monkeypatch)
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME, "--json-out"])
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, f"expected PASS, got rc={rc}\noutput:\n{out}"

    # The script writes "--- JSON output ---" then the object. Find the
    # first '{' after that marker and parse from there.
    marker = "--- JSON output ---"
    assert marker in out, f"--json-out marker missing from output:\n{out}"
    tail = out[out.index(marker) + len(marker) :].strip()
    payload = json.loads(tail)

    # Pin the schema the issue's machine-readable inventory requires.
    assert payload["duplicates_baseline"] == 45
    assert payload["live_count"] == 3
    assert payload["over_budget"] is False
    assert payload["delta"] == -42  # 3 - 45
    assert payload["summary_warnings"] == 7  # 2 + 2 + 3
    assert sorted(payload["crates"]) == ["alpha", "beta", "gamma"]
    assert payload["schema_version"] == 1
    assert "baseline_file" in payload
    assert "clusters_in_baseline" in payload
