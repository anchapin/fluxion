"""
Tests for ``scripts/check_required_checks_sync.py`` -- Issue #2866.

Regression guard for the ``release_gates.yaml`` <-> ``.github/workflows/``
drift gate. Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern
from ``test_check_known_issues_stale.py`` and
``test_check_physics_sim_cycle.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* redirect the module-level path constants at a synthetic ``tmp_path``
  fixture that contains a minimal but realistic ``release_gates.yaml``
  + ``.github/workflows/`` tree, then
* drive ``main()`` through clean and planted-violation scenarios.

The script reads two paths at module-import time (``RELEASE_GATES_YAML``,
``WORKFLOWS_DIR``), so the freshly-loaded module carries the *real* repo
paths. Each test that wants a synthetic fixture must therefore redirect
those constants before calling ``main()``.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

SCRIPT_NAME = "check_required_checks_sync"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the drift-detection script."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> Path:
    """Point the script's ``RELEASE_GATES_YAML`` and ``WORKFLOWS_DIR`` at a
    synthetic ``tmp_path`` tree and return the resolved ``RELEASE_GATES_YAML``
    path.

    Both constants are computed at import time from the script's location
    via ``Path(__file__).resolve().parent.parent`` / ``Path(...)``.
    Each test that wants a synthetic fixture must therefore redirect the
    constants before calling ``main()``.
    """
    target = tmp_path / "release_gates.yaml"
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(checker, "RELEASE_GATES_YAML", target)
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", wf_dir)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    return target


def _write(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(text), encoding="utf-8")
    return p


def _workflow_yaml(job_name: str, triggers=("pull_request",)) -> str:
    """Build a minimal single-job workflow with the given ``name:`` and
    ``on:`` triggers.

    Default trigger set is just ``pull_request``. The job name is placed
    after ``runs-on:`` to mirror the real workflows that have it second
    (e.g. ``architecture_drift.yml``) — the parser must tolerate that.
    """
    on_block = "\n".join(f"  {t}:" for t in triggers)
    return (
        f"name: Test Workflow\n"
        f"\n"
        f"on:\n"
        f"{on_block}\n"
        f"\n"
        f"jobs:\n"
        f"  the-job:\n"
        f"    runs-on: ubuntu-latest\n"
        f"    name: {job_name}\n"
        f"    steps:\n"
        f"      - run: echo hello\n"
    )


def _release_gates_yaml(required_checks, workflow_index) -> str:
    """Build a minimal ``release_gates.yaml`` with the given required_checks
    and workflow_index lists.

    YAML quoting is left to PyYAML on the output side; we just emit the
    shape verbatim so a typo in the script's parser shows up as a clear
    failure rather than a YAML round-trip surprise.
    """
    rc_lines = "\n".join(f'    - "{c}"' for c in required_checks)
    wi_lines = []
    for entry in workflow_index:
        wi_lines.append(f"    - job: \"{entry['job']}\"")
        wi_lines.append(f'      workflow: "{entry["workflow"]}"')
        if "issue" in entry:
            wi_lines.append(f"      issue: {entry['issue']}")
    return (
        f"ci:\n"
        f"  required_checks:\n"
        f"{rc_lines}\n"
        f"  workflow_index:\n"
        f"{chr(10).join(wi_lines)}\n"
    )


# ---------------------------------------------------------------------------
# main() — clean / violation scenarios
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_release_gates_and_workflows_are_in_sync(
    checker, tmp_path, monkeypatch, capsys
):
    """Clean fixture: required_check + workflow_index + workflow job all match.

    This is the post-fix state of the real repo (issue #2866 acceptance
    criteria): every required_check has a workflow_index entry pointing
    at a workflow whose ``jobs.<id>.name`` equals the required_check.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["My Gate (Issue #1234)"],
            workflow_index=[
                {
                    "job": "My Gate (Issue #1234)",
                    "workflow": ".github/workflows/my.yml",
                    "issue": 1234,
                },
            ],
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / ".github" / "workflows" / "my.yml",
        _workflow_yaml("My Gate (Issue #1234)"),
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "No drift" in out


def test_main_fails_when_required_check_has_no_workflow_index_entry(
    checker, tmp_path, monkeypatch, capsys
):
    """Planted violation: a required_check without a workflow_index entry.

    This is exactly the "Energy Conservation (GH)" / "Rustfmt (GH)" / ...
    drift that issue #2866 documents. The gate must refuse to pass.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["Orphan Required Check"],
            workflow_index=[],
        ),
        encoding="utf-8",
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "DRIFT DETECTED" in out
    assert "Orphan Required Check" in out
    assert "no workflow_index entry" in out


def test_main_fails_when_workflow_index_job_does_not_match_actual_job_name(
    checker, tmp_path, monkeypatch, capsys
):
    """Planted violation: workflow_index points at a job name that the
    workflow does not actually define.

    Issue #2866's headline bug: a workflow_index entry says
    ``job: "Foo"`` but the workflow's only job is ``name: Bar``. Branch
    protection wouldn't match either — the gate must catch this.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["Renamed Job"],
            workflow_index=[
                {
                    "job": "Renamed Job",
                    "workflow": ".github/workflows/wf.yml",
                },
            ],
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / ".github" / "workflows" / "wf.yml",
        _workflow_yaml("Different Job Name"),
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found in .github/workflows/wf.yml" in out


def test_main_fails_when_workflow_has_no_pull_request_or_workflow_run_trigger(
    checker, tmp_path, monkeypatch, capsys
):
    """Planted violation: workflow_index references a workflow that
    triggers only on ``schedule`` / ``workflow_dispatch``. A check run
    that never fires on a PR cannot be a branch-protection required
    check, so the gate must refuse.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["Nightly Gate"],
            workflow_index=[
                {"job": "Nightly Gate", "workflow": ".github/workflows/nightly.yml"},
            ],
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / ".github" / "workflows" / "nightly.yml",
        _workflow_yaml("Nightly Gate", triggers=("schedule", "workflow_dispatch")),
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "no pull_request or workflow_run trigger" in out
    assert "['schedule', 'workflow_dispatch']" in out


def test_main_fails_when_workflow_index_references_missing_workflow_file(
    checker, tmp_path, monkeypatch, capsys
):
    """Planted violation: workflow_index points at a workflow file that
    does not exist on disk.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["Ghost Gate"],
            workflow_index=[
                {"job": "Ghost Gate", "workflow": ".github/workflows/ghost.yml"},
            ],
        ),
        encoding="utf-8",
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "workflow file not found" in out
    assert "ghost.yml" in out


# ---------------------------------------------------------------------------
# Canonical-vs-suffix drift (Issue #3116): strict-match semantics
# ---------------------------------------------------------------------------


def test_main_fails_on_canonical_vs_suffix_drift(
    checker, tmp_path, monkeypatch, capsys
):
    """Planted violation: YAML holds the bare canonical name but the
    workflow only emits suffixed variants.

    Issue #3116's regression guard: GitHub branch protection's contexts
    array matches the emitted ``jobs.<id>.name`` *verbatim* — there is
    no canonical-vs-suffix tolerance. Before the #3116 fix, the YAML
    held the canonical name (e.g. ``My Multi Runner Check``) and the
    workflow emitted ``My Multi Runner Check (GH)`` / ``(Hetzner
    Overflow)``, which the script's regex tolerance silently accepted.
    The post-fix contract requires the YAML to name the suffixed
    variant explicitly. The script must FAIL this scenario with a
    canonical-vs-suffix drift message that names the suffixed variants.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["My Multi Runner Check"],
            workflow_index=[
                {
                    "job": "My Multi Runner Check",
                    "workflow": ".github/workflows/multi.yml",
                },
            ],
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / ".github" / "workflows" / "multi.yml",
        dedent(
            """\
            name: Multi Runner
            on:
              pull_request:
            jobs:
              gh:
                runs-on: ubuntu-latest
                name: My Multi Runner Check (GH)
                steps:
                  - run: echo gh
              hz:
                runs-on: [self-hosted, overflow]
                name: My Multi Runner Check (Hetzner Overflow)
                steps:
                  - run: echo hz
            """
        ),
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "canonical-vs-suffix drift" in out or "in canonical form but" in out
    assert "My Multi Runner Check (GH)" in out
    assert "My Multi Runner Check (Hetzner Overflow)" in out


def test_main_passes_when_workflow_index_uses_suffixed_name(
    checker, tmp_path, monkeypatch, capsys
):
    """Correct post-fix shape: YAML holds the suffixed name explicitly.

    This is the contract `release_gates.yaml` uses after the #3116 fix:
    ``- "Workspace Check (GH)"`` is in ``ci.required_checks`` because
    the workflow job emits exactly that name. The script must accept
    the suffixed entry verbatim.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["My Multi Runner Check (GH)"],
            workflow_index=[
                {
                    "job": "My Multi Runner Check (GH)",
                    "workflow": ".github/workflows/multi.yml",
                },
            ],
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / ".github" / "workflows" / "multi.yml",
        dedent(
            """\
            name: Multi Runner
            on:
              pull_request:
            jobs:
              gh:
                runs-on: ubuntu-latest
                name: My Multi Runner Check (GH)
                steps:
                  - run: echo gh
              hz:
                runs-on: [self-hosted, overflow]
                name: My Multi Runner Check (Hetzner Overflow)
                steps:
                  - run: echo hz
            """
        ),
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "No drift" in out


def test_main_passes_when_single_job_name_matches_workflow_name(
    checker, tmp_path, monkeypatch, capsys
):
    """Single-job workflow where the job's ``name:`` field equals the
    workflow's own ``name:`` field — both match the YAML required_check.

    Post-#3116: the YAML must match the emitted ``jobs.<id>.name`` exactly.
    For a single-job workflow whose job name happens to equal the
    workflow name (the post-fix ``Architecture Drift Detection`` case),
    ``job_in_workflow`` returns True because the job name matches. No
    workflow-name fallback is needed — the script doesn't apply one.

    Contrast with the pre-#3116 case where the job was named
    ``Check ARCHITECTURE.md drift`` (different from the workflow name)
    and only the workflow-name fallback made the gate accept it; that
    shape is now rejected (see ``test_main_fails_when_single_job_name
    _diverges_from_workflow_name`` below).
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["Architecture Drift Detection"],
            workflow_index=[
                {
                    "job": "Architecture Drift Detection",
                    "workflow": ".github/workflows/arch.yml",
                },
            ],
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / ".github" / "workflows" / "arch.yml",
        dedent(
            """\
            name: Architecture Drift Detection
            on:
              schedule:
                - cron: '0 3 * * *'
              pull_request:
            jobs:
              check-drift:
                runs-on: ubuntu-latest
                name: Architecture Drift Detection
                steps:
                  - run: echo drift
            """
        ),
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "No drift" in out


def test_main_fails_when_single_job_name_diverges_from_workflow_name(
    checker, tmp_path, monkeypatch, capsys
):
    """Single-job workflow where the job's ``name:`` field differs from
    the workflow's own ``name:`` field.

    Pre-#3116 shape: YAML held ``Architecture Drift Detection`` (matching
    the workflow's top-level ``name:``), but the actual job was named
    ``Check ARCHITECTURE.md drift``. The pre-fix script's
    ``job_in_workflow`` had a single-job workflow-name fallback that
    silently accepted this — and GitHub branch protection would never
    satisfy because the emitted check name was the job name, not the
    workflow name. Post-#3116 the script refuses to fall back to the
    workflow name for single-job workflows.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["Architecture Drift Detection"],
            workflow_index=[
                {
                    "job": "Architecture Drift Detection",
                    "workflow": ".github/workflows/arch.yml",
                },
            ],
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / ".github" / "workflows" / "arch.yml",
        dedent(
            """\
            name: Architecture Drift Detection
            on:
              schedule:
                - cron: '0 3 * * *'
              pull_request:
            jobs:
              check:
                runs-on: ubuntu-latest
                name: Check Drift
                steps:
                  - run: echo drift
            """
        ),
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "not found in" in out


# ---------------------------------------------------------------------------
# Informational findings (WASM-style: intentionally not in required_checks)
# ---------------------------------------------------------------------------


def test_main_reports_informational_when_workflow_index_entry_not_required(
    checker, tmp_path, monkeypatch, capsys
):
    """WASM Build pattern: workflow_index entry intentionally not in
    required_checks (per the YAML comment). The gate reports this as
    INFORMATIONAL (not a failure) so a future drift isn't silent but
    doesn't block this documented opt-in case.
    """
    target = _redirect(checker, tmp_path, monkeypatch)
    target.write_text(
        _release_gates_yaml(
            required_checks=["Required Gate"],
            workflow_index=[
                {"job": "Required Gate", "workflow": ".github/workflows/r.yml"},
                {
                    "job": "Opt In Gate",
                    "workflow": ".github/workflows/o.yml",
                },
            ],
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / ".github" / "workflows" / "r.yml",
        _workflow_yaml("Required Gate"),
    )
    _write(
        tmp_path / ".github" / "workflows" / "o.yml",
        _workflow_yaml("Opt In Gate"),
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "INFORMATIONAL" in out
    assert "'Opt In Gate'" in out


# ---------------------------------------------------------------------------
# Parsing primitives
# ---------------------------------------------------------------------------


def test_parse_workflow_extracts_triggers_and_job_names(checker, tmp_path):
    """Unit-level test of the regex-based workflow parser. Guards against
    a silent regex regression (the script can't fall back to PyYAML —
    GitHub Actions-specific ``${{ }}`` expressions choke it).
    """
    p = _write(
        tmp_path / "wf.yml",
        dedent(
            """\
            name: Example
            on:
              pull_request:
              workflow_run:
                workflows: ["Other"]
              schedule:
                - cron: '0 3 * * *'
            jobs:
              alpha:
                runs-on: ubuntu-latest
                name: Alpha Job
                steps:
                  - run: echo a
              beta:
                runs-on: ubuntu-latest
                name: Beta Job (${{ matrix.os }})
                steps:
                  - run: echo b
            """
        ),
    )
    result = checker.parse_workflow(p)
    assert result["workflow_name"] == "Example"
    assert sorted(result["triggers"]) == ["pull_request", "schedule", "workflow_run"]
    assert result["jobs"]["alpha"] == "Alpha Job"
    assert result["jobs"]["beta"] == "Beta Job (${{ matrix.os }})"


def test_job_in_workflow_strict_exact_match(checker):
    """Unit-level test of ``job_in_workflow`` post-#3116: exact match only.

    The pre-#3116 implementation applied a ``CANONICAL_NAME_SUFFIXES``
    regex tolerance (matched ``canonical`` against ``canonical + " (GH)"``
    / ``canonical + " (Hetzner Overflow)"``) and a single-job workflow-
    name fallback. Both were the root cause of #3116's drift and have
    been removed. This test pins the new contract: ``job_in_workflow``
    matches iff a job's emitted ``name:`` equals the entry byte-for-byte.
    """
    wf = {
        "jobs": {
            "exact": "Exact Job",
            "gh": "Suffixed (GH)",
            "hz": "Suffixed (Hetzner Overflow)",
        },
        "workflow_name": "Some Workflow",
        "triggers": ["pull_request"],
    }
    # Single-job workflow with divergent job name (pre-#3116 foot-gun)
    single_divergent = {
        "jobs": {"only": "Job Name"},
        "workflow_name": "Different From Job",
        "triggers": ["pull_request"],
    }

    # Exact match
    assert checker.job_in_workflow(wf, "Exact Job")
    # Suffixed variants are NOT matched by the bare canonical anymore
    assert not checker.job_in_workflow(wf, "Suffixed")
    # The suffixed variants themselves DO match themselves
    assert checker.job_in_workflow(wf, "Suffixed (GH)")
    assert checker.job_in_workflow(wf, "Suffixed (Hetzner Overflow)")
    # Single-job workflow with divergent name: workflow_name fallback is
    # gone, so the YAML name must match the job name (or fail).
    assert not checker.job_in_workflow(single_divergent, "Different From Job")
    assert checker.job_in_workflow(single_divergent, "Job Name")
    # Mismatch
    assert not checker.job_in_workflow(wf, "No Such Job")


def test_workflow_has_suffixed_variant(checker):
    """Unit-level test of ``workflow_has_suffixed_variant`` — the helper
    that drives the canonical-vs-suffix drift detection added in #3116.
    """
    wf_with_suffixes = {
        "jobs": {
            "gh": "Workspace Check (GH)",
            "hz": "Workspace Check (Hetzner Overflow)",
        },
        "workflow_name": "rust-tests",
        "triggers": ["pull_request"],
    }
    wf_no_suffixes = {
        "jobs": {"only": "Some Job"},
        "workflow_name": "Single",
        "triggers": ["pull_request"],
    }
    # Bare canonical → workflow has suffixed variants
    assert checker.workflow_has_suffixed_variant(wf_with_suffixes, "Workspace Check")
    # Already-suffixed query → no (it's asking about the bare canonical)
    assert not checker.workflow_has_suffixed_variant(
        wf_with_suffixes, "Workspace Check (GH)"
    )
    # Workflow without suffixed jobs → no
    assert not checker.workflow_has_suffixed_variant(wf_no_suffixes, "Some Job")
    assert not checker.workflow_has_suffixed_variant(wf_no_suffixes, "Single")


def test_has_blocking_trigger_accepts_pull_request_and_workflow_run(checker):
    """Only ``pull_request`` and ``workflow_run`` produce PR-blockable
    check runs. ``push``, ``schedule``, ``workflow_dispatch`` alone do
    not — they cannot be required status checks.
    """
    assert checker.has_blocking_trigger({"triggers": ["pull_request"]})
    assert checker.has_blocking_trigger({"triggers": ["workflow_run"]})
    assert checker.has_blocking_trigger({"triggers": ["push", "pull_request"]})
    assert not checker.has_blocking_trigger({"triggers": ["push", "schedule"]})
    assert not checker.has_blocking_trigger({"triggers": ["workflow_dispatch"]})
    assert not checker.has_blocking_trigger({"triggers": []})


# ---------------------------------------------------------------------------
# Module-level constants pin
# ---------------------------------------------------------------------------


def test_blocking_triggers_constant(checker):
    """``BLOCKING_TRIGGERS`` must remain ``{pull_request, workflow_run}``.

    Adding e.g. ``push`` here would silently allow non-PR-blockable
    workflows into required_checks; removing ``workflow_run`` would
    break the listener-gate pattern (Determinism, Performance, etc.).
    """
    assert checker.BLOCKING_TRIGGERS == frozenset({"pull_request", "workflow_run"})


def test_canonical_name_suffixes_constant(checker):
    """``CANONICAL_NAME_SUFFIXES`` documents the
    `` (GH) / (Hetzner Overflow)`` pattern (note the leading space —
    the suffix is appended to the canonical listener name). Adding a
    third canonical suffix without updating the comment block above
    each ``workflow_index`` entry would silently broaden what the gate
    accepts; removing a suffix would re-introduce the original drift.
    """
    assert " (GH)" in checker.CANONICAL_NAME_SUFFIXES
    assert " (Hetzner Overflow)" in checker.CANONICAL_NAME_SUFFIXES