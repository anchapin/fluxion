"""Tests for ``scripts/check_concurrency_keys.py`` -- Issue #3444.

The guard enforces the ADR-0015 per-``head_sha`` ``concurrency:``
template (Issue #3366) on every ``.github/workflows/*.yml`` but, until
#3444 wired it up, had zero CI references and no pytest coverage — its
docstring claimed wiring that did not exist. These tests drive the
matcher's documented invariants against hermetic ``tmp_path`` fixture
workflows so a regex regression (e.g. the block extractor matching a
nested ``concurrency:`` key, or the folded-scalar span shrinking below
the ``|| github.ref`` line) fails here instead of on a real PR.

Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_branch_protection_diff.py`` (#3426):

* ``extract_block`` — top-level ``concurrency:`` block detection,
* ``block_uses_per_sha_group`` / ``block_uses_conditional_cancel`` —
  the ADR-0015 marker tests,
* ``check_workflow`` — per-file findings for each planted violation,
* ``main()`` exit-code contract: 0 no drift / 1 drift / 2 script error.

All fixture workflows are synthetic files written under ``tmp_path`` —
no test depends on the network or on the repo's real workflows (the
end-to-end against the real tree is the direct
``check_concurrency_keys.py`` invocation in ``scripts-tests.yml``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_concurrency_keys"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_concurrency_keys.py``."""
    return load_script(SCRIPT_NAME)


# The ADR-0015 template exactly as `scripts/update_concurrency_keys.py`
# applies it (mirrors the block carried by every real workflow). Kept as
# hand-built strings — not round-tripped through a YAML emitter — so a
# parser/matcher regression surfaces instead of being hidden by
# re-serialization (same rationale as the #3426 companion).
_PER_SHA_GROUP = """\
  group: >-
    ${{ github.workflow }}-
    ${{
      github.event_name == 'pull_request' &&
      github.event.pull_request.head.sha
      || github.ref
    }}
"""

_CONDITIONAL_CANCEL = """\
  cancel-in-progress: >-
    ${{
      github.event_name == 'push'
      && contains('refs/heads/main,refs/heads/develop', github.ref)
    }}
"""

_COMPLIANT_WORKFLOW = (
    "name: CI\n"
    "\n"
    "on:\n"
    "  push:\n"
    "\n"
    "concurrency:\n"
    + _PER_SHA_GROUP
    + _CONDITIONAL_CANCEL
    + "\n"
    "jobs:\n"
    "  build:\n"
    "    runs-on: ubuntu-latest\n"
)

# Invariant 1 violation: no `concurrency:` key anywhere.
_NO_BLOCK_WORKFLOW = (
    "name: CI\n"
    "\n"
    "on:\n"
    "  push:\n"
    "\n"
    "jobs:\n"
    "  build:\n"
    "    runs-on: ubuntu-latest\n"
)

# `concurrency:` declared only inside a job — must NOT satisfy the
# top-level invariant (the extractor anchors on column 0).
_NESTED_ONLY_WORKFLOW = (
    "name: CI\n"
    "\n"
    "on:\n"
    "  push:\n"
    "\n"
    "jobs:\n"
    "  build:\n"
    "    runs-on: ubuntu-latest\n"
    "    concurrency:\n"
    "      group: ${{ github.ref }}\n"
    "      cancel-in-progress: true\n"
)

# Invariant 2 (group half) violation: block present, `group:` absent.
_NO_GROUP_WORKFLOW = (
    "name: CI\n"
    "\n"
    "concurrency:\n"
    + _CONDITIONAL_CANCEL
    + "\n"
    "jobs:\n"
)

# Invariant 2 (cancel half) violation: block present,
# `cancel-in-progress:` absent.
_NO_CANCEL_WORKFLOW = (
    "name: CI\n"
    "\n"
    "concurrency:\n"
    + _PER_SHA_GROUP
    + "\n"
    "jobs:\n"
)

# Invariant 3 violation: `group:` keyed on `github.ref` only — the
# pre-ADR-0015 shape that lets a force-push cancel a sibling's run.
_REF_ONLY_GROUP_WORKFLOW = (
    "name: CI\n"
    "\n"
    "concurrency:\n"
    "  group: ${{ github.ref }}\n"
    + _CONDITIONAL_CANCEL
    + "\n"
    "jobs:\n"
)

# Invariant 3 violation: per-SHA `group:` without the `|| github.ref`
# fallback (non-PR events would render an empty key).
_SHA_WITHOUT_REF_FALLBACK_WORKFLOW = (
    "name: CI\n"
    "\n"
    "concurrency:\n"
    "  group: >-\n"
    "    ${{ github.workflow }}-\n"
    "    ${{\n"
    "      github.event_name == 'pull_request' &&\n"
    "      github.event.pull_request.head.sha\n"
    "    }}\n"
    + _CONDITIONAL_CANCEL
    + "\n"
    "jobs:\n"
)

# Invariant 4 violation: unconditional `cancel-in-progress: true` —
# would cancel in-progress branch pushes too.
_UNCONDITIONAL_CANCEL_WORKFLOW = (
    "name: CI\n"
    "\n"
    "concurrency:\n"
    + _PER_SHA_GROUP
    + "  cancel-in-progress: true\n"
    "\n"
    "jobs:\n"
)

# Invariant 4 violation: gated on push but not on main/develop.
_PUSH_ONLY_CANCEL_WORKFLOW = (
    "name: CI\n"
    "\n"
    "concurrency:\n"
    + _PER_SHA_GROUP
    + "  cancel-in-progress: >-\n"
    "    ${{\n"
    "      github.event_name == 'push'\n"
    "    }}\n"
    "\n"
    "jobs:\n"
)

# Invariant 4 violation: gated on main/develop but not on push events.
_CONTAINS_ONLY_CANCEL_WORKFLOW = (
    "name: CI\n"
    "\n"
    "concurrency:\n"
    + _PER_SHA_GROUP
    + "  cancel-in-progress: >-\n"
    "    ${{\n"
    "      contains('refs/heads/main,refs/heads/develop', github.ref)\n"
    "    }}\n"
    "\n"
    "jobs:\n"
)

# The classic legacy shape: both invariant-3 and invariant-4 violations
# in one workflow — must yield exactly two findings.
_LEGACY_WORKFLOW = (
    "name: CI\n"
    "\n"
    "concurrency:\n"
    "  group: ${{ github.ref }}\n"
    "  cancel-in-progress: true\n"
    "\n"
    "jobs:\n"
)

_COMPLIANT_BLOCK = (
    "concurrency:\n" + _PER_SHA_GROUP + _CONDITIONAL_CANCEL
)


def _redirect(checker, monkeypatch, tmp_path: Path) -> Path:
    """Point the checker's module constants at a ``tmp_path`` mock repo
    and return its `.github/workflows` directory."""
    workflows = tmp_path / ".github" / "workflows"
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "WORKFLOWS_DIR", workflows)
    return workflows


def _write_workflow(tmp_path: Path, name: str, text: str) -> Path:
    """Write a synthetic workflow into the mock repo's workflows dir."""
    target = tmp_path / ".github" / "workflows" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _findings_for(checker, monkeypatch, tmp_path, text, name="ci.yml"):
    """`check_workflow` findings for a single synthetic workflow."""
    _redirect(checker, monkeypatch, tmp_path)
    return checker.check_workflow(_write_workflow(tmp_path, name, text))


# ---------------------------------------------------------------------------
# extract_block
# ---------------------------------------------------------------------------


def test_extract_block_returns_the_top_level_block(checker):
    """The compliant template's raw block text is extracted verbatim."""
    block = checker.extract_block(_COMPLIANT_WORKFLOW)
    assert block is not None
    assert block.startswith("concurrency:")
    assert "group: >-" in block
    assert "cancel-in-progress: >-" in block


def test_extract_block_missing_returns_none(checker):
    assert checker.extract_block(_NO_BLOCK_WORKFLOW) is None


def test_extract_block_ignores_nested_concurrency_key(checker):
    """A `concurrency:` key nested inside a job is not top-level drift
    coverage — the extractor anchors on column 0."""
    assert checker.extract_block(_NESTED_ONLY_WORKFLOW) is None


# ---------------------------------------------------------------------------
# block_uses_per_sha_group (invariant 3)
# ---------------------------------------------------------------------------


def test_per_sha_group_accepts_adr_template(checker):
    block = checker.extract_block(_COMPLIANT_WORKFLOW)
    assert block is not None
    assert checker.block_uses_per_sha_group(block) is True


def test_per_sha_group_rejects_ref_only_key(checker):
    """`group: ${{ github.ref }}` — the pre-ADR-0015 shared-ref shape."""
    block = checker.extract_block(_REF_ONLY_GROUP_WORKFLOW)
    assert block is not None
    assert checker.block_uses_per_sha_group(block) is False


def test_per_sha_group_rejects_sha_without_ref_fallback(checker):
    """Per-SHA `group:` without the `|| github.ref` fallback still
    fails the marker test (the `||` combination is load-bearing)."""
    block = checker.extract_block(_SHA_WITHOUT_REF_FALLBACK_WORKFLOW)
    assert block is not None
    assert checker.block_uses_per_sha_group(block) is False


def test_per_sha_group_rejects_missing_group_line(checker):
    block = checker.extract_block(_NO_GROUP_WORKFLOW)
    assert block is not None
    assert checker.block_uses_per_sha_group(block) is False


# ---------------------------------------------------------------------------
# block_uses_conditional_cancel (invariant 4)
# ---------------------------------------------------------------------------


def test_conditional_cancel_accepts_adr_template(checker):
    block = checker.extract_block(_COMPLIANT_WORKFLOW)
    assert block is not None
    assert checker.block_uses_conditional_cancel(block) is True


def test_conditional_cancel_rejects_unconditional_true(checker):
    block = checker.extract_block(_UNCONDITIONAL_CANCEL_WORKFLOW)
    assert block is not None
    assert checker.block_uses_conditional_cancel(block) is False


def test_conditional_cancel_rejects_push_gate_only(checker):
    """`push` gate without the main/develop ref containment."""
    block = checker.extract_block(_PUSH_ONLY_CANCEL_WORKFLOW)
    assert block is not None
    assert checker.block_uses_conditional_cancel(block) is False


def test_conditional_cancel_rejects_contains_gate_only(checker):
    """Ref containment without the `github.event_name == 'push'` gate."""
    block = checker.extract_block(_CONTAINS_ONLY_CANCEL_WORKFLOW)
    assert block is not None
    assert checker.block_uses_conditional_cancel(block) is False


def test_conditional_cancel_rejects_missing_cancel_line(checker):
    block = checker.extract_block(_NO_CANCEL_WORKFLOW)
    assert block is not None
    assert checker.block_uses_conditional_cancel(block) is False


# ---------------------------------------------------------------------------
# check_workflow — per-file findings for each planted violation
# ---------------------------------------------------------------------------


def test_check_workflow_compliant_has_no_findings(
    checker, monkeypatch, tmp_path
):
    assert _findings_for(checker, monkeypatch, tmp_path,
                         _COMPLIANT_WORKFLOW) == []


def test_check_workflow_reports_missing_block_with_rel_path(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _NO_BLOCK_WORKFLOW)
    assert len(findings) == 1
    assert ".github/workflows/ci.yml: missing top-level" in findings[0]


def test_check_workflow_reports_nested_only_as_missing_block(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _NESTED_ONLY_WORKFLOW)
    assert len(findings) == 1
    assert "missing top-level" in findings[0]


def test_check_workflow_reports_missing_group_key(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _NO_GROUP_WORKFLOW)
    assert len(findings) == 1
    assert "`concurrency.group` does not reference" in findings[0]


def test_check_workflow_reports_missing_cancel_key(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _NO_CANCEL_WORKFLOW)
    assert len(findings) == 1
    assert "`concurrency.cancel-in-progress` is not gated" in findings[0]


def test_check_workflow_reports_ref_only_group(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _REF_ONLY_GROUP_WORKFLOW)
    assert len(findings) == 1
    assert "per-`head_sha` shape" in findings[0]


def test_check_workflow_reports_sha_without_ref_fallback(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _SHA_WITHOUT_REF_FALLBACK_WORKFLOW)
    assert len(findings) == 1
    assert "`|| github.ref`" in findings[0]


def test_check_workflow_reports_unconditional_cancel(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _UNCONDITIONAL_CANCEL_WORKFLOW)
    assert len(findings) == 1
    assert "not gated" in findings[0]


def test_check_workflow_reports_push_only_cancel_gate(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _PUSH_ONLY_CANCEL_WORKFLOW)
    assert len(findings) == 1
    assert "refs/heads/main,refs/heads/develop" in findings[0]


def test_check_workflow_reports_contains_only_cancel_gate(
    checker, monkeypatch, tmp_path
):
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _CONTAINS_ONLY_CANCEL_WORKFLOW)
    assert len(findings) == 1
    assert "not gated" in findings[0]


def test_check_workflow_legacy_shape_yields_both_findings(
    checker, monkeypatch, tmp_path
):
    """The pre-ADR-0015 `github.ref` + `true` combo trips both the
    group invariant and the cancel invariant in one file."""
    findings = _findings_for(checker, monkeypatch, tmp_path,
                             _LEGACY_WORKFLOW)
    assert len(findings) == 2
    assert any("`concurrency.group` does not reference" in f
               for f in findings)
    assert any("`concurrency.cancel-in-progress` is not gated" in f
               for f in findings)


# ---------------------------------------------------------------------------
# main() exit-code contract: 0 no drift / 1 drift / 2 script error
# ---------------------------------------------------------------------------


def test_main_exit_0_when_all_workflows_compliant(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(tmp_path, "alpha.yml", _COMPLIANT_WORKFLOW)
    _write_workflow(tmp_path, "beta.yml", _COMPLIANT_WORKFLOW)
    monkeypatch.setattr(
        checker.sys, "argv", ["check_concurrency_keys.py"]
    )

    assert checker.main() == 0
    assert "OK: all 2 workflow(s)" in capsys.readouterr().out


def test_main_exit_1_on_planted_violation(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(tmp_path, "alpha.yml", _COMPLIANT_WORKFLOW)
    _write_workflow(tmp_path, "bad.yml", _NO_BLOCK_WORKFLOW)
    monkeypatch.setattr(
        checker.sys, "argv", ["check_concurrency_keys.py"]
    )

    assert checker.main() == 1
    err = capsys.readouterr().err
    assert "ADR-0015 concurrency drift detected" in err
    assert ".github/workflows/bad.yml: missing top-level" in err
    assert "1 finding(s) across 2 workflow(s)" in err


def test_main_exit_2_when_workflows_dir_missing(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)  # dir never created
    monkeypatch.setattr(
        checker.sys, "argv", ["check_concurrency_keys.py"]
    )

    assert checker.main() == 2
    assert "not found" in capsys.readouterr().err


def test_main_exit_2_on_unknown_workflow_filter(
    checker, monkeypatch, tmp_path, capsys
):
    workflows = _redirect(checker, monkeypatch, tmp_path)
    workflows.mkdir(parents=True)
    _write_workflow(tmp_path, "alpha.yml", _COMPLIANT_WORKFLOW)
    monkeypatch.setattr(
        checker.sys,
        "argv",
        ["check_concurrency_keys.py", "--workflow", "nope.yml"],
    )

    assert checker.main() == 2
    assert "workflow nope.yml not found" in capsys.readouterr().err


def test_main_workflow_filter_restricts_scan_to_named_file(
    checker, monkeypatch, tmp_path, capsys
):
    """`--workflow` scopes the scan: a repo with one compliant and one
    violating workflow passes when only the compliant one is named."""
    _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(tmp_path, "alpha.yml", _COMPLIANT_WORKFLOW)
    _write_workflow(tmp_path, "bad.yml", _NO_BLOCK_WORKFLOW)
    monkeypatch.setattr(
        checker.sys,
        "argv",
        ["check_concurrency_keys.py", "--workflow", "alpha.yml"],
    )

    assert checker.main() == 0
    assert "OK: all 1 workflow(s)" in capsys.readouterr().out
