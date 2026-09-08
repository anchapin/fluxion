"""Tests for ``scripts/check_workflow_pin.py`` -- Issue #3475.

The script enforces the SHA-pinning baseline documented in
``docs/SECURITY.md`` §5 across every ``.github/workflows/*.yml``. Until
the guard lands, a mutable ``@v4`` / ``@stable`` / ``@main`` ref could
merge into any workflow without review catching it; PR #3472 repinned
the last tag-pinned actions but, by scope, did not add the general
hygiene check (issue #3472 body, deferring the gate to #3475).

The tests below drive the matcher's documented invariants against
hermetic ``tmp_path`` fixture workflows so a regex / classification
regression surfaces here instead of on a real PR. Mirrors the
``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_concurrency_keys.py`` (Issue #3444).

Coverage:

* ``classify()`` -- the SHA / local / tag / branch / bare / empty
  taxonomy (parametrized so each kind is a discrete test),
* ``iter_uses_lines()`` -- line extractor with comment-skip semantics,
* ``check_workflow()`` -- per-file findings for each planted violation,
* ``main()`` exit-code contract: 0 clean / 1 drift / 2 script error.

No test depends on the network or on the repo's real workflows; the
end-to-end against the real tree is the direct
``check_workflow_pin.py`` invocation in ``scripts-tests.yml`` (Issue
#3475 acceptance criterion #2).

Note on real-tree state: at the time #3475 was filed, exactly one
non-pinned ``uses:`` remains in the production tree --
``nightly-ashrae-140-gauge.yml:63: dtolnay/rust-toolchain@stable``
(deferred from #3472's scope guard). The pytest suite therefore does
NOT add a ``test_main_returns_zero_on_clean_real_repo`` regression
test, because the strict checker correctly flags that line as drift.
Once the dtolnay line is repinned in a follow-up, the live
``check_workflow_pin.py`` step in scripts-tests.yml will turn green
end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_workflow_pin"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of ``scripts/check_workflow_pin.py``."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, monkeypatch, tmp_path: Path) -> Path:
    """Point the checker's module constants at a ``tmp_path`` mock repo
    and return its ``.github/workflows`` directory."""
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


def _findings_for(checker, monkeypatch, tmp_path, text,
                  name: str = "ci.yml") -> list[str]:
    """``check_workflow()`` findings for a single synthetic workflow."""
    _redirect(checker, monkeypatch, tmp_path)
    return checker.check_workflow(_write_workflow(tmp_path, name, text))


# ---------------------------------------------------------------------------
# classify() -- taxonomy of `uses:` ref shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref,expected",
    [
        # SHA-pinned (40 lowercase / 40 uppercase hex).
        ("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "sha"),
        ("actions/checkout@3D3C42E5AAC5BA805825DA76410C181273BA90B1", "sha"),
        ("nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60", "sha"),
        # Local composite / reusable path.
        ("./.github/actions/setup-rust-env", "local"),
        ("./.github/actions/setup-rust-python-env", "local"),
        ("./.github/workflows/ci-steps.yml", "local"),
        # Tag-pinned (mutable, FAIL).
        ("actions/checkout@v4", "tag"),
        ("actions/checkout@v4.1.7", "tag"),
        # Branch-pinned (mutable, FAIL).
        ("dtolnay/rust-toolchain@stable", "branch"),
        ("actions/checkout@main", "branch"),
        ("actions/checkout@my-feature/foo", "branch"),
        # Bare (no `@`, defaults to repo default branch, FAIL).
        ("actions/checkout", "bare"),
        # Empty ref (malformed).
        ("", "empty"),
        # Short non-SHA hex (NOT a SHA, FAIL).
        ("owner/repo@deadbeef", "branch"),
    ],
)
def test_classify_returns_expected_kind(checker, ref, expected):
    """Issue #3475 acceptance criterion #1: SHA / local pass; tag / branch
    / bare / empty / unknown all FAIL. The exact kind label is the
    remediation hint shown in the FAIL message."""
    assert checker.classify(ref) == expected


# ---------------------------------------------------------------------------
# iter_uses_lines() -- line extractor with comment-skip semantics
# ---------------------------------------------------------------------------


# A fixture workflow exercising the line extractor's documented
# behaviors: list-step `uses:`, bare-key `uses:` (under `with:`),
# `if:` conditionals, commented-out refs, and a tag-pinned drift.
_EXTRACTOR_SAMPLE = (
    "name: CI\n"
    "on:\n"
    "  push:\n"
    "jobs:\n"
    "  build:\n"
    "    steps:\n"
    # line 7: list-step SHA-pinned
    "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n"
    # line 8: list-step local composite
    "      - uses: ./.github/actions/setup-rust-env\n"
    "        with:\n"
    "          key: foo\n"
    # line 11: bare-key SHA-pinned (nested under `with:` in source)
    "        uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9  # v6.1.0\n"
    "      - if: ${{ github.event_name == 'pull_request' }}\n"
    # line 13: bare-key SHA-pinned (after `if:`)
    "        uses: nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60  # v4.0.0\n"
    # line 14: commented-out local composite -- must be SKIPPED
    "#       uses: ./.github/workflows/ci-steps.yml\n"
    # line 15: tag-pinned drift -- must be yielded
    "        uses: actions/checkout@v4\n"
)


def test_iter_uses_lines_extracts_active_uses_lines(checker):
    """The line extractor yields ``(lineno, ref)`` for every active
    ``uses:`` line in source order, skipping commented-out refs and
    non-``uses:`` keys (``if:``, ``with:``, etc.)."""
    expected = [
        (7, "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"),
        (8, "./.github/actions/setup-rust-env"),
        (11, "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"),
        (13, "nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60"),
        (15, "actions/checkout@v4"),
    ]
    assert checker.iter_uses_lines(_EXTRACTOR_SAMPLE) == expected


def test_iter_uses_lines_skips_commented_uses(checker):
    """Issue #3475: YAML allows structural commenting; a commented-out
    ref is not executed and must NOT trigger the gate. The fixture above
    pins this: ``# uses: ./.github/workflows/ci-steps.yml`` is on line 14
    but absent from the yielded list."""
    out = checker.iter_uses_lines(_EXTRACTOR_SAMPLE)
    assert not any(ref.startswith("./.github/workflows/ci-steps.yml")
                   for _, ref in out)


def test_iter_uses_lines_on_empty_input(checker):
    assert checker.iter_uses_lines("") == []


def test_iter_uses_lines_ignores_if_and_with(checker):
    """``- if:`` and ``with:`` lines are not ``uses:`` and must be skipped
    even if they happen to live near a ``uses:`` step."""
    text = (
        "steps:\n"
        "  - if: ${{ github.event_name == 'pull_request' }}\n"
        "    uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n"
        "    with:\n"
        "      fetch-depth: 0\n"
    )
    assert checker.iter_uses_lines(text) == [
        (3, "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"),
    ]


# ---------------------------------------------------------------------------
# check_workflow() -- per-file findings for each planted violation
# ---------------------------------------------------------------------------


_COMPLIANT_TEXT = (
    "name: ok\n"
    "jobs:\n"
    "  build:\n"
    "    steps:\n"
    "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n"
    "      - uses: ./.github/actions/setup-rust-env\n"
    "      - uses: nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60  # v4.0.0\n"
)


def test_check_workflow_compliant_has_no_findings(checker, monkeypatch, tmp_path):
    """A workflow whose every ``uses:`` is SHA-pinned or a local composite
    yields zero findings."""
    assert _findings_for(checker, monkeypatch, tmp_path,
                         _COMPLIANT_TEXT) == []


def test_check_workflow_flags_tag_pinned(checker, monkeypatch, tmp_path):
    """Acceptance criterion #2: tag-pinned refs are a hard FAIL."""
    text = (
        "name: drift\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
    )
    findings = _findings_for(checker, monkeypatch, tmp_path, text)
    assert len(findings) == 1
    assert "actions/checkout@v4" in findings[0]
    assert "tag" in findings[0]
    assert ".github/workflows/ci.yml:5" in findings[0]


def test_check_workflow_flags_branch_pinned(checker, monkeypatch, tmp_path):
    """Acceptance criterion #2: branch-pinned refs are a hard FAIL.
    Mirrors the dtolnay/rust-toolchain@stable case deferred from #3472."""
    text = (
        "name: drift\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: dtolnay/rust-toolchain@stable\n"
    )
    findings = _findings_for(checker, monkeypatch, tmp_path, text)
    assert len(findings) == 1
    assert "dtolnay/rust-toolchain@stable" in findings[0]
    assert "branch" in findings[0]


def test_check_workflow_flags_bare_ref(checker, monkeypatch, tmp_path):
    """Acceptance criterion #2: a bare ref (no ``@``) is a hard FAIL."""
    text = (
        "name: drift\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/setup-node\n"
    )
    findings = _findings_for(checker, monkeypatch, tmp_path, text)
    assert len(findings) == 1
    assert "actions/setup-node" in findings[0]
    assert "bare" in findings[0]


def test_check_workflow_flags_each_independent_drift(
    checker, monkeypatch, tmp_path
):
    """All three violation shapes (tag / branch / bare) planted in one
    workflow produce three findings -- one per line, in source order."""
    text = (
        "name: drift\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n"
        "      - uses: actions/checkout@v4\n"          # line 6: tag
        "      - uses: dtolnay/rust-toolchain@stable\n"  # line 7: branch
        "      - uses: actions/setup-node\n"            # line 8: bare
        "#       uses: ./.github/workflows/ci-steps.yml\n"  # line 9: commented
    )
    findings = _findings_for(checker, monkeypatch, tmp_path, text)
    assert len(findings) == 3
    # Findings are in source order (sorted by line number).
    assert findings[0].startswith(".github/workflows/ci.yml:6")
    assert "tag" in findings[0]
    assert findings[1].startswith(".github/workflows/ci.yml:7")
    assert "branch" in findings[1]
    assert findings[2].startswith(".github/workflows/ci.yml:8")
    assert "bare" in findings[2]
    # The commented-out line is absent from the findings.
    joined = "\n".join(findings)
    assert "ci-steps.yml" not in joined


def test_check_workflow_findings_carry_remediation_hint(
    checker, monkeypatch, tmp_path
):
    """The FAIL message includes actionable remediation text, not just a
    raw classification. Mirrors the manual-recovery path documented in
    docs/SECURITY.md §5."""
    text = (
        "name: drift\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
    )
    findings = _findings_for(checker, monkeypatch, tmp_path, text)
    assert len(findings) == 1
    assert "mutable" in findings[0]
    assert "git ls-remote" in findings[0]


# ---------------------------------------------------------------------------
# main() exit-code contract: 0 clean / 1 drift / 2 script error
# ---------------------------------------------------------------------------


def test_main_exit_0_when_all_workflows_compliant(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(tmp_path, "alpha.yml", _COMPLIANT_TEXT)
    _write_workflow(tmp_path, "beta.yml", _COMPLIANT_TEXT)
    monkeypatch.setattr(checker.sys, "argv", ["check_workflow_pin.py"])

    assert checker.main([]) == 0
    out = capsys.readouterr().out
    assert "OK:" in out
    assert "SHA-pinned or a local composite path" in out
    assert "0 violation(s)" in out


def test_main_exit_1_on_planted_violation(
    checker, monkeypatch, tmp_path, capsys
):
    text = (
        "name: drift\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
    )
    _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(tmp_path, "alpha.yml", _COMPLIANT_TEXT)
    _write_workflow(tmp_path, "bad.yml", text)
    monkeypatch.setattr(checker.sys, "argv", ["check_workflow_pin.py"])

    assert checker.main([]) == 1
    err = capsys.readouterr().err
    assert "Non-SHA-pinned `uses:` drift detected" in err
    assert ".github/workflows/bad.yml:" in err
    assert "actions/checkout@v4" in err
    assert "Remediation:" in err


def test_main_exit_2_when_workflows_dir_missing(
    checker, monkeypatch, tmp_path, capsys
):
    _redirect(checker, monkeypatch, tmp_path)  # dir never created
    monkeypatch.setattr(checker.sys, "argv", ["check_workflow_pin.py"])

    assert checker.main([]) == 2
    assert "not found" in capsys.readouterr().err


def test_main_exit_2_when_no_yml_files_present(
    checker, monkeypatch, tmp_path, capsys
):
    """A workflows dir that exists but has no ``.yml`` files is a script
    error, not a clean pass -- a contributor moving all workflows to
    ``.yaml`` would otherwise silently disable the gate."""
    _redirect(checker, monkeypatch, tmp_path)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "README.md").write_text(
        "x", encoding="utf-8"
    )
    monkeypatch.setattr(checker.sys, "argv", ["check_workflow_pin.py"])

    assert checker.main([]) == 2
    assert "no .yml workflows" in capsys.readouterr().err


def test_main_reports_uses_count_in_summary(
    checker, monkeypatch, tmp_path, capsys
):
    """The summary line includes the number of ``uses:`` lines scanned
    so a future regression that drops refs from the iteration loop is
    visible without reading the FAIL findings."""
    _redirect(checker, monkeypatch, tmp_path)
    _write_workflow(tmp_path, "alpha.yml", _COMPLIANT_TEXT)
    _write_workflow(tmp_path, "beta.yml", _COMPLIANT_TEXT)
    monkeypatch.setattr(checker.sys, "argv", ["check_workflow_pin.py"])

    checker.main([])
    out = capsys.readouterr().out
    # _COMPLIANT_TEXT has 3 `uses:` lines; two files = 6 refs total.
    assert "6 `uses:` ref(s)" in out


# ---------------------------------------------------------------------------
# bundled --self-test
# ---------------------------------------------------------------------------


def test_main_self_test_passes_deterministically(checker, capsys):
    """The bundled ``--self-test`` builds mock fixtures and must exit 0.
    This protects against a regression that breaks the bundled test
    runner even when the standalone pytest suite passes."""
    assert checker.main(["--self-test"]) == 0
    assert "PASS:" in capsys.readouterr().out
