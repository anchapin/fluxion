"""Prose test-count citation guard — Issue #4754.

Issue #4754: the *prose* test-count citations in ``AGENTS.md`` and
``README.md`` had drifted away from the canonical inventory
(``tests/test_inventory.json::totals``):

* ``AGENTS.md`` cited "8,865 cargo-listed tests / 143 ignored" while the
  inventory said 9,266 / 156.
* ``AGENTS.md`` said bare ``cargo test`` skips "4,158 sibling-crate
  tests" while the same file's header said 4,825.
* ``README.md`` said "~4,400 sibling-crate tests".

The drift gate (``scripts/check_test_inventory_drift.py``) only
checked the AGENTS.md gate-inventory *table rows* plus the
workspace-scope paragraph, so none of these prose sites was covered.

This module asserts, against the real repository tree:

1. every citation site exists (presence),
2. every cited figure equals the canonical inventory figure
   (correctness),
3. the sibling-crate figure uses ONE definition everywhere:
   ``workspace_tests - lib_tests_root`` per
   ``tests/test_inventory.json::totals``,
4. the drift gate's own prose-citation check agrees with the
   canonical inventory (the gate is the CI enforcement point).

The numbers are re-derived from ``tests/test_inventory.json`` at test
time — nothing here hardcodes 9,266 / 156 / 4,825 — so the guard
keeps working when the inventory legitimately moves (the gate then
demands that the prose is refreshed in the same PR).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = REPO_ROOT / "AGENTS.md"
README_MD = REPO_ROOT / "README.md"
INVENTORY = REPO_ROOT / "tests" / "test_inventory.json"
DRIFT_GATE = REPO_ROOT / "scripts" / "check_test_inventory_drift.py"


@pytest.fixture(scope="module")
def totals() -> dict:
    """Canonical inventory totals."""
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return payload["totals"]


def _fmt(value: int) -> str:
    """Render an int the way the docs render it (thousands separator)."""
    return f"{value:,}"


def test_inventory_has_the_expected_totals_shape(totals):
    """Guard the fixture: the fields this module relies on must exist."""
    for key in (
        "lib_tests_root",
        "lib_ignored_root",
        "workspace_tests",
        "workspace_ignored",
        "test_binaries",
    ):
        assert key in totals, f"inventory totals missing {key!r}"
        assert isinstance(totals[key], int)


def test_agents_md_workspace_command_cites_inventory_counts(totals):
    """AGENTS.md's workspace-command comment cites the live counts."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    match = re.search(
        r"full workspace \(([\d,]+) cargo-listed tests / ([\d,]+) ignored\)",
        text,
    )
    assert match is not None, (
        "AGENTS.md no longer carries a 'full workspace (N cargo-listed tests / "
        "M ignored)' citation — re-add it so the drift gate can check it"
    )
    cited_tests = int(match.group(1).replace(",", ""))
    cited_ignored = int(match.group(2).replace(",", ""))
    assert cited_tests == totals["workspace_tests"], (
        f"AGENTS.md cites {cited_tests} cargo-listed tests, inventory says "
        f"{totals['workspace_tests']}"
    )
    assert cited_ignored == totals["workspace_ignored"], (
        f"AGENTS.md cites {cited_ignored} ignored tests, inventory says "
        f"{totals['workspace_ignored']}"
    )


def test_agents_md_bare_cargo_test_cites_sibling_crate_count(totals):
    """AGENTS.md's bare-`cargo test` line cites the sibling-crate count."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    match = re.search(
        r"bare `cargo test` skip(?:s)? the ([\d,]+) sibling-crate tests",
        text,
    )
    assert match is not None, (
        "AGENTS.md no longer cites the sibling-crate test count on the bare "
        "`cargo test` line"
    )
    cited = int(match.group(1).replace(",", ""))
    expected = totals["workspace_tests"] - totals["lib_tests_root"]
    assert cited == expected, (
        f"AGENTS.md cites {cited} sibling-crate tests, inventory-derived "
        f"count is {expected}"
    )


def test_readme_cites_sibling_crate_count(totals):
    """README.md's quick-start comment cites the sibling-crate count."""
    text = README_MD.read_text(encoding="utf-8")
    match = re.search(r"skips (~?[\d,]+) sibling-crate tests", text)
    assert match is not None, (
        "README.md no longer cites the sibling-crate test count in the "
        "quick-start comment"
    )
    cited = int(match.group(1).replace("~", "").replace(",", ""))
    expected = totals["workspace_tests"] - totals["lib_tests_root"]
    assert cited == expected, (
        f"README.md cites {cited} sibling-crate tests, inventory-derived "
        f"count is {expected}"
    )


def test_sibling_crate_figure_is_consistent_across_citation_sites(totals):
    """One definition, one number: every sibling-crate citation agrees."""
    expected = totals["workspace_tests"] - totals["lib_tests_root"]
    agents_text = AGENTS_MD.read_text(encoding="utf-8")
    readme_text = README_MD.read_text(encoding="utf-8")

    cited = set()
    for match in re.finditer(
        r"bare `cargo test` skip(?:s)? the ([\d,]+) sibling-crate tests",
        agents_text,
    ):
        cited.add(int(match.group(1).replace(",", "")))
    for match in re.finditer(
        r"skips (~?[\d,]+) sibling-crate tests", readme_text
    ):
        cited.add(int(match.group(1).replace("~", "").replace(",", "")))

    assert cited, "no sibling-crate citation sites found"
    assert cited == {expected}, (
        f"sibling-crate citations disagree: found {sorted(cited)}, "
        f"expected only [{expected}]"
    )


def test_no_stale_prose_figures_remain():
    """The stale Issue #4754 figures must not reappear in the prose."""
    stale = (
        "8,865 cargo-listed tests",
        "143 ignored",
        "4,158 sibling-crate",
        "~4,400 sibling-crate",
        "4,400 sibling-crate",
    )
    for path in (AGENTS_MD, README_MD):
        text = path.read_text(encoding="utf-8")
        for figure in stale:
            assert figure not in text, (
                f"{path.name} still cites the stale figure {figure!r}"
            )


def test_drift_gate_prose_check_passes_on_the_real_tree():
    """The gate's prose-citation check must agree with the inventory."""
    result = subprocess.run(
        [sys.executable, str(DRIFT_GATE), "--no-verify"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"drift gate failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "do not match tests/test_inventory.json" not in result.stderr
