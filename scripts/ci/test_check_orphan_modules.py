"""
Tests for ``scripts/check_orphan_modules.py`` -- Issue #2875 (detector)
and Issue #3074 (harness).

Regression guard for the orphan-module detector. Mirrors the
``load_script`` + ``tmp_path`` mock-repo pattern from
``test_check_stub_modules.py`` (Issue #2896) -- both target dead-code
detection under ``src/**``, both landed in the same 48h window, and
the harness is a one-for-one sibling so a future contributor reading
either file finds the same shape.

The script computes its path constants (``REPO_ROOT`` / ``SRC_DIR`` /
``LIB_RS`` / ``BIN_DIR``) at import time from
``Path(__file__).resolve().parent.parent``. Each test that wants a
synthetic fixture must therefore redirect every constant (via
``_redirect``) before calling ``main()`` so the BFS walks the
synthetic tree instead of the real one.

Issue #3074 acceptance criteria are realised as seven scenarios:

1. **Clean state (real repo)** -- ``scripts/check_orphan_modules.py``
   exits 0 against the production workspace. A regression in the BFS,
   regex, or allowlist that mistreats a real ``src/`` file as a NEW
   orphan flips this red. Run via ``subprocess.run([...])`` per the
   issue brief so the test exercises the script exactly as CI does,
   not the in-process ``main()``.
2. **Detected orphan** -- a synthetic ``src/`` tree with one extra
   ``.rs`` file that no ``mod`` declaration references -> ``FAIL``
   (exit 1) with the injected path in stdout.
3. **cfg-gated module is wired in** -- a ``#[cfg(feature = "x")] mod foo;``
   on a reachable parent must NOT trip the detector; the cfg attribute
   is stripped before matching, so a feature-gated mod is still
   structurally wired in whenever the mod target file exists.
4. **Inline ``mod foo { ... }`` body** -- nested ``mod bar;``
   declarations inside an inline body must still be walked by the
   BFS; the inline namespace's directory is the out-of-line parent's
   directory + the inline name.
5. **``pub use`` is not a mod declaration** -- a
   ``pub use crate::alpha::Beta;`` line inside a reachable ``mod.rs``
   does NOT make ``src/alpha.rs`` reachable; the BFS only follows
   ``mod`` declarations, not path re-exports.
6. **``_walk_reachable`` BFS contract** -- pins the BFS primitive:
   starting file is always in the visited set; every file referenced
   by a ``mod`` declaration is included; an undeclared file is not.
7. **``_all_rs_under_src`` excludes ``src/bin/``** -- each file under
   ``src/bin/`` is its own Cargo target root, not a child of the
   library module graph, and must not appear in the orphan universe.

Issue #3748 adds the wired-but-dead disposition registry gate and its
scenarios (8-13): lock-step pass, missing-row / stale-row /
invalid-row / missing-registry failures, the mock-repo skip guard,
and a real-repo coverage pin asserting every ``WIRED_BUT_DEAD``
entry carries an owner-assigned wire-or-delete disposition.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

SCRIPT_NAME = "check_orphan_modules"

# ``REPO_ROOT`` here is used only by the real-repo subprocess test
# (Test 1) -- it points at the worktree that pytest itself was
# launched from. Per-test ``tmp_path`` redirects via the ``_redirect``
# helper never touch this constant.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / f"{SCRIPT_NAME}.py"

# Checked-in disposition registry (Issue #3748) backing the
# wired-but-dead gate under test.
DISPOSITION_REGISTRY = (
    REPO_ROOT / "tests" / "reference_data" / "wired_but_dead_dispositions.json"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the orphan-modules detector."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path: Path, monkeypatch) -> Path:
    """Point the script's path constants at a synthetic ``tmp_path/src/``
    tree and return the resolved ``src/`` path.

    All four constants must be overridden because the script computes
    them once at import time from ``Path(__file__).resolve().parent.parent``:

    * ``REPO_ROOT`` -- used in the print banner and as the
      ``relative_to`` base for orphan paths
    * ``SRC_DIR`` -- used by ``_all_rs_under_src``
    * ``LIB_RS`` -- used by ``main()`` as the BFS starting point
    * ``BIN_DIR`` -- used by ``_all_rs_under_src`` to skip standalone
      binary targets

    ``KNOWN_ORPHANS`` is a ``frozenset`` of paths relative to the real
    ``REPO_ROOT``; the synthetic tree has none of the legacy orphans,
    so no override is required. The allowlist is effectively empty
    for synthetic trees, which is the correct behaviour.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "SRC_DIR", src_dir)
    monkeypatch.setattr(checker, "LIB_RS", src_dir / "lib.rs")
    monkeypatch.setattr(checker, "BIN_DIR", src_dir / "bin")
    return src_dir


def _scrub_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset ``sys.argv`` so the script's argparse doesn't see pytest's
    CLI. The orphan-modules detector's ``main()`` does not actually
    read ``sys.argv`` (it takes no args), but the scrub keeps the
    pattern consistent with the sibling harnesses (e.g.
    ``test_check_stub_modules._scrub_argv``).
    """
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME])


def _write(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(text), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test 1: clean real repo (the regression-locking smoke test)
# ---------------------------------------------------------------------------


def test_script_exits_zero_on_real_repo():
    """Clean-tree smoke test against the real repo.

    Runs ``scripts/check_orphan_modules.py`` against the production
    workspace (no monkey-patching) and asserts it exits 0. A regression
    in the BFS, regex, or allowlist that mistreats a real ``src/``
    file as a NEW orphan flips this red.

    Per the issue brief, the test is driven via ``subprocess.run`` so
    it exercises the script exactly as ``architecture_drift.yml`` does
    when #3073 wires it in -- not the in-process ``main()``.
    """
    result = subprocess.run(
        ["python3", str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"expected exit 0 (no NEW orphans), got rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # Output banner phrases from the script's main(). These pin the
    # public output shape against accidental renames in the
    # architecture_drift wiring step (#3073).
    assert "Orphan-modules detector" in result.stdout
    assert "NEW orphans (regression): 0" in result.stdout
    # The known-orphan baseline line is present (the script has 30
    # baseline entries as of the #2875 allowlist commit).
    assert "No new orphan modules" in result.stdout
    # Issue #3748: the disposition gate section runs on the real repo
    # and reports lock-step.
    assert "Wired-but-dead dispositions (#3748)" in result.stdout
    assert "Disposition registry in lock-step" in result.stdout


# ---------------------------------------------------------------------------
# Test 2: injected orphan -> non-zero exit
# ---------------------------------------------------------------------------


def test_main_returns_one_when_orphan_introduced(
    checker, tmp_path, monkeypatch, capsys
):
    """An unreachable ``.rs`` file is reported as a NEW orphan -> exit 1.

    Builds a minimal ``src/lib.rs`` that declares ``mod alive;`` and
    a sibling ``src/orphan.rs`` that no ``mod`` declaration
    references. The BFS visits ``alive.rs`` only; ``orphan.rs`` lands
    in the diff and the script exits 1.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "lib.rs", "mod alive;\n")
    _write(src_dir / "alive.rs", "pub fn live() {}\n")
    _write(src_dir / "orphan.rs", "pub fn dead() {}\n")
    _scrub_argv(monkeypatch)

    rc = checker.main()
    out = capsys.readouterr().out

    assert rc == 1, f"expected FAIL (orphan introduced), got rc={rc}\noutput:\n{out}"
    assert "NEW ORPHAN MODULES DETECTED" in out
    assert "orphan.rs" in out
    # The reachable file must NOT appear in the orphan list.
    assert "alive.rs" not in out
    # The summary counter surfaces the count of NEW orphans.
    assert "NEW orphans (regression): 1" in out


# ---------------------------------------------------------------------------
# Test 3: cfg-gated module is still reachable
# ---------------------------------------------------------------------------


def test_cfg_gated_mod_is_still_reachable(
    checker, tmp_path, monkeypatch, capsys
):
    """A ``#[cfg(feature = \"x\")] mod foo;`` on a reachable parent
    must still wire the file in.

    The regex strips ``#[cfg(...)]`` / ``#[cfg_attr(...)]`` attributes
    before matching, so a feature-gated mod declaration is
    *structurally* wired in (the file is referenced by the module
    graph) even when the feature flag is not currently active. This
    is by design: the script verifies that the module *can* be wired
    in, not that every feature is currently active. A regression that
    over-strictly stripped the mod when the cfg attribute is present
    would silently turn every feature-gated module into an "orphan"
    on a build without that feature.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(
        src_dir / "lib.rs",
        "#[cfg(feature = \"nightly\")]\nmod foo;\nmod alive;\n",
    )
    _write(src_dir / "foo.rs", "pub fn nightly() {}\n")
    _write(src_dir / "alive.rs", "pub fn live() {}\n")
    _scrub_argv(monkeypatch)

    rc = checker.main()
    out = capsys.readouterr().out

    assert rc == 0, (
        f"expected PASS (cfg-gated mod is wired in), got rc={rc}\n"
        f"output:\n{out}"
    )
    # Neither the cfg-gated file nor the plain file should be reported.
    assert "foo.rs" not in out
    assert "alive.rs" not in out


# ---------------------------------------------------------------------------
# Test 4: inline `mod foo { ... }` body still reaches nested children
# ---------------------------------------------------------------------------


def test_inline_mod_body_walks_nested_children(
    checker, tmp_path, monkeypatch, capsys
):
    """A nested ``mod baz;`` inside an inline ``mod bar { ... }`` body
    inside an out-of-line ``mod foo;`` parent must still be walked
    by the BFS.

    The script's BFS descends into inline bodies with the same
    ``_MOD_RE`` regex; the inline namespace's directory is the
    file-module's own child directory + the inline name, so
    ``mod baz;`` inside ``src/foo.rs:mod bar { ... }`` resolves to
    ``src/foo/bar/baz.rs`` under the child-dir semantics this walker
    implements (a file module ``dir/foo.rs`` resolves nested mods
    against ``dir/foo/`` — the surrogate.rs + surrogate/ decomposition
    in this branch depends on that). An unrelated orphan is also
    planted to confirm the gate still fires on the rest of the tree.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "lib.rs", "mod foo;\n")
    # Inline body: `mod bar { mod baz; }` -> baz resolves under
    # `child_dir/foo/bar/baz.rs` (see docstring above).
    _write(
        src_dir / "foo.rs",
        "pub mod bar { mod baz; }\n",
    )
    _write(src_dir / "foo" / "bar" / "baz.rs", "pub fn baz() {}\n")
    # Unrelated orphan to confirm the gate still fires.
    _write(src_dir / "orphan.rs", "pub fn orphan() {}\n")
    _scrub_argv(monkeypatch)

    rc = checker.main()
    out = capsys.readouterr().out

    assert rc == 1, (
        f"expected FAIL (orphan.rs introduced), got rc={rc}\n"
        f"output:\n{out}"
    )
    # orphan.rs is reported.
    assert "orphan.rs" in out
    # The inline-body child is NOT reported (it was reached by descent).
    assert "baz.rs" not in out
    # The out-of-line parent `foo.rs` is also not reported.
    assert "foo.rs" not in out


# ---------------------------------------------------------------------------
# Test 4b: out-of-line mod AFTER inline bodies is not dropped
# (offset-desynchronisation regression)
# ---------------------------------------------------------------------------


def test_out_of_line_mod_after_inline_bodies_is_reached(
    checker, tmp_path, monkeypatch, capsys
):
    """A trailing out-of-line ``mod bar;`` in a file that ALSO has inline
    ``mod tests { ... }`` bodies must still be resolved as wired.

    Regression: ``_collect_declared_mods`` strips the inline bodies from the
    scanned text but indexed the terminator character in the ORIGINAL text,
    so every offset after the first stripped body was shifted and the
    out-of-line declaration was silently dropped from `declared` — making
    the declared child file look like a phantom orphan. Exactly the layout
    ``PR #3688``'s extracted ``coverage_tests`` child modules hit: inline
    ``mod tests { ... }`` / ``mod expm_debug_tests { ... }`` bodies followed
    by ``#[cfg(test)] mod coverage_tests;`` at EOF.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "lib.rs", "mod parent;\n")
    _write(
        src_dir / "parent" / "mod.rs",
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    #[test]\n"
        "    fn t() {}\n"
        "}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod more_tests {\n"
        "    #[test]\n"
        "    fn t2() {}\n"
        "}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod coverage_tests;\n",
    )
    _write(
        src_dir / "parent" / "coverage_tests.rs", "#[test]\nfn c() {}\n"
    )
    _scrub_argv(monkeypatch)

    rc = checker.main()
    out = capsys.readouterr().out

    assert rc == 0, (
        f"expected PASS (coverage_tests.rs is wired via the trailing "
        f"out-of-line mod), got rc={rc}\noutput:\n{out}"
    )
    assert "coverage_tests.rs" not in out


# ---------------------------------------------------------------------------
# Test 5: `pub use` in mod.rs does NOT count as a mod declaration
# ---------------------------------------------------------------------------


def test_pub_use_in_mod_rs_does_not_create_phantom_reachability(
    checker, tmp_path, monkeypatch, capsys
):
    """A ``pub use crate::alpha::Beta;`` line inside a reachable
    ``mod.rs`` does NOT make ``src/alpha.rs`` reachable.

    The BFS only follows ``mod foo;`` declarations. A ``pub use`` is
    a re-export statement (a path reference) and must not be
    confused with a mod declaration -- otherwise the detector would
    silently allow the underlying file to drift as an undetected
    orphan. The regex's required ``mod`` keyword is the safeguard
    against this; the test pins that contract.

    The reachable file (a real ``mod bar;`` declaration) MUST stay
    reachable; the unrelated ``src/alpha.rs`` (referenced only via
    ``pub use``) MUST be flagged as an orphan.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "lib.rs", "mod foo;\n")
    # The mod.rs declares `mod bar;` (real mod) AND a `pub use`
    # re-export whose path happens to point at a file that is NOT
    # declared via `mod alpha;` from anywhere in the tree.
    _write(
        src_dir / "foo" / "mod.rs",
        (
            "mod bar;\n"
            "pub use crate::alpha::Beta;\n"
        ),
    )
    _write(src_dir / "foo" / "bar.rs", "pub fn bar() {}\n")
    # The pub-use target. Not declared via `mod alpha;` from any
    # reachable parent, so it IS an orphan.
    _write(src_dir / "alpha.rs", "pub struct Beta;\n")
    _scrub_argv(monkeypatch)

    rc = checker.main()
    out = capsys.readouterr().out

    assert rc == 1, (
        f"expected FAIL (pub-use target is orphan), got rc={rc}\n"
        f"output:\n{out}"
    )
    # alpha.rs is reported as an orphan.
    assert "alpha.rs" in out
    # bar.rs (real mod declaration) must NOT be reported.
    assert "bar.rs" not in out


# ---------------------------------------------------------------------------
# Test 6: BFS primitive contract
# ---------------------------------------------------------------------------


def test_walk_reachable_starts_from_lib_rs(checker, tmp_path, monkeypatch):
    """``_walk_reachable`` accepts a starting ``.rs`` file and returns
    the set of every ``.rs`` file transitively reachable from it.

    Pins the BFS contract:
    * the starting file is always in the visited set,
    * every file referenced by a ``mod`` declaration is included,
    * an undeclared file is NOT included.

    Uses the 2015-edition ``mod.rs`` style (``mod X;`` in
    ``src/X/mod.rs`` resolves to ``src/X/Y.rs`` or ``src/X/Y/mod.rs``)
    because that is the path resolution the script's
    ``_candidate_paths_for_mod`` implements: the candidate paths are
    ``parent_dir / name.rs`` and ``parent_dir / name/mod.rs``, where
    ``parent_dir`` is the directory of the file containing the
    declaration. The fluxion codebase uses the 2015 ``mod.rs`` style
    for nested module trees, so this test exercises the same
    resolution path as the real repo.
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "lib.rs", "mod a;\nmod b;\n")
    _write(src_dir / "a" / "mod.rs", "pub mod c;\n")
    _write(src_dir / "b" / "mod.rs", "pub mod d;\n")
    _write(src_dir / "a" / "c.rs", "pub fn a_c() {}\n")
    _write(src_dir / "b" / "d.rs", "pub fn b_d() {}\n")
    _write(src_dir / "ghost.rs", "pub fn ghost() {}\n")

    reachable = checker._walk_reachable(src_dir / "lib.rs")

    assert src_dir / "lib.rs" in reachable
    assert src_dir / "a" / "mod.rs" in reachable
    assert src_dir / "b" / "mod.rs" in reachable
    assert src_dir / "a" / "c.rs" in reachable
    assert src_dir / "b" / "d.rs" in reachable
    # ghost.rs has no `mod` declaration referencing it.
    assert (src_dir / "ghost.rs") not in reachable


# ---------------------------------------------------------------------------
# Test 7: `_all_rs_under_src` excludes `src/bin/`
# ---------------------------------------------------------------------------


def test_all_rs_under_src_excludes_bin(checker, tmp_path, monkeypatch):
    """``_all_rs_under_src`` excludes ``src/bin/*.rs`` -- each file
    there is its own Cargo target root, not a child of the library
    module graph, and must not appear in the orphan universe.

    Pins the contract from the script's docstring (line 32-35):
    "Files under ``src/bin/`` are excluded from the universe: each
    ``.rs`` there is its own crate root / target (Cargo auto-
    discovers binaries), not a child of the ``fluxion`` library
    module tree."
    """
    src_dir = _redirect(checker, tmp_path, monkeypatch)
    _write(src_dir / "lib.rs", "mod alive;\n")
    _write(src_dir / "alive.rs", "pub fn live() {}\n")
    # Standalone binary target.
    _write(src_dir / "bin" / "mybin.rs", "fn main() {}\n")
    # Orphan under the lib tree (NOT in bin/).
    _write(src_dir / "ghost.rs", "pub fn ghost() {}\n")

    all_rs = checker._all_rs_under_src()
    assert (src_dir / "lib.rs") in all_rs
    assert (src_dir / "alive.rs") in all_rs
    assert (src_dir / "ghost.rs") in all_rs
    # bin/ is excluded from the lib-module orphan universe.
    assert (src_dir / "bin" / "mybin.rs") not in all_rs


# ---------------------------------------------------------------------------
# Issue #3748: wired-but-dead disposition registry gate
# ---------------------------------------------------------------------------


def _real_registry_rows() -> list[dict]:
    """The checked-in registry's rows, verbatim from the real repo."""
    payload = json.loads(DISPOSITION_REGISTRY.read_text(encoding="utf-8"))
    return list(payload["modules"])


def _write_registry(tmp_path: Path, rows: list[dict]) -> Path:
    """Serialise ``rows`` into a synthetic registry at ``tmp_path``."""
    path = tmp_path / "wired_but_dead_dispositions.json"
    payload = {
        "schema_version": 1,
        "description": "synthetic registry for pytest",
        "governing_issue": 3748,
        "module_count": len(rows),
        "modules": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_disposition_gate_passes_in_lockstep(
    checker, tmp_path, monkeypatch, capsys
):
    """A registry covering exactly the live wired-but-dead set passes.

    Mirrors the green state of the real repo: every live module has a
    row, every allowlist entry has a row, no stale rows, and each row
    carries a valid disposition / issue / owner / rationale.
    """
    names = ["ems", "topsis"]
    rows = [r for r in _real_registry_rows() if r["module"] in set(names)]
    assert len(rows) == 2, "seed rows for ems/topsis must exist"
    monkeypatch.setattr(checker, "WIRED_BUT_DEAD", frozenset(names))
    monkeypatch.setattr(
        checker, "WIRED_BUT_DEAD_DISPOSITIONS_PATH", _write_registry(tmp_path, rows)
    )
    _scrub_argv(monkeypatch)

    rc = checker._check_wired_but_dead_dispositions(names)
    out = capsys.readouterr().out

    assert rc == 0, f"expected PASS (lock-step), got rc={rc}\noutput:\n{out}"
    assert "Disposition registry in lock-step (2 row(s)" in out


def test_disposition_gate_fails_when_row_missing(
    checker, tmp_path, monkeypatch, capsys
):
    """A live wired-but-dead module without a registry row fails (exit 1).

    This is the #3748 forcing function: an allowlist entry (or live
    detection) without an owner-assigned wire-or-delete disposition is
    exactly the drift the ratchet alone could not stop.
    """
    rows = [r for r in _real_registry_rows() if r["module"] == "ems"]
    monkeypatch.setattr(checker, "WIRED_BUT_DEAD", frozenset({"ems", "topsis"}))
    monkeypatch.setattr(
        checker, "WIRED_BUT_DEAD_DISPOSITIONS_PATH", _write_registry(tmp_path, rows)
    )
    _scrub_argv(monkeypatch)

    rc = checker._check_wired_but_dead_dispositions(["ems", "topsis"])
    out = capsys.readouterr().out

    assert rc == 1, f"expected FAIL (missing row), got rc={rc}\noutput:\n{out}"
    assert "MISSING DISPOSITION ROWS DETECTED" in out
    assert "topsis" in out


def test_disposition_gate_fails_on_stale_row(
    checker, tmp_path, monkeypatch, capsys
):
    """A registry row whose module is no longer wired-but-dead fails.

    Lock-step convention (same as the dead-code inventory): a cleanup
    PR that wires up or deletes a module must drop its row in the same
    PR, alongside the allowlist entry and the BASELINE_WIRED_BUT_DEAD
    decrement.
    """
    rows = _real_registry_rows() + [
        {
            "module": "ghost_mod",
            "disposition": "reject",
            "issue": 3748,
            "owner": "anchapin",
            "rationale": "stale synthetic row",
        }
    ]
    monkeypatch.setattr(checker, "WIRED_BUT_DEAD", frozenset({"ems", "topsis"}))
    monkeypatch.setattr(
        checker, "WIRED_BUT_DEAD_DISPOSITIONS_PATH", _write_registry(tmp_path, rows)
    )
    _scrub_argv(monkeypatch)

    rc = checker._check_wired_but_dead_dispositions(["ems", "topsis"])
    out = capsys.readouterr().out

    assert rc == 1, f"expected FAIL (stale row), got rc={rc}\noutput:\n{out}"
    assert "STALE DISPOSITION ROWS DETECTED" in out
    assert "ghost_mod" in out


def test_disposition_gate_fails_on_invalid_row(
    checker, tmp_path, monkeypatch, capsys
):
    """Schema violations fail: unknown disposition enum, non-integer
    issue reference, and a blank rationale are each rejected.
    """
    rows = [
        {
            "module": "ems",
            "disposition": "delete-later",
            "issue": "TBD",
            "owner": "anchapin",
            "rationale": "",
        }
    ]
    monkeypatch.setattr(checker, "WIRED_BUT_DEAD", frozenset({"ems"}))
    monkeypatch.setattr(
        checker, "WIRED_BUT_DEAD_DISPOSITIONS_PATH", _write_registry(tmp_path, rows)
    )
    _scrub_argv(monkeypatch)

    rc = checker._check_wired_but_dead_dispositions(["ems"])
    out = capsys.readouterr().out

    assert rc == 1, f"expected FAIL (invalid row), got rc={rc}\noutput:\n{out}"
    assert "INVALID DISPOSITION ROWS DETECTED" in out
    assert "unknown disposition 'delete-later'" in out
    assert "missing/invalid 'issue' reference" in out
    assert "missing 'rationale'" in out


def test_disposition_gate_fails_when_registry_missing(
    checker, tmp_path, monkeypatch, capsys
):
    """A missing registry file is a hard error (exit 2), not a pass."""
    monkeypatch.setattr(checker, "WIRED_BUT_DEAD", frozenset({"ems"}))
    monkeypatch.setattr(
        checker,
        "WIRED_BUT_DEAD_DISPOSITIONS_PATH",
        tmp_path / "does_not_exist.json",
    )
    _scrub_argv(monkeypatch)

    rc = checker._check_wired_but_dead_dispositions(["ems"])
    out = capsys.readouterr().out

    assert rc == 2, f"expected ERROR (registry missing), got rc={rc}\noutput:\n{out}"
    assert "disposition registry missing" in out


def test_disposition_gate_skipped_on_mock_repo(checker, tmp_path, monkeypatch, capsys):
    """The registry-relative gate skips synthetic mock trees (exit 0).

    Same trade-off the #3752 dead-code gate accepted: the gate is
    inherently registry-relative to the real repo, so redirecting
    ``REPO_ROOT`` at a ``tmp_path`` fixture must short-circuit instead
    of false-failing every synthetic-tree test in this harness.
    """
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    _scrub_argv(monkeypatch)

    rc = checker._check_wired_but_dead_dispositions(["ems"])
    out = capsys.readouterr().out

    assert rc == 0, f"expected skip, got rc={rc}\noutput:\n{out}"
    assert "Skipped: REPO_ROOT redirected to a synthetic mock tree" in out


def test_real_registry_covers_real_allowlist(checker):
    """Real-repo coverage pin: the checked-in registry and the
    ``WIRED_BUT_DEAD`` allowlist are in exact lock-step, and every row
    is schema-valid (disposition enum, integer issue, owner, rationale).

    Catches allowlist edits that forget the registry row (and vice
    versa) without paying for a full ripgrep sweep.
    """
    rows = _real_registry_rows()
    by_module = {str(r["module"]): r for r in rows}
    assert set(checker.WIRED_BUT_DEAD) == set(by_module), (
        "WIRED_BUT_DEAD allowlist and disposition registry must match "
        "exactly (Issue #3748 lock-step seed)"
    )
    for name, row in by_module.items():
        assert row["disposition"] in checker.ALLOWED_DISPOSITIONS, name
        assert isinstance(row["issue"], int), name
        assert str(row["owner"]).strip(), name
        assert str(row["rationale"]).strip(), name


def test_disposition_gate_passes_on_real_allowlist(checker, capsys):
    """The gate returns 0 when fed the real repo's live wired-but-dead
    set (the allowlist itself) against the checked-in registry.

    This is the in-process positive path: identical inputs to the
    subprocess smoke test without re-paying the ripgrep sweep.
    """
    rc = checker._check_wired_but_dead_dispositions(sorted(checker.WIRED_BUT_DEAD))
    out = capsys.readouterr().out

    assert rc == 0, f"expected PASS on real repo, got rc={rc}\noutput:\n{out}"
    assert "Rows missing for live modules: 0" in out
    assert "Stale rows (module no longer wired-but-dead): 0" in out
    assert "Disposition registry in lock-step (22 row(s)" in out
