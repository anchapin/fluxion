#!/usr/bin/env python3
"""
Test-inventory generator — Issue #3442.

Derives the workspace's test inventory (binary count, lib-test count,
ignored count, doc-test count) from the on-disk source tree and
(optionally) cross-references it against ``cargo test --workspace
--exclude fluxion-tauri -- --list`` output.

The derived inventory is written to ``tests/test_inventory.json`` so
the drift gate (``scripts/check_test_inventory_drift.py``) and the
``AGENTS.md`` test-count citation can compare live numbers against a
frozen baseline. Mirrors the ``generate_quarantine_registry.py``
pattern (attribute parsing + workspace walk + JSON output) so future
contributors see one consistent shape across the testing gates.

Two modes:

  * ``--default`` (default): pure AST-regex scan over the
    **git-tracked** ``**/*.rs`` files (fast, no compilation). Issue
    #4069: enumeration is strictly ``git ls-files '*.rs'`` (sorted),
    so untracked scratch files (e.g. agent ``.planning/worktrees/``
    copies) cannot shift the counts between checkouts of the same
    commit. Falls back to the on-disk walk when git is unavailable.
    Counts include unconditional ``#[test]`` and ``#[ignore]`` attributes and ignore ``cfg_attr``-
    gated variants (consistent with the quarantine registry's ratchet
    pattern: a ``cfg_attr(..., ignore, ...)`` test that lacks an
    unconditional fallback is still reported, but tag the conditional
    flag in the inventory so the drift gate can prefer the lower of
    unconditional / cfg-gated counts).

  * ``--verify``: invokes ``cargo test --workspace --exclude
    fluxion-tauri -- --list`` (and ``-- --list --ignored``) AFTER the
    AST-regex pass and prefers those values when available. The
    ``cargo`` invocation requires the workspace to compile; if it
    fails, the gate falls back to the regex numbers.

The inventory schema::

    {
      "schema_version": 1,
      "generated_at": "2026-09-08T...Z",
      "repo_root": "<commit_sha-or-'unknown'>",
      "scope": {
        "workspace_exclude": ["fluxion-tauri"]
      },
      "totals": {
        "test_binaries": <int>,           # Cargo auto-discovered <pkg>/tests/*.rs
        "test_source_files": <int>,       # Any **/*.rs under a tests/ dir
        "lib_tests": <int>,               # root crate --lib
        "lib_ignored": <int>,             # root crate --lib + ignored
        "workspace_tests": <int>,         # all tests in --workspace --exclude fluxion-tauri
        "workspace_ignored": <int>,
        "doc_tests": <int>                # doctests across workspace
      },
      "by_crate": {
        "<pkg-name>": {
          "test_binaries": <int>,
          "lib_tests": <int>,
          "lib_ignored": <int>
        },
        ...
      },
      "verify": {                         # present only with --verify
        "command": ["cargo", "test", ...],
        "executed_at": "...",
        "matched": true|false,
        "lib_tests": <int>,
        "lib_ignored": <int>,
        "workspace_tests": <int>,
        "workspace_ignored": <int>,
        "doc_tests": <int>
      }
    }

Binary-count shapes counted as ``test_binaries`` (Issue #3764 update):

* auto-discovered top-level files (``<pkg>/tests/*.rs``) — since the
  consolidation these are only the process-isolation keepers (env-var
  mutation, ``#[global_allocator]``, ``std::process::exit``) plus sibling
  crates' files;
* hand-wired ``[[test]] path = "<pkg>/tests/<sub>/<foo>.rs"`` entries —
  which include the consolidated root runner
  (``tests/all_tests/main.rs``, name ``all_tests``) whose 263 former
  standalone binaries live as modules under ``tests/all_tests/``.

Module files inside a consolidated runner (``tests/all_tests/*.rs``
other than ``main.rs``) are counted under ``test_source_files`` only —
they are not Cargo binaries.

Exit codes
----------
  0 — inventory generated successfully
  1 — workspace walk failed (e.g. ``Cargo.toml`` missing)
  2 — ``--verify`` invocation failed AND the gate cannot recover

Usage
-----
  python3 scripts/generate_test_inventory.py                            # AST-only
  python3 scripts/generate_test_inventory.py --verify                    # cross-check with cargo
  python3 scripts/generate_test_inventory.py --output path/to/file.json   # custom path
  python3 scripts/generate_test_inventory.py --json                      # print to stdout (CI)
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "test_inventory.json"

SCHEMA_VERSION = 1

# Workspace packages that contribute to the inventory. ``fluxion-tauri``
# is excluded (Issue #3126: its frontend dist/ build needs `npm run
# build` before its proc-macro compile, so it is independently excluded
# from ``cargo test --workspace`` per AGENTS.md §"Commands That Are
# Easy to Guess Wrong").
DEFAULT_WORKSPACE_EXCLUDES = ["fluxion-tauri"]

# ---------------------------------------------------------------------------
# Attribute regexes — mirror ``generate_quarantine_registry.py`` so the
# inventory + the quarantine ratchet share a single attribute grammar.
# ---------------------------------------------------------------------------

# `#[test]` or `#[test(...)]` (the latter form is unusual but the AST is
# tolerant of both). Whitespace tolerant.
_TEST_RE = re.compile(
    r"^\s*#\s*\[\s*test(?:\s*(?:\(\s*\)|\s*\]))",
    re.MULTILINE,
)

# `#[ignore]`, `#[ignore = "reason"]`, `#[ignore(...)]`. Tolerant of
# whitespace.
_IGNORE_RE = re.compile(
    r"^\s*#\s*\[\s*ignore(?:\s*(?:=\s*\"[^\"]*\")?\s*\])",
    re.MULTILINE,
)

# `#[cfg_attr(<expr>, test)]` and `#[cfg_attr(<expr>, ignore)]` —
# conditional gating. The first capture group is the cfg expression
# (informational), the second is the inner attribute verb
# (`test` / `ignore`).
_CFG_ATTR_TEST_RE = re.compile(
    r"^\s*#\s*\[\s*cfg_attr\s*\(\s*[^\n]+?,\s*test(?:\s*(?:\(\s*\)|\s*\]))\s*\)",
    re.MULTILINE,
)
_CFG_ATTR_IGNORE_RE = re.compile(
    r"^\s*#\s*\[\s*cfg_attr\s*\(\s*[^\n]+?,\s*ignore(?:\s*(?:=\s*\"[^\"]*\")?\s*\])\s*\)",
    re.MULTILINE,
)


def _is_comment_line(line: str) -> bool:
    """A Rust doc-comment or block-comment line is not an attribute."""
    stripped = line.lstrip()
    return stripped.startswith(("/", "*"))


def _scan_attributes(text: str, lines: list[str]) -> tuple[int, int, int, int]:
    """Count ``#[test]`` / ``#[ignore]`` / conditional variants in a file.

    Returns ``(test_count, cfg_test_count, ignore_count, cfg_ignore_count)``.
    Doc-comment lines are skipped so a comment like
    ``/// add a `#[ignore]` here`` does not inflate the count. Matches
    the dedup convention of ``generate_quarantine_registry.py``: when
    the same line carries both an unconditional and a ``cfg_attr``
    variant, the unconditional wins (it is always compiled).
    """
    test_count = 0
    cfg_test_count = 0
    ignore_count = 0
    cfg_ignore_count = 0

    for match in _TEST_RE.finditer(text):
        line_idx = text.count("\n", 0, match.start())
        if line_idx < len(lines) and _is_comment_line(lines[line_idx]):
            continue
        test_count += 1
    for match in _IGNORE_RE.finditer(text):
        line_idx = text.count("\n", 0, match.start())
        if line_idx < len(lines) and _is_comment_line(lines[line_idx]):
            continue
        ignore_count += 1
    for match in _CFG_ATTR_TEST_RE.finditer(text):
        line_idx = text.count("\n", 0, match.start())
        if line_idx < len(lines) and _is_comment_line(lines[line_idx]):
            continue
        cfg_test_count += 1
    for match in _CFG_ATTR_IGNORE_RE.finditer(text):
        line_idx = text.count("\n", 0, match.start())
        if line_idx < len(lines) and _is_comment_line(lines[line_idx]):
            continue
        cfg_ignore_count += 1
    return test_count, cfg_test_count, ignore_count, cfg_ignore_count


def _walk_workspace_members() -> list[Path]:
    """Return the ``[members]`` of ``Cargo.toml`` minus the excludes.

    Falls back to a hardcoded set when the workspace cannot be parsed
    (e.g. running outside the repo). The hardcoded set is the
    2026-09-08 ``[workspace].members`` from the Issue #3442 PR.
    """
    cargo_toml = REPO_ROOT / "Cargo.toml"
    if not cargo_toml.exists():
        return [REPO_ROOT / "src"]
    text = cargo_toml.read_text(encoding="utf-8")
    members_match = re.search(
        r"\[workspace\][^\n]*\n(?:[^\[]+\n)*?\s*members\s*=\s*\[(.*?)\]",
        text,
        re.MULTILINE,
    )
    if members_match is None:
        return [REPO_ROOT / "src"]
    members: list[Path] = []
    raw = members_match.group(1)
    for entry in re.finditer(r'"([^"]+)"', raw):
        rel = entry.group(1).rstrip("/")
        if not rel:
            continue
        members.append(REPO_ROOT / rel)
    if not members:
        return [REPO_ROOT / "src"]
    return members


def _tracked_rs_files() -> list[str] | None:
    """Sorted POSIX paths of git-tracked ``*.rs`` files (Issue #4069).

    The AST inventory scan must be a pure function of the tracked
    tree, not the working directory: a dirty checkout (untracked
    scratch ``.rs`` files, agent ``.planning/worktrees/`` copies, …)
    must scan identically to a fresh CI checkout of the same commit.

    Returns ``None`` when git is unavailable or ``REPO_ROOT`` is not a
    git checkout, so the caller can fall back to the on-disk walk.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.rs"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    paths = [
        p for p in proc.stdout.decode("utf-8", errors="replace").split("\0") if p
    ]
    return sorted(paths)


def _crate_name_for(rel: Path, workspace_excludes: list[str]) -> str | None:
    """Map a member's repo-relative path to its inventory crate name.

    Returns ``None`` for excluded members. Mirrors the naming in
    ``scan_source_inventory`` (root crate ``.`` → ``"fluxion"``).
    """
    rel_s = rel.as_posix()
    if rel_s == ".":
        return "fluxion"
    if rel_s in workspace_excludes:
        return None
    return rel_s.replace("/", "_")


def _scan_file_list(
    members: list[Path], workspace_excludes: list[str]
) -> list[tuple[str, Path, str, Path]]:
    """Enumerate ``(crate_name, member, section, path)`` deterministically.

    Issue #4069: when git is available the enumeration is strictly the
    sorted ``git ls-files '*.rs'`` set, restricted to each member's
    ``src/`` and ``tests/`` sections and to files that exist on disk
    (a tracked-but-locally-deleted file is not compiled by cargo, so it
    is not scanned). Without git, falls back to the legacy on-disk
    ``rglob`` walk.
    """
    tracked = _tracked_rs_files()
    if tracked is None:
        return _scan_file_list_ondisk(members, workspace_excludes)
    # Longest-prefix-first so a nested member wins over the root "."
    # catch-all; the trailing "/" blocks "fluxion-core" from matching
    # "fluxion-core2/...".
    indexed: list[tuple[str, str, Path]] = []
    for member in members:
        rel = member.relative_to(REPO_ROOT)
        crate_name = _crate_name_for(rel, workspace_excludes)
        if crate_name is None:
            continue
        prefix = "" if rel.as_posix() == "." else rel.as_posix() + "/"
        indexed.append((prefix, crate_name, member))
    indexed.sort(key=lambda t: len(t[0]), reverse=True)
    out: list[tuple[str, Path, str, Path]] = []
    for posix_path in tracked:
        for prefix, crate_name, member in indexed:
            if not posix_path.startswith(prefix):
                continue
            rest = posix_path[len(prefix) :]
            if "/" not in rest:
                break  # tracked .rs outside src//tests/ (e.g. benches/)
            section, _ = rest.split("/", 1)
            if section not in ("src", "tests"):
                break
            path = REPO_ROOT / posix_path
            if path.is_file():
                out.append((crate_name, member, section, path))
            break
    return out


def _scan_file_list_ondisk(
    members: list[Path], workspace_excludes: list[str]
) -> list[tuple[str, Path, str, Path]]:
    """Legacy on-disk enumeration (pre-#4069 behavior, git-less fallback)."""
    out: list[tuple[str, Path, str, Path]] = []
    for member in members:
        rel = member.relative_to(REPO_ROOT)
        crate_name = _crate_name_for(rel, workspace_excludes)
        if crate_name is None:
            continue
        for section in ("src", "tests"):
            scan_dir = member / section
            if scan_dir.exists():
                for f in sorted(scan_dir.rglob("*.rs")):
                    out.append((crate_name, member, section, f))
    return out


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_test_blocks(cargo_toml_text: str) -> list[str]:
    """Return the ``path`` of every ``[[test]]`` block in ``Cargo.toml``.

    A ``[[test]]`` block declares a hand-wired integration test that
    lives OUTSIDE the auto-discovery convention (``<crate>/tests/*.rs``
    directly). The Drift Gate treats both shapes equivalently: each is
    one Cargo test binary.
    """
    paths: list[str] = []
    for match in re.finditer(
        r"\[\[test\]\][^\[]*?path\s*=\s*\"([^\"]+)\"",
        cargo_toml_text,
        re.DOTALL,
    ):
        paths.append(match.group(1))
    return paths


def scan_source_inventory(workspace_excludes: list[str]) -> dict:
    """Walk the workspace and tally test attributes per package.

    Returns the inventory dict (see module docstring for schema).
    Counts are derived from the on-disk source tree only (no cargo
    invocation); cross-checking with ``cargo test -- --list`` is the
    caller's responsibility (see ``verify_with_cargo`` below).
    """
    members = _walk_workspace_members()
    # The root crate at REPO_ROOT is implicit in Cargo workspaces
    # (``default-members = ["."]``). Add it explicitly so its
    # src/ + tests/ both contribute to the inventory.
    members = [REPO_ROOT] + list(members)

    # Issue #4069: enumerate strictly the sorted ``git ls-files
    # '*.rs'`` set (tracked files only), grouped per member/section.
    # The per-file counting below is unchanged from the on-disk walk.
    scan_files = _scan_file_list(members, workspace_excludes)
    src_files: dict[tuple[str, Path], list[Path]] = defaultdict(list)
    test_files: dict[tuple[str, Path], list[Path]] = defaultdict(list)
    for crate_name, member, section, path in scan_files:
        key = (crate_name, member)
        if section == "src":
            src_files[key].append(path)
        else:
            test_files[key].append(path)

    by_crate: dict[str, dict[str, int]] = defaultdict(
        lambda: {"test_binaries": 0, "lib_tests": 0, "lib_ignored": 0}
    )

    test_binary_count = 0
    test_source_file_count = 0
    workspace_test_count = 0  # integration test functions across workspace
    workspace_ignore_count = 0

    for member in members:
        rel = member.relative_to(REPO_ROOT)
        crate_name = _crate_name_for(rel, workspace_excludes)
        if crate_name is None:
            continue

        # lib / inline-`#[cfg(test)]` tests under src/
        src_dir = member / "src"
        if src_dir.exists():
            lib_tests = 0
            lib_ignored = 0
            for f in src_files.get((crate_name, member), []):
                text = f.read_text(encoding="utf-8")
                lines = text.split("\n")
                t, _ct, i, _ci = _scan_attributes(text, lines)
                lib_tests += t
                lib_ignored += i
            by_crate[crate_name]["lib_tests"] = lib_tests
            by_crate[crate_name]["lib_ignored"] = lib_ignored

        # Both auto-discovered (``<crate>/tests/*.rs``) and
        # hand-wired (``[[test]] path = "<crate>/tests/<sub>/foo.rs"``)
        # test binaries are Cargo test binaries. We scan EVERY
        # ``.rs`` file under ``tests/`` to cover both shapes.
        tests_dir = member / "tests"
        if tests_dir.exists():
            cargo_toml = member / "Cargo.toml"
            hand_wired_paths: set[str] = set()
            if cargo_toml.exists():
                hand_wired_paths = set(
                    _parse_test_blocks(cargo_toml.read_text(encoding="utf-8"))
                )
            for f in test_files.get((crate_name, member), []):
                text = f.read_text(encoding="utf-8")
                lines = text.split("\n")
                t, _ct, i, _ci = _scan_attributes(text, lines)
                # ``rel_to_member`` keeps the ``tests/`` prefix so it
                # can be compared against ``hand_wired_paths`` (which
                # also carry the ``tests/`` prefix from
                # ``Cargo.toml``).
                rel_to_member = f.relative_to(member).as_posix()
                rel_to_tests = f.relative_to(tests_dir).as_posix()
                if "/" not in rel_to_tests:
                    # top-level tests/*.rs auto-discovered binary
                    by_crate[crate_name]["test_binaries"] += 1
                    test_binary_count += 1
                elif rel_to_member in hand_wired_paths:
                    # Hand-wired ``[[test]] path = "tests/<sub>/<foo>.rs"``
                    # binary.
                    by_crate[crate_name]["test_binaries"] += 1
                    test_binary_count += 1
                else:
                    # Files in tests/** subdirs that are NOT
                    # hand-wired — they are test source files but
                    # not Cargo test binaries. Still counted under
                    # ``test_source_files`` for drift purposes.
                    test_source_file_count += 1
                workspace_test_count += t
                workspace_ignore_count += i

    by_crate_sorted = {k: dict(v) for k, v in sorted(by_crate.items())}

    # The root crate's lib_tests can be reported as a separate row
    # so the drift gate can verify ``cargo test --lib`` directly.
    root_lib_tests = by_crate_sorted.get("fluxion", {}).get("lib_tests", 0)
    root_lib_ignored = by_crate_sorted.get("fluxion", {}).get("lib_ignored", 0)

    # Aggregate per-crate lib totals so the workspace totals combine
    # lib + integration tests (mirrors ``cargo test --workspace ...``
    # which compiles both).
    workspace_lib_tests = sum(
        v.get("lib_tests", 0) for v in by_crate_sorted.values()
    )
    workspace_lib_ignored = sum(
        v.get("lib_ignored", 0) for v in by_crate_sorted.values()
    )

    # ``test_source_files`` = total number of ``*.rs`` files
    # under any ``tests/`` directory across the workspace. This
    # is a superset of ``test_binaries``: it counts every test
    # source, including subdir files referenced by ``mod ...``
    # inside a binary or simply lying in source control but not
    # wired into any test binary.
    total_test_source_files = test_binary_count + test_source_file_count

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_root": _git_head_sha() or "unknown",
        "scope": {"workspace_exclude": list(workspace_excludes)},
        "totals": {
            "test_binaries": test_binary_count,
            "test_source_files": total_test_source_files,
            "lib_tests_root": root_lib_tests,
            "lib_ignored_root": root_lib_ignored,
            "workspace_lib_tests": workspace_lib_tests,
            "workspace_lib_ignored": workspace_lib_ignored,
            "workspace_integration_tests": workspace_test_count,
            "workspace_integration_ignored": workspace_ignore_count,
            "workspace_tests": workspace_lib_tests + workspace_test_count,
            "workspace_ignored": workspace_lib_ignored + workspace_ignore_count,
            "doc_tests": 0,
        },
        "by_crate": by_crate_sorted,
    }
    # doc_tests default of 0 is overwritten by ``--verify``; pure-AST
    # mode cannot estimate it without running cargo.
    return inventory


def verify_with_cargo(
    inventory: dict,
    workspace_excludes: list[str],
    cargo_target_dir: str | None,
) -> dict:
    """Cross-check the inventory against ``cargo test --list``.

    Runs (sequentially, in separate subprocesses):

      cargo test --workspace --exclude <each-exclude> -- --list
      cargo test --workspace --exclude <each-exclude> -- --list --ignored
      cargo test --lib -- --list
      cargo test --lib -- --list --ignored

    Returns a copy of ``inventory`` with the ``verify`` block filled
    in. If the cargo invocation fails, ``matched`` is set to False and
    the drift gate will rely on the AST numbers only.
    """
    cmd_prefix = ["cargo", "test"]
    if cargo_target_dir:
        cmd_prefix = ["env", f"CARGO_TARGET_DIR={cargo_target_dir}", *cmd_prefix]

    workspace_cmd = cmd_prefix + ["--workspace"]
    for excl in workspace_excludes:
        workspace_cmd += ["--exclude", excl]
    workspace_cmd += ["--", "--list"]

    try:
        ws_proc = subprocess.run(
            workspace_cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "command": workspace_cmd,
            "executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "matched": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if ws_proc.returncode != 0:
        return {
            "command": workspace_cmd,
            "executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "matched": False,
            "error": f"cargo returned exit {ws_proc.returncode}",
            "stderr_tail": ws_proc.stderr[-1000:],
        }

    cargo_workspace_tests, _, cargo_doc_tests = _parse_cargo_listing(
        ws_proc.stdout, include_ignored=False
    )

    # workspace_cmd ends with `-- --list`; extend it with `--ignored`
    # so the test binary sees the right argument order.
    ignored_proc = subprocess.run(
        workspace_cmd + ["--ignored"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    cargo_workspace_ignored = 0
    if ignored_proc.returncode == 0:
        # In ignored-listing mode, every line is an ignored test;
        # ``_parse_cargo_listing(... ignored=True)`` returns
        # ``(test_count, ignored_count, doc_count)`` where
        # ``ignored_count == test_count + doc_count``.
        _, cargo_workspace_ignored, _ = _parse_cargo_listing(
            ignored_proc.stdout, include_ignored=True
        )

    # Lib tests
    lib_proc = subprocess.run(
        cmd_prefix + ["--lib", "--", "--list"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    cargo_lib_tests = None
    cargo_lib_ignored = 0
    if lib_proc.returncode == 0:
        cargo_lib_tests, _, _ = _parse_cargo_listing(
            lib_proc.stdout, include_ignored=False
        )

    lib_ignored_proc = subprocess.run(
        cmd_prefix + ["--lib", "--", "--list", "--ignored"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if lib_ignored_proc.returncode == 0:
        # In ignored-listing mode, the entire listing is ignored
        # tests, so the test_count == ignored count.
        _lib_test, _lib_doc, _ = _parse_cargo_listing(
            lib_ignored_proc.stdout, include_ignored=False
        )
        cargo_lib_ignored = _lib_test + _lib_doc

    return {
        "command": workspace_cmd,
        "executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "matched": True,
        "lib_tests": cargo_lib_tests,
        "lib_ignored": cargo_lib_ignored,
        "workspace_tests": cargo_workspace_tests,
        "workspace_ignored": cargo_workspace_ignored,
        "doc_tests": cargo_doc_tests,
    }


def _parse_cargo_listing(
    stdout: str, include_ignored: bool
) -> tuple[int, int, int]:
    """Parse ``cargo test -- --list`` stdout into ``(tests, ignored, doctests)``.

    The cargo ``--list`` output groups test names in blocks; each block
    belongs to one binary. Within a block, lines ending in ``: test``
    are ``#[test]`` functions (or doctest entries). Lines containing
    `` - `` like ``src/foo.rs - module::path (line 10): test`` are
    doctests; the rest are unit / integration tests.

    The ``--list --ignored`` variant omits the running-binary banners
    but otherwise has the same shape; we skip the binary-name tracking
    in that mode. In that mode, every listed test is *ignored*, so
    ``test_count`` and ``doc_count`` are both pure ignored entries.
    """
    test_count = 0
    doc_count = 0
    for line in stdout.splitlines():
        line = line.rstrip()
        if not line.endswith(": test"):
            continue
        # Heuristic: a doc-test line contains " - " separating the
        # source file from the qualified doc path.
        if " - " in line:
            doc_count += 1
        else:
            test_count += 1
    if include_ignored:
        # In ignored-listing mode, every entry is *already* an ignored
        # test; the caller treats test_count + doc_count as the
        # ignored universe.
        return test_count, test_count + doc_count, doc_count
    return test_count, 0, doc_count


def _git_head_sha() -> str | None:
    """Return the current HEAD commit SHA, or None if not a git repo."""
    head = REPO_ROOT / ".git" / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref_path = REPO_ROOT / ".git" / text.split(" ", 1)[1]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()[:12]
        return None
    return text[:12]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test-inventory generator (Issue #3442). Default mode is "
            "fast AST-regex over the on-disk source tree; --verify "
            "cross-checks against `cargo test -- --list`."
        )
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Cross-check the AST counts with `cargo test -- --list`.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Workspace package to exclude from the inventory (repeatable). "
            f"Defaults to {DEFAULT_WORKSPACE_EXCLUDES!r}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the inventory JSON to stdout (in addition to writing it).",
    )
    parser.add_argument(
        "--cargo-target-dir",
        default=None,
        help="CARGO_TARGET_DIR to pass to the verify invocation (default: reuse the workspace's target/).",
    )
    args = parser.parse_args()

    workspace_excludes = args.exclude if args.exclude else DEFAULT_WORKSPACE_EXCLUDES
    if any(e for e in workspace_excludes if "/" in e):
        print(
            f"ERROR: --exclude values must be package names (no slashes), got {workspace_excludes!r}",
            file=sys.stderr,
        )
        return 2

    inventory = scan_source_inventory(workspace_excludes)
    if args.verify:
        verify_block = verify_with_cargo(
            inventory, workspace_excludes, args.cargo_target_dir
        )
        inventory["verify"] = verify_block
        if verify_block.get("matched"):
            # Prefer the cargo counts when they are available.
            v = verify_block
            cargo = inventory["totals"]
            # Root-crate specific overrides.
            if v.get("lib_tests") is not None:
                cargo["lib_tests_root"] = v["lib_tests"]
            if v.get("lib_ignored") is not None:
                cargo["lib_ignored_root"] = v["lib_ignored"]
            # Workspace-wide overrides (cargo's --workspace invocation
            # naturally combines lib + integration + doctests).
            if v.get("workspace_tests") is not None:
                cargo["workspace_tests"] = v["workspace_tests"]
            if v.get("workspace_ignored") is not None:
                cargo["workspace_ignored"] = v["workspace_ignored"]
            if v.get("doc_tests") is not None:
                cargo["doc_tests"] = v["doc_tests"]
            # ``workspace_lib_tests`` stays as the per-crate AST sum
            # (we only validated ``--lib`` for the root crate).

    output_path: Path = args.output
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.json:
        json.dump(inventory, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        t = inventory["totals"]
        # --output may point outside the repo (e.g. a scratch temp file
        # from the drift gate, Issue #4134); fall back to the absolute
        # path instead of crashing on relative_to().
        try:
            out_display = str(output_path.relative_to(REPO_ROOT))
        except ValueError:
            out_display = str(output_path)
        print(
            f"Inventory written: {out_display} "
            f"(schema={SCHEMA_VERSION}, head={inventory['repo_root']})"
        )
        print(
            f"  test_binaries               = {t['test_binaries']} "
            f"(Cargo auto-discovered)"
        )
        print(
            f"  test_source_files           = {t['test_source_files']} "
            f"(incl. tests/** subdir files)"
        )
        print(f"  lib_tests_root              = {t['lib_tests_root']}")
        print(f"  lib_ignored_root            = {t['lib_ignored_root']}")
        print(f"  workspace_lib_tests         = {t['workspace_lib_tests']}")
        print(f"  workspace_lib_ignored       = {t['workspace_lib_ignored']}")
        print(f"  workspace_integration_tests = {t['workspace_integration_tests']}")
        print(f"  workspace_tests             = {t['workspace_tests']}")
        print(f"  workspace_ignored           = {t['workspace_ignored']}")
        print(f"  doc_tests                   = {t['doc_tests']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
