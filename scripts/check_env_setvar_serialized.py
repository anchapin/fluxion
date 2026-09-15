#!/usr/bin/env python3
"""
CI guard: verify that every integration test under ``tests/**/*.rs`` --
in the ROOT crate AND in every workspace member's ``tests/`` tree
(``fluxion-core/tests/``, ``crates/*/tests/``, ...; Issue #3746) -- that
mutates a process-wide env var (``std::env::set_var`` /
``std::env::remove_var``) serialises the mutation through a file-local
mutex. Issue #3453 closes the ADR-0014 nextest-migration safety-net gap
that let ``tests/ashrae_140_diagnostic_integration_test`` ship with
unprotected env mutation under ``--test-threads=2``. Issue #3746 widens
the scan from the root ``tests/`` tree alone to every sibling-crate
``tests/`` tree: sibling crates run inside the same
``cargo nextest run --workspace --all-targets`` invocation with the same
in-binary parallelism, so an env-mutating test added there would race
unchecked by a root-only gate.

Companion to ``scripts/update_concurrency_keys.py`` -- that script applies
the per-``head_sha`` concurrency template, this script enforces the
env-mutation serialisation convention. Wired into the ``Scripts Test
Suite`` job of ``.github/workflows/scripts-tests.yml`` ("Enforce
ENV_LOCK on env-mutating integration tests" step, Issue #3453) so any
future contributor adding an unguarded ``env::set_var`` call in
``tests/`` fails the CI gate before it can merge. The matcher's
invariants are additionally covered by
``scripts/ci/test_check_env_setvar_serialized.py`` against hermetic
``tmp_path`` fixtures.

For each ``<workspace-member>/tests/**/*.rs`` file (the scan roots are
the root ``tests/`` directory plus the ``tests/`` directory of every
``[workspace].members`` entry in ``Cargo.toml`` that has one; the root
crate is ``default-members = ["."]`` so it is always included):

  1. If the file does NOT call ``std::env::set_var`` /
     ``std::env::remove_var`` (or the unqualified ``env::set_var`` /
     ``env::remove_var``) anywhere, it is exempt -- no finding.
  2. If the file calls one of those functions, the file MUST declare at
     least one ``static <NAME>_LOCK: <TYPE>`` whose type contains
     ``Mutex`` (the convention is ``ENV_LOCK: Mutex<()>`` /
     ``std::sync::Mutex<()>``, but any ``*_LOCK: ... Mutex ...`` works).
     The convention is to acquire the lock with
     ``.lock().unwrap_or_else(|e| e.into_inner())`` so a poisoned
     mutex from a sibling panic does not deadlock the suite, but the
     gate enforces the declaration not the lock pattern (the lock
     pattern is enforced by code review per the AGENTS.md "Physics and
     Validation Guardrails" section).

Any file that mutates env vars without a visible ``*_LOCK: ... Mutex ...``
declaration is reported as drift with the file path. Exit code is 1 on
drift, 0 on clean.

Usage:
    python3 scripts/check_env_setvar_serialized.py

Exit codes:
    0 -- every scanned ``tests/**/*.rs`` that mutates env vars carries a
        file-local ``*_LOCK: ... Mutex ...`` declaration.
    1 -- one or more env-mutating tests lack the serialisation guard.
    2 -- script error (e.g. ``tests/`` missing).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

# Parse ``[workspace].members`` out of Cargo.toml so the scan scope is
# derived from the single source of truth (adding a workspace member
# widens this gate automatically, with no hardcoded crate list to
# forget). Mirrors ``scripts/generate_test_inventory.py::
# _walk_workspace_members`` (Issue #3442).
_WORKSPACE_MEMBERS_RE = re.compile(
    r"\[workspace\][^\n]*\n(?:[^\[]+\n)*?\s*members\s*=\s*\[(.*?)\]",
    re.MULTILINE,
)

# Sub-directory names skipped under EVERY tests/ root -- they contain
# data fixtures and ASCII golden files, not Rust source. Applied
# per-root (``<member>/tests/reference_data/`` etc.) so sibling crates
# get the same exclusions as the root tree (Issue #3746).
_SKIP_DIR_NAMES = {"reference_data", "fixtures"}

# Match `env::set_var` / `env::remove_var` with optional `std::` prefix.
# We intentionally do NOT match the `std::env` module declaration itself
# (e.g. `use std::env;`) -- only call sites. The regex anchors on the
# closing `(` so `set_var` inside an identifier (variable name, function
# name) cannot trip a false positive.
_ENV_MUTATION_RE = re.compile(
    r"\b(?:std::)?env::(?:set_var|remove_var)\s*\("
)

# Match a `static <NAME>_LOCK: <TYPE>` declaration whose type contains
# `Mutex`. The convention is `static ENV_LOCK: Mutex<()>` /
# `static ENV_LOCK: std::sync::Mutex<()>`, but the matcher accepts any
# name ending in `_LOCK` (e.g. `SHUTDOWN_ENV_LOCK`) so it generalises
# beyond `ENV_LOCK`. Anchored to the `static` keyword + the `_LOCK`
# suffix + a type that contains `Mutex` so a stray `Mutex` import in a
# doc-comment cannot satisfy the gate.
_LOCK_DECL_RE = re.compile(
    r"\bstatic\s+[A-Z][A-Z0-9_]*_LOCK\s*:\s*[^;=\n]*\bMutex\b[^;=\n]*",
    re.MULTILINE,
)


def workspace_member_tests_dirs(repo_root: Path) -> list[Path]:
    """Return every ``[workspace].members`` entry's ``tests/`` directory
    that exists on disk (Issue #3746).

    Parses ``Cargo.toml`` at ``repo_root`` with the same regex as
    ``scripts/generate_test_inventory.py::_walk_workspace_members`` so
    the sibling-crate scope tracks the workspace manifest instead of a
    hardcoded crate list. Entries without a ``tests/`` directory (e.g.
    ``fluxion-city``, ``fluxion-tauri/src-tauri``) are omitted; the root
    crate is handled separately by :func:`scan_roots` because it is the
    workspace ``default-members = ["."]`` package, not a ``members``
    entry. Returns ``[]`` when ``Cargo.toml`` is missing or has no
    parsable ``members`` list (e.g. hermetic ``tmp_path`` mock repos in
    the pytest suite that only exercise the root-tree semantics).
    """
    cargo_toml = repo_root / "Cargo.toml"
    if not cargo_toml.exists():
        return []
    text = cargo_toml.read_text(encoding="utf-8")
    members_match = _WORKSPACE_MEMBERS_RE.search(text)
    if members_match is None:
        return []
    dirs: list[Path] = []
    for entry in re.finditer(r'"([^"]+)"', members_match.group(1)):
        rel = entry.group(1).rstrip("/")
        if not rel:
            continue
        tests_dir = repo_root / rel / "tests"
        if tests_dir.is_dir():
            dirs.append(tests_dir)
    return sorted(set(dirs))


def scan_roots() -> list[Path]:
    """Return every ``tests/`` root the gate must scan (Issue #3746).

    The root crate's ``tests/`` first (it is the ``default-members``
    package and must exist -- ``main()`` exits 2 otherwise), then each
    workspace member's ``tests/`` directory, deduplicated and sorted for
    deterministic output. A ``members`` entry of ``"."`` would resolve
    back to the root ``tests/`` and is collapsed by the dedup.
    """
    roots = [TESTS_DIR]
    for member_dir in workspace_member_tests_dirs(REPO_ROOT):
        if member_dir not in roots:
            roots.append(member_dir)
    return roots


def iter_test_files(root: Path) -> list[Path]:
    """Return every ``*.rs`` file under a ``tests/`` root (recursive).

    Skips ``reference_data/`` and ``fixtures/`` sub-directories under
    that root because those directories contain data fixtures and ASCII
    golden files, not Rust source. The skip is per-root (relative to
    ``root``) so sibling-crate ``tests/`` trees get the same exclusions
    as the root tree (Issue #3746), and a directory that merely happens
    to be named ``fixtures`` ABOVE the tests root (e.g. a checkout under
    ``~/fixtures/fluxion``) is not skipped. The matcher is robust
    against false positives in those files (the ``_ENV_MUTATION_RE``
    matches call sites, not module-level strings), but skipping them
    keeps the scan fast and the output legible.
    """
    out: list[Path] = []
    for path in sorted(root.rglob("*.rs")):
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in _SKIP_DIR_NAMES for part in rel_parts[:-1]):
            continue
        out.append(path)
    return out


def has_env_mutation(text: str) -> bool:
    """True if ``text`` contains at least one ``env::set_var`` /
    ``env::remove_var`` call site (with optional ``std::`` prefix)."""
    return _ENV_MUTATION_RE.search(text) is not None


def has_lock_declaration(text: str) -> bool:
    """True if ``text`` declares at least one ``static <NAME>_LOCK: ...
    Mutex ...`` (the file-local serialisation guard convention)."""
    return _LOCK_DECL_RE.search(text) is not None


def check_test_file(path: Path) -> list[str]:
    """Return a list of drift findings for the test file at ``path``.

    Empty list means the file is compliant (no env mutation, OR the
    file carries a ``*_LOCK: ... Mutex ...`` declaration).
    """
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(REPO_ROOT).as_posix()

    if not has_env_mutation(text):
        return []

    if has_lock_declaration(text):
        return []

    return [
        (
            f"{rel}: mutates process-wide env vars (std::env::set_var / "
            "remove_var) without a file-local "
            "`static <NAME>_LOCK: ... Mutex ...` serialisation guard. "
            "See Issue #3453; the convention is documented in "
            "`tests/onnx_signature_integration.rs`, "
            "`tests/email_notifier_header_safety.rs`, and "
            "`src/ai/surrogate.rs`."
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Restrict to a single test file (e.g. "
             "ashrae_140_diagnostic_integration_test.rs).",
    )
    args = parser.parse_args()

    if not TESTS_DIR.is_dir():
        print(f"ERROR: {TESTS_DIR} not found", file=sys.stderr)
        return 2

    roots = scan_roots()
    files: list[Path] = []
    for root in roots:
        files.extend(iter_test_files(root))
    if args.test:
        files = [f for f in files if f.name == args.test]
        if not files:
            print(f"ERROR: test file {args.test} not found", file=sys.stderr)
            return 2

    all_findings: list[str] = []
    env_mutation_files = 0
    for path in files:
        findings = check_test_file(path)
        if has_env_mutation(path.read_text(encoding="utf-8")):
            env_mutation_files += 1
        all_findings.extend(findings)

    if all_findings:
        print(
            f"ENV_LOCK drift detected ({len(all_findings)} finding(s) "
            f"across {env_mutation_files} env-mutating test file(s)):",
            file=sys.stderr,
        )
        for f in all_findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nAdd a file-local `static ENV_LOCK: Mutex<()> = "
            "Mutex::new(());` (or equivalent `<NAME>_LOCK: ... Mutex ...`) "
            "and acquire it around every `env::set_var` / "
            "`env::remove_var` site.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: all {len(files)} test file(s) scanned across "
        f"{len(roots)} tests/ root(s); "
        f"{env_mutation_files} env-mutating file(s) carry the "
        "serialisation guard."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
