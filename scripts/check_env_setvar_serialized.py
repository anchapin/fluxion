#!/usr/bin/env python3
"""
CI guard: verify that every integration test under ``tests/**/*.rs`` that
mutates a process-wide env var (``std::env::set_var`` /
``std::env::remove_var``) serialises the mutation through a file-local
mutex. Issue #3453 closes the ADR-0014 nextest-migration safety-net gap
that let ``tests/ashrae_140_diagnostic_integration_test`` ship with
unprotected env mutation under ``--test-threads=2``.

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

For each ``tests/**/*.rs`` file:

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
    0 -- every ``tests/**/*.rs`` that mutates env vars carries a
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


def iter_test_files(root: Path) -> list[Path]:
    """Return every ``*.rs`` file under ``tests/`` (recursive).

    Skips ``tests/reference_data/`` and ``tests/fixtures/`` because those
    directories contain data fixtures and ASCII golden files, not
    Rust source. The matcher is robust against false positives in those
    files (the `_ENV_MUTATION_RE` matches call sites, not module-level
    strings), but skipping them keeps the scan fast and the output
    legible.
    """
    skip_dirs = {REPO_ROOT / "tests" / "reference_data",
                 REPO_ROOT / "tests" / "fixtures"}
    out: list[Path] = []
    for path in sorted(root.rglob("*.rs")):
        if any(parent in skip_dirs for parent in path.parents):
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

    files = iter_test_files(TESTS_DIR)
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
        f"OK: all {len(files)} test file(s) scanned; "
        f"{env_mutation_files} env-mutating file(s) carry the "
        "serialisation guard."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
