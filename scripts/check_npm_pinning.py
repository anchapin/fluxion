#!/usr/bin/env python3
"""Deterministic npm install gate: exact-pinned deps + lockfile + `npm ci`.

Issue #4202: ``node-bindings.yml`` used bare ``npm install`` (resolves from
``package.json`` at run time) and ``npm/package.json`` floated
``@napi-rs/cli`` on ``^3.0.0-alpha.0`` with no committed lockfile -- every
CI run could resolve a different dependency tree. The fix exact-pins
``@napi-rs/cli`` to ``3.10.5``, commits ``npm/package-lock.json``, and
installs with ``npm ci``. This gate keeps it that way.

The compliance rule:

  1. Every entry in ``npm/package.json`` ``dependencies`` /
     ``devDependencies`` (and ``optionalDependencies`` /
     ``peerDependencies`` when present) MUST be an exact semver version
     (``3.10.5``, ``3.0.0-alpha.0``) -- no ``^`` / ``~`` / ranges /
     ``*`` / ``latest`` / ``file:`` / ``git:`` / bare-tag specs.
  2. ``npm/package-lock.json`` MUST exist, parse as JSON, and its root
     ``packages[""]`` dependency maps MUST equal the manifest's -- a
     lockfile that drifted from ``package.json`` is a fail (no ``npm``
     invocation needed for the check).
  3. ``.github/workflows/node-bindings.yml`` MUST install with ``npm ci``:
     a bare ``run: npm install`` step fails (it re-resolves instead of
     reproducing the lockfile).

Usage::

    python3 scripts/check_npm_pinning.py             # scan the real tree
    python3 scripts/check_npm_pinning.py --self-test # deterministic self-test

Exit codes:
    0 -- manifest exact-pinned, lockfile present and in sync, workflow on npm ci
    1 -- drift detected
    2 -- script error (e.g. npm/package.json missing or unparseable)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NPM_DIR = REPO_ROOT / "npm"
PACKAGE_JSON = NPM_DIR / "package.json"
PACKAGE_LOCK = NPM_DIR / "package-lock.json"
NODE_BINDINGS_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "node-bindings.yml"
)

# Exact semver: 3.10.5, 3.0.0-alpha.0, 1.2.3+build. Anything with a range
# operator, prefix, or non-numeric scheme fails.
_EXACT_VERSION_RE = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)

# Dependency sections whose specs must be exact.
_DEP_SECTIONS = (
    "dependencies",
    "devDependencies",
    "optionalDependencies",
    "peerDependencies",
)

# A bare `npm install` run-step in the workflow (not `npm ci`, not
# `npm install <pkg>@<ver>`): re-resolves the tree instead of reproducing
# the lockfile.
_BARE_NPM_INSTALL_RE = re.compile(r"^\s*run:\s*npm install\s*$")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_manifest_exact_pins(manifest: dict) -> list[str]:
    """Fail every non-exact dependency spec in the manifest."""
    findings: list[str] = []
    for section in _DEP_SECTIONS:
        deps = manifest.get(section) or {}
        for name, spec in deps.items():
            if not isinstance(spec, str) or not _EXACT_VERSION_RE.match(spec):
                findings.append(
                    f"npm/package.json::{section}[{name!r}] spec {spec!r} "
                    f"is not an exact semver version -- pin it (e.g. "
                    f"`\"3.10.5\"`, no `^`/`~`/range) so `npm ci` "
                    f"reproduces the same tree on every run."
                )
    return findings


def check_lockfile_in_sync(manifest: dict, lock: dict) -> list[str]:
    """Fail when the lockfile's root dependency maps differ from the manifest."""
    findings: list[str] = []
    root = (lock.get("packages") or {}).get("")
    if not isinstance(root, dict):
        return [
            "npm/package-lock.json has no root `packages[\"\"]` entry -- "
            "regenerate with `npm install --package-lock-only` inside npm/."
        ]
    for section in _DEP_SECTIONS:
        manifest_deps = manifest.get(section) or {}
        if not manifest_deps:
            continue
        lock_deps = root.get(section) or {}
        if lock_deps != manifest_deps:
            findings.append(
                f"npm/package-lock.json root `{section}` {lock_deps!r} does "
                f"not match npm/package.json `{section}` {manifest_deps!r} "
                f"-- regenerate the lockfile (`npm install "
                f"--package-lock-only` inside npm/) and commit it."
            )
    return findings


def check_workflow_uses_npm_ci(text: str) -> list[str]:
    """Fail a bare `npm install` install step in node-bindings.yml."""
    findings: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.lstrip().startswith("#"):
            continue
        if _BARE_NPM_INSTALL_RE.match(raw):
            findings.append(
                f".github/workflows/node-bindings.yml:{lineno}: bare "
                f"`npm install` re-resolves the dependency tree -- use "
                f"`npm ci` (Issue #4202: npm/package-lock.json is committed "
                f"and is the single source of truth)."
            )
    return findings


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"ERROR: {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return _self_test()

    findings: list[str] = []

    if not PACKAGE_JSON.is_file():
        print(f"ERROR: {PACKAGE_JSON} not found", file=sys.stderr)
        return 2
    manifest = _load_json(PACKAGE_JSON)
    findings.extend(check_manifest_exact_pins(manifest))

    if not PACKAGE_LOCK.is_file():
        findings.append(
            "npm/package-lock.json is missing -- generate it with "
            "`npm install --package-lock-only` inside npm/ and commit it "
            "(Issue #4202: `npm ci` requires a lockfile)."
        )
    else:
        lock = _load_json(PACKAGE_LOCK)
        findings.extend(check_lockfile_in_sync(manifest, lock))

    if not NODE_BINDINGS_WORKFLOW.is_file():
        print(f"ERROR: {NODE_BINDINGS_WORKFLOW} not found", file=sys.stderr)
        return 2
    findings.extend(
        check_workflow_uses_npm_ci(
            NODE_BINDINGS_WORKFLOW.read_text(encoding="utf-8")
        )
    )

    if findings:
        print("npm pinning drift detected:", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        "OK: npm/package.json exact-pinned, npm/package-lock.json present "
        "and in sync, node-bindings.yml installs with `npm ci`."
    )
    return 0


# ---------------------------------------------------------------------------
# Self-test (deterministic; no repo state required)
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Exercise the three checks against hermetic fixtures."""
    print("=== check_npm_pinning.py self-test ===")
    failures: list[str] = []

    # --- manifest exact-pin classification ---
    if check_manifest_exact_pins(
        {"devDependencies": {"@napi-rs/cli": "3.10.5"}}
    ) != []:
        failures.append("exact pin 3.10.5 should pass")
    if check_manifest_exact_pins(
        {"devDependencies": {"@napi-rs/cli": "3.0.0-alpha.0"}}
    ) != []:
        failures.append("exact prerelease pin should pass")
    for bad in ("^3.0.0-alpha.0", "~3.10.5", ">=3.0.0", "*", "latest",
                "3.10.x", "file:../x", "git+https://x"):
        got = check_manifest_exact_pins({"dependencies": {"a": bad}})
        if len(got) != 1:
            failures.append(f"spec {bad!r} should yield exactly 1 finding")

    # --- lockfile sync ---
    manifest = {"devDependencies": {"@napi-rs/cli": "3.10.5"}}
    in_sync = {"packages": {"": {"devDependencies":
                                 {"@napi-rs/cli": "3.10.5"}}}}
    if check_lockfile_in_sync(manifest, in_sync) != []:
        failures.append("in-sync lockfile should pass")
    drifted = {"packages": {"": {"devDependencies":
                                 {"@napi-rs/cli": "3.10.4"}}}}
    if len(check_lockfile_in_sync(manifest, drifted)) != 1:
        failures.append("drifted lockfile should yield exactly 1 finding")
    if len(check_lockfile_in_sync(manifest, {"packages": {}})) != 1:
        failures.append("lockfile without root entry should fail")

    # --- workflow npm ci ---
    ok_wf = (
        "      - name: Install npm dependencies\n"
        "        working-directory: npm\n"
        "        run: npm ci\n"
    )
    if check_workflow_uses_npm_ci(ok_wf) != []:
        failures.append("`npm ci` step should pass")
    bad_wf = (
        "      - name: Install npm dependencies\n"
        "        working-directory: npm\n"
        "        run: npm install\n"
    )
    bad_findings = check_workflow_uses_npm_ci(bad_wf)
    if len(bad_findings) != 1 or ":3:" not in bad_findings[0]:
        failures.append("bare `npm install` should fail naming line 3")
    commented = "#        run: npm install\n        run: npm ci\n"
    if check_workflow_uses_npm_ci(commented) != []:
        failures.append("commented-out `npm install` should be skipped")

    if failures:
        print("FAIL: self-test assertions did not hold:")
        for f in failures:
            print(f"  - {f}")
        return 2
    print("PASS: manifest / lockfile-sync / npm-ci checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
