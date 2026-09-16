#!/usr/bin/env python3
"""Fail any workflow whose credential-bearing job uses unpinned ``pip install``.

Issue #3812: an unpinned ``pip install <package>`` step in a workflow
job that holds or can mint high-value credentials (``secrets.*`` env,
``permissions: id-token: write``, or a branch-protection PAT) re-resolves
the package on every run, leaving the credential scope exposed to a
dependency-confusion / compromised-transitive-dep PyPI artifact. The
project pins every ``uses:`` action by SHA via
``check_workflow_pin.py`` / ``check_action_pinning.py`` — the Python
install layer of the same jobs must be pinned at parity.

The gate accepts two compliance shapes (per the issue's acceptance
criteria):

  1. ``pip install -r <pinned-requirements-file>`` where the file lives
     at a path that exists in the repo and contains at least one
     version-pinned specifier (``==``, ``>=``, ``~=``, etc.) — bare
     ``-r`` references without an on-disk file or with no pin are
     rejected.
  2. ``pip install <package><specifier>`` where ``<specifier>`` is a
     version pin (``==``, ``>=``, ``~=``, ``<=``, etc.). Bare
     ``pip install <package>`` invocations fail.

The credential-bearing criterion is satisfied when EITHER of these
holds at the workflow level:

  * Any step ``env:`` contains a ``secrets.*`` reference.
  * The job declares ``permissions: id-token: write``.
  * The job declares ``permissions: contents: write`` (low-bar
    proxy for branch-mutation credentials).

Jobs without any of the above are out of scope — bare ``pip install``
in a non-credential-bearing job is informational only (not a fail).

Usage::

    python3 scripts/check_pip_pinning.py
    python3 scripts/check_pip_pinning.py --self-test

Exit codes:

    0 -- every credential-bearing job's ``pip install`` is pinned or
         uses a constraints file with at least one pinned specifier.
    1 -- one or more unpinned installs detected.
    2 -- script error (e.g. ``.github/workflows/`` missing).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# A pip install invocation. Captures the argv-after-pip text.
# Examples matched:
#   pip install pyyaml
#   pip install -r scripts/requirements-ci.txt
#   python -m pip install --upgrade pip
#   pip install boto3==1.35.36
#   pip install 'twine==5.1.1'
_PIP_INSTALL_RE = re.compile(
    r"""
    \b pip \s+ install \b       # the literal "pip install" word
    (?: \s+ --[^\s]+ )*          # optional pip flags (e.g. --upgrade, --quiet)
    \s+
    (?P<args> .+?)               # the rest of the line
    $                            # end-of-line
    """,
    re.VERBOSE | re.MULTILINE,
)

# Packages that are not third-party dependencies and don't need
# pinning (self-installs of the package manager itself, stdlib
# wrappers, etc.).
_ALWAYS_OK_PACKAGES = frozenset({"pip", "setuptools", "wheel", "pipx"})

# A pinned package specifier: `name==X.Y.Z`, `name>=X.Y`, etc.
# Allows quoted forms ('name==X' or "name==X") to handle shell
# contexts where the specifier must be quoted.
_PINNED_SPEC_RE = re.compile(
    r"""
    (?P<name>[A-Za-z0-9_.+-]+)     # package name (PEP 508)
    (?P<op>==|>=|<=|~=|!=|>|<)     # version operator
    (?P<ver>[A-Za-z0-9_.+!*-]+)    # version spec
    """,
    re.VERBOSE,
)

# A line that pins a version specifier — used to detect that a
# constraints file contains at least one pin.
_LINE_PIN_RE = _PINNED_SPEC_RE


def _is_credential_bearing_job(job_text: str) -> bool:
    """Return True if `job_text` (the YAML body of a single job) holds
    or mints a high-value credential."""
    if re.search(r"^\s*env:\s*(?:\n|$)", job_text, re.MULTILINE):
        # Crude: any `secrets.*` reference anywhere in the job text
        # (env, env.* keys, run blocks, action withs).
        if re.search(r"\$\{\{\s*secrets\.", job_text):
            return True
    if re.search(r"^\s*id-token:\s*write\b", job_text, re.MULTILINE):
        return True
    # contents:write is a low-bar proxy for branch-mutation credentials
    # (most branch-protection / release workflows declare this).
    if re.search(r"^\s*contents:\s*write\b", job_text, re.MULTILINE):
        return True
    return False


def _classify_pip_install_args(args: str, workflow_path: Path) -> tuple[bool, str]:
    """Return (compliant, reason) for a single ``pip install`` invocation.

    Compliant when EITHER:
      * at least one package is pinned (==, >=, etc.) — bare
        ``pip install foo`` is non-compliant.
      * the first non-flag token is ``-r`` / ``--requirement`` pointing
        to a constraints file that exists and contains at least one
        pinned specifier.
    """
    tokens = args.split()
    # Drop pip flags (anything starting with '-' that's not a value).
    positional: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"-r", "--requirement"}:
            # Next token is the file path.
            if i + 1 >= len(tokens):
                return False, "-r requires a file argument"
            req_path = tokens[i + 1].strip("'\"")
            # Resolve relative to repo root (CI runners run from
            # $GITHUB_WORKSPACE which is the repo root).
            full_path = (REPO_ROOT / req_path).resolve()
            if not full_path.is_file():
                return False, f"-r file does not exist: {req_path}"
            try:
                content = full_path.read_text(encoding="utf-8")
            except OSError as exc:
                return False, f"-r file unreadable: {req_path} ({exc})"
            if not any(_LINE_PIN_RE.search(line) for line in content.splitlines()):
                return False, (
                    f"-r file {req_path} contains no version-pinned "
                    f"specifier — at least one `name==X.Y.Z` is required"
                )
            return True, f"constraints file: {req_path}"
        if tok.startswith("-"):
            # Skip the flag and any value it takes inline / as next
            # token. We don't enumerate every flag; conservative:
            # consume the flag and, if it has an `=` style value,
            # consume it inline; otherwise treat as boolean and move on.
            if "=" in tok:
                i += 1
                continue
            # Common "value-takes" flags: --find-links, --target, --index-url,
            # --extra-index-url, --trusted-host, --root, --prefix, --src, --cache-dir
            value_taking = {
                "--find-links", "--target", "--index-url", "--extra-index-url",
                "--trusted-host", "--root", "--prefix", "--src", "--cache-dir",
                "--platform", "--abi", "--implementation", "--python-version",
                "--no-binary", "--only-binary", "--progress-bar",
            }
            if tok in value_taking and i + 1 < len(tokens):
                i += 2
                continue
            i += 1
            continue
        positional.append(tok)
        i += 1

    if not positional:
        return False, "no positional package arguments"

    # At least one positional package must be pinned. Self-upgrades
    # of the package manager itself (pip, setuptools, wheel, pipx)
    # are explicitly allowed without a version specifier — these
    # are not third-party deps and pinning the package manager is
    # outside the gate's scope.
    for tok in positional:
        if _PINNED_SPEC_RE.search(tok):
            return True, f"pinned: {tok}"
        # Strip trailing quotes / parens that may wrap a bare name
        # (e.g. `pip` inside a quoted list).
        bare = tok.strip("'\"()").strip()
        if bare in _ALWAYS_OK_PACKAGES:
            continue
        # Non-OK package and not pinned → remember as failure.
        # (We keep looping in case a later positional is pinned, but
        # if at the end nothing is pinned we report the first non-OK
        # failure.)
    if all(
        tok.strip("'\"()").strip() in _ALWAYS_OK_PACKAGES
        or _PINNED_SPEC_RE.search(tok)
        for tok in positional
    ):
        return True, "self-upgrade only"
    non_ok = [
        tok
        for tok in positional
        if tok.strip("'\"()").strip() not in _ALWAYS_OK_PACKAGES
        and not _PINNED_SPEC_RE.search(tok)
    ]
    return False, f"unpinned: {' '.join(non_ok)}"


def _iter_jobs(workflow_text: str) -> list[tuple[str, str]]:
    """Yield (job_id, job_text) pairs from a workflow file body.

    Workflows use 2-space indentation. Each job under ``jobs:`` is
    keyed by ``  <id>:`` and its value is the indented (4-space)
    mapping that follows. We locate the ``jobs:`` block, then walk
    through its top-level children by indentation rather than brace
    balancing (YAML mappings aren't brace-delimited).
    """
    lines = workflow_text.splitlines(keepends=True)
    # Locate the `jobs:` top-level key.
    jobs_idx: int | None = None
    for i, line in enumerate(lines):
        if re.match(r"^jobs:\s*(#.*)?$", line):
            jobs_idx = i
            break
    if jobs_idx is None:
        return []

    # Job keys live at exactly 2-space indentation under `jobs:`.
    # Everything indented 4+ spaces belongs to the current job.
    jobs: list[tuple[str, str]] = []
    i = jobs_idx + 1
    n = len(lines)
    while i < n:
        line = lines[i]
        # Skip blank lines (don't let their 0-indent close the block).
        if not line.strip():
            i += 1
            continue
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        # Skip comment lines (don't let their 0-indent close the block).
        if stripped.lstrip().startswith("#"):
            i += 1
            continue
        # End of jobs block: any NON-EMPTY non-comment line at indent 0
        # (top-level sibling key) closes the jobs block.
        if indent == 0:
            break
        # A 2-space key starts a new job. Capture its key. Strip
        # trailing newline before checking the `:` terminator (the
        # splitlines(keepends=True) leaves `\n` on every line).
        clean = stripped.rstrip("\n").rstrip("\r")
        if indent == 2 and clean.endswith(":"):
            key = clean[:-1].strip()
            value_start = i + 1
            # Walk forward as long as lines are indented 4+ spaces
            # (job body) — including blank/comment lines that are
            # between body lines.
            j = value_start
            while j < n:
                ln = lines[j]
                if not ln.strip():
                    j += 1
                    continue
                ln_stripped = ln.lstrip(" ")
                if ln_stripped.lstrip().startswith("#"):
                    j += 1
                    continue
                ln_indent = len(ln) - len(ln_stripped)
                if ln_indent < 4:
                    break
                j += 1
            job_text = "".join(lines[value_start:j])
            jobs.append((key, job_text))
            i = j
            continue
        # Unrecognised indent inside jobs block — skip defensively.
        i += 1
    return jobs


def scan_workflow(workflow_path: Path) -> list[str]:
    """Return a list of human-readable failure messages for `workflow_path`.

    An empty list means the workflow is compliant.
    """
    failures: list[str] = []
    try:
        text = workflow_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{workflow_path}: cannot read ({exc})"]

    for job_id, job_text in _iter_jobs(text):
        if not _is_credential_bearing_job(job_text):
            continue
        for match in _PIP_INSTALL_RE.finditer(job_text):
            args = match.group("args").strip()
            # Skip `pip install --help`, etc. — heuristic: require at
            # least one non-flag character beyond whitespace.
            if not args:
                continue
            ok, reason = _classify_pip_install_args(args, workflow_path)
            if not ok:
                rel = workflow_path.relative_to(REPO_ROOT)
                failures.append(
                    f"{rel}::{job_id}: unpinned pip install in "
                    f"credential-bearing job: {args!r} — {reason}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reject unpinned pip install steps in credential-bearing "
            "workflow jobs (Issue #3812)."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run an in-process compliance matrix and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _selftest()

    if not WORKFLOWS_DIR.is_dir():
        print(f"::error::workflows dir missing: {WORKFLOWS_DIR}", file=sys.stderr)
        return 2

    all_failures: list[str] = []
    workflows = sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))
    for wf in workflows:
        all_failures.extend(scan_workflow(wf))

    print("=== Fluxion Pip-Pinning Gate (Issue #3812) ===")
    print(f"Workflows: {len(workflows)}")
    print(f"Failures: {len(all_failures)}")
    if all_failures:
        print()
        for line in all_failures:
            print(f"  {line}")
        print()
        print(f"FAIL: {len(all_failures)} unpinned pip install(s) detected.")
        return 1
    print()
    print("PASS: every credential-bearing workflow's pip install is pinned.")
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_SELF_TEST_WORKFLOW = """\
name: test-workflow
on: [push]
jobs:
  cred-job:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    steps:
      - run: pip install pyyaml
      - run: pip install -r scripts/requirements-ci.txt
      - run: pip install boto3==1.35.36
      - run: pip install 'twine==5.1.1'
  safe-job:
    runs-on: ubuntu-latest
    steps:
      - run: pip install pyyaml
      - run: pip install --quiet pyyaml==6.0
"""


def _selftest() -> int:
    """In-process compliance matrix; exits 0 if all assertions hold."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        req_file = tmp_path / "scripts" / "requirements-ci.txt"
        req_file.parent.mkdir(parents=True, exist_ok=True)
        req_file.write_text("pyyaml>=6.0\nboto3==1.35.36\n", encoding="utf-8")
        wf_path = wf_dir / "ci.yml"
        wf_path.write_text(_SELF_TEST_WORKFLOW, encoding="utf-8")

        # Monkey-patch REPO_ROOT and WORKFLOWS_DIR for this test.
        global REPO_ROOT, WORKFLOWS_DIR
        saved_root, saved_dir = REPO_ROOT, WORKFLOWS_DIR
        try:
            REPO_ROOT = tmp_path  # type: ignore[assignment]
            WORKFLOWS_DIR = wf_dir  # type: ignore[assignment]
            failures = scan_workflow(wf_path)
        finally:
            REPO_ROOT, WORKFLOWS_DIR = saved_root, saved_dir

    # `cred-job` has `secrets.GITHUB_TOKEN` and `contents: write`:
    #   * `pip install pyyaml` — unpinned → FAIL.
    #   * `pip install -r scripts/requirements-ci.txt` — constraints
    #     file exists and has pins → PASS.
    #   * `pip install boto3==1.35.36` — pinned → PASS.
    #   * `pip install 'twine==5.1.1'` — pinned (quoted) → PASS.
    # `safe-job` has no credentials → no failures from it.
    bare_pyyaml_hits = [line for line in failures if "pyyaml" in line]
    if not bare_pyyaml_hits:
        print("FAIL: expected unpinned-bare-pyyaml failure missing", file=sys.stderr)
        print(f"  got: {failures}", file=sys.stderr)
        return 1
    if any("boto3==1.35.36" in line for line in failures):
        print("FAIL: pinned boto3 falsely flagged", file=sys.stderr)
        return 1
    if any("twine==5.1.1" in line for line in failures):
        print("FAIL: pinned (quoted) twine falsely flagged", file=sys.stderr)
        return 1
    if any("requirements-ci.txt" in line for line in failures):
        print("FAIL: constraints file install falsely flagged", file=sys.stderr)
        return 1
    # safe-job's `pip install pyyaml` should NOT appear in failures
    # because the job has no credentials.
    safe_job_hits = [line for line in failures if "safe-job" in line]
    if safe_job_hits:
        print(f"FAIL: safe-job (no creds) was flagged: {safe_job_hits}", file=sys.stderr)
        return 1
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())