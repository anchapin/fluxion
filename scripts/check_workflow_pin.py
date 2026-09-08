#!/usr/bin/env python3
"""
CI hygiene guard: fail on non-SHA-pinned ``uses:`` in ``.github/workflows/*.yml``.

Issue #3475: PR #3472 repinned the repo's last tag-pinned third-party actions
but, by scope, did not add a general enforcement gate. The guard added here
catches the next contributor who copies a mutable ``@vN`` / ``@stable`` /
``@main`` ref into any workflow -- a re-resolving reference that can swap
the action's code out from under us on every run.

The compliance rule (mirrors ``docs/SECURITY.md`` §5 *Action pinning*):

    Every ``uses:`` under ``.github/workflows/`` MUST be one of:

      1. A 40-hex commit SHA, optionally followed by a trailing version
         comment (``# vN.M.P``). SHA-pinned references never re-resolve.
         Examples:
             uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1
             uses: nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60  # v4.0.0

      2. A local composite / reusable path beginning with ``./`` (or ``.\\``
         on Windows -- N/A on Linux CI runners). Local actions are versioned
         by the repository itself, not by a third party.
         Example:
             uses: ./.github/actions/setup-rust-env

    Anything else -- tags (``@v4``), branches (``@stable``, ``@main``),
    bare refs without ``@``, etc. -- is a hard FAIL. The acceptance
    criteria explicitly cite ``tag-pinned, branch-pinned, and bare refs``
    as failure cases (``#3475`` acceptance criterion #2).

Commented-out ``uses:`` lines (lines whose first non-whitespace character
is ``#``) are skipped. YAML allows structural commenting; a commented-out
ref is not executed.

Usage::

    python3 scripts/check_workflow_pin.py             # scan all workflows
    python3 scripts/check_workflow_pin.py --self-test # deterministic self-test

Exit codes:
    0 -- every ``uses:`` is SHA-pinned or a local composite path
    1 -- one or more ``uses:`` fail the pin rule (drift detected)
    2 -- script error (e.g. ``.github/workflows/`` missing)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------
#
# `uses:` lines come in two shapes in the repo's workflows:
#
#   - list step form:      `      - uses: owner/action@<ref>  # vN`
#   - bare key form:       `        uses: owner/action@<ref>  # vN`
#
# Both shapes carry the ref as a single inline string. Quoted multi-line
# refs (`uses: |` or `uses: >`) are not used in any of the repo's
# workflows -- if a contributor introduces one, this regex will skip it,
# which is the safe default (silently passing a brand-new shape would be
# worse than a false negative the contributor fixes during review).
#
# The ref capture stops at the first `` # `` (a trailing version comment)
# or end-of-line. A bare ref (``uses: owner/action``) is matched with
# the same group-1 capture so the classifier can flag it as `bare`.
_USES_LINE_RE = re.compile(
    r"^\s*-?\s*uses:\s+(?P<ref>\S+)(?:\s+#.*)?$"
)

# SHA pin: GitHub commit SHAs are 40 lowercase hex characters. The repo's
# workflows use lowercase consistently; this regex still tolerates mixed
# case (a hash with uppercase letters would still be a valid SHA, just
# non-canonical).
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

# Local composite / reusable path. Per docs/SECURITY.md §5, the only
# non-SHA class that is acceptable is a path beginning with `./` (or
# `.\` for completeness, though CI runners are Linux). Anchored at the
# start of the ref so a ref like `./.github/actions/setup-rust-env@v1`
# still passes (a local path can carry a version, even though the repo
# does not currently use that pattern).
_LOCAL_PATH_RE = re.compile(r"^\.[\\/]")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify(ref: str) -> str:
    """Classify a single ``uses:`` reference.

    Returns one of:

      - ``"sha"``       -- ``owner/repo@<40hex>``
      - ``"local"``     -- ``./local/path`` (composite or reusable)
      - ``"tag"``       -- ``owner/repo@v4`` (or any other non-SHA ``@<ref>``)
      - ``"branch"``    -- ``owner/repo@stable`` / ``@main`` / etc.
      - ``"bare"``      -- ``owner/repo`` with no ``@`` separator
      - ``"empty"``     -- ``uses: `` with no ref at all (malformed)
      - ``"unknown"``   -- anything else (defensive default)

    The split between ``tag`` and ``branch`` is informational only (both
    are FAIL); the function returns whichever non-SHA suffix is present
    so the remediation message can name the offending pattern.
    """
    if not ref:
        return "empty"

    if _LOCAL_PATH_RE.match(ref):
        return "local"

    if "@" not in ref:
        return "bare"

    # Split on the LAST `@` so a ref like ``foo/bar@baz@qux`` (defensive)
    # is classified by the trailing component.
    suffix = ref.rsplit("@", 1)[1]

    if _SHA_RE.match(suffix):
        return "sha"

    # Non-SHA suffix. Distinguish tag from branch heuristically by the
    # shape of the suffix: digits-only / dotted / ``v<n>`` style reads as
    # a tag; anything else (alphanumeric with dashes, ``stable``,
    # ``main``, ``master``, a long branch name) reads as a branch. The
    # heuristic is intentionally narrow -- it never promotes a FAIL to
    # a PASS, only changes the remediation text.
    if re.fullmatch(r"v?\d+(?:\.\d+){0,2}", suffix):
        return "tag"
    return "branch"


# Human-readable remediation hint keyed on the classify() return value.
_REMEDIATION = {
    "tag": (
        "Tags are mutable. Resolve the tag to its current commit SHA via "
        "`git ls-remote https://github.com/<owner>/<repo> refs/tags/<tag>` "
        "and pin with `<sha>  # <tag>` (house style -- trailing version "
        "comment)."
    ),
    "branch": (
        "Branches are mutable. Pin to the branch's current HEAD commit SHA "
        "via `git ls-remote https://github.com/<owner>/<repo> refs/heads/"
        "<branch>`, then record the SHA with a trailing version comment."
    ),
    "bare": (
        "Bare references default to the repo's default branch and are "
        "mutable. Pin to a specific commit SHA via `git ls-remote "
        "https://github.com/<owner>/<repo> HEAD` (then resolve the SHA "
        "to a stable tag)."
    ),
    "empty": "Malformed `uses:` line (empty ref). Add a SHA-pinned action.",
    "unknown": (
        "Unrecognized `uses:` shape. Use a 40-hex SHA (`@<sha>  # <tag>`) "
        "or a local composite path (`./<path>`)."
    ),
}


# ---------------------------------------------------------------------------
# Per-line scanner
# ---------------------------------------------------------------------------
def iter_uses_lines(text: str) -> list[tuple[int, str]]:
    """Yield ``(line_number, ref_string)`` for every active ``uses:``
    line in ``text``.

    A line is *active* iff its first non-whitespace character is not
    ``#``. A line that does not match :data:`_USES_LINE_RE` is silently
    skipped (non-``uses:`` keys, blank lines, ``- if:`` clauses, etc.).
    The returned line numbers are 1-indexed so error messages can be
    copy/pasted into a code review.
    """
    out: list[tuple[int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.lstrip().startswith("#"):
            continue
        m = _USES_LINE_RE.match(raw)
        if not m:
            continue
        out.append((lineno, m.group("ref")))
    return out


# ---------------------------------------------------------------------------
# Per-workflow checker
# ---------------------------------------------------------------------------
def check_workflow(path: Path) -> list[str]:
    """Return a list of drift findings for the workflow at ``path``.

    Each finding is a single ``"<rel>:N: uses: <ref> -- <reason>"`` line
    suitable for ``print``-style output. Empty list means the workflow
    is compliant.

    Findings are returned in line-number order so the FAIL summary in
    ``main()`` is deterministic.
    """
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(REPO_ROOT).as_posix()
    findings: list[str] = []
    for lineno, ref in iter_uses_lines(text):
        kind = classify(ref)
        if kind in {"sha", "local"}:
            continue
        remediation = _REMEDIATION.get(kind, _REMEDIATION["unknown"])
        findings.append(
            f"{rel}:{lineno}: uses: {ref} -- not SHA-pinned "
            f"(classified as {kind!r}). {remediation}"
        )
    return findings


# ---------------------------------------------------------------------------
# Reporting / main
# ---------------------------------------------------------------------------
def _print_summary(file_count: int, ref_count: int, finding_count: int) -> None:
    print(
        f"Scanned {file_count} workflow file(s); verified "
        f"{ref_count} `uses:` ref(s); "
        f"{finding_count} violation(s)."
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return _self_test()

    if not WORKFLOWS_DIR.is_dir():
        print(f"ERROR: {WORKFLOWS_DIR} not found", file=sys.stderr)
        return 2

    files = sorted(WORKFLOWS_DIR.glob("*.yml"))
    if not files:
        print(f"ERROR: no .yml workflows found in {WORKFLOWS_DIR}",
              file=sys.stderr)
        return 2

    all_findings: list[str] = []
    total_refs = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        total_refs += len(iter_uses_lines(text))
        all_findings.extend(check_workflow(path))

    if all_findings:
        print("Non-SHA-pinned `uses:` drift detected:", file=sys.stderr)
        for f in all_findings:
            print(f"  - {f}", file=sys.stderr)
        _print_summary(len(files), total_refs, len(all_findings))
        print(
            "\nRemediation: per docs/SECURITY.md §5, every `uses:` must be a "
            "40-hex commit SHA (`@<sha>  # vN`) or a local composite path "
            "(`./<path>`). Tags and branches are mutable and must not be "
            "used as the resolved ref.",
            file=sys.stderr,
        )
        return 1

    _print_summary(len(files), total_refs, 0)
    print("OK: every `uses:` is SHA-pinned or a local composite path.")
    return 0


# ---------------------------------------------------------------------------
# Self-test (deterministic; no repo state required)
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Build mock workflow fixtures in a tmpdir and assert classify(),
    iter_uses_lines(), check_workflow(), and main() all behave per the
    documented contract. Returns 0 on success, 2 on failure.
    """
    import tempfile

    print("=== check_workflow_pin.py self-test ===")
    failures: list[str] = []

    # --- classify() invariants ---
    expected_classify: dict[str, str] = {
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1": "sha",
        "actions/checkout@3D3C42E5AAC5BA805825DA76410C181273BA90B1": "sha",  # upper-case SHA still valid
        "nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60": "sha",
        "./.github/actions/setup-rust-env": "local",
        "./.github/actions/setup-rust-python-env": "local",
        "./.github/workflows/ci-steps.yml": "local",
        "actions/checkout@v4": "tag",
        "actions/checkout@v4.1.7": "tag",
        "actions/checkout@v4.1.7-rc.1": "branch",  # non-numeric pre-release suffix; still a non-SHA FAIL, classified as branch for the heuristic
        "dtolnay/rust-toolchain@stable": "branch",
        "actions/checkout@main": "branch",
        "actions/checkout@my-feature/foo": "branch",
        "actions/checkout": "bare",
        "": "empty",
        "weird-no-at-symbol-but-long": "bare",
        "owner/repo@abc": "branch",  # short non-SHA suffix => branch
        "owner/repo@deadbeef": "branch",  # 8 hex chars => branch, NOT sha
    }
    for ref, expected in expected_classify.items():
        got = classify(ref)
        if got != expected:
            failures.append(
                f"classify({ref!r}) -> {got!r}, expected {expected!r}"
            )

    # --- iter_uses_lines() invariants ---
    sample = (
        "name: CI\n"
        "on:\n"
        "  push:\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n"
        "      - uses: ./.github/actions/setup-rust-env\n"
        "        with:\n"
        "          key: foo\n"
        "        uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9  # v6.1.0\n"
        "      - if: ${{ github.event_name == 'pull_request' }}\n"
        "        uses: nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60  # v4.0.0\n"
        "#       uses: ./.github/workflows/ci-steps.yml\n"
        "        uses: actions/checkout@v4\n"
    )
    expected_uses = [
        (7, "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"),
        (8, "./.github/actions/setup-rust-env"),
        (11, "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"),
        (13, "nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60"),
        (15, "actions/checkout@v4"),
    ]
    got_uses = iter_uses_lines(sample)
    if got_uses != expected_uses:
        failures.append(
            f"iter_uses_lines() expected {expected_uses}, got {got_uses}"
        )

    # --- check_workflow() compliance + drift ---
    compliant_text = (
        "name: ok\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n"
        "      - uses: ./.github/actions/setup-rust-env\n"
        "      - uses: nick-fields/retry@ad984534de44a9489a53aefd81eb77f87c70dc60  # v4.0.0\n"
    )
    drift_text = (
        "name: drift\n"
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1\n"
        "      - uses: actions/checkout@v4\n"          # tag
        "      - uses: dtolnay/rust-toolchain@stable\n"  # branch
        "      - uses: actions/setup-node\n"            # bare
        "#       uses: ./.github/workflows/ci-steps.yml\n"
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / ".github" / "workflows").mkdir(parents=True)
        compliant = root / ".github" / "workflows" / "ok.yml"
        drift = root / ".github" / "workflows" / "drift.yml"
        compliant.write_text(compliant_text, encoding="utf-8")
        drift.write_text(drift_text, encoding="utf-8")

        # Redirect module-level path constants at the mock repo so main()
        # scans our tmpdir, not the real one.
        import check_workflow_pin as mod
        orig_root = mod.REPO_ROOT
        orig_wf = mod.WORKFLOWS_DIR
        mod.REPO_ROOT = root
        mod.WORKFLOWS_DIR = root / ".github" / "workflows"
        try:
            ok_findings = mod.check_workflow(compliant)
            if ok_findings != []:
                failures.append(
                    f"compliant workflow yielded findings: {ok_findings}"
                )
            drift_findings = mod.check_workflow(drift)
            # Expected: 3 findings (tag, branch, bare); the commented line
            # is skipped; the SHA-pinned line is silent.
            if len(drift_findings) != 3:
                failures.append(
                    f"drift workflow expected 3 findings, got "
                    f"{len(drift_findings)}: {drift_findings}"
                )
            else:
                joined = "\n".join(drift_findings)
                for needle in (
                    "actions/checkout@v4",
                    "dtolnay/rust-toolchain@stable",
                    "actions/setup-node",
                ):
                    if needle not in joined:
                        failures.append(
                            f"drift workflow missing {needle!r} in "
                            f"findings: {joined}"
                        )
                # And the remediation strings must be present.
                for needle in ("mutable",):
                    if joined.count(needle) < 3:
                        failures.append(
                            f"drift findings missing remediation "
                            f"({needle!r} < 3 occurrences): {joined}"
                        )

            # main() exit codes: compliant-only dir -> 0; drift dir -> 1.
            # Pass explicit argv=[] so main() does not see the
            # `--self-test` flag from the outer invocation and recurse
            # back into _self_test() until Python's recursion limit
            # trips.
            compliant_only = root / "only_ok"
            (compliant_only / ".github" / "workflows").mkdir(parents=True)
            (compliant_only / ".github" / "workflows" / "ok.yml").write_text(
                compliant_text, encoding="utf-8"
            )
            mod.REPO_ROOT = compliant_only
            mod.WORKFLOWS_DIR = (
                compliant_only / ".github" / "workflows"
            )
            if mod.main([]) != 0:
                failures.append("main() should exit 0 on compliant-only "
                                "tmpdir")

            drift_only = root / "only_drift"
            (drift_only / ".github" / "workflows").mkdir(parents=True)
            (drift_only / ".github" / "workflows" / "drift.yml").write_text(
                drift_text, encoding="utf-8"
            )
            mod.REPO_ROOT = drift_only
            mod.WORKFLOWS_DIR = drift_only / ".github" / "workflows"
            if mod.main([]) != 1:
                failures.append("main() should exit 1 on drift-only tmpdir")
        finally:
            mod.REPO_ROOT = orig_root
            mod.WORKFLOWS_DIR = orig_wf

    if failures:
        print("FAIL: self-test assertions did not hold:")
        for f in failures:
            print(f"  - {f}")
        return 2

    print(
        f"PASS: {len(expected_classify)} classify() cases, "
        f"{len(expected_uses)} iter_uses_lines() cases, "
        f"compliant + drift workflow round-trips green."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
