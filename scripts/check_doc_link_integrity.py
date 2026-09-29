#!/usr/bin/env python3
"""Fluxion Doc Link-Integrity Check.

Verifies that path-shaped references inside markdown files actually
resolve on disk. Currently the docs-hygiene gates only verify (a) root
allow-list (b) 7-line summaries (c) docs/doc-inventory.md freshness. They
do NOT verify path references such as `(docs/foo.md)`, `path/to/file`,
or `<path>` resolve on disk.

A checked reference is one of:

1. Markdown link text inside `[..](PATH)` — `PATH` is the reference.
2. Path inside angle brackets `<PATH>` — `PATH` is the reference.
3. `cargo test --test <target>` commands inside code fences — `<target>` is
   checked against the `[[test]]` targets declared in `Cargo.toml`
   (Issue #4199: issue #3764 consolidated the standalone test binaries into
   the single `all_tests` runner, so doc commands naming the old binaries
   abort with `error: no test target named 'X'`).
4. References to `.github/workflows/*.yml` paths — verified to exist on disk
   (Issue #4184: prevents doc rot where citations point at deleted workflows).
   Matches backticked and bare tokens in prose; fenced code blocks are
   excluded (they hold example commands like `ci-local.sh
   .github/workflows/foo.yml`). Known proposal-record exceptions are listed
   in WORKFLOW_PATH_REF_EXCEPTIONS.

Bare path-shaped tokens are NOT heuristically matched (this avoids false
positives on code-block-like text such as `dyn Trait` or `release_gates.yaml`
embedded in prose). Use explicit markdown-link or angle-bracket syntax
for cross-references inside markdown.

For each reference, the script tries (in order):

1. `os.path.join(REPO_ROOT, ref_path)` — relative to repo root.
2. The reference path resolved relative to the file's own directory.

References that fail BOTH lookups and are not URLs (http/https) are
counted as failures.

The script exits 0 when no failures are detected, 1 otherwise.

Targets the AGENTS.md allow-listed root docs and all `docs/**/*.md`
files. Wired into `.github/workflows/docs-hygiene.yml` as a new
"Run doc-link-integrity check" step (see #2885).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 1. Markdown link: [text](PATH)
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# 2. Angle-bracket path: <PATH>
ANGLE_RE = re.compile(r"<([^>]+)>")

# 3. `cargo test --test <target>` command in a code fence (Issue #4199).
# `--test` may be preceded by other flags (e.g. `--features x -p fluxion`),
# so match it anywhere on the line. Target names are alphanumeric +
# underscore + hyphen; placeholders like `<name>` are never matched.
CARGO_TEST_TARGET_RE = re.compile(r"--test\s+([a-zA-Z0-9_\-]+)")

# Directory holding the per-module sources consolidated into the
# single `all_tests` runner by issue #3764.
ALL_TESTS_DIR = REPO_ROOT / "tests" / "all_tests"

# Directory holding the CI workflow files cited by docs.
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# 4. Inline `.github/workflows/<name>.yml` references in prose (Issue #4184).
# Matches backticked and bare tokens; fenced code blocks are stripped before
# matching so example commands (e.g. `ci-local.sh .github/workflows/foo.yml`)
# are not treated as citations.
WORKFLOW_PATH_RE = re.compile(r"\.github/workflows/([A-Za-z0-9_.\-]+\.ya?ml)")

# Issue #4184 — (repo-relative markdown file, workflow filename) pairs that
# name a workflow file which does not exist on disk but are NOT doc rot:
#   - docs/ripr_investigation_1254.md names `ripr.yml` as a PROPOSAL inside a
#     historical investigation record (the proposal later landed as
#     `ripr-preflight.yml`); editing the record would falsify history.
#   - .planning/** names `validation.yml` in historical plan/research notes;
#     those docs describe intended wiring, not current CI state.
WORKFLOW_PATH_REF_EXCEPTIONS = frozenset(
    {
        ("docs/ripr_investigation_1254.md", "ripr.yml"),
        (
            ".planning/phases/47-performance-validation-optimization/47-03-PLAN.md",
            "validation.yml",
        ),
        (".planning/research/ARCHITECTURE.md", "validation.yml"),
    }
)

# Files in scope: AGENTS.md allow-listed root docs + docs/**/*.md
ROOT_ALLOW = (
    "README.md", "ARCHITECTURE.md", "CODEBASE_MAP.md", "CONTRIBUTING.md",
    "RULES.md", "CHANGELOG.md", "AGENTS.md", "SCORECARD.md",
)


# Issue #3808 — `.planning/worktrees/**` is the wave orchestrator's
# live git worktree store (gitignored runtime state). It is never
# present in CI but exists during parallel-agent local runs and
# contains markdown files with cross-references resolved against the
# worktree root rather than the repo root, which produces false
# "broken reference" failures. Skip the directory at collection time.
WORKTREE_SKIP_PARTS = (REPO_ROOT / ".planning" / "worktrees",)


def is_skipped_worktree(path: Path) -> bool:
    """Return True if `path` lives under any gitignored runtime
    worktree directory (Issue #3808)."""
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    parts = rel.parts
    return any(
        REPO_ROOT.joinpath(*parts[: i + 1]) in WORKTREE_SKIP_PARTS
        for i in range(len(parts))
    )


def load_cargo_test_targets() -> set[str]:
    """Return every `[[test]]` target name declared in the root Cargo.toml."""
    targets: set[str] = set()
    cargo_toml = REPO_ROOT / "Cargo.toml"
    if not cargo_toml.is_file():
        return targets
    in_test_section = False
    for line in cargo_toml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[[test]]":
            in_test_section = True
        elif stripped.startswith("[["):
            in_test_section = False
        elif in_test_section:
            match = re.match(r'name\s*=\s*"([^"]+)"', stripped)
            if match:
                targets.add(match.group(1))
                in_test_section = False
    return targets


def extract_cargo_test_targets(text: str) -> list[tuple[str, int]]:
    """Return `(target, line_no)` for every `--test <target>` found inside
    fenced code blocks in `text` (Issue #4199)."""
    found: list[tuple[str, int]] = []
    in_fence = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            for match in CARGO_TEST_TARGET_RE.finditer(line):
                found.append((match.group(1), line_no))
    return found


def is_consolidation_drift_target(target: str, valid_targets: set[str]) -> bool:
    """Return True when `target` is the issue #3764 consolidation-drift
    class: not a declared `[[test]]` target, but a module of that name
    exists under `tests/all_tests/` — i.e. a doc command naming a
    standalone binary that was merged into the `all_tests` runner.
    Names that are neither declared targets nor consolidated modules
    (placeholders, filters, other-crate targets) are NOT this class."""
    return target not in valid_targets and (ALL_TESTS_DIR / f"{target}.rs").is_file()


def extract_workflow_path_refs(text: str) -> list[tuple[str, int]]:
    """Return `(workflow_filename, line_no)` for every
    `.github/workflows/<name>.yml` token in prose `text` (Issue #4184).
    Fenced code blocks are stripped first, so example commands are not
    treated as citations."""
    found: list[tuple[str, int]] = []
    cleaned = strip_code_fences(text)
    for line_no, line in enumerate(cleaned.splitlines(), start=1):
        for match in WORKFLOW_PATH_RE.finditer(line):
            found.append((match.group(1), line_no))
    return found


def collect_markdown_files() -> list[Path]:
    """Return all in-scope markdown files."""
    files: list[Path] = []
    for name in ROOT_ALLOW:
        path = REPO_ROOT / name
        if path.is_file():
            files.append(path)
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.is_dir():
        for path in sorted(docs_dir.rglob("*.md")):
            files.append(path)
    # Include .planning/**/*.md for cross-references from CHANGELOG/AGENTS/etc.
    # Issue #3808 — but skip `.planning/worktrees/**` (parallel-agent
    # git worktrees; see WORKTREE_SKIP_PARTS above).
    planning_dir = REPO_ROOT / ".planning"
    if planning_dir.is_dir():
        for path in sorted(planning_dir.rglob("*.md")):
            if is_skipped_worktree(path):
                continue
            files.append(path)
    return files


def strip_code_fences(text: str) -> str:
    """Remove fenced code blocks from `text` so links inside code are
    not treated as references."""
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")  # blank out fence markers
            continue
        if in_fence:
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def extract_references(text: str, file_path: Path) -> list[tuple[str, int]]:
    """Return list of (reference, line_number) extracted from `text`."""
    refs: list[tuple[str, int]] = []
    cleaned = strip_code_fences(text)
    for i, line in enumerate(cleaned.splitlines(), 1):
        for match in MD_LINK_RE.finditer(line):
            refs.append((match.group(1), i))
        for match in ANGLE_RE.finditer(line):
            refs.append((match.group(1), i))
    return refs


def looks_like_path(ref: str) -> bool:
    """Heuristic: does `ref` look like a filesystem path rather than
    Rust generic syntax, inline code, or prose?"""
    if not ref or any(ch.isspace() for ch in ref):
        return False
    if "::" in ref:
        return False  # Rust path separator
    if "," in ref or ";" in ref:
        return False  # generic param lists
    # Must end in a known extension OR have at least one path separator
    exts = (".md", ".rs", ".toml", ".yml", ".yaml", ".sh", ".py", ".json", ".txt", ".csv")
    if any(ref.endswith(ext) for ext in exts):
        return True
    if "/" in ref or ref.startswith("./") or ref.startswith("../"):
        return True
    return False


def is_external_ref(ref: str) -> bool:
    """Return True if ref is not a relative path (URL or anchor)."""
    if ref.startswith(("http://", "https://", "ftp://", "mailto:", "#")):
        return True
    if ref.startswith(("/", "//")):
        return True
    return False


def resolve_reference(ref: str, file_path: Path) -> bool:
    """Return True if `ref` resolves to an existing file."""
    # Strip any anchor / query
    ref_clean = ref.split("#")[0].split("?")[0]
    if not ref_clean:
        return True  # pure anchor

    # Skip absolute URLs that slipped through
    if ref_clean.startswith(("/", "//", "http", "https", "ftp", "mailto")):
        return True

    # 1. Relative to repo root
    candidate_root = REPO_ROOT / ref_clean
    if candidate_root.exists():
        return True

    # 2. Relative to the file's directory
    candidate_local = file_path.parent / ref_clean
    if candidate_local.exists():
        return True

    return False


def main() -> int:
    files = collect_markdown_files()
    if not files:
        sys.stderr.write("::error::No markdown files found in scope\n")
        return 1

    total_refs = 0
    failures: list[tuple[Path, int, str]] = []
    # Issue #4199: (target, file, line) for cargo test targets in doc code
    # fences that name a consolidated-away standalone binary.
    drift_failures: list[tuple[str, Path, int]] = []
    drift_warnings: list[tuple[str, Path, int]] = []
    # Issue #4184: (file, line, workflow filename) for prose references to
    # `.github/workflows/*.yml` paths that do not exist on disk.
    workflow_failures: list[tuple[Path, int, str]] = []
    valid_test_targets = load_cargo_test_targets()
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            sys.stderr.write(f"::warning::could not read {file_path}: {exc}\n")
            continue
        rel_posix = file_path.relative_to(REPO_ROOT).as_posix()
        for wf_name, wf_line in extract_workflow_path_refs(text):
            if (rel_posix, wf_name) in WORKFLOW_PATH_REF_EXCEPTIONS:
                continue
            if not (WORKFLOWS_DIR / wf_name).is_file():
                workflow_failures.append((file_path, wf_line, wf_name))
        for target, line_no in extract_cargo_test_targets(text):
            if target in valid_test_targets:
                continue
            if is_consolidation_drift_target(target, valid_test_targets):
                drift_failures.append((target, file_path, line_no))
            else:
                # Placeholder, filter, or other-crate target: warn, don't fail.
                drift_warnings.append((target, file_path, line_no))
        for ref, line_no in extract_references(text, file_path):
            if is_external_ref(ref):
                continue
            if not looks_like_path(ref):
                continue
            total_refs += 1
            if not resolve_reference(ref, file_path):
                failures.append((file_path, line_no, ref))

    print("=== Fluxion Doc Link-Integrity Check ===")
    print(f"Repo: {REPO_ROOT}")
    print(f"Files scanned: {len(files)}")
    print(f"References checked: {total_refs}")
    print(f"Failures: {len(failures)}")

    if failures:
        print()
        for file_path, line_no, ref in failures:
            rel = file_path.relative_to(REPO_ROOT)
            print(f"  {rel}:{line_no}: {ref}")
        print()
        print(f"FAIL: {len(failures)} broken doc reference(s) detected.")

    if drift_warnings:
        print()
        print("Warnings: --test targets that are neither declared [[test]] "
              "targets nor consolidated all_tests modules (not failures):")
        for target, file_path, line_no in drift_warnings:
            rel = file_path.relative_to(REPO_ROOT)
            print(f"  {rel}:{line_no}: --test {target}")

    if drift_failures:
        print()
        for target, file_path, line_no in drift_failures:
            rel = file_path.relative_to(REPO_ROOT)
            print(f"  {rel}:{line_no}: --test {target} "
                  f"(consolidated into all_tests; use "
                  f"--test all_tests {target}::)")
        print()
        print(f"FAIL: {len(drift_failures)} stale cargo test target(s) "
              f"detected (issue #4199).")
        return 1

    if workflow_failures:
        print()
        print("Doc references to .github/workflows/*.yml paths that do not "
              "exist on disk (issue #4184):")
        for file_path, line_no, wf_name in workflow_failures:
            rel = file_path.relative_to(REPO_ROOT)
            print(f"  {rel}:{line_no}: .github/workflows/{wf_name}")
        print()
        print(f"FAIL: {len(workflow_failures)} doc reference(s) to missing "
              f"workflow file(s) detected.")
        return 1

    if failures:
        return 1

    print()
    print("PASS: All doc references resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())