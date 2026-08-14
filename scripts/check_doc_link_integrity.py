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

# Files in scope: AGENTS.md allow-listed root docs + docs/**/*.md
ROOT_ALLOW = (
    "README.md", "ARCHITECTURE.md", "CODEBASE_MAP.md", "CONTRIBUTING.md",
    "RULES.md", "CHANGELOG.md", "AGENTS.md", "SCORECARD.md",
)


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
    planning_dir = REPO_ROOT / ".planning"
    if planning_dir.is_dir():
        for path in sorted(planning_dir.rglob("*.md")):
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
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            sys.stderr.write(f"::warning::could not read {file_path}: {exc}\n")
            continue
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
        return 1

    print()
    print("PASS: All doc references resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())