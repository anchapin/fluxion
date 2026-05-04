#!/usr/bin/env python3
"""Fast Repository Cleanup Script"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024

# Fast patterns (just check top-level dirs)
SKIP_DIRS = {
    "node_modules",
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".coverage",
    "coverage.json",
    "htmlcov",
}


def get_file_hash(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read(8192)).hexdigest()
    except (OSError, IOError):
        return ""


def quick_scan():
    """Fast scan using os.walk instead of rglob"""
    duplicates = defaultdict(list)
    seen_hashes = {}
    unnecessary = defaultdict(list)
    empty_files = []
    empty_dirs = []
    suspicious = []
    missing_gitignore = []

    # Quick top-level scan for suspicious dirs
    for item in REPO_ROOT.iterdir():
        if item.is_dir():
            name = item.name
            if len(name) > 20 and not name.startswith(".") and name not in SKIP_DIRS:
                suspicious.append(str(item))

    # Check .gitignore
    gitignore = REPO_ROOT / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        for pattern in [
            "__pycache__/",
            "*.py[cod]",
            ".pytest_cache/",
            ".ruff_cache/",
            "node_modules/",
            "nohup.out",
        ]:
            if pattern.rstrip("/") not in content:
                missing_gitignore.append(pattern)

    # Walk tree
    for root, dirs, files in os.walk(REPO_ROOT):
        root_path = Path(root)
        rel = root_path.relative_to(REPO_ROOT)

        # Prune skip dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        # Check for unnecessary dirs
        for d in dirs:
            if (
                d.startswith("__pycache__")
                or d.startswith(".pytest")
                or d.startswith(".ruff")
                or d.startswith(".mypy")
            ):
                unnecessary["cache_dirs"].append(str(root_path / d))

        # Check files
        for f in files:
            fp = root_path / f
            try:
                stat = fp.stat()
                size = stat.st_size

                # Empty files
                if size == 0 and f not in {".gitkeep"}:
                    empty_files.append(str(fp))

                # Temp files
                elif f in {".coverage", "nohup.out"}:
                    unnecessary["temp_files"].append(str(fp))

                # Large files
                elif size > LARGE_FILE_THRESHOLD:
                    unnecessary["large_files"].append(str(fp))

                # Duplicates (only small files)
                elif size > 0 and size < 1024 * 1024:  # Only < 1MB
                    h = get_file_hash(fp)
                    if h:
                        if h in seen_hashes:
                            duplicates[h].append(str(fp))
                        else:
                            seen_hashes[h] = str(fp)
            except OSError:
                pass

        # Empty dirs
        if not files and not dirs:
            dir_rel = rel.relative_to(rel) if rel != Path(".") else Path(".")
            if str(dir_rel) not in SKIP_DIRS:
                empty_dirs.append(str(root_path))

    # Filter duplicates
    duplicates = {h: paths for h, paths in duplicates.items() if len(paths) > 1}

    return (
        duplicates,
        unnecessary,
        empty_files,
        empty_dirs,
        suspicious,
        missing_gitignore,
    )


def generate_report(
    duplicates, unnecessary, empty_files, empty_dirs, suspicious, missing_gitignore
) -> str:
    lines = [
        f"# Repository Cleanup Report - {REPO_ROOT.name}\n",
        f"Generated: {datetime.now().isoformat()}\n",
    ]

    total = (
        len(duplicates)
        + len(unnecessary.get("cache_dirs", []))
        + len(unnecessary.get("temp_files", []))
        + len(empty_files)
        + len(empty_dirs)
        + len(suspicious)
    )

    lines.append("## Summary\n")
    lines.append(f"- Duplicate file groups: {len(duplicates)}")
    lines.append(f"- Cache directories: {len(unnecessary.get('cache_dirs', []))}")
    lines.append(f"- Temp files: {len(unnecessary.get('temp_files', []))}")
    lines.append(f"- Empty files: {len(empty_files)}")
    lines.append(f"- Empty directories: {len(empty_dirs)}")
    lines.append(f"- Suspicious directories: {len(suspicious)}")
    lines.append(f"- Missing .gitignore entries: {len(missing_gitignore)}")
    lines.append(f"- **Total: {total}**\n")

    if duplicates:
        lines.append("## Duplicate Files\n")
        for h, paths in list(duplicates.items())[:10]:
            lines.append(f"### Hash: {h[:12]}...")
            for p in paths[:5]:
                lines.append(f"- `{Path(p).relative_to(REPO_ROOT)}`")
            if len(paths) > 5:
                lines.append(f"- ... and {len(paths) - 5} more")
            lines.append("")

    if unnecessary:
        lines.append("## Unnecessary Items\n")
        for cat in ["cache_dirs", "temp_files", "large_files"]:
            items = unnecessary.get(cat, [])
            if items:
                lines.append(f"### {cat.replace('_', ' ').title()}")
                for item in items[:30]:
                    lines.append(f"- `{Path(item).relative_to(REPO_ROOT)}`")
                if len(items) > 30:
                    lines.append(f"- ... and {len(items) - 30} more")
                lines.append("")

    if empty_files:
        lines.append("## Empty Files\n")
        for item in empty_files[:30]:
            lines.append(f"- `{Path(item).relative_to(REPO_ROOT)}`")
        if len(empty_files) > 30:
            lines.append(f"- ... and {len(empty_files) - 30} more")
        lines.append("")

    if empty_dirs:
        lines.append("## Empty Directories\n")
        for item in empty_dirs[:30]:
            lines.append(f"- `{Path(item).relative_to(REPO_ROOT)}/`")
        if len(empty_dirs) > 30:
            lines.append(f"- ... and {len(empty_dirs) - 30} more")
        lines.append("")

    if suspicious:
        lines.append("## Suspicious Directories\n")
        for d in suspicious:
            lines.append(f"- `{Path(d).relative_to(REPO_ROOT)}/`")
        lines.append("")

    if missing_gitignore:
        lines.append("## Missing .gitignore Entries\n")
        for item in missing_gitignore:
            lines.append(f"- `{item}`")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("-o", "--output", help="Output report to file")
    args = parser.parse_args()

    print("Scanning repository (fast mode)...")

    duplicates, unnecessary, empty_files, empty_dirs, suspicious, missing_gitignore = (
        quick_scan()
    )
    report = generate_report(
        duplicates, unnecessary, empty_files, empty_dirs, suspicious, missing_gitignore
    )

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    total = (
        len(duplicates)
        + len(unnecessary.get("cache_dirs", []))
        + len(unnecessary.get("temp_files", []))
        + len(empty_files)
        + len(empty_dirs)
        + len(suspicious)
    )

    if total == 0:
        print("No cleanup issues found.")
        sys.exit(0)

    sys.exit(0 if args.report_only else 1)
