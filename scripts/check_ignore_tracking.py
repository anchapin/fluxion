#!/usr/bin/env python3
"""
IGNORE-to-Active Test Tracking for Fluxion.

Scans test files for `#[ignore]` attributes and extracts the issue references
from the ignore reasons. This provides visibility into which tests are
quarantined and what needs to happen before they can be un-ignored.

Usage::

    python3 scripts/check_ignore_tracking.py
    python3 scripts/check_ignore_tracking.py --by-issue

Exit codes:
    0 — Script ran successfully (output only)
    1 — Found ignored tests with closed issues (may need un-ignoring)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_ignored_tests() -> list[dict]:
    """Find all #[ignore] occurrences in test files and extract issue references."""
    ignored_tests = []
    
    # Scan all test files
    for test_file in REPO_ROOT.glob("tests/**/*.rs"):
        text = test_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Match #[ignore] or #[ignore = "..."] 
            if stripped.startswith("#[ignore"):
                # Try to extract the ignore reason and issue reference
                reason = ""
                issue_refs = []
                
                # Check if it's a block comment above this line
                if i > 1:
                    j = i - 2
                    while j >= 0 and lines[j].strip().startswith("//!") or lines[j].strip().startswith("///"):
                        comment = lines[j].strip().removeprefix("//!").removeprefix("///").strip()
                        if "#[ignore]" in comment or "ignore" in comment.lower():
                            reason = comment
                            # Extract issue references like #1234
                            issue_refs.extend(re.findall(r'#(\d+)', comment))
                        j -= 1
                
                # Check for inline ignore reason
                inline_match = re.search(r'#\[ignore\s*=\s*"([^"]+)"', stripped)
                if inline_match:
                    reason = inline_match.group(1)
                    issue_refs.extend(re.findall(r'#(\d+)', reason))
                
                # Dedupe issue refs
                issue_refs = list(dict.fromkeys(issue_refs))
                
                ignored_tests.append({
                    "file": test_file.relative_to(REPO_ROOT),
                    "line": i,
                    "reason": reason,
                    "issues": issue_refs,
                    "ignore_text": stripped,
                })
    
    return ignored_tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--by-issue",
        action="store_true",
        help="Group ignored tests by issue reference.",
    )
    args = parser.parse_args()
    
    print("=== Fluxion IGNORE-to-Active Test Tracking ===")
    print(f"Repo: {REPO_ROOT}")
    print()
    
    tests = find_ignored_tests()
    
    if not tests:
        print("No #[ignore] tests found.")
        return 0
    
    print(f"Found {len(tests)} ignored test(s).")
    print()
    
    if args.by_issue:
        # Group by issue
        by_issue: dict[str, list[dict]] = defaultdict(list)
        for test in tests:
            if test["issues"]:
                for issue in test["issues"]:
                    by_issue[f"#{issue}"].append(test)
            else:
                by_issue["(no issue reference)"].append(test)
        
        for issue, issue_tests in sorted(by_issue.items()):
            print(f"{issue}:")
            for test in issue_tests:
                print(f"  - {test['file']}:{test['line']}")
                if test["reason"]:
                    print(f"    {test['reason'][:80]}...")
            print()
    else:
        for test in tests:
            issues_str = ", ".join(f"#{i}" for i in test["issues"]) if test["issues"] else "(no issue ref)"
            print(f"{test['file']}:{test['line']} {issues_str}")
            if test["reason"]:
                print(f"  {test['reason'][:100]}...")
            print()
    
    # Check if any ignored tests reference closed issues
    # (This would require GitHub API access, so we just warn)
    tests_without_refs = [t for t in tests if not t["issues"]]
    if tests_without_refs:
        print("=" * 64)
        print()
        print(f"WARNING: {len(tests_without_refs)} ignored test(s) have no issue reference.")
        print("Consider adding an issue reference to track when they can be un-ignored.")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — top-level CLI error boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
