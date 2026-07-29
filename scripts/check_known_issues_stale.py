#!/usr/bin/env python3
# scripts/check_known_issues_stale.py
#
# Check that docs/KNOWN_ISSUES.md has been updated within 60 days.
# Run as a CI gate on every PR and main push.
#
# Exit codes:
#   0 — Last Updated is within 60 days (or file/date not found)
#   1 — Last Updated is stale (>60 days old)

import re
import sys
from datetime import date, timedelta

KNOWN_ISSUES_PATH = "docs/KNOWN_ISSUES.md"
STALE_THRESHOLD_DAYS = 60

def main() -> int:
    path = KNOWN_ISSUES_PATH
    threshold = timedelta(days=STALE_THRESHOLD_DAYS)
    cutoff = date.today() - threshold

    try:
        content = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        # If the file doesn't exist, skip the check (not a failure)
        print(f"{path} not found — skipping stale check")
        return 0

    # Match "*Last Updated: YYYY-MM-DD*" (with optional italics markers)
    m = re.search(r"\*Last Updated:\s*(\d{4}-\d{2}-\d{2})\*", content)
    if not m:
        print(f"WARN: Could not find '*Last Updated: YYYY-MM-DD*' in {path}")
        return 0

    last_updated = date.fromisoformat(m.group(1))

    if last_updated >= cutoff:
        print(f"OK  {path} Last Updated: {last_updated} (within {STALE_THRESHOLD_DAYS}-day threshold)")
        return 0
    else:
        age = (date.today() - last_updated).days
        print(f"FAIL: {path} Last Updated: {last_updated} is {age} days old (threshold: {STALE_THRESHOLD_DAYS} days)")
        print(f"  Update the '*Last Updated:*' line in {path}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
