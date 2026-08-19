#!/usr/bin/env python3
"""Write the KNOWN_ISSUES.md staleness issue body to a file.

Invoked by the `check-stale` job in
`.github/workflows/known-issues-stale.yml`. The bash heredoc that used to
embed this body (`<<'ISSUE_BODY'`) failed to parse on every workflow run
because the YAML multi-line string preserves indentation, but bash's
quoted heredoc terminator (`<<'ISSUE_BODY'`, no dash) requires the closing
delimiter at column 0. The same root cause was fixed for the determinism
listener (`post_determinism_gate_comment.py`); this script extends the
established pattern to the known-issues-stale workflow.

See: https://github.com/anchapin/fluxion/issues/3117
"""
import os
import sys

OUTPUT_FILE = os.environ["OUTPUT_FILE"]

BODY = """## KNOWN_ISSUES.md Staleness Alert

`docs/KNOWN_ISSUES.md` is more than **45 days** old and approaching the 60-day CI gate.

### Action Required

1. Review each section of `docs/KNOWN_ISSUES.md` and update the Last Updated date and section notes as needed
2. Run the refresh script: `bash scripts/refresh_known_issues.sh`
3. Open a PR with the refreshed date and a brief summary of any section updates

### Refresh Checklist

- [ ] **Foundation Issues (BASE)** — verify all BASE-0x entries still reflect current state
- [ ] **Solar Issues (SOLAR)** — verify SOLAR-0x entries and update if new issues discovered
- [ ] **Free-Floating Temperature Issues (FREE)** — verify FREE-0x entries
- [ ] **Temperature Issues (TEMP)** — verify TEMP-0x entries
- [ ] **Multi-Zone Issues (MULTI)** — verify MULTI-0x entries
- [ ] **5R1C Model Limitations (LIMIT)** — verify LIMIT-0x entries, especially LIMIT-05
- [ ] Update `*Last Updated: YYYY-MM-DD*` at the top of the file

### Reference

- Staleness check: `scripts/check_known_issues_stale.py` (60-day CI gate)
- Refresh script: `scripts/refresh_known_issues.sh`
- CI gate fails at 60 days; this issue was created at 45 days to allow time for review
"""

with open(OUTPUT_FILE, "w") as f:
    f.write(BODY)
