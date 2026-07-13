# Worksheets

Index of all worksheets organized by issue/tag.

## By Issue

| Issue | Worksheet | Description |
|-------|-----------|-------------|
| #1532 | [issue-template.md](./issue-template.md) | Session template for agent worksheets |
| #1541 | doc-inventory | This inventory — self-healing docs catalog with 7-line summary convention |

## By Tag

| Tag | Worksheets |
|-----|------------|
| architecture | [ARCHITECTURE.md](../../ARCHITECTURE.md) |
| validation | [validation_report.md](../../validation_report.md), [ASHRAE140_VALIDATION.md](../../ASHRAE140_VALIDATION.md) |
| performance | [documentation/performance.md](../../documentation/performance.md), [documentation/performance_guide.md](../../documentation/performance_guide.md) |
| contributing | [CONTRIBUTING.md](../../CONTRIBUTING.md) |
| rules | [RULES.md](../../RULES.md) |
| codebase | [CODEBASE_MAP.md](../../CODEBASE_MAP.md) |
| issues | [docs/KNOWN_ISSUES.md](../../docs/KNOWN_ISSUES.md) |
| fix | [FIX.md](../../FIX.md) |
| process | [issue-template.md](./issue-template.md) |

## Git Tagging Convention

Tag worksheets when a phase completes:

```bash
git tag -a issue-{N} -m "Issue #{N} - {title}"
```

Example:
```bash
git tag -a issue-703 -m "Issue #703 - 900-Series Peak Cooling Root Cause"
```
