# Worksheets

> **TL;DR**: Index of all agent worksheets organized by issue and tag.
> **Owned by**: Fluxion team
> **Reviewed**: 2026-07-13

This directory contains session worksheets used by agents to document their investigation, planning, and implementation work.

## By Issue

| Issue | Worksheet | Description |
|-------|-----------|-------------|
| #1532 | [issue-template.md](./issue-template.md) | Session template for agent worksheets |
| #1541 | [doc-inventory.md](../doc-inventory.md) | This inventory — self-healing docs catalog with 7-line summary convention |

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
| linter | [docs/linter-rules.md](../../docs/linter-rules.md) |
| testing | [scripts/audit_false_confidence.py](../../scripts/audit_false_confidence.py) |

## Worksheet Structure

Each worksheet follows the [issue-template.md](./issue-template.md) structure:

1. **Session Meta** — Issue number, title, agent, branch, tags
2. **Problem Statement** — Expected vs actual behavior, impact
3. **Research Findings** — Code locations, data analysis, root cause hypothesis
4. **Plan** — Step-by-step implementation plan with approval
5. **Implementation Log** — Chronological log of changes made
6. **Blockers** — Current blockers with impact and resolution
7. **Decisions Made** — Architectural decisions with rationale and outcome
8. **Test Results** — Test comparisons against expected values
9. **Wrap-up Checklist** — Final checklist before closing issue

## Git Tagging Convention

Tag worksheets when a phase completes:

```bash
git tag -a issue-{N} -m "Issue #{N} - {title}"
```

Example:
```bash
git tag -a issue-703 -m "Issue #703 - 900-Series Peak Cooling Root Cause"
```

## Creating a New Worksheet

1. Copy `issue-template.md` to a new file named after the issue
2. Fill in the Session Meta section
3. Update this README to include the new worksheet
4. Tag the worksheet when complete

