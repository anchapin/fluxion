# Issue #{N} — {Short Title}

**Status:** {OPEN | IN_PROGRESS | COMPLETE | BLOCKED}
**Created:** {YYYY-MM-DD}
**Agent:** {agent-name}
**Branch:** {branch-name}

## Session Meta

| Field | Value |
|-------|-------|
| Created | {YYYY-MM-DD} |
| Agent | {agent-name} |
| Branch | {branch-name} |
| Status | {OPEN} |
| Tags | {tag1, tag2} |

## Problem Statement

What issue is this session addressing? What is the expected vs actual behavior?

## Research Findings

What did you discover during investigation? Include relevant code locations, data, and analysis.

## Plan (approved: yes/no)

- [ ] Step 1 description
- [ ] Step 2 description

*Plan approved by: {name} on {YYYY-MM-DD}*

## Implementation Log

### {YYYY-MM-DD} — {description}

```
{snippet of changes made}
```

## Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| {description} | {impact} | {resolution or TBD} |

## Decisions Made

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| {what was decided} | {why} | {result} |

## Test Results

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| {test name} | {expected} | {actual} | {PASS/FAIL} |

## Wrap-up Checklist

- [ ] All planned changes implemented
- [ ] Tests added/updated and passing
- [ ] Documentation updated (ARCHITECTURE.md, etc.)
- [ ] No regressions in affected modules
- [ ] Reference data comparison complete (if applicable)
- [ ] Branch pushed and PR opened
- [ ] Git tag created: `git tag -a issue-{N} -m "Issue #{N} - {title}"`
