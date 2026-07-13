# Issue #{N} — {Short Title}

> **TL;DR**: One sentence describing what this worksheet is for.
> **Owned by**: {owner}
> **Reviewed**: {YYYY-MM-DD}

**Status:** {OPEN | IN_PROGRESS | COMPLETE | BLOCKED}
**Created:** {YYYY-MM-DD}
**Agent:** {agent-name}
**Branch:** {branch-name}

---

## Session Meta

| Field | Value |
|-------|-------|
| Issue Number | #{N} |
| Title | {Short Title} |
| Created | {YYYY-MM-DD} |
| Agent | {agent-name} |
| Branch | {branch-name} |
| Status | {OPEN} |
| Tags | {tag1, tag2} |
| Related Issues | #{N1}, #{N2} |

---

## Problem Statement

What issue is this session addressing? What is the expected vs actual behavior?

**Expected behavior:**
> Describe what should happen

**Actual behavior:**
> Describe what actually happens

**Impact:**
> Describe the impact of this issue

---

## Research Findings

What did you discover during investigation? Include relevant code locations, data, and analysis.

### Code Locations

| File | Function/Struct | Relevant Finding |
|------|------------------|-----------------|
| {path} | {name} | {finding} |

### Data Analysis

```
{relevant data, logs, test outputs}
```

### Root Cause Hypothesis

> Hypothesis: {description}

---

## Plan (approved: yes/no)

- [ ] Step 1 description
- [ ] Step 2 description
- [ ] Step 3 description

*Plan approved by: {name} on {YYYY-MM-DD}*

---

## Implementation Log

### {YYYY-MM-DD} — {description}

**Changes:**

```diff
{diff of changes made}
```

**Files modified:**
- {file1}
- {file2}

**Tests added/updated:**
- {test name}: {description}

---

## Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| {description} | {impact} | {resolution or TBD} |

---

## Decisions Made

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| {what was decided} | {why} | {result} |

---

## Test Results

| Test | Module | Expected | Actual | Status |
|------|--------|----------|--------|--------|
| {test name} | {module} | {expected} | {actual} | PASS/FAIL |

### Validation Results

> Describe any EnergyPlus reference comparisons or other validation performed

---

## Wrap-up Checklist

- [ ] All planned changes implemented
- [ ] Tests added/updated and passing
- [ ] Documentation updated (ARCHITECTURE.md, etc.)
- [ ] No regressions in affected modules
- [ ] Reference data comparison complete (if applicable)
- [ ] Branch pushed and PR opened
- [ ] Git tag created: `git tag -a issue-{N} -m "Issue #{N} - {title}"`

---

## Notes

Additional notes and observations from this session.

