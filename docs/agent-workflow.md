# Agent Workflow

> **TL;DR**: Standard 4-phase workflow for implementing issues with cross-agent review and validation gates.
> **Key decisions**: Research before planning | Plan before implement | Validate after implement
> **Owned by**: Wave orchestrator
> **Reviewed**: 2026-07-13

## Overview

Every issue follows a 4-phase workflow. Each phase has explicit entry/exit criteria and checklist items.

## Phase 1 — Research

Understand the issue before writing any code.

### Entry Criteria
- Issue is assigned and triaged
- Branch created from `develop`

### Checklist
- [ ] Read `ARCHITECTURE.md` — understand module boundaries and data flow
- [ ] Run affected code paths — trace inputs/outputs manually
- [ ] Read existing tests for the module — understand test patterns
- [ ] Identify all files that will be modified
- [ ] Note any physics/math that needs verification (use Python to verify, not mental math)

### Exit Criteria
- You can explain what the code does and why it needs to change
- You know where the relevant tests live

---

## Phase 2 — Plan

Write a concrete plan before touching production code.

### Entry Criteria
- Phase 1 complete

### Checklist
- [ ] Write plan to `docs/worksheets/issue-{N}-plan.md`
- [ ] Tag related documentation in `docs/doc-inventory.md`
- [ ] Request cross-agent review (Physics Auditor for physics code, Security Engineer for auth/infrastructure)
- [ ] Address review feedback before proceeding

### Plan Template
```markdown
# Issue {N} Plan

## Problem
{One paragraph description of the issue}

## Approach
{Step-by-step approach}

## Files to Modify
- `src/...`
- `tests/...`

## Validation
- [ ] Unit tests pass
- [ ] Reference tests pass
- [ ] No regression in system tests
```

### Cross-Agent Review Routing
| Issue Type | Reviewer |
|------------|----------|
| Physics/math | `bem-engineer` |
| Security/CVE | `agency-security-engineer` |
| Performance | `agency-performance-benchmarker` |
| API/endpoint | `oma-backend` |
| Frontend/UI | `oma-frontend` |

---

## Phase 3 — Implement

Write code and tests in lockstep.

### Entry Criteria
- Plan approved via cross-agent review

### Checklist
- [ ] Write minimal code to pass the first test
- [ ] Write test covering the fix
- [ ] Run app on every significant change
- [ ] Run affected unit tests after each logical unit of work
- [ ] Keep commits atomic (one logical change per commit)

### Rules
- **No ASHRAE 140 system-level testing** until individual modules pass E+ reference tests
- **No parameter tuning** to make system tests pass — fix the underlying math
- Physics modules must match EnergyPlus within 1% tolerance on isolated scenarios
- Run `npm run lint` / `npm run typecheck` before committing

---

## Phase 4 — Wrap-up

Finalize and prepare for merge.

### Entry Criteria
- All tests pass
- Code reviewed and approved

### Checklist
- [ ] Cross-agent review completed and approved
- [ ] Full test suite passes (`npm test` / `cargo test`)
- [ ] Write end-of-session worksheet to `docs/worksheets/issue-{N}-wrap-up.md`
- [ ] Update `docs/doc-inventory.md` 7-line summaries for modified modules
- [ ] Commit with git tag: `git tag -a issue-{N} -m "Issue {N} complete: {one-line summary}"`
- [ ] Push branch: `git push -u origin fix/issue-{N}-{slug}`
- [ ] Create PR with clear description linking to issue

### PR Description Template
```markdown
## Summary
{What this PR does}

## Testing
- [ ] Unit tests pass
- [ ] Reference data tests pass
- [ ] No ASHRAE 140 regressions (if applicable)

## Checklist
- [ ] Cross-agent review: {reviewer}
- [ ] Documentation updated
- [ ] No debug code or TODO comments left
```

---

## Wave 2 Specific Additions

### Worktree Management
```bash
# Create worktree
git worktree add ../worktrees/issue-{N}-{slug} -b fix/issue-{N}-{slug} develop

# Sync before starting
git pull origin develop

# Commit and push
git add . && git commit -m "docs: resolve #{N} — {title}"
git push -u origin fix/issue-{N}-{slug}
```

### Wave 2 Issue List
| Issue | Title | Priority |
|-------|-------|----------|
| 1531 | agent-workflow.md | high |
| 1533 | agent-conventions.md | high |
| 1534 | scripts/README.md | medium |
