# Agent Workflow

> **TL;DR**: Standard 4-phase workflow for resolving issues with research, planning, implementation, and wrap-up.
> **Key decisions**: Always read ARCHITECTURE.md first | Use ctx tools for GATHER/FOLLOW-UP | Write worksheet for complex issues
> **Owned by**: All agents
> **Reviewed**: 2026-07-13

## 4-Phase Workflow

### Phase 1 — Research

Understand the issue. Read `ARCHITECTURE.md`. Run affected code paths. Read tests.

- Read `ARCHITECTURE.md` for module boundaries and I/O contracts
- Read relevant test files to understand expected behavior
- Run the affected code path to observe the issue
- Document findings in `docs/worksheets/issue-{N}-plan.md`

### Phase 2 — Plan

Write the plan to `docs/worksheets/issue-{N}-plan.md`. Tag related docs. Get cross-agent review.

- Create worksheet at `docs/worksheets/issue-{N}-{slug}.md`
- Tag related architecture docs
- Get cross-agent review from relevant personas
- Get explicit approval before implementing

### Phase 3 — Implement

Write code + tests in lockstep. Run app on every significant change.

- Write the minimal viable change
- Add/update tests alongside implementation
- Run tests on every significant change
- Use Python for mathematical verification

### Phase 4 — Wrap-up

Cross-agent review. Run full test suite. Write end-of-session worksheet. Commit with git tag.

- Run `pr-review-merge` for multi-model review
- Run full test suite
- Update worksheet to "complete"
- Commit with git tag: `git tag -a issue-{N} -m "Issue #{N} — {title}"`

## Per-Phase Checklist

### Phase 1 Checklist
- [ ] Read `ARCHITECTURE.md`
- [ ] Read relevant test files
- [ ] Ran affected code path
- [ ] Identified root cause
- [ ] Created worksheet

### Phase 2 Checklist
- [ ] Plan written in worksheet
- [ ] Related docs tagged
- [ ] Cross-agent review obtained
- [ ] Approval received

### Phase 3 Checklist
- [ ] Code written
- [ ] Tests added/updated
- [ ] Tests passing
- [ ] Mathematical verification done

### Phase 4 Checklist
- [ ] Cross-agent review done
- [ ] Full test suite passing
- [ ] Worksheet updated
- [ ] Git tag applied

## Cross-Agent Review Routing

| Milestone | Reviewer Persona | Scope |
|-----------|-----------------|-------|
| Research | BEM Domain Expert | Issue understanding, ARCHITECTURE.md alignment |
| Plan | Physics Auditor + Integration Reviewer | Plan soundness, module boundaries |
| Implementation | Code Quality Auditor + ML Surrogate Reviewer | Code correctness, contract adherence |
| Wrap-up | Safety Engineer + Performance Reviewer | Full validation, no regressions |

## Related Docs

- Worksheet template: `@/docs/worksheets/issue-template.md`
- Cross-agent review guide: `@/docs/agent-review-guide.md`
- Coding conventions: `@/docs/agent-conventions.md`
