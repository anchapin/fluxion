# Agent Workflow

> **TL;DR**: Standard four-phase workflow for all agent coding sessions.
> **Phases**: Research → Plan → Implement → Wrap-up
> **Owned by**: All contributors
> **Reviewed**: 2026-07-13

## Overview

Every agent session follows this four-phase structure. Tag `@/docs/agent-workflow.md` at session start.

```
Phase 1: Research    → Understand the issue
Phase 2: Plan        → Write plan to docs/worksheets/
Phase 3: Implement   → Code + tests in lockstep
Phase 4: Wrap-up     → Review, test, commit
```

---

## Phase 1 — Research

Understand the issue before writing any code.

### Checklist

- [ ] Read the full issue description and comments
- [ ] Read `ARCHITECTURE.md` for relevant module boundaries and contracts
- [ ] Read the affected source files — understand the current implementation
- [ ] Run the affected code paths (e.g., `cargo test --lib`, integration runs)
- [ ] Read the existing tests for the affected module
- [ ] Check `docs/KNOWN_ISSUES.md` for related known issues
- [ ] Check recent commits on the relevant module for context

### Exit criterion

You can explain the issue in your own words and know which files need to change.

---

## Phase 2 — Plan

Write the plan before writing code.

### Checklist

- [ ] Create `docs/worksheets/issue-{N}-plan.md` (see worksheet format below)
- [ ] Tag related docs in the plan (e.g., `ARCHITECTURE.md §3`, `tests/README.md`)
- [ ] Identify all files that need to change
- [ ] Identify test files that need new or updated tests
- [ ] Get cross-agent review of the plan before implementing (optional but recommended for large changes)
- [ ] Ensure the plan respects the validation strategy: **no parameter tuning to make tests pass — fix the underlying math**

### Worksheet format

```markdown
# Issue {N} — Plan

## Issue
Brief description of the issue.

## Root Cause
What's actually wrong?

## Files to Change
- `src/path/file.rs` — reason
- `tests/path/test.rs` — reason

## Implementation Steps
1. Step description
2. Step description

## Validation
- [ ] Unit tests pass
- [ ] Reference data comparison (if applicable)
- [ ] No regression in related modules
```

---

## Phase 3 — Implement

Write code and tests together.

### Checklist

- [ ] Write tests **before** or **in parallel** with implementation (TDD for bug fixes)
- [ ] Use reference data CSVs for EnergyPlus parity checks
- [ ] Name tests descriptively: `fn solar_altitude_40N_summer_solstice_matches_epplus()`
- [ ] Run the app on every significant change (`cargo build`, `cargo test`)
- [ ] For physics/math: use `ctx_execute` with Python to verify calculations — never mental math
- [ ] Follow existing code style and conventions in the module
- [ ] Update ARCHITECTURE.md if the code changes the documented interfaces

### Validation targets

- Each physics module must match EnergyPlus within **1% tolerance** on isolated scenarios
- **No ASHRAE 140 system-level testing** until individual modules pass E+ reference tests
- **No parameter tuning** to make system tests pass — fix the underlying math

---

## Phase 4 — Wrap-up

Final review, testing, and commit.

### Checklist

- [ ] **Cross-agent review**: have another agent or model review the changes
- [ ] Run the **full test suite**: `make test-fast` or `cargo test`
- [ ] Run linting: `make lint` or `cargo fmt --check && cargo clippy`
- [ ] Write end-of-session worksheet: `docs/worksheets/issue-{N}-wrapup.md`
- [ ] Commit with a **git tag** indicating the phase/state: `git tag -a issue-{N}-done -m "Issue {N} resolved"`
- [ ] Push branch and open PR with reference to the issue

### Cross-agent review notes

- Use the `pr-review-merge` skill or request a peer review
- Review checks: correctness, security, performance, test coverage, documentation drift
- If the change affects physics: verify math with Python against reference data

### Worksheet index

Worksheets are stored in `docs/worksheets/`. Each worksheet is named:
- `issue-{N}-plan.md` — Phase 2 plan
- `issue-{N}-wrapup.md` — Phase 4 wrap-up

---

## References

- `ARCHITECTURE.md` — Module boundaries, I/O contracts, physics diagrams
- `docs/AI_CODING_STRATEGY_ADOPTION_PLAN.md` §2 — Origin of this document
- `docs/worksheets/` — Worksheet storage directory
- `AGENTS.md` — Context-mode routing rules, tool hierarchy
