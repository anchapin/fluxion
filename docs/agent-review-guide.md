# Agent Review Guide

> **TL;DR**: Defines cross-agent review personas, their responsibilities, and review schedule for Fluxion PRs.
> **Key decisions**: 8 reviewer personas | Scheduled reviews vs. ad-hoc | Staged rollout
> **Owned by**: Agent coordination
> **Reviewed**: 2026-07-13

## Overview

This guide defines the cross-agent review protocol for Fluxion, ensuring each PR receives appropriate scrutiny from domain experts before merging.

## Reviewer Personas

### 1. Physics Auditor
**Trigger**: Any change to physics modules (`src/sim/`, `src/physics/`)
**Responsibilities**:
- Verify mathematical correctness of thermal, solar, or ventilation calculations
- Check energy balance integrity
- Validate units and dimensional analysis
- Confirm alignment with ASHRAE standards when cited

**Review Checklist**:
- [ ] Governing equations are correctly implemented
- [ ] Energy balance closes within tolerance
- [ ] Reference data comparison shows < 1% deviation
- [ ] No hardcoded constants without justification

---

### 2. Safety Engineer
**Trigger**: Any change affecting system safety, fail-safes, or boundary conditions
**Responsibilities**:
- Identify potential failure modes
- Verify appropriate error handling
- Check for unsafe edge cases
- Ensure no resource leaks or unbounded operations

**Review Checklist**:
- [ ] All error paths handled gracefully
- [ ] No panics in production code paths
- [ ] Resource cleanup on failure (files, handles, memory)
- [ ] Boundary conditions validated

---

### 3. ML Surrogate Reviewer
**Trigger**: Any change to surrogate models, ML components, or surrogate swap points
**Responsibilities**:
- Verify surrogate model inputs/outputs match physics contract
- Check interpolation/extrapolation behavior
- Validate surrogate swap-in/swap-out logic
- Ensure reproducibility (seed handling, determinism)

**Review Checklist**:
- [ ] Surrogate contract (trait) unchanged without review
- [ ] Input normalization matches training distribution
- [ ] Fallback to physics when surrogate fails
- [ ] Benchmark results reported for accuracy

---

### 4. Performance Reviewer
**Trigger**: Any change with performance implications, algorithmic changes, or benchmark regressions
**Responsibilities**:
- Analyze computational complexity
- Identify potential bottlenecks
- Review memory allocation patterns
- Verify benchmark coverage

**Review Checklist**:
- [ ] Algorithmic complexity is acceptable
- [ ] No O(n²) where O(n log n) is possible
- [ ] Allocation patterns reviewed
- [ ] Benchmarks show no regression

---

### 5. Security Auditor
**Trigger**: Any external I/O, serialization, authentication, or dependency changes
**Responsibilities**:
- Review for injection vulnerabilities
- Check dependency供应链 security
- Verify data sanitization
- Ensure secrets handling is safe

**Review Checklist**:
- [ ] No unsanitized external input
- [ ] Dependencies are trusted and pinned
- [ ] No secrets in code or logs
- [ ] File I/O uses safe paths

---

### 6. Code Quality Auditor
**Trigger**: All PRs
**Responsibilities**:
- Enforce Rust idioms and best practices
- Check test coverage
- Verify documentation on public APIs
- Lint and format compliance

**Review Checklist**:
- [ ] `cargo clippy` passes
- [ ] `cargo fmt` compliant
- [ ] Public API docs present
- [ ] Test coverage maintained or improved
- [ ] No `unsafe` blocks without justification

---

### 7. BEM Domain Expert
**Trigger**: Changes to building energy modeling components, ASHRAE standard implementations
**Responsibilities**:
- Verify compliance with ASHRAE 90.1, 62.1, 140
- Check weather data handling
- Validate HVAC modeling assumptions
- Review zone balance calculations

**Review Checklist**:
- [ ] Standard citations are correct
- [ ] Implementation matches specification
- [ ] Edge cases per standard covered
- [ ] Reference test data validation

---

### 8. Integration Reviewer
**Trigger**: Multi-module changes, API changes, or system-level modifications
**Responsibilities**:
- Verify module boundaries respected
- Check trait implementation compatibility
- Validate end-to-end flows
- Review migration paths

**Review Checklist**:
- [ ] Module interfaces unchanged or properly migrated
- [ ] Trait implementations compile
- [ ] No breaking changes to downstream consumers
- [ ] Integration tests pass

---

## Review Schedule

### Staged Review Protocol

| Stage | Reviewer | Gate | Notes |
|-------|----------|------|-------|
| 1 | Code Quality Auditor | Required | Auto-assign on PR open |
| 2 | Physics Auditor | Required | If physics code changed |
| 3 | ML Surrogate Reviewer | Required | If ML/surrogate changed |
| 4 | BEM Domain Expert | Required | If BEM components changed |
| 5 | Safety Engineer | Required | If safety-relevant |
| 6 | Security Auditor | Required | If external I/O changed |
| 7 | Performance Reviewer | Required | If perf-critical path |
| 8 | Integration Reviewer | Required | If multi-module change |

### Review Timeline

1. **PR Open**: Auto-assign Code Quality Auditor + relevant domain reviewers
2. **24h Hold**: Allow initial review cycle
3. **Review Round 1**: All assigned reviewers submit feedback
4. **Author Response**: Address feedback or escalate
5. **Review Round 2**: Re-review if major changes
6. **Approval**: All required reviewers must approve
7. **Merge**: Squash-merge with descriptive commit

### Escalation

- **Blocking issue**: Any reviewer can request freeze
- **Unresolved disagreement**: Escalate to human maintainer
- **Stale PR (>7 days)**: Auto-flag for author follow-up

---

## Review Assignment

Reviewers are auto-assigned based on file path patterns:

| Pattern | Reviewers |
|---------|-----------|
| `src/physics/**` | Physics Auditor, BEM Domain Expert |
| `src/sim/surrogate/**` | ML Surrogate Reviewer |
| `src/sim/*.rs` | Physics Auditor, BEM Domain Expert |
| `src/**/ventilation*.rs` | Physics Auditor, BEM Domain Expert |
| `src/**/solar*.rs` | Physics Auditor |
| `src/**/thermal_model*.rs` | Physics Auditor, BEM Domain Expert |
| `src/**/io*.rs` | Security Auditor |
| `src/**/network*.rs` | Security Auditor |
| `**/Cargo.toml` | Security Auditor, Code Quality Auditor |
| `src/**/*performance*.rs` | Performance Reviewer |
| Any multi-module change | Integration Reviewer |

---

## Quality Gates

All PRs must pass:

1. [ ] `cargo check --all-targets`
2. [ ] `cargo test`
3. [ ] `cargo clippy -- -D warnings`
4. [ ] `cargo fmt --check`
5. [ ] All required reviewer approvals
6. [ ] No unresolved blocking comments

---

## Notes

- Reviewers should respond within 24h during business days
- Use `clang-format` for C/C++ components
- Performance benchmarks required for algorithmic changes
- Regression tests required for bug fixes
