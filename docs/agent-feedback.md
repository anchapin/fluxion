# Agent Feedback Template

> **TL;DR**: Standardized feedback collection and ingestion process for agent sessions.
> **Key decisions**: Structured template | Continuous ingestion | Feedback-driven improvement
> **Owned by**: Agent coordination
> **Reviewed**: 2026-07-13

## Purpose

This document provides the standard template for collecting end-of-session feedback from agents and defines the process for ingesting that feedback into the project improvement cycle.

---

## End-of-Session Feedback Template

### Session Header

```
## Session Feedback

**Session ID**: `<uuid>`
**Agent**: `<agent-name>`
**Date**: `<YYYY-MM-DD>`
**Duration**: `<duration>`
**Task Type**: `<feature|bugfix|refactor|review|research|documentation>`
**Outcome**: `<completed|partial|blocked|failed>`
```

---

### Technical Feedback

#### What Worked Well?
- **Code quality**: Describe any high-quality code patterns, clean abstractions, or excellent tests
- **Architecture**: Note good architectural decisions or module boundaries
- **Tooling**: Positive observations about build, test, or CI/CD

#### What Could Be Improved?
- **Code complexity**: Areas that were overly complex or hard to understand
- **Technical debt**: Accumulated debt noticed during the session
- **Missing abstractions**: Opportunities for better separation of concerns

#### Unresolved Technical Questions
- `<question 1>`
- `<question 2>`

#### Bugs or Issues Discovered
- `<bug description and location>`

---

### Process Feedback

#### Task Clarity
- **Was the task well-defined?** Yes/No
- **Were requirements clear?** Yes/No
- **Feedback**: `<free text>`

#### Information Access
- **Had necessary context?** Yes/No
- **Documentation was adequate?** Yes/No
- **Feedback**: `<free text>`

#### Collaboration
- **Tooling worked well?** Yes/No
- **Communication was clear?** Yes/No
- **Feedback**: `<free text>`

---

### Improvement Suggestions

#### High Priority
1. `<suggestion>`
2. `<suggestion>`

#### Medium Priority
1. `<suggestion>`
2. `<suggestion>`

#### Low Priority
1. `<suggestion>`
2. `<suggestion>`

---

### Metrics (Optional)

| Metric | Value |
|--------|-------|
| Lines changed | `<number>` |
| Files touched | `<number>` |
| Tests added | `<number>` |
| Bugs found | `<number>` |
| TODO comments | `<number>` |

---

### Notes for Next Session

```
<free text for context that should persist to next session>
```

---

## Feedback Ingestion Process

### Collection

1. **Automated Collection**
   - Session logs are captured automatically
   - Error events are tagged with session ID
   - Code quality metrics are collected via CI

2. **Manual Collection**
   - Agent completes feedback template at session end
   - Human reviewer adds annotations during code review
   - Stakeholder feedback is collected via issues

### Aggregation

1. **Weekly Review**
   - Aggregate feedback by category
   - Identify recurring themes
   - Prioritize improvements by frequency and impact

2. **Monthly Analysis**
   - Trend analysis across sessions
   - Measure improvement initiative impact
   - Update processes based on data

### Action Items

| Feedback Type | Owner | SLA |
|---------------|-------|-----|
| Code quality issues | Code author | Next PR |
| Process gaps | Team lead | 1 week |
| Tooling problems | DevOps | 2 weeks |
| Documentation gaps | Tech writer | 1 week |

### Feedback Loops

1. **Short-loop**: Feedback from current sprint informs next sprint
2. **Medium-loop**: Monthly review adjusts processes
3. **Long-loop**: Quarterly strategy updates based on trends

---

## Feedback Categories

### Code Quality
- Complexity issues
- Test coverage gaps
- Documentation missing
- Type safety concerns
- Error handling patterns

### Process Efficiency
- Task definition quality
- Information access friction
- Waiting time (blocked on reviews, context)
- Tool limitations

### Architecture
- Module boundary issues
- Coupling concerns
- Extensibility limitations
- Performance bottlenecks

### Collaboration
- Review turnaround time
- Communication clarity
- Context preservation
- Handoff smoothness

---

## Privacy Considerations

- Session IDs are pseudonymous
- No personal data in feedback templates
- Aggregated data only shared internally
- Individual feedback confidential unless explicit consent

---

## Related Documents

- [Agent Review Guide](agent-review-guide.md) — Review personas and schedule
- [Agent Loop Skill](../templates/skills/agent-loop/SKILL.md) — Autonomous work protocol
