# Agent Loop Skill

> **TL;DR**: Autonomous work protocol enabling agents to execute tasks continuously with structured checkpoints.
> **Key decisions**: Pre-flight checklist | Task selection queue | Execution loop with health checks | Night shift mode
> **Owned by**: Agent coordination
> **Reviewed**: 2026-07-13

## Overview

The agent loop skill defines a structured protocol for autonomous task execution, enabling agents to pick tasks from a queue, execute them with checkpoints, and continue working across sessions.

## Pre-Flight Checklist

Before starting work, verify:

### Environment
- [ ] Working directory exists and is clean
- [ ] Required tools available (`git`, `cargo`, `gh`, `cargo-rtk`)
- [ ] Network access verified
- [ ] Credentials available (GH_TOKEN, etc.)

### Context
- [ ] Project documentation reviewed (`ARCHITECTURE.md`, `AGENTS.md`)
- [ ] Recent session context loaded (check `CONTEXT.md` if exists)
- [ ] Active issues reviewed for current priorities
- [ ] Any blocking issues acknowledged

### State
- [ ] Git branch created or checked out
- [ ] No uncommitted changes in worktree
- [ ] Upstream is up to date

---

## Task Selection

### Priority Queue

Tasks are selected in priority order:

1. **Blocked tasks** — Unblock any blocked work first
2. **In-progress tasks** — Continue existing work before starting new
3. **High-priority issues** — P0/P1 from issue tracker
4. **Medium-priority issues** — P2/P3 for steady progress
5. **Low-priority maintenance** — P4/P5 when queue is empty

### Selection Criteria

- Match task type to agent capability
- Prefer tasks where context is available
- Avoid starting multiple large tasks simultaneously
- Respect task dependencies

### Task Validation

Before starting:
- Verify issue exists and is not already being worked
- Check for prerequisites (approved designs, dependencies)
- Confirm acceptance criteria are clear

---

## Execution Loop

### Loop Structure

```
WHILE has_tasks AND is_healthy:
    task = select_next_task()
    IF can_start(task):
        start_task(task)
        execute_task(task)
        IF task.complete:
            finalize_task(task)
        ELSE IF task.blocked:
            flag_blocked(task)
        ELSE IF task.failed:
            flag_failed(task)
    ELSE:
        wait_for_prerequisites(task)
    
    health_check()
    update_progress()
END WHILE
```

### Task Execution Phases

#### 1. Start
- Assign issue to self (if tracked)
- Create branch if needed: `fix/issue-{N}-{slug}`
- Update task status

#### 2. Research
- Read relevant documentation
- Understand architecture context
- Identify affected components
- Plan implementation approach

#### 3. Implement
- Make changes incrementally
- Run tests frequently
- Check lint/type errors early

#### 4. Verify
- All tests pass
- Lint clean
- Typecheck clean
- Documentation updated if needed

#### 5. Finalize
- Commit with descriptive message
- Push branch
- Create PR (if applicable)
- Update issue status

### Health Check

Every 5 iterations or 30 minutes, verify:

- [ ] No infinite loops or memory leaks
- [ ] Progress is being made
- [ ] No repeated failures on same issue
- [ ] Context window not exceeded
- [ ] Can still reach external services

### Progress Tracking

Track in session state:
- Tasks attempted
- Tasks completed
- Tasks blocked
- Time spent
- Key decisions made

---

## End-of-Loop

### Before Exiting

1. **Commit any partial work**
   - Use `WIP: ` prefix in commit message
   - Push to remote for persistence

2. **Update issue status**
   - Move to appropriate column or add comment
   - Note what was accomplished

3. **Document context**
   - Write `CONTEXT.md` with current state
   - Note what needs to happen next
   - Include any unresolved decisions

4. **Session summary**
   - Complete feedback template
   - Report metrics if tracked

### Graceful Exit Conditions

- All tasks in queue completed
- No more high-priority work available
- Health check failed (needs human intervention)
- Explicit stop signal received
- Time limit reached

---

## Night Shift Mode

For extended autonomous operation (e.g., overnight runs):

### Enabling Night Shift

```
/night-shift on --max-tasks 5 --max-hours 12
```

### Constraints

- Only touch P2 or lower priority issues
- Do not modify shared infrastructure
- Do not merge any PRs automatically
- Pause at any security-relevant change
- Email summary to maintainers at 8am

### Safety Limits

- Maximum 5 tasks per night
- Maximum 12 hours runtime
- Mandatory 30-minute pause at 2am
- Wake check at 7am (respond to any urgent issues)

### Night Shift Checklist

- [ ] CI/CD credentials are read-only
- [ ] No production system access
- [ ] Monitoring alerts are active
- [ ] Emergency contact listed
- [ ] Rollback plan documented

---

## Error Handling

### Retry Policy

| Error Type | Retry | Backoff |
|------------|-------|---------|
| Network timeout | 3x | 30s |
| Test failure | 1x | N/A |
| Lint failure | 1x | N/A |
| Build failure | 2x | 60s |
| API rate limit | Wait | 5min |

### Dead Letter Queue

Tasks that fail 3 times or are blocked > 24h:
- Flag in issue tracker
- Notify maintainers
- Skip to next task
- Log for weekly review

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_ITERATIONS` | 100 | Max loop iterations |
| `AGENT_HEALTH_CHECK_INTERVAL` | 30min | Health check frequency |
| `AGENT_NIGHT_SHIFT_START` | 22:00 | Night shift start time |
| `AGENT_NIGHT_SHIFT_END` | 07:00 | Night shift end time |

### Skill Settings

```yaml
agent_loop:
  pre_flight_required: true
  health_check_interval: 30
  max_retries: 3
  night_shift_enabled: false
  progress_tracking: true
```

---

## Related Documents

- [Agent Review Guide](../docs/agent-review-guide.md) — Review personas and schedule
- [Agent Feedback Template](../docs/agent-feedback.md) — Feedback collection
