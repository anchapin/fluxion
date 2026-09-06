# ADR-0015: PR-event concurrency key — per-`head_sha` grouping (Issue #3366)

> **Summary 1/7:** 5 of 6 recent `Rust Tests & Linting` PR runs ended in `cancelled` status (2026-09-04 to 2026-09-06 baseline), because the existing `concurrency.cancel-in-progress: true` on `${{ github.workflow }}-${{ github.ref }}` cancels a run the instant the developer pushes a new commit to the same PR branch.
> **Summary 2/7:** Adopt per-`head_sha` concurrency grouping for `pull_request` events: `${{ github.workflow }}-${{ github.event.pull_request.head.sha || github.ref }}` — different head_shas run in parallel and never cancel each other.
> **Summary 3/7:** Push events to `develop`/`main` keep the current `${{ github.workflow }}-${{ github.ref }}` + `cancel-in-progress: true` — superseded pushes are rare (1-2/day) and you want the runner freed quickly.
> **Summary 4/7:** No soft cap on parallel runs per PR — public OSS has unlimited GH minutes, and real queue starvation from rapid rebase/fixup pushes is rare enough to not justify the workflow complexity.
> **Summary 5/7:** The merge commit's required-check status is always the latest `head_sha` by definition — the previous run becoming stale is moot; GitHub surfaces the latest green check.
> **Summary 6/7:** This change ships in the same coordinated PR as ADR-0014 (nextest adoption); both decisions share the Issue #3366 tracking envelope.
> **Summary 7/7:** Acceptance: ≤1 of 6 PR runs of `Rust Tests & Linting` ends in `cancelled` for reasons unrelated to the developer actually closing/reopening the PR (from 2026-09 baseline 5/6).

- **Status:** Accepted
- **Date:** 2026-09-06 (record created)
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** Issue #3366 (parent tracking issue)
- **Issue:** [#3366](https://github.com/anchapin/fluxion/issues/3366)
- **Related:** ADR-0014 (`cargo nextest` test-runner adoption, the coordinated first leg of #3366); `release_gates.yaml::ci.required_checks` (no change); `AGENTS.md` §"Runner routing — GH probe + Hetzner overflow" (orthogonal pattern, not affected); all 46 workflows under `.github/workflows/`

---

## Context

Every workflow in `.github/workflows/` currently declares:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

This means **any new push to the same ref cancels the in-progress run**. For PR branches (`refs/pull/N/merge` or `refs/heads/<branch>`), a developer pushing a fix-up commit while CI is running kills the previous run.

The 2026-09-04 to 2026-09-06 PR-fleet data shows the cost:

| Workflow | Successful PR runs / total | Cancellation rate |
|---|---|---|
| Rust Tests & Linting | 1 / 6 | 83% |
| Surrogate Drift Tolerance Gate | 12 / 12 | 0% (long pole but rare commits) |
| ASHRAE 140 Validation | 12 / 12 | 0% |
| Python Tests (OSimFlow) | 6 / 6 | 0% |
| Code Coverage | 6 / 6 | 0% |
| … | | |

Why `Rust Tests & Linting` is uniquely cancelled: it is the slowest per-commit workflow (~38 min on a successful run). Any developer iterating faster than 38 min triggers a cancellation. The other long workflows (Python Bindings, Cross-Platform Determinism, ASHRAE 140 CI Gate) have similar wall-clock but appear with single runs per PR cycle (the developer pushes once, waits for green, merges).

The cancel pattern is invisible to the developer as a *cancellation*: from the developer's view, CI "starts but never finishes" — they push, watch the run spinner for ~30 minutes, push a fix-up, see the first run stop, push again, and eventually the last run completes or is itself cancelled. The pattern matches the user's description of "gets stuck."

For `Rust Tests & Linting` specifically, the cancellation is also asymmetric: the *long-pole matrix entry* (`Test (ubuntu-latest, multi-zone)`, 305s of `cargo test`) is the entry the developer cares about most (it exercises the feature under change), and it is the entry most likely to be killed by a re-push.

Three concurrency strategies were considered; see the Plan section for the rejected alternatives and their cost analysis.

---

## Decision

**Change the `concurrency.group` key for `pull_request` events to include `github.event.pull_request.head.sha`.** Keep the current key + `cancel-in-progress: true` for `push` events to `develop`/`main`.

The concrete change (illustrative for `.github/workflows/rust-tests.yml`; same pattern applies to all 46 workflows):

```yaml
concurrency:
  group: >-
    ${{ github.workflow }}-
    ${{
      github.event_name == 'pull_request' &&
      github.event.pull_request.head.sha
      || github.ref
    }}
  cancel-in-progress: >-
    ${{
      github.event_name == 'push'
      && contains('refs/heads/main,refs/heads/develop', github.ref)
    }}
```

This is a pure configuration change. No job structure changes; no required-check changes; no `release_gates.yaml` changes.

---

## Plan

### Step 1 — Apply the new `concurrency` block to all workflows

The change is mechanical and identical across the 46 workflow files in `.github/workflows/`. The implementation PR applies it via a script (`scripts/update_concurrency_keys.py` or similar, to be added in the PR) that:

- Reads each `.github/workflows/*.yml`.
- Replaces the existing `concurrency: { group: ..., cancel-in-progress: true }` block with the new template above.
- Leaves any workflow-specific `group:` strings (e.g., the workflow `name`-based prefixes) untouched.

### Step 2 — Verify with a controlled experiment

Before merging:

- Run a small "two-commit-on-same-PR" simulation: push commit A, immediately push commit B (a no-op change like a doc fix). Confirm that both runs execute (in parallel, on different runners) and both reach `success` / `failure`. The merge commit (B's `head_sha`) shows B's results.
- Confirm that a `push` to `develop` still cancels a previous in-flight push run (current behavior preserved).

### Step 3 — Post-merge watch (1 week, parallel to ADR-0014)

- Monitor `gh run list --repo anchapin/fluxion --workflow "Rust Tests & Linting" --event pull_request --json conclusion` for 1 week.
- Acceptance: ≤1 of 6 PR runs ends in `cancelled` for reasons unrelated to actual PR close/reopen.

### Step 4 — If queue starvation materialises (YAGNI)

If real-world data shows that rapid rebase / fixup pushes (≥5 commits in <5 minutes) starve other PRs of GH free runner capacity, add a soft cap as a follow-up. The follow-up would either:

- Drop the oldest queued run (impossible in current GHA — `cancel-in-progress: true` is the only signal that drops work), or
- Use a workflow-level `if:` condition that gates on `github.event.pull_request.head.sha` being among the last 2 `head_shas` for that PR (requires a custom action that reads PR commit history).

Today, this is speculative. Skip until needed.

---

## Rejected Alternatives

### α' — Conditional cancel: `cancel-in-progress: ${{ github.event_name == 'push' }}` only

This cancels nothing on PR events; both A and B execute fully, but **sequentially in queue** (because they share the same `ref`-based group key). Net effect: PR feedback latency stays at ~38 min because the queue holds B until A finishes. Rejected because it preserves the wall-clock pain.

### γ' — Drop `cancel-in-progress` entirely

Both A and B execute, but **both queue serially**. Worse than α' because no parallelism at all, and CI minutes double or triple per PR iteration cycle. Rejected because it preserves the wall-clock pain *and* wastes public OSS CI minutes.

### δ' — Per-PR "latest N=2 head_shas" soft cap

Adds a workflow-level guard that only runs CI for the latest 2 head_shas on a PR. Rejected because: (a) GHA has no native "queue length" semantic for the same group; (b) implementing it requires a custom action that reads PR commit history via the API — a meaningful complexity cost; (c) the failure mode it's protecting against (queue starvation from ≥5 rapid pushes) is rare enough to not justify the complexity. YAGNI.

### ε' — Keep current key, just lower the wall-clock

Status quo on concurrency; rely entirely on ADR-0014 (nextest) to drop the wall-clock. Rejected because: (a) ADR-0014 alone cannot bring the wall-clock below the developer's commit cadence (15 min target vs. typical 5-10 min iteration cadence), so cancellations will still happen; (b) the "stuck" feeling (user's original complaint) is more about *seeing* a green check than *waiting* for one — cancellation suppresses the green.

---

## Consequences

### Positive

- The "stuck" feeling goes away: every `head_sha` runs to completion and reports its own green check. The developer's last push (the merge commit) is what GitHub shows on the PR.
- Public OSS unlimited GH minutes mean no cost pressure from the new parallelism — two PR runs in flight per PR is fine.
- The change is mechanical and applies uniformly across all 46 workflows, so the rollout is consistent.
- The `push`-event path preserves the current behavior: superseded pushes to `develop`/`main` are still cancelled quickly, freeing runners.

### Negative

- A `pull_request` event from a malicious or careless contributor pushing 10 commits in 60 seconds will spawn up to 10 parallel CI runs. Public OSS has no cost ceiling, but real queue pressure on GH free runners could starve other PRs of capacity. The mitigation is the per-workflow matrix: not every workflow runs in parallel from one contributor's pushes. Empirically, this is rare enough to defer.
- The `cancel-in-progress: >-` conditional expression is slightly harder to read than a boolean literal. Future maintainers may wonder "why is this conditional?" — this ADR is the answer.

### Neutral

- The pattern is identical across all 46 workflows, so it can be applied by a script and verified by a CI check (e.g., `scripts/check_concurrency_keys.py`).
- The Hetzner overflow pattern (`*-gh-probe` / `*-gh` / `*-hz` jobs under AGENTS.md §"Runner routing") is orthogonal and unaffected.
- The probe pattern's 5-min timeout is unaffected — probes are still per-runner, not per-SHA.

---

## References

- Issue #3366 — parent tracking issue.
- ADR-0014 — coordinated `cargo nextest` test-runner adoption.
- `release_gates.yaml::ci.required_checks` — canonical required-check list; no change.
- `AGENTS.md` §"Runner routing — GH probe + Hetzner overflow" — orthogonal pattern, unaffected.
- `.github/workflows/*.yml` — all 46 workflows receive the same `concurrency:` block change.
- `gh run list --repo anchapin/fluxion` — 2026-09-04 to 2026-09-06 baseline runs used to derive the cancellation-rate acceptance criterion.
- `copilot-instructions.md` §"Workflow author rules" (if present) — preserve any project-specific concurrency overrides when applying the script.
- `scripts/check_concurrency_keys.py` — to be added in the implementation PR; verifies all 46 workflows have the per-SHA pattern (or an explicit documented exception).