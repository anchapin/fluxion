# ADR-0016: Three-lane CI gates — fast-lane required, path-filtered physics gauntlet, nightly authority with merge freeze

> **Summary 1/7:** Measured 2026-09 baseline: per-PR fan-out ≈ 20–28 workflows (~60–84 concurrent jobs per 3-PR wave) through a Free-plan 20-job pipe; cycle time 1–5 h (p50 ~3 h); the gates that actually fail (ASHRAE suites, 8/29 and 5/13 real-failure rates) run ~20–38 min, while the slowest stragglers (determinism matrix ~40 min, 5 perf gates, coverage, CUDA) show ~0 recent real failures.
> **Summary 2/7:** Decision: split `ci.required_checks` into three lanes — **Lane 1 fast lane** (fmt, clippy, workspace tests, cycle/script gates: required per-PR on every PR, ~20–27 min pole = `rust-tests.yml`); **Lane 2 path-filtered** (ASHRAE blind validation + isolation suites, Strict Energy #1333, h_tr_em, Fast-Math, surrogate + binding gates: required per-PR only when the physics/binding path class changes); **Lane 3 nightly authority** (determinism #1351, perf gates #1618/#2693/#2772/#2919/#2922, coverage #1932, CUDA #1603: run nightly on `develop`, non-blocking per-PR).
> **Summary 3/7:** Lane-2 path filters use **deny-list semantics** (`paths-ignore` for docs/deps-only change sets): an unexpected file type triggers the gauntlet rather than skipping it — fail-safe by construction; physics path class = `src/**`, `fluxion-core/**`, `fluxion-fluid/**`, `tests/**`, `models/**`, `weather/**`.
> **Summary 4/7:** Strict Energy #1333 and h_tr_em stay per-PR (path-filtered) **deliberately** despite zero recent failures — RULES.md/ADR-0001 crown jewels; their zero-failure record is partly *because* they run everywhere.
> **Summary 5/7:** Nightly authority is enforced by the wave orchestrator, not branch protection (GitHub cannot express "last nightly green" as a required check): red nightly ⇒ merge freeze — no merges, no new waves, one freeze-breaker diagnostic agent (failing log + candidate merge list since last green), user notified. Blind window ≤ ~18 h by construction.
> **Summary 6/7:** Rejected alternative — "gate holiday" (Tier 3: drop time-consuming checks for ~1 week, substitute qualitative review, reapply after): the gates that fail are the fast ones already kept per-PR; the slow ones rarely fire, so the marginal cycle win over this ADR is ~10 min, while reconciliation across ~100 merges becomes a bisection problem (see #3770 for the cost of one escaped regression) and the repo's meta-gates (`check_required_checks_sync`, downward ratchets) would surface a drift explosion on reapplication.
> **Summary 7/7:** Acceptance (30 days post-rollout): hygiene-PR cycle p50 ≤ 75 min; physics-PR p50 ≤ 2.5 h; fix-loop rate ≤ 15 %; cancelled-run share ≤ 3 %; nightly green ≥ 6/7 nights; blind window ≤ 24 h; ≥ 2 `milestone`-labeled merges/week. Companion capacity change: GitHub Pro (20→40 concurrent) + this dev machine as a 2-slot `[self-hosted,fluxion-overflow]` runner with fork-PR isolation.

- **Status:** Accepted
- **Date:** 2026-09-15 (record created)
- **Deciders:** Fluxion maintainers
- **Supersedes:** None (refines the required-checks policy documented in `release_gates.yaml` and `docs/ci/branch-protection-strict-mode.md`)
- **Depends on:** Planning session 2026-09-15 (throughput plan); `.planning/CONTEXT.md` (pipeline glossary); Issue #1505 (queue-stall detector precedent); ADR-0015 (per-`head_sha` concurrency); Issue #3804 (weekly progress telegram that tracks Summary 7/7 acceptance)

## Context

The v1.3 wave pipeline (batches of ≤3 issues → parallel PRs → CI → fix loop → merge) showed 1–5 h cycle times with near-zero runner queueing and a median of one commit per PR — the wall clock hides in the ~40-min CI critical path, the fix-loop re-runs (33 % of PRs), orchestration latency, and Free-plan concurrency contention at wave bursts. A proposal to suspend expensive checks for a week ("gate holiday") was evaluated against 250 recent workflow runs and rejected: real failures concentrate in the fast ASHRAE/test suites, not the slow stragglers.

## Decision

As per Summary 2/7: three lanes, deny-list path filters, nightly authority bound by an orchestrator-enforced merge freeze. The canonical classification lives in `release_gates.yaml`; branch protection must be synchronized in the same PR (`apply_branch_protection.py` / `check_branch_protection_diff.py`). The determinism matrix gains a nightly `schedule:` (the only lane-3 gate lacking one today).

## Consequences

- Hygiene PRs (docs/deps/scripts/CI, ≈60 % of wave volume) drop to the lane-1 floor (~20–27 min); physics PRs keep the full gauntlet — exactly where the gates empirically fire.
- The blind window for lane-3 properties (cross-platform determinism, perf, coverage) extends from 0 to ≤ one nightly cycle (~18 h), bounded by the merge freeze.
- `docs/agents/pre-push-checklist.md` (pre-push ladder + fix-push etiquette) reduces fix-loop iterations; the wave-mix rule (≥1 `milestone`-labeled issue per wave) redirects throughput at the v1.3 scorecard rather than pure hygiene burn-down.
- Rollback: revert the lane-split PR and re-apply prior branch protection (the diff/protection checker scripts verify both directions).

## Companion artifacts

- **Weekly progress telegram — Issue #3804.** The pinned v1.3 progress tracker. Every Monday, [`scripts/progress_telegram.py`](../../scripts/progress_telegram.py) (triggered by [`.github/workflows/progress_telegram.yml`](../../.github/workflows/progress_telegram.yml) on cron `0 7 * * 1`) posts a markdown comment on issue #3804 that reports the Summary 7/7 acceptance metrics in flight: ASHRAE 140 metric pass-rate + cases-fully-passing (parsed from `validation/performance_history.latest.json` and the canonical `SCORECARD.md`); `milestone`-labelled vs hygiene merge mix; physics/hygiene cycle-time p50s; fix-loop rate; cancelled-run share; and the LIMIT-gap count from `docs/KNOWN_ISSUES.md`. Week-over-week deltas are derived statelessly from a machine-readable marker embedded in the previous comment (no committed state file — same pattern as the β-soak streak tracker, Issue #3286). The 30-day acceptance window closes when the weekly metrics clear all six targets; the issue stays open as the ongoing tracker.
- **Glossary.** `.planning/CONTEXT.md` defines the `Progress telegram` entry in the wave-pipeline vocabulary so future agents do not reinvent the term.
