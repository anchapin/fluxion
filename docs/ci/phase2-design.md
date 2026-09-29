# CI Phase 2 design: merge queue + physics-pr.yml + sharded ASHRAE suite

<!-- summary: Design for #4007 — merge queue on develop, one physics-pr.yml (~6 checks), sharded ASHRAE suite with sccache-warmed shared build, nightly cache-prime. -->
<!-- status: implemented 2026-09-27 (no merge queue per Alex's decision); see §8. Triplicate fold deferred pending 30-day data. -->
<!-- scope: CI only; no physics, no tolerance, no test changes. -->
<!-- lane: workflow-only (required_checks_workflow_only) when implemented. -->
<!-- risks: merge-queue changes how Alex merges; merge_group trigger rollout; check renames need release_gates.yaml + branch-protection sync. -->
<!-- decisions: merge queue yes/no; PR-vs-queue check split; strict-gate + nextest-subset promotion; sharding scheme; probe triplicates. -->
<!-- see: #4007, ADR-0016, docs/ci/nextest-rollout.md, release_gates.yaml -->

Issue #4007 proposes three moves: (1) GitHub merge queue on `develop`, (2) one
`physics-pr.yml` consolidating the per-PR physics path to ~6 required checks,
(3) a sharded ASHRAE suite sharing one sccache-warmed build plus a nightly
cache-prime job. This doc scopes each so the decisions are easy. No workflow
code is written here — Alex picks the shape first.

## 1. Current-state inventory (measured 2026-09-26, develop @ 13772f6)

**Per-PR fan-out.** 27 of 54 workflows trigger on `pull_request`/`pull_request_target`.
Workflows doing full workspace compiles per PR (each its own checkout + toolchain +
build — the ~15+ tariff from the issue):

| Workflow | What compiles per PR |
|---|---|
| `ci-gates.yml` (phase 2) | `cargo nextest` full matrix (~38 min documented) |
| `rust-tests.yml` | workspace check + listener jobs + CUDA smoke |
| `ci.yml` | python-examples + integration-tests matrices (probe/GH/Hetzner triplicates) |
| `ashrae_validation.yml` (via `workflow_run` on CI Gates) | blind validation + 5 isolation suites + surrogate MAE gate |
| `performance_dashboard.yml` (via `workflow_run`) | perf gates |
| `determinism_check.yml` (via `workflow_run`) | 3-OS determinism |
| `code-coverage.yml` (via `workflow_run`) | coverage |
| `.github/workflows/physics-pr.yml` (job: "ASHRAE 140 Strict Energy Gate (Issue #1333)") | strict ±15% gate |
| `ashrae_140_validation.yml`, `fast_math_check.yml`, `h_tr_em_regression_gate.yml` | path-filtered physics gates |

**Phase-gated CI today (ADR-0016 + #4018/#4019).** `ci-gates.yml` holds the 14
required checks as phase 1 (9 unfiltered + 5 path-filtered Lane-2). Seven heavy
workflows (ASHRAE validation, performance dashboard, determinism, coverage,
strict energy, h_tr_em, fast-math) run only after CI Gates succeeds, via
`workflow_run: workflows: ["CI Gates"]` + a docs-only-gate precheck + PR-head-SHA
checkout. Required-check names are byte-matched by branch protection (Issue #3116);
`check_required_checks_sync.py` enforces `release_gates.yaml` ↔ live parity.

**Probe/GH/Hetzner triplicates.** Five workflows use the probe pattern
(`ci-gates.yml`, `ci.yml`, `rust-tests.yml`, `tracked-vs-ignored.yml`,
`fast_math_check.yml` + reusable `ci-steps.yml`): a lightweight `<job>-gh-probe`
(5-min timeout) tests GH-runner availability; `<job>-gh` runs if the probe
succeeds, `<job>-hz` (self-hosted Hetzner overflow) runs if it fails/is cancelled.
Security driver is Issue #3445: persistent self-hosted runners must never execute
PR-controlled code (cache poisoning), so Hetzner overflow is reserved for
main-merge pushes of trusted code.

**sccache.** `setup-rust-env` enables sccache via the GHA-cache backend by default
(content-addressed; PR cache writes are branch-scoped by GitHub, so they cannot
poison the base-branch cache — the #3445 concern is persistent runners, not
sccache itself). Explicitly disabled (`sccache: 'false'`) on a few PR-path jobs
(surrogate-drift-gate uses a `target/`-dir cache instead; ashrae_validation's
surrogate gate; wasm builds). Enabled on code-coverage and node-bindings.

**Timing samples** (recent completed runs, wall-clock incl. queueing):
CI Gates (PR) ~10–12 min; ASHRAE 140 Validation ~5–9 min; Performance Dashboard
~10–14 min; Cross-Platform Determinism ~20–24 min; Strict Energy Gate ~8–12 min;
Code Coverage ~3–6 min. (Directional — the ci-gates header documents the nextest
matrix alone at ~38 min.)

**Merge method today:** squash (merge commits are single-parent, e.g. #4097).

## 2. Merge-queue design

### Exact GitHub settings (all in repo/branch settings, no workflow code)

1. Repo Settings → General → Pull Requests → enable **Allow merge queue**.
2. Branch protection rule for `develop`: enable **Require merge queue**, with:
   - Merge method: **Squash** (preserves today's history shape).
   - Max entries to build: **1** (no speculative batching — single-maintainer repo;
     avoids ambiguous multi-PR failure attribution).
   - Min entries to merge: 1; max entries to merge: 1 (serial merges, same as today).
   - Status check timeout: 60 min (covers the ~38-min nextest matrix + queueing).
   - "Require all checks to pass" against the required-check list below.
3. Every workflow emitting a required check must add `merge_group` to its `on:`
   triggers — a check that never runs on the `merge_group` event can never satisfy
   branch protection in the queue.

### How it changes Alex's workflow

Today: PR green → click **Merge pull request** (squash) → done. With the queue:
PR checks green → click **Merge when ready** (adds to queue) → GitHub builds a
merge commit of `develop` HEAD + PR, runs the `merge_group` checks, merges
automatically if green, removes from queue if red. Differences that matter:

- Merges are no longer instant: queue latency = merge_group check runtime.
- No more "merge now, fix CI as follow-up" for checks in the queue set — a red
  queue entry blocks the merge. (This cuts against the stated CI philosophy of
  moving fast; the mitigation is keeping the queue set to the true must-pass
  gates and leaving the rest advisory.)
- Rebase-then-merge still works: the queue handles the rebase onto current
  `develop` HEAD itself, which actually removes today's manual rebase-before-push
  step.

### `workflow_run` listeners and `merge_group`

`workflow_run` fires when the named workflow's run completes — including runs
triggered by `merge_group`. So listeners keyed on "CI Gates" keep working, **but**
two code changes are needed:

- The phase-gate precheck (`docs-only-gate` + upstream-success check) parses the
  `workflow_run` event payload; `merge_group` runs have a different payload shape
  (`github.event.merge_group.head_sha` instead of PR context). The precheck
  scripts need a `merge_group` branch.
- ADR-0015 concurrency (`check_concurrency_keys.py`): the per-`head_sha` group
  expression needs a `merge_group.head_sha` arm, or queue runs cancel each other.

### With-queue vs without-queue

| | With queue (recommended) | Without queue |
|---|---|---|
| Trigger split | `pull_request`: light checks only (physics-pr.yml ~6 + docs + drift + deny). `merge_group`: heavy suite (full nextest, sharded ASHRAE, determinism, perf, coverage). | Everything on `pull_request` as today; consolidation + sharding only. |
| Heavy-gate tariff | Once per merge, on the true merge commit. | Per push, on the PR head (stale the moment develop moves). |
| `push: [develop]` triggers | Drop on heavy workflows — the queue already validated the merge commit. | Keep (post-merge signal). |
| Alex's merge action | "Merge when ready" → automatic. | Unchanged. |
| Risk | Queue misconfiguration blocks all merges; red queue entry = no merge-now option. | No new failure modes. |

Recommendation: **with queue**, because the single biggest tariff cut in the
issue ("heavy gates run once per merge, not per push") is only real with it, and
validating the actual merge commit (rather than a PR head that has since drifted
from `develop`) is strictly stronger. Keep the queue set minimal so the
move-fast philosophy survives: queue = physics-pr.yml checks + full nextest +
sharded ASHRAE + determinism. Everything else advisory.

## 3. `physics-pr.yml` consolidation

Fold the per-PR physics path into one workflow. Keep every required-check **name**
byte-identical (Issue #3116) — only the file/owner changes.

| Current required check (14) | Verdict | Rationale |
|---|---|---|
| Rustfmt (GH) | **Fold** | Fast; same name, job moves into physics-pr.yml |
| Clippy (GH) | **Fold** | Same |
| Workspace Check (GH) | **Fold** | Same |
| Energy Conservation (GH) | **Fold** | The energy-conservation invariant; core physics signal |
| Physics-Sim-Cycle-Check (GH) | **Fold** | Script-only, seconds |
| Ashrae Cases Cycle Check (GH) | **Fold** | Script-only, seconds |
| Cycle Downward Trend Guard (#2768) | **Fold** | Script-only, seconds |
| ASHRAE 140 Strict Energy Gate (#1333) | **Promote → required** | Currently path-filtered non-required (ADR-0016); the issue wants it per-PR. Needs the docs-only-gate skip pattern so docs-only PRs stay mergeable |
| Curated nextest subset | **New required** | Policy change: today the nextest matrix is advisory phase-2. A curated subset (lib tests + energy-conservation + isolation smoke, ~10 min budget) becomes the PR-time regression signal; the full matrix moves to the queue/nightly |
| Surrogate Drift Tolerance Gate (#1784) | **Keep standalone** | Lives in ci-gates.yml phase 1; path-filtered to surrogate files; leave it |
| Cargo Deny | **Keep standalone** | Supply-chain; fast; already in phase 1 |
| Docs Hygiene Gate (#2466) | **Keep standalone** | Docs-only lane; untouched |
| Architecture Drift Detection | **Keep standalone** | Path-filtered Lane-2; untouched |
| Module Size (#2878) | **Keep standalone** | Path-filtered Lane-2; untouched |
| MSRV Check (#2934) | **Keep standalone** | Path-filtered; untouched |
| Crate Size Gate (#2930) | **Keep standalone** | Path-filtered; untouched |

Steady-state required: **~9–10** (6 core physics-pr checks + 3 cycle/script gates
folded in the same file + surrogate drift + deny + docs hygiene; arch/module/msrv/
crate-size add 0–4 more when their paths change) vs 14 today. The bigger win is
structural: one checkout + one toolchain + one workspace compile serves all
folded jobs (cargo's target dir is shared within a job; split across jobs in one
workflow on the same runner via a build-once fan-out).

`required_checks_workflow_only` lane: the 9 workflow-only checks today are
`{surrogate drift, physics-sim-cycle, workspace check, energy conservation,
rustfmt, clippy, ashrae-cases-cycle, cycle-trend, deny}` — all of these live in
physics-pr.yml under the proposal, and all run fine on workflow-only PRs (none
depend on workflow files). So the workflow-only lane becomes
**physics-pr.yml's 9 checks + docs hygiene** with no special-casing, and the
`required_checks_workflow_only` list shrinks to a comment pointing at the file.

`check_required_checks_sync.py` + branch protection: the strict-gate promotion and
the new nextest-subset check are **renames/additions** — `release_gates.yaml`
must be updated in the same PR and the live branch-protection PUT re-applied
(the established `scripts/apply_branch_protection.py` flow). The strict gate's
current check name ("ASHRAE 140 Strict Energy Gate (Issue #1333)" in
workflow_index) should be kept verbatim to avoid churn.

## 4. ASHRAE sharding

`ashrae_validation.yml` today: 15 jobs, all sequential-ish — blind `validate`,
5 isolation suites (weather, solar, conduction, ventilation, zone-balance),
surrogate MAE gate, quantum/gauge diagnostics, 2 listener gates. The 5 isolation
suites + blind validation are independent modules in the `all_tests` runner and
shard cleanly.

Proposed `ashrae-shard.yml` (replaces the per-PR role of `ashrae_validation.yml`):

- **`build` job** (once): checkout + toolchain + `cargo build --release --tests`
  with sccache enabled (GHA backend). Warms the shared content-addressed cache;
  uploads nothing (sccache is the sharing mechanism, not artifacts).
- **`shard` matrix** (6 shards, `needs: [build]`, all on ephemeral
  `ubuntu-24.04`): `validate`, `isolation-weather`, `isolation-solar`,
  `isolation-conduction`, `isolation-ventilation`, `isolation-zone-balance` —
  each runs its module filter against the warm sccache (incremental link only).
- Surrogate MAE gate, quantum/gauge diagnostics stay as-is (different
  dependencies); determinism/performance listeners keep observing their upstream
  workflows by name.
- Security: shards run on ephemeral GH-hosted runners only (Issue #3445 —
  never persistent self-hosted for PR code). sccache GHA-backend writes from PRs
  are branch-scoped and cannot poison the base cache.

**Nightly cache-prime** (`cache-prime.yml`, schedule ~06:00 UTC on `develop`):
`cargo build --release --tests` (+ the `--features grid` and default matrices'
feature sets) with sccache enabled, so the morning's first PRs hit a warm cache.
Informational, never required. Estimated saving: the cold→warm delta on the
release test build (measure in implementation; the determinism job's ~20–24 min
is the proxy baseline).

## 5. Probe/Hetzner triplicates: justified vs foldable

| Triplicate site | Assessment |
|---|---|
| `rust-tests.yml` workspace-check GH/hz | **Justified** — Hetzner arm restricted to main-merge pushes (#3445); PR path stays ephemeral GH |
| `ci.yml` python-examples / integration-tests probes | **Needs Alex's data call** — the probe adds up to 5 min stall per PR when GH is saturated. Need 30-day hit-rate data: how often does the `-hz` arm actually fire on PRs? Query: `gh run list --workflow ci.yml` correlated with job-level outcomes. If the Hetzner arm fires <5% of PR runs, fold to GH-only + retry. |
| `tracked-vs-ignored.yml`, `fast_math_check.yml` probes | **Likely foldable** — both are seconds-to-minutes script/test jobs; the probe machinery costs more in complexity than Hetzner saves. Confirm with the same hit-rate query. |
| `ci-steps.yml` reusable | Keep the shape; callers decide. |

## 6. Risks

1. **Queue misbehavior blocks all merging.** Mitigation: conservative settings
   (max build entries 1, 60-min timeout), keep the queue check set minimal, and
   document the escape hatch (temporarily disable "Require merge queue" on
   develop — a settings click, no code).
2. **Move-fast tension.** A red queue entry cannot be merged around. The
   `pull_request` light set stays the fast signal; only true must-pass gates go
   in the queue.
3. **Double CI spend during transition.** Until triggers are split, checks run on
   both `pull_request` and `merge_group`. The implementation PR must do the
   split atomically.
4. **Check renames.** Strict-gate promotion + nextest-subset are new required
   names → `release_gates.yaml` sync + live branch-protection PUT in the same
   PR (`check_required_checks_sync.py` fails otherwise).
5. **Concurrency + precheck scripts** need `merge_group` arms (ADR-0015,
   docs-only-gate, workflow_run precheck) — small but must-ship in the same PR.
6. **Physics path classes untouched.** No test, tolerance, timestep, or
   validation-logic changes anywhere in this design; `physics-pr.yml` only
   re-homes existing jobs.

## 7. Decisions for Alex

1. **Merge queue on develop: yes or no?** (Recommended: yes — squash, max build
   entries 1, 60-min timeout.)
2. **PR vs queue check split:** light ~9 on `pull_request`, heavy suite
   (full nextest, sharded ASHRAE, determinism, perf, coverage) on `merge_group`
   only — agree?
3. **Promote** Strict Energy Gate and a curated nextest subset to required?
   (This is the one real policy change; everything else is re-homing.)
4. **ASHRAE sharding:** 6-shard scheme + 06:00 UTC cache-prime on develop —
   approve?
5. **Probe triplicates:** gather the 30-day Hetzner-arm hit-rate data first, or
   fold `tracked-vs-ignored`/`fast_math_check` now and measure `ci.yml` later?
6. Confirm **squash** stays the merge method under the queue.

## 8. Implementation status (2026-09-27, PR for Issue #4007)

**No merge queue — deliberate decision.** Alex approved proceeding with the
recommendation WITHOUT the merge queue: he is effectively the sole merger,
prefers fast landing + follow-up fixes, and queue latency would oppose that
policy. This implementation is consolidation + sharding only. The queue
design in §§2–3 remains documented for a future revisit; the triplicate-data
decision (below) is likewise deferred pending 30-day usage data.

**What shipped:**

- **`physics-pr.yml` (new):** consolidates the per-PR physics path. Nine
  required checks, all with byte-identical `name:` values (Issue #3116):
  `Workspace Check (GH)`, `Energy Conservation (GH)`, `Rustfmt (GH)`,
  `Clippy (GH)`, `Ashrae Cases Cycle Check (GH)`,
  `Physics-Sim-Cycle-Check (GH)`, `Cycle Downward Trend Guard (Issue #2768)`
  (all folded verbatim from `ci-gates.yml`); `ASHRAE 140 Strict Energy Gate
  (Issue #1333)` (PROMOTED from the deleted
  `.github/workflows/physics-pr.yml`, with `strict-precheck` docs-only-gate
  + `strict-energy-gate-listener` neutral-success so docs-only PRs stay
  mergeable — Issue #3810 pattern); `Nextest Subset (GH)` (NEW curated
  signal: root lib tests + the two `all_tests` regression modules, single
  default-feature leg, ~10 min budget; the full 3-feature-set matrix stays
  in `ci-gates.yml` as advisory).
- **`ci-gates.yml`:** the seven folded jobs removed; `test` job's `needs:`
  narrowed to `[surrogate-drift-gate, deny]`. Surrogate drift, Cargo deny,
  and the full nextest matrix stay.
- **`ashrae-shard.yml` (new):** `build` job warms sccache (GHA backend)
  once; six-shard matrix (`validate`, `weather`, `solar`, `conduction`,
  `ventilation`, `zone-balance`) runs against the warm cache. All on
  ephemeral `ubuntu-24.04`. Phase-gated on "CI Gates" via `workflow_run` +
  docs-only-gate precheck. Nightly `cache-prime` job at 06:00 UTC on
  `develop`.
- **`ashrae_validation.yml`:** the seven moved jobs
  (`queue-stall-detector`, `validate`, `validate-fallback`, five isolation
  suites) removed. Surrogate MAE gate, quantum/gauge diagnostics, and the
  determinism/performance listeners stay. The Issue #1505 self-hosted
  routing + fallback machinery was removed with the move (ephemeral-only
  per the #4007 decision).
- **`release_gates.yaml`:** `required_checks` 14 → 16 (strict gate
  promoted, nextest subset added); `required_checks_workflow_only` 9 → 11;
  `workflow_index` updated (8 entries repointed to `physics-pr.yml`,
  1 new entry for the nextest subset).
- **Docs:** `AGENTS.md` and `docs/ci/branch-protection-strict-mode.md`
  count literals updated (14→16, 9→11); the Issue #3898 removal text now
  notes the strict gate's #4007 promotion.

**What remains deferred (needs Alex's data call):**

- **Probe/GitHub/Hetzner triplicates:** NOT folded. The five workflows
  using the probe pattern keep their shape until 30-day hit-rate data is
  collected. To gather it: `gh run list --workflow <name> --created
  ">2026-08-27" --json databaseId,conclusion` correlated with job-level
  outcomes (`gh run view <id> --json jobs`), counting how often the `-hz`
  arm actually fires on PRs. If <5%, fold to GH-only + retry. The fold
  decision is explicitly Alex's call after seeing the data.

**Live branch-protection edits Alex must apply** (after merging; the
`check_required_checks_sync.py` live check `[7/7]` will fail until then):

1. ADD `ASHRAE 140 Strict Energy Gate (Issue #1333)` to develop (and main)
   required checks.
2. ADD `Nextest Subset (GH)` to develop (and main) required checks.
3. No removals — the seven folded checks keep their exact names; only
   their hosting workflow changed (`ci-gates.yml` → `physics-pr.yml`),
   which branch protection does not track.
4. Verify with `FLUXION_CHECK_LIVE_PROTECTION=1 python3
   scripts/check_required_checks_sync.py`.

### Follow-up cut: 16 → 9 required checks (2026-09-27, Alex-approved)

Alex asked for the cut list to reach ~9 required checks and approved the
exact list below. Shipped on branch `ci/cut-required-checks-4007`.

**Keep required (9):** `Energy Conservation (GH)`, `ASHRAE 140 Strict
Energy Gate (Issue #1333)`, `Nextest Subset (GH)`,
`Physics-Sim-Cycle-Check (GH)`, `Rustfmt (GH)`, `Clippy (GH)`, `Workspace
Check (GH)` (all always-run, in `required_checks_workflow_only`) plus
`Surrogate Drift Tolerance Gate (Issue #1784)` and `Cargo Deny`
(path-filtered required — in `required_checks`, enforced on PRs touching
their filters, not on live branch protection).

**Demoted to advisory (6)** — workflows keep running, names removed from
`required_checks` (and from `required_checks_workflow_only` where present):
`Docs Hygiene Gate (Issue #2466)`, `Ashrae Cases Cycle Check (GH)`
(Physics-Sim-Cycle-Check stays as the cycle representative), `Cycle
Downward Trend Guard (Issue #2768)`, `Architecture Drift Detection`,
`Module Size (Issue #2878)`, `Crate Size Gate (Issue #2930)`.

**Path-filtered instead of always-required (2):**
- `Cargo Deny` → moved from `ci-gates.yml` to new
  `.github/workflows/cargo-deny.yml` with `pull_request.paths` on
  `**/Cargo.toml`, `**/Cargo.lock`, `deny.toml`.
- `Surrogate Drift Tolerance Gate (Issue #1784)` → moved from
  `ci-gates.yml` to new `.github/workflows/surrogate-drift.yml` with
  `pull_request.paths` on `**/surrogate*.rs`, `**/Surrogate*.rs`,
  `models/**`, `tests/reference_data/surrogate/**`.
- `ci-gates.yml` now hosts only the advisory `test` nextest matrix (its
  `needs: [surrogate-drift-gate, deny]` removed); header rewritten.

**MSRV → nightly authority:** `msrv.yml` gained a `0 1 * * *` UTC schedule;
removed from per-PR required; added to `ci.nightly_authority` (now 9).
Still runs advisory on Cargo-manifest PRs and pushes.

**`release_gates.yaml`:** `required_checks` 16 → 9;
`required_checks_workflow_only` 11 → 7; `nightly_authority` 8 → 9;
`workflow_index` repointed (surrogate-drift, cargo-deny) and annotated
(advisory demotions, MSRV nightly).

**Live branch-protection edits Alex must apply** (develop; main has no
required checks configured): starting from the current 9 live contexts,
REMOVE `Cargo Deny`, `Surrogate Drift Tolerance Gate (Issue #1784)`,
`Ashrae Cases Cycle Check (GH)`, `Cycle Downward Trend Guard (Issue
#2768)`; ADD `ASHRAE 140 Strict Energy Gate (Issue #1333)` and `Nextest
Subset (GH)` (still pending from the #4007 PR). Net: 9 → 7. Verify with
`FLUXION_CHECK_LIVE_PROTECTION=1 python3
scripts/check_required_checks_sync.py`.
