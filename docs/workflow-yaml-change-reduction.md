# Reducing Workflow-YAML Changes (Issue #4254 Investigation)

> Investigation for Issue #4254: why do `.github/workflows/*.yml` files change
> so often, and what would let those changes ship through normal OAuth pushes
> instead of one-time classic PATs with `workflow` scope? Method: every commit
> on `develop` touching `.github/workflows/` was classified by diff intent.
> Headline finding: half of all workflow changes are CI-job *repairs* whose
> logic lives inline in YAML step bodies; moving that logic into `scripts/`
> is the single biggest PAT-reduction lever. The Gate Registry (already
> landed, `.github/ci-gates.yaml`) covers most of the second-biggest reason.

## Summary

| # | Reason workflow YAML changes | Commits | Share |
|---|------------------------------|--------:|------:|
| R2 | CI-job repair (fix a job's step logic, thresholds, filters, pins inside steps) | 49 | 49% |
| R1 | Gate-set change (add/remove/re-wire a gate or required check) | 20 | 20% |
| R5 | Rename / contract change (job, listener, or artifact contract renamed) | 13 | 13% |
| R3 | Pin / image bump (action or base-image version bumps) | 8 | 8% |
| R4 | CI topology change (phase-gating, listener pattern, lane splits) | 6 | 6% |
| R0 | Merge commit, no direct content | 4 | 4% |

## Method and evidence

- Command: `git log --oneline develop -- .github/workflows/` at
  `a0446c26` (2026-10-22) returns 101 commits; excluding the root commit,
  100 commits were classified by inspecting each diff and assigning one
  primary reason. (Supervisor re-ran the count independently: 101.)
- R2 examples: `ci(determinism): if-no-files-found error at source`
  (#4545), `fix(ci): perf lane bulk -E filter — fold exclusion into one
  expression` (#4450), a long run of `fix(docker): …` step repairs
  (#4133–#4220), `ci: gate Bench (all-features) on kafka feature
  consumability` (#4381). The common shape: a threshold, a filter
  expression, a tool pin, or a shell incantation edited *inside* a step.
- R1 examples: `ci: cut required checks 16 -> 9 for issue #4007` (#4100),
  `feat(ci): implement Gate Registry Pattern (Refs #4254)`, the phase-gated
  CI consolidation (#4098) and its listener-pattern follow-ups (#3810).
- R3 examples: the Dependabot `github-actions` group bumps (#3775, #3914,
  #4129, #4305, #4397) and base-image/MSRV digest refreshes.
- R5 examples: `#4294: rename the misnamed hvac_bestest target`,
  `fix(soak): write last_run_id as a string (#3286 gate contract)` (#4483).

## Proposals, per reason

### R2 — Extract step logic into `scripts/` (biggest lever)

Most R2 edits change *what a step does*, not *that the job exists*: shell
logic, tolerance/threshold literals, nextest filter expressions, Docker
build/push argument repair. All of that can live in versioned scripts under
`scripts/` (Python or shell), which are ordinary files — editable and
pushable via OAuth, unit-testable by the existing scripts test suite, and
reviewable without touching a workflow.

Concrete refactors:

1. One thin wrapper per heavy job: the YAML step becomes
   `run: python3 scripts/ci/<job>.py` (or `.sh` where shell is natural),
   with inputs passed as env/args. The Docker workflow cluster (at least
   eight R2 commits) is the first candidate: extract its build/tag/push
   logic into `scripts/docker/`.
2. Move numeric literals (tolerances, thresholds, sample counts) into
   `release_gates.yaml` / `.github/ci-gates.yaml` and have the scripts read
   them — several R2 commits are single-number changes.
3. Keep genuinely declarative YAML (triggers, permissions, matrix shape)
   in the workflow; only imperative logic moves.

Estimated PAT impact: if the step-logic share of R2 moves to scripts,
roughly 25–35 of the last 100 workflow-touching commits would not have
touched a workflow file — a **~25–35% reduction** in PAT-requiring pushes.
This is an estimate from the classification, not a guarantee; some R2
repairs are inseparable from job wiring.

### R1 — Finish the Gate Registry pattern (partially done)

The Gate Registry already landed (`47ba8eddb`, explicitly Refs #4254):
gates are declared in `.github/ci-gates.yaml`, and adding/tweaking a gate
is now a config edit, not a workflow edit. The remaining R1 edits are
*new-job wiring* (a brand-new gate needs a job somewhere) and one-off
required-check cuts (#4100).

Proposal: extend the registry so a new gate is (a) a registry entry plus
(b) a script under `scripts/`, executed by the existing generic
gate-runner workflow — no per-gate workflow job at all. Then a gate-set
change touches YAML only when the runner itself changes.

Estimated PAT impact: up to ~15 of the last 100 commits (**~10–15%**),
declining in practice because the registry already absorbed the common
case mid-window; expect the forward-looking R1 rate to be lower than the
historical 20% even without further work.

### R5 — Accept most renames; generate the mechanical ones

Renames/contracts (13%) are semantic changes a human should review in
place; they are not the friction driver. The one mechanical sub-pattern —
the path-filtered "listener" job names that must mirror their gate names
(#3830 consumed 16 files on a YAML-comment truncation fix) — could be
generated from the Gate Registry instead of hand-maintained. Estimated
impact if generated: **~3–5%**.

### R3 — No action: pins are already low-friction

Action/image bumps (8%) arrive as Dependabot group PRs; Alex does not
mint a PAT to author them. SHA-pinning is a deliberate supply-chain
control (see `fix(ci): SHA-pin actions in gate-runner.yml`) and should
not be traded for dynamic refs. Estimated PAT impact of any change here:
**~0%** — recorded so nobody "optimizes" it later.

### R4 — No action: topology migrations are one-offs

Phase-gating, the listener pattern, and lane splits (6%) were a bounded
migration series that has largely settled. Consolidating workflows now
would *increase* blast radius per change. No refactor proposed.

## Consolidated estimate

| Lever | Est. share of workflow pushes avoided |
|-------|--------------------------------------:|
| R2 step-logic extraction into `scripts/` | ~25–35% |
| R1 Gate Registry completion (generic runner) | ~10–15% |
| R5 listener generation from the registry | ~3–5% |
| **Total realistic reduction** | **~35–45%** |

Branch-protection side effect: proposals R1/R5 keep required-check *names*
stable (one generic gate-runner check instead of per-gate checks), so the
protected-branch check list — itself a PAT-adjacent friction source —
changes rarely.

## What not to do

- Do not consolidate workflows merely to reduce file count; it raises the
  blast radius of every remaining edit.
- Do not replace pinned action SHAs with floating refs to make bumps
  "free"; the pinning is intentional.
- Do not move trigger/permission declarations out of YAML; GitHub requires
  them there, and they are not what changes.

## Provenance

Classification data was produced by the token-plan worker for this issue
(100 commits, per-commit reason table in its run transcript); the
supervisor independently re-verified the 101-commit population and wrote
this document from that table after the worker hit its step cap during
analysis. Estimates are the supervisor's, labelled as estimates above.
