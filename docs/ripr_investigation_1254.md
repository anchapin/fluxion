# Investigation: ripr for Static Mutation Seam Analysis

**Issue:** [#1254 — Investigate incorporating ripr for static mutation seam analysis](https://github.com/anchapin/fluxion/issues/1254)
**Status:** Investigation complete — recommendation: **adopt ripr as a per-PR advisory pre-filter**
**Date:** 2026-06-24
**Author:** fluxion tooling

---

## 1. TL;DR

[ripr](https://github.com/EffortlessMetrics/ripr) is a **static** mutation-exposure
analyzer: it reads a PR diff and names which *changed* behavior the current tests
reach but do not actually check, **without compiling or running any mutants**.
Because it never invokes the compiler, its memory footprint is a small fraction
of `cargo-mutants`, which OOMs on fluxion (~28 GB RAM vs. the 7 GB CI runners
have, see Issue #1244).

**Recommendation:** use the two tools in complementary lanes:

| Lane | Tool | When | Runner | Cost |
|------|------|------|--------|------|
| Per-PR advisory gate | **ripr** | every pull request | `ubuntu-latest` (7 GB) | seconds, static |
| Confirmation run | `cargo-mutants` | nightly / `workflow_dispatch` | `ubuntu-latest-8-cores` (32 GB) | hours, runs mutants |

ripr is **advisory** — it does not replace mutation testing, coverage, or a merge
gate. It is the cheaper "draft-time" question that surfaces test-oracle gaps
*before* the expensive mutation run, and it is the only one of the two that fits
on the standard CI runners fluxion uses for per-PR gates.

---

## 2. What ripr is

- **Crate:** [`ripr`](https://crates.io/crates/ripr) v0.10.0 (published 2026-06-15)
- **Repo:** https://github.com/EffortlessMetrics/ripr/
- **License:** MIT OR Apache-2.0
- **MSRV:** Rust **1.95** (fluxion pins `stable` in `rust-toolchain.toml`, which is
  ≥ 1.95 as of this writing — compatible)
- **Category (self-described):** *Static Mutation Exposure Analysis* —
  "static oracle-gap analysis for diff-derived mutation probes"
- **Maturity:** alpha. The Rust analyzer loop is the mature path; Python repair
  routing is "usable alpha"; TypeScript is a preview.

### Core model

ripr is **diff-scoped**. Given a base..head diff it performs a four-evidence
analysis on each changed behavior and emits a **gap** where the tests are too weak
to notice the change breaking:

| Evidence | Question |
|----------|----------|
| Reachability | are there related tests? |
| Infection | can the change alter branch/value behavior? |
| Propagation | does the changed value influence an observable output? |
| Revealability | do tests assert on that output with a discriminating value? |

A gap that fails Revealability is an "oracle gap" — the behavior is *reached* but
not *checked*. ripr then routes one focused, test-only repair:

- **gap** — changed behavior lacking a discriminator (stable canonical ID)
- **card** — the repair: what to assert, where, why the proof is weak
- **packet** — a bounded, source-edit-free work order (allowed/forbidden files)
- **verify** — the focused command that checks the repair (`cargo test …`)
- **receipt** — before/after record of whether the gap closed

### How it works under the hood (memory-relevant)

ripr parses source with `ra_ap_syntax` — **rust-analyzer's syntax-tree crate only**.
It does **not** run the type checker, borrow checker, or compiler, and it does
**not** execute the test binary. This is the key reason it is light: the parts of
rust-analyzer/cargo-mutants that dominate memory (type inference over the whole
crate graph, repeated compilation) are entirely absent.

Dependencies of note: `ra_ap_syntax`, `tower-lsp-server` (optional LSP sidecar),
`serde`/`serde_json`, `tokio`, `toml`; optional `oxc` (TypeScript) and
`rustpython-parser` (Python).

---

## 3. ripr vs. cargo-mutants

| Dimension | cargo-mutants | ripr |
|-----------|---------------|------|
| Approach | **Dynamic** — mutates source, recompiles, runs tests | **Static** — parses diff, analyzes evidence |
| Runs mutants? | Yes (killed / survived / timeout) | **No** |
| Output | mutation score, surviving mutants | oracle gaps + next test to add |
| Scope | whole crate / module | **diff-scoped** (changed behavior only) |
| Compiles the crate? | Yes, many times | **No** |
| Memory on fluxion | **~28 GB → OOM on 7 GB runners** (#1244) | Syntax-only; expected to be a small fraction (see §4) |
| Wall-clock on fluxion | hours | seconds–low minutes |
| Position | confirmation engine | draft-time advisor |
| Cost to add to per-PR CI | too high (needs 32 GB runner) | low (fits 7 GB runner) |

ripr's own framing is explicit and matches this split:

> a real mutation runner like `cargo-mutants` confirms it under execution when the
> change is ready. Coverage stays the execution-surface signal; ripr is the cheaper
> draft-time question between them.

These are **not competitors**. ripr cannot produce a mutation score or list
surviving mutants; cargo-mutants cannot run on fluxion's per-PR runners. The two
answer different questions at different points in the pipeline.

---

## 4. RAM / time requirements

> **Honesty note:** the issue brief estimates "~1.5 GB RAM" for ripr. That figure
> is **not stated in ripr's own documentation** and was not independently
> measured on fluxion during this investigation (cargo-mutants is what is
> installed; ripr is not yet integrated). Treat the number as an *expectation to
> verify*, not a fact.

What we *can* assert with confidence:

1. ripr does **not** compile or execute the crate, so it avoids the two largest
   memory consumers in cargo-mutants (full type-checking of the crate graph and
   repeated `rustc`/test-binary invocations).
2. Its parser (`ra_ap_syntax`) is the lightweight, syntax-only slice of
   rust-analyzer — not the IDE engine whose memory footprint is the well-known
   heavy part.
3. It is diff-scoped, so the analyzed surface is bounded by the size of the PR,
   not the whole codebase.

**Action item before merging the integration:** measure actual peak RSS on a real
fluxion PR with `/usr/bin/time -v ripr … | grep Maximum resident`. Update this
document and the CI plan with the measured figure. Until measured, the safe claim
is *"dramatically lower than cargo-mutants; expected to fit on a 7 GB runner."*

Expected wall-clock: seconds to low single-digit minutes for a typical PR, since
it is bounded by diff size and parse cost (no compilation).

---

## 5. Fluxion context (current state)

- **Crate structure:** fluxion is a **single crate** (`fluxion`, `cdylib` + `rlib`),
  *not* a Cargo workspace. (There is no `docs/mutation_testing_crate_split.md` —
  that document does not exist in the repo.) This is relevant: cargo-mutants must
  type-check the entire single crate's complex type hierarchies at once, which is
  what drives its ~28 GB peak. ripr sidesteps this entirely.
- **OOM history (Issue #1244):** `cargo-mutants` 27.x needs ~28 GB to analyze
  fluxion's type hierarchies. Standard `ubuntu-latest` runners have 7 GB.
  Commits `c8480ef`, `e83a35e`, and the revert `8b4012e` all tried to scope
  mutation testing to smaller modules; final state `7c47842` **disabled** CI
  mutation testing and restricted it to `workflow_dispatch` on a 32 GB runner.
- **Current CI layout:**
  - `.github/workflows/rust-tests.yml` — per-PR fast gate on `ubuntu-latest` (7 GB).
  - `.github/workflows/ci.yml` — main-merge jobs (Python, integration).
  - `.github/workflows/mutation-testing.yml` — **disabled**, `workflow_dispatch`
    only, runs on `ubuntu-latest-8-cores` (32 GB), runs `cargo mutants`.
- **Net effect today:** fluxion has *no* mutation signal on per-PR CI, and the
  confirmation run is manual. This is exactly the gap ripr is designed to fill:
  cheap, advisory, per-PR.

---

## 6. Proposed CI integration plan

### 6.1 Phase 1 — advisory ripr job on per-PR CI (standard runner)

Add a new lightweight workflow `.github/workflows/ripr.yml`:

- **Trigger:** `pull_request` (same as `rust-tests.yml`).
- **Runner:** `ubuntu-latest` (7 GB) — or `${{ vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}` to honor the existing self-hosted routing.
- **Steps:** checkout (full history for the diff), install stable Rust, `cargo install ripr --locked`, run ripr diff-scoped against the PR base, upload the gap artifact + post an **advisory** PR comment (never failing).
- **Gate semantics:** `continue-on-error: true` initially. ripr is advisory; it must
  not block merges until the team has triaged its signal-to-noise on fluxion.

The job posts a summary comment listing the top gap(s) and the suggested test, plus
an artifact with the full packet. A follow-up issue is filed per actionable gap.

### 6.2 Phase 2 — promote to soft gate (after triage)

Once ripr's gaps are consistently actionable on fluxion (few false positives),
flip the job from advisory to a **soft gate**: fail only on high-severity gaps in
physics-critical modules (`src/physics/`, `src/sim/`), keep `continue-on-error`
for everything else. Decision deferred to a separate issue after Phase 1 data.

### 6.3 cargo-mutants stays where it is

Keep `.github/workflows/mutation-testing.yml` as the confirmation engine on the
32 GB runner, scheduled **nightly** (add `schedule: - cron:`) instead of purely
manual. ripr's per-PR gaps become the prioritized input list for the nightly
cargo-mutants run — focus mutants on modules where ripr flagged weak oracles.

### 6.4 Routing summary

```
PR opened ──> rust-tests.yml (7 GB, hard gate)
          ──> ripr.yml        (7 GB, advisory, names test-oracle gaps)
                                     │
                                     ▼
nightly  ──> mutation-testing.yml (32 GB, cargo-mutants confirmation)
```

---

## 7. Commands to run ripr on fluxion

Install (one-time; ripr requires Rust ≥ 1.95, which fluxion's `stable` satisfies):

```bash
cargo install ripr --locked
```

First-run (the intended zero-config loop — names the single top repairable gap):

```bash
ripr first-pr --root . --base origin/main --head HEAD
```

Diff-scoped pilot over the whole PR (full gap list):

```bash
ripr pilot --root .
```

CI integration setup (generates the GitHub advisory wiring):

```bash
ripr init --ci github
```

Agent/automation status (for the fluxion-mcp / coding-agent loop):

```bash
ripr agent status --root .
```

`ripr.toml` is **optional** — the zero-config run is the intended first interface.
If fluxion needs to scope/ignore paths, a minimal `ripr.toml` can be added later
(out of scope for this investigation; ripr is no-config by default).

---

## 8. Recommended workflow

1. **Per PR (cheap, every time):** `ripr pilot` on the 7 GB runner, advisory only.
   Reviewer + author see the top test-oracle gap and decide whether to add the
   suggested test before merge. This restores a mutation-*adjacent* signal to
   per-PR CI that cargo-mutants cannot provide.
2. **Nightly (expensive, scheduled):** `cargo mutants` on the 32 GB runner, seeded
   by the modules ripr flagged as weakly-exposed during the day. This is where the
   real killed/survived confirmation happens.
3. **Manual (on demand):** `cargo mutants` via `workflow_dispatch` for a focused
   module before release, exactly as today.
4. **Measure first:** before promoting ripr past advisory, capture peak RSS
   (`/usr/bin/time -v`) and false-positive rate on ~10 real PRs. Update §4 with
   the measured numbers.

---

## 9. Risks & limitations

- **Alpha software.** ripr self-describes as alpha. It is advisory and not a proof
  system; expect some noise. Mitigation: start `continue-on-error`, triage before
  gating.
- **No mutation score.** ripr does not report killed/survived. It cannot replace
  cargo-mutants; it precedes it.
- **Diff-scoped only.** ripr analyzes *changed* behavior. It will not surface
  pre-existing oracle gaps in untouched code (cargo-mutants nightly covers that).
- **MSRV 1.95.** Compatible with fluxion's `stable` pin today, but the CI job must
  not downgrade the toolchain.
- **The ~1.5 GB figure is unverified.** Must be measured on fluxion (see §4).
- **`ra_ap_syntax` version coupling.** ripr pins `ra_ap_syntax ^0.0.330`; very new
  Rust syntax features could lag. Fluxion currently targets edition 2021, so risk
  is low.

---

## 10. Recommendation

**Adopt ripr as a Phase-1 advisory per-PR job.** It directly addresses the
consequence of Issue #1244 — the total loss of any mutation-adjacent signal on
per-PR CI — without requiring the 32 GB runner that makes cargo-mutants
un-runnable there. Pair it with a nightly-scheduled cargo-mutants confirmation
run on the existing 32 GB runner, seeded by ripr's weak-oracle output.

Track follow-up work in dedicated issues:
- [ ] Implement `.github/workflows/ripr.yml` (advisory, per-PR).
- [ ] Measure peak RSS + false-positive rate on 10 PRs; update §4.
- [ ] Schedule `mutation-testing.yml` nightly (cron) instead of manual-only.
- [ ] (Phase 2) Soft-gate ripr on `src/physics/`, `src/sim/` after triage.
