# Cleanup Pass — Fluxion (post Aug 12–14 agent session)

**Date:** 2026-08-14
**Window:** last 48 hours (default anchor = `2a77424`; HEAD = `42f2e89`)
**Scope anchor:** v1.3 "Blind ASHRAE 140 Validation" — CHANGELOG.md:44 (Series 800–810 HVAC,
Series 195–470 diagnostic cases), validation min pass rate 60% (40% for patches),
multi-zone perf gate #2772, strict ±15% annual-energy gate #1333.
**Branch:** `develop` (origin/develop, **clean working tree**)
**Mode:** subtractive verification + drift audit

---

## Status

**YELLOW** — Working tree is clean and the bulk of hygiene was already done in two
deliberate commits inside the 48h window (`b4c714e`, `e17fe55`, `0290741`, `fcdfd9a`).
**One regression was introduced by the window and must be fixed before release:**
`fluxion-behavior` build break from `SmallRng::from_entropy()` after the `rand 0.8 → 0.9`
bump in `3cd78c9`. This is not a cleanup job — it is a one-package repair PR.

## Summary

| Window | Files | Insertions | Deletions |
|---|---|---|---|
| Working tree (staged + unstaged) | 0 | 0 | 0 |
| 48h commits (`2a77424^..HEAD`) | 721 | 10,540 | 20,625 |
| 48h commits excluding `.sdd/` (swept) | **155** | **10,540** | **6,843** |

- **Working tree:** clean (`git status --porcelain` → empty, no untracked, no diffs).
- **Root hygiene gate:** expected PASS — `scripts/check_root_hygiene.py` and the new
  dotfile/dotdir sub-check (commit `134a956`) swept 33 root scratch artifacts in the
  window.
- **Architecture drift gate:** expected PASS — `scripts/check_architecture_drift.py`
  (cycle / drift / micro-architecture checks).
- **Known-issues freshness gate:** last `KNOWN_ISSUES.md` update was in the window; OK.
- **Doc-inventory freshness gate:** expected PASS — `f996d75` repaired the 11 broken
  references and `b203265` enumerated git-tracked docs.
- **Cargo build (root):** `cargo check --lib` → **0 errors, 1 warning** (unrelated).
- **Cargo build (workspace):** `cargo check --workspace` → **5 errors** (PRE-EXISTING
  REGRESSION — see *Build regression* below).
- **rustfmt:** `cargo fmt --check` → clean (no diffs).
- **48h commits:** 50 commits, all map to v1.3 requirements OR legitimate supporting
  infrastructure. **No scope creep detected.**

## Build regression (the only finding)

`cargo check --workspace` fails with **5 errors** in `fluxion-behavior`, all of the
form `SmallRng::from_entropy()` on `rand 0.9.5`:

```
error[E0599]: no associated function or constant named `from_entropy` found for struct `SmallRng`
  --> fluxion-behavior/src/occupancy.rs:520:39
  --> fluxion-behavior/src/occupancy.rs:579:29
  --> fluxion-behavior/src/occupancy.rs:633:29
  --> fluxion-behavior/src/plug_loads.rs:52:28
  --> fluxion-behavior/src/plug_loads.rs:61:28
```

**Root cause:** commit `3cd78c9 chore(deps): cargo-major + sha256_hex fix (#2960)` (in
this window) bumped `rand 0.8 → 0.9`. In `rand 0.9`, `SmallRng::from_entropy()` was
removed; the API is now `SeedableRng::from_rng(&mut thread_rng())` or `from_seed(...)`.

**Files untouched in the 48h window** (last edits predate the cargo-major bump):
`fluxion-behavior/src/occupancy.rs`, `fluxion-behavior/src/plug_loads.rs`. The
sha256_hex consolidation in `95ddafc` only touched the root `src/` tree and `fluxion-core`.

**Recommended fix (single small PR, not part of this cleanup):**

```rust
// before
rng: SmallRng::from_entropy(),
// after
rng: SmallRng::from_rng(&mut rand::rng()),
```

Add `use rand::SeedableRng;` if not already imported. The `fluxion-behavior` crate
already pulls in `rand` 0.9.5 (`cargo tree -p fluxion-behavior` confirms).

**Why this is not in the cleanup pass:** the cleanup protocol is subtraction, not
editing. The regression was introduced by a deliberate deps bump, not by drift; the
fix is a one-file mechanical change deserving its own PR with a regression test.

---

## Files / categories

### Removed (working tree)

None — working tree is clean.

### Removed (committed in 48h window — already done)

| Window | Path | Reason |
|---|---|---|
| [commit `e17fe55`] | `tests_tmp_dummy.onnx` | scratch ONNX file at repo root; gate `check_root_hygiene.py` extended |
| [commit `0290741`] | `.automaker/features/.../feature.json.bak{1,2,3}` × 6 | leftover editor backups; `.gitignore` extended with `*.bak[0-9]*` |
| [commit `fcdfd9a`] | `.sdd/` × 566 files | runtime dir added to `.gitignore` per AGENTS.md rule; tracked state swept |
| [commit `b4c714e`] | `bem-engineer.skill`, `bem-engineer/{SKILL.md,evals/,scripts/,templates/}`, `validation_tests.toml`, `fluxion-mcp/Cargo.lock` | root scratch artifacts (issue #2819); `bem-engineer/` is a duplicated skill bundle |

The `.sdd/` sweep is the single biggest deletion in the window
(`563 files, 13,782 deletions`). `.sdd/` was already configured as a local
runtime dir in AGENTS.md but tracked files survived the existing ignore. `fcdfd9a`
resolved `Check_root_hygiene.py` Issue #2837 by gitignoring the dir and removing
the tracked instances.

### Reverted (commits)

None — every commit in the window is in scope for v1.3 (or in the case of `2a77424`,
a CI-gate fix that survived review).

### Reverted (hunks)

None — no in-progress hunks spotted.

### Consolidated

| Kept | Removed | Reason |
|---|---|---|
| `src/util/mod.rs`, `src/util/sha256_hex.rs` | 4 inline `sha256_hex` implementations in `src/ai/{batch_runner.rs, batch_runner_9r4c.rs, surrogate.rs}` and 2 in `src/validation/{reference_catalog.rs, reference_loader.rs}` | [commit `95ddafc`] consolidation; 5 callers updated. Imports verified: `src/validation/reference_loader.rs`, `src/validation/reference_catalog.rs`, `src/ai/batch_runner.rs`, `src/ai/batch_runner_9r4c.rs`, `src/ai/surrogate.rs` |
| `tests/reference_data/zone_balance/generate_case_800_810_energy.py` | (none — new file) | [commit `c05c6cc`] CHANGELOG.md:44 confirms Series 800–810 is in v1.3 scope (Phase 40-07) |

### Renames

| From | To | Reason |
|---|---|---|
| `src/bin/tdd_validator.rs.broken` | `tools/tdd_validator.rs.broken` | [commit `385c52c`] move out of `src/bin/` so cargo stops trying to build it (it no longer compiles against the current `fluxion::testing::tdd_framework`). `.broken` suffix is the project's "keep for reference" convention. **No callers anywhere in the codebase** (`grep -rn "tdd_validator" -- ':!*.broken' -- ':!*.lock'` → 0 hits). The file still imports a real module (`fluxion::testing::tdd_framework`, which exists at `src/testing/tdd_framework.rs`), so it would compile if moved back. Kept as archive. |

### Kept (intentional / in scope)

| Anchor | Item | Why it stays |
|---|---|---|
| [commit `42f2e89`] | `src/validation/ashrae140/mod.rs` — compute `ComparisonMetrics` from actual benchmark_report | v1.3 validation pass-rate work |
| [commit `827152d`] | `src/bin/fluxion.rs`, `tests/integration/test_cli.rs` — 8 workflow subcommands marked unimplemented | AGENTS.md "CLI Subcommands — Partial Implementation Status" already documents these stubs; gates the implicit "not silently succeeding" contract |
| [commit `54ec9d1`] | `src/validation/case_195_calibration.rs` (456 lines), `src/validation/performance/parallel_executor.rs` (544 lines) | v1.3 ASHRAE 140 Case 195 (diagnostic) + multi-zone perf gate #2772 |
| [commit `0cd5343`] | `scripts/autonomous_parameter_sweep.py` --brief YAML/JSON, `scripts/ci/test_autonomous_parameter_sweep.py`, `scripts/requirements-test.txt` | issue #2951 — coverage ratchet depends on it |
| [commit `e70748d`, `b203265`, `f996d75`, `f3a55ba`] | five new ci gate tests + `check_doc_link_integrity.py` | makes the existing drift audits runnable in pytest harness |
| [commit `b48bdec`, `41072f9`] | `docs/adr/0001-no-parameter-tuning-rule.md`, `docs/adr/0005-acausal-hvac-fluid-port-traits.md` | records the v1.3 architectural decisions |
| [commit `57570ca`] | conditional CUDA skip | efficiency, not scope |
| [commit `3cd78c9`, `d13be8b`, `3b428b2`, `c843a65`, `611523b`, `385c52c`, `f5add18`, `734001d`, `ce0f3b6`] | deps + physics + build-cli fixes | routine hygiene, all map to v1.3 release readiness |
| [commit `939b3fc`] | blind-validation feature for #2748 | core v1.3 scope |
| [commit `00f33a9`] | `docs/agents/v1.3-cleanup-pass-audit.md` (the previous PM pass) | the audit log; keep |
| [commit `fcdfd9a`] | `.sdd/` → `.gitignore` + deletion | already-cleaned, see *Removed* above |
| [add] | `.agents/results/plan-cleanup-followup.json` (added) and `.agents/results/result-pm.md` (modified) | directory is the agent audit log; previous PM run also wrote `result-pm.md` (last modified 2026-08-13). Not drift. |

### Verifications performed

- `cargo fmt --check` → **clean** (no diffs)
- `cargo check --lib` → **0 errors, 1 warning** (profile warning, unrelated)
- `cargo check --workspace` → **5 errors** (see *Build regression*)
- `git status --porcelain` → **empty**
- `git ls-files .sdd/` → **0** (fully untracked after `fcdfd9a`)
- `git ls-files .automaker/ .serena/ .planning/ .jules/ .sisyphus/ .gitnexus/ .superset/ .opencode/ .claude/ CLAUDE.md` → many tracked; these are also listed in AGENTS.md as
  "Local-only runtime dirs (gitignored — never commit, never create at repo root)" but
  some are still tracked. **This is a long-standing inconsistency outside the 48h
  window** — none were added in the window. The cleanup window removed the worst
  offenders (`.sdd/` × 566). The remaining cleanup is a separate project.

### Boundary checks

- No new files in `src/` lack inbound references.
- All `src/util/sha256_hex.rs` callers verified (5 sites).
- The `tdd_validator.rs.broken` orphan has zero callers anywhere it is referenced.
- No `git push --force` or `git reset --hard` was performed.
- No `git revert` was needed — every commit is in scope.

---

## Summary tables (skill "Reporting back" format)

```
Removed (working tree):
  - (none — working tree is clean)

Removed (committed in 48h window):
  - [commit fcdfd9a]   .sdd/ (566 files, 13,782 deletions) — runtime dir, gitignored
  - [commit b4c714e]   bem-engineer.skill, bem-engineer/{SKILL.md,evals,scripts,templates}, validation_tests.toml, fluxion-mcp/Cargo.lock — root scratch
  - [commit e17fe55]   tests_tmp_dummy.onnx — root scratch
  - [commit 0290741]   .automaker/.../feature.json.bak{1,2,3} × 6 — leftover editor backups

Reverted (commits):
  - (none — all 50 commits in 48h window are in v1.3 scope)

Reverted (hunks):
  - (none)

Consolidated:
  - kept src/util/sha256_hex.rs; deleted 6 inline dupes — consolidation refactor 95ddafc

Renames:
  - src/bin/tdd_validator.rs.broken → tools/tdd_validator.rs.broken — moved out of build [commit 385c52c]

Kept (intentional):
  - [commit 42f2e89] src/validation/ashrae140/mod.rs — v1.3 validation
  - [commit 54ec9d1] src/validation/case_195_calibration.rs, src/validation/performance/parallel_executor.rs — Case 195 + multi-zone perf (#2772)
  - [commit 827152d] src/bin/fluxion.rs, tests/integration/test_cli.rs — gated 8 stubbed CLI subcommands (issue #2947)
  - [commit 0cd5343] scripts/autonomous_parameter_sweep.py + pytest — --brief YAML/JSON (#2951)
  - [commit c05c6cc] tests/reference_data/zone_balance/generate_case_800_810_energy.py — Series 800-810 (Phase 40-07)
  - [commits b48bdec, 41072f9] docs/adr/0001, 0005 — v1.3 ADRs
  - [commit 134a956] scripts/check_root_hygiene.py — dotfile/dotdir sub-check
  - [commit f996d75] docs/doc-inventory.md + 11 broken refs fixed — drift hygiene
  - [commit e70748d + 8 more] scripts/ci/test_*.py — pytest harness for ci gates
  - [.agents/results/plan-cleanup-followup.json, result-pm.md] — agent audit log; previous PM run also wrote here
  - [tools/tdd_validator.rs.broken] — moved-from-bin orphan; zero callers; keep as archive per project .broken convention

Verification:
  - tests: not run (full suite not requested; pre-existing workspace build break)
  - fmt:    pass — cargo fmt --check clean
  - clippy: not run (would mix with the build break noise)
  - build (root): pass — cargo check --lib 0 errors
  - build (workspace): FAIL — 5 errors in fluxion-behavior (rand 0.9 API drift, introduced by 3cd78c9)
  - git log --since="48 hours ago": 50 commits scanned, 0 reverted, 1 regression flagged for fix
```

---

## Out-of-scope follow-ups (not in this cleanup pass)

1. **Required fix** — `fluxion-behavior` `SmallRng::from_entropy()` → `from_rng(&mut thread_rng())` on 5 lines. Creator PR for commit `3cd78c9` follow-up.
2. **Optional cleanup** — the remaining AGENTS.md-listed runtime dirs still tracked
   (`.automaker/` 5+ files, `.serena/` 5, `.planning/` many, `.jules/`, `.sisyphus/`,
   `.gitnexus/`, `.superset/`) are inconsistent with the "Local-only runtime dirs
   (gitignored — never commit)" rule. They were not added in this window, so they
   fall outside the 48h cleanup scope per the skill's "do not review commits older
   than 48 hours" boundary, but a follow-up issue could schedule them.
3. **Optional cleanup** — `tools/tdd_validator.rs.broken` has zero callers. If the
   long-term intent is to revive it, the move-out-of-bin is correct; if not, the
   next cleanup pass should delete it. Current `.broken` convention says keep.
