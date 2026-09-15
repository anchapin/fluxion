# Pre-push checklist (wave pipeline)

> **Summary 1/7:** The proportional local validation ladder every implementation sub-agent runs BEFORE `git push` — ratified 2026-09-15 as part of the throughput plan (ADR-0016 companion).
> **Summary 2/7:** Always: `cargo fmt -- --check`, `cargo clippy -p <touched crates> -- -D warnings`, `./scripts/disk-space-check.sh && ./scripts/ci-local.sh` (~4–6 min total).
> **Summary 3/7:** Code changes (`src/**`, sibling crates): add scoped `cargo test -p <crate> <module>` for changed modules. Physics path class (`src/**`, `fluxion-core/**`, `fluxion-fluid/**`, `tests/**`, `models/**`, `weather/**`): add the relevant scoped ASHRAE suites (lane-2 preview), e.g. `cargo test --test all_tests ashrae_140_<module>::`.
> **Summary 4/7:** Tests added: regenerate the inventory — `python3 scripts/generate_test_inventory.py --verify` — and commit BOTH `tests/test_inventory.json` AND `tests/reference_data/test_inventory_baseline.json` in lock-step (the #1 ratchet trap).
> **Summary 5/7:** `#[ignore]` added: add the matching `tests/QUARANTINE.md` row in the same PR (Category, Issue, Owner, Un-Ignore Criteria, Status) per the #3629 protocol.
> **Summary 6/7:** Docs changed: `python3 scripts/check_docs_summaries.py` (7-line summary block at lines 2–8) and `python3 scripts/generate_doc_inventory.py` (commit `docs/doc-inventory.md`).
> **Summary 7/7:** Fix-push etiquette: when a CI check fails, reproduce locally first, then batch ALL known fixes into ONE push — never push per-check (unbatched pushes cause ADR-0015 cancel-storms). Acceptance: fix-loop rate ≤ 15 %, cancelled-run share ≤ 3 %.

## Why this exists

Baseline 2026-09: 33 % of merged PRs needed 2–6 commits; each failed push costs a ~25–40 min CI round-trip plus diagnosis. The ladder front-loads the cheap catches; the etiquette stops wasting runner slots.

## The ladder

| PR touches | Run locally before push | Cost |
|---|---|---|
| anything | `cargo fmt -- --check`; `cargo clippy -p <touched crates> -- -D warnings`; `./scripts/disk-space-check.sh && ./scripts/ci-local.sh` | ~4–6 min |
| code | + scoped `cargo test -p <crate> <module>` for changed modules | ~2–10 min |
| physics path class | + scoped ASHRAE suites (`cargo test --test all_tests ashrae_140_<module>::`) | ~5–15 min |
| tests added | + `python3 scripts/generate_test_inventory.py --verify`; commit BOTH inventory files in lock-step | ~2 min |
| `#[ignore]` added | + `tests/QUARANTINE.md` row in the same PR | ~1 min |
| docs | + `python3 scripts/check_docs_summaries.py`; `python3 scripts/generate_doc_inventory.py` (commit the inventory) | ~1 min |

Deliberately **excluded**: the full `cargo test --workspace --exclude fluxion-tauri` run — lane-2 CI runs the real suites in parallel; duplicating them locally would double agent wall-clock per physics PR.

## Fix-push etiquette (CI sub-agents)

1. Fetch the failing check's log (`gh run view <id> --log-failed`).
2. Reproduce locally (`act` for the workflow, or the underlying `scripts/check_*.py` / `cargo test` command).
3. Fix, re-run the local reproduction, and re-run the ladder rows relevant to what you changed.
4. Push ONCE with all known fixes. If a sibling check is likely to fail for the same root cause, fix it in the same push.

## Lane prediction

Know your lane before pushing (ADR-0016): a PR whose diff is docs/deps/scripts-only triggers lane 1 (~20–27 min CI); anything touching the physics path class triggers the full lane-2 gauntlet (~35–40 min). Scope your local ladder accordingly — and never narrow the diff just to dodge lane 2.
