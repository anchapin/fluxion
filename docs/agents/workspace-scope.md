# Workspace-scope rule (Issue #3587)

The Fluxion root crate is also workspace package `fluxion` with `default-members = ["."]` (root `Cargo.toml`).
A bare `cargo test` therefore runs the root crate ONLY (4,311 lib tests + the explicit `[[test]]` entries) and silently SKIPS the remaining 4,369 sibling-crate tests.
The developer-facing default is `cargo test --workspace --exclude fluxion-tauri --no-fail-fast`; bare `cargo test` is footgun.
The `.githooks/pre-push` hook is the opt-in companion that enforces this on `git push` so a root-only local green cannot land a PR that breaks siblings.
This rule is the canonical reading of `AGENTS.md` §"Commands That Are Easy to Guess Wrong" and `CODEBASE_MAP.md` §"Build & Test Commands" — both were edited to point here as part of the Issue #3587 acceptance criteria.

## Why it matters

Running the wrong command locally produces a false-positive green that hides regressions until CI. The known failure modes:

- **Sibling-crate regressions** — a change to `fluxion-core`, `fluxion-city`, `fluxion-fluid`, `fluxion-grid`, `fluxion-mcp`, or `fluxion-wasm` that breaks its own tests is invisible until CI. The `fluxion-fluid` energy-conservation gate (`tests/integration/test_fluid_energy_conservation.rs`), the MCP package's JSON-RPC harness, the grid solver, and the city radiation solver have all been bitten by this in the past.
- **Default-feature drift** — CI exercises a wider feature graph than the bare root-crate `cargo test` (see `.githooks/cargo-check-features.sh` for the local mirror). A bare-`cargo test` green does not imply a `--features wiring-tracing,multi-zone` green.
- **Tauri proc-macro trap** — `cargo test --workspace` WITHOUT `--exclude fluxion-tauri` fails locally because `fluxion-tauri`'s proc-macro build needs `npm run build` in `fluxion-tauri/frontend/` to materialise `../frontend/dist` (Issue #3126). The `--exclude fluxion-tauri` flag is therefore load-bearing in the developer-facing default; CI handles it via a separate matrix entry, not via the exclusion.

## The developer-facing default

```bash
cargo test --workspace --exclude fluxion-tauri --no-fail-fast
```

| Flag | Reason |
|------|--------|
| `--workspace` | exercises every member of `[workspace] members = [...]` in root `Cargo.toml` (root + 7 siblings). |
| `--exclude fluxion-tauri` | skips the Tauri crate locally; CI uses a separate matrix entry. Issue #3126. |
| `--no-fail-fast` | surfaces every failing crate instead of bailing on the first; matches CI's behaviour. |

This command is the local-debug equivalent of the CI `cargo nextest run --workspace --all-targets --test-threads=2 --no-fail-fast` (Issue #3366 / ADR-0014, PR #3369). Both runners share `.config/nextest.toml::concurrency = 2` defaults; the test inventory (`tests/test_inventory.json`) reports 8,680 tests / 108 ignored for the workspace run.

## Opt-in pre-push gate

```bash
# One-time per clone:
ln -s ../../.githooks/pre-push .git/hooks/pre-push

# Verify it's wired:
ls -l .git/hooks/pre-push    # -> .githooks/pre-push

# Skip a single push when the local suite is already green another way:
FLUXION_SKIP_PRE_PUSH_TESTS=1 git push
```

The hook is **opt-in by design** — it is not installed automatically so that ad-hoc pushes (e.g. WIP branches, hotfix sketches) are not blocked. It is intentionally minimal:

- Runs the developer-facing default above (NOT `cargo nextest` — that's CI-only).
- Never modifies the working tree, never tightens test tolerance bands, never tunes baselines (RULES.md forbids that).
- Has a single escape hatch (`FLUXION_SKIP_PRE_PUSH_TESTS=1`).

## What this rule does NOT change

- `cargo test` (bare, root-only) still works as a fast iteration loop for engineers who know they only changed the root crate. It is now annotated `⚠ ROOT CRATE ONLY (NOT the full suite)` in `AGENTS.md` so a casual reader sees the warning.
- `cargo test -p fluxion <test_name>` is unchanged — single-crate scope is an intentional, narrow request.
- `cargo check --workspace` is unchanged — build-only, no test execution; useful when you only need a compile signal.
- The CI command (`cargo nextest run --workspace --all-targets --test-threads=2 --no-fail-fast`) is unchanged. The Issue #3366 nextest rollout is the authoritative runner for CI; this rule is purely about the local-developer surface.

## References

- `AGENTS.md` §"Read Before Changing Boundaries" (workspace-scope paragraph, line 9).
- `AGENTS.md` §"Commands That Are Easy to Guess Wrong" (Issue #3587 edit promoted the workspace form to first position).
- `CODEBASE_MAP.md` §"Build & Test Commands" (line ~940 area) — updated to point at this doc.
- `.githooks/pre-push` — the opt-in hook script.
- `docs/ci/nextest-rollout.md` — the CI-side nextest rollout runbook (Issue #3366).
- Issue #3587 — original acceptance criteria (a) / (b) / (c).
- PR #3568 — `scripts/ci-local.sh` (the `act`-based pre-push for non-Rust checks; this hook complements, not replaces, it).
