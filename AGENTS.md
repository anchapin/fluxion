# Repository Guidelines

Fluxion is a Rust-first building-energy-modeling engine with Python and Node bindings. The current v1.3 work is ASHRAE 140 validation; `SCORECARD.md` is generated and is the canonical status snapshot. Its known failing validation gates are structural `LIMIT-*` gaps in `docs/KNOWN_ISSUES.md`, not permission to tune constants or relax baselines.

## Read Before Changing Boundaries

- Read `ARCHITECTURE.md` before physics or cross-module work; read `CODEBASE_MAP.md` for FFI ownership, serialization, and workspace boundaries. `scripts/check_architecture_drift.py` enforces alignment.
- `RULES.md` is binding for physics: execute Python for numerical reasoning, preserve energy balance, and never hardcode/tune outputs to pass ASHRAE 140.
- The root package is also workspace package `fluxion`; `default-members = ["."]`, so bare `cargo build/test/check` does **not** cover siblings. Use `--workspace` or `-p <package>`.
- `fluxion-core` is a dependency-light leaf. It must not import root `sim`, `physics`, `ai`, or `validation` modules. `scripts/check_ashrae_cases_cycle.py` and `scripts/check_physics_sim_cycle.py` enforce cycle rules. Keep `src/sim/assembly.rs` and `src/sim/multi_node_thermal.rs` as re-export shims.
- Main swap points are `HeatConductionSolver` (`src/physics/solver_trait.rs`), `VentilationSchedule` (`src/sim/ventilation.rs`), `ThermalSelector` (`src/sim/thermal_selector.rs`), and `ThermalModelTrait` (`src/sim/thermal_model.rs`). `BatchOracle` parallelizes populations only; do not add nested inner-loop Rayon parallelism.
- **Phase A8 (Issue #3291, merged via PR #3482) — `GaugeSolver` is the unconditional `selector` default zone solver (and the unconditional `production` default once `--features gauge-solver` is on, pending §LIMIT-21 closure).** `ThermalSelector::default()` resolves to `ZoneSolverKind::Gauge` (`src/sim/thermal_selector.rs`). With `--features gauge-solver` the dispatcher's `step_physics` runs the gauge path with no fall-through to legacy 5R1C/9R4C — a missing gauge backend is a programming error and panics loudly rather than silently switching solvers. `FiveROneC` and `NineRFourC` remain available as explicit opt-in legacy paths via `ThermalSelector`. The `gauge-solver` cargo feature is intentionally retained as the production-path gate pending §LIMIT-21 (Issue #3297) closure — the β-soak gate (Issue #3286, `#3286 β-soak` convention in CI comment threads) is currently at 0/30 nights green; the unconditional production default applies once the feature is on. See `ARCHITECTURE.md` Module 5 swap-point update and `docs/KNOWN_ISSUES.md` §LIMIT-21.
- `fluxion-fluid/` is the acausal HVAC crate; it is not `fluxion-core/src/fluid/`. `fluxion-mcp` is a separate package and is run with `cargo run -p fluxion-mcp`, not `--bin`.

## Commands That Are Easy to Guess Wrong

> **Workspace-scope rule (Issue #3587)** — The root crate is also workspace package `fluxion` with `default-members = ["."]`. The bare `cargo test` therefore runs the root crate ONLY (4,007 lib tests + the root `[[test]]` entries) and silently SKIPS the remaining 4,398 sibling-crate tests. **Always use the workspace form below** unless you have a deliberate reason to scope to one crate. See `docs/agents/workspace-scope.md` for the full rationale and `.githooks/pre-push` for the opt-in pre-push gate.

```bash
./scripts/disk-space-check.sh                         # before large builds/orchestration; 10 GB minimum (also the first half of the pre-push pair below)
./scripts/disk-space-check.sh && ./scripts/ci-local.sh   # pre-push: disk-space gate then curated `act` suite (default = scorecard-drift + docs-hygiene + architecture_drift + scripts-tests, ~3m total); catches CI-shape failures locally before burning a GH runner slot — see .actrc for image/arch pinning and docs/ci/local-validation.md for the full workflow (Issue #3577, PR #3568)
cargo test --workspace --exclude fluxion-tauri --no-fail-fast   # ★ DEVELOPER-FACING DEFAULT ★ — full workspace (8,405 tests / 119 ignored) minus fluxion-tauri (its proc-macro build needs `npm run build` in fluxion-tauri/frontend/ to materialise ../frontend/dist — Issue #3126); this is the LOCAL-DEBUG equivalent of the CI `cargo nextest run --workspace --all-targets --test-threads=2 --no-fail-fast` (Issue #3366 / ADR-0014, PR #3369); both runners share `.config/nextest.toml::concurrency = 2` defaults. Install `.githooks/pre-push` (see below) to enforce this on `git push`.
cargo nextest run --workspace --all-targets --test-threads=2 --no-fail-fast   # canonical CI command (Issue #3366 / ADR-0014, PR #3369); see docs/ci/nextest-rollout.md for rationale and .github/workflows/rust-tests.yml::test for the actual matrix invocation
cargo test                                           # ⚠ ROOT CRATE ONLY (NOT the full suite) — `default-members = ["."]` makes bare `cargo test` skip the 4,398 sibling-crate tests; do NOT rely on this as a green-light signal
cargo test -p fluxion <test_name>                    # one named test (intentional single-crate scope)
cargo test --test zone_balance_eplus_isolation       # energy-conservation gate
cargo test --test ashrae_140_validation              # ASHRAE suite (one of several ashrae_140 binaries)
cargo test --test integration-cli                    # CLI behavior/stub guards
cargo test --profile ci                              # faster local iteration profile
cargo check --workspace                              # all workspace siblings (build-only, no tests)
cargo test -p fluxion-mcp                            # MCP package
cargo test --features ort                            # ONNX runtime is opt-in
python3 scripts/generate_test_inventory.py --verify   # regenerate tests/test_inventory.json — Issue #3442; cross-checks AST counts against cargo test -- --list
python3 scripts/check_test_inventory_drift.py        # Issue #3442 drift gate; fails PRs that grow the test suite above the documented baseline without an explicit baseline bump
```

**Pre-push gate (opt-in, Issue #3587 acceptance criterion #b)** — to stop a bare-`cargo test` local green from landing a PR that breaks siblings, install the opt-in hook once:

```bash
ln -s ../../.githooks/pre-push .git/hooks/pre-push    # opt-in; symlink-relative so a fresh clone picks it up
# Skip a single push when the local suite is already green another way:
FLUXION_SKIP_PRE_PUSH_TESTS=1 git push
```

The hook invokes `cargo test --workspace --exclude fluxion-tauri --no-fail-fast` and aborts the push on non-zero exit; it never modifies the working tree, never invokes `cargo nextest` (that's a CI-only runner choice, see Issue #3366), and never tightens test tolerance bands.

**Test runner policy (Issue #3366 / ADR-0014, merged via PR #3369 on 2026-09-06):** CI uses `cargo nextest run` with per-binary `--test-threads=2` (matching GH free-runner vCPU count); local developers may continue to use `cargo test --workspace --exclude fluxion-tauri --no-fail-fast` (see the workspace-scope rule above — promoted via Issue #3587) for the local-debug equivalent. The nextest rollout runbook (`docs/ci/nextest-rollout.md`) is the source of truth for the audit, re-audit triggers, and `.config/nextest.toml` overrides. Do **not** relax ASHRAE 140 / energy-conservation / `h_tr_em` / surrogate-drift tolerance bands to compensate for any nextest race — tighten `.config/nextest.toml` instead (Issue #3366 §"Step 1 — Audit").

**Test suite overview (Issue #3442 — citation source-of-truth = `tests/test_inventory.json`):** `cargo test --workspace --exclude fluxion-tauri` runs the inventory below; verify mode cross-checks against `cargo test --workspace --exclude fluxion-tauri -- --list` and prefers the cargo counts. The headline numbers are derived from the head commit's verified run; treat them as informational and cite the inventory JSON for the durable record. The ASHRAE 140 suite is distributed across multiple `--test` binaries (run `ls tests/ashrae_140*.rs` to see them all). Running `cargo test` without `--workspace` only runs the root crate tests and misses the full suite.

| Source | Suite | Tests | Ignored | Notes |
|---|---|---|---|---|
| `cargo test --lib` | root crate unit tests | 4,007 | 8 | matches `tests/test_inventory.json::totals.lib_tests_root` (the AST-regex committed inventory) |
| `cargo test --workspace --exclude fluxion-tauri` | full workspace (lib + integration + bin) | 8,405 | 119 | `tests/test_inventory.json::totals.workspace_tests` (AST-regex; `cargo test --workspace -- --list` produces a slightly different count — see `tests/reference_data/test_inventory_baseline.json::verified_at_head_baseline` for the cargo-verified figures) |
| AST-regex inventory | committed in `tests/test_inventory.json` | 8,405 | 119 | non-runtime snapshot, used by the drift gate (`--no-verify`) |
| Cargo auto-discovered test binaries | `<crate>/tests/*.rs` + `[[test]] path = "tests/<sub>/<foo>.rs"` | 308 | n/a | matches `tests/test_inventory.json::totals.test_binaries` |

Refreshing the canonical inventory (Issue #3442 acceptance): run `python3 scripts/generate_test_inventory.py --verify` locally and commit the regenerated `tests/test_inventory.json`. The drift gate (next section) will fail any test-adding PR that does not bump the baseline ratchet in the same PR.

CI-quality order is significant:

```bash
cargo fmt -- --check
cargo clippy --lib -- -D warnings
cargo audit
cargo deny check
```

Install hooks with `pip install pre-commit && pre-commit install && pre-commit install --hook-type commit-msg -f`; run `pre-commit run --all-files`. Hooks use Ruff, not Black; mypy is intentionally disabled because of pre-existing errors. Docs hooks are `stages: [manual]`, so run the scripts below directly when docs change.

Bindings are feature-gated: `maturin develop` for Python; run `npm run build` inside `npm/` for NAPI. Root default features are empty. Full Loom/mutation suites need about 32 GB RAM; do not launch them casually.

## Physics and Validation Guardrails

- Canonical exterior film coefficient is `18.3 W/m²K` (`fluxion-core/src/construction.rs`). Do not reintroduce legacy `29.3` in production computation paths; `tests/regression_exterior_film_unification.rs` guards this.
- Diagnose bottom-up: Weather → Solar → Conduction → Ventilation → Zone Balance. Prefer the module-isolation tests under `tests/` before system-level ASHRAE runs.
- Never raise `tests/reference_data/zone_balance/strict_energy_gate_baseline.json` to hide Case 600/900 regressions. Check `SCORECARD.md`, `docs/ASHRAE140_RESULTS.md`, and `docs/KNOWN_ISSUES.md` before classifying a validation failure.
- The `fluxion` help includes intentionally stubbed paths (direct simulation, workflow/measure execution, and diagnostic case ranges). They must fail non-zero with issue `#2947`; do not turn them into silent success. The removed `parallel-issue-workflow` binary must not be revived without its fail-closed source-diff requirement.

## Toolchain, Security, and Generated Artifacts

- MSRV is Rust `1.98.0`. A bump must update all workspace manifests plus the toolchain SHA/field in `.github/workflows/msrv.yml`.
- `.rustfmt.toml` pins edition 2021. Stable rustfmt cannot exclude files; preserve `#[rustfmt::skip]` on generated fixture items such as `tests/per_tilt_per_azimuth_fixture_data.rs`.
- ONNX loads are fail-closed against `<model>.sha256`; `FLUXION_ONNX_MODEL_SIGNATURE` is only an explicit digest override. Do not bypass `verify_onnx_signature`.
- Release REST builds reject insecure public bind/auth combinations unless `FLUXION_REST_ALLOW_INSECURE=1`; TLS proxy auth requires `FLUXION_REST_TRUSTED_PROXIES`. Untrusted forwarded IP headers are intentionally ignored. The MQTT telemetry consumer (`fluxion-twin`) is always TLS with validated certificates — there is no runtime bypass (#3162).

### Environment Variables

Runtime configuration knobs referenced above. All are fail-closed by design — there are no runtime bypasses:

- `FLUXION_ONNX_MODEL_SIGNATURE` — explicit digest override for the `<model>.sha256` verifier. Leave unset in production; ONNX loads are rejected if the on-disk signature does not match (`verify_onnx_signature`).
- `FLUXION_REST_ALLOW_INSECURE=1` — opt-in escape hatch for release REST builds that bind publicly without TLS or accepted-auth combinations. Leave unset for any release reachable from the public internet.
- `FLUXION_REST_TRUSTED_PROXIES` — comma-separated CIDR list of proxies whose `Forwarded`/`X-Forwarded-*` headers may be honored for client-IP and TLS-terminated-auth context. Untrusted forwarded headers are always ignored outside this set.

## Documentation and Repository Hygiene

- Do not hand-edit generated `SCORECARD.md`; regenerate with `python3 scripts/generate_scorecard.py`. The `scorecard-drift` workflow auto-regenerates on PRs (issue #3128), so manual regen is only needed for local citations or to fix drift that leaks through to `develop`/`main`. See `docs/agents/scorecard-regen.md`.
- Every `docs/**/*.md` file needs the 7-line summary block at lines 2–8. After adding/removing docs, run `python3 scripts/generate_doc_inventory.py` and commit `docs/doc-inventory.md`.
- Verify docs/root hygiene with `python3 scripts/check_docs_summaries.py`, `python3 scripts/check_doc_inventory_fresh.py`, and `python3 scripts/check_root_hygiene.py`. Root scratch reports/blobs are rejected; use `tmp/`. For the two known stale root CSV operator artifacts that the gate cannot untrack (e.g. `case_900ff_profile_hourly.csv`, `hourly_output.csv`), run `./scripts/cleanup_root_strays.sh` (Issue #3438; dry-run by default, add `--apply --yes` to delete) — the script's predicates mirror this gate's gitignored-and-untracked check, so it cannot remove anything you would not lose to `rm`. Agent runtime directories such as `.agents/`, `.opencode/`, `.planning/worktrees/`, and the XDG `.local/` state dir (e.g. `.local/state/gh/device-id` written by `gh auth login`; issue #3626) are gitignored and must never be committed.
- Keep `.cargo/audit.toml` and `deny.toml` advisory exceptions synchronized. Do not increase the duplicate-version budget without documenting the unavoidable dependency.
- **Test-inventory drift gate (Issue #3442)** — `tests/test_inventory.json` is the canonical test-count citation; do not edit by hand. The drift gate (`scripts/check_test_inventory_drift.py`, wired into `scripts-tests.yml`) rejects a PR whose live counts grow above the `BASELINE_*` ratchet constants in the gate. To add tests in a PR, regenerate via `python3 scripts/generate_test_inventory.py --verify` and update both `tests/test_inventory.json` AND `tests/reference_data/test_inventory_baseline.json` (commit both; the drift gate's per-PR `--update-baseline` invocation rewrites the second). The mirror of the `BASELINE_KNOWN_ORPHANS` / `BASELINE_WIRED_BUT_DEAD` ratchets in `scripts/check_orphan_modules.py` (Issues #3459 / #3458): shrinking the test suite is the only authorised baseline change that does NOT require both files in lock-step.
- **Quarantined tests (Issue #3629)** — any PR that adds an `#[ignore = "awaiting #N"]` (or `LIMIT-*`) attribute must add a matching row to `tests/QUARANTINE.md` in the same PR (Category, Issue, Owner, Un-Ignore Criteria, Status). The auditor is `python3 scripts/generate_quarantine_registry.py --strict`; the downward-only ratchets are `BASELINE_ORPHANED_IGNORES` and `BASELINE_GHOST_ROWS` in `scripts/generate_quarantine_registry.py` (Issue #3443). See `CONTRIBUTING.md` §"Quarantined tests" for the full contributor-facing protocol and `docs/KNOWN_ISSUES.md` §1.1 / `docs/ci/nextest-rollout.md` ~L215 for the operational context.

## Git and CI Workflow

Branch from and target `develop`; `main` accepts release PRs only from `develop`. Never push directly to either branch. Use Conventional Commits (`fix(scope):`, `feat(scope):`, `test(scope):`, `docs(scope):`, etc.), and include `Closes #N` or `Fixes #N` in PR bodies.

`release_gates.yaml -> ci.required_checks` is canonical. Required names include `(GH)` suffixes; synchronize that file before renaming workflow jobs. In the GH-probe/Hetzner-overflow pattern, a cancelled probe is the fallback trigger, not the final failure; judge the `(GH)` or overflow job result.

For workflow-only PRs (touching only `scripts/`, `.github/workflows/`, or `docs/`), use `release_gates.yaml -> ci.required_checks_workflow_only` (26 checks). Path-filtered checks (`Docs Hygiene Gate`, `Architecture Drift Detection`, `Module Size (Issue #2878)`, `Crate Size Gate`, `MSRV Check`) cannot run on such PRs by design. See `docs/ci/branch-protection-strict-mode.md` for the full rationale (Issue #3142).

