# Scripts Catalog

> **TL;DR**: Complete catalog of every executable script under `scripts/` — CI gate checks, generators, operator tooling, and the `scripts/ci/` pytest harness — with one-line purpose and wiring per script.
> **Counts at head**: 39 `check_*.py` gates | 41 top-level Python tools | 29 shell scripts | 55 `scripts/ci/` pytest files (+fixtures, +1 bash test) | 1 Node drift gate — 167 code files total.
> **Wiring legend**: bare workflow names are `.github/workflows/*.yml` files; `pre-commit (manual)` = `.pre-commit-config.yaml` manual-stage hook; "operator" = intentionally unwired, run locally/manually per AGENTS.md.
> **Source of truth**: purposes merge each script's docstring/argparse help with the AGENTS.md §"CI Gates You Can Run Locally" table; wiring was derived by scanning every workflow for `scripts/` references (verified 2026-09-10 vs. head, post-#3654).
> **Status**: current as of 2026-09-10; the stale 2026-07-13 snapshot and the `in-progress` self-flag (issue #1534 tracking) were replaced by this regeneration (issue #3645).
> **Maintenance contract**: PRs that add, rename, or remove a script under `scripts/` must update this catalog in the same PR; the 7-line summary block is enforced by `scripts/check_docs_summaries.py` (docs-hygiene.yml + pre-commit manual stage), and the `scripts/ci/` pytest harness runs in `scripts-tests.yml` / `python-tests.yml`.
> **Related**: AGENTS.md §"CI Gates You Can Run Locally", `release_gates.yaml` (`ci.required_checks` / `ci.required_checks_workflow_only`), `scripts/README.md` (OSimFlow orchestration subset), `docs/doc-inventory.md`.

---

## Inventory at a glance

| Category | Count | Notes |
|---|---|---|
| CI gate checks (`scripts/check_*.py`) | 39 | One row per gate below; wiring matches AGENTS.md and `release_gates.yaml` |
| Top-level Python tools (`scripts/*.py`, non-gate) | 41 | Generators, verifiers, campaign infra, shared `conftest.py` |
| Shell scripts (`scripts/*.sh`) | 29 | 27 top-level + 2 under `scripts/ci/` |
| `scripts/ci/` pytest harness | 55 pytest files | Plus `__init__.py`, `conftest.py`, and `ci/test_build_pgo.sh` (bash) |
| Node drift gate (`scripts/*.mjs`) | 1 | npm README code-block validation |
| **Total executable scripts** | **167** | 137 `.py` + 29 `.sh` + 1 `.mjs` |

Non-executable support files are listed at the end.

---

## CI gate checks (`scripts/check_*.py` — 39)

Every active gate is wired into a workflow job, a pre-commit (manual) hook, or is an explicit operator diagnostic. The table matches AGENTS.md §"CI Gates You Can Run Locally"; run the matching command locally before opening a PR.

| Script | Purpose | Wiring |
|---|---|---|
| `check_action_pinning.py` | Reject named mutable `@vN` / `@stable` / `@main` action refs (#3530) | scripts-tests.yml |
| `check_architecture_drift.py` | ARCHITECTURE.md ↔ Rust-code drift (traits, modules, cycle-baseline counts; #3460) | architecture_drift.yml |
| `check_ashrae_cases_cycle.py` | `sim ↔ validation` cycle edge-count guard (#1441 / #2495) | rust-tests.yml |
| `check_audit_config_unique.py` | Single canonical `cargo audit` config — rejects stray root `audit.toml` (#2773) | security.yml |
| `check_audit_deny_agree.py` | cargo-audit ignore list ↔ cargo-deny advisory agreement (#3654) | security.yml |
| `check_audit_ignores_fresh.py` | `.cargo/audit.toml` ignore-block removal-condition audit (#2912) | security.yml, rumqttc-upstream.yml |
| `check_beta_soak_gate.py` | β-soak 30-night GaugeSolver production-path gate (Issue #3286) | nightly-ashrae-140-gauge.yml |
| `check_branch_protection_diff.py` | `develop` branch-protection diagnostic vs. `release_gates.yaml` (#3383); never applies PUTs | operator diagnostic (pair with `apply_branch_protection.py`) |
| `check_cli_doc_stubs.py` | `fluxion` CLI stub-path fail-loud contract per issue #2947 (#3550) | docs-hygiene.yml, pre-commit (manual) |
| `check_concurrency_keys.py` | ADR-0015 per-`head_sha` concurrency block on every workflow (#3366 / #3444) | scripts-tests.yml |
| `check_cycle_downward_trend.py` | `sim ↔ validation` coupling must shrink toward zero, not stay frozen (#2768) | rust-tests.yml |
| `check_doc_drift.py` | Stale cycle-claim sentences inside Rust doc-comments (#2895) | architecture_drift.yml, pre-commit (manual) |
| `check_doc_inventory_fresh.py` | `docs/doc-inventory.md` byte-equal to the generator's output (#2765) | docs-hygiene.yml, pre-commit (manual) |
| `check_doc_link_integrity.py` | Markdown `[..](path)` / `<path>` references resolve on disk | docs-hygiene.yml |
| `check_docs_summaries.py` | 7-line summary at lines 2–8 of every `docs/*.md` (#2466) | docs-hygiene.yml, pre-commit (manual) |
| `check_env_setvar_serialized.py` | `ENV_LOCK` guard for `tests/**/*.rs` `std::env::set_var` callers (#3453) | scripts-tests.yml |
| `check_evaluator_dynamic_secure.py` | `fluxion-evaluator` dynamic-loader security acceptance (#3554) | security.yml |
| `check_fluxion_core_dep_budget.py` | `fluxion-core` default-feature dependency-budget regression (#3467) | run locally before changing `fluxion-core/Cargo.toml` (doc cross-ref only) |
| `check_ignore_tracking.py` | `#[ignore]` attribute ↔ issue-reference visibility report (informational) | operator diagnostic (`--by-issue` quarantine triage) |
| `check_known_issues_links.py` | `docs/FAQ.md` + `docs/TROUBLESHOOTING.md` link targets resolve (#2541) | run locally before editing either file |
| `check_known_issues_stale.py` | `docs/KNOWN_ISSUES.md` ≤60d "Last Updated" marker (#3105; skip-if-missing per #1723) | known-issues-stale.yml, rust-tests.yml |
| `check_known_issues_summary.py` | `## Summary` table in `KNOWN_ISSUES.md` matches section headers (#3513) | docs-hygiene.yml |
| `check_module_size.py` | Module-size ratchet — hard ceiling on the largest `.rs` files (#2878 / #3457) | architecture_drift.yml |
| `check_no_ignored_tracked_files.py` | Tracked file must not match any `.gitignore` pattern (#3174 / #3356) | tracked-vs-ignored.yml |
| `check_orphan_modules.py` | Orphan Rust source file detector (#2875 / #3458 / #3459) | architecture_drift.yml, scripts-tests.yml |
| `check_osimflow_coverage.py` | OSimFlow per-file coverage thresholds (#1847 / #1864) | python-tests.yml |
| `check_physics_sim_cycle.py` | `physics ↔ sim` cycle edge-count guard (#2463) | rust-tests.yml |
| `check_pyi_drift.py` | `fluxion.pyi` stub ↔ `#[pyclass]` / `#[pyfunction]` drift (#2509) | run locally before editing Python bindings |
| `check_required_checks_sync.py` | `release_gates.yaml::ci.required_checks` ↔ live `develop` branch protection (#2866 / #3116 / #3123 / #3441) | architecture_drift.yml, required-checks-sync-cron.yml, scripts-tests.yml |
| `check_root_hygiene.py` | Root-level transient-file hygiene, widened beyond `.md` (#3438) | via alias `check_root_md_policy.py` in docs-hygiene.yml |
| `check_root_md_policy.py` | Back-compat alias for `check_root_hygiene.py` (keeps historical filename alive) | docs-hygiene.yml, pre-commit (manual) |
| `check_rumqttc_upstream.py` | rumqttc / rustls-webpki security-cluster upstream watch (#2853) | rumqttc-upstream.yml |
| `check_runner_routing_policy.py` | `FLUXION_LINUX_RUNNER` trust-boundary on PR triggers (#3445 / #3531) | scripts-tests.yml |
| `check_scorecard_data_sources_consistent.py` | SCORECARD headline figures consistent across `performance_history.latest.json` vs. `ASHRAE140_RESULTS.md` (#3535) | run locally before regenerating SCORECARD.md (transitively enforced by scorecard-drift.yml) |
| `check_strict_energy_gate_regression.py` | Strict ±15% ASHRAE 140 annual-energy tolerance regression (#2506 / #3572) | ashrae_140_strict_energy_gate.yml |
| `check_stub_modules.py` | Stub-module detector — future-extraction marker files (#2896) | architecture_drift.yml |
| `check_tdqs_regression.py` | TDQS (Temporal Decision Quality Score) criterion-bench regression | tdqs_regression.yml |
| `check_test_inventory_drift.py` | Test-count ratchet vs. `test_inventory_baseline.json` (#3442) | scripts-tests.yml |
| `check_workflow_pin.py` | Fail on non-SHA-pinned `uses:` in `.github/workflows/*.yml` (#3475) | scripts-tests.yml |

---

## Top-level Python tools (`scripts/*.py`, non-gate — 41)

### Scorecard, release & branch protection

| Script | Purpose | Wiring |
|---|---|---|
| `apply_branch_protection.py` | Idempotent applier for branch-protection required-checks sync (#3386); deliberate counterpart to `check_branch_protection_diff.py` | operator (branch-protection PUTs) |
| `display_gate_status.py` | Display gate status from `gate_status.json` | ashrae_140_validation.yml, performance.yml |
| `generate_quarantine_registry.py` | Quarantine registry synchroniser (#3211, #3393); downward-only ratchets (#3443) | scripts-tests.yml |
| `generate_scorecard.py` | Fluxion Release Scorecard Generator (regenerates `SCORECARD.md`) | scorecard-drift.yml |
| `release_gate_checker.py` | Release gate checker (`--benchmark-gates throughput,latency,…`) | ashrae_140_validation.yml, crate-size.yml, performance.yml, performance_dashboard.yml, pypi-release.yml |

### Docs, inventory & test-suite bookkeeping

| Script | Purpose | Wiring |
|---|---|---|
| `doc_inventory_check.py` | Doc Inventory Check — summary blocks + table/path sync for `docs/doc-inventory.md` | docs-hygiene.yml, pre-commit (manual); shell wrapper: `doc_inventory_check.sh` |
| `generate_doc_inventory.py` | Regenerate `docs/doc-inventory.md` enumerating every `docs/**/*.md` | docs-hygiene.yml |
| `generate_test_inventory.py` | Test-inventory generator (#3442); regenerate `tests/test_inventory.json` with `--verify` | operator (commit regenerated JSON in the same PR) |

### Performance & coverage gates

| Script | Purpose | Wiring |
|---|---|---|
| `coverage_baseline.py` | Record the current coverage as the ratchet baseline for the Code Coverage gate (#1932) | operator |
| `coverage_critical_paths.py` | Per-critical-path coverage analysis + ratchet gate (#1932, #3395) | code-coverage.yml |
| `generate_perf_baseline.py` | Generate/regenerate `tests/perf_baseline.json` | operator |
| `performance_gate.py` | Flags performance regressions >10% on PRs | wrapper `ci/perf-check.sh`; pytest harness (`ci/test_performance_gate.py`) |
| `test_coverage_ratchet.py` | Self-test for the per-critical-path coverage ratchet (#1932, #2533, #2710) | operator / pytest |

### ASHRAE 140 validation, calibration & analysis

| Script | Purpose | Wiring |
|---|---|---|
| `ashrae_benchmark_harness.py` | ASHRAE 140 Benchmark Harness (#1847 / #1488) | ashrae_benchmark_harness.yml |
| `audit_false_confidence.py` | Audit tests for false-confidence issues (pass-without-assertion patterns) | operator |
| `autonomous_parameter_sweep.py` | Autonomous diagnostic parameter sweep for ASHRAE 140 discrepancies (#1450) | operator (coverage-tracked by python-tests.yml) |
| `compare_ff_profiles.py` | Analyze free-float temperature profiles | operator |
| `compare_peak_profiles.py` | Compare Fluxion vs. reference peak profiles | operator |
| `generate_diagnostic_data.py` | Synthetic reference data for ASHRAE 140 Cases 195–470 (diagnostic validation) | operator |
| `generate_monthly_aggregate.py` | Aggregate hourly EnergyPlus reference data into monthly heating/cooling totals (#2748) | operator |
| `generate_reference_data.py` | Synthetic reference data for ASHRAE 140 Cases 800–810 (HVAC equipment) | operator |
| `grid_search_h_si.py` | Grid search for `H_SI` (interior surface convective coefficient) calibration | operator |
| `issue-2448-seasonal-attribution.py` | Issue #2453 / #2448 seasonal-attribution analyser | operator |
| `sweep_h_ms_coeff.py` | Sweep `h_ms_coeff` for Case 900FF to locate the tolerance-satisfying window | operator |
| `verify_gauge_solver_regression.py` | TDD snapshot diff verifier for the `ThermalModelData` god-struct split (#3070) | pytest harness (`ci/test_verify_gauge_solver_regression.py`) |
| `verify_h_tr_em_regression.py` | `h_tr_em` wind-dependent per-step recompute regression verifier (#3549) | h_tr_em_regression_gate.yml |

### Surrogate / ML tooling

| Script | Purpose | Wiring |
|---|---|---|
| `export_onnx.py` | Export and validate ONNX surrogate models (v3.0) | operator |
| `gen_golden_outputs.py` | Generate `tests/surrogate_models/golden/golden_v{version}.json` | operator |
| `gen_registry.py` | Generate `tests/surrogate_models/registry.json` | operator |
| `gen_surrogate_v3_1.py` | Generate Surrogate v3.1.0 ONNX model locally (#1334) | operator |
| `generate_training_data.py` | Synthetic training data generation for surrogate models (v2.1) | operator |
| `train_surrogate.py` | Train surrogate model for a physics component (v3.0) | operator |
| `validate_surrogate.py` | Validate surrogate models against the physics baseline (v3.0) | operator |

### Cloud campaign infrastructure (OSimFlow)

See `scripts/README.md` for the orchestration overview. Coverage floor ≥60% enforced by `check_osimflow_coverage.py`.

| Script | Purpose | Wiring |
|---|---|---|
| `cloud_campaign_manager.py` | AWS/Nomad cloud campaign manager (#1192) | cloud_campaign.yml, scripts-tests.yml |
| `s3_aggregator.py` | S3 aggregator for Fluxion campaigns | cloud_campaign.yml |
| `s3_worker.py` | S3 worker for Fluxion campaigns (#2830) | pytest harness (`ci/test_s3_worker.py`) |
| `state_store.py` | State store for Fluxion campaigns — DynamoDB/Redis/in-memory backends (#1787 / T7.3) | imported by campaign scripts; pytest-covered |
| `sync_planning.py` | Sync Fluxion planning artifacts | operator |

### Sweep & agent tooling

| Script | Purpose | Wiring |
|---|---|---|
| `periodic_sweep.py` | Monthly agent sweep through recent commits | operator |
| `update_concurrency_keys.py` | Mechanical one-shot: replace the `concurrency:` block in every workflow (#3366 / #3444) | operator (one-shot migration) |

### Test fixtures

| Script | Purpose | Wiring |
|---|---|---|
| `conftest.py` | Shared pytest fixtures + path config for the OSimFlow suite (#1847); hermetic AWS fakes | picked up by pytest (python-tests.yml, scripts-tests.yml) |

---

## Shell scripts (29)

### Top level (27)

| Script | Purpose | Wiring |
|---|---|---|
| `agent-lint.sh` | LLM-assisted lint for physics correctness and common bugs | operator |
| `annual_ashrae_revalidation.sh` | Automate the annual ASHRAE 140 re-validation (January ritual, ≥80% blind pass rate) | operator |
| `build_pgo.sh` | Profile-Guided Optimization build pipeline (#2563) | pgo-nightly.yml |
| `build_pgo_common.sh` | Sourceable `pg::` helpers for `build_pgo.sh` | sourced by `build_pgo.sh`; tested by `ci/test_build_pgo.sh` |
| `check_mojo_toolchain.sh` | Advisory Mojo SDK / Modular MAX detect-gate (#2979); never blocks | operator / wave pre-flight |
| `check_pr_closing_refs.sh` | Verify a PR's `closingIssuesReferences` count (wave-orchestrator post-`gh pr create`) | operator / wave sub-agents; pytest: `ci/test_check_pr_closing_refs.py` |
| `ci-local.sh` | Local pre-push CI validation via `act` (curated gate suite, ~3m) | operator (pre-push pair with `disk-space-check.sh`) |
| `cleanup-build.sh` | Free disk space by removing build artifacts | operator |
| `cleanup_root_strays.sh` | Delete stale root transient artifacts tripping `check_root_hygiene.py` (#3438); dry-run default | operator; pytest: `ci/test_cleanup_root_strays.py` |
| `cleanup_stale_worktrees.sh` | Classify + clean stale worktrees and `fix/issue-*` branches (#3069, #3118) | operator; pytest: `ci/test_cleanup_stale_worktrees.py` |
| `disk-space-check.sh` | Check available disk space before large builds/orchestration (10 GB minimum) | fuzz.yml, nightly-ashrae-140-gauge.yml, pgo-nightly.yml; pre-push pair |
| `disk-space-gate.sh` | Warn or exit if disk space is critically low (defaults: warn 10 GB, exit 5 GB) | operator |
| `doc_inventory_check.sh` | Shell wrapper for `doc_inventory_check.py` | operator convenience |
| `end_of_shift_validation.sh` | Comprehensive pre-handoff validation (tests, perf, drift, ASHRAE, mutation, audit, lint) | operator |
| `install_ripr.sh` | Install ripr — static mutation-exposure analyzer (#1254) | operator |
| `memory-budget-gate.sh` | Warn or exit if peak RSS exceeds budget | operator |
| `memory_profile.sh` | Run simulation with dhat heap profiling | operator |
| `mutants_diff_files.sh` | Unified diff of changed Rust files for scoped mutation testing (#1891) | mutation-testing.yml |
| `pin_docker_base_images.sh` | Resolve and pin Docker base-image digests for `Dockerfile` | docker.yml |
| `precommit-lint.sh` | Fluxion pre-commit lint hook | git hook (manual symlink install) |
| `provision-hetzner-runner.sh` | Provision a Hetzner VM as a GitHub Actions self-hosted runner (#3445, #3448; no Docker group) | operator |
| `refresh_known_issues.sh` | Refresh the `Last Updated` date in `docs/KNOWN_ISSUES.md` (`--dry-run` supported) | operator |
| `release_v0.8.0.sh` | v0.8.0 release automation (historical) | operator |
| `run_mutation_testing.sh` | Run cargo-mutants on specified modules (32 GB RAM; do not run casually) | operator / nightly |
| `setup_parallel_worktrees.sh` | Race-safe parallel worktree setup from `origin/develop` (#2489) | operator / wave orchestrator |
| `verify_issues_closed.sh` | Verify an issue is closed via `gh` (wave-orchestrator confirmation) | operator / wave sub-agents |
| `wt-add.sh` | Race-safe single-worktree creation with `--check` self-test (#2489) | operator / wave orchestrator |

### Under `scripts/ci/` (2)

| Script | Purpose | Wiring |
|---|---|---|
| `ci/perf-check.sh` | CI integration wrapper for the performance regression gate (`performance_gate.py`) | invocable post-`cargo test`; pytest harness covers the gate logic |
| `ci/test_build_pgo.sh` | Bash test suite for `build_pgo_common.sh` helpers (no cargo/llvm needed) | operator / pre-commit-runnable |

---

## `scripts/ci/` pytest harness (55 test files)

Runs in `python-tests.yml` (Python 3.10–3.13 matrix, coverage via `scripts/pytest.ini`) and in `scripts-tests.yml` (`python3 -m pytest scripts/`). Shared fixtures live in `ci/conftest.py`; `ci/__init__.py` makes the directory a pytest package.

| Test file | Covers |
|---|---|
| `test_apply_branch_protection.py` | `apply_branch_protection.py` (#3386) |
| `test_ashrae_benchmark_harness.py` | `ashrae_benchmark_harness.py` (#1847) |
| `test_autonomous_parameter_sweep.py` | `autonomous_parameter_sweep.py` (#1847) |
| `test_check_architecture_drift.py` | `check_architecture_drift.py` |
| `test_check_ashrae_cases_cycle.py` | `check_ashrae_cases_cycle.py` (#1441, #2495) |
| `test_check_audit_config_unique.py` | `check_audit_config_unique.py` (#2773) |
| `test_check_audit_deny_agree.py` | `check_audit_deny_agree.py` (#3654) |
| `test_check_audit_ignores_fresh.py` | `check_audit_ignores_fresh.py` (#2912) |
| `test_check_beta_soak_gate.py` | `check_beta_soak_gate.py` (#3286) |
| `test_check_branch_protection_diff.py` | `check_branch_protection_diff.py` (#3426) |
| `test_check_code_coverage_docs_skip.py` | Code Coverage gate docs-only skip (#3662) |
| `test_check_concurrency_keys.py` | `check_concurrency_keys.py` (#3444) |
| `test_check_cycle_downward_trend.py` | `check_cycle_downward_trend.py` (#2768) |
| `test_check_doc_drift.py` | `check_doc_drift.py` (#2895) |
| `test_check_doc_inventory_fresh.py` | `check_doc_inventory_fresh.py` (#2765) |
| `test_check_doc_link_integrity.py` | `check_doc_link_integrity.py` |
| `test_check_docs_summaries.py` | `check_docs_summaries.py` (#2466) |
| `test_check_env_setvar_serialized.py` | `check_env_setvar_serialized.py` (#3453) |
| `test_check_fluxion_core_dep_budget.py` | `check_fluxion_core_dep_budget.py` (#3467) |
| `test_check_ignore_tracking.py` | `check_ignore_tracking.py` |
| `test_check_known_issues_links.py` | `check_known_issues_links.py` |
| `test_check_known_issues_stale.py` | `check_known_issues_stale.py` (#1723) |
| `test_check_module_size.py` | `check_module_size.py` (#2878, #3457) |
| `test_check_no_ignored_tracked_files.py` | `check_no_ignored_tracked_files.py` |
| `test_check_orphan_modules.py` | `check_orphan_modules.py` (#2875) |
| `test_check_osimflow_coverage.py` | `check_osimflow_coverage.py` (#1864) |
| `test_check_physics_sim_cycle.py` | `check_physics_sim_cycle.py` (#2463) |
| `test_check_pr_closing_refs.py` | `check_pr_closing_refs.sh` (wave-orchestrator) |
| `test_check_pyi_drift.py` | `check_pyi_drift.py` (#2509) |
| `test_check_required_checks_sync.py` | `check_required_checks_sync.py` (#2866, #3116) |
| `test_check_root_hygiene.py` | `check_root_hygiene.py` (#2466) |
| `test_check_root_md_policy.py` | `check_root_md_policy.py` (back-compat alias) |
| `test_check_rumqttc_upstream.py` | `check_rumqttc_upstream.py` (#2853) |
| `test_check_strict_energy_gate_regression.py` | `check_strict_energy_gate_regression.py` (#1333, #2506) |
| `test_check_stub_modules.py` | `check_stub_modules.py` (#2896) |
| `test_check_tdqs_regression.py` | `check_tdqs_regression.py` |
| `test_check_test_inventory_drift.py` | `check_test_inventory_drift.py` (#3442) |
| `test_check_workflow_pin.py` | `check_workflow_pin.py` (#3475) |
| `test_cleanup_root_strays.py` | `cleanup_root_strays.sh` (#3438) |
| `test_cleanup_stale_worktrees.py` | `cleanup_stale_worktrees.sh` (#3069, #3118) |
| `test_cloud_campaign_manager.py` | `cloud_campaign_manager.py` (#1847) |
| `test_coverage_critical_paths.py` | `coverage_critical_paths.py` (#3395) |
| `test_generate_diagnostic_data.py` | `generate_diagnostic_data.py` |
| `test_generate_doc_inventory.py` | `generate_doc_inventory.py` (#2765) |
| `test_generate_monthly_aggregate.py` | `generate_monthly_aggregate.py` (#2748) |
| `test_generate_perf_baseline.py` | `generate_perf_baseline.py` |
| `test_generate_quarantine_registry.py` | `generate_quarantine_registry.py` (#3211, #3393, #3443) |
| `test_generate_reference_data.py` | `generate_reference_data.py` |
| `test_generate_scorecard.py` | `generate_scorecard.py` (#2496, #3436) |
| `test_generate_training_data.py` | `generate_training_data.py` |
| `test_performance_gate.py` | `performance_gate.py` (PR-time benchmark gate) |
| `test_release_gate_checker.py` | `release_gate_checker.py` (#505) |
| `test_s3_worker.py` | `s3_worker.py` (#2830) |
| `test_verify_gauge_solver_regression.py` | `verify_gauge_solver_regression.py` (#3070) |
| `test_verify_h_tr_em_regression.py` | `verify_h_tr_em_regression.py` (#3549) |

---

## Node drift gate

| Script | Purpose | Wiring |
|---|---|---|
| `check_npm_readme_codeblocks.mjs` | Extract and validate every fenced code block in `npm/README.md` (#2511) | docs-hygiene.yml |

---

## Support files (non-executable)

| File | Purpose |
|---|---|
| `README.md` | OSimFlow Python-orchestration subset overview (#1847) |
| `beta_soak_admin_allowlist.txt` | Admin allowlist consumed by `check_beta_soak_gate.py` (#3286) |
| `cycle_baseline_history.json` | Historical `sim ↔ validation` cycle edge counts (cycle guards) |
| `pytest.ini` | pytest config for `scripts/ci/` (test paths, coverage options) |
| `requirements-test.txt` | Pinned test dependencies for the scripts pytest harness |
| `trait_contract_baseline.json` | Trait-contract baseline consumed by `check_architecture_drift.py` |

---

## Usage notes

```bash
# Run every CI gate check that does not need special credentials
for f in scripts/check_*.py; do python3 "$f"; done   # some gates are expected to fail locally pre-PR-fix

# Run the scripts pytest harness (what CI runs)
python3 -m pytest scripts/                          # scripts-tests.yml
pytest scripts/ci/ -c scripts/pytest.ini            # python-tests.yml (coverage matrix)

# Pre-push validation pair (AGENTS.md §"Commands That Are Easy to Guess Wrong")
./scripts/disk-space-check.sh && ./scripts/ci-local.sh
```

Prerequisites: Python 3.10+, `pytest` + `pytest-asyncio` + `pytest-cov` (see `scripts/requirements-test.txt`), `gh` CLI for the PR/issue helpers, and the Rust toolchain for mutation/PGO scripts. AWS credentials are only needed for the cloud-campaign scripts (tests use hermetic fakes).
