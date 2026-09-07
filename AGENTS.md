# Repository Guidelines

Fluxion is a Rust-first building-energy-modeling engine with Python and Node bindings. The current v1.3 work is ASHRAE 140 validation; `SCORECARD.md` is generated and is the canonical status snapshot. Its known failing validation gates are structural `LIMIT-*` gaps in `docs/KNOWN_ISSUES.md`, not permission to tune constants or relax baselines.

## Read Before Changing Boundaries

- Read `ARCHITECTURE.md` before physics or cross-module work; read `CODEBASE_MAP.md` for FFI ownership, serialization, and workspace boundaries. `scripts/check_architecture_drift.py` enforces alignment.
- `RULES.md` is binding for physics: execute Python for numerical reasoning, preserve energy balance, and never hardcode/tune outputs to pass ASHRAE 140.
- The root package is also workspace package `fluxion`; `default-members = ["."]`, so bare `cargo build/test/check` does **not** cover siblings. Use `--workspace` or `-p <package>`.
- `fluxion-core` is a dependency-light leaf. It must not import root `sim`, `physics`, `ai`, or `validation` modules. `scripts/check_ashrae_cases_cycle.py` and `scripts/check_physics_sim_cycle.py` enforce cycle rules. Keep `src/sim/assembly.rs` and `src/sim/multi_node_thermal.rs` as re-export shims.
- Main swap points are `HeatConductionSolver` (`src/physics/solver_trait.rs`), `VentilationSchedule` (`src/sim/ventilation.rs`), and `ThermalModelTrait` (`src/sim/thermal_model.rs`). `BatchOracle` parallelizes populations only; do not add nested inner-loop Rayon parallelism.
- `fluxion-fluid/` is the acausal HVAC crate; it is not `fluxion-core/src/fluid/`. `fluxion-mcp` is a separate package and is run with `cargo run -p fluxion-mcp`, not `--bin`.

## Commands That Are Easy to Guess Wrong

```bash
./scripts/disk-space-check.sh                         # before large builds/orchestration; 10 GB minimum
cargo test --workspace --exclude fluxion-tauri        # ALL workspace tests (~6299 passed / 110 ignored across 331 test binaries as of HEAD 7d4a1f1); --exclude fluxion-tauri is required because fluxion-tauri's proc-macro build needs `npm run build` in fluxion-tauri/frontend/ to materialise ../frontend/dist (Issue #3126)
cargo test                                           # root crate only (NOT the full suite)
cargo test -p fluxion <test_name>                    # one named test
cargo test --test zone_balance_eplus_isolation       # energy-conservation gate
cargo test --test ashrae_140_validation              # ASHRAE suite (one of several ashrae_140 binaries)
cargo test --test integration-cli                    # CLI behavior/stub guards
cargo test --profile ci                              # faster local iteration profile
cargo check --workspace                              # all workspace siblings
cargo test -p fluxion-mcp                            # MCP package
cargo test --features ort                            # ONNX runtime is opt-in
```

**Test suite overview:** `cargo test --workspace --exclude fluxion-tauri` runs ~6299 passed / 110 ignored tests across 331 test binaries (HEAD 7d4a1f1); `cargo test --lib` runs ~3923 passed / 0 ignored tests in the root crate. The ASHRAE 140 suite is distributed across multiple `--test` binaries (run `ls tests/ashrae_140*.rs` to see them all). Running `cargo test` without `--workspace` only runs the root crate tests and misses the full suite.

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
- Release REST builds reject insecure public bind/auth combinations unless `FLUXION_REST_ALLOW_INSECURE=1`; TLS proxy auth requires `FLUXION_REST_TRUSTED_PROXIES`. Untrusted forwarded IP headers are intentionally ignored. MQTT is TLS-only by default; insecure transport requires the explicit opt-out guard.

## Documentation and Repository Hygiene

- Do not hand-edit generated `SCORECARD.md`; regenerate with `python3 scripts/generate_scorecard.py`. The `scorecard-drift` workflow auto-regenerates on PRs (issue #3128), so manual regen is only needed for local citations or to fix drift that leaks through to `develop`/`main`. See `docs/agents/scorecard-regen.md`.
- Every `docs/**/*.md` file needs the 7-line summary block at lines 2–8. After adding/removing docs, run `python3 scripts/generate_doc_inventory.py` and commit `docs/doc-inventory.md`.
- Verify docs/root hygiene with `python3 scripts/check_docs_summaries.py`, `python3 scripts/check_doc_inventory_fresh.py`, and `python3 scripts/check_root_hygiene.py`. Root scratch reports/blobs are rejected; use `tmp/`. Agent runtime directories such as `.agents/`, `.opencode/`, and `.planning/worktrees/` are gitignored and must never be committed.
- Keep `.cargo/audit.toml` and `deny.toml` advisory exceptions synchronized. Do not increase the duplicate-version budget without documenting the unavoidable dependency.

## Git and CI Workflow

Branch from and target `develop`; `main` accepts release PRs only from `develop`. Never push directly to either branch. Use Conventional Commits (`fix(scope):`, `feat(scope):`, `test(scope):`, `docs(scope):`, etc.), and include `Closes #N` or `Fixes #N` in PR bodies.

`release_gates.yaml -> ci.required_checks` is canonical. Required names include `(GH)` suffixes; synchronize that file before renaming workflow jobs. In the GH-probe/Hetzner-overflow pattern, a cancelled probe is the fallback trigger, not the final failure; judge the `(GH)` or overflow job result.

For workflow-only PRs (touching only `scripts/`, `.github/workflows/`, or `docs/`), use `release_gates.yaml -> ci.required_checks_workflow_only` (19 checks). Path-filtered checks (`Docs Hygiene Gate`, `Architecture Drift Detection`, `Crate Size Gate`, `MSRV Check`) cannot run on such PRs by design. See `docs/ci/branch-protection-strict-mode.md` for the full rationale (Issue #3142).

