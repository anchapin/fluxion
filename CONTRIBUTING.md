# Contributing to Fluxion

Contribution guide for the Fluxion building energy modeling engine.
Read this before submitting PRs or branching strategies.
Covers: PR workflow, hotfix process, merge strategy, branch conventions.
Companion to RULES.md (coding rules) and CODEBASE_MAP.md (code navigation).
Status: Active — follows --no-ff merge policy and expedited hotfix process.
Action: Use `gh pr create --base develop` for all changes; never push directly to main or develop.

## Quick Rules

1. **Never force-push `main` or `develop`**
2. **`develop` is the default branch** — create all feature branches from `develop` and target all PRs to `develop`
3. **All changes go through PR** - even hotfixes
4. **CI must pass** before merging
5. **Use `--no-ff` merges** to preserve history

## Workflow

### Normal Changes
1. Update from develop and create a feature branch: `git checkout develop && git pull && git checkout -b fix/issue-123`
2. Make changes, commit
3. Push and create PR targeting develop: `gh pr create --base develop`
4. Get review approval
5. Merge via PR (not squash-merge)

### Hotfixes (Unblocked CI)
1. Create branch from develop: `git checkout develop && git pull && git checkout -b hotfix/urgent-fix`
2. Make minimal fix
3. Create PR targeting develop: `gh pr create --base develop`
4. Request expedited review
5. Merge after approval

### When `main` is Broken
1. **Do NOT force-push to fix it**
2. Create a `hotfix/broken-main` branch from `develop`
3. Apply minimal fix (revert broken commits or restore working versions)
4. Create PR targeting `develop` with clear explanation
5. Get emergency review approval
6. Merge via PR to `develop`, then fast-forward `develop` → `main` via release PR

## Branch Protection

- **`develop`** (default branch): requires PR review; direct pushes are blocked.
- **`main`** (release branch): requires PR review and passing CI; PRs to `main` are only accepted from the `develop` branch.
- Protected branches cannot be force-pushed.
- Branch protection and the `protect-main-branch.yml` workflow enforce these rules automatically.

## Commit Messages

Use conventional commits:
- `fix(scope): description` for bug fixes
- `feat(scope): description` for new features
- `refactor(scope): description` for refactoring
- `test(scope): description` for tests

## Testing

- All tests must pass before merge
- Add tests for new functionality
- Update tests when changing behavior

## Build Notes

### Debug builds on disk-space-constrained systems

Debug builds (`cargo build` without `--release`) of large targets such as
`fluxion-rest` may crash during linking with a SIGSEGV in rust-lld on systems
with limited disk space or high memory pressure. This is an environmental issue
— release builds (`cargo build --release`) work correctly and CI uses release
builds by default.

If you encounter a linker segfault during a debug build, use release builds
for local development: `cargo build --release` / `cargo test --release`.
See also: `docs/KNOWN_ISSUES.md` §CI-02 (issue #2297).

### Profile-Guided Optimization (PGO) build

The crate ships with a PGO pipeline (issue #2563) that typically yields
10–20 % throughput improvement on compute-bound paths such as the 5R1C/9R4C
solvers and the conduction transfer functions.

The pipeline is implemented as a single shell script that can be run
locally and in CI. It expects a release of `llvm-tools-preview` (for
`llvm-profdata`).

```bash
# One-shot: profile-generate → train → merge → profile-use
./scripts/build_pgo.sh --clean
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--pgo-dir DIR` | Where to store profile data (default: `target/pgo`) |
| `--train-workload CMD` | Override the training workload (default: `cargo test --profile release --test ashrae_140_validation -- --nocapture`) |
| `--skip-generate` | Reuse an existing instrumented binary |
| `--skip-train` | Skip the training workload (use existing `.profraw` files) |
| `--skip-use` | Generate-only; do not produce the optimized binary |
| `--clean` | Wipe `target/pgo/` before starting |

The pipeline produces a PGO-optimized release binary in `target/release/`
alongside the merged profile data at `target/pgo/merged/profdata`. Build
logs are captured under `target/pgo/logs/`.

For manual reuse of the merged profile data:

```bash
RUSTFLAGS="-Cprofile-use=$(pwd)/target/pgo/merged/profdata" \
    cargo build --profile release-pgo
```

A dedicated `[profile.release-pgo]` Cargo profile is provided with
`lto = "fat"` and `codegen-units = 1` — the PGO instrumentation flags are
passed via `RUSTFLAGS` because Cargo profiles cannot express PGO directly.
See `Cargo.toml` for the exact settings.

The nightly `pgo-nightly.yml` GitHub Actions workflow runs the same
script at 02:30 UTC and uploads the optimized binaries plus profile data
as artifacts. Trigger it manually via `gh workflow run pgo-nightly.yml`
when you want to validate a change end-to-end.

## Code Style

### Rust formatting

- `cargo fmt -- --check` is a required CI check.
- The workspace has a `.rustfmt.toml` that pins `edition = "2021"` (rustfmt 1.9-stable defaults to 2015 which fails on async/await and `?` syntax).
- **rustfmt 1.9-stable does not support the `exclude` config option** (that requires rustfmt-nightly with `unstable_features = true`). When you have auto-generated data files containing thousands of literal array elements that you do NOT want rustfmt to reformat, mark each item with `#[rustfmt::skip]`. Example:
  ```rust
  #[rustfmt::skip]
  pub const FIXTURE_DATA: [f64; 1000] = [/* ... */];
  ```
  See `tests/per_tilt_per_azimuth_fixture_data.rs` for a working example.
- If `cargo fmt --check` fails on your PR with many unrelated drift items, the drift is pre-existing on `develop`. Rebase onto `develop` first; if the drift persists after rebase, file a follow-up issue — do NOT auto-fix 200 files of mechanical drift as part of a feature PR.

### Avoid scope creep on CI failures

If your PR's CI fails on a check that is not in scope for your change (e.g., `ASHRAE 140 Strict Energy Gate`, `MultiZoneValidator` lib tests, `Cargo Audit` advisories in dependencies you didn't touch):
1. **First** verify the failure is pre-existing on `develop` HEAD without your changes (`git stash && cargo ...` or check out `develop` and reproduce).
2. **If pre-existing**: file a separate follow-up issue (label `bug`, link your PR in the body). Do NOT bloat your PR diff with the unrelated fix.
3. **If introduced by your PR**: fix it in scope or STOP and report.

## Disk Space Requirements

Minimum disk space requirements for local development:

| Requirement | Space | Notes |
|------------|-------|-------|
| Minimum | 10 GB | Sufficient for basic build/test cycle |
| Recommended | 50 GB | Needed for release builds, mutation testing, validation runs |

**Critical operations requiring significant space:**
- `cargo build --release` — full release build
- `cargo test --features ort` — with ONNX runtime
- Mutation testing — requires 32 GB RAM + significant disk for results
- Large validation runs with ASHRAE 140 cases

**Free up disk space:**
```bash
# Remove build artifacts
cargo clean

# Remove large generated files
rm -rf mutation_testing_results/
rm -rf validation_artifacts.zip crossval_logs.zip
rm -rf test_results/

# Check space before operations
./scripts/disk-space-check.sh
```

Disk space exhaustion during wave orchestration can cause credential lock failures, PR creation failures, and git ref lock failures.

## KNOWN_ISSUES.md Maintenance

`docs/KNOWN_ISSUES.md` is audited by CI for staleness. The CI gate
(`scripts/check_known_issues_stale.py`) fails if the `*Last Updated: YYYY-MM-DD*`
line is more than **60 days** old.

A weekly GitHub Actions workflow (`.github/workflows/known-issues-stale.yml`)
monitors the file and creates an issue when it exceeds **45 days** old, giving
ahead-of-time notice before the CI gate fails.

### Refresh Workflow

When the staleness issue is triggered:

1. **Review the document** — check each section (BASE, SOLAR, FREE, TEMP,
   MULTI, LIMIT) for accuracy and add notes about any recent fixes or new
   issues discovered since the last refresh.
2. **Run the refresh script** — this updates the `*Last Updated:*` date
   in-place:
   ```bash
   bash scripts/refresh_known_issues.sh
   ```
   Use `--dry-run` to preview the change without modifying the file.
3. **Open a PR** with the refreshed date and a brief summary of section
   updates (or lack thereof).

### Why This Matters

The document is the primary reference for AI agents and engineers attributing
validation failures — citing stale numbers can misdirect investigations.
Keeping it current avoids "known issues" being reopened unnecessarily.

## SCORECARD.md Maintenance

`SCORECARD.md` is generated by `scripts/generate_scorecard.py` from three
committed sources: `docs/ASHRAE140_RESULTS.md`, `release_gates.yaml`, and
`README.md`. The `Scorecard Drift Gate` workflow enforces consistency on
every PR via `python3 scripts/generate_scorecard.py --check`.

**Since issue #3128**, the gate auto-regenerates `SCORECARD.md` and pushes
the result back to your PR branch when drift is detected, so manual
regeneration is no longer required for routine PRs. If the gate fails on
a `push` to `develop`/`main` (the workflow does NOT auto-commit on
protected branches), regenerate locally and open a follow-up PR:

```bash
python3 scripts/generate_scorecard.py
git add SCORECARD.md && git commit -m "docs(scorecard): regenerate for #<issue>"
```

See `docs/agents/scorecard-regen.md` for the full agent-facing reference.

## Questions?

Open an issue or ask in the PR review.
