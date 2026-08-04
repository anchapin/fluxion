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

## Questions?

Open an issue or ask in the PR review.
