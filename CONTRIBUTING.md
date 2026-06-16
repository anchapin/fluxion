# Contributing to Fluxion

## Quick Rules

1. **Never force-push `main`**
2. **All changes go through PR** - even hotfixes
3. **CI must pass** before merging
4. **Use `--no-ff` merges** to preserve history

## Workflow

### Normal Changes
1. Create a feature branch: `git checkout -b fix/issue-123`
2. Make changes, commit
3. Push and create PR: `gh pr create`
4. Get review approval
5. Merge via PR (not squash-merge)

### Hotfixes (Unblocked CI)
1. Create branch: `git checkout -b hotfix/urgent-fix`
2. Make minimal fix
3. Create PR immediately
4. Request expedited review
5. Merge after approval

### When `main` is Broken
1. **Do NOT force-push to fix it**
2. Create a `hotfix/broken-main` branch
3. Apply minimal fix (revert broken commits or restore working versions)
4. Create PR with clear explanation
5. Get emergency review approval
6. Merge via PR

## Branch Protection

- `main` requires PR review and passing CI
- Protected branches cannot be force-pushed

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

## Questions?

Open an issue or ask in the PR review.
